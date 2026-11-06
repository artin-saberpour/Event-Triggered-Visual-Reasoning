"""Live webcam demo with open-ended descriptions and CLIP context memory.

The gate detects visual changes; the VLM describes individual still frames.
Its descriptions are not proof of a completed action or a person's intent.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, TextIO

import numpy as np

from .context_memory import ContextMemory
from .gate import RegionGate


TASK_QUESTION = (
    "Describe in one short sentence what the person is visibly doing right now. "
    "Focus on their posture, hands, and any object they are interacting with. "
    "If the activity is unclear, describe only what is visible. Do not guess "
    "their intent, spoken words, or the contents of a screen."
)


def parse_observation(response: str) -> str:
    """Keep the model's open-ended wording, with a fallback for empty output."""
    return " ".join(response.split()) or "No description returned"


def cacheable_description(response: str) -> bool:
    """Avoid caching empty and some explicitly uncertain responses.

    This is a small heuristic, not a reliable model-confidence estimator.
    """
    normalized = " ".join(response.lower().split())
    if not normalized:
        return False
    uncertain = ("cannot tell", "can't tell", "cannot determine", "can't determine",
                 "unable to determine", "unable to tell", "cannot see", "can't see",
                 "not sure", "unclear")
    return not any(phrase in normalized for phrase in uncertain)


class TaskDemoEngine:
    """Resolve gate events through session memory or a fresh VLM response."""

    def __init__(self, model: Any, threshold: float, mode: str = "hybrid",
                 min_similarity: float = 0.30, min_margin: float = 0.04) -> None:
        if mode not in ("hybrid", "vlm"):
            raise ValueError("mode must be hybrid or vlm")
        self.model = model
        self.gate = RegionGate(threshold)
        self.mode = mode
        self.memory = ContextMemory(min_similarity, min_margin)
        self.samples = 0
        self.triggers = 0
        self.vlm_calls = 0
        self.cache_hits = 0
        self.task: str | None = None
        self.answer_time: float | None = None
        self.last_raw: str | None = None
        self.last_source: str | None = None
        self.last_match: dict[str, Any] | None = None
        self.last_resolution_ms: float | None = None
        self.last_trigger: str | None = None
        self.last_trigger_time: float | None = None
        self.last_distances: dict[str, float] = {}
        self.timeline: list[dict[str, Any]] = []

    def process(self, frame: np.ndarray, captured_at: float) -> tuple[dict[str, Any], dict[str, Any]]:
        started = time.perf_counter()
        images, embeddings = self.model.encode_frame(frame)
        decision = self.gate.observe(embeddings)
        gate_ms = (time.perf_counter() - started) * 1000
        self.samples += 1
        self.last_distances = decision.distances
        record: dict[str, Any] = {
            "sample": self.samples,
            "captured_at_seconds": captured_at,
            "baseline": decision.baseline,
            "trigger_region": decision.region,
            "cosine_distances": decision.distances,
            "inference": None,
            "recognition": None,
        }
        resolution_ms = vlm_ms = None
        if decision.baseline or decision.region is not None:
            resolving = time.perf_counter()
            if decision.region is not None:
                self.triggers += 1
                self.last_trigger = decision.region
                self.last_trigger_time = captured_at
            match_info = None
            task = None
            if self.mode == "hybrid" and not decision.baseline:
                match = self.memory.match(embeddings["full"])
                match_info = {"best_similarity": match.best_similarity,
                              "margin": match.margin, "reason": match.reason,
                              "best_description": match.candidate}
                task = match.description
            cache_added = False
            if task is not None:
                source, response = "cache", None
                self.cache_hits += 1
            else:
                source = "vlm"
                self.vlm_calls += 1
                vlm_start = time.perf_counter()
                # Use the full frame so the object interacting with the person
                # is not removed by a person-only mask.
                response = self.model.describe(images["full"], TASK_QUESTION, max_new_tokens=48)
                vlm_ms = (time.perf_counter() - vlm_start) * 1000
                task = parse_observation(response)
                record["inference"] = {"image": "full", "raw_response": response, "task": task}
                if self.mode == "hybrid" and cacheable_description(response):
                    text_embedding = self.model.encode_texts([task])[0]
                    cache_added = self.memory.add(task, text_embedding)
            resolution_ms = (time.perf_counter() - resolving) * 1000
            self.timeline.append({"time_seconds": captured_at, "task": task, "source": source})
            self.task, self.answer_time, self.last_raw = task, captured_at, response
            self.last_source, self.last_match = source, match_info
            self.last_resolution_ms = resolution_ms
            record["recognition"] = {"source": source, "description": task, "match": match_info,
                                     "cache_added": cache_added, "cache_size": len(self.memory)}
        record["timings_ms"] = {"gate": gate_ms, "context_resolution": resolution_ms,
                                "vlm": vlm_ms, "total_processing": (time.perf_counter() - started) * 1000}
        return record, self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "triggers": self.triggers,
            "vlm_calls": self.vlm_calls,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.memory),
            "mode": self.mode,
            "task": self.task,
            "answer_time": self.answer_time,
            "last_raw": self.last_raw,
            "last_source": self.last_source,
            "last_match": self.last_match,
            "last_resolution_ms": self.last_resolution_ms,
            "last_trigger": self.last_trigger,
            "last_trigger_time": self.last_trigger_time,
            "last_distances": dict(self.last_distances),
            "timeline": copy.deepcopy(self.timeline),
        }


class LatestFrameWorker:
    """Keep inference off the display thread and discard stale queued frames."""

    def __init__(self, engine: TaskDemoEngine, log: TextIO | None = None) -> None:
        self.engine, self.log = engine, log
        self._condition = threading.Condition()
        self._pending: tuple[np.ndarray, float] | None = None
        self._stopping = False
        self._busy = False
        self._dropped = 0
        self._error: str | None = None
        self._state = engine.snapshot()
        self._thread = threading.Thread(target=self._run, name="vision-task-demo", daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray, captured_at: float) -> None:
        with self._condition:
            if self._stopping:
                return
            if self._pending is not None:
                self._dropped += 1
            self._pending = (frame.copy(), captured_at)
            self._condition.notify()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            state = copy.deepcopy(self._state)
            state.update(busy=self._busy, dropped=self._dropped, error=self._error)
            return state

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                item = self._pending
                self._pending = None
                self._busy = True
            assert item is not None
            try:
                record, state = self.engine.process(*item)
                line = json.dumps(record, ensure_ascii=True)
                print(line, flush=True)
                if self.log is not None:
                    self.log.write(line + "\n")
                    self.log.flush()
            except Exception as exc:
                with self._condition:
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._busy = False
                    self._stopping = True
                return
            with self._condition:
                self._state = state
                self._busy = False

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify_all()
        self._thread.join()


def wrap_display_text(cv2: Any, value: str, max_width: int, scale: float, max_lines: int) -> list[str]:
    """Wrap text to actual panel width; the complete response stays in the log."""
    value = " ".join(value.split()).encode("ascii", "replace").decode("ascii")
    font = cv2.FONT_HERSHEY_SIMPLEX

    def fits(text: str) -> bool:
        return cv2.getTextSize(text, font, scale, 1)[0][0] <= max_width

    lines: list[str] = []
    line = ""
    for word in value.split():
        proposed = f"{line} {word}".strip()
        if fits(proposed):
            line = proposed
            continue
        if line:
            lines.append(line)
            line = ""
        while word and not fits(word):
            end = len(word)
            while end > 1 and not fits(word[:end]):
                end -= 1
            lines.append(word[:end])
            word = word[end:]
        line = word
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and not fits(last + "..."):
            last = last[:-1]
        lines[-1] = last.rstrip() + "..."
    return lines


def render_dashboard(cv2: Any, frame: np.ndarray, state: dict[str, Any], elapsed: float) -> np.ndarray:
    """Draw a small dashboard around the actual camera image."""
    height, video_width, panel_width = 600, 800, 480
    canvas = np.full((height, video_width + panel_width, 3), (19, 22, 28), dtype=np.uint8)
    source_height, source_width = frame.shape[:2]
    scale = min(video_width / source_width, height / source_height)
    width, scaled_height = max(1, int(source_width * scale)), max(1, int(source_height * scale))
    view = cv2.resize(frame, (width, scaled_height))
    x0, y0 = (video_width - width) // 2, (height - scaled_height) // 2
    canvas[y0:y0 + scaled_height, x0:x0 + width] = view
    cv2.rectangle(canvas, (0, 0), (video_width, 40), (22, 25, 31), -1)
    cv2.putText(canvas, f"LIVE CAMERA    {elapsed:05.1f}s", (16, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)

    x = video_width + 22
    white, muted, accent = (245, 245, 245), (180, 185, 190), (110, 215, 180)

    def text(line: str, y: int, color: tuple[int, int, int] = white, scale: float = 0.5) -> None:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    text("EVENT-TRIGGERED TASK DEMO", 34, accent, 0.58)
    text("LATEST CONTEXT", 67, muted, 0.45)
    description = state["task"] or "Waiting for first observation"
    for index, line in enumerate(wrap_display_text(cv2, description, 430, 0.57, 3)):
        text(line, 96 + index * 23, white, 0.57)
    if state["answer_time"] is not None:
        text(f"Image captured at {state['answer_time']:.1f}s  |  age {max(0, elapsed - state['answer_time']):.1f}s", 170, muted)
    else:
        text("Model is preparing the first observation", 170, muted)
    source = {"vlm": "VLM", "cache": "CLIP CACHE"}.get(state["last_source"], "--")
    status = "RUNNING" if state["busy"] else "WAITING"
    text(f"Source: {source}   |   {status}", 196, accent)
    text(f"Samples: {state['samples']}   Triggers: {state['triggers']}", 218)
    text(f"VLM calls: {state['vlm_calls']}   Reuses: {state['cache_hits']}   Stored: {state['cache_size']}", 240)
    if state["last_resolution_ms"] is not None:
        text(f"Resolve after gate: {state['last_resolution_ms']:.0f} ms", 262, muted, 0.45)
    else:
        text("Resolve after gate: --", 262, muted, 0.45)
    match = state["last_match"]
    if match and match["best_similarity"] is not None:
        gap = "--" if match["margin"] is None else f"{match['margin']:.3f}"
        text(f"Match: {match['best_similarity']:.3f}   gap: {gap}   {match['reason']}", 284, muted, 0.45)
    else:
        text("Match: no comparison yet", 284, muted, 0.45)
    distances = state["last_distances"]
    if distances:
        last_trigger = state["last_trigger"] or "none"
        text(f"Gate P/B: {distances['person']:.3f}/{distances['background']:.3f}   Last: {last_trigger}", 306, muted, 0.45)
    text("RECENT CONTEXTS  [V] VLM  [C] CACHE", 329, accent, 0.47)
    for index, event in enumerate(state["timeline"][-4:]):
        source_letter = "V" if event["source"] == "vlm" else "C"
        entry = f"{event['time_seconds']:5.1f}s [{source_letter}]  {event['task']}"
        for line_index, line in enumerate(wrap_display_text(cv2, entry, 430, 0.44, 2)):
            text(line, 354 + index * 43 + line_index * 17, white, 0.44)
    text("Single frames can be ambiguous.", 548, muted, 0.45)
    text("Q or Esc: quit", 578, muted, 0.45)
    return canvas


def run_demo(args: argparse.Namespace) -> None:
    if not math.isfinite(args.sample_seconds) or args.sample_seconds <= 0:
        raise ValueError("--sample-seconds must be positive and finite")
    # Validate the threshold before opening the camera or loading weights.
    RegionGate(args.threshold)
    ContextMemory(args.min_similarity, args.min_margin)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install the model dependencies with: pip install -e '.[models]'") from exc
    from .models import VisionModels

    print(f"Opening webcam {args.camera}...", flush=True)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open webcam {args.camera}. Try another --camera index.")
    model = worker = log = None
    try:
        if args.log:
            path = Path(args.log)
            path.parent.mkdir(parents=True, exist_ok=True)
            log = path.open("x", encoding="utf-8")
        print("Loading segmentation, CLIP and LLaVA; model weights may need to download...", flush=True)
        model = VisionModels(device=args.device, use_vlm=True)
        engine = TaskDemoEngine(model, args.threshold, mode=args.mode,
                                min_similarity=args.min_similarity, min_margin=args.min_margin)
        worker = LatestFrameWorker(engine, log)
        start = next_sample = time.monotonic()
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a webcam frame. Check the camera connection.")
            now = time.monotonic()
            if now >= next_sample:
                worker.submit(frame, now - start)
                next_sample = now + args.sample_seconds
            state = worker.snapshot()
            if state["error"]:
                raise RuntimeError(f"Inference failed: {state['error']}")
            cv2.imshow("Event-triggered task demo", render_dashboard(cv2, frame, state, now - start))
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    finally:
        capture.release()
        if worker is not None:
            worker.stop()
        if model is not None:
            model.close()
        if log is not None:
            log.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Show open-ended event-triggered observations from a webcam")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.1, help="Gate cosine-distance threshold")
    parser.add_argument("--sample-seconds", type=float, default=1.0, help="Seconds between submitted camera frames")
    parser.add_argument("--mode", choices=["hybrid", "vlm"], default="hybrid",
                        help="Reuse CLIP memory or always call the VLM on a trigger")
    parser.add_argument("--min-similarity", type=float, default=0.30,
                        help="Minimum image/text cosine similarity for cache reuse (uncalibrated)")
    parser.add_argument("--min-margin", type=float, default=0.04,
                        help="Minimum gap between top two cache matches (uncalibrated)")
    parser.add_argument("--log", help="Optional new JSONL path for observations; no images are saved")
    args = parser.parse_args()
    try:
        run_demo(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
