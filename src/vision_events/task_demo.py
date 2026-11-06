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
from PIL import Image

from .context_memory import ContextMemory
from .dashboard import render_dashboard, wrap_display_text
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


def make_region_previews(images: dict[str, Any]) -> dict[str, np.ndarray]:
    """Make small BGR previews of the actual masked views; never save frames."""
    previews: dict[str, np.ndarray] = {}
    for region in ("person", "background"):
        image = images.get(region)
        if isinstance(image, Image.Image):
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((256, 144), Image.Resampling.BILINEAR)
            previews[region] = np.asarray(thumbnail)[:, :, ::-1].copy()
    return previews


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
        self.region_previews: dict[str, np.ndarray] = {}
        self.preview_time: float | None = None
        self.timeline: list[dict[str, Any]] = []

    def process(self, frame: np.ndarray, captured_at: float) -> tuple[dict[str, Any], dict[str, Any]]:
        started = time.perf_counter()
        images, embeddings = self.model.encode_frame(frame)
        decision = self.gate.observe(embeddings)
        gate_ms = (time.perf_counter() - started) * 1000
        self.region_previews = make_region_previews(images)
        self.preview_time = captured_at
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
            "gate_threshold": self.gate.threshold,
            "min_similarity": self.memory.min_similarity,
            "min_margin": self.memory.min_margin,
            "task": self.task,
            "answer_time": self.answer_time,
            "last_raw": self.last_raw,
            "last_source": self.last_source,
            "last_match": self.last_match,
            "last_resolution_ms": self.last_resolution_ms,
            "last_trigger": self.last_trigger,
            "last_trigger_time": self.last_trigger_time,
            "last_distances": dict(self.last_distances),
            "region_previews": {key: value.copy() for key, value in self.region_previews.items()},
            "preview_time": self.preview_time,
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
        cv2.namedWindow("Event-triggered task demo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Event-triggered task demo", 1280, 720)
        start = next_sample = time.monotonic()
        show_details = False
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
            cv2.imshow("Event-triggered task demo",
                       render_dashboard(cv2, frame, state, now - start, show_details=show_details))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("d"), ord("D")):
                show_details = not show_details
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
