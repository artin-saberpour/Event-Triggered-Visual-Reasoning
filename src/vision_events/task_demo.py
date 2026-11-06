"""Live, event-triggered webcam demo for drinking and keyboard use.

The gate detects changes; the VLM labels individual still frames. The labels
are observations, not proof of a completed action or a sent message.
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

from .gate import RegionGate


TASK_QUESTION = (
    "Look at this single image. Choose exactly one label: DRINKING if a person "
    "is visibly drinking from a cup or bottle; TYPING if the person's hands "
    "are visibly in a typing posture on a computer keyboard; UNCERTAIN if "
    "neither is clear. Do not guess or infer that a message was sent. "
    "Reply with only the label DRINKING, TYPING, or UNCERTAIN."
)

TASK_LABELS = {"DRINKING", "TYPING", "UNCERTAIN"}


def parse_task(response: str) -> str:
    """Accept only an unambiguous label; retain unexpected answers in the log."""
    label = response.strip().upper().rstrip(".! ")
    return label if label in TASK_LABELS else "UNCERTAIN"


class TaskDemoEngine:
    """Apply the existing gate and classify only its baseline and events."""

    def __init__(self, model: Any, threshold: float) -> None:
        self.model = model
        self.gate = RegionGate(threshold)
        self.samples = 0
        self.triggers = 0
        self.vlm_calls = 0
        self.task: str | None = None
        self.answer_time: float | None = None
        self.last_raw: str | None = None
        self.last_trigger: str | None = None
        self.last_trigger_time: float | None = None
        self.last_distances: dict[str, float] = {}
        self.timeline: list[dict[str, Any]] = []

    def process(self, frame: np.ndarray, captured_at: float) -> tuple[dict[str, Any], dict[str, Any]]:
        images, embeddings = self.model.encode_frame(frame)
        decision = self.gate.observe(embeddings)
        self.samples += 1
        self.last_distances = decision.distances
        record: dict[str, Any] = {
            "sample": self.samples,
            "captured_at_seconds": captured_at,
            "baseline": decision.baseline,
            "trigger_region": decision.region,
            "cosine_distances": decision.distances,
            "inference": None,
        }
        if decision.baseline or decision.region is not None:
            if decision.region is not None:
                self.triggers += 1
                self.last_trigger = decision.region
                self.last_trigger_time = captured_at
            # The gate still chooses the region; the VLM sees the full frame so
            # that the keyboard or cup is not lost in a person-only mask.
            self.vlm_calls += 1
            response = self.model.describe(images["full"], TASK_QUESTION, max_new_tokens=16)
            task = parse_task(response)
            if task != self.task:
                self.timeline.append({"time_seconds": captured_at, "task": task})
            self.task, self.answer_time, self.last_raw = task, captured_at, response
            record["inference"] = {"image": "full", "raw_response": response, "task": task}
        return record, self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "triggers": self.triggers,
            "vlm_calls": self.vlm_calls,
            "task": self.task,
            "answer_time": self.answer_time,
            "last_raw": self.last_raw,
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
    text("LAST MODEL OBSERVATION", 70, muted, 0.45)
    text(state["task"] or "Waiting for first observation", 102, white, 0.76)
    if state["answer_time"] is not None:
        text(f"Image captured at {state['answer_time']:.1f}s  |  age {max(0, elapsed - state['answer_time']):.1f}s", 126, muted)
    else:
        text("Model is preparing the first observation", 126, muted)
    text("Inference running" if state["busy"] else "Waiting for next sample", 156, accent)
    text(f"Samples: {state['samples']}   Triggers: {state['triggers']}   VLM calls: {state['vlm_calls']}", 190)
    text(f"Stale queued samples replaced: {state['dropped']}", 213, muted, 0.46)
    if state["last_trigger"] is not None:
        text(f"Last trigger: {state['last_trigger']} at {state['last_trigger_time']:.1f}s", 247)
    else:
        text("Last trigger: none", 247, muted)
    distances = state["last_distances"]
    if distances:
        text(f"CLIP distances  person {distances['person']:.3f}  background {distances['background']:.3f}", 272, muted, 0.45)
    text("OBSERVED LABEL CHANGES", 320, accent, 0.52)
    for index, event in enumerate(state["timeline"][-6:]):
        text(f"{event['time_seconds']:6.1f}s   {event['task']}", 350 + index * 27)
    text("Single frames can be ambiguous.", 548, muted, 0.45)
    text("Q or Esc: quit", 578, muted, 0.45)
    return canvas


def run_demo(args: argparse.Namespace) -> None:
    if not math.isfinite(args.sample_seconds) or args.sample_seconds <= 0:
        raise ValueError("--sample-seconds must be positive and finite")
    # Validate the threshold before opening the camera or loading weights.
    RegionGate(args.threshold)
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
        worker = LatestFrameWorker(TaskDemoEngine(model, args.threshold), log)
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
    parser = argparse.ArgumentParser(description="Show event-triggered task observations from a webcam")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.1, help="Gate cosine-distance threshold")
    parser.add_argument("--sample-seconds", type=float, default=1.0, help="Seconds between submitted camera frames")
    parser.add_argument("--log", help="Optional new JSONL path for observations; no images are saved")
    args = parser.parse_args()
    try:
        run_demo(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
