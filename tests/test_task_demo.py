import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from vision_events.task_demo import (
    LatestFrameWorker, TaskDemoEngine, make_region_previews, parse_observation, render_dashboard, run_demo,
    wrap_display_text,
)


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.queried_images = []

    def encode_frame(self, frame):
        index = int(frame[0, 0, 0])
        person = [1, 0] if index in (0, 2) else [0, 1]
        images = {region: f"{region}-{index}" for region in ("full", "person", "background")}
        return images, {"full": person, "person": person, "background": [1, 0]}

    def describe(self, image, question, max_new_tokens=80):
        self.queried_images.append(image)
        return next(self.responses)

    def encode_texts(self, descriptions):
        return np.array([[1.0, 0.0] for _ in descriptions])


def frame(index):
    return np.full((2, 2, 3), index, dtype=np.uint8)


class TaskDemoTests(unittest.TestCase):
    def test_region_previews_preserve_actual_masks_and_do_not_enter_logs(self):
        class ImageModel(FakeModel):
            def encode_frame(self, frame):
                images = {"full": Image.new("RGB", (640, 480), (15, 25, 35)),
                          "person": Image.new("RGB", (640, 480), (255, 0, 0)),
                          "background": Image.new("RGB", (640, 480), (0, 0, 255))}
                return images, {key: [1, 0] for key in ("full", "person", "background")}

        model = ImageModel(["Holding a cup."])
        engine = TaskDemoEngine(model, 0.1)
        record, state = engine.process(frame(0), 3.0)
        person = state["region_previews"]["person"]
        background = state["region_previews"]["background"]
        self.assertEqual(person.shape, (144, 192, 3))
        self.assertEqual(person[0, 0].tolist(), [0, 0, 255])
        self.assertEqual(background[0, 0].tolist(), [255, 0, 0])
        self.assertEqual(state["preview_time"], 3.0)
        person[:] = 0
        self.assertEqual(engine.snapshot()["region_previews"]["person"][0, 0].tolist(), [0, 0, 255])
        self.assertNotIn("region_previews", json.dumps(record))
        self.assertEqual(make_region_previews({"person": "not an image"}), {})

    def test_gate_controls_real_model_calls_and_timeline_follows_model_output(self):
        model = FakeModel(["Typing on a keyboard.", "Holding a phone to their ear.", "I cannot tell."])
        engine = TaskDemoEngine(model, threshold=0.1, mode="vlm")

        baseline, state = engine.process(frame(0), 0.0)
        self.assertTrue(baseline["baseline"])
        self.assertIsNone(baseline["trigger_region"])
        self.assertEqual(state["task"], "Typing on a keyboard.")

        unchanged, state = engine.process(frame(0), 1.0)
        self.assertIsNone(unchanged["inference"])
        self.assertEqual(state["vlm_calls"], 1)

        changed, state = engine.process(frame(1), 2.0)
        self.assertEqual(changed["trigger_region"], "person")
        self.assertEqual(changed["inference"]["task"], "Holding a phone to their ear.")
        self.assertEqual(changed["inference"]["image"], "full")
        self.assertEqual(state["triggers"], 1)

        unknown, state = engine.process(frame(2), 3.0)
        self.assertEqual(unknown["inference"]["raw_response"], "I cannot tell.")
        self.assertEqual(state["task"], "I cannot tell.")
        self.assertEqual([item["task"] for item in state["timeline"]], [
            "Typing on a keyboard.", "Holding a phone to their ear.", "I cannot tell."
        ])
        self.assertEqual(model.queried_images, ["full-0", "full-1", "full-2"])
        self.assertEqual((state["samples"], state["triggers"], state["vlm_calls"]), (4, 2, 3))

    def test_open_ended_descriptions_and_uncertainty_are_retained(self):
        for response in ("Petting a dog.", "Holding a book.", "I cannot see the person's activity."):
            self.assertEqual(parse_observation(response), response)
        self.assertEqual(parse_observation("  Looking\n at a phone. "), "Looking at a phone.")
        self.assertEqual(parse_observation(" \n "), "No description returned")

    def test_repeated_observations_are_not_treated_as_verified_task_changes(self):
        engine = TaskDemoEngine(FakeModel(["Holding a phone.", "Holding a phone."]), 0.1, mode="vlm")
        engine.process(frame(0), 0.0)
        _, state = engine.process(frame(1), 1.0)
        self.assertEqual([item["task"] for item in state["timeline"]], ["Holding a phone."] * 2)

    def test_worker_replaces_stale_frames_while_inference_is_busy(self):
        class BlockingEngine:
            def __init__(self):
                self.started, self.release, self.finished = threading.Event(), threading.Event(), threading.Event()
                self.frames = []

            def snapshot(self):
                return {"frames": list(self.frames)}

            def process(self, image, captured_at):
                index = int(image[0, 0, 0])
                self.frames.append(index)
                if index == 0:
                    self.started.set()
                    if not self.release.wait(timeout=5):
                        raise RuntimeError("test timed out")
                if index == 2:
                    self.finished.set()
                return {"frame": index}, self.snapshot()

        engine = BlockingEngine()
        log = io.StringIO()
        worker = LatestFrameWorker(engine, log)
        try:
            worker.submit(frame(0), 0.0)
            self.assertTrue(engine.started.wait(timeout=5))
            worker.submit(frame(1), 1.0)
            worker.submit(frame(2), 2.0)
            self.assertEqual(worker.snapshot()["dropped"], 1)
            engine.release.set()
            self.assertTrue(engine.finished.wait(timeout=5))
        finally:
            engine.release.set()
            worker.stop()
        self.assertEqual(engine.frames, [0, 2])
        self.assertEqual([json.loads(line)["frame"] for line in log.getvalue().splitlines()], [0, 2])
        self.assertIsNone(worker.snapshot()["error"])

    def test_dashboard_shows_label_timestamp_and_counts(self):
        class FakeCV2:
            FONT_HERSHEY_SIMPLEX = LINE_AA = 0
            lines = []

            @staticmethod
            def resize(image, size):
                return np.zeros((size[1], size[0], 3), dtype=np.uint8)

            @staticmethod
            def rectangle(*args):
                pass

            @staticmethod
            def circle(*args):
                pass

            @staticmethod
            def line(*args):
                pass

            @staticmethod
            def getTextSize(value, font, scale, thickness):
                return ((len(value) * 9, 12), 2)

            @classmethod
            def putText(cls, image, value, *args):
                cls.lines.append(value)

        engine = TaskDemoEngine(FakeModel(["Holding a phone to their ear."]), 0.1)
        _, state = engine.process(frame(0), 2.0)
        state.update(busy=False, dropped=0, error=None)
        dashboard = render_dashboard(FakeCV2, frame(0), state, 5.0)
        self.assertEqual(dashboard.shape, (900, 1600, 3))
        self.assertIn("Holding a phone to their ear.", FakeCV2.lines)
        self.assertIn("RECENT OBSERVATIONS", FakeCV2.lines)
        self.assertIn("GENERATED BY LLaVA", FakeCV2.lines)
        self.assertIn("Frame captured 3.0s ago", FakeCV2.lines)
        self.assertIn("MediaPipe", FakeCV2.lines)
        self.assertIn("CLIP", FakeCV2.lines)
        self.assertIn("LLaVA", FakeCV2.lines)
        self.assertIn("Person / background distance", FakeCV2.lines)
        self.assertIn("LLaVA calls / memory reuses", FakeCV2.lines)
        self.assertIn("Match / gap thresholds", FakeCV2.lines)

        FakeCV2.lines.clear()
        state.update(last_source="cache", cache_hits=1, cache_size=2, last_resolution_ms=0.7,
                     last_trigger_time=1.0,
                     last_match={"best_similarity": 0.38, "margin": 0.08, "reason": "matched"},
                     timeline=[{"time_seconds": 2.0, "task": "Holding a phone.", "source": "cache"}])
        render_dashboard(FakeCV2, frame(0), state, 5.0)
        self.assertIn("REUSED FROM CLIP MEMORY", FakeCV2.lines)
        self.assertIn("Memory", FakeCV2.lines)
        self.assertIn("Last memory similarity / gap", FakeCV2.lines)
        self.assertIn("0.380 / 0.080", FakeCV2.lines)

        FakeCV2.lines.clear()
        render_dashboard(FakeCV2, frame(0), state, 5.0, show_details=True)
        self.assertIn("EXPANDED DETAILS", FakeCV2.lines)
        self.assertIn("BEST CACHED CANDIDATE", FakeCV2.lines)
        self.assertIn("Similarity: 0.380   Gap: 0.080   Decision: matched", FakeCV2.lines)
        self.assertIn("REUSED FROM CLIP MEMORY", FakeCV2.lines)
        self.assertIn("MediaPipe", FakeCV2.lines)

        wrapped = wrap_display_text(FakeCV2, "A very long open ended description " * 8, 120, 0.5, 2)
        self.assertEqual(len(wrapped), 2)
        self.assertTrue(wrapped[-1].endswith("..."))
        self.assertTrue(all(len(line) * 9 <= 120 for line in wrapped))

    def test_camera_loop_initializes_inference_and_cleans_up(self):
        class Camera:
            released = False

            def isOpened(self):
                return True

            def read(self):
                return True, frame(0)

            def release(self):
                self.released = True

        camera = Camera()
        finished = threading.Event()

        class DemoModel(FakeModel):
            closed = False

            def describe(self, image, question, max_new_tokens=80):
                result = super().describe(image, question, max_new_tokens)
                finished.set()
                return result

            def close(self):
                self.closed = True

        model = DemoModel(["Drinking from a cup."])

        class FakeCV2:
            FONT_HERSHEY_SIMPLEX = LINE_AA = WINDOW_NORMAL = 0
            destroyed = False
            lines = []
            key_count = 0

            @staticmethod
            def VideoCapture(index):
                return camera

            @staticmethod
            def namedWindow(*args):
                pass

            @staticmethod
            def resizeWindow(*args):
                pass

            @staticmethod
            def resize(image, size):
                return np.zeros((size[1], size[0], 3), dtype=np.uint8)

            @staticmethod
            def rectangle(*args):
                pass

            @staticmethod
            def circle(*args):
                pass

            @staticmethod
            def line(*args):
                pass

            @staticmethod
            def getTextSize(value, font, scale, thickness):
                return ((len(value) * 9, 12), 2)

            @classmethod
            def putText(cls, image, value, *args):
                cls.lines.append(value)

            @staticmethod
            def imshow(*args):
                pass

            @classmethod
            def waitKey(cls, delay):
                if not finished.wait(timeout=5):
                    raise RuntimeError("test timed out")
                cls.key_count += 1
                return ord("d") if cls.key_count <= 2 else ord("q")

            @classmethod
            def destroyAllWindows(cls):
                cls.destroyed = True

        with tempfile.TemporaryDirectory() as temp:
            log_path = Path(temp) / "events.jsonl"
            args = SimpleNamespace(sample_seconds=1.0, threshold=0.1, camera=0,
                                   log=str(log_path), device="cpu", mode="hybrid",
                                   min_similarity=0.30, min_margin=0.04)
            with patch.dict("sys.modules", {"cv2": FakeCV2}), \
                    patch("vision_events.models.VisionModels", return_value=model):
                run_demo(args)
            records = [json.loads(line) for line in log_path.read_text().splitlines()]
        self.assertEqual(records[0]["inference"]["task"], "Drinking from a cup.")
        self.assertTrue(camera.released)
        self.assertTrue(model.closed)
        self.assertTrue(FakeCV2.destroyed)
        self.assertGreaterEqual(FakeCV2.lines.count("RECENT OBSERVATIONS"), 2)
        self.assertIn("EXPANDED DETAILS", FakeCV2.lines)


if __name__ == "__main__":
    unittest.main()
