import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vision_events.task_demo import LatestFrameWorker, TaskDemoEngine, parse_task, render_dashboard, run_demo


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


def frame(index):
    return np.full((2, 2, 3), index, dtype=np.uint8)


class TaskDemoTests(unittest.TestCase):
    def test_gate_controls_real_model_calls_and_timeline_follows_model_output(self):
        model = FakeModel(["TYPING", "DRINKING", "I cannot tell"])
        engine = TaskDemoEngine(model, threshold=0.1)

        baseline, state = engine.process(frame(0), 0.0)
        self.assertTrue(baseline["baseline"])
        self.assertIsNone(baseline["trigger_region"])
        self.assertEqual(state["task"], "TYPING")

        unchanged, state = engine.process(frame(0), 1.0)
        self.assertIsNone(unchanged["inference"])
        self.assertEqual(state["vlm_calls"], 1)

        changed, state = engine.process(frame(1), 2.0)
        self.assertEqual(changed["trigger_region"], "person")
        self.assertEqual(changed["inference"]["task"], "DRINKING")
        self.assertEqual(changed["inference"]["image"], "full")
        self.assertEqual(state["triggers"], 1)

        unknown, state = engine.process(frame(2), 3.0)
        self.assertEqual(unknown["inference"]["raw_response"], "I cannot tell")
        self.assertEqual(state["task"], "UNCERTAIN")
        self.assertEqual([item["task"] for item in state["timeline"]], ["TYPING", "DRINKING", "UNCERTAIN"])
        self.assertEqual(model.queried_images, ["full-0", "full-1", "full-2"])
        self.assertEqual((state["samples"], state["triggers"], state["vlm_calls"]), (4, 2, 3))

    def test_unknown_or_extra_model_text_does_not_become_a_confident_label(self):
        self.assertEqual(parse_task(" typing. \n"), "TYPING")
        for response in ("", "DRINKING or TYPING", "The person is drinking", "not typing"):
            self.assertEqual(parse_task(response), "UNCERTAIN")

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

            @classmethod
            def putText(cls, image, value, *args):
                cls.lines.append(value)

        engine = TaskDemoEngine(FakeModel(["TYPING"]), 0.1)
        _, state = engine.process(frame(0), 2.0)
        state.update(busy=False, dropped=0, error=None)
        dashboard = render_dashboard(FakeCV2, frame(0), state, 5.0)
        self.assertEqual(dashboard.shape, (600, 1280, 3))
        self.assertIn("TYPING", FakeCV2.lines)
        self.assertTrue(any("age 3.0s" in line for line in FakeCV2.lines))
        self.assertTrue(any("Samples: 1   Triggers: 0   VLM calls: 1" in line for line in FakeCV2.lines))

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

        model = DemoModel(["DRINKING"])

        class FakeCV2:
            FONT_HERSHEY_SIMPLEX = LINE_AA = 0
            destroyed = False

            @staticmethod
            def VideoCapture(index):
                return camera

            @staticmethod
            def resize(image, size):
                return np.zeros((size[1], size[0], 3), dtype=np.uint8)

            @staticmethod
            def rectangle(*args):
                pass

            @staticmethod
            def putText(*args):
                pass

            @staticmethod
            def imshow(*args):
                pass

            @staticmethod
            def waitKey(delay):
                if not finished.wait(timeout=5):
                    raise RuntimeError("test timed out")
                return ord("q")

            @classmethod
            def destroyAllWindows(cls):
                cls.destroyed = True

        with tempfile.TemporaryDirectory() as temp:
            log_path = Path(temp) / "events.jsonl"
            args = SimpleNamespace(sample_seconds=1.0, threshold=0.1, camera=0,
                                   log=str(log_path), device="cpu")
            with patch.dict("sys.modules", {"cv2": FakeCV2}), \
                    patch("vision_events.models.VisionModels", return_value=model):
                run_demo(args)
            records = [json.loads(line) for line in log_path.read_text().splitlines()]
        self.assertEqual(records[0]["inference"]["task"], "DRINKING")
        self.assertTrue(camera.released)
        self.assertTrue(model.closed)
        self.assertTrue(FakeCV2.destroyed)


if __name__ == "__main__":
    unittest.main()
