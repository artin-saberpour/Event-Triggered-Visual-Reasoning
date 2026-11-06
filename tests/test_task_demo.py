import argparse
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

from vision_events.dashboard import BACKGROUND, INK, PAPER, PERSON, RESET, clip_chart_window, draw_clip_chart
from vision_events.task_demo import (
    LatestFrameWorker, TaskDemoEngine, make_region_previews, parse_camera_resolution, parse_observation,
    render_dashboard, run_demo, wrap_display_text,
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
    def test_distance_trace_records_real_gate_values_and_reference_reset(self):
        engine = TaskDemoEngine(FakeModel(["Typing.", "Holding a phone."]), 0.1, mode="vlm")
        _, state = engine.process(frame(0), 0.0)
        self.assertEqual(state["distance_history"], [])
        unchanged, _ = engine.process(frame(0), 1.0)
        triggered, _ = engine.process(frame(1), 2.0)
        after_reset, state = engine.process(frame(1), 3.0)
        history = state["distance_history"]
        self.assertEqual([item["time_seconds"] for item in history], [1.0, 2.0, 3.0])
        self.assertEqual([item["trigger_region"] for item in history], [None, "person", None])
        self.assertAlmostEqual(history[0]["person"], unchanged["cosine_distances"]["person"])
        self.assertAlmostEqual(history[1]["person"], triggered["cosine_distances"]["person"])
        self.assertAlmostEqual(history[1]["person"], 1.0)
        self.assertAlmostEqual(history[2]["person"], after_reset["cosine_distances"]["person"])
        self.assertAlmostEqual(history[2]["person"], 0.0)
        history[1]["person"] = 123.0
        self.assertAlmostEqual(engine.snapshot()["distance_history"][1]["person"], 1.0)

    def test_distance_history_is_bounded_and_chart_uses_only_recent_samples(self):
        engine = TaskDemoEngine(FakeModel(["Typing."]), 0.1, mode="vlm")
        engine.process(frame(0), 0.0)
        for index in range(1, 250):
            engine.process(frame(0), float(index))
        history = engine.snapshot()["distance_history"]
        self.assertEqual(len(history), 240)
        self.assertEqual((history[0]["time_seconds"], history[-1]["time_seconds"]), (10.0, 249.0))
        samples, start, end, upper = clip_chart_window(history, 249.0, 0.1)
        self.assertEqual((start, end), (219.0, 249.0))
        self.assertEqual((samples[0]["time_seconds"], samples[-1]["time_seconds"]), (219.0, 249.0))
        self.assertAlmostEqual(upper, 0.20)

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
            INTER_LINEAR, INTER_AREA = 1, 2
            lines = []
            segments = []
            circles = []
            interpolations = []

            @classmethod
            def resize(cls, image, size, interpolation):
                cls.interpolations.append(interpolation)
                return np.zeros((size[1], size[0], 3), dtype=np.uint8)

            @staticmethod
            def rectangle(*args):
                pass

            @classmethod
            def circle(cls, image, point, radius, color, thickness, line_type):
                cls.circles.append((point, radius, color, thickness))

            @classmethod
            def line(cls, image, start, end, color, thickness, line_type):
                cls.segments.append((start, end, color, thickness))

            @staticmethod
            def getTextSize(value, font, scale, thickness):
                return ((len(value) * 9, 12), 2)

            @classmethod
            def putText(cls, image, value, *args):
                cls.lines.append(value)

        def capture_labels(canvas, labels):
            FakeCV2.lines.extend(label[0] for label in labels)
            return canvas

        def render(*args, **kwargs):
            with patch("vision_events.dashboard._paint_labels", side_effect=capture_labels):
                return render_dashboard(*args, **kwargs)

        engine = TaskDemoEngine(FakeModel(["Holding a phone to their ear."]), 0.1)
        _, state = engine.process(frame(0), 2.0)
        state.update(busy=False, dropped=0, error=None,
                     last_distances={"person": 0.12, "background": 0.02},
                     distance_history=[
                         {"time_seconds": 3.0, "person": 0.02, "background": 0.01, "trigger_region": None},
                         {"time_seconds": 4.0, "person": 0.12, "background": 0.02, "trigger_region": "person"},
                     ])
        dashboard = render(FakeCV2, frame(0), state, 5.0)
        self.assertEqual(dashboard.shape, (1080, 1920, 3))
        self.assertIn("Holding a phone to their ear.", FakeCV2.lines)
        self.assertIn("RECENT OBSERVATIONS", FakeCV2.lines)
        self.assertIn("GENERATED BY LLaVA", FakeCV2.lines)
        self.assertIn("Frame captured 3.0s ago", FakeCV2.lines)
        self.assertIn("MediaPipe", FakeCV2.lines)
        self.assertIn("CLIP", FakeCV2.lines)
        self.assertIn("LLaVA", FakeCV2.lines)
        self.assertIn("CLIP CHANGE OVER TIME", FakeCV2.lines)
        self.assertIn("0.120 / 0.020", FakeCV2.lines)
        self.assertIn("Person", FakeCV2.lines)
        self.assertIn("Background", FakeCV2.lines)
        self.assertIn("Threshold 0.10", FakeCV2.lines)
        self.assertIn("Trigger  |  dashed drop = reset", FakeCV2.lines)
        self.assertTrue(any(segment[2] == PERSON and segment[3] == 2 for segment in FakeCV2.segments))
        self.assertTrue(any(segment[2] == BACKGROUND and segment[3] == 2 for segment in FakeCV2.segments))
        self.assertTrue(any(c[1:] == (5, INK, 1) for c in FakeCV2.circles))
        self.assertIn("LLaVA calls / memory reuses", FakeCV2.lines)
        self.assertIn("Match / gap thresholds", FakeCV2.lines)

        painted = render_dashboard(FakeCV2, frame(0), state, 5.0)
        self.assertTrue(np.any(painted[10:55, 20:650] != PAPER))
        render(FakeCV2, np.zeros((1080, 1920, 3), dtype=np.uint8), state, 5.0)
        self.assertIn(FakeCV2.INTER_AREA, FakeCV2.interpolations)
        self.assertIn(FakeCV2.INTER_LINEAR, FakeCV2.interpolations)

        FakeCV2.lines.clear()
        state.update(last_source="cache", cache_hits=1, cache_size=2, last_resolution_ms=0.7,
                     last_trigger_time=1.0,
                     last_match={"best_similarity": 0.38, "margin": 0.08, "reason": "matched"},
                     timeline=[{"time_seconds": 2.0, "task": "Holding a phone.", "source": "cache"}])
        render(FakeCV2, frame(0), state, 5.0)
        self.assertIn("REUSED FROM CLIP MEMORY", FakeCV2.lines)
        self.assertIn("Memory", FakeCV2.lines)
        self.assertIn("Memory similarity / gap", FakeCV2.lines)
        self.assertIn("0.380 / 0.080", FakeCV2.lines)

        FakeCV2.lines.clear()
        render(FakeCV2, frame(0), state, 5.0, show_details=True)
        self.assertIn("EXPANDED DETAILS", FakeCV2.lines)
        self.assertIn("BEST CACHED CANDIDATE", FakeCV2.lines)
        self.assertIn("Similarity: 0.380   Gap: 0.080   Decision: matched", FakeCV2.lines)
        self.assertIn("REUSED FROM CLIP MEMORY", FakeCV2.lines)
        self.assertIn("MediaPipe", FakeCV2.lines)

        wrapped = wrap_display_text(FakeCV2, "A very long open ended description " * 8, 120, 0.5, 2)
        self.assertEqual(len(wrapped), 2)
        self.assertTrue(wrapped[-1].endswith("..."))
        self.assertTrue(all(len(line) * 9 <= 120 for line in wrapped))

        FakeCV2.segments.clear()
        gap_history = [
            {"time_seconds": 1.0, "person": 0.02, "background": 0.01, "trigger_region": None},
            {"time_seconds": 10.0, "person": 0.12, "background": 0.02, "trigger_region": "person"},
        ]
        draw_clip_chart(FakeCV2, dashboard, gap_history, 12.0, 0.1, 0, 0, 400, 100)
        self.assertFalse(any(s[2] in (PERSON, BACKGROUND) and s[3] == 2
                             for s in FakeCV2.segments))
        self.assertTrue(any(s[2] == RESET and s[3] == 1 for s in FakeCV2.segments))

        FakeCV2.segments.clear()
        reset_history = [
            {"time_seconds": 1.0, "person": 0.02, "background": 0.01, "trigger_region": None},
            {"time_seconds": 2.0, "person": 0.12, "background": 0.02, "trigger_region": "person"},
            {"time_seconds": 3.0, "person": 0.01, "background": 0.03, "trigger_region": None},
        ]
        draw_clip_chart(FakeCV2, dashboard, reset_history, 4.0, 0.1, 0, 0, 400, 100)
        person_segments = [s for s in FakeCV2.segments if s[2] == PERSON and s[3] == 2]
        background_segments = [s for s in FakeCV2.segments if s[2] == BACKGROUND and s[3] == 2]
        self.assertEqual(len(person_segments), 2)
        self.assertEqual(len(background_segments), 2)
        self.assertEqual(person_segments[1][0], (222, 76))
        self.assertNotEqual(background_segments[1][0], (222, 76))

        FakeCV2.segments.clear()
        reset_history[1]["trigger_region"] = "full"
        draw_clip_chart(FakeCV2, dashboard, reset_history, 4.0, 0.1, 0, 0, 400, 100)
        background_segments = [s for s in FakeCV2.segments if s[2] == BACKGROUND and s[3] == 2]
        self.assertEqual(len(background_segments), 2)
        self.assertEqual(background_segments[1][0], (222, 76))

        FakeCV2.segments.clear()
        gap_after_reset = [
            {"time_seconds": 1.0, "person": 0.12, "background": 0.01, "trigger_region": "person"},
            {"time_seconds": 10.0, "person": 0.02, "background": 0.02, "trigger_region": None},
        ]
        draw_clip_chart(FakeCV2, dashboard, gap_after_reset, 12.0, 0.1, 0, 0, 400, 100)
        self.assertFalse(any(s[2] in (PERSON, BACKGROUND) and s[3] == 2
                             for s in FakeCV2.segments))

        FakeCV2.lines.clear()
        FakeCV2.segments.clear()
        draw_clip_chart(FakeCV2, dashboard, [], 0.0, 0.1, 0, 0, 400, 100)
        self.assertIn("Waiting for processed CLIP samples", FakeCV2.lines)
        self.assertFalse(any(s[2] in (PERSON, BACKGROUND) and s[3] == 2
                             for s in FakeCV2.segments))

    def test_camera_loop_initializes_inference_and_cleans_up(self):
        class Camera:
            released = False
            settings = []

            def isOpened(self):
                return True

            def read(self):
                return True, frame(0)

            def set(self, prop, value):
                self.settings.append((prop, value))
                return True

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
            INTER_LINEAR, INTER_AREA = 1, 2
            CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT = 3, 4
            WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN = 5, 6
            destroyed = False
            lines = []
            properties = []
            key_count = 0

            @staticmethod
            def VideoCapture(index):
                return camera

            @staticmethod
            def namedWindow(*args):
                pass

            @staticmethod
            def resizeWindow(*args):
                raise AssertionError("Fullscreen should not use windowed sizing")

            @classmethod
            def setWindowProperty(cls, *args):
                cls.properties.append(args)

            @staticmethod
            def resize(image, size, interpolation):
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

        def capture_labels(canvas, labels):
            FakeCV2.lines.extend(label[0] for label in labels)
            return canvas

        with tempfile.TemporaryDirectory() as temp:
            log_path = Path(temp) / "events.jsonl"
            args = SimpleNamespace(sample_seconds=1.0, threshold=0.1, camera=0,
                                   log=str(log_path), device="cpu", mode="hybrid",
                                   min_similarity=0.30, min_margin=0.04,
                                   camera_resolution=(1280, 720), fullscreen=True)
            output = io.StringIO()
            with patch.dict("sys.modules", {"cv2": FakeCV2}), \
                    patch("vision_events.models.VisionModels", return_value=model), \
                    patch("vision_events.dashboard._paint_labels", side_effect=capture_labels), \
                    patch("sys.stdout", output):
                run_demo(args)
            records = [json.loads(line) for line in log_path.read_text().splitlines()]
        self.assertEqual(records[0]["inference"]["task"], "Drinking from a cup.")
        self.assertTrue(camera.released)
        self.assertTrue(model.closed)
        self.assertTrue(FakeCV2.destroyed)
        self.assertEqual(camera.settings, [(FakeCV2.CAP_PROP_FRAME_WIDTH, 1280),
                                           (FakeCV2.CAP_PROP_FRAME_HEIGHT, 720)])
        self.assertEqual(FakeCV2.properties, [("Event-triggered task demo", FakeCV2.WND_PROP_FULLSCREEN,
                                                FakeCV2.WINDOW_FULLSCREEN)])
        self.assertIn("Actual webcam frames: 2x2 (different from the requested size)", output.getvalue())
        self.assertGreaterEqual(FakeCV2.lines.count("RECENT OBSERVATIONS"), 2)
        self.assertIn("EXPANDED DETAILS", FakeCV2.lines)

    def test_camera_resolution_argument(self):
        self.assertEqual(parse_camera_resolution("1280x720"), (1280, 720))
        self.assertEqual(parse_camera_resolution("1920X1080"), (1920, 1080))
        for invalid in ("bad", "1280", "0x720", "-1x720", "640x480x30"):
            with self.subTest(invalid=invalid), self.assertRaises(argparse.ArgumentTypeError):
                parse_camera_resolution(invalid)


if __name__ == "__main__":
    unittest.main()
