import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from vision_events.gate import RegionGate


spec = importlib.util.spec_from_file_location("record_captures", Path(__file__).resolve().parents[1] / "scripts" / "record_captures.py")
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


class FakeVisionModel:
    def encode_frame(self, bgr):
        image = Image.fromarray(bgr[:, :, ::-1].copy())
        vector = [1, 0] if bgr.mean() < 100 else [0, 1]
        return {key: image.copy() for key in ("full", "person", "background")}, {
            key: vector for key in ("full", "person", "background")
        }


class CaptureReportTests(unittest.TestCase):
    def test_report_pairs_each_input_with_its_decision_and_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = []
            for index, color in enumerate((10, 200, 200)):
                path = root / f"frame_{index}.png"
                Image.new("RGB", (4, 4), (color, color, color)).save(path)
                paths.append(path)
            output = root / "report"
            records = recorder.record_frames(paths, FakeVisionModel(), RegionGate(0.1), output)
            self.assertEqual([r["selected_region"] for r in records], [None, "full", None])
            self.assertTrue(records[0]["baseline"])
            self.assertAlmostEqual(records[1]["cosine_distances"]["person"], 1.0)
            self.assertAlmostEqual(records[2]["cosine_distances"]["person"], 0.0)
            logged = [json.loads(row) for row in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual(logged, records)
            for record in records:
                for image_path in record["images"].values():
                    self.assertTrue((output / image_path).is_file())
            self.assertIn("frame_1.png", (output / "report.html").read_text())
            manifest = json.loads((output / "run.json").read_text())
            self.assertFalse(manifest["vlm_enabled"])
            self.assertEqual(manifest["clip_revision"], recorder.CLIP_SAFE_REVISION)

    def test_does_not_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "frame.png"
            Image.new("RGB", (2, 2)).save(source)
            with self.assertRaises(FileExistsError):
                recorder.record_frames([source], FakeVisionModel(), RegionGate(), root)


if __name__ == "__main__":
    unittest.main()
