import unittest

import numpy as np

from vision_events.context_memory import ContextMemory
from vision_events.task_demo import TaskDemoEngine, cacheable_description


class ContextMemoryTests(unittest.TestCase):
    def test_cold_start_requires_two_distinct_descriptions(self):
        memory = ContextMemory(0.3, 0.04)
        self.assertEqual(memory.match([1, 0, 0]).reason, "empty")
        self.assertTrue(memory.add("Typing at keyboard", [2, 0, 0]))
        first = memory.match([1, 0, 0])
        self.assertEqual(first.reason, "need_more_examples")
        self.assertIsNone(first.description)
        self.assertEqual(first.candidate, "Typing at keyboard")
        self.assertAlmostEqual(first.best_similarity, 1.0)
        self.assertFalse(memory.add("typing at keyboard", [2, 0, 0]))
        self.assertEqual(len(memory), 1)
        self.assertTrue(memory.add("Phone at ear", [0, 3, 0]))
        match = memory.match([4, 0, 0])
        self.assertEqual(match.reason, "matched")
        self.assertEqual(match.description, "Typing at keyboard")
        self.assertAlmostEqual(match.best_similarity, 1.0)
        self.assertAlmostEqual(match.margin, 1.0)

    def test_unknown_and_ambiguous_frames_fall_back(self):
        memory = ContextMemory(0.3, 0.04)
        memory.add("Keyboard", [1, 0, 0])
        memory.add("Phone", [0, 1, 0])
        unknown = memory.match([0, 0, 1])
        self.assertEqual(unknown.reason, "low_similarity")
        self.assertIsNone(unknown.description)
        ambiguous = memory.match([1, 1, 0])
        self.assertEqual(ambiguous.reason, "ambiguous")
        self.assertIsNone(ambiguous.description)
        self.assertAlmostEqual(ambiguous.margin, 0.0)

    def test_invalid_embeddings_and_thresholds_are_rejected(self):
        for similarity in (-1.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                ContextMemory(similarity, 0.04)
        for margin in (-0.1, 2.1, float("nan")):
            with self.assertRaises(ValueError):
                ContextMemory(0.3, margin)
        memory = ContextMemory()
        for vector in ([0, 0], [float("nan"), 1], [[1, 2]]):
            with self.assertRaises(ValueError):
                memory.add("Invalid", vector)
        memory.add("Valid", [1, 0])
        with self.assertRaises(ValueError):
            memory.match([1, 0, 0])
        with self.assertRaises(ValueError):
            memory.add("Wrong dimensions", [1, 0, 0])
        self.assertEqual(len(memory), 1)

    def test_explicitly_uncertain_output_is_not_cached(self):
        for response in ("", "I cannot tell what the person is doing.",
                         "The activity is unclear.", "I can't see the person's hands."):
            self.assertFalse(cacheable_description(response))
        self.assertTrue(cacheable_description("The person is holding a phone."))

    def test_hybrid_reuses_returning_context_and_calls_vlm_for_uncertain_matches(self):
        class FakeHybridModel:
            def __init__(self):
                self.responses = iter([
                    "Typing on a keyboard.", "Holding a phone to their ear.",
                    "I cannot tell.", "Stirring a bowl.",
                ])
                self.queried = []
                self.encoded_descriptions = []

            def encode_frame(self, frame):
                index = int(frame[0, 0, 0])
                vector = {0: [1, 0, 0], 1: [0, 1, 0], 2: [1, 0, 0],
                          3: [1, 1, 0], 4: [0, 0, 1]}[index]
                images = {region: f"{region}-{index}" for region in ("full", "person", "background")}
                return images, {"full": vector, "person": vector, "background": [1, 0, 0]}

            def describe(self, image, question, max_new_tokens=80):
                self.queried.append(image)
                return next(self.responses)

            def encode_texts(self, descriptions):
                self.encoded_descriptions.extend(descriptions)
                vectors = {"Typing on a keyboard.": [1, 0, 0],
                           "Holding a phone to their ear.": [0, 1, 0],
                           "Stirring a bowl.": [0, 0, 1]}
                return np.array([vectors[text] for text in descriptions])

        def frame(index):
            return np.full((2, 2, 3), index, dtype=np.uint8)

        model = FakeHybridModel()
        engine = TaskDemoEngine(model, 0.1)
        baseline, _ = engine.process(frame(0), 0.0)
        self.assertEqual(baseline["recognition"]["source"], "vlm")
        self.assertTrue(baseline["recognition"]["cache_added"])

        phone, _ = engine.process(frame(1), 1.0)
        self.assertEqual(phone["recognition"]["source"], "vlm")
        self.assertEqual(phone["recognition"]["match"]["reason"], "need_more_examples")
        self.assertEqual(phone["recognition"]["match"]["best_description"], "Typing on a keyboard.")

        returned, state = engine.process(frame(2), 2.0)
        self.assertIsNone(returned["inference"])
        self.assertEqual(returned["recognition"]["source"], "cache")
        self.assertEqual(returned["recognition"]["description"], "Typing on a keyboard.")
        self.assertEqual((state["vlm_calls"], state["cache_hits"], state["cache_size"]), (2, 1, 2))
        self.assertGreaterEqual(returned["timings_ms"]["context_resolution"], 0.0)
        self.assertIsNone(returned["timings_ms"]["vlm"])

        ambiguous, _ = engine.process(frame(3), 3.0)
        self.assertEqual(ambiguous["recognition"]["source"], "vlm")
        self.assertEqual(ambiguous["recognition"]["match"]["reason"], "ambiguous")
        self.assertFalse(ambiguous["recognition"]["cache_added"])
        self.assertEqual(ambiguous["recognition"]["cache_size"], 2)

        novel, state = engine.process(frame(4), 4.0)
        self.assertEqual(novel["recognition"]["source"], "vlm")
        self.assertEqual(novel["recognition"]["match"]["reason"], "low_similarity")
        self.assertTrue(novel["recognition"]["cache_added"])
        self.assertEqual((state["vlm_calls"], state["cache_hits"], state["cache_size"]), (4, 1, 3))
        self.assertEqual(model.queried, ["full-0", "full-1", "full-3", "full-4"])
        self.assertEqual(len(model.encoded_descriptions), 3)


if __name__ == "__main__":
    unittest.main()
