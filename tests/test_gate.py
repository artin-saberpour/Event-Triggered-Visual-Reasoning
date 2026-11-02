import unittest

from vision_events.gate import RegionGate


def vectors(full=(1, 0), person=(1, 0), background=(1, 0)):
    return {"full": full, "person": person, "background": background}


class GateTests(unittest.TestCase):
    def setUp(self):
        self.gate = RegionGate(0.1)
        self.assertTrue(self.gate.observe(vectors()).baseline)

    def test_unchanged_frame_does_not_trigger(self):
        self.assertIsNone(self.gate.observe(vectors()).region)

    def test_person_and_background_route_separately(self):
        self.assertEqual(self.gate.observe(vectors(person=(0, 1))).region, "person")
        self.assertIsNone(self.gate.observe(vectors(person=(0, 1))).region)
        self.assertEqual(self.gate.observe(vectors(person=(0, 1), background=(0, 1))).region, "background")

    def test_simultaneous_change_refreshes_both_region_references(self):
        changed = vectors(full=(0, 1), person=(0, 1), background=(0, 1))
        self.assertEqual(self.gate.observe(changed).region, "full")
        self.assertIsNone(self.gate.observe(changed).region)

    def test_subthreshold_changes_accumulate_until_trigger(self):
        self.assertIsNone(self.gate.observe(vectors(person=(1, 0.2))).region)
        self.assertEqual(self.gate.observe(vectors(person=(1, 0.8))).region, "person")

    def test_invalid_vectors_do_not_corrupt_references(self):
        for bad in [(0, 0), (float("nan"), 1), (1, 2, 3)]:
            with self.assertRaises(ValueError):
                self.gate.observe(vectors(person=bad))
        self.assertIsNone(self.gate.observe(vectors()).region)

    def test_rejects_invalid_thresholds(self):
        for threshold in [-1, 2.1, float("nan")]:
            with self.assertRaises(ValueError):
                RegionGate(threshold)


if __name__ == "__main__":
    unittest.main()
