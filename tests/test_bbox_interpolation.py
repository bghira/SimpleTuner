import unittest

from simpletuner.helpers.training.grounding.interpolation import interpolate_bbox_keyframes


class InterpolateBboxKeyframesTestCase(unittest.TestCase):

    def test_single_keyframe(self):
        keyframes = [
            {"frame": 0, "entities": [{"label": "cat", "bbox": [0.1, 0.2, 0.5, 0.6]}]},
        ]
        result = interpolate_bbox_keyframes(keyframes, num_frames=5)
        self.assertEqual(len(result), 5)
        for frame_entities in result:
            self.assertEqual(len(frame_entities), 1)
            self.assertEqual(frame_entities[0]["label"], "cat")
            self.assertEqual(frame_entities[0]["bbox"], [0.1, 0.2, 0.5, 0.6])

    def test_two_keyframes_linear(self):
        keyframes = [
            {"frame": 0, "entities": [{"label": "cat", "bbox": [0.0, 0.0, 0.4, 0.4]}]},
            {"frame": 10, "entities": [{"label": "cat", "bbox": [0.2, 0.2, 0.6, 0.6]}]},
        ]
        result = interpolate_bbox_keyframes(keyframes, num_frames=11)
        self.assertEqual(len(result), 11)
        # Frame 0: original
        self.assertEqual(result[0][0]["bbox"], [0.0, 0.0, 0.4, 0.4])
        # Frame 10: target
        self.assertEqual(result[10][0]["bbox"], [0.2, 0.2, 0.6, 0.6])
        # Frame 5: midpoint
        mid = result[5][0]["bbox"]
        for i in range(4):
            self.assertAlmostEqual(mid[i], [0.1, 0.1, 0.5, 0.5][i], places=5)

    def test_label_appears_midway(self):
        keyframes = [
            {"frame": 0, "entities": [{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]}]},
            {
                "frame": 5,
                "entities": [
                    {"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]},
                    {"label": "dog", "bbox": [0.6, 0.6, 0.9, 0.9]},
                ],
            },
        ]
        result = interpolate_bbox_keyframes(keyframes, num_frames=10)
        # Dog has only one keyframe (frame 5), so it holds before and after
        self.assertEqual(result[0][0]["label"], "cat")
        # "dog" appears in all frames since it's in the label set, held at frame 5 values
        dog_at_0 = next(e for e in result[0] if e["label"] == "dog")
        self.assertEqual(dog_at_0["bbox"], [0.6, 0.6, 0.9, 0.9])
        dog_at_9 = next(e for e in result[9] if e["label"] == "dog")
        self.assertEqual(dog_at_9["bbox"], [0.6, 0.6, 0.9, 0.9])

    def test_label_disappears(self):
        keyframes = [
            {
                "frame": 0,
                "entities": [
                    {"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]},
                    {"label": "dog", "bbox": [0.6, 0.6, 0.9, 0.9]},
                ],
            },
            {
                "frame": 10,
                "entities": [
                    {"label": "cat", "bbox": [0.2, 0.2, 0.6, 0.6]},
                ],
            },
        ]
        result = interpolate_bbox_keyframes(keyframes, num_frames=11)
        # Dog has only one keyframe (frame 0), holds at last position
        dog_at_10 = next(e for e in result[10] if e["label"] == "dog")
        self.assertEqual(dog_at_10["bbox"], [0.6, 0.6, 0.9, 0.9])

    def test_out_of_range_frame_clamped(self):
        keyframes = [
            {"frame": 0, "entities": [{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]}]},
            {"frame": 100, "entities": [{"label": "cat", "bbox": [0.9, 0.9, 1.0, 1.0]}]},
        ]
        result = interpolate_bbox_keyframes(keyframes, num_frames=10)
        self.assertEqual(len(result), 10)
        # Frame 100 is clamped to frame 9
        # So interpolation from 0 to 9 across 10 frames
        self.assertEqual(result[0][0]["bbox"], [0.1, 0.1, 0.5, 0.5])
        self.assertEqual(result[9][0]["bbox"], [0.9, 0.9, 1.0, 1.0])

    def test_empty_keyframes(self):
        result = interpolate_bbox_keyframes([], num_frames=5)
        self.assertEqual(len(result), 5)
        for frame_entities in result:
            self.assertEqual(frame_entities, [])

    def test_zero_frames(self):
        result = interpolate_bbox_keyframes(
            [{"frame": 0, "entities": [{"label": "a", "bbox": [0, 0, 1, 1]}]}],
            num_frames=0,
        )
        self.assertEqual(result, [])

    def test_multiple_labels_interpolated_independently(self):
        keyframes = [
            {
                "frame": 0,
                "entities": [
                    {"label": "a", "bbox": [0.0, 0.0, 0.2, 0.2]},
                    {"label": "b", "bbox": [0.8, 0.8, 1.0, 1.0]},
                ],
            },
            {
                "frame": 4,
                "entities": [
                    {"label": "a", "bbox": [0.4, 0.4, 0.6, 0.6]},
                    {"label": "b", "bbox": [0.0, 0.0, 0.2, 0.2]},
                ],
            },
        ]
        result = interpolate_bbox_keyframes(keyframes, num_frames=5)
        # At frame 2 (midpoint): both labels should be at midpoint values
        a_mid = next(e for e in result[2] if e["label"] == "a")
        b_mid = next(e for e in result[2] if e["label"] == "b")
        for i in range(4):
            self.assertAlmostEqual(a_mid["bbox"][i], [0.2, 0.2, 0.4, 0.4][i], places=5)
            self.assertAlmostEqual(b_mid["bbox"][i], [0.4, 0.4, 0.6, 0.6][i], places=5)

    def test_three_keyframes(self):
        keyframes = [
            {"frame": 0, "entities": [{"label": "x", "bbox": [0.0, 0.0, 0.2, 0.2]}]},
            {"frame": 4, "entities": [{"label": "x", "bbox": [0.4, 0.4, 0.6, 0.6]}]},
            {"frame": 8, "entities": [{"label": "x", "bbox": [0.0, 0.0, 0.2, 0.2]}]},
        ]
        result = interpolate_bbox_keyframes(keyframes, num_frames=9)
        # Frame 2: midpoint between kf0 and kf1
        mid1 = result[2][0]["bbox"]
        for i in range(4):
            self.assertAlmostEqual(mid1[i], [0.2, 0.2, 0.4, 0.4][i], places=5)
        # Frame 6: midpoint between kf1 and kf2
        mid2 = result[6][0]["bbox"]
        for i in range(4):
            self.assertAlmostEqual(mid2[i], [0.2, 0.2, 0.4, 0.4][i], places=5)

    def test_unsorted_keyframes_sorted_internally(self):
        keyframes = [
            {"frame": 10, "entities": [{"label": "a", "bbox": [0.5, 0.5, 0.7, 0.7]}]},
            {"frame": 0, "entities": [{"label": "a", "bbox": [0.1, 0.1, 0.3, 0.3]}]},
        ]
        result = interpolate_bbox_keyframes(keyframes, num_frames=11)
        # Frame 0 should have first keyframe values
        self.assertEqual(result[0][0]["bbox"], [0.1, 0.1, 0.3, 0.3])
        self.assertEqual(result[10][0]["bbox"], [0.5, 0.5, 0.7, 0.7])


if __name__ == "__main__":
    unittest.main()
