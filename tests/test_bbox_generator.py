"""
Tests for BboxGenerator (Florence-2 backend): coordinate normalisation,
degenerate bbox filtering, resume logic, empty labels, detection mode
selection, and round-trip validation against BboxMetadata.from_string().
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from simpletuner.helpers.data_generation.bbox_generator import BboxGenerator
from simpletuner.helpers.training.grounding.metadata import BboxMetadata


def _make_accelerator(device="cpu"):
    acc = MagicMock()
    acc.device = device
    acc.main_process_first.return_value.__enter__ = MagicMock(return_value=None)
    acc.main_process_first.return_value.__exit__ = MagicMock(return_value=False)
    return acc


class TestPostprocess(unittest.TestCase):
    """Test BboxGenerator._postprocess: normalisation and degenerate filtering."""

    def test_normalise_coordinates(self):
        raw = {"bboxes": [[100, 200, 300, 400]], "labels": ["cat"]}
        result = BboxGenerator._postprocess(raw, img_w=1000, img_h=1000)
        self.assertEqual(len(result), 1)
        bbox = result[0]["bbox"]
        self.assertAlmostEqual(bbox[0], 0.1)
        self.assertAlmostEqual(bbox[1], 0.2)
        self.assertAlmostEqual(bbox[2], 0.3)
        self.assertAlmostEqual(bbox[3], 0.4)
        self.assertEqual(result[0]["label"], "cat")

    def test_degenerate_bbox_filtered(self):
        raw = {"bboxes": [[500, 0, 500, 500]], "labels": ["cat"]}
        result = BboxGenerator._postprocess(raw, img_w=1000, img_h=1000)
        self.assertEqual(len(result), 0)

    def test_inverted_bbox_filtered(self):
        raw = {"bboxes": [[800, 0, 200, 500]], "labels": ["cat"]}
        result = BboxGenerator._postprocess(raw, img_w=1000, img_h=1000)
        self.assertEqual(len(result), 0)

    def test_clamping(self):
        raw = {"bboxes": [[-50, -50, 1050, 1050]], "labels": ["cat"]}
        result = BboxGenerator._postprocess(raw, img_w=1000, img_h=1000)
        self.assertEqual(len(result), 1)
        bbox = result[0]["bbox"]
        self.assertAlmostEqual(bbox[0], 0.0)
        self.assertAlmostEqual(bbox[1], 0.0)
        self.assertAlmostEqual(bbox[2], 1.0)
        self.assertAlmostEqual(bbox[3], 1.0)

    def test_multiple_detections(self):
        raw = {
            "bboxes": [[0, 0, 500, 500], [500, 500, 1000, 1000]],
            "labels": ["cat", "dog"],
        }
        result = BboxGenerator._postprocess(raw, img_w=1000, img_h=1000)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["label"], "cat")
        self.assertEqual(result[1]["label"], "dog")

    def test_empty_input(self):
        raw = {"bboxes": [], "labels": []}
        result = BboxGenerator._postprocess(raw, img_w=1000, img_h=1000)
        self.assertEqual(result, [])

    def test_malformed_bbox_skipped(self):
        raw = {"bboxes": [[1, 2, 3]], "labels": ["bad"]}
        result = BboxGenerator._postprocess(raw, img_w=1000, img_h=1000)
        self.assertEqual(len(result), 0)


class TestDetectionModeSelection(unittest.TestCase):
    """Test that the correct Florence-2 task is selected based on labels."""

    def test_labels_provided_uses_open_vocabulary(self):
        gen = BboxGenerator(
            config={"labels": ["person", "dog"]},
            accelerator=_make_accelerator(),
        )
        gen._run_florence2 = MagicMock(
            return_value={"<OPEN_VOCABULARY_DETECTION>": {"bboxes": [[10, 20, 50, 80]], "labels": ["person"]}}
        )
        result = gen._detect_open_vocabulary(MagicMock())
        gen._run_florence2.assert_called_once()
        call_args = gen._run_florence2.call_args
        self.assertEqual(call_args[0][1], "<OPEN_VOCABULARY_DETECTION>")
        self.assertEqual(call_args[0][2], "person, dog")

    def test_no_labels_uses_caption_grounding(self):
        gen = BboxGenerator(
            config={"labels": []},
            accelerator=_make_accelerator(),
        )
        gen._run_florence2 = MagicMock(
            side_effect=[
                {"<CAPTION>": "a cat sitting on a mat"},
                {"<CAPTION_TO_PHRASE_GROUNDING>": {"bboxes": [[10, 20, 50, 80]], "labels": ["cat"]}},
            ]
        )
        result = gen._detect_caption_grounding(MagicMock())
        self.assertEqual(gen._run_florence2.call_count, 2)
        first_call = gen._run_florence2.call_args_list[0]
        self.assertEqual(first_call[0][1], "<CAPTION>")
        second_call = gen._run_florence2.call_args_list[1]
        self.assertEqual(second_call[0][1], "<CAPTION_TO_PHRASE_GROUNDING>")
        self.assertEqual(second_call[0][2], "a cat sitting on a mat")

    def test_empty_caption_returns_empty(self):
        gen = BboxGenerator(config={}, accelerator=_make_accelerator())
        gen._run_florence2 = MagicMock(return_value={"<CAPTION>": ""})
        result = gen._detect_caption_grounding(MagicMock())
        self.assertEqual(result, {"bboxes": [], "labels": []})
        self.assertEqual(gen._run_florence2.call_count, 1)


class TestResumeLogic(unittest.TestCase):
    """Test that images with existing .bbox files are skipped."""

    def test_skips_existing_bbox_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img1 = Path(tmpdir) / "a.png"
            img2 = Path(tmpdir) / "b.png"
            img1.write_bytes(b"fake")
            img2.write_bytes(b"fake")
            img1.with_suffix(".bbox").write_text("[]")

            gen = BboxGenerator(config={}, accelerator=_make_accelerator())
            paths = gen._collect_image_paths(Path(tmpdir))
            pending = [p for p in paths if not p.with_suffix(".bbox").exists()]
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].name, "b.png")


class TestWriteBboxFile(unittest.TestCase):
    """Test _write_bbox_file creates correct JSON."""

    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "photo.jpg"
            img_path.write_bytes(b"fake")
            detections = [
                {"label": "cat", "bbox": [0.1, 0.2, 0.5, 0.8]},
                {"label": "dog", "bbox": [0.5, 0.1, 0.9, 0.9]},
            ]
            BboxGenerator._write_bbox_file(img_path, detections)

            bbox_path = img_path.with_suffix(".bbox")
            self.assertTrue(bbox_path.exists())
            parsed = json.loads(bbox_path.read_text(encoding="utf-8"))
            self.assertEqual(len(parsed), 2)
            self.assertEqual(parsed[0]["label"], "cat")

    def test_empty_detections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "empty.jpg"
            img_path.write_bytes(b"fake")
            BboxGenerator._write_bbox_file(img_path, [])
            content = img_path.with_suffix(".bbox").read_text()
            self.assertEqual(json.loads(content), [])


class TestRoundTrip(unittest.TestCase):
    """Test that BboxGenerator output passes BboxMetadata.from_string() validation."""

    def test_output_parseable(self):
        detections = [
            {"label": "person", "bbox": [0.1, 0.2, 0.5, 0.8]},
            {"label": "car", "bbox": [0.3, 0.1, 0.7, 0.6]},
        ]
        json_str = json.dumps(detections, indent=2)
        entities = BboxMetadata.from_string(json_str)
        self.assertEqual(len(entities), 2)
        self.assertEqual(entities[0].label, "person")
        self.assertEqual(entities[0].bbox, (0.1, 0.2, 0.5, 0.8))

    def test_empty_output_parseable(self):
        entities = BboxMetadata.from_string(json.dumps([]))
        self.assertEqual(entities, [])


class TestLabelsParsing(unittest.TestCase):
    """Test that labels can be provided as a list or comma-separated string."""

    def test_list_labels(self):
        gen = BboxGenerator(
            config={"labels": ["cat", "dog"]},
            accelerator=_make_accelerator(),
        )
        self.assertEqual(gen.labels, ["cat", "dog"])

    def test_string_labels(self):
        gen = BboxGenerator(
            config={"labels": "cat, dog, bird"},
            accelerator=_make_accelerator(),
        )
        self.assertEqual(gen.labels, ["cat", "dog", "bird"])

    def test_empty_string_labels(self):
        gen = BboxGenerator(
            config={"labels": ""},
            accelerator=_make_accelerator(),
        )
        self.assertEqual(gen.labels, [])

    def test_default_model(self):
        gen = BboxGenerator(config={}, accelerator=_make_accelerator())
        self.assertEqual(gen.model_name, "microsoft/Florence-2-large")


class TestGenerateEndToEnd(unittest.TestCase):
    """Integration test of generate() with mocked Florence-2 model."""

    @patch("simpletuner.helpers.data_generation.bbox_generator.BboxGenerator._load_model")
    @patch("simpletuner.helpers.data_generation.bbox_generator.BboxGenerator._unload_model")
    def test_generate_writes_bbox_files(self, mock_unload, mock_load):
        with tempfile.TemporaryDirectory() as tmpdir:
            from PIL import Image

            for name in ["a.png", "b.jpg"]:
                Image.new("RGB", (100, 100)).save(str(Path(tmpdir) / name))

            gen = BboxGenerator(
                config={"labels": ["cat"], "batch_size": 2},
                accelerator=_make_accelerator(),
            )

            # Mock _run_florence2 to return fixed open vocabulary detections
            gen._run_florence2 = MagicMock(
                return_value={
                    "<OPEN_VOCABULARY_DETECTION>": {
                        "bboxes": [[10, 20, 50, 80]],
                        "labels": ["cat"],
                    }
                }
            )

            count = gen.generate(instance_data_dir=tmpdir)
            self.assertEqual(count, 2)

            for name in ["a.bbox", "b.bbox"]:
                bbox_path = Path(tmpdir) / name
                self.assertTrue(bbox_path.exists(), f"{name} should exist")
                entities = BboxMetadata.from_file(str(bbox_path))
                self.assertEqual(len(entities), 1)
                self.assertEqual(entities[0].label, "cat")

    @patch("simpletuner.helpers.data_generation.bbox_generator.BboxGenerator._load_model")
    @patch("simpletuner.helpers.data_generation.bbox_generator.BboxGenerator._unload_model")
    def test_generate_skips_existing(self, mock_unload, mock_load):
        with tempfile.TemporaryDirectory() as tmpdir:
            from PIL import Image

            Image.new("RGB", (100, 100)).save(str(Path(tmpdir) / "a.png"))
            Image.new("RGB", (100, 100)).save(str(Path(tmpdir) / "b.png"))

            (Path(tmpdir) / "a.bbox").write_text('[{"label": "existing", "bbox": [0.1, 0.1, 0.5, 0.5]}]')

            gen = BboxGenerator(
                config={"labels": ["cat"]},
                accelerator=_make_accelerator(),
            )
            gen._run_florence2 = MagicMock(
                return_value={
                    "<OPEN_VOCABULARY_DETECTION>": {
                        "bboxes": [[10, 20, 50, 80]],
                        "labels": ["cat"],
                    }
                }
            )

            count = gen.generate(instance_data_dir=tmpdir)
            self.assertEqual(count, 1)

            a_entities = BboxMetadata.from_file(str(Path(tmpdir) / "a.bbox"))
            self.assertEqual(a_entities[0].label, "existing")

    @patch("simpletuner.helpers.data_generation.bbox_generator.BboxGenerator._load_model")
    @patch("simpletuner.helpers.data_generation.bbox_generator.BboxGenerator._unload_model")
    def test_generate_caption_grounding_mode(self, mock_unload, mock_load):
        with tempfile.TemporaryDirectory() as tmpdir:
            from PIL import Image

            Image.new("RGB", (100, 100)).save(str(Path(tmpdir) / "a.png"))

            gen = BboxGenerator(
                config={"labels": []},
                accelerator=_make_accelerator(),
            )
            gen._run_florence2 = MagicMock(
                side_effect=[
                    {"<CAPTION>": "a dog playing in the park"},
                    {
                        "<CAPTION_TO_PHRASE_GROUNDING>": {
                            "bboxes": [[5, 10, 60, 90]],
                            "labels": ["dog"],
                        }
                    },
                ]
            )

            count = gen.generate(instance_data_dir=tmpdir)
            self.assertEqual(count, 1)

            entities = BboxMetadata.from_file(str(Path(tmpdir) / "a.bbox"))
            self.assertEqual(len(entities), 1)
            self.assertEqual(entities[0].label, "dog")

    def test_generate_invalid_dir_raises(self):
        gen = BboxGenerator(config={}, accelerator=_make_accelerator())
        with self.assertRaises(ValueError):
            gen.generate(instance_data_dir="/nonexistent/path/that/does/not/exist")


if __name__ == "__main__":
    unittest.main()
