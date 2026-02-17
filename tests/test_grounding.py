"""
Tests for the grounding pipeline: BboxMetadata parsing, GroundingCollate, and downsample_mask.
"""

import json
import os
import tempfile
import unittest

import torch

from simpletuner.helpers.training.grounding.collate import GroundingCollate
from simpletuner.helpers.training.grounding.metadata import BboxMetadata
from simpletuner.helpers.training.grounding.types import BboxEntity, GroundingBatch


class TestBboxMetadataFromString(unittest.TestCase):
    """Test BboxMetadata.from_string with all supported formats."""

    def test_empty_string(self):
        result = BboxMetadata.from_string("")
        self.assertEqual(result, [])

    def test_whitespace_string(self):
        result = BboxMetadata.from_string("   \n  ")
        self.assertEqual(result, [])

    def test_json_array(self):
        data = json.dumps(
            [
                {"label": "cat", "bbox": [0.1, 0.2, 0.5, 0.8]},
                {"label": "dog", "bbox": [0.5, 0.1, 0.9, 0.9], "mask": "masks/dog.png"},
            ]
        )
        result = BboxMetadata.from_string(data)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].label, "cat")
        self.assertEqual(result[0].bbox, (0.1, 0.2, 0.5, 0.8))
        self.assertIsNone(result[0].mask_path)
        self.assertEqual(result[1].label, "dog")
        self.assertEqual(result[1].mask_path, "masks/dog.png")

    def test_json_lines(self):
        data = '{"label": "person", "bbox": [0.0, 0.0, 0.5, 0.5]}\n{"label": "car", "bbox": [0.5, 0.5, 1.0, 1.0]}'
        result = BboxMetadata.from_string(data)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].label, "person")
        self.assertEqual(result[1].label, "car")

    def test_json_lines_with_blank_lines(self):
        data = '{"label": "a", "bbox": [0.1, 0.1, 0.5, 0.5]}\n\n{"label": "b", "bbox": [0.5, 0.5, 0.9, 0.9]}\n'
        result = BboxMetadata.from_string(data)
        self.assertEqual(len(result), 2)

    def test_yolo_txt(self):
        data = "0 0.3 0.5 0.4 0.6\n1 0.7 0.5 0.4 0.8"
        result = BboxMetadata.from_string(data)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].label, "0")
        # XYWH center (0.3, 0.5, 0.4, 0.6) -> XYXY (0.1, 0.2, 0.5, 0.8)
        self.assertAlmostEqual(result[0].bbox[0], 0.1, places=5)
        self.assertAlmostEqual(result[0].bbox[1], 0.2, places=5)
        self.assertAlmostEqual(result[0].bbox[2], 0.5, places=5)
        self.assertAlmostEqual(result[0].bbox[3], 0.8, places=5)

    def test_yolo_malformed_line_skipped(self):
        data = "0 0.3\n1 0.5 0.5 0.2 0.2"
        result = BboxMetadata.from_string(data)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, "1")

    def test_bbox_clamping(self):
        data = json.dumps([{"label": "overflow", "bbox": [-0.1, -0.2, 1.5, 1.3]}])
        result = BboxMetadata.from_string(data)
        self.assertEqual(len(result), 1)
        x1, y1, x2, y2 = result[0].bbox
        self.assertGreaterEqual(x1, 0.0)
        self.assertGreaterEqual(y1, 0.0)
        self.assertLessEqual(x2, 1.0)
        self.assertLessEqual(y2, 1.0)

    def test_invalid_bbox_raises(self):
        # x1 == x2 after clamping -> invalid
        data = json.dumps([{"label": "degenerate", "bbox": [0.5, 0.5, 0.5, 0.5]}])
        with self.assertRaises(ValueError):
            BboxMetadata.from_string(data)

    def test_inverted_bbox_raises(self):
        # x1 > x2 after clamping
        data = json.dumps([{"label": "inverted", "bbox": [0.8, 0.1, 0.2, 0.9]}])
        with self.assertRaises(ValueError):
            BboxMetadata.from_string(data)

    def test_missing_bbox_raises(self):
        data = json.dumps([{"label": "no_bbox"}])
        with self.assertRaises(ValueError):
            BboxMetadata.from_string(data)

    def test_mask_path_alias(self):
        """Both 'mask' and 'mask_path' should be accepted."""
        data = json.dumps([{"label": "a", "bbox": [0.1, 0.1, 0.5, 0.5], "mask_path": "m.png"}])
        result = BboxMetadata.from_string(data)
        self.assertEqual(result[0].mask_path, "m.png")


class TestBboxMetadataFromFile(unittest.TestCase):
    """Test BboxMetadata.from_file."""

    def test_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            BboxMetadata.from_file("/nonexistent/path.bbox")

    def test_reads_json_array(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bbox", delete=False) as f:
            json.dump([{"label": "tree", "bbox": [0.2, 0.3, 0.7, 0.9]}], f)
            path = f.name
        try:
            result = BboxMetadata.from_file(path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].label, "tree")
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bbox", delete=False) as f:
            f.write("")
            path = f.name
        try:
            result = BboxMetadata.from_file(path)
            self.assertEqual(result, [])
        finally:
            os.unlink(path)


class TestGroundingCollateDownsampleMask(unittest.TestCase):
    """Test GroundingCollate.downsample_mask."""

    def test_basic_downsample(self):
        mask = torch.ones(64, 64)
        result = GroundingCollate.downsample_mask(mask, vae_scale_factor=8)
        self.assertEqual(result.shape, (8, 8))
        self.assertTrue(torch.allclose(result, torch.ones(8, 8)))

    def test_3d_input(self):
        mask = torch.ones(1, 32, 32)
        result = GroundingCollate.downsample_mask(mask, vae_scale_factor=4)
        self.assertEqual(result.shape, (8, 8))

    def test_4d_input(self):
        mask = torch.ones(1, 1, 16, 16)
        result = GroundingCollate.downsample_mask(mask, vae_scale_factor=2)
        self.assertEqual(result.shape, (8, 8))


class TestGroundingCollateBboxToMask(unittest.TestCase):
    """Test GroundingCollate._bbox_to_mask."""

    def test_full_image_bbox(self):
        collate = GroundingCollate(max_entities=4, vae_scale_factor=8)
        mask = collate._bbox_to_mask(0.0, 0.0, 1.0, 1.0, h=8, w=8)
        self.assertEqual(mask.shape, (8, 8))
        self.assertTrue(torch.all(mask == 1.0))

    def test_partial_bbox(self):
        collate = GroundingCollate(max_entities=4, vae_scale_factor=8)
        mask = collate._bbox_to_mask(0.0, 0.0, 0.5, 0.5, h=10, w=10)
        self.assertEqual(mask.shape, (10, 10))
        # Top-left quadrant should be 1, rest should be 0
        self.assertTrue(torch.all(mask[:5, :5] == 1.0))
        self.assertTrue(torch.all(mask[5:, :] == 0.0))
        self.assertTrue(torch.all(mask[:, 5:] == 0.0))

    def test_zero_area_bbox(self):
        collate = GroundingCollate(max_entities=4, vae_scale_factor=8)
        # Very small bbox that maps to at least 1 pixel
        mask = collate._bbox_to_mask(0.0, 0.0, 0.01, 0.01, h=8, w=8)
        self.assertEqual(mask.shape, (8, 8))
        # Should have at least one pixel set (min 1 pixel)
        self.assertGreater(mask.sum().item(), 0)


class TestGroundingCollatePoolTextEncoderOutput(unittest.TestCase):
    """Test GroundingCollate._pool_text_encoder_output."""

    def test_with_pooled_prompt_embeds(self):
        output = {"pooled_prompt_embeds": torch.randn(1, 768)}
        result = GroundingCollate._pool_text_encoder_output(output)
        self.assertEqual(result.shape, (768,))

    def test_with_prompt_embeds_and_mask(self):
        seq_len = 10
        dim = 512
        embeds = torch.randn(1, seq_len, dim)
        mask = torch.ones(1, seq_len)
        mask[0, 5:] = 0  # Half the sequence is masked
        output = {"prompt_embeds": embeds, "attention_masks": mask}
        result = GroundingCollate._pool_text_encoder_output(output)
        self.assertEqual(result.shape, (dim,))

    def test_with_prompt_embeds_no_mask(self):
        embeds = torch.randn(1, 10, 256)
        output = {"prompt_embeds": embeds}
        result = GroundingCollate._pool_text_encoder_output(output)
        self.assertEqual(result.shape, (256,))

    def test_empty_output(self):
        result = GroundingCollate._pool_text_encoder_output({})
        self.assertEqual(result.shape, (768,))

    def test_with_attention_mask_key(self):
        """Test that 'attention_mask' (singular) key is recognized."""
        seq_len = 10
        dim = 512
        embeds = torch.randn(1, seq_len, dim)
        mask = torch.ones(1, seq_len)
        mask[0, 5:] = 0
        output = {"prompt_embeds": embeds, "attention_mask": mask}
        result = GroundingCollate._pool_text_encoder_output(output)
        self.assertEqual(result.shape, (dim,))

    def test_with_prompt_attention_mask_key(self):
        """Test that 'prompt_attention_mask' key is recognized."""
        seq_len = 10
        dim = 512
        embeds = torch.randn(1, seq_len, dim)
        mask = torch.ones(1, seq_len)
        mask[0, 7:] = 0
        output = {"prompt_embeds": embeds, "prompt_attention_mask": mask}
        result = GroundingCollate._pool_text_encoder_output(output)
        self.assertEqual(result.shape, (dim,))


class TestGroundingBatchDataclass(unittest.TestCase):
    """Test GroundingBatch dataclass structure."""

    def test_creation(self):
        batch = GroundingBatch(
            boxes=torch.zeros(2, 4, 4),
            validity_mask=torch.zeros(2, 4),
            spatial_masks=torch.zeros(2, 4, 8, 8),
            text_embeds=torch.zeros(2, 4, 768),
            image_embeds=None,
            text_masks=torch.zeros(2, 4),
            image_masks=torch.zeros(2, 4),
            max_entities=4,
        )
        self.assertEqual(batch.boxes.shape, (2, 4, 4))
        self.assertEqual(batch.validity_mask.shape, (2, 4))
        self.assertEqual(batch.spatial_masks.shape, (2, 4, 8, 8))
        self.assertEqual(batch.text_embeds.shape, (2, 4, 768))
        self.assertIsNone(batch.image_embeds)
        self.assertEqual(batch.text_masks.shape, (2, 4))
        self.assertEqual(batch.image_masks.shape, (2, 4))
        self.assertEqual(batch.max_entities, 4)

    def test_creation_with_image_embeds(self):
        batch = GroundingBatch(
            boxes=torch.zeros(2, 4, 4),
            validity_mask=torch.ones(2, 4),
            spatial_masks=torch.zeros(2, 4, 8, 8),
            text_embeds=torch.zeros(2, 4, 768),
            image_embeds=torch.zeros(2, 4, 1024),
            text_masks=torch.ones(2, 4),
            image_masks=torch.ones(2, 4),
            max_entities=4,
        )
        self.assertEqual(batch.image_embeds.shape, (2, 4, 1024))
        self.assertEqual(batch.text_masks.shape, (2, 4))
        self.assertEqual(batch.image_masks.shape, (2, 4))


class TestRandomDropFeatures(unittest.TestCase):
    """Test GroundingCollate._random_drop_features."""

    def test_no_image_mode(self):
        validity = torch.ones(2, 4)
        text_masks, image_masks = GroundingCollate._random_drop_features(validity, has_image=False)
        self.assertTrue(torch.equal(text_masks, validity))
        self.assertTrue(torch.equal(image_masks, torch.zeros_like(validity)))

    def test_with_image_mode_shapes(self):
        validity = torch.ones(2, 4)
        text_masks, image_masks = GroundingCollate._random_drop_features(validity, has_image=True)
        self.assertEqual(text_masks.shape, (2, 4))
        self.assertEqual(image_masks.shape, (2, 4))

    def test_invalid_entities_stay_zero(self):
        validity = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        text_masks, image_masks = GroundingCollate._random_drop_features(validity, has_image=True)
        # Invalid entities must remain 0 in both masks
        self.assertEqual(text_masks[0, 2].item(), 0.0)
        self.assertEqual(text_masks[0, 3].item(), 0.0)
        self.assertEqual(image_masks[0, 2].item(), 0.0)
        self.assertEqual(image_masks[0, 3].item(), 0.0)

    def test_never_drops_both(self):
        """Over many runs, valid entities should never have both masks zeroed."""
        validity = torch.ones(1, 1)
        for _ in range(100):
            text_masks, image_masks = GroundingCollate._random_drop_features(validity, has_image=True)
            # At least one of text or image mask should be active for valid entity
            self.assertGreater(text_masks[0, 0].item() + image_masks[0, 0].item(), 0.0)


class TestTrainingBatchCompat(unittest.TestCase):
    """Test TrainingBatch dict-like access for backwards compatibility."""

    def test_getitem(self):
        from simpletuner.helpers.training.batch_types import TrainingBatch

        batch = TrainingBatch(
            latent_batch=None,
            latent_metadata=None,
            prompts=["test"],
            text_encoder_output={},
            prompt_embeds=None,
            add_text_embeds=None,
            batch_time_ids=None,
            batch_luminance=None,
            conditioning_pixel_values=None,
            conditioning_latents=None,
            conditioning_image_embeds=None,
            conditioning_captions=None,
            encoder_attention_mask=None,
            is_regularisation_data=False,
            is_i2v_data=False,
            conditioning_type=None,
            loss_mask_type=None,
            audio_latent_batch=None,
            audio_latent_mask=None,
            video_latent_mask=None,
            is_audio_only=False,
            s2v_audio_paths=None,
            s2v_audio_backend_ids=None,
            grounding_batch=None,
            slider_strength=None,
        )
        self.assertEqual(batch["prompts"], ["test"])
        self.assertEqual(batch.get("prompts"), ["test"])
        self.assertIsNone(batch.get("nonexistent", None))
        self.assertIn("prompts", batch)
        self.assertIn("grounding_batch", batch.keys())


class TestDatasetTypeGrounding(unittest.TestCase):
    """Test that DatasetType.GROUNDING is properly registered."""

    def test_grounding_enum_exists(self):
        from simpletuner.helpers.data_backend.dataset_types import DatasetType

        self.assertEqual(DatasetType.GROUNDING.value, "grounding")

    def test_grounding_from_value(self):
        from simpletuner.helpers.data_backend.dataset_types import DatasetType

        result = DatasetType.from_value("grounding")
        self.assertEqual(result, DatasetType.GROUNDING)


if __name__ == "__main__":
    unittest.main()
