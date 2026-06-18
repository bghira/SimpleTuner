import unittest
from types import SimpleNamespace

import torch

from simpletuner.helpers.models.common import PipelineTypes
from simpletuner.helpers.models.qwen_image.model import QwenImage
from simpletuner.helpers.models.qwen_image.pipeline import _pad_qwen_cfg_prompt_tensors
from simpletuner.helpers.models.qwen_image.pipeline import QwenImagePipeline as SimpleTunerQwenImagePipeline


class QwenImageCFGBatchingTests(unittest.TestCase):
    def test_qwen_image_model_uses_batched_cfg_pipeline(self):
        self.assertIs(QwenImage.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG], SimpleTunerQwenImagePipeline)

    def test_cached_text_embeds_keep_masks_through_cfg_batching(self):
        model = QwenImage.__new__(QwenImage)
        model.config = SimpleNamespace(validation_guidance_real=4.0)

        positive_cached = {
            "prompt_embeds": torch.arange(1 * 5 * 2, dtype=torch.float32).view(1, 5, 2),
            "attention_masks": torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.int64),
        }
        negative_cached = {
            "prompt_embeds": torch.full((1, 3, 2), -1.0),
            "attention_masks": torch.tensor([[1, 1, 0]], dtype=torch.int64),
        }

        positive = model.convert_text_embed_for_pipeline(positive_cached)
        negative = model.convert_negative_text_embed_for_pipeline(negative_cached)
        cfg_embeds, cfg_mask = _pad_qwen_cfg_prompt_tensors(
            negative_prompt_embeds=negative["negative_prompt_embeds"],
            prompt_embeds=positive["prompt_embeds"],
            negative_prompt_embeds_mask=negative["negative_prompt_embeds_mask"],
            prompt_embeds_mask=positive["prompt_embeds_mask"],
        )

        self.assertEqual(cfg_embeds.shape, torch.Size([2, 5, 2]))
        self.assertEqual(cfg_mask.dtype, torch.int64)
        self.assertTrue(torch.equal(cfg_mask[0], torch.tensor([1, 1, 0, 0, 0])))
        self.assertTrue(torch.equal(cfg_mask[1], torch.tensor([1, 1, 1, 0, 0])))
        self.assertTrue(torch.equal(cfg_embeds[0, 3:], torch.zeros_like(cfg_embeds[0, 3:])))
        self.assertEqual(negative["true_cfg_scale"], 4.0)

    def test_cfg_prompt_tensors_pad_variable_lengths(self):
        negative_prompt_embeds = torch.ones(1, 2, 3)
        prompt_embeds = torch.full((1, 5, 3), 2.0)
        negative_mask = torch.tensor([[True, True]])
        prompt_mask = torch.tensor([[True, True, True, False, False]])

        cfg_embeds, cfg_mask = _pad_qwen_cfg_prompt_tensors(
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds_mask=negative_mask,
            prompt_embeds_mask=prompt_mask,
        )

        self.assertEqual(cfg_embeds.shape, torch.Size([2, 5, 3]))
        self.assertEqual(cfg_mask.shape, torch.Size([2, 5]))
        self.assertTrue(torch.equal(cfg_embeds[0, :2], negative_prompt_embeds[0]))
        self.assertTrue(torch.equal(cfg_embeds[0, 2:], torch.zeros_like(cfg_embeds[0, 2:])))
        self.assertTrue(torch.equal(cfg_embeds[1], prompt_embeds[0]))
        self.assertTrue(torch.equal(cfg_mask[0], torch.tensor([True, True, False, False, False])))
        self.assertTrue(torch.equal(cfg_mask[1], prompt_mask[0]))

    def test_cfg_prompt_tensors_synthesize_missing_mask_with_padding(self):
        negative_prompt_embeds = torch.ones(1, 3, 2)
        prompt_embeds = torch.full((1, 5, 2), 2.0)

        cfg_embeds, cfg_mask = _pad_qwen_cfg_prompt_tensors(
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds_mask=None,
            prompt_embeds_mask=torch.tensor([[True, True, False, False, False]]),
        )

        self.assertEqual(cfg_embeds.shape, torch.Size([2, 5, 2]))
        self.assertTrue(torch.equal(cfg_mask[0], torch.tensor([True, True, True, False, False])))
        self.assertTrue(torch.equal(cfg_mask[1], torch.tensor([True, True, False, False, False])))

    def test_cfg_prompt_tensors_keep_mask_none_when_both_unmasked(self):
        negative_prompt_embeds = torch.ones(2, 4, 3)
        prompt_embeds = torch.full((2, 4, 3), 2.0)

        cfg_embeds, cfg_mask = _pad_qwen_cfg_prompt_tensors(
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds_mask=None,
            prompt_embeds_mask=None,
        )

        self.assertEqual(cfg_embeds.shape, torch.Size([4, 4, 3]))
        self.assertIsNone(cfg_mask)
        self.assertTrue(torch.equal(cfg_embeds[:2], negative_prompt_embeds))
        self.assertTrue(torch.equal(cfg_embeds[2:], prompt_embeds))


if __name__ == "__main__":
    unittest.main()
