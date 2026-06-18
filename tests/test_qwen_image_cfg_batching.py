import unittest

import torch

from simpletuner.helpers.models.common import PipelineTypes
from simpletuner.helpers.models.qwen_image.model import QwenImage
from simpletuner.helpers.models.qwen_image.pipeline import _pad_qwen_cfg_prompt_tensors
from simpletuner.helpers.models.qwen_image.pipeline import QwenImagePipeline as SimpleTunerQwenImagePipeline


class QwenImageCFGBatchingTests(unittest.TestCase):
    def test_qwen_image_model_uses_batched_cfg_pipeline(self):
        self.assertIs(QwenImage.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG], SimpleTunerQwenImagePipeline)

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
