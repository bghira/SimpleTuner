import unittest
from types import SimpleNamespace

import torch
from PIL import Image

from simpletuner.helpers.models.common import PipelineTypes
from simpletuner.helpers.models.qwen_image.model import QwenImage


class NoImageEncodePipeline:
    def __init__(self):
        self.calls = []

    def encode_prompt(self, prompt, device=None, num_images_per_prompt=1):
        self.calls.append(
            {
                "prompt": prompt,
                "device": device,
                "num_images_per_prompt": num_images_per_prompt,
            }
        )
        batch = len(prompt)
        prompt_embeds = torch.ones(batch, 2, 4)
        prompt_mask = torch.ones(batch, 2, dtype=torch.int64)
        return prompt_embeds, prompt_mask


class ImageEncodePipeline:
    def __init__(self):
        self.calls = []
        self.last_image = None

    def encode_prompt(self, prompt, image=None, device=None, num_images_per_prompt=1):
        self.last_image = image
        self.calls.append(
            {
                "prompt": prompt,
                "image_len": len(image) if image is not None else 0,
                "device": device,
                "num_images_per_prompt": num_images_per_prompt,
            }
        )
        batch = len(prompt)
        prompt_embeds = torch.zeros(batch, 2, 4)
        prompt_mask = torch.ones(batch, 2, dtype=torch.int64)
        return prompt_embeds, prompt_mask


class PromptEncodingQwen(QwenImage):
    """
    Minimal QwenImage wrapper that exposes _encode_prompts without loading real models.
    """

    def __init__(self, pipeline, flavour="v1.0"):
        self.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.config = SimpleNamespace(
            model_flavour=flavour,
            pretrained_model_name_or_path="dummy",
            pretrained_vae_model_name_or_path="dummy",
            vae_path="dummy",
            controlnet=False,
            control=False,
            weight_dtype=torch.float32,
            flow_matching=True,
        )
        self.vae_scale_factor = 8
        self.pipelines = {PipelineTypes.TEXT2IMG: pipeline}
        self.tokenizers = [SimpleNamespace()]
        text_encoder = SimpleNamespace(device=torch.device("cpu"))

        def to(device):
            text_encoder.device = device
            return text_encoder

        text_encoder.to = to
        self.text_encoders = [text_encoder]
        self.noise_schedule = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))

    def get_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True):
        return self.pipelines[pipeline_type]


class QwenPromptEncodingTests(unittest.TestCase):
    def test_base_encode_prompts_omits_image_kwarg(self):
        pipeline = NoImageEncodePipeline()
        model = PromptEncodingQwen(pipeline, flavour="v1.0")

        embeds, mask = model._encode_prompts(["hello world"])

        self.assertEqual(len(pipeline.calls), 1)
        self.assertEqual(pipeline.calls[0]["prompt"], ["hello world"])
        self.assertEqual(mask.shape, torch.Size([1, 2]))
        self.assertEqual(embeds.shape, torch.Size([1, 2, 4]))

        dropout = model.encode_dropout_caption()
        self.assertIn("prompt_embeds", dropout)
        self.assertEqual(dropout["attention_masks"].shape, torch.Size([1, 2]))

    def test_edit_encode_prompts_requires_and_uses_prompt_image(self):
        pipeline = ImageEncodePipeline()
        model = PromptEncodingQwen(pipeline, flavour="edit-v1")

        with self.assertRaises(ValueError):
            model._encode_prompts(["needs image"])

        model._current_prompt_contexts = [
            {"conditioning_pixel_values": torch.ones(3, 2, 2)},
        ]
        embeds, mask = model._encode_prompts(["with image"])

        self.assertEqual(embeds.shape, torch.Size([1, 2, 4]))
        self.assertEqual(mask.shape, torch.Size([1, 2]))
        self.assertIsNotNone(pipeline.last_image)
        self.assertEqual(len(pipeline.last_image), 1)
        self.assertIsInstance(pipeline.last_image[0], Image.Image)

        dropout = model.encode_dropout_caption()
        self.assertIsNotNone(pipeline.last_image)
        self.assertTrue(torch.is_tensor(pipeline.last_image))
        self.assertEqual(tuple(pipeline.last_image.shape), (1, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()
