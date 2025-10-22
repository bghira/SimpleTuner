import unittest
from types import SimpleNamespace

import torch

from simpletuner.helpers.models.common import PipelineTypes
from simpletuner.helpers.models.qwen_image.model import (
    QwenImage,
    QwenImageEditPipeline,
    QwenImageEditPlusPipeline,
)


class DummyProcessor:
    def __call__(self, images, return_tensors="pt", **kwargs):
        batch_size = len(images)
        pixel_values = torch.arange(batch_size * 3 * 2 * 2, dtype=torch.float32).view(batch_size, 3, 2, 2)
        image_grid_thw = torch.ones(batch_size, 3, dtype=torch.int64)
        return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}


class DummyPipeline:
    def __init__(self, prompt_embeds, prompt_mask):
        self._prompt_embeds = prompt_embeds
        self._prompt_mask = prompt_mask
        self.processor = DummyProcessor()
        self.text_encoder = SimpleNamespace(dtype=torch.float32)
        self.captured_images = None

    def encode_prompt(self, prompt, image=None, device=None, num_images_per_prompt=1):
        self.captured_images = image
        return self._prompt_embeds.clone(), self._prompt_mask.clone()


class DummyLatentDist:
    def __init__(self, tensor):
        self._tensor = tensor

    def sample(self):
        return self._tensor


class DummyVAE:
    config = SimpleNamespace(
        z_dim=16,
        latents_mean=[0.0] * 16,
        latents_std=[1.0] * 16,
    )

    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def encode(self, tensor):
        spatial = tensor.shape[-2:]
        sample = torch.ones(tensor.shape[0], self.config.z_dim, *spatial, device=tensor.device, dtype=tensor.dtype)
        return SimpleNamespace(latent_dist=DummyLatentDist(sample))


class DummyTransformer:
    config = SimpleNamespace(in_channels=64)

    def __call__(
        self,
        hidden_states,
        timestep,
        guidance=None,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        img_shapes=None,
        txt_seq_lens=None,
        return_dict=False,
    ):
        return (hidden_states, )


class TestableQwenImage(QwenImage):
    def __init__(self, pipeline, transformer, vae, model_flavour="edit-v1"):
        self.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.config = SimpleNamespace(
            model_flavour=model_flavour,
            pretrained_model_name_or_path="dummy",
            pretrained_vae_model_name_or_path="dummy",
            vae_path="dummy",
            controlnet=False,
            control=False,
            weight_dtype=torch.float32,
            flow_matching=True,
        )
        self.pipelines = {}
        if model_flavour == "edit-v2":
            self.PIPELINE_CLASSES = {PipelineTypes.TEXT2IMG: QwenImageEditPlusPipeline}
        else:
            self.PIPELINE_CLASSES = {PipelineTypes.TEXT2IMG: QwenImageEditPipeline}
        self.vae_scale_factor = 8
        self.vae = vae
        self.model = transformer
        self.noise_schedule = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        self._pipeline = pipeline

    def get_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True):
        return self._pipeline


class QwenEditTests(unittest.TestCase):
    def test_conditioning_image_embedder_returns_dict_entries(self):
        processor = DummyProcessor()
        embedder = QwenImage._EditV1ConditioningImageEmbedder(
            processor=processor,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        outputs = embedder.encode(["img-a", "img-b"])
        self.assertEqual(len(outputs), 2)
        for entry in outputs:
            self.assertIn("pixel_values", entry)
            self.assertIsInstance(entry["pixel_values"], torch.Tensor)
            self.assertEqual(entry["pixel_values"].shape, torch.Size([3, 2, 2]))

    def test_prepare_edit_batch_sets_prompt_and_control_latents(self):
        prompt_embeds = torch.ones(2, 4, 8)
        prompt_mask = torch.ones(2, 4, dtype=torch.int64)
        pipeline = DummyPipeline(prompt_embeds, prompt_mask)
        transformer = DummyTransformer()
        vae = DummyVAE()
        model = TestableQwenImage(pipeline, transformer, vae)

        batch = {
            "prompts": ["a", "b"],
            "conditioning_pixel_values": torch.zeros(2, 3, 4, 4),
            "conditioning_image_embeds": {"pixel_values": torch.zeros(2, 3, 4, 4)},
        }

        updated = model._prepare_edit_batch_v1(batch)

        torch.testing.assert_close(updated["prompt_embeds"], prompt_embeds.to(dtype=model.config.weight_dtype))
        torch.testing.assert_close(updated["encoder_attention_mask"], prompt_mask)
        self.assertIn("edit_control_latents", updated)
        self.assertEqual(updated["edit_control_latents"].shape, torch.Size([2, 16, 4, 4]))
        self.assertIsNotNone(pipeline.captured_images)

    def test_model_predict_edit_path_restores_latent_shape(self):
        prompt_embeds = torch.randn(1, 4, 8)
        prompt_mask = torch.ones(1, 4, dtype=torch.int64)
        pipeline = DummyPipeline(prompt_embeds, prompt_mask)
        transformer = DummyTransformer()
        vae = DummyVAE()
        model = TestableQwenImage(pipeline, transformer, vae)

        latents = torch.randn(1, 16, 8, 8)
        control_latents = torch.randn(1, 16, 8, 8)
        prepared_batch = {
            "noisy_latents": latents.clone(),
            "edit_control_latents": control_latents.clone(),
            "prompt_embeds": prompt_embeds.clone(),
            "encoder_attention_mask": prompt_mask.clone(),
            "timesteps": torch.tensor([250.0]),
        }

        result = model._model_predict_edit_v1(prepared_batch)
        self.assertIn("model_prediction", result)
        prediction = result["model_prediction"]
        self.assertEqual(prediction.shape, latents.shape)

    def test_prepare_edit_batch_v2_builds_control_tensors(self):
        prompt_embeds = torch.ones(1, 4, 8)
        prompt_mask = torch.ones(1, 4, dtype=torch.int64)
        pipeline = DummyPipeline(prompt_embeds, prompt_mask)
        transformer = DummyTransformer()
        vae = DummyVAE()
        model = TestableQwenImage(pipeline, transformer, vae, model_flavour="edit-v2")

        cond_a = torch.zeros(2, 3, 4, 4)
        cond_b = torch.ones(2, 3, 4, 4)
        batch = {
            "prompts": ["a", "b"],
            "latents": torch.zeros(2, 16, 8, 8),
            "conditioning_pixel_values_multi": [cond_a, cond_b],
        }

        updated = model._prepare_edit_batch_v2(batch)

        self.assertEqual(updated["prompt_embeds"].shape, torch.Size([2, 4, 8]))
        self.assertEqual(updated["encoder_attention_mask"].shape, torch.Size([2, 4]))
        self.assertIn("control_tensor_list", updated)
        self.assertEqual(len(updated["control_tensor_list"]), 2)
        self.assertEqual(len(updated["control_tensor_list"][0]), 2)
        self.assertIsInstance(model._pipeline.captured_images, list)

    def test_model_predict_edit_plus_restores_latent_shape(self):
        prompt_embeds = torch.randn(1, 4, 8)
        prompt_mask = torch.ones(1, 4, dtype=torch.int64)
        pipeline = DummyPipeline(prompt_embeds, prompt_mask)
        transformer = DummyTransformer()
        vae = DummyVAE()
        model = TestableQwenImage(pipeline, transformer, vae, model_flavour="edit-v2")

        latents = torch.randn(1, 16, 8, 8)
        control_tensor_list = [[torch.zeros(3, 4, 4)]]
        prepared_batch = {
            "noisy_latents": latents.clone(),
            "control_tensor_list": control_tensor_list,
            "prompt_embeds": prompt_embeds.clone(),
            "encoder_attention_mask": prompt_mask.clone(),
            "timesteps": torch.tensor([250.0]),
        }

        result = model._model_predict_edit_plus(prepared_batch)
        self.assertIn("model_prediction", result)
        prediction = result["model_prediction"]
        self.assertEqual(prediction.shape, latents.shape)


if __name__ == "__main__":
    unittest.main()
