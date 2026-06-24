import types
import unittest
from unittest import mock

import torch

from simpletuner.helpers.models.z_image.model import ZImage


class DummyAccelerator:
    device = torch.device("cpu")

    def unwrap_model(self, model):
        return model


class DummyTransformer:
    def __init__(self):
        self.received = None
        self.config = types.SimpleNamespace(all_patch_size=(2,), all_f_patch_size=(1,))

    def __call__(self, latent_list, timestep, prompt_list, **kwargs):
        self.received = (latent_list, timestep, prompt_list, kwargs)
        if "hidden_states_buffer" in kwargs and kwargs["hidden_states_buffer"] is not None:
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((len(latent_list), 16, 8), 2.0)
            kwargs["hidden_states_buffer"]["layer_7"] = torch.full((len(latent_list), 16, 8), 7.0)
        outputs = [torch.zeros_like(lat) for lat in latent_list]
        return outputs, {}


class EchoTransformer(DummyTransformer):
    def __call__(self, latent_list, timestep, prompt_list, **kwargs):
        self.received = (latent_list, timestep, prompt_list, kwargs)
        return [lat.clone() for lat in latent_list], {}


class ZImageModelTests(unittest.TestCase):
    def _build_model(self):
        model = ZImage.__new__(ZImage)
        model.accelerator = DummyAccelerator()
        model.config = types.SimpleNamespace(
            weight_dtype=torch.float32,
            model_family="z_image",
            pretrained_model_name_or_path=None,
            pretrained_vae_model_name_or_path=None,
            vae_path=None,
            flow_schedule_shift=1.0,
            crepa_self_flow_mask_ratio=0.5,
            controlnet=False,
        )
        model.model = DummyTransformer()
        model._new_hidden_state_buffer = lambda: {}
        model.crepa_regularizer = types.SimpleNamespace(enabled=True, block_index=2)
        return model

    def test_ic_lora_capability_flags(self):
        model = self._build_model()

        self.assertTrue(model.supports_conditioning_dataset())
        self.assertTrue(model.requires_conditioning_latents())

    def test_prepare_batch_conditions_accepts_reference_conditioning_only(self):
        model = self._build_model()

        with self.assertRaisesRegex(ValueError, "Unsupported conditioning_type"):
            model.prepare_batch_conditions(
                {"conditioning_latents": torch.zeros(1, 4, 8, 8), "conditioning_type": "mask"},
                state={},
            )

    def test_convert_text_embed_for_pipeline_masks_tokens(self):
        model = self._build_model()
        prompt_embeds = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
        attention_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 1]], dtype=torch.bool)

        converted = model.convert_text_embed_for_pipeline(
            {"prompt_embeds": prompt_embeds, "attention_mask": attention_mask}
        )["prompt_embeds"]

        self.assertEqual(len(converted), 2)
        self.assertTrue(torch.equal(converted[0], prompt_embeds[0][:2]))
        self.assertTrue(torch.equal(converted[1], prompt_embeds[1][[0, 2, 3]]))

    def test_model_predict_shapes_and_timesteps(self):
        model = self._build_model()
        latents = torch.randn(2, 4, 8, 8)
        timesteps = torch.tensor([100.0, 500.0])
        prompt_embeds = torch.randn(2, 5, 6)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)

        output = model.model_predict(
            {
                "noisy_latents": latents,
                "timesteps": timesteps,
                "encoder_hidden_states": prompt_embeds,
                "encoder_attention_mask": attention_mask,
            }
        )

        noise_pred = output["model_prediction"]
        self.assertEqual(noise_pred.shape, latents.shape)

        latent_list, received_t, prompt_list, _ = model.model.received
        self.assertEqual(len(latent_list), latents.shape[0])
        self.assertEqual(len(prompt_list[0]), attention_mask[0].sum().item())
        self.assertEqual(len(prompt_list[1]), attention_mask[1].sum().item())
        # Timesteps are normalized and flipped as in the pipeline: (1000 - t) / 1000
        self.assertTrue(torch.allclose(received_t, torch.tensor([0.9, 0.5])))

    def test_model_predict_concats_ic_lora_reference_and_crops_prediction(self):
        model = self._build_model()
        model.model = EchoTransformer()
        model.crepa_regularizer = None
        latents = torch.ones(1, 4, 8, 8)
        conditioning_latents = torch.full((1, 4, 8, 4), 2.0)
        prompt_embeds = torch.randn(1, 5, 6)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)

        output = model.model_predict(
            {
                "noisy_latents": latents,
                "conditioning_latents": conditioning_latents,
                "conditioning_type": "reference_strict",
                "timesteps": torch.tensor([100.0]),
                "encoder_hidden_states": prompt_embeds,
                "encoder_attention_mask": attention_mask,
            }
        )

        noise_pred = output["model_prediction"]
        self.assertEqual(noise_pred.shape, latents.shape)
        torch.testing.assert_close(noise_pred, -latents)

        latent_list, received_t, _, _ = model.model.received
        self.assertEqual(latent_list[0].shape, torch.Size([4, 1, 8, 12]))
        self.assertTrue(torch.equal(latent_list[0][..., :4], conditioning_latents[0].unsqueeze(1)))
        self.assertTrue(torch.equal(latent_list[0][..., 4:], latents[0].unsqueeze(1)))
        self.assertEqual(received_t.shape, torch.Size([1, 24]))
        torch.testing.assert_close(received_t[:, :8], torch.zeros(1, 8))
        torch.testing.assert_close(received_t[:, 8:], torch.full((1, 16), 0.9))

    def test_model_predict_preserves_target_tokenwise_timesteps_with_ic_lora_reference(self):
        model = self._build_model()
        model.model = EchoTransformer()
        model.crepa_regularizer = None
        latents = torch.ones(1, 4, 8, 8)
        conditioning_latents = torch.full((1, 4, 8, 4), 2.0)
        prompt_embeds = torch.randn(1, 5, 6)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)

        model.model_predict(
            {
                "noisy_latents": latents,
                "conditioning_latents": conditioning_latents,
                "conditioning_type": "reference_strict",
                "timesteps": torch.full((1, 16), 100.0),
                "encoder_hidden_states": prompt_embeds,
                "encoder_attention_mask": attention_mask,
            }
        )

        _, received_t, _, _ = model.model.received
        self.assertEqual(received_t.shape, torch.Size([1, 24]))
        torch.testing.assert_close(received_t[:, :8], torch.zeros(1, 8))
        torch.testing.assert_close(received_t[:, 8:], torch.full((1, 16), 0.9))

    def test_model_predict_rejects_ic_lora_with_hidden_state_capture(self):
        model = self._build_model()
        model.model = EchoTransformer()
        model.crepa_regularizer = types.SimpleNamespace(wants_hidden_states=lambda: True)
        prompt_embeds = torch.randn(1, 5, 6)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "hidden-state capture"):
            model.model_predict(
                {
                    "noisy_latents": torch.ones(1, 4, 8, 8),
                    "conditioning_latents": torch.full((1, 4, 8, 4), 2.0),
                    "conditioning_type": "reference_strict",
                    "timesteps": torch.tensor([100.0]),
                    "encoder_hidden_states": prompt_embeds,
                    "encoder_attention_mask": attention_mask,
                }
            )

    def test_model_predict_accepts_tokenwise_timesteps_and_capture_override(self):
        model = self._build_model()
        latents = torch.randn(1, 4, 8, 8)
        prompt_embeds = torch.randn(1, 5, 6)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)

        output = model.model_predict(
            {
                "noisy_latents": latents,
                "timesteps": torch.tensor([[100.0] * 16], dtype=torch.float32),
                "encoder_hidden_states": prompt_embeds,
                "encoder_attention_mask": attention_mask,
                "crepa_capture_block_index": 7,
            }
        )

        _, received_t, _, _ = model.model.received
        self.assertEqual(received_t.shape, torch.Size([1, 16]))
        torch.testing.assert_close(received_t, torch.full((1, 16), 0.9, dtype=torch.float32))
        self.assertTrue(torch.equal(output["crepa_hidden_states"], torch.full((1, 16, 8), 7.0)))

    def test_infer_crepa_hidden_size_uses_transformer_dim(self):
        model = self._build_model()
        model.model.config = types.SimpleNamespace(dim=3840)

        self.assertEqual(model._infer_crepa_hidden_size(), 3840)

    def test_prepare_crepa_self_flow_batch_creates_tokenwise_timesteps(self):
        model = self._build_model()
        model.sample_flow_sigmas = mock.MagicMock(
            return_value=(torch.tensor([0.75], dtype=torch.float32), torch.tensor([750.0], dtype=torch.float32))
        )

        batch = {
            "latents": torch.zeros(1, 16, 8, 8),
            "input_noise": torch.ones(1, 16, 8, 8),
            "sigmas": torch.tensor([0.25], dtype=torch.float32),
            "timesteps": torch.tensor([250.0], dtype=torch.float32),
        }

        updated = model._prepare_crepa_self_flow_batch(batch, state={"global_step": 0})

        self.assertEqual(updated["timesteps"].shape, torch.Size([1, 16]))
        self.assertEqual(updated["crepa_teacher_timesteps"].shape, torch.Size([1]))
        self.assertEqual(updated["crepa_teacher_noisy_latents"].shape, torch.Size([1, 16, 8, 8]))


if __name__ == "__main__":
    unittest.main()
