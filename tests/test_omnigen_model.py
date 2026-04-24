import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from simpletuner.helpers.models.omnigen.model import OmniGen


class OmniGenModelTests(unittest.TestCase):
    def setUp(self):
        self.model = OmniGen.__new__(OmniGen)
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model.config = SimpleNamespace(weight_dtype=torch.float32, base_weight_dtype=torch.float32)
        self.model._load_preprocessor = lambda: None
        self.model.processor = SimpleNamespace(
            process_multi_modal_prompt=lambda prompt, input_images=None: {"prompt": prompt},
            collator=lambda all_features: {
                "output_latents": torch.randn(1, 4, 2, 2),
                "input_ids": torch.ones(1, 2, dtype=torch.long),
                "attention_mask": torch.ones(1, 1, 7, 7),
                "position_ids": torch.arange(7).view(1, 7),
                "input_img_latents": [],
                "input_image_sizes": {},
            },
        )
        self.model._new_hidden_state_buffer = MagicMock(return_value={"layer_7": torch.randn(1, 4, 8)})
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=3)
        self.model.model = MagicMock()
        self.model.model.config = SimpleNamespace(patch_size=1)
        self.model.model.return_value = (torch.randn(1, 4, 2, 2),)
        self.model.sample_flow_sigmas = OmniGen.sample_flow_sigmas.__get__(self.model, OmniGen)
        self.model.unwrap_model = lambda model=None, wrapped=None: model if model is not None else wrapped

    def test_model_supports_crepa_self_flow(self):
        self.assertTrue(self.model.supports_crepa_self_flow())

    def test_prepare_crepa_self_flow_batch_creates_tokenwise_timesteps(self):
        self.model.config.crepa_self_flow_mask_ratio = 0.5
        batch = {
            "latents": torch.zeros(1, 4, 2, 2, dtype=torch.float32),
            "noise": torch.ones(1, 4, 2, 2, dtype=torch.float32),
            "timesteps": torch.tensor([0.2], dtype=torch.float32),
            "sigmas": torch.tensor([0.2], dtype=torch.float32),
        }
        self.model.sample_flow_sigmas = MagicMock(return_value=(torch.tensor([0.8]), torch.tensor([0.8])))
        fake_mask_rand = torch.tensor([[[0.2, 0.7], [0.9, 0.1]]], dtype=torch.float32)

        with unittest.mock.patch("torch.rand", return_value=fake_mask_rand):
            result = self.model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["timesteps"].shape, (1, 4))
        self.assertEqual(sorted(round(float(x), 4) for x in result["timesteps"].view(-1)), [0.2, 0.2, 0.8, 0.8])
        self.assertEqual(result["sigmas"].shape, (1, 1, 2, 2))

    def test_model_predict_accepts_tokenwise_timesteps_and_capture_override(self):
        prepared_batch = {
            "noisy_latents": torch.randn(1, 4, 2, 2),
            "timesteps": torch.tensor([[200.0, 800.0, 200.0, 800.0]], dtype=torch.float32),
            "prompts": ["test"],
            "crepa_capture_block_index": 7,
        }

        result = self.model.model_predict(prepared_batch)

        self.assertIsNotNone(result["crepa_hidden_states"])
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], torch.tensor([[0.2, 0.8, 0.2, 0.8]])))


if __name__ == "__main__":
    unittest.main()
