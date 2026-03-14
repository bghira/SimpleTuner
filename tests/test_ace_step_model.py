import unittest
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.ace_step.model import ACEStep


class TestACEStepModel(unittest.TestCase):
    def setUp(self):
        self.mock_accelerator = MagicMock()
        self.mock_accelerator.device = torch.device("cpu")

        self.config = MagicMock()
        self.config.weight_dtype = torch.float32
        self.config.logit_mean = 0.0
        self.config.logit_std = 1.0
        self.config.flow_schedule_shift = 3.0
        self.config.model_family = "ace_step"
        self.config.pretrained_model_name_or_path = "dummy_path"

        # Mock init to avoid side effects like loading tokenizers
        with (
            patch("simpletuner.helpers.models.ace_step.model.VoiceBpeTokenizer") as mock_tokenizer,
            patch("simpletuner.helpers.models.ace_step.model.LangSegment") as mock_lang_segment,
            patch("simpletuner.helpers.models.ace_step.model.FlowMatchEulerDiscreteScheduler") as mock_scheduler_cls,
        ):

            self.model = ACEStep(self.config, self.mock_accelerator)

            # Setup scheduler mock
            self.mock_scheduler = MagicMock()
            self.mock_scheduler.timesteps = torch.linspace(1000, 0, 1000)
            self.mock_scheduler.sigmas = torch.linspace(0, 1, 1000)
            self.model.noise_schedule = self.mock_scheduler

            # Mock tokenizer instance
            self.model.lyric_tokenizer = MagicMock()
            self.model.lyric_tokenizer.encode.return_value = [10, 11, 12]  # Dummy tokens

    def test_prepare_batch_masks(self):
        batch_size = 2
        seq_len = 16
        channels = 8
        latents = torch.randn(batch_size, channels, 1, seq_len)  # (B, C, H, W)

        # latent_metadata with differing lengths
        # Item 0: full length (16)
        # Item 1: half length (8)
        latent_metadata = [{"latent_length": 16}, {"latent_length": 8}]

        batch = {
            "latent_batch": latents,
            "latent_metadata": latent_metadata,
            "prompt_embeds": torch.randn(batch_size, 10, 32),  # (B, T, D)
            "speaker_embeds": torch.randn(batch_size, 512),
        }

        prepared = self.model.prepare_batch(batch, state={"is_validation": True})

        self.assertIn("attention_mask", prepared)
        mask = prepared["attention_mask"]
        self.assertEqual(mask.shape, (batch_size, seq_len))

        # Check mask values
        # Item 0 should be all 1s
        self.assertTrue(torch.all(mask[0] == 1.0))
        # Item 1 should be 1s up to index 8, then 0s
        self.assertTrue(torch.all(mask[1, :8] == 1.0))
        self.assertTrue(torch.all(mask[1, 8:] == 0.0))

    def test_loss_masking(self):
        batch_size = 2
        seq_len = 4
        channels = 2
        height = 1

        # Create prediction and target
        # Item 0: valid
        # Item 1: masked (last 2 tokens)
        model_pred = torch.ones(batch_size, channels, height, seq_len)
        target = torch.zeros(batch_size, channels, height, seq_len)  # Loss should be 1 if unmasked

        # Mask:
        # Item 0: [1, 1, 1, 1]
        # Item 1: [1, 1, 0, 0]
        attention_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])

        prepared_batch = {"latents": target, "attention_mask": attention_mask}
        model_output = {"model_prediction": model_pred, "sample": model_pred}  # Fallback

        loss = self.model.loss(prepared_batch, model_output)

        # Expected loss:
        # Item 0: MSE(1, 0) over all 4 tokens = 1.0
        # Item 1: MSE(1, 0) over first 2 tokens = 1.0. Last 2 masked out.
        # Mean loss = (1.0 + 1.0 * (0.5 weighting?)) / something?

        # Let's trace the loss calculation in model.py:
        # mask = attention_mask expanded to (B, C, W, T)
        # selected_model_pred = (pred * mask).reshape
        # selected_target = (target * mask).reshape
        # loss = MSE(..., reduction='none')
        # loss = loss.mean(1) -> Per sample mean MSE
        # loss = loss * mask.reshape(bsz, -1).mean(1) -> Weighted by valid ratio
        # loss = loss.mean()

        # Item 0:
        #   diff = 1. Valid ratio = 1.0. Loss = 1.0 * 1.0 = 1.0
        # Item 1:
        #   diff: first 2 are 1, last 2 are 0 (masked).
        #   MSE over 4*C*H elements?
        #   selected_model_pred has 0s where masked. selected_target has 0s where masked.
        #   So squared error is 0 for masked regions.
        #   MSE sum = (1^2 + 1^2 + ... for valid) + (0 + 0 for masked).
        #   Mean MSE = Sum / Total Elements (including masked).
        #   If C=2, H=1. Total elements = 8.
        #   Valid elements = 4 (Item 0), 2 (Item 1)? No.
        #   Item 0: 4 valid time steps * 2 channels = 8 elements. All valid. MSE = 1.
        #   Item 1: 2 valid time steps * 2 channels = 4 elements. 4 masked.
        #     Sum errors = 4 * (1-0)^2 = 4.
        #     Mean MSE = 4 / 8 = 0.5.
        #   Weighting: mask mean for Item 1 = 0.5 (2/4 time steps).
        #   Item 1 Loss contribution = 0.5 (MSE) * 0.5 (Weight) = 0.25?

        # Wait, the code says:
        # loss = F.mse_loss(..., reduction="none") -> shape (B, flattened_dim)
        # loss = loss.mean(1) -> (B,)
        # loss = loss * mask.reshape(bsz, -1).mean(1)

        # Let's calculate manually:
        # Item 0: MSE vector is all 1s. Mean = 1. Mask mean = 1. Result = 1.
        # Item 1: MSE vector has 1s for valid, 0s for masked.
        #   Valid elements: 4 (out of 8). So 4 ones, 4 zeros.
        #   Mean = 0.5.
        #   Mask mean = 0.5.
        #   Result = 0.5 * 0.5 = 0.25.

        # Final Loss = (1 + 0.25) / 2 = 0.625.

        self.assertAlmostEqual(loss.item(), 0.625, places=4)

    def test_model_supports_crepa_self_flow(self):
        self.assertTrue(self.model.supports_crepa_self_flow())

    def test_prepare_crepa_self_flow_batch_creates_tokenwise_timesteps(self):
        self.model.model = MagicMock(config=type("Cfg", (), {"patch_size": [16, 1]})())
        self.model.get_trained_component = MagicMock(return_value=self.model.model)
        self.model.sample_flow_sigmas = MagicMock(return_value=(torch.tensor([0.8]), torch.tensor([800.0])))
        batch = {
            "latents": torch.randn(1, 8, 16, 8),
            "input_noise": torch.randn(1, 8, 16, 8),
            "sigmas": torch.tensor([0.2]),
            "timesteps": torch.tensor([200.0]),
        }

        result = self.model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["timesteps"].shape, (1, 8))
        self.assertEqual(result["crepa_self_flow_mask"].shape, (1, 1, 8))
        self.assertEqual(result["crepa_teacher_timesteps"].shape, (1,))

    def test_model_predict_uses_crepa_capture_block_override(self):
        transformer = MagicMock()

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((1, 8, 16), 2.0)
            kwargs["hidden_states_buffer"]["layer_9"] = torch.full((1, 8, 16), 9.0)
            return type("Out", (), {"sample": torch.randn(1, 8, 16, 8), "proj_losses": None})()

        transformer.side_effect = _forward
        self.model.get_trained_component = MagicMock(return_value=transformer)
        self.model._new_hidden_state_buffer = MagicMock(return_value={})
        self.model.crepa_regularizer = type("Crepa", (), {"block_index": 2})()

        prepared_batch = {
            "noisy_latents": torch.randn(1, 8, 16, 8),
            "attention_mask": torch.ones(1, 8),
            "encoder_hidden_states": torch.randn(1, 4, 32),
            "encoder_attention_mask": torch.ones(1, 4),
            "speaker_embeds": torch.randn(1, 512),
            "lyric_token_ids": torch.zeros(1, 1, dtype=torch.long),
            "lyric_mask": torch.zeros(1, 1, dtype=torch.long),
            "timesteps": torch.tensor([200.0]),
            "sigmas": torch.ones(1, 1, 1, 1),
            "crepa_capture_block_index": 9,
        }

        result = self.model.model_predict(prepared_batch)

        self.assertTrue(torch.equal(result["crepa_hidden_states"], torch.full((1, 8, 16), 9.0)))
        transformer_kwargs = transformer.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], torch.tensor([200.0], dtype=torch.float32)))

    def test_model_predict_accepts_tokenwise_timesteps(self):
        transformer = MagicMock()

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((1, 8, 16), 2.0)
            return type("Out", (), {"sample": torch.randn(1, 8, 16, 8), "proj_losses": None})()

        transformer.side_effect = _forward
        self.model.get_trained_component = MagicMock(return_value=transformer)
        self.model._new_hidden_state_buffer = MagicMock(return_value={})

        prepared_batch = {
            "noisy_latents": torch.randn(1, 8, 16, 8),
            "attention_mask": torch.ones(1, 8),
            "encoder_hidden_states": torch.randn(1, 4, 32),
            "encoder_attention_mask": torch.ones(1, 4),
            "speaker_embeds": torch.randn(1, 512),
            "lyric_token_ids": torch.zeros(1, 1, dtype=torch.long),
            "lyric_mask": torch.zeros(1, 1, dtype=torch.long),
            "timesteps": torch.tensor([[100.0, 900.0, 200.0, 800.0, 300.0, 700.0, 400.0, 600.0]]),
            "sigmas": torch.ones(1, 1, 1, 1),
        }

        self.model.model_predict(prepared_batch)

        transformer_kwargs = transformer.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], prepared_batch["timesteps"].to(torch.float32)))


if __name__ == "__main__":
    unittest.main()
