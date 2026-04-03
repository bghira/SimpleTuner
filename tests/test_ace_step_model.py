import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
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
        self.config.pretrained_transformer_model_name_or_path = None
        self.config.pretrained_transformer_subfolder = None
        self.config.model_flavour = "base"
        self.config.controlnet = False
        self.config.peft_lora_target_modules = None
        self.config.lora_type = "standard"

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

    def test_build_v15_text_prompt_includes_metadata_template(self):
        prompt = self.model._build_v15_text_prompt(
            "warm analog synthwave",
            {
                "bpm": 120,
                "timesignature": "4/4",
                "keyscale": "C major",
                "duration": 12,
            },
        )
        self.assertIn("# Instruction", prompt)
        self.assertIn("# Caption", prompt)
        self.assertIn("# Metas", prompt)
        self.assertIn("warm analog synthwave", prompt)
        self.assertIn("- bpm: 120", prompt)
        self.assertIn("- timesignature: 4/4", prompt)
        self.assertIn("- keyscale: C major", prompt)
        self.assertIn("- duration: 12 seconds", prompt)

    def test_resolve_v15_layout_uses_requested_variant(self):
        self.config.model_flavour = "v15-base"
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "vae").mkdir()
            (root / "Qwen3-Embedding-0.6B").mkdir()
            (root / "acestep-v15-base").mkdir()
            torch.save(torch.zeros(1, 64, 4), root / "acestep-v15-base" / "silence_latent.pt")

            layout = self.model._resolve_v15_layout(str(root))

        self.assertIsNotNone(layout)
        self.assertEqual(layout["variant_path"], str(root / "acestep-v15-base"))
        self.assertEqual(layout["tokenizer_path"], str(root / "Qwen3-Embedding-0.6B"))
        self.assertEqual(layout["vae_path"], str(root / "vae"))

    def test_resolve_v15_layout_caches_negative_result_for_same_base_path(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            self.assertIsNone(self.model._resolve_v15_layout(str(root)))

            with patch("simpletuner.helpers.models.ace_step.model.Path.is_dir", side_effect=AssertionError("re-scanned")):
                self.assertIsNone(self.model._resolve_v15_layout(str(root)))

    def test_embed_v15_lyrics_batch_preserves_integer_attention_mask(self):
        embedding_layer = torch.nn.Embedding(8, 4)
        self.model.text_encoders = [SimpleNamespace(embed_tokens=embedding_layer)]
        self.model.tokenizers = [
            MagicMock(
                return_value=SimpleNamespace(
                    input_ids=torch.tensor([[1, 2, 0]], dtype=torch.long),
                    attention_mask=torch.tensor([[1, 1, 0]], dtype=torch.long),
                )
            )
        ]

        lyric_hidden_states, attention_mask = self.model._embed_v15_lyrics_batch(["verse"])

        self.assertEqual(lyric_hidden_states.dtype, torch.float32)
        self.assertEqual(attention_mask.dtype, torch.long)
        self.assertTrue(torch.equal(attention_mask, torch.tensor([[1, 1, 0]], dtype=torch.long)))

    def test_prepare_batch_v15_uses_latent_time_axis(self):
        self.model._v15_layout = {"variant_path": "dummy"}
        self.model._embed_v15_lyrics_batch = MagicMock(
            return_value=(torch.randn(2, 512, 32), torch.ones(2, 512, dtype=torch.float32))
        )
        self.model._run_v15_encoder = MagicMock(
            return_value=(torch.randn(2, 20, 32), torch.ones(2, 20, dtype=torch.float32))
        )
        self.model._sample_v15_timesteps = MagicMock(return_value=torch.tensor([0.25, 0.75], dtype=torch.float32))
        self.model._build_v15_context_latents = MagicMock(return_value=torch.ones(2, 12, 128, dtype=torch.float32))

        batch = {
            "latent_batch": torch.randn(2, 12, 64),
            "latent_metadata": [{"latent_length": 12}, {"latent_length": 5}],
            "prompt_embeds": torch.randn(2, 10, 32),
            "lyrics": ["one", "two"],
        }

        prepared = self.model.prepare_batch(batch, state={"is_validation": True})

        self.assertEqual(prepared["attention_mask"].shape, (2, 12))
        self.assertTrue(torch.all(prepared["attention_mask"][0] == 1.0))
        self.assertTrue(torch.all(prepared["attention_mask"][1, :5] == 1.0))
        self.assertTrue(torch.all(prepared["attention_mask"][1, 5:] == 0.0))
        self.assertEqual(prepared["noisy_latents"].shape, (2, 12, 64))
        self.assertEqual(prepared["context_latents"].shape, (2, 12, 128))
        self.assertTrue(torch.allclose(prepared["flow_target"], prepared["noise"] - prepared["latents"]))

    def test_get_pipeline_v15_wires_loaded_components(self):
        self.model._v15_layout = {"variant_path": "dummy"}
        self.model.model = MagicMock()
        self.model.vae = MagicMock()
        self.model.text_encoder_1 = MagicMock()
        self.model.tokenizer_1 = MagicMock()
        self.model.silence_latent = torch.zeros(1, 8, 64)

        pipeline = self.model.get_pipeline()

        self.assertTrue(pipeline.is_v15_pipeline)
        self.assertIs(pipeline.v15_model, self.model.model)
        self.assertIs(pipeline.music_dcae, self.model.vae)
        self.assertIs(pipeline.text_encoder_model, self.model.text_encoder_1)
        self.assertIs(pipeline.text_tokenizer, self.model.tokenizer_1)
        self.assertTrue(pipeline.loaded)

    def test_validation_audio_sample_rate_uses_v15_rate(self):
        self.assertEqual(self.model.validation_audio_sample_rate(), 44100)
        self.model._v15_layout = {"variant_path": "dummy"}
        self.assertEqual(self.model.validation_audio_sample_rate(), 48000)


if __name__ == "__main__":
    unittest.main()
