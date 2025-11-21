import unittest
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.training.collate import check_latent_shapes, collate_fn


class TestAudioCollate(unittest.TestCase):
    def setUp(self):
        self.mock_state_tracker_patcher = patch("simpletuner.helpers.training.collate.StateTracker")
        self.mock_state_tracker = self.mock_state_tracker_patcher.start()

        self.mock_accelerator = MagicMock()
        self.mock_accelerator.device = "cpu"
        self.mock_state_tracker.get_accelerator.return_value = self.mock_accelerator

        self.mock_args = MagicMock()
        self.mock_args.model_family = "ace_step"
        self.mock_args.caption_dropout_probability = 0.0
        self.mock_args.controlnet = False
        self.mock_args.vae_cache_ondemand = False
        self.mock_args.vae_cache_disable = False
        self.mock_args.conditioning_multidataset_sampling = "random"
        self.mock_state_tracker.get_args.return_value = self.mock_args
        self.mock_state_tracker.get_model_family.return_value = "ace_step"
        self.mock_state_tracker.get_weight_dtype.return_value = torch.float32

        # Mock model
        self.mock_model = MagicMock()
        self.mock_model.requires_conditioning_image_embeds.return_value = False
        self.mock_model.requires_conditioning_dataset.return_value = False
        self.mock_model.requires_conditioning_latents.return_value = False
        self.mock_state_tracker.get_model.return_value = self.mock_model

        # Mock Data Backend
        self.mock_backend = MagicMock()
        self.mock_backend.get.return_value = {"instance_data_dir": "/tmp"}
        self.mock_state_tracker.get_data_backend.return_value = self.mock_backend

        # Mock Text Embed Cache
        self.mock_text_cache = MagicMock()
        self.mock_text_cache.disabled = False
        self.mock_text_cache.compute_prompt_embeddings_with_model.return_value = {
            "prompt_embeds": torch.randn(1, 10, 32),
            "attention_masks": torch.ones(1, 10),
        }
        self.mock_backend.__getitem__.return_value = self.mock_text_cache  # backend['text_embed_cache']

    def tearDown(self):
        self.mock_state_tracker_patcher.stop()

    def test_collate_audio_equal_lengths(self):
        # Test batch with 2 samples, same shape
        batch = [
            {
                "training_samples": [
                    {"image_path": "a.wav", "data_backend_id": "b1", "instance_prompt_text": "a"},
                    {"image_path": "b.wav", "data_backend_id": "b1", "instance_prompt_text": "b"},
                ],
                "conditioning_samples": [],
            }
        ]

        # Mock compute_latents output
        # Latents: [C, T] or [C, H, W]? ACE-Step: [C, H, W].
        # For 10s audio @ 48k -> 480000 samples
        # Latent shape approx: [8, 16, 100] ?
        latent_shape = (8, 16, 100)
        latents = [{"latents": torch.randn(*latent_shape)}, {"latents": torch.randn(*latent_shape)}]

        with patch("simpletuner.helpers.training.collate.compute_latents", return_value=latents):
            result = collate_fn(batch)

        self.assertIsNotNone(result["latent_batch"])
        self.assertEqual(result["latent_batch"].shape, (2, *latent_shape))

    def test_check_latent_shapes_variable_lengths(self):
        # Verify check_latent_shapes raises error for differing lengths in training data
        latents = [torch.randn(8, 16, 100), torch.randn(8, 16, 110)]
        filepaths = ["a.wav", "b.wav"]

        with self.assertRaises(ValueError) as cm:
            check_latent_shapes(latents, filepaths, "b1", [], is_conditioning=False)

        self.assertIn("latent shape mismatch", str(cm.exception))

    def test_check_latent_shapes_conditioning_variable_lengths(self):
        # Verify check_latent_shapes allows differing lengths for conditioning
        latents = [torch.randn(8, 16, 100), torch.randn(8, 16, 110)]
        filepaths = ["a.wav", "b.wav"]

        result = check_latent_shapes(latents, filepaths, "b1", [], is_conditioning=True)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape[-1], 100)
        self.assertEqual(result[1].shape[-1], 110)


if __name__ == "__main__":
    unittest.main()
