import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

from helpers.training.collate import (
    collate_fn,
)  # Adjust this import according to your project structure
from helpers.training.state_tracker import StateTracker  # Adjust imports as needed


class TestCollateFn(unittest.TestCase):
    def setUp(self):
        # Set up any common variables or mocks used in multiple tests
        self.mock_batch = [
            {
                "training_samples": [
                    {
                        "image_path": "fake_path_1.png",
                        "instance_prompt_text": "caption 1",
                        "luminance": 0.5,
                        "original_size": (100, 100),
                        "image_data": MagicMock(),
                        "crop_coordinates": [0, 0, 100, 100],
                        "data_backend_id": "foo",
                        "aspect_ratio": 1.0,
                    }
                ],
                "conditioning_samples": [],
            },
            # Add more examples as needed
        ]
        # Mock StateTracker.get_args() to return a mock object with required attributes
        StateTracker.set_args(
            MagicMock(caption_dropout_probability=0.5, controlnet=False, flux=False)
        )
        fake_accelerator = MagicMock(device="cpu")
        StateTracker.set_accelerator(fake_accelerator)

    @patch("helpers.training.collate.compute_latents")
    @patch("helpers.training.collate.compute_prompt_embeddings")
    @patch("helpers.training.collate.gather_conditional_sdxl_size_features")
    def test_collate_fn(self, mock_gather, mock_compute_embeds, mock_compute_latents):
        # Mock the responses from the compute functions
        mock_compute_latents.return_value = torch.randn(
            2, 512
        )  # Adjust dimensions as needed
        mock_compute_embeds.return_value = {
            "prompt_embeds": torch.randn(2, 768),
            "pooled_prompt_embeds": torch.randn(2, 768),
        }  # Example embeddings
        mock_gather.return_value = torch.tensor(
            [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
        )
        mock_compute_latents.to = MagicMock(return_value=mock_compute_latents)

        # Call collate_fn with a mock batch
        with patch("helpers.training.state_tracker.StateTracker.get_data_backend"):
            # Mock get_data_backend() to return a mock object with required attributes
            StateTracker.get_data_backend.return_value = MagicMock(
                compute_embeddings_for_legacy_prompts=MagicMock()
            )
            result = collate_fn(self.mock_batch)

        # Assert that the results are as expected
        self.assertIn("latent_batch", result)
        self.assertIn("prompt_embeds", result)
        self.assertIn("add_text_embeds", result)
        self.assertIn("batch_time_ids", result)
        self.assertIn("batch_luminance", result)

        # Check that the conditioning dropout was correctly applied (random elements should be zeros)
        # This can be tricky since the dropout is random; you may want to set a fixed random seed or test the structure more than values

    # You can add more test methods to cover different aspects like different dropout probabilities, edge cases, etc.


if __name__ == "__main__":
    unittest.main()
