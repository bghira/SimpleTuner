import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.training.validation import ValidationPreviewer


class _DummyModel:
    NAME = "Dummy"

    def supports_validation_preview(self):
        return True

    def get_validation_preview_spec(self):
        return object()

    def get_validation_preview_decoder(self):
        return object()

    def denormalize_latents_for_preview(self, latents):
        return latents


class ValidationPreviewerTests(unittest.TestCase):
    @patch("simpletuner.helpers.training.validation.StateTracker")
    def test_should_emit_interval(self, mock_state_tracker):
        handler = MagicMock()
        mock_state_tracker.get_webhook_handler.return_value = handler
        config = types.SimpleNamespace(validation_preview=True, validation_preview_steps=3)
        accelerator = types.SimpleNamespace(is_main_process=True)
        previewer = ValidationPreviewer(_DummyModel(), accelerator, config)
        self.assertFalse(previewer._should_emit_for_step(0))
        self.assertFalse(previewer._should_emit_for_step(1))
        self.assertTrue(previewer._should_emit_for_step(2))
        self.assertFalse(previewer._should_emit_for_step(3))
        self.assertFalse(previewer._should_emit_for_step(4))
        self.assertTrue(previewer._should_emit_for_step(5))

    @patch("simpletuner.helpers.training.validation.StateTracker")
    def test_decode_preview_skips_rescaling_when_not_required(self, mock_state_tracker):
        handler = MagicMock()
        mock_state_tracker.get_webhook_handler.return_value = handler
        config = types.SimpleNamespace(validation_preview=True, validation_preview_steps=1)
        accelerator = types.SimpleNamespace(is_main_process=True)
        model = _DummyModel()
        model.denormalize_latents_for_preview = MagicMock(side_effect=lambda x: x)
        previewer = ValidationPreviewer(model, accelerator, config)
        previewer._decoder = types.SimpleNamespace(
            is_video=False,
            requires_vae_rescaling=False,
            device=torch.device("cpu"),
            dtype=torch.float32,
            decode=lambda latents: torch.zeros(1, 3, 8, 8),
        )
        previewer._emit_event = MagicMock()
        previewer._decode_preview(torch.zeros(1, 4, 8, 8))
        model.denormalize_latents_for_preview.assert_not_called()

    @patch("simpletuner.helpers.training.validation.StateTracker")
    def test_decode_preview_rescales_when_requested(self, mock_state_tracker):
        handler = MagicMock()
        mock_state_tracker.get_webhook_handler.return_value = handler
        config = types.SimpleNamespace(validation_preview=True, validation_preview_steps=1)
        accelerator = types.SimpleNamespace(is_main_process=True)
        model = _DummyModel()
        model.denormalize_latents_for_preview = MagicMock(side_effect=lambda x: x)
        previewer = ValidationPreviewer(model, accelerator, config)
        previewer._decoder = types.SimpleNamespace(
            is_video=False,
            requires_vae_rescaling=True,
            device=torch.device("cpu"),
            dtype=torch.float32,
            decode=lambda latents: torch.zeros(1, 3, 8, 8),
        )
        previewer._emit_event = MagicMock()
        previewer._decode_preview(torch.zeros(1, 4, 8, 8))
        model.denormalize_latents_for_preview.assert_called_once()


if __name__ == "__main__":
    unittest.main()
