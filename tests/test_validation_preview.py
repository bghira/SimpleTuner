import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.training.validation import ValidationPreviewer, _PreviewMetadata


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

    def pre_validation_preview_decode(self, latents):
        return latents


class ValidationPreviewerTests(unittest.TestCase):
    @patch("simpletuner.helpers.training.validation.StateTracker")
    def test_should_emit_interval(self, mock_state_tracker):
        handler = MagicMock()
        mock_state_tracker.get_webhook_handler.return_value = handler
        config = types.SimpleNamespace(validation_preview=True, validation_preview_steps=3)
        accelerator = types.SimpleNamespace(is_main_process=True)
        previewer = ValidationPreviewer(_DummyModel(), accelerator, config)
        self.assertTrue(previewer._should_emit_for_step(0))
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

    @patch("simpletuner.helpers.training.validation.StateTracker")
    def test_emit_event_formats_step_label_with_config_total(self, mock_state_tracker):
        handler = MagicMock()
        mock_state_tracker.get_webhook_handler.return_value = handler
        config = types.SimpleNamespace(validation_preview=True, validation_preview_steps=1, validation_num_inference_steps=4)
        accelerator = types.SimpleNamespace(is_main_process=True)
        previewer = ValidationPreviewer(_DummyModel(), accelerator, config)
        previewer._webhook_handler = handler
        metadata = _PreviewMetadata(shortname="foo", prompt="hello", resolution=(512, 512), validation_type="checkpoint")
        previewer._emit_event([], None, metadata, step=1, timestep=0.5)
        handler.send_raw.assert_called_once()
        payload = handler.send_raw.call_args.kwargs["structured_data"]
        self.assertEqual(payload["message"], "Validation (step 2/4): foo")
        self.assertEqual(payload["title"], "Validation (step 2/4): foo")
        self.assertEqual(payload["body"], "hello")
        self.assertEqual(payload["data"]["step_label"], "2/4")
        handler.reset_mock()

    @patch("simpletuner.helpers.training.validation.StateTracker")
    def test_emit_event_uses_metadata_total_override(self, mock_state_tracker):
        handler = MagicMock()
        mock_state_tracker.get_webhook_handler.return_value = handler
        config = types.SimpleNamespace(validation_preview=True, validation_preview_steps=1, validation_num_inference_steps=4)
        accelerator = types.SimpleNamespace(is_main_process=True)
        previewer = ValidationPreviewer(_DummyModel(), accelerator, config)
        previewer._webhook_handler = handler
        metadata = _PreviewMetadata(
            shortname="bar",
            prompt="prompt",
            resolution=(256, 256),
            validation_type="checkpoint",
            total_steps=10,
        )
        previewer._emit_event([], None, metadata, step=2, timestep=None)
        payload = handler.send_raw.call_args.kwargs["structured_data"]
        self.assertEqual(payload["message"], "Validation (step 3/10): bar")
        self.assertEqual(payload["title"], "Validation (step 3/10): bar")
        self.assertEqual(payload["data"]["step_label"], "3/10")


if __name__ == "__main__":
    unittest.main()
