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

    def pre_latent_decode(self, latents):
        return latents

    # Backward compatibility alias
    pre_validation_preview_decode = pre_latent_decode

    def decode_latents_to_pixels(self, latents, *, use_tae=False):
        # Return (B, T, C, H, W) format - single frame image
        return torch.zeros(1, 1, 3, 8, 8)


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
    def test_decode_preview_uses_unified_interface(self, mock_state_tracker):
        """Test that _decode_preview uses the unified decode_latents_to_pixels interface."""
        handler = MagicMock()
        mock_state_tracker.get_webhook_handler.return_value = handler
        config = types.SimpleNamespace(validation_preview=True, validation_preview_steps=1)
        accelerator = types.SimpleNamespace(is_main_process=True)
        model = _DummyModel()
        model.decode_latents_to_pixels = MagicMock(return_value=torch.zeros(1, 1, 3, 8, 8))
        previewer = ValidationPreviewer(model, accelerator, config)
        previewer._emit_event = MagicMock()
        previewer._decode_preview(torch.zeros(1, 4, 8, 8))
        model.decode_latents_to_pixels.assert_called_once()
        # Verify use_tae=True is passed
        call_kwargs = model.decode_latents_to_pixels.call_args.kwargs
        self.assertTrue(call_kwargs.get("use_tae", False))

    @patch("simpletuner.helpers.training.validation.StateTracker")
    def test_decode_preview_handles_video_output(self, mock_state_tracker):
        """Test that _decode_preview correctly handles multi-frame video output."""
        handler = MagicMock()
        mock_state_tracker.get_webhook_handler.return_value = handler
        config = types.SimpleNamespace(validation_preview=True, validation_preview_steps=1)
        accelerator = types.SimpleNamespace(is_main_process=True)
        model = _DummyModel()
        # Return video with 4 frames: (B=1, T=4, C=3, H=8, W=8)
        model.decode_latents_to_pixels = MagicMock(return_value=torch.rand(1, 4, 3, 8, 8))
        previewer = ValidationPreviewer(model, accelerator, config)
        images, video_payload = previewer._decode_preview(torch.zeros(1, 4, 8, 8))
        # Should return first frame as image and video payload
        self.assertEqual(len(images), 1)
        self.assertIsNotNone(video_payload)

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
