import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.training.validation import ValidationPreviewer, _PreviewMetadata, parse_validation_resolutions


class _DummyModel:
    NAME = "Dummy"
    PIXEL_PREVIEW_INPUTS = {}

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
    def test_if_pipelines_stream_their_post_step_rgb(self, tracker):
        from diffusers import DDPMScheduler, IFPipeline, IFSuperResolutionPipeline, UNet2DConditionModel

        from simpletuner.helpers.models.deepfloyd.model import DeepFloydIF

        preview = ValidationPreviewer(
            DeepFloydIF.__new__(DeepFloydIF),
            None,
            types.SimpleNamespace(validation_preview=True, validation_preview_steps=2),
        )
        handler = tracker.get_webhook_handler.return_value
        for stage, pipeline_type in enumerate((IFPipeline, IFSuperResolutionPipeline), 1):
            with self.subTest(stage=stage):
                handler.reset_mock()
                unet = UNet2DConditionModel(
                    sample_size=8,
                    in_channels=3 * stage,
                    out_channels=6,
                    block_out_channels=(8,),
                    down_block_types=("DownBlock2D",),
                    up_block_types=("UpBlock2D",),
                    layers_per_block=1,
                    cross_attention_dim=8,
                    attention_head_dim=4,
                    norm_num_groups=4,
                )
                components = dict(
                    unet=unet,
                    scheduler=DDPMScheduler(),
                    text_encoder=None,
                    tokenizer=None,
                    safety_checker=None,
                    feature_extractor=None,
                    watermarker=None,
                    requires_safety_checker=False,
                )
                kwargs = dict(
                    prompt_embeds=torch.ones(1, 2, 8),
                    num_inference_steps=2,
                    guidance_scale=1.0,
                    height=8,
                    width=8,
                    output_type="pt",
                )
                if stage == 2:
                    components["image_noising_scheduler"] = DDPMScheduler()
                    kwargs["image"] = torch.zeros(1, 3, 4, 4)
                pipeline = pipeline_type(**components)
                pipeline.set_progress_bar_config(disable=True)
                metadata = _PreviewMetadata("test", "test prompt", (8, 8), "checkpoint", 2, f"stage{stage}")
                with preview.attach(pipeline, kwargs, metadata):
                    result = pipeline(**kwargs).images
                self.assertEqual(handler.send_raw.call_count, 2)
                last = handler.send_raw.call_args.kwargs
                self.assertEqual(last["structured_data"]["data"]["stage"], f"stage{stage}")
                expected = preview._tensor_to_pil((result[0] + 1) / 2)
                self.assertEqual(last["images"][0].tobytes(), expected.tobytes())
                self.assertNotIn("callback", kwargs)

        class LatentUpscaler:
            def __call__(self, callback=None, callback_steps=1):
                self.callback = callback

        upscaler = LatentUpscaler()
        kwargs = {}
        with preview.attach(upscaler, kwargs, metadata):
            upscaler(**kwargs)
        self.assertIsNone(upscaler.callback)

    @patch("simpletuner.helpers.training.validation.StateTracker")
    def test_tae_preview_preserves_existing_step_callback_and_inputs(self, tracker):
        class Pipeline:
            def __call__(self, callback_on_step_end=None, callback_on_step_end_tensor_inputs=None):
                tensors = {key: torch.zeros(1, 4, 8, 8) for key in callback_on_step_end_tensor_inputs}
                return callback_on_step_end(self, 0, 1, tensors)

        model = _DummyModel()
        pipeline = Pipeline()
        preview = ValidationPreviewer(model, None, types.SimpleNamespace(validation_preview=True))
        previous = MagicMock(side_effect=lambda pipe, step, timestep, tensors: {"latents": tensors["latents"] + 1})
        kwargs = {"callback_on_step_end": previous, "callback_on_step_end_tensor_inputs": ["prompt_embeds"]}
        original = kwargs.copy()
        metadata = _PreviewMetadata("test", "prompt", (8, 8), "checkpoint")
        with preview.attach(pipeline, kwargs, metadata):
            out = pipeline(**kwargs)
        self.assertEqual(out["latents"].mean().item(), 1)
        previous.assert_called_once()
        tracker.get_webhook_handler.return_value.send_raw.assert_called_once()
        self.assertEqual(kwargs, original)

    def test_parse_validation_resolutions_supports_comma_separated_formats(self):
        self.assertEqual(
            parse_validation_resolutions("512, 768x512", model_flavour=""),
            [(512, 512), (768, 512)],
        )
        self.assertEqual(parse_validation_resolutions(512, model_flavour=""), [(512, 512)])

    def test_parse_validation_resolutions_rejects_invalid_items(self):
        with self.assertRaisesRegex(ValueError, "Invalid validation resolution"):
            parse_validation_resolutions("512,wide", model_flavour="")

    @patch("simpletuner.helpers.training.validation.StateTracker")
    def test_internal_callback_enables_preview_without_webhook(self, mock_state_tracker):
        mock_state_tracker.get_webhook_handler.return_value = None
        callback = MagicMock()
        config = types.SimpleNamespace(
            validation_preview=True,
            validation_preview_steps=1,
            _validation_preview_callback=callback,
        )
        previewer = ValidationPreviewer(_DummyModel(), types.SimpleNamespace(is_main_process=True), config)
        metadata = _PreviewMetadata(
            shortname="inference",
            prompt="preview prompt",
            resolution=(512, 512),
            validation_type="checkpoint",
        )

        previewer._emit_event([object()], None, metadata, step=0, timestep=1.0)

        self.assertTrue(previewer.enabled)
        callback.assert_called_once()
        self.assertEqual(callback.call_args.kwargs["structured_data"]["data"]["step_label"], "1")

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
