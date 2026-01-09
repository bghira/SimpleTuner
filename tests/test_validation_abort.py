"""Test that validation properly responds to abort signals during pipeline execution."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.common import AudioModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.training.validation import Validation, ValidationAbortedException


class MockAudioModel(AudioModelFoundation):
    PREDICTION_TYPE = PredictionTypes.EPSILON
    MODEL_TYPE = ModelTypes.TRANSFORMER
    NAME = "MockAudioModel"
    DEFAULT_PIPELINE_TYPE = PipelineTypes.TEXT2AUDIO
    PIPELINE_CLASSES = {PipelineTypes.TEXT2AUDIO: MagicMock}
    VALIDATION_USES_NEGATIVE_PROMPT = False

    def __init__(self):
        config = MagicMock()
        config.model_family = "ace_step"
        config.pretrained_model_name_or_path = "dummy_path"
        config.twinflow_enabled = False
        super().__init__(config=config, accelerator=MagicMock())
        self.pipeline = MagicMock()
        self.controlnet = None
        self.vae = None
        self.text_encoders = None

    def setup_training_noise_schedule(self):
        self.noise_schedule = MagicMock()
        return self.config, self.noise_schedule

    def requires_conditioning_validation_inputs(self):
        return False

    def validation_image_input_edge_length(self):
        return None

    def update_pipeline_call_kwargs(self, kwargs):
        return kwargs

    def convert_text_embed_for_pipeline(self, embed, prompt=None):
        return {}

    def _encode_prompts(self, prompts, is_validation=False, is_negative_prompt=False):
        return {}

    def convert_negative_text_embed_for_pipeline(self, prompt, text_embedding):
        return {}

    def model_predict(self, *args, **kwargs):
        return None

    def get_pipeline(self, **kwargs):
        return self.pipeline

    def move_models(self, device):
        pass


class TestValidationAbort(unittest.TestCase):
    @patch("simpletuner.helpers.training.validation.StateTracker")
    @patch("simpletuner.helpers.training.validation.validation_audio.save_audio")
    @patch("simpletuner.helpers.training.validation.prepare_validation_prompt_list")
    @patch("simpletuner.helpers.training.validation._normalise_validation_sample")
    @patch("simpletuner.helpers.training.validation.Validation.setup_scheduler")
    @patch("simpletuner.helpers.training.validation.tqdm")
    def test_abort_during_pipeline_execution(
        self, mock_tqdm, mock_setup_scheduler, mock_normalise, mock_prepare_prompts, mock_save_audio, mock_state_tracker
    ):
        """Test that abort signal during pipeline execution raises ValidationAbortedException."""
        # Setup mocks
        mock_tqdm.side_effect = lambda x, **kwargs: x
        mock_setup_scheduler.return_value = None
        mock_normalise.return_value = "mock_sample"
        mock_prepare_prompts.return_value = {
            "validation_prompts": ["test prompt"],
            "validation_shortnames": ["test"],
            "validation_sample_images": [],
        }
        mock_state_tracker.get_global_resume_step.return_value = 0
        mock_state_tracker.get_global_step.return_value = 1
        mock_state_tracker.get_epoch.return_value = 0
        mock_state_tracker.get_epoch_step.return_value = 0
        mock_state_tracker.get_webhook_handler.return_value = None

        mock_accelerator = MagicMock()
        mock_accelerator.is_main_process = True
        mock_accelerator.device = torch.device("cpu")
        mock_accelerator.num_processes = 1

        mock_model = MockAudioModel()

        # Track callback invocations and abort on second call
        callback_count = [0]

        def mock_pipeline_call(
            prompt=None,
            num_inference_steps=None,
            guidance_scale=None,
            num_images_per_prompt=None,
            generator=None,
            callback_on_step_end=None,
            **kwargs,
        ):
            # Simulate the pipeline calling the callback
            if callback_on_step_end:
                # Simulate 3 denoising steps
                for step in range(3):
                    callback_count[0] += 1
                    mock_callback_kwargs = {"latents": torch.zeros(1, 4, 8, 8)}
                    callback_on_step_end(None, step, torch.tensor(1000 - step * 100), mock_callback_kwargs)

            # Return mock result
            mock_result = MagicMock()
            mock_result.audios = [torch.randn(1, 16000)]
            del mock_result.frames
            del mock_result.images
            return mock_result

        # Create a mock with a proper signature
        mock_model.pipeline = MagicMock(side_effect=mock_pipeline_call)
        mock_model.pipeline.__call__ = mock_pipeline_call

        # Create a config that will abort on second callback
        mock_config = MagicMock()
        mock_config.validation_num_inference_steps = 3
        mock_config.num_validation_images = 1
        mock_config.validation_guidance = 1.0
        mock_config.validation_guidance_real = 1.0
        mock_config.validation_no_cfg_until_timestep = None
        mock_config.model_family = "ace_step"
        mock_config.validation_randomize = False
        mock_config.use_ema = False
        mock_config.validation_adapter_mode = "none"
        mock_config.validation_adapter_path = None
        mock_config.validation_adapter_config = None
        mock_config.validation_adapter_name = None
        mock_config.validation_adapter_strength = 1.0
        mock_config.validation_noise_scheduler = "ddim"
        mock_config.validation_seed_source = "cpu"
        mock_config.validation_seed = 42
        mock_config.weight_dtype = torch.float32
        mock_config.controlnet = False
        mock_config.control = False
        mock_config.output_dir = "/tmp/test_validation"
        mock_config.validation_preview = False

        # Abort on second callback invocation
        def should_abort_check():
            return callback_count[0] >= 2

        mock_config.should_abort = should_abort_check

        mock_embed_cache = MagicMock()
        mock_embed_cache.compute_embeddings_for_prompts.return_value = None

        # Instantiate Validation
        validation = Validation(
            accelerator=mock_accelerator,
            model=mock_model,
            distiller=None,
            args=mock_config,
            validation_prompt_metadata={
                "validation_prompts": ["test prompt"],
                "validation_shortnames": ["test"],
                "validation_sample_images": ["mock_sample"],
            },
            vae_path=None,
            weight_dtype=torch.float32,
            embed_cache=mock_embed_cache,
            ema_model=None,
        )

        # Run validation - should raise ValidationAbortedException
        validation.global_resume_step = 0
        with self.assertRaises(ValidationAbortedException):
            validation.run_validations(step=1, validation_type="intermediary", force_evaluation=True)

        # Verify the callback was invoked at least twice (abort happened during execution)
        self.assertGreaterEqual(callback_count[0], 2, "Callback should have been called at least twice before abort")


if __name__ == "__main__":
    unittest.main()
