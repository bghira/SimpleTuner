import unittest
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.common import AudioModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.training.validation import Validation


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
        super().__init__(config=config, accelerator=MagicMock())
        self.pipeline = MagicMock()
        self.controlnet = None
        self.vae = None
        self.text_encoders = None

    def setup_training_noise_schedule(self):
        self.noise_schedule = MagicMock()
        return self.config, self.noise_schedule
        self.NAME = "MockAudioModel"
        self.PIPELINE_CLASSES = {}
        self.DEFAULT_PIPELINE_TYPE = None
        self.VALIDATION_USES_NEGATIVE_PROMPT = False
        self.config = MagicMock()
        self.config.controlnet = False
        self.config.weight_dtype = torch.float32
        self.model = MagicMock()
        self.accelerator = MagicMock()
        self.accelerator.device = "cpu"
        self.controlnet = None
        self.vae = None
        self.text_encoders = None

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


class TestAudioValidation(unittest.TestCase):
    @patch("simpletuner.helpers.training.validation.StateTracker")
    @patch("simpletuner.helpers.training.validation.validation_audio.save_audio")
    @patch("simpletuner.helpers.training.validation.prepare_validation_prompt_list")
    @patch("simpletuner.helpers.training.validation._normalise_validation_sample")
    @patch("simpletuner.helpers.training.validation.Validation.setup_scheduler")
    def test_audio_validation_flow(
        self, mock_setup_scheduler, mock_normalise, mock_prepare_prompts, mock_save_audio, mock_state_tracker
    ):
        # Setup mocks
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

        mock_accelerator = MagicMock()
        mock_accelerator.is_main_process = True
        mock_accelerator.device = torch.device("cpu")
        mock_accelerator.num_processes = 1

        mock_model = MockAudioModel()
        mock_model.pipeline = MagicMock()

        # Mock pipeline output
        mock_pipeline_result = MagicMock()
        mock_pipeline_result.audios = [torch.randn(1, 16000)]  # 1 sec audio
        del mock_pipeline_result.frames
        del mock_pipeline_result.images
        mock_model.pipeline.return_value = mock_pipeline_result

        mock_config = MagicMock()
        mock_config.validation_num_inference_steps = 1
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

        # Run validation
        validation.global_resume_step = 0
        validation.run_validations(step=1, validation_type="intermediary", force_evaluation=True)

        # Verify save_audio was called
        self.assertTrue(mock_save_audio.called)
        call_args = mock_save_audio.call_args
        # args: (save_dir, validation_images, validation_shortname)
        # validation_images is a dict {shortname: [audio_tensors]}
        self.assertIn("test", call_args[0][1])
        self.assertEqual(len(call_args[0][1]["test"]), 1)


if __name__ == "__main__":
    unittest.main()
