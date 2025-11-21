import unittest
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.common import PipelineTypes
from simpletuner.helpers.models.sanavideo.model import SanaVideo


class TestSanaVideoModel(unittest.TestCase):
    def setUp(self):
        self.mock_accelerator = MagicMock()
        self.mock_accelerator.device = torch.device("cpu")

        self.config = MagicMock()
        self.config.weight_dtype = torch.float32
        self.config.model_family = "sanavideo"
        self.config.pretrained_model_name_or_path = "dummy_path"
        self.config.vae_latent_channels = 16
        self.config.base_model_precision = "fp16"
        self.config.sana_motion_score = 30.0
        self.config.sana_complex_human_instruction = True
        self.config.validation_num_video_frames = 81
        self.config.flow_schedule_shift = 1.0
        self.config.tracker_run_name = "test_run"

        # Mock the pipeline classes so they don't try to load real models
        with (
            patch("simpletuner.helpers.models.sanavideo.model.FlowMatchEulerDiscreteScheduler"),
            patch("simpletuner.helpers.models.sanavideo.model.AutoencoderKLWan"),
            patch("simpletuner.helpers.models.sanavideo.model.SanaVideoTransformer3DModel"),
            patch("simpletuner.helpers.models.sanavideo.model.GemmaTokenizerFast"),
            patch("simpletuner.helpers.models.sanavideo.model.Gemma2Model"),
            patch("simpletuner.helpers.models.sanavideo.model.SanaVideoPipeline") as mock_pipeline_cls,
        ):

            # Setup pipeline mock instance
            self.mock_pipeline = MagicMock()
            mock_pipeline_cls.return_value = self.mock_pipeline
            self.mock_pipeline.encode_prompt.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

            # We need to mock the pipeline dictionary that ModelFoundation creates
            # But ModelFoundation creates it by checking PIPELINE_CLASSES.
            # Since we patch the module where SanaVideo imports from, it should work?
            # Wait, SanaVideo imports SanaVideoPipeline from .pipeline.
            # So patching 'simpletuner.helpers.models.sanavideo.model.SanaVideoPipeline' is correct.

            self.model = SanaVideo(self.config, self.mock_accelerator)

            # Inject the mocked pipeline into the pipelines dict manually to be sure
            self.model.pipelines = {PipelineTypes.TEXT2IMG: self.mock_pipeline}

            # Setup scheduler mock
            self.mock_scheduler = MagicMock()
            self.mock_scheduler.config.num_train_timesteps = 1000
            self.mock_scheduler.sigmas = torch.linspace(0, 1, 1000)
            self.mock_scheduler.timesteps = torch.linspace(1000, 0, 1000)
            self.model.noise_schedule = self.mock_scheduler

    def test_update_pipeline_call_kwargs(self):
        kwargs = {}
        updated_kwargs = self.model.update_pipeline_call_kwargs(kwargs)
        self.assertEqual(updated_kwargs["frames"], 81)

    def test_format_text_embedding(self):
        # text_embedding is expected to be a 4-tuple in SanaVideo
        embed_tuple = ("prompt_embeds", "prompt_mask", "neg_embeds", "neg_mask")
        formatted = self.model._format_text_embedding(embed_tuple)
        self.assertEqual(formatted["prompt_embeds"], "prompt_embeds")
        self.assertEqual(formatted["prompt_attention_mask"], "prompt_mask")
        self.assertEqual(formatted["negative_prompt_embeds"], "neg_embeds")
        self.assertEqual(formatted["negative_prompt_attention_mask"], "neg_mask")

    def test_convert_text_embed_for_pipeline(self):
        embed_dict = {
            "prompt_embeds": "p",
            "prompt_attention_mask": "pm",
            "negative_prompt_embeds": "n",
            "negative_prompt_attention_mask": "nm",
        }
        converted = self.model.convert_text_embed_for_pipeline(embed_dict)
        self.assertEqual(converted, embed_dict)

    def test_encode_prompts_with_motion_score(self):
        prompts = ["A cat"]
        self.model._encode_prompts(prompts, is_negative_prompt=False)

        args, kwargs = self.mock_pipeline.encode_prompt.call_args

        # Check that prompt was modified
        self.assertIn("motion score: 30.0", kwargs["prompt"][0])
        # Check complex human instruction
        self.assertEqual(kwargs["complex_human_instruction"], self.model.COMPLEX_HUMAN_INSTRUCTION)

    def test_encode_prompts_negative(self):
        prompts = ["Bad quality"]
        self.model._encode_prompts(prompts, is_negative_prompt=True)

        args, kwargs = self.mock_pipeline.encode_prompt.call_args

        # Check that prompt was NOT modified with motion score
        self.assertNotIn("motion score", kwargs["prompt"][0])
        # Check complex human instruction is None
        self.assertIsNone(kwargs["complex_human_instruction"])

    def test_check_user_config_errors(self):
        self.config.base_model_precision = "fp8-quanto"
        with self.assertRaises(ValueError):
            self.model.check_user_config()

    def test_loss_calculation(self):
        batch_size = 2
        channels = 16
        frames = 5
        height = 16
        width = 16

        model_pred = torch.randn(batch_size, channels, frames, height, width)
        target = torch.randn(batch_size, channels, frames, height, width)
        sigmas = torch.rand(batch_size)

        prepared_batch = {
            "sigmas": sigmas,
            "noisy_latents": torch.randn_like(model_pred),
        }
        model_output = {"model_prediction": model_pred}

        # Mock get_prediction_target to return our target
        with patch.object(self.model, "get_prediction_target", return_value=target):
            loss = self.model.loss(prepared_batch, model_output)

        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.ndim, 0)


if __name__ == "__main__":
    unittest.main()
