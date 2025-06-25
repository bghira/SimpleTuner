import unittest
from unittest.mock import MagicMock, patch
import os
import json

from helpers.publishing.metadata import (
    _negative_prompt,
    _torch_device,
    _model_imports,
    _model_load,
    _validation_resolution,
    _skip_layers,
    _guidance_rescale,
)
from helpers.publishing.metadata import *


class TestMetadataFunctions(unittest.TestCase):
    def setUp(self):
        # Mock the args object
        self.args = MagicMock()
        self.args.lora_type = "standard"
        self.args.model_type = "lora"
        self.args.model_family = "sdxl"
        self.args.validation_prompt = "A test prompt"
        self.args.validation_negative_prompt = "A negative prompt"
        self.args.validation_num_inference_steps = 50
        self.args.validation_guidance = 7.5
        self.args.validation_guidance_rescale = 0.7
        self.args.validation_resolution = "512x512"
        self.args.pretrained_model_name_or_path = "test-model"
        self.args.output_dir = "test-output"
        self.args.lora_rank = 4
        self.args.lora_alpha = 1.0
        self.args.lora_dropout = 0.0
        self.args.lora_init_type = "kaiming_uniform"
        self.args.model_card_note = "Test note"
        self.args.validation_using_datasets = False
        self.args.flow_matching_loss = "compatible"
        self.args.flux_fast_schedule = False
        self.args.flow_schedule_auto_shift = False
        self.args.flow_schedule_shift = None
        self.args.flux_guidance_value = None
        self.args.flux_guidance_min = None
        self.args.flux_guidance_max = None
        self.args.flow_use_beta_schedule = False
        self.args.flow_beta_schedule_alpha = None
        self.args.flow_beta_schedule_beta = None
        self.args.flux_attention_masked_training = False
        self.args.flow_use_uniform_schedule = False
        self.args.flux_lora_target = None
        self.args.validation_guidance_skip_layers = None
        self.args.validation_seed = 1234
        self.args.validation_noise_scheduler = "ddim"
        self.args.model_card_safe_for_work = True
        self.args.learning_rate = 1e-4
        self.args.max_grad_norm = 1.0
        self.args.train_batch_size = 4
        self.args.gradient_accumulation_steps = 1
        self.args.optimizer = "AdamW"
        self.args.optimizer_config = ""
        self.args.mixed_precision = "fp16"
        self.args.base_model_precision = "no_change"
        self.args.flux_guidance_mode = "constant"
        self.args.flux_guidance_value = 1.0
        self.args.t5_padding = "unmodified"
        self.args.enable_xformers_memory_efficient_attention = False
        self.args.attention_mechanism = "diffusers"
        self.mock_model = MagicMock(MODEL_TYPE=MagicMock(value="unet"))

    def test_model_imports(self):
        self.args.lora_type = "standard"
        self.args.model_type = "lora"
        expected_output = "import torch\nfrom diffusers import DiffusionPipeline"
        output = _model_imports(self.args)
        self.assertEqual(output.strip(), expected_output.strip())

        self.args.lora_type = "lycoris"
        output = _model_imports(self.args)
        self.assertIn("from lycoris import create_lycoris_from_weights", output)

    def test_model_load(self):
        self.args.pretrained_model_name_or_path = "pretrained-model"
        self.args.output_dir = "output-dir"
        self.args.lora_type = "standard"
        self.args.model_type = "lora"

        with patch(
            "helpers.publishing.metadata.StateTracker.get_hf_username",
            return_value="testuser",
        ):
            output = _model_load(self.args, repo_id="repo-id", model=self.mock_model)
            self.assertIn("pipeline.load_lora_weights", output)
            self.assertIn("adapter_id = 'testuser/repo-id'", output)

        self.args.lora_type = "lycoris"
        output = _model_load(self.args, model=self.mock_model)
        self.assertIn("pytorch_lora_weights.safetensors", output)

    def test_torch_device(self):
        output = _torch_device()
        expected_output = "'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'"
        self.assertEqual(output.strip(), expected_output.strip())

    def test_negative_prompt(self):
        self.args.model_family = "sdxl"
        output = _negative_prompt(self.args)
        expected_output = "negative_prompt = 'A negative prompt'"
        self.assertEqual(output.strip(), expected_output.strip())

        output_in_call = _negative_prompt(self.args, in_call=True)
        self.assertIn("negative_prompt=negative_prompt", output_in_call)

    def test_guidance_rescale(self):
        self.args.model_family = "sdxl"
        output = _guidance_rescale(self.mock_model)
        self.assertNotEqual(output.strip(), "")

    def test_skip_layers(self):
        self.args.model_family = "sd3"
        self.args.validation_guidance_skip_layers = 2
        output = _skip_layers(self.args)
        expected_output = "\n    skip_guidance_layers=2,"
        self.assertEqual(output.strip(), expected_output.strip())

    def test_validation_resolution(self):
        self.args.validation_resolution = "512x512"
        output = _validation_resolution(self.args)
        expected_output = "width=512,\n    height=512,"
        self.assertEqual(output.strip(), expected_output.strip())

        self.args.validation_resolution = ""
        output = _validation_resolution(self.args)
        expected_output = "width=1024,\n    height=1024,"
        self.assertEqual(output.strip(), expected_output.strip())

    def test_code_example(self):
        with patch(
            "helpers.publishing.metadata._model_imports",
            return_value="import torch\nfrom diffusers import DiffusionPipeline",
        ):
            with patch(
                "helpers.publishing.metadata._model_load", return_value="pipeline = ..."
            ):
                with patch(
                    "helpers.publishing.metadata._torch_device", return_value="'cuda'"
                ):
                    with patch(
                        "helpers.publishing.metadata._negative_prompt",
                        return_value="negative_prompt = 'A negative prompt'",
                    ):
                        with patch(
                            "helpers.publishing.metadata._validation_resolution",
                            return_value="width=512,\n    height=512,",
                        ):
                            output = code_example(self.args, None, self.mock_model)
                            self.assertIn("import torch", output)
                            self.assertIn("pipeline = ...", output)
                            self.assertIn("pipeline.to('cuda')", output)

    def test_model_type(self):
        self.args.model_type = "lora"
        self.args.lora_type = "standard"
        output = model_type(self.args)
        self.assertEqual(output, "standard PEFT LoRA")

        self.args.lora_type = "lycoris"
        output = model_type(self.args)
        self.assertEqual(output, "LyCORIS adapter")

        self.args.model_type = "full"
        output = model_type(self.args)
        self.assertEqual(output, "full rank finetune")

    def test_lora_info(self):
        self.args.model_type = "lora"
        self.args.lora_type = "standard"
        output = lora_info(self.args)
        self.assertIn("LoRA Rank: 4", output)

        self.args.lora_type = "lycoris"
        # Mocking the file reading
        lycoris_config = {"key": "value"}
        with patch(
            "builtins.open",
            unittest.mock.mock_open(read_data=json.dumps(lycoris_config)),
        ):
            output = lora_info(self.args)
            self.assertIn('"key": "value"', output)

    def test_model_card_note(self):
        output = model_card_note(self.args)
        self.assertIn("Test note", output)

        self.args.model_card_note = ""
        output = model_card_note(self.args)
        self.assertEqual(output.strip(), "")

    def test_save_model_card(self):
        # Mocking StateTracker methods
        self.args.model_family = "flux"
        self.args.model_type = "lora"
        self.args.lora_type = "lycoris"
        self.args.base_model_precision = "int8-quanto"
        with patch(
            "helpers.publishing.metadata.StateTracker.get_model_family",
            return_value="sdxl",
        ):
            with patch(
                "helpers.publishing.metadata.StateTracker.get_data_backends",
                return_value={},
            ):
                with patch(
                    "helpers.publishing.metadata.StateTracker.get_epoch", return_value=1
                ):
                    with patch(
                        "helpers.publishing.metadata.StateTracker.get_global_step",
                        return_value=1000,
                    ):
                        with patch(
                            "helpers.publishing.metadata.StateTracker.get_weight_dtype",
                            return_value=torch.bfloat16,
                        ):
                            with patch(
                                "helpers.publishing.metadata.StateTracker.get_accelerator",
                                return_value=MagicMock(num_processes=1),
                            ):
                                with patch(
                                    "helpers.training.state_tracker.StateTracker.get_args",
                                    return_value=self.args,
                                ):
                                    with patch(
                                        "builtins.open", unittest.mock.mock_open()
                                    ) as mock_file:
                                        save_model_card(
                                            repo_id="test-repo",
                                            images=None,
                                            base_model="test-base-model",
                                            train_text_encoder=True,
                                            prompt="Test prompt",
                                            validation_prompts=["Test prompt"],
                                            validation_shortnames=["shortname"],
                                            repo_folder="test-folder",
                                            model=MagicMock(),
                                        )
                                        # Ensure the README.md was written
                                        mock_file.assert_called_with(
                                            os.path.join("test-folder", "README.md"),
                                            "w",
                                            encoding="utf-8",
                                        )

    def test_adapter_download_fn(self):
        with patch("huggingface_hub.hf_hub_download", return_value="path/to/adapter"):
            from helpers.publishing.metadata import lycoris_download_info

            output = lycoris_download_info()
            self.assertIn("hf_hub_download", output)

    def test_pipeline_move_full_bf16(self):
        from helpers.publishing.metadata import _pipeline_move_to

        with patch(
            "helpers.training.state_tracker.StateTracker.get_weight_dtype",
            return_value=torch.bfloat16,
        ):
            output = _pipeline_move_to(args=self.args)

        self.assertNotIn("torch.bfloat16", output)

    def test_pipeline_move_lycoris_bf16(self):
        from helpers.publishing.metadata import _pipeline_move_to

        with patch(
            "helpers.training.state_tracker.StateTracker.get_weight_dtype",
            return_value=torch.bfloat16,
        ):
            self.args.model_type = "lora"
            self.args.lora_type = "lycoris"
            self.args.base_model_precision = "no_change"
            output = _pipeline_move_to(args=self.args)
        self.assertNotIn("torch.bfloat16", output)

    def test_pipeline_move_lycoris_int8(self):
        from helpers.publishing.metadata import _pipeline_move_to

        with patch(
            "helpers.training.state_tracker.StateTracker.get_weight_dtype",
            return_value=torch.bfloat16,
        ):
            self.args.model_type = "lora"
            self.args.lora_type = "lycoris"
            self.args.base_model_precision = "int8-quanto"
            output = _pipeline_move_to(args=self.args)
        self.assertNotIn("torch.bfloat16", output)

    def test_pipeline_quanto_hint_unet(self):
        from helpers.publishing.metadata import _pipeline_quanto

        self.mock_model.MODEL_TYPE = MagicMock(value="unet")
        output = _pipeline_quanto(args=self.args, model=self.mock_model)

        self.assertIn("quantize", output)
        self.assertIn("optimum.quanto", output)
        self.assertIn("pipeline.unet", output)

    def test_pipeline_quanto_hint_transformer(self):
        from helpers.publishing.metadata import _pipeline_quanto

        self.args.model_family = "flux"
        self.mock_model.MODEL_TYPE = MagicMock(value="transformer")
        output = _pipeline_quanto(args=self.args, model=self.mock_model)
        self.assertIn("quantize", output)
        self.assertIn("optimum.quanto", output)
        self.assertIn("pipeline.transformer", output)


if __name__ == "__main__":
    unittest.main()
