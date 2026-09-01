import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from simpletuner.helpers.publishing.huggingface import HubManager
from simpletuner.helpers.publishing.metadata import *
from simpletuner.helpers.publishing.metadata import (
    _audio_dataset_overview,
    _guidance_rescale,
    _license_metadata,
    _model_imports,
    _model_load,
    _negative_prompt,
    _pipeline_tag,
    _secondary_pipeline_tag,
    _skip_layers,
    _torch_device,
    _validation_resolution,
)
from simpletuner.helpers.training.attention_backend import AttentionBackendMode


class TestMetadataFunctions(unittest.TestCase):
    def tearDown(self):
        """Clean up test-folder after each test."""
        import shutil

        if os.path.exists("test-folder"):
            shutil.rmtree("test-folder", ignore_errors=True)
        super().tearDown()

    def setUp(self):
        # Mock the args object
        self.args = MagicMock()
        self.args.lora_type = "standard"
        self.args.model_type = "lora"
        self.args.model_family = "sdxl"
        self.args.validation_prompt = "A test prompt"
        self.args.validation_disable = False
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
        self.args.validation_image_format = "png"
        self.args.validation_image_quality = 90
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
        self.args.peft_lora_mode = "standard"
        self.args.t5_padding = "unmodified"
        self.args.enable_xformers_memory_efficient_attention = False
        self.args.attention_mechanism = "diffusers"
        self.args.minimax_music_train_component = None
        self.args.minimax_music_lm_max_frames = None
        self.args.minimax_music_lm_window_mode = "prefix"
        self.args.nextlat_enabled = False
        self.args.nextlat_block_index = -1
        self.args.nextlat_weight = 0.0
        self.args.nextlat_state_loss = "smooth_l1"
        self.args.nextlat_kl_weight = 0.0
        self.args.xm_enabled = False
        self.args.xm_candidate_count = 1
        self.args.xm_selection_scope = "sample"
        self.args.xm_training_target = "noise"
        self.args.xm_block_size = 0
        self.mock_model = MagicMock(MODEL_TYPE=MagicMock(value="unet"))

    def test_model_imports(self):
        self.args.lora_type = "standard"
        self.args.model_type = "lora"
        expected_output = "import torch\nfrom diffusers import DiffusionPipeline"
        with patch(
            "simpletuner.helpers.publishing.metadata.StateTracker.get_model",
            return_value=None,
        ):
            output = _model_imports(self.args)
            self.assertEqual(output.strip(), expected_output.strip())

            self.args.lora_type = "lycoris"
            output = _model_imports(self.args)
            self.assertIn("from lycoris import create_lycoris_from_weights", output)

    def test_audio_dataset_overview_does_not_reload_live_metadata(self):
        metadata_backend = MagicMock()
        metadata_backend.image_metadata = {"0.wav": {"sample_rate": 44100, "num_channels": 2, "dataset_type": "audio"}}
        metadata_backend.aspect_ratio_bucket_indices = {"180s": ["0.wav"]}
        sampler = SimpleNamespace(is_regularisation_data=True, conditioning_type=None)

        with patch(
            "simpletuner.helpers.publishing.metadata.StateTracker.get_data_backend_config",
            return_value={"repeats": 0},
        ):
            overview = _audio_dataset_overview(
                "audio",
                {"metadata_backend": metadata_backend, "sampler": sampler},
            )

        metadata_backend.load_image_metadata.assert_not_called()
        self.assertIn("Total number of audio files: 1", overview)
        self.assertIn("Sample rate: 44.1 kHz", overview)
        self.assertIn("Channels: 2", overview)

    def test_hub_manager_loads_standard_environment_token(self):
        manager = object.__new__(HubManager)
        manager.config = SimpleNamespace(push_to_hub=True)

        with patch.dict(os.environ, {"HF_TOKEN": "environment-token"}, clear=False):
            self.assertEqual(manager._load_hub_token(), "environment-token")

    def test_model_load(self):
        self.args.pretrained_model_name_or_path = "pretrained-model"
        self.args.output_dir = "output-dir"
        self.args.lora_type = "standard"
        self.args.model_type = "lora"

        with (
            patch(
                "simpletuner.helpers.publishing.metadata.StateTracker.get_hf_username",
                return_value="testuser",
            ),
            patch(
                "simpletuner.helpers.publishing.metadata.StateTracker.get_model",
                return_value=None,
            ),
        ):
            output = _model_load(self.args, repo_id="repo-id", model=self.mock_model)
            self.assertIn("pipeline.load_lora_weights", output)
            self.assertIn("adapter_id = 'testuser/repo-id'", output)

            output = _model_load(self.args, repo_id="testuser/repo-id", model=self.mock_model)
            self.assertIn("adapter_id = 'testuser/repo-id'", output)
            self.assertNotIn("testuser/testuser/repo-id", output)

        with patch(
            "simpletuner.helpers.publishing.metadata.StateTracker.get_model",
            return_value=None,
        ):
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

    def test_model_type(self):
        self.args.model_type = "lora"
        self.args.lora_type = "standard"
        self.args.controlnet = False
        self.args.control = False
        with patch(
            "simpletuner.helpers.publishing.metadata.StateTracker.get_model",
            return_value=None,
        ):
            output = model_type(self.args)
            self.assertEqual(output, "PEFT LoRA")

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

    def test_minimax_h3_model_card_metadata(self):
        self.args.model_family = "minimaxh3"
        model = SimpleNamespace(
            MODEL_LICENSE="other",
            MODEL_LICENSE_NAME="minimax-h3-community-license-agreement",
            MODEL_LICENSE_LINK="https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE",
        )

        self.assertEqual("text-to-video", _pipeline_tag(self.args))
        self.assertEqual(
            "license: other\n"
            'license_name: "minimax-h3-community-license-agreement"\n'
            'license_link: "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"',
            _license_metadata(model),
        )

    def test_audio_model_card_writes_audio_widget_asset(self):
        self.args.model_family = "minimaxmusic"
        self.args.model_type = "lora"
        self.args.lora_type = "standard"
        self.args.peft_lora_mode = "standard"
        self.args.controlnet = False
        self.args.control = False
        model = MagicMock(
            MODEL_LICENSE="other",
            PREDICTION_TYPE=SimpleNamespace(value="flow_matching"),
            gligen=False,
        )
        model.validation_audio_sample_rate.return_value = 44100
        model.custom_model_card_schedule_info.return_value = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("simpletuner.helpers.publishing.metadata.StateTracker.get_model_family", return_value="minimaxmusic"),
                patch("simpletuner.helpers.publishing.metadata.StateTracker.get_data_backends", return_value={}),
                patch("simpletuner.helpers.publishing.metadata.StateTracker.get_weight_dtype", return_value=torch.bfloat16),
                patch(
                    "simpletuner.helpers.publishing.metadata.StateTracker.get_accelerator",
                    return_value=MagicMock(num_processes=1),
                ),
                patch("simpletuner.helpers.publishing.metadata.StateTracker.get_args", return_value=self.args),
                patch("simpletuner.helpers.publishing.metadata.StateTracker.get_model", return_value=model),
            ):
                save_model_card(
                    repo_id="test-repo",
                    images=None,
                    audios={"song": [torch.zeros(1, 1600)]},
                    base_model="test-base-model",
                    train_text_encoder=False,
                    prompt="Test prompt",
                    validation_prompts=["bright reggae"],
                    validation_shortnames=["song"],
                    repo_folder=tmpdir,
                    model=model,
                    global_step=1000,
                    epoch=1,
                )

            readme = Path(tmpdir, "README.md").read_text(encoding="utf-8")
            self.assertIn("pipeline_tag: text-to-audio", readme)
            self.assertIn("  - audio", readme)
            self.assertIn("url: ./assets/audio_0_0.wav", readme)
            self.assertIn("You can find some example audio samples", readme)
            self.assertTrue(Path(tmpdir, "assets", "audio_0_0.wav").exists())

    def test_audio_model_families_use_audio_pipeline_tag(self):
        for model_family in ("ace_step", "heartmula", "minimaxmusic"):
            with self.subTest(model_family=model_family):
                self.args.model_family = model_family
                self.assertEqual(_pipeline_tag(self.args), "text-to-audio")
                self.assertEqual(_secondary_pipeline_tag(self.args), "audio")

    def test_minimax_music_model_card_reports_modes_and_disabled_validation(self):
        self.args.model_family = "minimaxmusic"
        self.args.model_type = "lora"
        self.args.lora_type = "standard"
        self.args.peft_lora_mode = "standard"
        self.args.controlnet = False
        self.args.control = False
        self.args.validation_disable = True
        self.args.model_card_note = ""
        self.args.minimax_music_train_component = "language_model"
        self.args.minimax_music_lm_max_frames = 128
        self.args.minimax_music_lm_window_mode = "random"
        self.args.nextlat_enabled = True
        self.args.nextlat_block_index = -1
        self.args.nextlat_weight = 0.1
        self.args.nextlat_state_loss = "smooth_l1"
        self.args.nextlat_kl_weight = 0.0
        self.args.xm_enabled = True
        self.args.xm_candidate_count = 2
        self.args.xm_selection_scope = "block"
        self.args.xm_training_target = "route"
        self.args.xm_block_size = 16
        model = MagicMock(
            MODEL_LICENSE="other",
            PREDICTION_TYPE=SimpleNamespace(value="autoregressive_next_token"),
            gligen=False,
        )
        model.validation_audio_sample_rate.return_value = 44100
        model.custom_model_card_schedule_info.return_value = ""
        model.custom_model_card_code_example.return_value = "```python\npass\n```"
        model.custom_model_card_training_mode_info.return_value = (
            "- MiniMax Music train component: `language_model (global LM / RVQ planner)`\n"
            "- MiniMax Music LM max frames: `128`\n"
            "- MiniMax Music LM window mode: `random`"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("simpletuner.helpers.publishing.metadata.StateTracker.get_model_family", return_value="minimaxmusic"),
                patch("simpletuner.helpers.publishing.metadata.StateTracker.get_data_backends", return_value={}),
                patch("simpletuner.helpers.publishing.metadata.StateTracker.get_weight_dtype", return_value=torch.bfloat16),
                patch(
                    "simpletuner.helpers.publishing.metadata.StateTracker.get_accelerator",
                    return_value=MagicMock(num_processes=1),
                ),
                patch("simpletuner.helpers.publishing.metadata.StateTracker.get_args", return_value=self.args),
                patch("simpletuner.helpers.publishing.metadata.StateTracker.get_model", return_value=model),
            ):
                save_model_card(
                    repo_id="test-repo",
                    images=None,
                    audios=None,
                    base_model="MiniMaxAI/MiniMax-Music3",
                    train_text_encoder=False,
                    prompt="",
                    validation_prompts=None,
                    validation_shortnames=None,
                    repo_folder=tmpdir,
                    model=model,
                    global_step=6000,
                    epoch=250,
                )

            readme = Path(tmpdir, "README.md").read_text(encoding="utf-8")
            self.assertIn("Validation was disabled during training.", readme)
            self.assertNotIn("## Validation settings", readme)
            self.assertNotIn("<Gallery />", readme)
            self.assertIn("## Training modes", readme)
            self.assertIn("- MiniMax Music train component: `language_model (global LM / RVQ planner)`", readme)
            self.assertIn("- MiniMax Music LM max frames: `128`", readme)
            self.assertIn("- MiniMax Music LM window mode: `random`", readme)
            self.assertIn("- NextLat: Enabled", readme)
            self.assertIn("  - Weight: `0.1`", readme)
            self.assertIn("- XM: Enabled", readme)
            self.assertIn("  - Candidate count: `2`", readme)
            self.assertIn("  - Training target: `route`", readme)
            model.custom_model_card_training_mode_info.assert_called_once_with(self.args)

    def test_minimax_music_model_card_training_mode_info(self):
        from simpletuner.helpers.models.minimaxmusic.model import MiniMaxMusic

        self.args.minimax_music_train_component = "language_model"
        self.args.minimax_music_lm_max_frames = 128
        self.args.minimax_music_lm_window_mode = "random"
        model = MiniMaxMusic.__new__(MiniMaxMusic)

        details = model.custom_model_card_training_mode_info(self.args)

        self.assertIn("- MiniMax Music train component: `language_model (global LM / RVQ planner)`", details)
        self.assertIn("- MiniMax Music LM max frames: `128`", details)
        self.assertIn("- MiniMax Music LM window mode: `random`", details)

    def test_hub_commit_message_omits_diffusion_schedule_fields_for_flow_matching(self):
        hub_manager = object.__new__(HubManager)
        hub_manager.collected_data_backend_str = "['alt-embed-cache', 'h3-drift0-anyflow-openvid-39f-480']"
        hub_manager.model = SimpleNamespace(PREDICTION_TYPE=SimpleNamespace(value="flow_matching"))
        hub_manager.config = SimpleNamespace(
            learning_rate=6e-5,
            train_batch_size=1,
            gradient_accumulation_steps=1,
            prediction_type=None,
            rescale_betas_zero_snr=False,
            training_scheduler_timestep_spacing="trailing",
            distillation_method="h3_drift",
            distillation_config={
                "h3_drift": {
                    "loss_weight": 0.5,
                    "inner_distillation_method": "anyflow",
                    "inner_distillation_config": {"stage": "forward"},
                }
            },
            pretrained_model_name_or_path="MiniMaxAI/MiniMax-H3",
            pretrained_vae_model_name_or_path=(
                "https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/"
                "minimax_h3_video_vae_int8_convrot.safetensors"
            ),
            model_type="lora",
        )

        message = hub_manager._commit_message(global_step=1000, epoch=2)

        self.assertIn("Learning rate 6e-05 and batch size 1.", message)
        self.assertIn("Training objective: flow matching.", message)
        self.assertIn("Distillation method: h3_drift.", message)
        self.assertIn('"inner_distillation_method": "anyflow"', message)
        self.assertIn('"stage": "forward"', message)
        self.assertNotIn("gradient accumulation", message)
        self.assertNotIn("None prediction type", message)
        self.assertNotIn("rescale_betas_zero_snr", message)
        self.assertNotIn("timestep spacing", message)

    def test_hub_commit_message_keeps_diffusion_schedule_fields_for_epsilon_models(self):
        hub_manager = object.__new__(HubManager)
        hub_manager.collected_data_backend_str = "['dataset']"
        hub_manager.model = SimpleNamespace(PREDICTION_TYPE=SimpleNamespace(value="epsilon"))
        hub_manager.config = SimpleNamespace(
            learning_rate=1e-4,
            train_batch_size=2,
            gradient_accumulation_steps=4,
            prediction_type="epsilon",
            rescale_betas_zero_snr=True,
            training_scheduler_timestep_spacing="trailing",
            distillation_method=None,
            distillation_config=None,
            pretrained_model_name_or_path="base-model",
            pretrained_vae_model_name_or_path="base-vae",
            model_type="lora",
        )

        message = hub_manager._commit_message(global_step=20, epoch=2)

        self.assertIn(
            "Learning rate 0.0001, batch size 2, and 4 gradient accumulation steps.",
            message,
        )
        self.assertIn(
            "Trained with epsilon prediction type and rescale_betas_zero_snr=True",
            message,
        )
        self.assertIn("Using 'trailing' timestep spacing.", message)
        self.assertNotIn("Distillation method:", message)

    def test_save_model_card(self):
        # Mocking StateTracker methods
        self.args.model_family = "flux"
        self.args.model_type = "lora"
        self.args.lora_type = "lycoris"
        self.args.base_model_precision = "int8-quanto"
        with patch(
            "simpletuner.helpers.publishing.metadata.StateTracker.get_model_family",
            return_value="sdxl",
        ):
            with patch(
                "simpletuner.helpers.publishing.metadata.StateTracker.get_data_backends",
                return_value={},
            ):
                with patch(
                    "simpletuner.helpers.publishing.metadata.StateTracker.get_weight_dtype",
                    return_value=torch.bfloat16,
                ):
                    with patch(
                        "simpletuner.helpers.publishing.metadata.StateTracker.get_accelerator",
                        return_value=MagicMock(num_processes=1),
                    ):
                        with patch(
                            "simpletuner.helpers.training.state_tracker.StateTracker.get_args",
                            return_value=self.args,
                        ):
                            with patch("builtins.open", unittest.mock.mock_open()) as mock_file:
                                model = MagicMock(
                                    MODEL_LICENSE="other",
                                    MODEL_LICENSE_NAME="minimax-h3-community-license-agreement",
                                    MODEL_LICENSE_LINK="https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE",
                                )
                                model.custom_model_card_schedule_info.return_value = ""
                                save_model_card(
                                    repo_id="test-repo",
                                    images=None,
                                    base_model="test-base-model",
                                    train_text_encoder=True,
                                    prompt="Test prompt",
                                    validation_prompts=["Test prompt"],
                                    validation_shortnames=["shortname"],
                                    repo_folder="test-folder",
                                    model=model,
                                    global_step=1000,
                                    epoch=1,
                                )
                                # Ensure the README.md was written
                                mock_file.assert_called_with(
                                    os.path.join("test-folder", "README.md"),
                                    "w",
                                    encoding="utf-8",
                                )
                                written = mock_file().write.call_args.args[0]
                                self.assertIn("license: other\n", written)
                                self.assertIn('license_name: "minimax-h3-community-license-agreement"\n', written)
                                self.assertNotIn(" license_name:", written)
                                self.assertNotIn(" license_link:", written)

    def test_upload_full_model_uses_empty_repo_path_for_top_level_uploads(self):
        hub_manager = object.__new__(HubManager)
        hub_manager.config = SimpleNamespace(push_to_hub=True, output_dir="output")
        hub_manager._repo_id = "owner/model"
        hub_manager.hub_token = "token"
        hub_manager._hub_api = MagicMock()
        hub_manager._commit_message = MagicMock(return_value="message")

        hub_manager.upload_full_model()

        hub_manager._hub_api.upload_folder.assert_called_once()
        self.assertEqual(hub_manager._hub_api.upload_folder.call_args.kwargs["path_in_repo"], "")

    def test_save_model_card_honors_validation_image_format_for_assets(self):
        self.args.model_family = "sdxl"
        self.args.model_type = "lora"
        self.args.lora_type = "standard"
        self.args.validation_image_format = "jpeg"
        self.args.validation_image_quality = 81
        self.args.controlnet = False
        self.args.control = False
        model = MagicMock(
            MODEL_LICENSE="other",
            PREDICTION_TYPE=SimpleNamespace(value="epsilon"),
            MODEL_TYPE=SimpleNamespace(value="unet"),
            gligen=False,
        )
        model.validation_audio_sample_rate.return_value = 44100
        model.custom_model_card_schedule_info.return_value = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("simpletuner.helpers.publishing.metadata.StateTracker.get_model_family", return_value="sdxl"),
                patch("simpletuner.helpers.publishing.metadata.StateTracker.get_data_backends", return_value={}),
                patch("simpletuner.helpers.publishing.metadata.StateTracker.get_weight_dtype", return_value=torch.bfloat16),
                patch(
                    "simpletuner.helpers.publishing.metadata.StateTracker.get_accelerator",
                    return_value=MagicMock(num_processes=1),
                ),
                patch("simpletuner.helpers.publishing.metadata.StateTracker.get_args", return_value=self.args),
                patch("simpletuner.helpers.publishing.metadata.StateTracker.get_model", return_value=model),
            ):
                save_model_card(
                    repo_id="test-repo",
                    images={"prompt": [Image.new("RGBA", (8, 8), color=(255, 0, 0, 128))]},
                    base_model="test-base-model",
                    train_text_encoder=False,
                    prompt="Test prompt",
                    validation_prompts=["Test prompt"],
                    validation_shortnames=["prompt"],
                    repo_folder=tmpdir,
                    model=model,
                    global_step=1000,
                    epoch=1,
                )

            readme = Path(tmpdir, "README.md").read_text(encoding="utf-8")
            self.assertIn("url: ./assets/image_0_0.jpg", readme)
            self.assertTrue(Path(tmpdir, "assets", "image_0_0.jpg").exists())
            self.assertFalse(Path(tmpdir, "assets", "image_0_0.png").exists())

    def test_checkpoint_validation_media_filter_matches_shortnames_exactly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            validation_dir = output_dir / "validation_images"
            validation_dir.mkdir()
            (validation_dir / "step_50_prompt_0_64x64.jpg").write_bytes(b"jpg")
            (validation_dir / "step_50_prompt_adapter_0_64x64.jpg").write_bytes(b"adapter")
            (validation_dir / "step_49_prompt_0_64x64.jpg").write_bytes(b"old")
            (validation_dir / "step_50_prompt_0.wav").write_bytes(b"wav")

            hub_manager = object.__new__(HubManager)
            hub_manager.config = SimpleNamespace(output_dir=str(output_dir))

            images, audios = hub_manager._filter_checkpoint_validation_media(
                50,
                {"prompt": [], "prompt_adapter": []},
                {"prompt": []},
            )

            self.assertEqual(
                {key: [Path(path).name for path in paths] for key, paths in images.items()},
                {
                    "prompt": ["step_50_prompt_0_64x64.jpg"],
                    "prompt_adapter": ["step_50_prompt_adapter_0_64x64.jpg"],
                },
            )
            self.assertEqual(
                {key: [Path(path).name for path in paths] for key, paths in audios.items()},
                {"prompt": ["step_50_prompt_0.wav"]},
            )

    def test_save_training_config_sanitizes_public_export(self):
        config = SimpleNamespace(
            output_dir="output/test",
            sageattention_usage=AttentionBackendMode.INFERENCE,
            publishing_config=[
                {
                    "provider": "s3",
                    "bucket": "training",
                    "access_key": "dummy-access-key",
                    "secret_key": "dummy-secret-key",
                }
            ],
            webhook_config={"url": "https://example.invalid/webhook", "auth_token": "dummy-token"},
            nested={
                "safe": "keep",
                "tokenizer_max_length": 77,
                "access_key": "nested-access-key",
                "aws_secret_access_key": "nested-secret-key",
                "token": "nested-token",
            },
            dtype=torch.bfloat16,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            save_training_config(repo_folder=temp_dir, config=config)
            with open(os.path.join(temp_dir, "simpletuner_config.json"), "r", encoding="utf-8") as handle:
                payload = json.load(handle)

        self.assertEqual(payload["sageattention_usage"], "inference")
        self.assertNotIsInstance(payload["sageattention_usage"], dict)
        self.assertNotIn("publishing_config", payload)
        self.assertNotIn("webhook_config", payload)
        self.assertEqual(payload["nested"], {"safe": "keep", "tokenizer_max_length": 77})
        self.assertEqual(payload["dtype"], "torch.bfloat16")

    def test_adapter_download_fn(self):
        with patch("huggingface_hub.hf_hub_download", return_value="path/to/adapter"):
            from simpletuner.helpers.publishing.metadata import lycoris_download_info

            output = lycoris_download_info()
            self.assertIn("hf_hub_download", output)

    def test_pipeline_move_full_bf16(self):
        from simpletuner.helpers.publishing.metadata import _pipeline_move_to

        with patch(
            "simpletuner.helpers.training.state_tracker.StateTracker.get_weight_dtype",
            return_value=torch.bfloat16,
        ):
            output = _pipeline_move_to(args=self.args)

        self.assertNotIn("torch.bfloat16", output)

    def test_pipeline_move_lycoris_bf16(self):
        from simpletuner.helpers.publishing.metadata import _pipeline_move_to

        with patch(
            "simpletuner.helpers.training.state_tracker.StateTracker.get_weight_dtype",
            return_value=torch.bfloat16,
        ):
            self.args.model_type = "lora"
            self.args.lora_type = "lycoris"
            self.args.base_model_precision = "no_change"
            output = _pipeline_move_to(args=self.args)
        self.assertNotIn("torch.bfloat16", output)

    def test_pipeline_move_lycoris_int8(self):
        from simpletuner.helpers.publishing.metadata import _pipeline_move_to

        with patch(
            "simpletuner.helpers.training.state_tracker.StateTracker.get_weight_dtype",
            return_value=torch.bfloat16,
        ):
            self.args.model_type = "lora"
            self.args.lora_type = "lycoris"
            self.args.base_model_precision = "int8-quanto"
            output = _pipeline_move_to(args=self.args)
        self.assertNotIn("torch.bfloat16", output)

    def test_pipeline_quanto_hint_unet(self):
        from simpletuner.helpers.publishing.metadata import _pipeline_quanto

        self.mock_model.MODEL_TYPE = MagicMock(value="unet")
        output = _pipeline_quanto(args=self.args, model=self.mock_model)

        self.assertIn("quantize", output)
        self.assertIn("optimum.quanto", output)
        self.assertIn("pipeline.unet", output)

    def test_pipeline_quanto_hint_transformer(self):
        from simpletuner.helpers.publishing.metadata import _pipeline_quanto

        self.args.model_family = "flux"
        self.mock_model.MODEL_TYPE = MagicMock(value="transformer")
        output = _pipeline_quanto(args=self.args, model=self.mock_model)
        self.assertIn("quantize", output)
        self.assertIn("optimum.quanto", output)
        self.assertIn("pipeline.transformer", output)


class TestGligenModelCard(unittest.TestCase):
    def setUp(self):
        self.args = MagicMock()
        self.args.lora_type = "standard"
        self.args.model_type = "lora"
        self.args.model_family = "sdxl"
        self.args.controlnet = False
        self.args.control = False
        self.args.peft_lora_mode = "standard"

    def test_model_type_gligen_prefix(self):
        mock_model = MagicMock()
        mock_model.gligen = True
        mock_model.MODEL_TYPE = MagicMock(value="unet")
        with patch(
            "simpletuner.helpers.publishing.metadata.StateTracker.get_model",
            return_value=mock_model,
        ):
            result = model_type(self.args)
            self.assertEqual(result, "GLIGEN PEFT LoRA")

    def test_model_type_no_gligen_prefix(self):
        mock_model = MagicMock()
        mock_model.gligen = False
        mock_model.MODEL_TYPE = MagicMock(value="unet")
        with patch(
            "simpletuner.helpers.publishing.metadata.StateTracker.get_model",
            return_value=mock_model,
        ):
            result = model_type(self.args)
            self.assertEqual(result, "PEFT LoRA")

    def test_model_type_controlnet_and_gligen(self):
        self.args.controlnet = True
        mock_model = MagicMock()
        mock_model.gligen = True
        mock_model.MODEL_TYPE = MagicMock(value="unet")
        with patch(
            "simpletuner.helpers.publishing.metadata.StateTracker.get_model",
            return_value=mock_model,
        ):
            result = model_type(self.args)
            self.assertTrue(result.startswith("ControlNet GLIGEN "))

    def test_model_imports_gligen(self):
        mock_model = MagicMock()
        mock_model.gligen = True
        with patch(
            "simpletuner.helpers.publishing.metadata.StateTracker.get_model",
            return_value=mock_model,
        ):
            result = _model_imports(self.args)
            self.assertIn("inject_gligen_layers", result)

    def test_model_imports_no_gligen(self):
        mock_model = MagicMock()
        mock_model.gligen = False
        with patch(
            "simpletuner.helpers.publishing.metadata.StateTracker.get_model",
            return_value=mock_model,
        ):
            result = _model_imports(self.args)
            self.assertNotIn("inject_gligen_layers", result)

    def test_gligen_injection_code_in_model_load(self):
        from simpletuner.helpers.publishing.metadata import _gligen_injection_code

        mock_model = MagicMock()
        mock_model.gligen = True
        mock_model.MODEL_TYPE = MagicMock(value="unet")
        mock_component = MagicMock()
        mock_component.config.cross_attention_dim = 768
        mock_model.get_trained_component.return_value = mock_component
        with (
            patch(
                "simpletuner.helpers.publishing.metadata.StateTracker.get_model",
                return_value=mock_model,
            ),
            patch(
                "simpletuner.helpers.publishing.metadata.StateTracker.get_args",
                return_value=MagicMock(pretrained_grounding_model_name_or_path=None),
            ),
        ):
            result = _gligen_injection_code(mock_model)
            self.assertIn("inject_gligen_layers", result)
            self.assertIn("positive_len=768", result)
            self.assertIn('feature_type="text-only"', result)

    def test_gligen_injection_code_text_image(self):
        from simpletuner.helpers.publishing.metadata import _gligen_injection_code

        mock_model = MagicMock()
        mock_model.gligen = True
        mock_model.MODEL_TYPE = MagicMock(value="unet")
        mock_component = MagicMock()
        mock_component.config.cross_attention_dim = 768
        mock_model.get_trained_component.return_value = mock_component
        with (
            patch(
                "simpletuner.helpers.publishing.metadata.StateTracker.get_model",
                return_value=mock_model,
            ),
            patch(
                "simpletuner.helpers.publishing.metadata.StateTracker.get_args",
                return_value=MagicMock(pretrained_grounding_model_name_or_path="/some/path"),
            ),
        ):
            result = _gligen_injection_code(mock_model)
            self.assertIn('feature_type="text-image"', result)

    def test_gligen_injection_code_no_gligen(self):
        from simpletuner.helpers.publishing.metadata import _gligen_injection_code

        mock_model = MagicMock()
        mock_model.gligen = False
        mock_model.MODEL_TYPE = MagicMock(value="unet")
        with patch(
            "simpletuner.helpers.publishing.metadata.StateTracker.get_model",
            return_value=mock_model,
        ):
            result = _gligen_injection_code(mock_model)
            self.assertEqual(result, "")

    def test_model_load_includes_gligen_injection(self):
        self.args.pretrained_model_name_or_path = "pretrained-model"
        self.args.output_dir = "output-dir"
        mock_model = MagicMock()
        mock_model.gligen = True
        mock_model.MODEL_TYPE = MagicMock(value="unet")
        mock_component = MagicMock()
        mock_component.config.cross_attention_dim = 768
        mock_model.get_trained_component.return_value = mock_component
        with (
            patch(
                "simpletuner.helpers.publishing.metadata.StateTracker.get_model",
                return_value=mock_model,
            ),
            patch(
                "simpletuner.helpers.publishing.metadata.StateTracker.get_hf_username",
                return_value="testuser",
            ),
            patch(
                "simpletuner.helpers.publishing.metadata.StateTracker.get_weight_dtype",
                return_value="torch.bfloat16",
            ),
            patch(
                "simpletuner.helpers.publishing.metadata.StateTracker.get_args",
                return_value=MagicMock(pretrained_grounding_model_name_or_path=None),
            ),
        ):
            output = _model_load(self.args, repo_id="repo-id", model=mock_model)
            self.assertIn("inject_gligen_layers", output)
            inject_pos = output.index("inject_gligen_layers")
            load_pos = output.index("pipeline.load_lora_weights")
            self.assertLess(inject_pos, load_pos)


if __name__ == "__main__":
    unittest.main()
