import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import load_file, save_file

from simpletuner.helpers.configuration.cmd_args import parse_cmdline_args
from simpletuner.helpers.models.moss_image.model import MOSSImage
from simpletuner.helpers.training.save_hooks import LORA_SAFETENSORS_FILENAME, SaveHookManager
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.training.trainer import Trainer
from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
from simpletuner.simpletuner_sdk.server.services.validation_service import ValidationService
from tests.test_moss_image import tiny_transformer


class AdapterModelPrecisionTests(unittest.TestCase):
    def setUp(self):
        previous_args = StateTracker.get_args()
        self.addCleanup(StateTracker.set_args, previous_args)

    def parse(self, *extra):
        return parse_cmdline_args(
            input_args=[
                "--model_family=pixart",
                "--output_dir=/tmp/adapter-precision-output",
                "--model_type=lora",
                "--optimizer=torch-adamw",
                "--data_backend_config=/tmp/adapter-precision-data.json",
                *extra,
            ],
            exit_on_error=True,
        )

    def trainer(self, precision="fp32"):
        module = torch.nn.Module()
        module.base = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.bfloat16), requires_grad=False)
        module.adapter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
        module.delta_timestep_embedder = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
        trainer = object.__new__(Trainer)
        trainer.config = SimpleNamespace(
            adapter_model_precision=precision,
            model_type="lora",
            lora_type="standard",
            train_text_encoder=False,
            weight_dtype=torch.bfloat16,
            use_fsdp=False,
            pipeline_quantization_base=False,
            musubi_blocks_to_swap=0,
            ramtorch=False,
            is_quantized=False,
            attention_mechanism="diffusers",
            controlnet=False,
        )
        trainer.model = SimpleNamespace(
            get_trained_component=lambda **kwargs: module,
            MODEL_TYPE=SimpleNamespace(value="transformer"),
            text_encoders=[],
        )
        trainer.accelerator = SimpleNamespace(device="cpu", state=SimpleNamespace(fsdp_plugin=None))
        trainer._report_cuda_usage_if_requested = Mock()
        return trainer, module

    def test_cli_default_and_explicit_fp32(self):
        self.assertEqual(self.parse().adapter_model_precision, "no_change")
        self.assertEqual(self.parse("--adapter_model_precision=fp32").adapter_model_precision, "fp32")
        self.assertEqual(self.parse("--optimizer=adamw_bf16").adapter_model_precision, "no_change")

    def test_cli_rejects_quantized_adapter_precision(self):
        with self.assertRaisesRegex(ValueError, "invalid choice"):
            self.parse("--adapter_model_precision=int8-quanto")

    def test_cli_rejects_incompatible_training_modes(self):
        for extra, message in [
            (["--model_type=full"], "requires model_type=lora"),
            (["--optimizer=adamw_bf16"], "incompatible with adamw_bf16"),
        ]:
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                self.parse("--adapter_model_precision=fp32", *extra)

    def test_no_change_preserves_parameters_and_gradients(self):
        trainer, module = self.trainer("no_change")
        module.adapter.grad = torch.ones_like(module.adapter)
        trainer._apply_adapter_model_precision()
        self.assertEqual(module.adapter.dtype, torch.bfloat16)
        self.assertEqual(module.adapter.grad.dtype, torch.bfloat16)
        self.assertEqual(module.base.dtype, torch.bfloat16)

    def test_fp32_upcasts_trainable_parameters_and_gradients_only(self):
        trainer, module = self.trainer()
        parameter_id = id(module.adapter)
        module.adapter.grad = torch.ones_like(module.adapter)
        trainer._apply_adapter_model_precision()
        self.assertEqual(id(module.adapter), parameter_id)
        self.assertEqual(module.adapter.dtype, torch.float32)
        self.assertEqual(module.adapter.grad.dtype, torch.float32)
        self.assertEqual(module.delta_timestep_embedder.weight.dtype, torch.float32)
        self.assertEqual(module.base.dtype, torch.bfloat16)

    def test_grouped_optimizer_parameters_leave_frozen_weights_unchanged(self):
        trainer, module = self.trainer()
        trainer._apply_adapter_model_precision([{"params": [module.base, module.adapter]}])
        self.assertEqual(module.adapter.dtype, torch.float32)
        self.assertEqual(module.base.dtype, torch.bfloat16)

    def test_trainable_text_encoder_parameters_are_upcast(self):
        trainer, module = self.trainer()
        encoder = torch.nn.Linear(1, 1, dtype=torch.bfloat16)
        encoder.bias.requires_grad_(False)
        trainer.model.text_encoders = [encoder]
        trainer.config.train_text_encoder = True
        trainer._apply_adapter_model_precision()
        self.assertEqual(encoder.weight.dtype, torch.float32)
        self.assertEqual(encoder.bias.dtype, torch.bfloat16)

    def test_model_moves_preserve_small_fp32_updates(self):
        trainer, module = self.trainer()
        trainer._apply_adapter_model_precision()
        with torch.no_grad():
            module.adapter.fill_(1.0001)
        expected = module.adapter.detach().clone()
        with patch("simpletuner.helpers.training.trainer.AttentionBackendController.apply"):
            trainer.move_models("cpu")
            trainer.move_models("accelerator")
        torch.testing.assert_close(module.adapter, expected, rtol=0, atol=0)
        self.assertEqual(module.base.dtype, torch.bfloat16)

    def test_fp32_adamw_retains_updates_smaller_than_bf16_resolution(self):
        trainer, module = self.trainer()
        trainer._apply_adapter_model_precision()
        optimizer = torch.optim.AdamW([module.adapter], lr=1e-5, weight_decay=0)
        before = module.adapter.detach().clone()
        module.adapter.sum().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, module.adapter))
        self.assertTrue(torch.equal(before.bfloat16(), module.adapter.bfloat16()))
        self.assertEqual(optimizer.state[module.adapter]["exp_avg"].dtype, torch.float32)

    def test_init_lora_preserves_saved_fp32_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            expected = {
                "transformer.layers.0.proj.q.lora_A.weight": torch.full((2, 32), 1.0001),
                "transformer.layers.0.proj.q.lora_B.weight": torch.full((32, 2), 0.00010001),
            }
            save_file(expected, path)
            foundation = object.__new__(MOSSImage)
            foundation.accelerator = SimpleNamespace(unwrap_model=lambda model, **kwargs: model)
            foundation.config = SimpleNamespace(
                adapter_model_precision="fp32",
                init_lora=str(path),
                controlnet=False,
                lora_rank=2,
                lora_alpha=2,
                lora_dropout=0,
                lora_initialisation_style=True,
                lora_type="standard",
                peft_lora_target_modules=["layers.0.proj.q"],
                ramtorch=False,
            )
            foundation.model = tiny_transformer().to(dtype=torch.bfloat16).requires_grad_(False)
            foundation.add_lora_adapter()
            layer = foundation.model.layers[0].proj["q"]
            torch.testing.assert_close(
                layer.lora_A["default"].weight, expected["transformer.layers.0.proj.q.lora_A.weight"], rtol=0, atol=0
            )
            torch.testing.assert_close(
                layer.lora_B["default"].weight, expected["transformer.layers.0.proj.q.lora_B.weight"], rtol=0, atol=0
            )
            self.assertEqual(layer.base_layer.weight.dtype, torch.bfloat16)

    def test_webui_rejects_incompatible_fp32_optimizer(self):
        for prefix in ["", "--"]:
            with self.subTest(prefix=prefix):
                config = {
                    prefix + key: value
                    for key, value in {
                        "model_type": "lora",
                        "lora_rank": 2,
                        "adapter_model_precision": "fp32",
                        "optimizer": "adamw_bf16",
                    }.items()
                }
                result = ValidationService().validate_configuration(config, validate_paths=False, estimate_vram=False)
                self.assertTrue(any(message.field == "adapter_model_precision" for message in result.messages))
                with tempfile.TemporaryDirectory() as directory:
                    store = ConfigStore(config_dir=Path(directory), config_type="model")
                    result = store.validate_config(config)
                self.assertTrue(any("adapter_model_precision" in error for error in result.errors))

    def test_lycoris_checkpoint_loading_preserves_fp32_values(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = torch.full((1, 1), 1.0001)
            save_file({"weight": expected}, Path(directory) / LORA_SAFETENSORS_FILENAME)
            network = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)

            def load_weights(path):
                network.load_state_dict(load_file(path), strict=True)
                return {}

            network.load_weights = load_weights
            hooks = object.__new__(SaveHookManager)
            hooks.args = SimpleNamespace(adapter_model_precision="fp32")
            hooks.accelerator = SimpleNamespace(_lycoris_wrapped_network=network, device="cpu")
            hooks.model = SimpleNamespace(get_trained_component=lambda: network)
            hooks._load_lycoris([network], directory)
            self.assertEqual(network.weight.dtype, torch.float32)
            torch.testing.assert_close(network.weight, expected, rtol=0, atol=0)
