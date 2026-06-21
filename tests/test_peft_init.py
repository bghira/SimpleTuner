import unittest
from types import SimpleNamespace

import torch

from simpletuner.helpers.training.peft_init import init_lokr_network_with_perturbed_normal

try:
    from torchao.prototype.quantized_training.int8 import Int8QuantizedTrainingLinearWeight
except Exception:  # pragma: no cover - optional dependency
    Int8QuantizedTrainingLinearWeight = None


@unittest.skipIf(Int8QuantizedTrainingLinearWeight is None, "torchao int8 quantized training is not available")
class LoKrInitTorchAOTests(unittest.TestCase):
    def test_init_lokr_with_torchao_int8_weight(self):
        org_weight = Int8QuantizedTrainingLinearWeight.from_float(torch.randn(8, 8))
        lokr_module = SimpleNamespace(
            org_weight=org_weight,
            lokr_w1=torch.zeros(8, 8),
            lokr_w2=torch.zeros(8, 8),
        )
        network = SimpleNamespace(loras=[lokr_module])

        init_lokr_network_with_perturbed_normal(network, scale=1e-3)

        self.assertTrue(torch.allclose(lokr_module.lokr_w1, torch.ones_like(lokr_module.lokr_w1)))
        self.assertGreater(lokr_module.lokr_w2.abs().sum().item(), 0.0)

    def test_peft_torchao_dispatcher_receives_requantize_config(self):
        import peft.tuners.lora.model as peft_lora_model
        import peft.tuners.lora.torchao as peft_lora_torchao
        from peft import LoraConfig
        from peft.tuners.lora.torchao import TorchaoLoraLinear

        import simpletuner.helpers.training.quantisation.torchao_workarounds  # noqa: F401

        linear = torch.nn.Linear(8, 8, bias=False)
        linear.weight = torch.nn.Parameter(Int8QuantizedTrainingLinearWeight.from_float(linear.weight.detach()))
        config = LoraConfig(r=2, lora_alpha=2, target_modules=["unused"])

        new_module = peft_lora_model.dispatch_torchao(
            linear,
            "default",
            config=config,
            r=config.r,
            lora_alpha=config.lora_alpha,
        )

        self.assertIs(peft_lora_model.dispatch_torchao, peft_lora_torchao.dispatch_torchao)
        self.assertIsInstance(new_module, TorchaoLoraLinear)
        self.assertTrue(callable(new_module.get_apply_tensor_subclass))

    @unittest.skipUnless(torch.cuda.is_available(), "TorchAO dynamic quantization dispatcher test requires CUDA")
    def test_peft_torchao_dispatcher_supports_dynamic_quantized_weights(self):
        import peft.tuners.lora.model as peft_lora_model
        from peft import LoraConfig
        from peft.tuners.lora.torchao import TorchaoLoraLinear
        from torchao.quantization import quantize_
        from torchao.quantization.quant_api import (
            Float8DynamicActivationFloat8WeightConfig,
            Float8WeightOnlyConfig,
            Int8DynamicActivationInt8WeightConfig,
            Int8DynamicActivationIntxWeightConfig,
        )

        import simpletuner.helpers.training.quantisation.torchao_workarounds  # noqa: F401
        from simpletuner.helpers.training.state_tracker import StateTracker

        cases = (
            ("int8dq-torchao", Int8DynamicActivationInt8WeightConfig()),
            ("int8dq-int4-torchao", Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4)),
            ("fp8-torchao", Float8DynamicActivationFloat8WeightConfig()),
            ("fp8wo-torchao", Float8WeightOnlyConfig()),
        )
        previous_args = StateTracker.get_args()
        config = LoraConfig(r=2, lora_alpha=2, target_modules=["unused"])
        try:
            for precision, torchao_config in cases:
                with self.subTest(precision=precision):
                    StateTracker.set_args(SimpleNamespace(base_model_precision=precision))
                    linear = torch.nn.Linear(32, 32, bias=False, device="cuda", dtype=torch.bfloat16)
                    quantize_(linear, torchao_config)

                    new_module = peft_lora_model.dispatch_torchao(
                        linear,
                        "default",
                        config=config,
                        r=config.r,
                        lora_alpha=config.lora_alpha,
                    )

                    self.assertIsInstance(new_module, TorchaoLoraLinear)
                    self.assertTrue(callable(new_module.get_apply_tensor_subclass))
        finally:
            StateTracker.set_args(previous_args)
