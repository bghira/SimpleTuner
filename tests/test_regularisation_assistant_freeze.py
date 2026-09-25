import unittest
from types import SimpleNamespace

import torch
from peft import LoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer

from simpletuner.helpers.assistant_lora import set_adapter_stack
from simpletuner.helpers.models.qwen_image.transformer_21 import QwenImage21Transformer2DModel
from simpletuner.helpers.training.trainer import Trainer


class AssistantRegularisationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.component = QwenImage21Transformer2DModel(
            in_channels=4,
            out_channels=4,
            num_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            context_in_dim=8,
            mlp_ratio=2,
            axes_dims_rope=(4, 6, 6),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.component.to(self.device)
        self.component.requires_grad_(False)
        for name in ("default", "assistant"):
            self.component.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q"]), adapter_name=name)
        with torch.no_grad():
            for name, parameter in self.component.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(0, 0.3)
        self.inputs = dict(
            hidden_states=torch.randn(1, 4, 4),
            encoder_hidden_states=torch.randn(1, 3, 8),
            encoder_hidden_states_mask=torch.ones(1, 3),
            timestep=torch.tensor([0.3]),
            img_shapes=[[(1, 2, 2)]],
            img_mask=torch.zeros(1, 5, dtype=torch.bool),
            return_dict=False,
        )
        self.inputs = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in self.inputs.items()}
        self.trainer = object.__new__(Trainer)
        self.trainer.config = SimpleNamespace(lora_type="standard", assistant_lora_strength=0.7)
        self.trainer.model = SimpleNamespace(
            assistant_lora_loaded=True,
            assistant_adapter_name="assistant",
            get_trained_component=lambda: self.component,
            configure_assistant_lora_for_training=self.restore,
        )
        self.trainer.model_predict = lambda prepared_batch: {"model_prediction": self.component(**self.inputs)[0]}
        self.restore()

    def restore(self):
        set_adapter_stack(self.component, ["assistant", "default"], weights=[0.7, 1.0], freeze_names=["assistant"])

    def assert_restored(self):
        self.assertEqual(self.component.active_adapters(), ["assistant", "default"])
        for name, parameter in self.component.named_parameters():
            if ".assistant." in name:
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
            if ".default." in name:
                self.assertTrue(parameter.requires_grad)

    def test_target_is_bare_base_and_student_updates_only_concept(self):
        set_adapter_stack(self.component, ["assistant"], weights=[0.7], freeze_names=["assistant"])
        with torch.no_grad():
            assisted = self.component(**self.inputs)[0]
        self.component.disable_lora()
        with torch.no_grad():
            expected = self.component(**self.inputs)[0]
        self.component.enable_lora()
        self.restore()
        self.assertFalse(torch.allclose(assisted, expected))
        frozen = {n: p.detach().clone() for n, p in self.component.named_parameters() if ".assistant." in n}
        batch = {}
        self.trainer._prepare_regularisation_parent_targets(batch)
        torch.testing.assert_close(batch["target"], expected)
        self.assertFalse(batch["target"].requires_grad)
        self.assert_restored()
        student = self.component(**self.inputs)[0]
        self.assertFalse(torch.allclose(student, expected))
        optimizer = torch.optim.SGD([p for p in self.component.parameters() if p.requires_grad], lr=0.1)
        (student - batch["target"]).square().mean().backward()
        self.assert_restored()
        optimizer.step()
        for name, parameter in self.component.named_parameters():
            if name in frozen:
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
        new_batch = {}
        self.trainer._prepare_regularisation_parent_targets(new_batch)
        torch.testing.assert_close(new_batch["target"], expected)

    def test_parent_failure_restores_training_stack(self):
        def fail(prepared_batch):
            self.assertTrue(
                all(layer.disable_adapters for layer in self.component.modules() if isinstance(layer, BaseTunerLayer))
            )
            raise RuntimeError("parent failed")

        self.trainer.model_predict = fail
        with self.assertRaisesRegex(RuntimeError, "parent failed"):
            self.trainer._prepare_regularisation_parent_targets({})
        self.assert_restored()
