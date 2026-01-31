"""
Tests for gradient checkpointing backend selection (torch vs unsloth).
"""

import unittest

import torch
import torch.nn as nn


class SimpleModule(nn.Module):
    """Simple module for testing checkpointing."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class TestOffloadedGradientCheckpointer(unittest.TestCase):
    """Tests for the offloaded gradient checkpointer."""

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dim = 64
        self.batch_size = 4

    def test_offloaded_checkpoint_forward_pass(self):
        """Test that offloaded checkpoint produces correct forward output."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import offloaded_checkpoint

        module = SimpleModule(self.dim).to(self.device)
        x = torch.randn(self.batch_size, self.dim, device=self.device)

        # Direct forward
        expected = module(x)

        # Checkpointed forward
        result = offloaded_checkpoint(module, x, use_reentrant=False)

        self.assertTrue(torch.allclose(expected, result, atol=1e-6))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required for offload test")
    def test_offloaded_checkpoint_backward_pass(self):
        """Test that offloaded checkpoint computes correct gradients."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import offloaded_checkpoint

        # Create two identical modules
        module1 = SimpleModule(self.dim).to(self.device)
        module2 = SimpleModule(self.dim).to(self.device)
        module2.load_state_dict(module1.state_dict())

        x1 = torch.randn(self.batch_size, self.dim, device=self.device, requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)

        # Direct backward
        out1 = module1(x1)
        loss1 = out1.sum()
        loss1.backward()

        # Checkpointed backward
        out2 = offloaded_checkpoint(module2, x2, use_reentrant=False)
        loss2 = out2.sum()
        loss2.backward()

        # Check gradients match
        for (n1, p1), (n2, p2) in zip(module1.named_parameters(), module2.named_parameters()):
            self.assertTrue(
                torch.allclose(p1.grad, p2.grad, atol=1e-5),
                f"Gradient mismatch for {n1}",
            )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required for offload test")
    def test_offloaded_checkpoint_tuple_output(self):
        """Test that offloaded checkpoint handles tuple outputs correctly."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import offloaded_checkpoint

        class TupleModule(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear = nn.Linear(dim, dim)

            def forward(self, x, y):
                return self.linear(x), self.linear(y)

        module = TupleModule(self.dim).to(self.device)
        x = torch.randn(self.batch_size, self.dim, device=self.device)
        y = torch.randn(self.batch_size, self.dim, device=self.device)

        # Direct forward
        expected_x, expected_y = module(x, y)

        # Checkpointed forward
        result_x, result_y = offloaded_checkpoint(module, x, y, use_reentrant=False)

        self.assertTrue(torch.allclose(expected_x, result_x, atol=1e-6))
        self.assertTrue(torch.allclose(expected_y, result_y, atol=1e-6))

    def test_cpu_offload_hooks_pack_unpack(self):
        """Test that CPUOffloadHooks correctly packs and unpacks tensors."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import CPUOffloadHooks

        hooks = CPUOffloadHooks()

        if torch.cuda.is_available():
            tensor = torch.randn(4, 4, device="cuda")
            packed = hooks.pack(tensor)
            # Pack returns (cpu_tensor, original_device) tuple
            self.assertIsInstance(packed, tuple)
            self.assertEqual(len(packed), 2)
            cpu_tensor, original_device = packed
            self.assertEqual(cpu_tensor.device.type, "cpu")
            self.assertEqual(original_device.type, "cuda")

            unpacked = hooks.unpack(packed)
            self.assertEqual(unpacked.device.type, "cuda")
        else:
            # On CPU, tensors should pass through with None device
            tensor = torch.randn(4, 4)
            packed = hooks.pack(tensor)
            self.assertIsInstance(packed, tuple)
            cpu_tensor, original_device = packed
            self.assertEqual(cpu_tensor.device.type, "cpu")
            self.assertIsNone(original_device)


class TestGradientCheckpointingBackend(unittest.TestCase):
    """Tests for the gradient checkpointing backend module."""

    def test_set_checkpoint_backend(self):
        """Test that checkpoint backend can be set."""
        from simpletuner.helpers.training.gradient_checkpointing_interval import (
            get_checkpoint_backend,
            set_checkpoint_backend,
        )

        # Default should be torch
        original = get_checkpoint_backend()

        try:
            set_checkpoint_backend("unsloth")
            self.assertEqual(get_checkpoint_backend(), "unsloth")

            set_checkpoint_backend("torch")
            self.assertEqual(get_checkpoint_backend(), "torch")
        finally:
            # Restore original
            set_checkpoint_backend(original)

    def test_set_checkpoint_backend_validation(self):
        """Test that invalid backend values raise ValueError."""
        from simpletuner.helpers.training.gradient_checkpointing_interval import set_checkpoint_backend

        with self.assertRaises(ValueError) as cm:
            set_checkpoint_backend("invalid_backend")

        self.assertIn("invalid_backend", str(cm.exception))
        self.assertIn("torch", str(cm.exception))
        self.assertIn("unsloth", str(cm.exception))

    def test_get_checkpoint_function_torch(self):
        """Test that get_checkpoint_function returns torch checkpoint for torch backend."""
        from simpletuner.helpers.training.gradient_checkpointing_interval import (
            get_checkpoint_backend,
            get_checkpoint_function,
            set_checkpoint_backend,
        )

        original_backend = get_checkpoint_backend()

        try:
            set_checkpoint_backend("torch")
            checkpoint_fn = get_checkpoint_function()
            self.assertEqual(checkpoint_fn, torch.utils.checkpoint.checkpoint)
        finally:
            set_checkpoint_backend(original_backend)

    def test_get_checkpoint_function_unsloth(self):
        """Test that get_checkpoint_function returns offloaded checkpoint for unsloth backend."""
        from simpletuner.helpers.training.gradient_checkpointing_interval import (
            get_checkpoint_backend,
            get_checkpoint_function,
            set_checkpoint_backend,
        )
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import offloaded_checkpoint

        original_backend = get_checkpoint_backend()

        try:
            set_checkpoint_backend("unsloth")
            checkpoint_fn = get_checkpoint_function()
            self.assertEqual(checkpoint_fn, offloaded_checkpoint)
        finally:
            set_checkpoint_backend(original_backend)

    def test_checkpoint_function_produces_correct_output(self):
        """Test that checkpoint functions produce correct forward output."""
        from simpletuner.helpers.training.gradient_checkpointing_interval import (
            get_checkpoint_backend,
            get_checkpoint_function,
            set_checkpoint_backend,
        )

        original_backend = get_checkpoint_backend()

        try:
            module = SimpleModule(32)
            x = torch.randn(2, 32)

            # Direct forward
            expected = module(x)

            # Test with torch backend
            set_checkpoint_backend("torch")
            checkpoint_fn = get_checkpoint_function()
            result_torch = checkpoint_fn(module, x, use_reentrant=False)
            self.assertTrue(torch.allclose(expected, result_torch, atol=1e-6))

            # Test with unsloth backend (only if CUDA available)
            if torch.cuda.is_available():
                module = module.cuda()
                x = x.cuda()
                expected = module(x)

                set_checkpoint_backend("unsloth")
                checkpoint_fn = get_checkpoint_function()
                result_unsloth = checkpoint_fn(module, x, use_reentrant=False)
                self.assertTrue(torch.allclose(expected, result_unsloth, atol=1e-6))
        finally:
            set_checkpoint_backend(original_backend)


class TestConfigFieldIntegration(unittest.TestCase):
    """Tests for the configuration field integration."""

    def test_gradient_checkpointing_backend_field_exists(self):
        """Test that the gradient_checkpointing_backend field is registered."""
        from simpletuner.simpletuner_sdk.server.services.field_registry import FieldRegistry

        registry = FieldRegistry()
        field = registry.get_field("gradient_checkpointing_backend")

        self.assertIsNotNone(field)
        self.assertEqual(field.default_value, "torch")
        self.assertIn({"value": "torch", "label": "PyTorch (recompute)"}, field.choices)
        self.assertIn({"value": "unsloth", "label": "Unsloth (CPU offload)"}, field.choices)

    def test_gradient_checkpointing_backend_validation(self):
        """Test that invalid backend values are rejected."""
        from simpletuner.simpletuner_sdk.server.services.field_registry import FieldRegistry

        registry = FieldRegistry()
        field = registry.get_field("gradient_checkpointing_backend")

        # Check validation rules
        choices_rule = None
        for rule in field.validation_rules:
            if rule.rule_type.value == "choices":
                choices_rule = rule
                break

        self.assertIsNotNone(choices_rule)
        self.assertIn("torch", choices_rule.value)
        self.assertIn("unsloth", choices_rule.value)


class TestTransformerBackendAttribute(unittest.TestCase):
    """Tests that transformer models have the backend attribute and setter."""

    def test_flux_transformer_has_backend_attribute(self):
        """Test that FluxTransformer2DModel has gradient_checkpointing_backend."""
        from simpletuner.helpers.models.flux.transformer import FluxTransformer2DModel

        self.assertTrue(hasattr(FluxTransformer2DModel, "set_gradient_checkpointing_backend"))

    def test_sana_transformer_has_backend_attribute(self):
        """Test that SanaTransformer2DModel has gradient_checkpointing_backend."""
        from simpletuner.helpers.models.sana.transformer import SanaTransformer2DModel

        self.assertTrue(hasattr(SanaTransformer2DModel, "set_gradient_checkpointing_backend"))

    def test_sd3_transformer_has_backend_attribute(self):
        """Test that SD3Transformer2DModel has gradient_checkpointing_backend."""
        from simpletuner.helpers.models.sd3.transformer import SD3Transformer2DModel

        self.assertTrue(hasattr(SD3Transformer2DModel, "set_gradient_checkpointing_backend"))

    def test_chroma_transformer_has_backend_attribute(self):
        """Test that ChromaTransformer2DModel has gradient_checkpointing_backend."""
        from simpletuner.helpers.models.chroma.transformer import ChromaTransformer2DModel

        self.assertTrue(hasattr(ChromaTransformer2DModel, "set_gradient_checkpointing_backend"))

    def test_auraflow_transformer_has_backend_attribute(self):
        """Test that AuraFlowTransformer2DModel has gradient_checkpointing_backend."""
        from simpletuner.helpers.models.auraflow.transformer import AuraFlowTransformer2DModel

        self.assertTrue(hasattr(AuraFlowTransformer2DModel, "set_gradient_checkpointing_backend"))


if __name__ == "__main__":
    unittest.main()
