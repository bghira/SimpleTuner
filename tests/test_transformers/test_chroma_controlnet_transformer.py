"""
Targeted unit tests for ChromaControlNetModel and interoperability with ChromaTransformer2DModel.

These tests follow the shared transformer test conventions:
- Base class inheritance from TransformerBaseTest
- Usage of helper utilities (TensorGenerator, MockDiffusersConfig, etc.)
- Standard test method patterns (instantiation, forward structure, typo prevention)
"""

import os
import sys
import unittest
from typing import Dict

import torch

# Make shared transformer test utilities importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

from transformer_base_test import TransformerBaseTest  # noqa: E402
from transformer_test_helpers import (  # noqa: E402
    MockDiffusersConfig,
    ShapeValidator,
    TensorGenerator,
    TypoTestUtils,
)

from simpletuner.helpers.models.chroma.controlnet import ChromaControlNetModel
from simpletuner.helpers.models.chroma.transformer import ChromaTransformer2DModel


def _build_minimal_transformer(config_overrides: Dict = None) -> ChromaTransformer2DModel:
    """Utility for constructing a tiny transformer used across tests."""
    overrides = {
        "patch_size": 1,
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 4,
        "num_attention_heads": 2,
        "joint_attention_dim": 8,
        "axes_dims_rope": (2, 2, 2),
        "approximator_num_channels": 8,
        "approximator_hidden_dim": 16,
        "approximator_layers": 1,
    }
    if config_overrides:
        overrides.update(config_overrides)
    return ChromaTransformer2DModel(**overrides)


class TestChromaControlNetModel(TransformerBaseTest):
    """Test suite covering basic behaviour of ChromaControlNetModel."""

    def setUp(self):
        super().setUp()
        self.tensor_gen = TensorGenerator()
        self.shape_validator = ShapeValidator()
        self.default_config = {
            "patch_size": 1,
            "in_channels": 4,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 4,
            "num_attention_heads": 2,
            "joint_attention_dim": 8,
            "axes_dims_rope": (2, 2, 2),
            "approximator_num_channels": 8,
            "approximator_hidden_dim": 16,
            "approximator_layers": 1,
        }

    @torch.no_grad()
    def test_basic_instantiation(self):
        """Ensure the model can be instantiated and copied from a transformer."""
        transformer = _build_minimal_transformer()
        controlnet = ChromaControlNetModel(**self.default_config)

        # Validate config registration behaves like other models
        mock_config = MockDiffusersConfig(
            num_attention_heads=controlnet.config.num_attention_heads,
            attention_head_dim=controlnet.config.attention_head_dim,
            patch_size=controlnet.config.patch_size,
            num_layers=controlnet.config.num_layers,
        )
        self.assertEqual(mock_config.patch_size, self.default_config["patch_size"])
        self.assertEqual(mock_config.num_layers, self.default_config["num_layers"])
        self.typo_utils.test_method_name_existence(controlnet, ["forward"])

        # Verify from_transformer preserves structure
        cloned = ChromaControlNetModel.from_transformer(transformer)
        self.assertEqual(controlnet.inner_dim, cloned.inner_dim)
        self.assertEqual(len(cloned.transformer_blocks), len(transformer.transformer_blocks))
        self.assertEqual(len(cloned.single_transformer_blocks), len(transformer.single_transformer_blocks))

    @torch.no_grad()
    def test_forward_pass_structure(self):
        """Run a minimal forward pass and check emitted residual shapes."""
        controlnet = ChromaControlNetModel(**self.default_config)

        hidden_states = self.tensor_gen.create_hidden_states(batch_size=2, seq_len=8, hidden_dim=4)
        controlnet_cond = self.tensor_gen.create_hidden_states(batch_size=2, seq_len=8, hidden_dim=4)
        encoder_hidden_states = self.tensor_gen.create_encoder_hidden_states(batch_size=2, seq_len=3, hidden_dim=8)
        timestep = self.tensor_gen.create_timestep(batch_size=2)
        txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3)
        img_ids = torch.zeros(hidden_states.shape[1], 3)

        block_samples, single_block_samples = controlnet(
            hidden_states=hidden_states,
            controlnet_cond=controlnet_cond,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )

        self.assertIsInstance(block_samples, tuple)
        self.assertIsInstance(single_block_samples, tuple)
        self.assertEqual(len(block_samples), controlnet.config.num_layers)
        self.assertEqual(len(single_block_samples), controlnet.config.num_single_layers)

        self.shape_validator.validate_transformer_output(block_samples[0], 2, hidden_states.shape[1], controlnet.inner_dim)
        self.shape_validator.validate_transformer_output(single_block_samples[0], 2, hidden_states.shape[1], controlnet.inner_dim)

    def test_typo_prevention_for_constructor(self):
        """Ensure common constructor typos raise errors."""
        with self.assertRaises(TypeError):
            # Intentional typo: num_single_layer vs num_single_layers
            ChromaControlNetModel(num_single_layer=1)

        typo_mappings = {"num_attention_head": "num_attention_heads", "joint_attn_dim": "joint_attention_dim"}
        for typo, correct in typo_mappings.items():
            invalid_kwargs = self.default_config.copy()
            invalid_kwargs[typo] = invalid_kwargs.pop(correct)
            with self.assertRaises(TypeError):
                ChromaControlNetModel(**invalid_kwargs)


if __name__ == "__main__":
    unittest.main()
