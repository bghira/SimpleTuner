"""
Focused unit tests for ChromaTransformer2DModel.

These tests follow the shared transformer test conventions around:
- Base class inheritance from TransformerBaseTest
- Usage of helper utilities (TensorGenerator, MockDiffusersConfig, etc.)
- Standard method naming patterns for instantiation, forward pass, and typo prevention
"""

import os
import sys
import unittest
from typing import Dict, Optional

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

from simpletuner.helpers.models.chroma.transformer import ChromaTransformer2DModel


def _build_minimal_config(overrides: Optional[Dict] = None) -> Dict:
    """Return a tiny configuration that keeps tests fast."""
    config = {
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
    if overrides:
        config.update(overrides)
    return config


class TestChromaTransformer2DModel(TransformerBaseTest):
    """Test suite covering core behaviour of ChromaTransformer2DModel."""

    def setUp(self):
        super().setUp()
        self.tensor_gen = TensorGenerator()
        self.shape_validator = ShapeValidator()
        self.typo_utils = TypoTestUtils()
        self.default_config = _build_minimal_config()

    def _build_transformer(self, overrides: Optional[Dict] = None) -> ChromaTransformer2DModel:
        return ChromaTransformer2DModel(**_build_minimal_config(overrides))

    @torch.no_grad()
    def test_basic_instantiation(self):
        """Ensure the transformer can be constructed and exposes expected config."""
        transformer = self._build_transformer()

        self.assertEqual(
            transformer.inner_dim, self.default_config["num_attention_heads"] * self.default_config["attention_head_dim"]
        )
        self.assertEqual(transformer.config.patch_size, self.default_config["patch_size"])
        self.assertEqual(transformer.config.num_layers, self.default_config["num_layers"])

        mock_config = MockDiffusersConfig(
            num_attention_heads=transformer.config.num_attention_heads,
            attention_head_dim=transformer.config.attention_head_dim,
            patch_size=transformer.config.patch_size,
            num_layers=transformer.config.num_layers,
        )
        self.assertEqual(mock_config.num_layers, self.default_config["num_layers"])

        self.typo_utils.test_method_name_existence(transformer, ["forward"])

    @torch.no_grad()
    def test_forward_pass_structure(self):
        """Run a minimal forward pass and validate shape semantics."""
        transformer = self._build_transformer()

        batch_size = 2
        hidden_states = self.tensor_gen.create_hidden_states(
            batch_size=batch_size, seq_len=8, hidden_dim=transformer.config.in_channels
        )
        encoder_hidden_states = self.tensor_gen.create_encoder_hidden_states(
            batch_size=batch_size, seq_len=3, hidden_dim=self.default_config["joint_attention_dim"]
        )
        timestep = self.tensor_gen.create_timestep(batch_size=batch_size)
        txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3)
        img_ids = torch.zeros(hidden_states.shape[1], 3)

        outputs = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 1)
        sample = outputs[0]

        expected_hidden = hidden_states.shape[1]
        expected_channels = transformer.config.patch_size * transformer.config.patch_size * transformer.config.out_channels
        self.assertEqual(sample.shape, (batch_size, expected_hidden, expected_channels))
        self.shape_validator.validate_transformer_output(sample, batch_size, expected_hidden, expected_channels)
        self.assert_no_nan_or_inf(sample)

    def test_typo_prevention_for_constructor(self):
        """Ensure common constructor typos raise helpful errors."""
        with self.assertRaises(TypeError):
            ChromaTransformer2DModel(num_attention_head=2)  # Missing trailing 's'

        invalid_kwargs = self.default_config.copy()
        invalid_kwargs["joint_attn_dim"] = invalid_kwargs.pop("joint_attention_dim")
        with self.assertRaises(TypeError):
            ChromaTransformer2DModel(**invalid_kwargs)


if __name__ == "__main__":
    unittest.main()
