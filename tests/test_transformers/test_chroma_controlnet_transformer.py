"""
Targeted unit tests for Chroma ControlNet components.

These tests focus on:
- Basic instantiation parity between the transformer and controlnet wrappers.
- Forward pass shape validation using the shared tensor helpers.
- Typo prevention for constructor parameters and critical method names.
"""

import os
import sys
import unittest
from typing import Dict

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

from transformer_base_test import ShapeValidator, TransformerBaseTest, TypoTestUtils
from transformer_test_helpers import MockDiffusersConfig, TensorGenerator

from simpletuner.helpers.models.chroma.controlnet import ChromaControlNetModel
from simpletuner.helpers.models.chroma.transformer import ChromaTransformer2DModel


class TestChromaControlNetTransformer(TransformerBaseTest):
    """Tests for Chroma ControlNet and its underlying transformer."""

    def setUp(self):
        super().setUp()
        self.tensor_gen = TensorGenerator()
        self.shape_validator = ShapeValidator()
        self.typo_utils = TypoTestUtils()

        # Minimal config shared between transformer and controlnet
        self.transformer_kwargs: Dict[str, int] = {
            "patch_size": 1,
            "in_channels": 4,
            "out_channels": 4,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 6,
            "num_attention_heads": 2,
            "joint_attention_dim": 8,
            "axes_dims_rope": (2, 2, 2),
            "approximator_num_channels": 12,
            "approximator_hidden_dim": 24,
            "approximator_layers": 1,
        }
        self.controlnet_kwargs = {
            key: value for key, value in self.transformer_kwargs.items() if key != "out_channels"
        }
        self.expected_inner_dim = (
            self.transformer_kwargs["attention_head_dim"] * self.transformer_kwargs["num_attention_heads"]
        )
        self.mock_config = MockDiffusersConfig(**self.controlnet_kwargs)

    def _create_transformer(self) -> ChromaTransformer2DModel:
        return ChromaTransformer2DModel(**self.transformer_kwargs)

    def _create_controlnet(self) -> ChromaControlNetModel:
        return ChromaControlNetModel(**self.controlnet_kwargs)

    # --------------------------------------------------------------------- #
    # Required pattern: test_basic_instantiation
    # --------------------------------------------------------------------- #
    def test_basic_instantiation_matches_transformer_layout(self):
        transformer = self._create_transformer()
        controlnet = self._create_controlnet()

        self.assertEqual(controlnet.inner_dim, self.expected_inner_dim)
        self.assertEqual(len(controlnet.transformer_blocks), len(transformer.transformer_blocks))
        self.assertEqual(len(controlnet.single_transformer_blocks), len(transformer.single_transformer_blocks))
        # Ensure the mock config mirrors important attributes
        self.assertEqual(self.mock_config.num_attention_heads, self.controlnet_kwargs["num_attention_heads"])
        self.assertEqual(self.mock_config.attention_head_dim, self.controlnet_kwargs["attention_head_dim"])

    # --------------------------------------------------------------------- #
    # Required pattern: test_forward_pass
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def test_forward_pass_produces_expected_shapes(self):
        controlnet = self._create_controlnet()

        batch_size = 2
        seq_len = 8

        hidden_states = self.tensor_gen.create_hidden_states(batch_size=batch_size, seq_len=seq_len, hidden_dim=4)
        controlnet_cond = self.tensor_gen.create_hidden_states(batch_size=batch_size, seq_len=seq_len, hidden_dim=4)
        encoder_hidden_states = self.tensor_gen.create_encoder_hidden_states(
            batch_size=batch_size, seq_len=0, hidden_dim=self.controlnet_kwargs["joint_attention_dim"]
        )
        timestep = self.tensor_gen.create_timestep(batch_size=batch_size)

        txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3)
        img_ids = torch.zeros(seq_len, 3)

        block_samples, single_block_samples = controlnet(
            hidden_states=hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=1.0,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )

        self.assertIsInstance(block_samples, tuple)
        self.assertIsInstance(single_block_samples, tuple)
        self.assertEqual(len(block_samples), len(controlnet.transformer_blocks))
        self.assertEqual(len(single_block_samples), len(controlnet.single_transformer_blocks))

        self.shape_validator.validate_transformer_output(
            block_samples[0], batch_size, seq_len, self.expected_inner_dim
        )
        self.shape_validator.validate_transformer_output(
            single_block_samples[0], batch_size, seq_len, self.expected_inner_dim
        )

    # --------------------------------------------------------------------- #
    # Additional structural and typo tests
    # --------------------------------------------------------------------- #
    def test_from_transformer_copies_structure(self):
        transformer = self._create_transformer()
        controlnet = ChromaControlNetModel.from_transformer(transformer)

        self.assertEqual(controlnet.inner_dim, transformer.inner_dim)
        # Confirm modules are copied without unexpected dtype/device issues
        self.assertTrue(torch.equal(controlnet.time_text_embed.mod_proj, transformer.time_text_embed.mod_proj))
        self.typo_utils.test_method_name_existence(
            controlnet,
            [
                "set_attn_processor",
                "enable_gradient_checkpointing",
                "disable_gradient_checkpointing",
            ],
        )

    # --------------------------------------------------------------------- #
    # Required pattern: test_typo_prevention
    # --------------------------------------------------------------------- #
    def test_typo_prevention_in_constructor(self):
        with self.assertRaises(TypeError):
            ChromaControlNetModel(
                patch_sizes=1,  # Intentional typo: should be patch_size
                in_channels=4,
                num_layers=1,
                num_single_layers=1,
                attention_head_dim=6,
                num_attention_heads=2,
                joint_attention_dim=8,
            )


if __name__ == "__main__":
    unittest.main()
