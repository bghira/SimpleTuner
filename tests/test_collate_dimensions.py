"""
Tests for collate_prompt_embeds dimension handling across different models.

These tests reproduce dimension mismatch issues that occur when:
1. Cached embeddings have different shapes (with/without batch dimension)
2. Different models collate embeddings differently
3. Batch size is 1 vs multiple samples
"""

import unittest
from unittest.mock import MagicMock

import torch

from simpletuner.helpers.models.auraflow.model import Auraflow
from simpletuner.helpers.models.chroma.model import Chroma
from simpletuner.helpers.models.hidream.model import HiDream
from simpletuner.helpers.models.qwen_image.model import QwenImage


class TestCollatePromptEmbedsDimensions(unittest.TestCase):
    """Test dimension handling in collate_prompt_embeds across different models."""

    def setUp(self):
        """Set up mock accelerator for model initialization."""
        self.mock_accelerator = MagicMock()
        self.mock_accelerator.device = "cpu"

        self.mock_config = MagicMock()
        self.mock_config.model_family = "test"
        self.mock_config.model_flavour = "test"
        self.mock_config.weight_dtype = torch.float32

    def test_qwen_collate_batch_size_1_with_batch_dim(self):
        """
        Test Qwen collate_prompt_embeds with batch size 1 when embeddings already have batch dim.

        This reproduces the error:
        IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
        at line: txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        """
        model = QwenImage(
            accelerator=self.mock_accelerator,
            config=self.mock_config,
        )

        # Simulate cached embeddings with batch dimension: [1, seq_len, hidden_dim]
        text_encoder_output = [
            {
                "prompt_embeds": torch.randn(1, 1386, 3584),  # Already has batch dim
                "attention_masks": torch.ones(1, 1386),  # Already has batch dim
            }
        ]

        result = model.collate_prompt_embeds(text_encoder_output)

        # Verify result has correct shape
        self.assertEqual(result["prompt_embeds"].dim(), 3, "prompt_embeds should be 3D: [batch, seq, dim]")
        self.assertEqual(result["prompt_embeds"].shape[0], 1, "batch size should be 1")
        self.assertEqual(result["attention_masks"].dim(), 2, "attention_masks should be 2D: [batch, seq]")
        self.assertEqual(result["attention_masks"].shape[0], 1, "batch size should be 1")

        # Verify we can do .sum(dim=1) on the mask
        try:
            seq_lens = result["attention_masks"].sum(dim=1).tolist()
            self.assertEqual(len(seq_lens), 1, "Should have 1 sequence length for batch size 1")
        except IndexError as e:
            self.fail(f"sum(dim=1) failed on attention mask: {e}")

    def test_qwen_collate_batch_size_1_without_batch_dim(self):
        """
        Test Qwen collate_prompt_embeds with batch size 1 when embeddings lack batch dim.

        This tests the case where cached embeddings are 2D: [seq_len, hidden_dim]
        """
        model = QwenImage(
            accelerator=self.mock_accelerator,
            config=self.mock_config,
        )

        # Simulate cached embeddings WITHOUT batch dimension: [seq_len, hidden_dim]
        text_encoder_output = [
            {
                "prompt_embeds": torch.randn(1386, 3584),  # No batch dim
                "attention_masks": torch.ones(1386),  # No batch dim
            }
        ]

        result = model.collate_prompt_embeds(text_encoder_output)

        # Verify result has batch dimension added
        self.assertEqual(result["prompt_embeds"].dim(), 3, "prompt_embeds should be 3D: [batch, seq, dim]")
        self.assertEqual(result["prompt_embeds"].shape[0], 1, "batch size should be 1")
        self.assertEqual(result["attention_masks"].dim(), 2, "attention_masks should be 2D: [batch, seq]")
        self.assertEqual(result["attention_masks"].shape[0], 1, "batch size should be 1")

        # Verify we can do .sum(dim=1) on the mask
        seq_lens = result["attention_masks"].sum(dim=1).tolist()
        self.assertEqual(len(seq_lens), 1, "Should have 1 sequence length for batch size 1")

    def test_qwen_collate_batch_size_multiple_with_batch_dim(self):
        """Test Qwen collate_prompt_embeds with multiple samples."""
        model = QwenImage(
            accelerator=self.mock_accelerator,
            config=self.mock_config,
        )

        # Simulate 4 cached embeddings, each with batch dimension
        text_encoder_output = [
            {
                "prompt_embeds": torch.randn(1, 1386, 3584),
                "attention_masks": torch.ones(1, 1386),
            }
            for _ in range(4)
        ]

        result = model.collate_prompt_embeds(text_encoder_output)

        # Verify concatenation along batch dimension
        self.assertEqual(result["prompt_embeds"].shape[0], 4, "batch size should be 4")
        self.assertEqual(result["attention_masks"].shape[0], 4, "batch size should be 4")

        # Verify we can do .sum(dim=1) on the mask
        seq_lens = result["attention_masks"].sum(dim=1).tolist()
        self.assertEqual(len(seq_lens), 4, "Should have 4 sequence lengths for batch size 4")

    def test_qwen_collate_batch_size_multiple_without_batch_dim(self):
        """Test Qwen collate_prompt_embeds with multiple samples without batch dim."""
        model = QwenImage(
            accelerator=self.mock_accelerator,
            config=self.mock_config,
        )

        # Simulate 4 cached embeddings WITHOUT batch dimension
        text_encoder_output = [
            {
                "prompt_embeds": torch.randn(1386, 3584),
                "attention_masks": torch.ones(1386),
            }
            for _ in range(4)
        ]

        result = model.collate_prompt_embeds(text_encoder_output)

        # Verify concatenation after adding batch dimension
        self.assertEqual(result["prompt_embeds"].shape[0], 4, "batch size should be 4")
        self.assertEqual(result["attention_masks"].shape[0], 4, "batch size should be 4")

        # Verify we can do .sum(dim=1) on the mask
        seq_lens = result["attention_masks"].sum(dim=1).tolist()
        self.assertEqual(len(seq_lens), 4, "Should have 4 sequence lengths for batch size 4")

    def test_auraflow_collate_dimensions(self):
        """
        Test Auraflow dimension handling.

        Auraflow doesn't implement collate_prompt_embeds, so it falls back to default stacking.
        This test verifies what happens with the default behavior.
        """
        # Use the base class implementation directly (Auraflow doesn't override it)
        from simpletuner.helpers.models.common import ImageModelFoundation

        class MockAuraflow(ImageModelFoundation):
            """Minimal mock to test collate_prompt_embeds without full init."""

            def _encode_prompts(self, prompts, is_negative_prompt=False):
                pass

            def convert_text_embed_for_pipeline(self, text_embedding):
                pass

            def convert_negative_text_embed_for_pipeline(self, text_embedding, prompt):
                pass

            def model_predict(self, prepared_batch):
                pass

        model = MockAuraflow.__new__(MockAuraflow)

        # Test that Auraflow returns empty dict (uses default collate logic)
        text_encoder_output = [
            {
                "prompt_embeds": torch.randn(256, 2048),  # [seq_len, hidden_dim]
                "prompt_attention_mask": torch.ones(256),
            }
        ]

        result = model.collate_prompt_embeds(text_encoder_output)

        # Auraflow should return empty dict, falling back to default stacking
        self.assertEqual(result, {}, "Auraflow should return empty dict from collate_prompt_embeds")

    def test_auraflow_default_stack_behavior(self):
        """
        Test what happens when Auraflow embeddings go through default torch.stack().

        This simulates the collate.py behavior when collate_prompt_embeds returns {}.
        """
        # Simulate what collate.py does when collate_prompt_embeds returns {}
        text_encoder_output = [
            {
                "prompt_embeds": torch.randn(256, 2048),  # [seq_len, hidden_dim]
                "prompt_attention_mask": torch.ones(256),
            }
            for _ in range(4)
        ]

        # This is what collate.py does:
        stacked_embeds = torch.stack([t["prompt_embeds"] for t in text_encoder_output])
        stacked_masks = torch.stack([t["prompt_attention_mask"] for t in text_encoder_output])

        # After stacking, we get [batch, seq_len, hidden_dim]
        self.assertEqual(stacked_embeds.shape, (4, 256, 2048))
        self.assertEqual(stacked_masks.shape, (4, 256))

    def test_auraflow_transformer_cat_mismatch(self):
        """
        Test the dimension mismatch that can occur in Auraflow transformer.

        At line 597 in transformer.py:
        encoder_hidden_states = torch.cat([self.register_tokens.repeat(...), encoder_hidden_states], dim=1)

        This can fail if encoder_hidden_states has wrong number of dimensions.
        """
        # Simulate register_tokens with shape [1, num_registers, hidden_dim]
        register_tokens = torch.randn(1, 8, 2048)

        # Case 1: Correct shape - [batch, seq, dim]
        encoder_hidden_states_correct = torch.randn(4, 256, 2048)
        repeated_registers = register_tokens.repeat(encoder_hidden_states_correct.size(0), 1, 1)

        # This should work
        try:
            result = torch.cat([repeated_registers, encoder_hidden_states_correct], dim=1)
            self.assertEqual(result.shape, (4, 264, 2048), "Concatenation should produce [4, 264, 2048]")
        except RuntimeError as e:
            self.fail(f"Concatenation failed with correct shapes: {e}")

        # Case 2: Wrong shape - [batch, batch, seq, dim] (extra dimension from incorrect stacking)
        encoder_hidden_states_wrong = torch.randn(4, 1, 256, 2048)

        # This should fail with dimension mismatch
        with self.assertRaises(RuntimeError, msg="Should fail with 4D tensor"):
            # register_tokens is 3D, encoder_hidden_states is 4D
            torch.cat([repeated_registers, encoder_hidden_states_wrong], dim=1)

    def test_chroma_collate_dimensions(self):
        """Test Chroma collate_prompt_embeds dimension handling."""
        # Create minimal mock without full initialization
        from simpletuner.helpers.models.chroma.model import Chroma

        model = Chroma.__new__(Chroma)

        # Chroma uses torch.stack in its collate_prompt_embeds
        text_encoder_output = [
            {
                "prompt_embeds": torch.randn(256, 4096),  # [seq_len, hidden_dim]
                "attention_masks": torch.ones(256),
            }
            for _ in range(4)
        ]

        result = model.collate_prompt_embeds(text_encoder_output)

        # Chroma stacks, so we should get [batch, seq_len, hidden_dim]
        self.assertEqual(result["prompt_embeds"].shape, (4, 256, 4096))
        self.assertEqual(result["attention_masks"].shape, (4, 256))

    def test_hidream_collate_dimensions(self):
        """Test HiDream collate_prompt_embeds dimension handling."""
        # Create minimal mock without full initialization
        from simpletuner.helpers.models.hidream.model import HiDream

        model = HiDream.__new__(HiDream)
        model.accelerator = self.mock_accelerator  # Set required attribute

        # HiDream has multiple embedding types
        text_encoder_output = [
            {
                "t5_prompt_embeds": torch.randn(256, 4096),
                "llama_prompt_embeds": torch.randn(256, 2048),
                "pooled_prompt_embeds": torch.randn(2048),
            }
            for _ in range(4)
        ]

        result = model.collate_prompt_embeds(text_encoder_output)

        # HiDream stacks all three types
        self.assertEqual(result["t5_prompt_embeds"].shape, (4, 256, 4096))
        self.assertEqual(result["llama_prompt_embeds"].shape, (4, 256, 2048))
        self.assertEqual(result["pooled_prompt_embeds"].shape, (4, 2048))

    def test_mixed_dimension_scenarios(self):
        """
        Test the scenario where some embeddings have batch dim and others don't.

        This can happen when cache was created at different times or by different code paths.
        """
        model = QwenImage(
            accelerator=self.mock_accelerator,
            config=self.mock_config,
        )

        # Mix of embeddings with and without batch dimension
        text_encoder_output = [
            {"prompt_embeds": torch.randn(1, 1386, 3584), "attention_masks": torch.ones(1, 1386)},  # With batch dim
            {"prompt_embeds": torch.randn(1386, 3584), "attention_masks": torch.ones(1386)},  # Without batch dim
            {"prompt_embeds": torch.randn(1, 1386, 3584), "attention_masks": torch.ones(1, 1386)},  # With batch dim
        ]

        result = model.collate_prompt_embeds(text_encoder_output)

        # All should be normalized to have batch dimension and concatenated
        self.assertEqual(result["prompt_embeds"].shape[0], 3, "Should concatenate to batch size 3")
        self.assertEqual(result["attention_masks"].shape[0], 3, "Should concatenate to batch size 3")


class TestDefaultCollateLogic(unittest.TestCase):
    """Test the default collate logic in collate.py when model returns empty dict."""

    def test_default_stack_with_2d_embeddings(self):
        """Test default stacking behavior with 2D embeddings (no batch dim)."""
        text_encoder_output = [{"prompt_embeds": torch.randn(256, 2048)} for _ in range(4)]

        # Simulate collate.py default behavior
        stacked = torch.stack([t["prompt_embeds"] for t in text_encoder_output])

        self.assertEqual(stacked.shape, (4, 256, 2048), "Should stack to [batch, seq, dim]")

    def test_default_stack_with_3d_embeddings(self):
        """
        Test default stacking behavior with 3D embeddings (already have batch dim).

        This creates the problem: stacking adds ANOTHER batch dimension!
        """
        text_encoder_output = [{"prompt_embeds": torch.randn(1, 256, 2048)} for _ in range(4)]  # Already has batch dim

        # Simulate collate.py default behavior
        stacked = torch.stack([t["prompt_embeds"] for t in text_encoder_output])

        # This is the BUG: we get [4, 1, 256, 2048] instead of [4, 256, 2048]
        self.assertEqual(stacked.shape, (4, 1, 256, 2048), "Stacking 3D tensors creates 4D tensor (the bug!)")

    def test_default_cat_would_work_for_3d(self):
        """Show that torch.cat would work correctly for 3D embeddings."""
        text_encoder_output = [{"prompt_embeds": torch.randn(1, 256, 2048)} for _ in range(4)]  # Already has batch dim

        # Using torch.cat instead of torch.stack
        concatenated = torch.cat([t["prompt_embeds"] for t in text_encoder_output], dim=0)

        # This works correctly: [4, 256, 2048]
        self.assertEqual(concatenated.shape, (4, 256, 2048), "Concatenating 3D tensors along dim=0 works correctly")


class TestCollateHelperFunction(unittest.TestCase):
    """Test the _collate_tensors helper function from collate.py."""

    def setUp(self):
        # Import the helper function from collate module
        # Since it's a local function, we'll test it through the collate_fn pathway
        # or recreate it here for testing
        def _collate_tensors(tensors):
            """
            Intelligently collate a list of tensors, handling both 2D and 3D cases.
            """
            if not tensors:
                return None

            first_tensor = tensors[0]
            dims = first_tensor.dim()
            all_same_dims = all(t.dim() == dims for t in tensors)

            if dims == 2:
                return torch.stack(tensors)
            elif dims == 3 and all_same_dims:
                return torch.cat(tensors, dim=0)
            elif dims == 1:
                return torch.stack(tensors)
            else:
                normalized = []
                for t in tensors:
                    if t.dim() == 2:
                        normalized.append(t.unsqueeze(0))
                    elif t.dim() == 3:
                        normalized.append(t)
                    elif t.dim() == 1:
                        normalized.append(t.unsqueeze(0))
                    else:
                        raise ValueError(f"Unexpected tensor dimension: {t.dim()} with shape {t.shape}")
                return torch.cat(normalized, dim=0)

        self._collate_tensors = _collate_tensors

    def test_collate_2d_tensors(self):
        """Test collating 2D tensors (no batch dimension)."""
        tensors = [torch.randn(256, 2048) for _ in range(4)]
        result = self._collate_tensors(tensors)
        self.assertEqual(result.shape, (4, 256, 2048), "2D tensors should stack to [batch, seq, dim]")

    def test_collate_3d_tensors(self):
        """Test collating 3D tensors (with batch dimension)."""
        tensors = [torch.randn(1, 256, 2048) for _ in range(4)]
        result = self._collate_tensors(tensors)
        self.assertEqual(result.shape, (4, 256, 2048), "3D tensors should concatenate to [batch, seq, dim]")

    def test_collate_1d_tensors(self):
        """Test collating 1D tensors."""
        tensors = [torch.randn(256) for _ in range(4)]
        result = self._collate_tensors(tensors)
        self.assertEqual(result.shape, (4, 256), "1D tensors should stack to [batch, dim]")

    def test_collate_mixed_2d_3d_tensors(self):
        """Test collating mixed 2D and 3D tensors."""
        tensors = [
            torch.randn(1, 256, 2048),  # 3D
            torch.randn(256, 2048),  # 2D
            torch.randn(1, 256, 2048),  # 3D
            torch.randn(256, 2048),  # 2D
        ]
        result = self._collate_tensors(tensors)
        self.assertEqual(result.shape, (4, 256, 2048), "Mixed tensors should normalize and concatenate")

    def test_collate_empty_list(self):
        """Test collating empty list."""
        result = self._collate_tensors([])
        self.assertIsNone(result, "Empty list should return None")

    def test_collate_single_2d_tensor(self):
        """Test collating a single 2D tensor."""
        tensors = [torch.randn(256, 2048)]
        result = self._collate_tensors(tensors)
        self.assertEqual(result.shape, (1, 256, 2048), "Single 2D tensor should get batch dimension")

    def test_collate_single_3d_tensor(self):
        """Test collating a single 3D tensor."""
        tensors = [torch.randn(1, 256, 2048)]
        result = self._collate_tensors(tensors)
        self.assertEqual(result.shape, (1, 256, 2048), "Single 3D tensor should maintain shape")

    def test_collate_batch_size_varies(self):
        """Test that 3D tensors with batch_size > 1 are concatenated correctly."""
        tensors = [
            torch.randn(2, 256, 2048),  # batch=2
            torch.randn(3, 256, 2048),  # batch=3
        ]
        result = self._collate_tensors(tensors)
        self.assertEqual(result.shape, (5, 256, 2048), "Should concatenate to total batch size of 5")


if __name__ == "__main__":
    unittest.main()
