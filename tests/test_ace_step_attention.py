import unittest
from unittest.mock import MagicMock

import torch

from simpletuner.helpers.models.ace_step.customer_attention_processor import (
    CustomerAttnProcessor2_0,
    CustomLiteLAProcessor2_0,
)


class TestACEStepAttentionProcessors(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.heads = 4
        self.head_dim = 8
        self.dim = self.heads * self.head_dim
        self.seq_len = 16

        # Mock Attention module
        self.attn = MagicMock()
        self.attn.heads = self.heads
        self.attn.to_q = torch.nn.Linear(self.dim, self.dim)
        self.attn.to_k = torch.nn.Linear(self.dim, self.dim)
        self.attn.to_v = torch.nn.Linear(self.dim, self.dim)
        self.attn.to_out = torch.nn.ModuleList([torch.nn.Linear(self.dim, self.dim), torch.nn.Identity()])
        self.attn.norm_q = None
        self.attn.norm_k = None
        self.attn.group_norm = None
        self.attn.norm_cross = None
        self.attn.rescale_output_factor = 1.0
        self.attn.residual_connection = True
        self.attn.is_cross_attention = False

        # For prepare_attention_mask
        self.attn.prepare_attention_mask.return_value = torch.zeros(self.batch_size, 1, self.seq_len, self.seq_len)

    def test_customer_attn_processor_basic(self):
        processor = CustomerAttnProcessor2_0()
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.dim)

        # RoPE mocks - must match head_dim (not half)
        cos = torch.randn(self.seq_len, self.head_dim)
        sin = torch.randn(self.seq_len, self.head_dim)
        rotary_freqs_cis = (cos, sin)

        output = processor(self.attn, hidden_states, rotary_freqs_cis=rotary_freqs_cis)

        self.assertEqual(output.shape, hidden_states.shape)

    def test_custom_lite_la_processor_basic(self):
        processor = CustomLiteLAProcessor2_0()
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.dim)

        # RoPE mocks - must match head_dim (not half)
        cos = torch.randn(self.seq_len, self.head_dim)
        sin = torch.randn(self.seq_len, self.head_dim)
        rotary_freqs_cis = (cos, sin)

        output = processor(self.attn, hidden_states, rotary_freqs_cis=rotary_freqs_cis)

        self.assertEqual(output.shape, hidden_states.shape)


if __name__ == "__main__":
    unittest.main()
