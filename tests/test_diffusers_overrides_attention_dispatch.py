import types
import unittest
from unittest.mock import patch

import torch
from diffusers.models import attention_dispatch

from simpletuner.helpers.training import diffusers_overrides  # noqa: F401


def _fake_attention_ctx() -> types.SimpleNamespace:
    batch_size, num_heads, seq_len, head_dim = 2, 3, 5, 7
    saved_layout = (batch_size, num_heads, seq_len, head_dim)
    saved_tensors = (
        torch.randn(saved_layout),
        torch.randn(saved_layout),
        torch.randn(saved_layout),
        torch.randn(saved_layout),
        torch.randn(batch_size, num_heads, seq_len),
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.zeros((), dtype=torch.int64),
        torch.zeros((), dtype=torch.int64),
    )
    return types.SimpleNamespace(
        saved_tensors=saved_tensors,
        attn_mask=None,
        max_q=seq_len,
        max_k=seq_len,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    )


class DiffusersTemplatedAttentionBackwardOverrideTests(unittest.TestCase):
    def test_uniform_varlen_metadata_compiles_with_dynamic_shapes(self):
        def metadata(query, key):
            return attention_dispatch._prepare_for_flash_attn_or_sage_varlen_without_mask(
                query.shape[0], query.shape[1], key.shape[1], query.device
            )

        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            compiled = torch.compile(metadata, backend="eager", fullgraph=True, dynamic=True)
            for batch, queries, keys in [(2, 5, 9), (4, 7, 11)]:
                with self.subTest(device=device, batch=batch):
                    lengths, offsets, maxima = compiled(
                        torch.empty(batch, queries, device=device), torch.empty(batch, keys, device=device)
                    )
                    self.assertEqual(maxima, (queries, keys))
                    for length, offset, size in zip(lengths, offsets, (queries, keys)):
                        torch.testing.assert_close(length, torch.full((batch,), size, dtype=torch.int32, device=device))
                        torch.testing.assert_close(offset, torch.arange(batch + 1, dtype=torch.int32, device=device) * size)

    def _assert_backward_uses_saved_bhsd_layout(self, backward_op, aten_op_name: str):
        ctx = _fake_attention_ctx()
        grad_out = torch.randn(2, 5, 3, 7)
        captured = {}

        def fake_aten_backward(grad_out_arg, query, key, value, out, **kwargs):
            captured["grad_out_shape"] = tuple(grad_out_arg.shape)
            captured["query_shape"] = tuple(query.shape)
            captured["key_shape"] = tuple(key.shape)
            captured["value_shape"] = tuple(value.shape)
            return torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)

        with patch.object(torch.ops.aten, aten_op_name, fake_aten_backward, create=True):
            grad_query, grad_key, grad_value = backward_op(ctx, grad_out)

        self.assertEqual(captured["grad_out_shape"], (2, 3, 5, 7))
        self.assertEqual(captured["query_shape"], (2, 3, 5, 7))
        self.assertEqual(captured["key_shape"], (2, 3, 5, 7))
        self.assertEqual(captured["value_shape"], (2, 3, 5, 7))
        self.assertEqual(tuple(grad_query.shape), (2, 5, 3, 7))
        self.assertEqual(tuple(grad_key.shape), (2, 5, 3, 7))
        self.assertEqual(tuple(grad_value.shape), (2, 5, 3, 7))

    def test_cudnn_backward_does_not_transpose_saved_key_value_twice(self):
        self._assert_backward_uses_saved_bhsd_layout(
            attention_dispatch._cudnn_attention_backward_op,
            "_scaled_dot_product_cudnn_attention_backward",
        )

    def test_native_flash_backward_does_not_transpose_saved_key_value_twice(self):
        self._assert_backward_uses_saved_bhsd_layout(
            attention_dispatch._native_flash_attention_backward_op,
            "_scaled_dot_product_flash_attention_backward",
        )


if __name__ == "__main__":
    unittest.main()
