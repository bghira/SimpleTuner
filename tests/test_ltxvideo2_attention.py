import unittest
from unittest.mock import patch

import torch
from diffusers.models.attention_dispatch import AttentionBackendName

from simpletuner.helpers.models.ltxvideo2 import transformer as ltx2_transformer


class TestLTX2AttentionDispatch(unittest.TestCase):
    def test_prepared_mask_skips_per_attention_preparation(self):
        attn = ltx2_transformer.LTX2Attention(query_dim=8, heads=2, dim_head=4, bias=False)
        hidden_states = torch.randn(1, 3, 8)
        prepared_mask = torch.ones(1, 2, 1, 3, dtype=torch.bool)
        attention_output = torch.randn(1, 3, 2, 4)

        with (
            patch.object(ltx2_transformer, "_ltx2_prepare_attention_mask") as prepare_mask,
            patch.object(ltx2_transformer, "_ltx2_dispatch_attention", return_value=attention_output),
        ):
            result = attn.processor(
                attn,
                hidden_states,
                attention_mask=prepared_mask,
                attention_mask_prepared=True,
            )

        prepare_mask.assert_not_called()
        self.assertEqual(result.shape, hidden_states.shape)

    def test_flash3_varlen_uses_eager_helper_while_compiling(self):
        query = torch.randn(1, 2, 1, 4)
        key = torch.randn(1, 3, 1, 4)
        value = torch.randn(1, 3, 1, 4)
        expected = torch.randn(1, 2, 1, 4)

        with (
            patch.object(ltx2_transformer.torch.compiler, "is_compiling", return_value=True),
            patch.object(
                ltx2_transformer,
                "_ltx2_flash3_varlen_hub_attention_eager",
                return_value=expected,
            ) as eager_attention,
            patch.object(
                ltx2_transformer,
                "_ltx2_flash3_varlen_hub_attention",
            ) as compiled_attention,
        ):
            result = ltx2_transformer._ltx2_dispatch_attention(
                query=query,
                key=key,
                value=value,
                attention_mask=torch.ones(1, 3, dtype=torch.bool),
                backend=AttentionBackendName._FLASH_3_VARLEN_HUB,
                parallel_config=None,
            )

        self.assertIs(result, expected)
        eager_attention.assert_called_once()
        compiled_attention.assert_not_called()

    def test_flash_varlen_hub_receives_bool_mask(self):
        query = torch.randn(1, 2, 1, 4)
        key = torch.randn(1, 3, 1, 4)
        value = torch.randn(1, 3, 1, 4)
        additive_mask = torch.tensor([[[[0.0, -10000.0, 0.0]]]], dtype=torch.bfloat16)
        expected = torch.randn(1, 2, 1, 4)

        with patch.object(ltx2_transformer, "dispatch_attention_fn", return_value=expected) as dispatch_attention:
            result = ltx2_transformer._ltx2_dispatch_attention(
                query=query,
                key=key,
                value=value,
                attention_mask=additive_mask,
                backend=AttentionBackendName.FLASH_VARLEN_HUB,
                parallel_config=None,
            )

        self.assertIs(result, expected)
        forwarded_mask = dispatch_attention.call_args.kwargs["attn_mask"]
        self.assertEqual(forwarded_mask.dtype, torch.bool)
        self.assertTrue(torch.equal(forwarded_mask, torch.tensor([[True, False, True]])))

    def test_flash_varlen_hub_uses_precomputed_metadata(self):
        query = torch.randn(1, 2, 1, 4)
        key = torch.randn(1, 3, 1, 4)
        value = torch.randn(1, 3, 1, 4)
        metadata = ltx2_transformer.LTX2FlashAttentionMetadata(
            indices_k=torch.tensor([0, 2]),
            cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 2], dtype=torch.int32),
            max_seqlen_q=2,
            max_seqlen_k=2,
        )
        expected = torch.randn(1, 2, 1, 4)

        with (
            patch.object(
                ltx2_transformer,
                "_ltx2_flash_varlen_hub_attention_prepared",
                return_value=expected,
            ) as prepared_attention,
            patch.object(ltx2_transformer, "dispatch_attention_fn") as dispatch_attention,
        ):
            result = ltx2_transformer._ltx2_dispatch_attention(
                query=query,
                key=key,
                value=value,
                attention_mask=None,
                backend=AttentionBackendName.FLASH_VARLEN_HUB,
                parallel_config=None,
                flash_attention_metadata=metadata,
            )

        self.assertIs(result, expected)
        prepared_attention.assert_called_once_with(query, key, value, metadata)
        dispatch_attention.assert_not_called()

    def test_prepares_flash_metadata_once_for_masked_keys(self):
        metadata = ltx2_transformer._ltx2_prepare_flash_attention_metadata(
            torch.tensor([[True, False, True], [True, True, False]]),
            batch_size=2,
            seq_len_q=4,
            seq_len_kv=3,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(metadata.indices_k, torch.tensor([0, 2, 3, 4])))
        self.assertTrue(torch.equal(metadata.cu_seqlens_q, torch.tensor([0, 4, 8], dtype=torch.int32)))
        self.assertTrue(torch.equal(metadata.cu_seqlens_k, torch.tensor([0, 2, 4], dtype=torch.int32)))
        self.assertEqual(metadata.max_seqlen_q, 4)
        self.assertEqual(metadata.max_seqlen_k, 2)

    def test_prompt_timesteps_use_batch_axis_from_tokenwise_timesteps(self):
        timesteps = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        prompt_timesteps = ltx2_transformer._ltx2_prompt_timesteps(timesteps, batch_size=2, name="video prompt")

        self.assertTrue(torch.equal(prompt_timesteps, torch.tensor([0.1, 0.4])))


if __name__ == "__main__":
    unittest.main()
