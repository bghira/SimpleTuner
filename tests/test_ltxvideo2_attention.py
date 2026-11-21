import unittest
from contextlib import ExitStack
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

    def test_native_efficient_uses_compile_safe_direct_operator_without_context_parallelism(self):
        query = torch.randn(1, 2, 1, 4)
        key = torch.randn(1, 3, 1, 4)
        value = torch.randn(1, 3, 1, 4)
        expected = torch.randn(1, 2, 1, 4)

        with (
            patch.object(ltx2_transformer, "_ltx2_native_efficient_attention", return_value=expected) as native_attention,
            patch.object(ltx2_transformer, "dispatch_attention_fn") as dispatch_attention,
        ):
            result = ltx2_transformer._ltx2_dispatch_attention(
                query=query,
                key=key,
                value=value,
                attention_mask=None,
                backend=AttentionBackendName._NATIVE_EFFICIENT,
                parallel_config=None,
            )

        self.assertIs(result, expected)
        native_attention.assert_called_once_with(query, key, value, None)
        dispatch_attention.assert_not_called()

    def test_native_efficient_expands_broadcast_query_mask(self):
        query = torch.randn(1, 5, 2, 8)
        key = torch.randn(1, 7, 2, 8)
        value = torch.randn(1, 7, 2, 8)
        attention_mask = torch.ones(1, 1, 1, 7, dtype=torch.bool)

        with patch.object(torch.ops.aten, "_scaled_dot_product_efficient_attention") as efficient_attention:
            efficient_attention.return_value = (torch.randn(1, 2, 5, 8), None, None, None)
            ltx2_transformer._ltx2_native_efficient_attention(query, key, value, attention_mask)

        self.assertEqual(efficient_attention.call_args.args[3].shape, (1, 1, 5, 7))

    def test_prepares_flash_metadata_once_for_masked_keys(self):
        metadata = ltx2_transformer._ltx2_prepare_flash_attention_metadata(
            torch.tensor([[True, False, True], [True, True, False]]),
            batch_size=2,
            seq_len_q=4,
            seq_len_kv=3,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(metadata.indices_k, torch.tensor([0, 2, 3, 4, 1, 5])))
        self.assertTrue(torch.equal(metadata.cu_seqlens_q, torch.tensor([0, 4, 8], dtype=torch.int32)))
        self.assertTrue(torch.equal(metadata.cu_seqlens_k, torch.tensor([0, 2, 4], dtype=torch.int32)))
        self.assertEqual(metadata.max_seqlen_q, 4)
        self.assertEqual(metadata.max_seqlen_k, 3)

    def test_prompt_timesteps_use_batch_axis_from_tokenwise_timesteps(self):
        timesteps = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        prompt_timesteps = ltx2_transformer._ltx2_prompt_timesteps(timesteps, batch_size=2, name="video prompt")

        self.assertTrue(torch.equal(prompt_timesteps, torch.tensor([0.1, 0.4])))


class TestLTX2Checkpointing(unittest.TestCase):
    def test_backward_recomputes_the_original_block_and_attention_flags(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model = ltx2_transformer.LTX2VideoTransformer3DModel(
                in_channels=2,
                out_channels=2,
                num_attention_heads=2,
                attention_head_dim=8,
                cross_attention_dim=16,
                audio_in_channels=4,
                audio_out_channels=4,
                audio_num_attention_heads=2,
                audio_attention_head_dim=4,
                audio_cross_attention_dim=8,
                num_layers=2,
                use_prompt_embeddings=False,
                perturbed_attn=True,
            )
            inputs = {
                "hidden_states": torch.randn(1, 4, 2),
                "audio_hidden_states": torch.randn(1, 2, 4),
                "encoder_hidden_states": torch.randn(1, 3, 16),
                "audio_encoder_hidden_states": torch.randn(1, 3, 8),
                "timestep": torch.tensor([250.0]),
                "num_frames": 1,
                "height": 2,
                "width": 2,
                "audio_num_frames": 2,
                "return_dict": False,
            }
        calls = []

        def run():
            output = model(**inputs)
            sum(tensor.square().mean() for tensor in output).backward()
            gradients = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
            return output, gradients

        with ExitStack() as hooks:
            for index, block in enumerate(model.transformer_blocks):
                hooks.enter_context(
                    block.register_forward_pre_hook(
                        lambda _module, _args, kwargs, index=index: calls.append((index, kwargs["stg_skip_self_attn"])),
                        with_kwargs=True,
                    )
                )
            for skip in ((), (0,)):
                with self.subTest(skipped_attention_blocks=skip):
                    inputs["attention_kwargs"] = {"stg_self_attn_blocks": skip}
                    model.disable_gradient_checkpointing()
                    model.zero_grad(set_to_none=True)
                    calls.clear()
                    eager_output, eager_gradients = run()
                    forward_calls = [(0, 0 in skip), (1, 1 in skip)]
                    self.assertEqual(calls, forward_calls)

                    model.enable_gradient_checkpointing()
                    model.zero_grad(set_to_none=True)
                    calls.clear()
                    checkpoint_output, checkpoint_gradients = run()
                    self.assertEqual(calls, forward_calls + forward_calls[::-1])
                    torch.testing.assert_close(checkpoint_output, eager_output, rtol=0, atol=0)
                    torch.testing.assert_close(checkpoint_gradients, eager_gradients, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
