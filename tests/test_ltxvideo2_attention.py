import unittest
from unittest.mock import patch

import torch
from diffusers.models.attention_dispatch import AttentionBackendName

from simpletuner.helpers.models.ltxvideo2 import transformer as ltx2_transformer


class TestLTX2AttentionDispatch(unittest.TestCase):
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

    def test_prompt_timesteps_use_batch_axis_from_tokenwise_timesteps(self):
        timesteps = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        prompt_timesteps = ltx2_transformer._ltx2_prompt_timesteps(timesteps, batch_size=2, name="video prompt")

        self.assertTrue(torch.equal(prompt_timesteps, torch.tensor([0.1, 0.4])))


if __name__ == "__main__":
    unittest.main()
