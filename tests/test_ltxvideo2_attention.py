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


if __name__ == "__main__":
    unittest.main()
