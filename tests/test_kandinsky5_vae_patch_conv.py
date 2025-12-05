import unittest

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders import autoencoder_kl_hunyuan_video as hv_mod

from simpletuner.helpers.models.kandinsky5_video.model import _patch_diffusers_hunyuanvideo_conv


class TestKandinsky5VaePatchConv(unittest.TestCase):
    def test_patch_applies_and_preserves_output(self):
        orig_forward = hv_mod.HunyuanVideoCausalConv3d.forward
        orig_flag = getattr(hv_mod, "_st_patch_conv_applied", False)
        try:
            cls = _patch_diffusers_hunyuanvideo_conv(memory_limit=8)
            self.assertIs(cls, hv_mod.HunyuanVideoCausalConv3d)
            self.assertTrue(getattr(hv_mod, "_st_patch_conv_applied", False))

            conv = hv_mod.HunyuanVideoCausalConv3d(1, 1, kernel_size=1, bias=False)
            sample = torch.arange(1 * 1 * 4 * 2 * 2, dtype=torch.float32).view(1, 1, 4, 2, 2)
            out_patched = conv(sample)
            expected = conv.conv(F.pad(sample, conv.time_causal_padding, mode=conv.pad_mode))
            self.assertTrue(torch.allclose(out_patched, expected))
        finally:
            hv_mod.HunyuanVideoCausalConv3d.forward = orig_forward
            if not orig_flag and hasattr(hv_mod, "_st_patch_conv_applied"):
                delattr(hv_mod, "_st_patch_conv_applied")


if __name__ == "__main__":
    unittest.main()
