import types
import unittest

import torch

from simpletuner.helpers.models.z_image.model import ZImage
from simpletuner.helpers.models.z_image.transformer import ZImageTransformer2DModel


class ZImageLoadArgsTests(unittest.TestCase):
    def test_pretrained_load_args_defaults_low_cpu_mem_usage_false(self):
        dummy = ZImage.__new__(ZImage)
        dummy.config = types.SimpleNamespace(low_cpu_mem_usage=False)

        args = dummy.pretrained_load_args({"foo": "bar"})

        self.assertIn("low_cpu_mem_usage", args)
        self.assertFalse(args["low_cpu_mem_usage"])

    def test_pretrained_load_args_respects_true(self):
        dummy = ZImage.__new__(ZImage)
        dummy.config = types.SimpleNamespace(low_cpu_mem_usage=True)

        args = dummy.pretrained_load_args({})

        self.assertTrue(args["low_cpu_mem_usage"])


class ZImageTransformerPaddingTests(unittest.TestCase):
    def test_patchify_handles_zero_padding_len(self):
        model = ZImageTransformer2DModel(
            all_patch_size=(2,),
            all_f_patch_size=(1,),
            in_channels=1,
            dim=4,
            n_layers=1,
            n_refiner_layers=0,
            n_heads=1,
            n_kv_heads=1,
            norm_eps=1e-5,
            qk_norm=False,
            cap_feat_dim=4,
            rope_theta=1.0,
            t_scale=1.0,
            axes_dims=[4],
            axes_lens=[1],
        )

        # 32 image tokens => padding length should be 0
        image = torch.zeros((1, 4, 4, 8))
        cap_feats = torch.zeros((1, 1, 4))

        (
            all_image_out,
            _,
            _,
            _,
            _,
            all_image_pad_mask,
            _,
        ) = model.patchify_and_embed(
            [image],
            [cap_feats[0]],
            patch_size=2,
            f_patch_size=1,
        )

        self.assertEqual(all_image_out[0].shape[0], 32)
        self.assertEqual(all_image_pad_mask[0].shape[0], 32)
        self.assertFalse(all_image_pad_mask[0].any())

    def test_gradient_checkpointing_flag_exists(self):
        model = ZImageTransformer2DModel(
            all_patch_size=(2,),
            all_f_patch_size=(1,),
            in_channels=1,
            dim=4,
            n_layers=1,
            n_refiner_layers=0,
            n_heads=1,
            n_kv_heads=1,
            norm_eps=1e-5,
            qk_norm=False,
            cap_feat_dim=4,
            rope_theta=1.0,
            t_scale=1.0,
            axes_dims=[4],
            axes_lens=[1],
        )

        # Should not raise
        model._set_gradient_checkpointing(enable=True)
        self.assertTrue(model.gradient_checkpointing)

    def test_mask_flattening_for_prompt_embeds(self):
        # Ensure 2D attention masks are flattened when selecting prompt embeddings
        zimage = ZImage.__new__(ZImage)
        prompt_embeds = torch.randn(1, 4, 3)
        attention_mask = torch.tensor([[1, 0, 1, 0]])

        out = zimage.convert_text_embed_for_pipeline({"prompt_embeds": prompt_embeds, "attention_mask": attention_mask})

        self.assertEqual(len(out["prompt_embeds"]), 1)
        self.assertEqual(out["prompt_embeds"][0].shape[0], 2)

    def test_mask_flattening_for_negative_prompt_embeds(self):
        zimage = ZImage.__new__(ZImage)
        prompt_embeds = torch.randn(1, 4, 3)
        attention_mask = torch.tensor([[1, 0, 0, 1]])

        out = zimage.convert_negative_text_embed_for_pipeline(
            {"prompt_embeds": prompt_embeds, "attention_mask": attention_mask}
        )

        self.assertEqual(len(out["negative_prompt_embeds"]), 1)
        self.assertEqual(out["negative_prompt_embeds"][0].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
