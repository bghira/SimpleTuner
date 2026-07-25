import inspect
import json
import unittest

import torch

from simpletuner.helpers.models.z_image.model import ZImage
from simpletuner.helpers.models.z_image.quantized_loading import _decode_comfy_quant, _map_zimage_comfy_key_to_diffusers
from simpletuner.helpers.models.z_image.transformer import ZImageTransformer2DModel


class ZImageTransformerPaddingTests(unittest.TestCase):
    def test_transformer_exposes_single_file_loader(self):
        parameters = inspect.signature(ZImageTransformer2DModel.from_single_file).parameters

        self.assertIn("pretrained_model_link_or_path", parameters)
        self.assertIn("filename", parameters)
        self.assertIn("torch_dtype", parameters)

    def test_convrot_comfy_key_mapping(self):
        self.assertEqual(
            _map_zimage_comfy_key_to_diffusers("x_embedder.weight"),
            ["all_x_embedder.2-1.weight"],
        )
        self.assertEqual(
            _map_zimage_comfy_key_to_diffusers("final_layer.linear.weight"),
            ["all_final_layer.2-1.linear.weight"],
        )
        self.assertEqual(
            _map_zimage_comfy_key_to_diffusers("layers.3.attention.q_norm.weight"),
            ["layers.3.attention.norm_q.weight"],
        )
        self.assertEqual(
            _map_zimage_comfy_key_to_diffusers("layers.3.attention.out.weight"),
            ["layers.3.attention.to_out.0.weight"],
        )
        self.assertEqual(
            _map_zimage_comfy_key_to_diffusers("layers.3.attention.qkv.weight"),
            [
                "layers.3.attention.to_q.weight",
                "layers.3.attention.to_k.weight",
                "layers.3.attention.to_v.weight",
            ],
        )

    def test_convrot_comfy_quant_metadata_decodes(self):
        metadata = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64, "per_row": True}
        json_bytes = json.dumps(metadata).encode("utf-8")
        payload = torch.tensor(list(json_bytes), dtype=torch.uint8)

        self.assertEqual(_decode_comfy_quant(payload), metadata)

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

    def test_context_parallel_only_marks_unified_transformer_blocks(self):
        model = ZImageTransformer2DModel(
            all_patch_size=(2,),
            all_f_patch_size=(1,),
            in_channels=1,
            dim=4,
            n_layers=2,
            n_refiner_layers=1,
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

        self.assertEqual(model._cp_plan, {})
        for block in model.noise_refiner:
            self.assertFalse(getattr(block.attention, "_zimage_allow_context_parallel", False))
        for block in model.context_refiner:
            self.assertFalse(getattr(block.attention, "_zimage_allow_context_parallel", False))
        for block in model.layers:
            self.assertTrue(block.attention._zimage_allow_context_parallel)

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

    def test_transformer_accepts_tokenwise_timesteps(self):
        model = ZImageTransformer2DModel(
            all_patch_size=(2,),
            all_f_patch_size=(1,),
            in_channels=1,
            dim=8,
            n_layers=1,
            n_refiner_layers=1,
            n_heads=1,
            n_kv_heads=1,
            norm_eps=1e-5,
            qk_norm=False,
            cap_feat_dim=4,
            rope_theta=1.0,
            t_scale=1.0,
            axes_dims=[2, 2, 4],
            axes_lens=[64, 64, 64],
        )

        x = [torch.zeros(1, 1, 8, 8)]
        cap_feats = [torch.zeros(2, 4)]
        t = torch.full((1, 16), 0.5)

        output = model(x, t, cap_feats)[0]

        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].shape, (1, 1, 8, 8))

    def test_transformer_rejects_wrong_tokenwise_timestep_length(self):
        model = ZImageTransformer2DModel(
            all_patch_size=(2,),
            all_f_patch_size=(1,),
            in_channels=1,
            dim=8,
            n_layers=1,
            n_refiner_layers=1,
            n_heads=1,
            n_kv_heads=1,
            norm_eps=1e-5,
            qk_norm=False,
            cap_feat_dim=4,
            rope_theta=1.0,
            t_scale=1.0,
            axes_dims=[2, 2, 4],
            axes_lens=[64, 64, 64],
        )

        with self.assertRaisesRegex(ValueError, "tokenwise timesteps expected shape"):
            model([torch.zeros(1, 1, 8, 8)], torch.full((1, 2), 0.5), [torch.zeros(2, 4)])[0]


if __name__ == "__main__":
    unittest.main()
