import unittest

from simpletuner.helpers.models.longcat_image.transformer import LongCatImageTransformer2DModel
from simpletuner.helpers.models.qwen_image.transformer import QwenImageTransformer2DModel
from simpletuner.helpers.models.sana.transformer import SanaTransformer2DModel
from simpletuner.helpers.models.sanavideo.transformer import SanaVideoTransformer3DModel


def _state_dict_has_time_sign_embed(model) -> bool:
    return any(key.endswith("time_sign_embed.weight") for key in model.state_dict().keys())


def _tiny_sana_transformer(enable_time_sign_embed: bool = False) -> SanaTransformer2DModel:
    return SanaTransformer2DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=1,
        num_cross_attention_heads=2,
        cross_attention_head_dim=4,
        cross_attention_dim=8,
        caption_channels=8,
        mlp_ratio=2.0,
        dropout=0.0,
        attention_bias=False,
        sample_size=8,
        patch_size=1,
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        interpolation_scale=None,
        enable_time_sign_embed=enable_time_sign_embed,
    )


def _tiny_longcat_image_transformer(enable_time_sign_embed: bool = False) -> LongCatImageTransformer2DModel:
    return LongCatImageTransformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        pooled_projection_dim=16,
        axes_dims_rope=[2, 2, 4],
        enable_time_sign_embed=enable_time_sign_embed,
    )


def _tiny_qwen_image_transformer(enable_time_sign_embed: bool = False) -> QwenImageTransformer2DModel:
    return QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        axes_dims_rope=(2, 2, 4),
        enable_time_sign_embed=enable_time_sign_embed,
    )


def _tiny_sanavideo_transformer(
    guidance_embeds: bool,
    enable_time_sign_embed: bool = False,
) -> SanaVideoTransformer3DModel:
    return SanaVideoTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=1,
        num_cross_attention_heads=2,
        cross_attention_head_dim=4,
        cross_attention_dim=8,
        caption_channels=8,
        mlp_ratio=2.0,
        dropout=0.0,
        attention_bias=False,
        sample_size=2,
        patch_size=(1, 2, 2),
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        interpolation_scale=None,
        guidance_embeds=guidance_embeds,
        rope_max_seq_len=64,
        enable_time_sign_embed=enable_time_sign_embed,
    )


class TimeSignEmbedGatingAcrossModelsTest(unittest.TestCase):
    def test_sana_time_sign_embed_gated(self):
        self.assertFalse(_state_dict_has_time_sign_embed(_tiny_sana_transformer()))
        self.assertTrue(_state_dict_has_time_sign_embed(_tiny_sana_transformer(enable_time_sign_embed=True)))

    def test_longcat_image_time_sign_embed_gated(self):
        self.assertFalse(_state_dict_has_time_sign_embed(_tiny_longcat_image_transformer()))
        self.assertTrue(_state_dict_has_time_sign_embed(_tiny_longcat_image_transformer(enable_time_sign_embed=True)))

    def test_qwen_image_time_sign_embed_gated(self):
        self.assertFalse(_state_dict_has_time_sign_embed(_tiny_qwen_image_transformer()))
        self.assertTrue(_state_dict_has_time_sign_embed(_tiny_qwen_image_transformer(enable_time_sign_embed=True)))

    def test_sanavideo_time_sign_embed_gated_without_guidance(self):
        self.assertFalse(_state_dict_has_time_sign_embed(_tiny_sanavideo_transformer(guidance_embeds=False)))
        self.assertTrue(
            _state_dict_has_time_sign_embed(_tiny_sanavideo_transformer(guidance_embeds=False, enable_time_sign_embed=True))
        )

    def test_sanavideo_time_sign_embed_gated_with_guidance(self):
        self.assertFalse(_state_dict_has_time_sign_embed(_tiny_sanavideo_transformer(guidance_embeds=True)))
        self.assertTrue(
            _state_dict_has_time_sign_embed(_tiny_sanavideo_transformer(guidance_embeds=True, enable_time_sign_embed=True))
        )


if __name__ == "__main__":
    unittest.main()
