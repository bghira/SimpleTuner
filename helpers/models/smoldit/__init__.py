from helpers.models.smoldit.transformer import SmolDiT2DModel
from helpers.models.smoldit.pipeline import SmolDiTPipeline

SmolDiTConfigurations = {
    "smoldit-small": {
        "sample_size": 64,
        "num_layers": 18,
        "patch_size": 2,
        "attention_head_dim": 64,
        "num_attention_heads": 16,
        "num_kv_heads": 4,
        "in_channels": 4,
        "cross_attention_dim": 768,
        "out_channels": 4,
        "activation_fn": "gelu-approximate",
    },
    "smoldit-swiglu": {
        "sample_size": 64,
        "num_layers": 24,
        "patch_size": 2,
        "attention_head_dim": 72,
        "num_attention_heads": 16,
        "num_kv_heads": 4,
        "in_channels": 4,
        "cross_attention_dim": 768,
        "out_channels": 4,
        "activation_fn": "swiglu",
    },
    "smoldit-base": {
        "sample_size": 64,
        "num_layers": 24,
        "patch_size": 2,
        "attention_head_dim": 72,
        "num_attention_heads": 16,
        "num_kv_heads": 4,
        "in_channels": 4,
        "cross_attention_dim": 768,
        "out_channels": 4,
        "activation_fn": "gelu-approximate",
    },
    "smoldit-large": {
        "sample_size": 64,
        "num_layers": 30,
        "patch_size": 2,
        "attention_head_dim": 72,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "in_channels": 4,
        "cross_attention_dim": 768,
        "out_channels": 4,
        "activation_fn": "gelu-approximate",
    },
    "smoldit-huge": {
        "sample_size": 64,
        "num_layers": 36,
        "patch_size": 2,
        "attention_head_dim": 96,
        "num_attention_heads": 64,
        "num_kv_heads": 16,
        "in_channels": 4,
        "cross_attention_dim": 768,
        "out_channels": 4,
        "activation_fn": "gelu-approximate",
    },
}
SmolDiTConfigurationNames = list(SmolDiTConfigurations.keys())


def get_resize_crop_region_for_grid(src, tgt_size):
    th = tw = tgt_size
    h, w = src

    r = h / w

    # resize
    if r > 1:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)
