from helpers.models.smoldit.transformer import SmolDiT2DModel
from helpers.models.smoldit.pipeline import SmolDiTPipeline

SmolDiTDefaultConfig = {
    "sample_size": 16,
    "num_layers": 2,
    "patch_size": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 4,
    "num_kv_heads": 2,
    "in_channels": 4,
    "cross_attention_dim": 768,
    "out_channels": 4,
    "activation_fn": "gelu-approximate",
}


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
