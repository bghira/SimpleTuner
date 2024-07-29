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
