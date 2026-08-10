import unittest

import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders.lora_base import LoraBaseMixin
from diffusers.loaders.peft import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from peft import LoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer

from simpletuner.helpers.training import diffusers_overrides  # noqa: F401


class TinyPeftModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    config_name = "config.json"

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)


class TwoComponentLoraPipeline(LoraBaseMixin):
    _lora_loadable_modules = ["unet", "text_encoder"]

    def __init__(self, unet, text_encoder):
        self._merged_adapters = set()
        self.unet = unet
        self.text_encoder = text_encoder


class DiffusersLoraOverrideTests(unittest.TestCase):
    def test_partial_unfuse_keeps_adapter_tracked_until_all_components_are_unmerged(self):
        unet = TinyPeftModel()
        text_encoder = TinyPeftModel()
        config = LoraConfig(r=4, lora_alpha=4, target_modules=["linear"], init_lora_weights=False)
        unet.add_adapter(config, adapter_name="adapter")
        text_encoder.add_adapter(config, adapter_name="adapter")

        pipeline = TwoComponentLoraPipeline(unet, text_encoder)
        pipeline.fuse_lora(components=["unet", "text_encoder"], adapter_names=["adapter"])
        self.assertEqual(pipeline.num_fused_loras, 1)

        pipeline.unfuse_lora(components=["text_encoder"])

        unet_still_merged = any(
            isinstance(module, BaseTunerLayer) and bool(module.merged_adapters) for module in unet.modules()
        )
        self.assertTrue(unet_still_merged)
        self.assertIn("adapter", pipeline.fused_loras)
        self.assertEqual(pipeline.num_fused_loras, 1)

        pipeline.unfuse_lora(components=["unet"])
        self.assertEqual(pipeline.fused_loras, set())
        self.assertEqual(pipeline.num_fused_loras, 0)


if __name__ == "__main__":
    unittest.main()
