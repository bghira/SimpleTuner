import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers.training_utils import _collate_lora_metadata
from peft import LoraConfig, get_peft_model_state_dict

from simpletuner.helpers.models.longcat_video.model import LongCatVideo
from simpletuner.helpers.models.longcat_video.pipeline import LongCatVideoPipeline
from simpletuner.helpers.models.longcat_video.transformer import LongCatVideoTransformer3DModel


class LongCatVideoLoraTests(unittest.TestCase):
    def test_final_and_checkpoint_exports_round_trip(self):
        transformer = LongCatVideoTransformer3DModel(
            hidden_size=64,
            depth=1,
            num_heads=8,
            caption_channels=64,
            mlp_ratio=2,
            adaln_tembed_dim=32,
            frequency_embedding_size=16,
        )
        transformer.add_adapter(LoraConfig(r=2, lora_alpha=8, target_modules=["qkv"], init_lora_weights=False))
        adapter_state = get_peft_model_state_dict(transformer)
        self.assertTrue(adapter_state)

        model = LongCatVideo.__new__(LongCatVideo)
        model.config = SimpleNamespace(controlnet=False, lora_format="diffusers")
        final_kwargs = {f"{model.MODEL_TYPE.value}_lora_layers": adapter_state}
        checkpoint_kwargs = {
            f"{model.MODEL_SUBFOLDER}_lora_layers": adapter_state,
            **_collate_lora_metadata({model.MODEL_SUBFOLDER: transformer}),
        }
        for label, kwargs in (("final", final_kwargs), ("checkpoint", checkpoint_kwargs)):
            with self.subTest(export=label), tempfile.TemporaryDirectory() as directory:
                model.save_lora_weights(save_directory=directory, **kwargs)

                self.assertTrue((Path(directory) / "pytorch_lora_weights.safetensors").is_file())
                weights, metadata = LongCatVideoPipeline.lora_state_dict(directory, local_files_only=True)
                if label == "checkpoint":
                    self.assertEqual(metadata["transformer.lora_alpha"], 8)
                reloaded = LongCatVideoTransformer3DModel.from_config(transformer.config)
                LongCatVideoPipeline.load_lora_into_transformer(
                    weights, reloaded, adapter_name="restored", metadata=metadata
                )
                restored = get_peft_model_state_dict(reloaded, adapter_name="restored")
                torch.testing.assert_close(restored, adapter_state)
                if label == "checkpoint":
                    self.assertEqual(reloaded.peft_config["restored"].lora_alpha, 8)
