import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from simpletuner.helpers.models.wan_s2v.model import WanS2V
from simpletuner.helpers.models.wan_s2v.transformer import WanS2VTransformer3DModel


class WanS2VCheckpointLoadingTestCase(unittest.TestCase):
    def _write_legacy_checkpoint(self, root, *, omit_average_weights=False):
        transformer = WanS2VTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            num_layers=1,
            ffn_dim=32,
            text_dim=8,
            audio_dim=8,
            freq_dim=16,
            audio_inject_layers=(0,),
        )
        subfolder = Path(root) / "transformer"
        transformer.save_config(subfolder)
        state_dict = transformer.state_dict()
        legacy_state = {}
        prefix = "condition_embedder.causal_audio_encoder."
        for name, value in state_dict.items():
            legacy_name = name
            if name == f"{prefix}weighted_avg.weights":
                if omit_average_weights:
                    continue
                legacy_name = f"{prefix}weights"
            elif name.startswith(f"{prefix}encoder.conv"):
                legacy_name = name.replace(".conv.conv.", ".conv.")
            legacy_state[legacy_name] = value.contiguous()

        shard_name = "diffusion_pytorch_model-00001-of-00001.safetensors"
        save_file(legacy_state, subfolder / shard_name)
        index = {"metadata": {}, "weight_map": {name: shard_name for name in legacy_state}}
        (subfolder / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index))
        return state_dict

    def _training_model(self, path):
        config = SimpleNamespace(
            model_flavour="s2v-14b-2.2",
            revision=None,
            variant=None,
            pretrained_model_name_or_path=path,
            pretrained_transformer_model_name_or_path=None,
            base_model_precision="no_change",
            weight_dtype=torch.bfloat16,
            fuse_qkv_projections=False,
            gradient_checkpointing_interval=None,
            gradient_checkpointing=False,
        )
        with patch.object(WanS2V, "setup_training_noise_schedule"):
            return WanS2V(config, accelerator=None)

    def test_sharded_legacy_audio_weights_are_loaded_in_fp32(self):
        with TemporaryDirectory() as folder:
            expected = self._write_legacy_checkpoint(folder)
            model = self._training_model(folder)
            model.load_model(move_to_device=False)

        for name, value in model.model.named_parameters():
            self.assertNotEqual(value.device.type, "meta", name)
            if name.startswith("condition_embedder.causal_audio_encoder."):
                torch.testing.assert_close(value, expected[name])

    def test_unresolved_weights_are_reported_by_name(self):
        with TemporaryDirectory() as folder:
            self._write_legacy_checkpoint(folder, omit_average_weights=True)
            model = self._training_model(folder)
            with self.assertRaisesRegex(RuntimeError, r"condition_embedder\.causal_audio_encoder\.weighted_avg\.weights"):
                model.load_model(move_to_device=False)
