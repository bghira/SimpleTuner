import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from simpletuner.helpers.acceleration import AccelerationBackend
from simpletuner.helpers.models.common import PipelineTypes
from simpletuner.helpers.models.mageflow.pipeline_edit import MageFlowEditPipeline
from simpletuner.helpers.models.mageflow.transformer import MageFlowTransformer2DModel
from simpletuner.helpers.models.registry import ModelRegistry


def _mageflow_class():
    registry_entry = ModelRegistry.get("mageflow")
    return registry_entry.get_real_class() if hasattr(registry_entry, "get_real_class") else registry_entry


class MageFlowModelTests(unittest.TestCase):
    def test_model_metadata_contains_mageflow(self):
        metadata_path = Path(__file__).parent.parent / "simpletuner/helpers/models/model_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        self.assertIn("mageflow", metadata)
        self.assertEqual(metadata["mageflow"]["class_name"], "MageFlow")
        self.assertEqual(
            metadata["mageflow"]["flavour_choices"],
            ["base", "default", "turbo", "edit-base", "edit", "edit-turbo"],
        )

    def test_registry_resolves_mageflow(self):
        model_cls = _mageflow_class()
        self.assertEqual(model_cls.NAME, "Mage-Flow")
        self.assertEqual(model_cls.DEFAULT_MODEL_FLAVOUR, "base")
        self.assertIn("edit-turbo", model_cls.get_flavour_choices())

    def test_acceleration_presets_include_ramtorch_and_musubi_levels(self):
        model_cls = _mageflow_class()
        presets = model_cls.get_acceleration_presets()
        ramtorch_levels = {preset.level for preset in presets if preset.backend is AccelerationBackend.RAMTORCH}
        musubi_levels = {preset.level for preset in presets if preset.backend is AccelerationBackend.MUSUBI_BLOCK_SWAP}

        self.assertEqual(ramtorch_levels, {"light", "balanced", "aggressive"})
        self.assertEqual(musubi_levels, {"light", "balanced", "aggressive"})

    def test_pretrained_load_args_include_musubi_settings(self):
        model_cls = _mageflow_class()
        model = object.__new__(model_cls)
        model.config = SimpleNamespace(
            attention_mechanism="sdpa",
            musubi_blocks_to_swap=6,
            musubi_block_swap_device="cpu",
            twinflow_enabled=False,
        )

        args = model.pretrained_load_args({})

        self.assertEqual(args["attn_type"], "sdpa")
        self.assertEqual(args["musubi_blocks_to_swap"], 6)
        self.assertEqual(args["musubi_block_swap_device"], "cpu")

    def test_edit_flavour_uses_edit_pipeline(self):
        model_cls = _mageflow_class()
        config = SimpleNamespace(
            model_flavour="edit-turbo",
            aspect_bucket_alignment=16,
            tokenizer_max_length=None,
            validation_using_datasets=False,
            validation_num_inference_steps=4,
        )
        model = object.__new__(model_cls)
        model.config = config
        model.PIPELINE_CLASSES = dict(model_cls.PIPELINE_CLASSES)
        model.check_user_config()
        self.assertIs(model.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG], MageFlowEditPipeline)
        self.assertTrue(model.requires_conditioning_dataset())
        self.assertTrue(model.requires_conditioning_validation_inputs())

    def test_small_transformer_instantiates(self):
        model = MageFlowTransformer2DModel(
            hidden_size=24,
            num_heads=2,
            axes_dim=[4, 4, 4],
            depth=1,
            context_in_dim=8,
        )
        self.assertEqual(model.config.in_channels, 128)
        self.assertEqual(len(model.transformer_blocks), 1)

    def test_gradient_checkpointing_uses_supplied_function(self):
        checkpoint_calls = []

        def checkpoint_func(function, *args, **kwargs):
            checkpoint_calls.append(dict(kwargs))
            kwargs.pop("use_reentrant", None)
            return function(*args)

        model = MageFlowTransformer2DModel(
            in_channels=4,
            out_channels=4,
            hidden_size=24,
            num_heads=2,
            axes_dim=[4, 4, 4],
            depth=1,
            context_in_dim=8,
            attn_type="sdpa",
        )
        model.train()
        model.enable_gradient_checkpointing(checkpoint_func)

        output = model(
            img=torch.randn(1, 4, 4),
            txt=torch.randn(1, 3, 8),
            timesteps=torch.tensor([0.5]),
            img_shapes=[[(1, 2, 2)]],
            img_cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
            txt_cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
        )

        self.assertEqual(output.shape, (1, 4, 4))
        self.assertEqual(len(checkpoint_calls), 1)
        self.assertEqual(checkpoint_calls[0], {"use_reentrant": False})

    def test_transformer_has_musubi_constructor_args_and_cpu_forward(self):
        model = MageFlowTransformer2DModel(
            in_channels=4,
            out_channels=4,
            hidden_size=24,
            num_heads=2,
            axes_dim=[4, 4, 4],
            depth=2,
            context_in_dim=8,
            attn_type="sdpa",
            musubi_blocks_to_swap=1,
            musubi_block_swap_device="cpu",
        )

        self.assertIsNotNone(model._musubi_block_swap)
        self.assertTrue(model._musubi_block_swap.is_managed_block(1))
        output = model(
            img=torch.randn(1, 4, 4),
            txt=torch.randn(1, 3, 8),
            timesteps=torch.tensor([0.5]),
            img_shapes=[[(1, 2, 2)]],
            img_cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
            txt_cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
        )

        self.assertEqual(output.shape, (1, 4, 4))


if __name__ == "__main__":
    unittest.main()
