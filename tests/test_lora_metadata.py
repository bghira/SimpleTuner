import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.common import PipelineTypes
from simpletuner.helpers.training.save_hooks import SaveHookManager


class _DummyAccelerator:
    is_main_process = True

    def unwrap_model(self, model):
        return model


class _DummyBaseModule:
    def __init__(self, name):
        self.name = name
        self.peft_config = {
            "default": SimpleNamespace(to_dict=lambda: {"name": name}),
        }


class _DummyTrainedComponent(_DummyBaseModule):
    pass


class _DummyControlNet(_DummyTrainedComponent):
    pass


class _DummyTextEncoder(_DummyBaseModule):
    pass


class _DummyModel:
    MODEL_CLASS = _DummyTrainedComponent
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: object(),
        PipelineTypes.IMG2IMG: object(),
        PipelineTypes.CONTROLNET: object(),
    }

    def __init__(self, trained_component, text_encoder=None):
        self._trained_component = trained_component
        self._text_encoders = {0: text_encoder} if text_encoder is not None else {}
        self.save_lora_weights = MagicMock()

    def get_trained_component(self, unwrap_model=False):
        return self._trained_component

    def get_text_encoder(self, index: int):
        return self._text_encoders.get(index)


_ema_stub = SimpleNamespace(store=lambda *args, **kwargs: None, copy_to=lambda *args, **kwargs: None, restore=lambda *args, **kwargs: None)


class SaveHookMetadataTests(unittest.TestCase):
    def _make_manager(self, args_overrides=None, text_encoder=None):
        args = {
            "use_ema": False,
            "model_type": "lora",
            "lora_type": "standard",
            "controlnet": False,
            "validation_using_datasets": False,
        }
        if args_overrides:
            args.update(args_overrides)
        args = SimpleNamespace(**args)

        trained_component = _DummyTrainedComponent("transformer")
        model = _DummyModel(trained_component=trained_component, text_encoder=text_encoder)

        accelerator = _DummyAccelerator()
        manager = SaveHookManager(
            args=args,
            model=model,
            ema_model=_ema_stub,
            accelerator=accelerator,
            use_deepspeed_optimizer=False,
        )
        return manager, model, trained_component

    def test_save_hook_collects_metadata_for_transformer_and_text_encoder(self):
        text_encoder = _DummyTextEncoder("text_encoder")
        manager, model, trained_component = self._make_manager(text_encoder=text_encoder)

        lora_state = {"weight": torch.tensor([1.0])}
        with patch("simpletuner.helpers.training.save_hooks.get_peft_model_state_dict", return_value=lora_state), patch(
            "simpletuner.helpers.training.save_hooks.convert_state_dict_to_diffusers", side_effect=lambda sd, original_type=None: sd
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                models = [trained_component, text_encoder]
                weights = [object() for _ in models]
                manager._save_lora(models=models, weights=weights, output_dir=tmpdir)

        call_args = model.save_lora_weights.call_args
        self.assertIsNotNone(call_args, "save_lora_weights was not invoked")
        kwargs = call_args.kwargs

        self.assertIn("transformer_lora_adapter_metadata", kwargs)
        self.assertEqual(kwargs["transformer_lora_adapter_metadata"], {"name": "transformer"})

        self.assertIn("text_encoder_lora_adapter_metadata", kwargs)
        self.assertEqual(kwargs["text_encoder_lora_adapter_metadata"], {"name": "text_encoder"})

        self.assertNotIn("text_encoder_2_lora_adapter_metadata", kwargs)
        self.assertIn("transformer_lora_layers", kwargs)

    def test_save_hook_handles_controlnet_metadata(self):
        text_encoder = _DummyTextEncoder("text_encoder")
        manager, model, trained_component = self._make_manager(args_overrides={"controlnet": True}, text_encoder=text_encoder)
        controlnet_module = _DummyControlNet("controlnet")

        lora_state = {"weight": torch.tensor([2.0])}
        with patch("simpletuner.helpers.training.save_hooks.get_peft_model_state_dict", return_value=lora_state), patch(
            "simpletuner.helpers.training.save_hooks.convert_state_dict_to_diffusers", side_effect=lambda sd, original_type=None: sd
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                models = [controlnet_module, text_encoder]
                weights = [object() for _ in models]
                manager._save_lora(models=models, weights=weights, output_dir=tmpdir)

        kwargs = model.save_lora_weights.call_args.kwargs
        self.assertIn("controlnet_lora_adapter_metadata", kwargs)
        self.assertEqual(kwargs["controlnet_lora_adapter_metadata"], {"name": "controlnet"})
        self.assertNotIn("transformer_lora_adapter_metadata", kwargs)
        self.assertIn("text_encoder_lora_adapter_metadata", kwargs)


class FluxPipelineMetadataTests(unittest.TestCase):
    def test_flux_pipeline_save_lora_weights_injects_metadata(self):
        from simpletuner.helpers.models.flux.pipeline import FluxPipeline

        def fake_pack(data, prefix):
            return {f"{prefix}.{key}": value for key, value in data.items()}

        transformer_weights = {"lora_down.weight": torch.tensor([3.0])}
        adapter_metadata = {"r": 4, "lora_alpha": 8}

        with patch.object(FluxPipeline, "pack_weights", side_effect=fake_pack) as mock_pack, patch.object(
            FluxPipeline, "write_lora_layers"
        ) as mock_write:
            FluxPipeline.save_lora_weights(
                save_directory="/tmp/out",
                transformer_lora_layers=transformer_weights,
                transformer_lora_adapter_metadata=adapter_metadata,
            )

        mock_pack.assert_called()
        self.assertTrue(mock_write.called, "write_lora_layers should be invoked")

        _, kwargs = mock_write.call_args
        self.assertIn("state_dict", kwargs)
        self.assertIs(kwargs["state_dict"]["transformer.lora_down.weight"], transformer_weights["lora_down.weight"])

        self.assertIn("lora_adapter_metadata", kwargs)
        self.assertEqual(kwargs["lora_adapter_metadata"]["transformer.r"], 4)
        self.assertEqual(kwargs["lora_adapter_metadata"]["transformer.lora_alpha"], 8)


if __name__ == "__main__":
    unittest.main()
