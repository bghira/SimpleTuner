import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers import assistant_lora
from simpletuner.helpers.models.common import ModelFoundation
from simpletuner.helpers.models.flux.model import Flux
from simpletuner.helpers.models.z_image.model import ZImage


class _Param:
    def __init__(self):
        self.requires_grad = True

    def requires_grad_(self, flag: bool):
        self.requires_grad = flag
        return self


class _DummyModule:
    def __init__(self, names):
        self._params = {name: _Param() for name in names}
        self.set_adapters_called = None
        self.set_adapter_called = None

    def named_parameters(self):
        return list(self._params.items())

    def set_adapters(self, names, weights=None):
        self.set_adapters_called = (names, weights)

    def set_adapter(self, name):
        self.set_adapter_called = name


class AssistantLoraTests(unittest.TestCase):
    def test_freeze_adapter_parameters_only_targets_adapter(self):
        p1 = _Param()
        p2 = _Param()
        p3 = _Param()
        module = types.SimpleNamespace(
            named_parameters=lambda: [
                ("layer.lora_A.assistant", p1),
                ("layer.lora_A.other", p2),
                ("layer.weight", p3),
            ]
        )

        assistant_lora.freeze_adapter_parameters(module, "assistant")

        self.assertFalse(p1.requires_grad)
        self.assertTrue(p2.requires_grad)
        self.assertTrue(p3.requires_grad)

    def test_set_adapter_stack_prefers_set_adapters_and_freezes(self):
        module = _DummyModule(["block.lora_B.assistant", "block.weight"])

        assistant_lora.set_adapter_stack(
            module,
            adapter_names=["assistant", "default"],
            weights=[0.5, 1.0],
            freeze_names=["assistant"],
        )

        self.assertEqual(module.set_adapters_called, (["assistant", "default"], [0.5, 1.0]))
        adapter_params = dict(module.named_parameters())
        self.assertFalse(adapter_params["block.lora_B.assistant"].requires_grad)

    def test_set_adapter_stack_falls_back_to_set_adapter(self):
        module = _DummyModule(["block.lora_B.assistant"])
        module.set_adapters = None

        assistant_lora.set_adapter_stack(
            module,
            adapter_names=["assistant", "other"],
            weights=[1.0, 0.0],
            freeze_names=["assistant"],
        )

        self.assertEqual(module.set_adapter_called, "assistant")

    def test_load_assistant_adapter_calls_pipeline_loader_and_freezes(self):
        called = {}

        class _Pipeline:
            @classmethod
            def lora_state_dict(cls, path, return_alphas=False, **kwargs):
                called["state_kwargs"] = kwargs
                called["return_alphas"] = return_alphas
                return {"transformer.lora_A": 1}, {"transformer.alpha": 2}

            @classmethod
            def load_lora_into_transformer(cls, state_dict, **kwargs):
                called["load_kwargs"] = kwargs
                called["state_dict"] = state_dict

        class _Transformer:
            def __init__(self):
                self.params = [("layer.lora_A.assistant", _Param())]

            def named_parameters(self):
                return self.params

        transformer = _Transformer()
        loaded = assistant_lora.load_assistant_adapter(
            transformer=transformer,
            pipeline_cls=_Pipeline,
            lora_path="dummy.safetensors",
            adapter_name="assistant",
            low_cpu_mem_usage=True,
        )

        self.assertTrue(loaded)
        self.assertEqual(called["return_alphas"], True)
        self.assertEqual(called["load_kwargs"]["adapter_name"], "assistant")
        self.assertTrue(called["load_kwargs"]["network_alphas"])
        adapter_params = dict(transformer.named_parameters())
        self.assertFalse(adapter_params["layer.lora_A.assistant"].requires_grad)
        self.assertIsNone(called["state_kwargs"].get("weight_name"))


class SupportsAssistantLoraTests(unittest.TestCase):
    def test_supports_assistant_lora_flavour_gate(self):
        class _Model(ModelFoundation):
            ASSISTANT_LORA_PATH = None
            ASSISTANT_LORA_FLAVOURS = ["target"]

            @classmethod
            def _pipeline_has_lora_loader(cls, pipeline_cls) -> bool:
                return False

            # abstract method stubs
            def model_predict(self, prepared_batch, custom_timesteps=None):
                raise NotImplementedError

            def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
                raise NotImplementedError

            def convert_text_embed_for_pipeline(self, text_embedding):
                raise NotImplementedError

            def convert_negative_text_embed_for_pipeline(self, text_embedding):
                raise NotImplementedError

        config = types.SimpleNamespace(model_flavour="target")
        self.assertTrue(_Model.supports_assistant_lora(config))

        config_bad = types.SimpleNamespace(model_flavour="other")
        self.assertFalse(_Model.supports_assistant_lora(config_bad))


class AssistantLoraModelDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.mock_accelerator = MagicMock()
        self.mock_accelerator.device = torch.device("cpu")

    def _build_zimage_config(self, flavour: str):
        config = MagicMock()
        config.model_family = "z-image"
        config.model_flavour = flavour
        config.model_type = "lora"
        config.assistant_lora_path = None
        config.disable_assistant_lora = False
        config.assistant_lora_weight_name = None
        config.weight_dtype = torch.float32
        config.pretrained_model_name_or_path = "TONGYI-MAI/Z-Image-Turbo"
        config.revision = None
        config.variant = None
        config.vae_path = None
        config.controlnet = False
        config.pretrained_transformer_model_name_or_path = None
        config.pretrained_unet_model_name_or_path = None
        config.pretrained_transformer_subfolder = None
        config.pretrained_unet_subfolder = None
        return config

    def test_flux_schnell_sets_default_assistant_path(self):
        config = MagicMock()
        config.model_family = "flux"
        config.model_flavour = "schnell"
        config.model_type = "lora"
        config.flux_fast_schedule = True
        config.i_know_what_i_am_doing = False
        config.validation_num_inference_steps = 4
        config.validation_guidance_real = 0.0
        config.flux_attention_masked_training = False
        config.fuse_qkv_projections = False
        config.aspect_bucket_alignment = 64
        config.unet_attention_slice = False
        config.prediction_type = None
        config.tokenizer_max_length = None
        config.assistant_lora_path = None
        config.disable_assistant_lora = False
        config.assistant_lora_weight_name = None
        config.weight_dtype = torch.float32
        config.pretrained_model_name_or_path = "black-forest-labs/flux.1-schnell"

        with (
            patch.object(Flux, "setup_model_flavour", lambda self: None),
            patch.object(Flux, "setup_training_noise_schedule", lambda self: None),
        ):
            model = Flux(config, self.mock_accelerator)
        model.check_user_config()
        self.assertEqual(config.assistant_lora_path, Flux.ASSISTANT_LORA_PATH)

    def test_zimage_turbo_requires_assistant_path(self):
        config = self._build_zimage_config("turbo")

        with (
            patch.object(ZImage, "setup_model_flavour", lambda self: None),
            patch.object(ZImage, "setup_training_noise_schedule", lambda self: None),
        ):
            model = ZImage(config, self.mock_accelerator)
        model.check_user_config()
        self.assertEqual(config.assistant_lora_path, ZImage.ASSISTANT_LORA_PATH)
        self.assertEqual(config.assistant_lora_weight_name, ZImage.ASSISTANT_LORA_WEIGHT_NAME)

    def test_zimage_turbo_v2_sets_assistant_defaults(self):
        config = self._build_zimage_config("turbo-ostris-v2")

        with (
            patch.object(ZImage, "setup_model_flavour", lambda self: None),
            patch.object(ZImage, "setup_training_noise_schedule", lambda self: None),
        ):
            model = ZImage(config, self.mock_accelerator)
        model.check_user_config()
        self.assertEqual(config.assistant_lora_path, ZImage.ASSISTANT_LORA_PATH)
        self.assertEqual(config.assistant_lora_weight_name, ZImage.ASSISTANT_LORA_WEIGHT_NAMES["turbo-ostris-v2"])

    def test_disable_flag_skips_requirements(self):
        config = MagicMock()
        config.model_family = "z-image"
        config.model_flavour = "turbo"
        config.model_type = "lora"
        config.assistant_lora_path = None
        config.disable_assistant_lora = True
        config.assistant_lora_weight_name = None
        config.weight_dtype = torch.float32
        config.pretrained_model_name_or_path = "TONGYI-MAI/Z-Image-Turbo"
        config.revision = None
        config.variant = None
        config.vae_path = None
        config.controlnet = False
        config.pretrained_transformer_model_name_or_path = None
        config.pretrained_unet_model_name_or_path = None
        config.pretrained_transformer_subfolder = None
        config.pretrained_unet_subfolder = None

        with (
            patch.object(ZImage, "setup_model_flavour", lambda self: None),
            patch.object(ZImage, "setup_training_noise_schedule", lambda self: None),
        ):
            model = ZImage(config, self.mock_accelerator)
        # Should not raise even without assistant_lora_path when disabled.
        model.check_user_config()

    def test_zimage_de_turbo_overrides_transformer_path(self):
        config = self._build_zimage_config("ostris-de-turbo")

        with (
            patch.object(ZImage, "setup_training_noise_schedule", lambda self: None),
            patch.object(ZImage, "setup_diff2flow_bridge", lambda self: None),
        ):
            ZImage(config, self.mock_accelerator)

        self.assertEqual(config.pretrained_model_name_or_path, ZImage.HUGGINGFACE_PATHS["ostris-de-turbo"])
        self.assertEqual(
            config.pretrained_transformer_model_name_or_path, ZImage.TRANSFORMER_PATH_OVERRIDES["ostris-de-turbo"]
        )


class ZImageAdapterMappingTests(unittest.TestCase):
    def test_set_adapters_mapping_registered(self):
        from diffusers.loaders import peft as diffusers_peft

        import simpletuner.helpers.models.z_image.transformer  # noqa: F401

        self.assertIn("ZImageTransformer2DModel", diffusers_peft._SET_ADAPTER_SCALE_FN_MAPPING)
        fn = diffusers_peft._SET_ADAPTER_SCALE_FN_MAPPING["ZImageTransformer2DModel"]
        self.assertEqual([1.0, 0.5], fn(None, [1.0, 0.5]))


if __name__ == "__main__":
    unittest.main()
