import unittest
from unittest import mock

import torch

from simpletuner.helpers.models.ideogram.constants import OUTPUT_IMAGE_INDICATOR
from simpletuner.helpers.models.ideogram.quantized_loading import is_fp8_state_dict, quantize_weight_to_fp8

CONDITIONAL_INDEX = "transformer/diffusion_pytorch_model.safetensors.index.json"
UNCONDITIONAL_INDEX = "unconditional_transformer/diffusion_pytorch_model.safetensors.index.json"


def _make_model(
    load_unconditional: bool = False,
    uncond_ramtorch: bool = False,
    upcast: bool = False,
    base_model_precision: str = "no_change",
):
    from simpletuner.helpers.models.ideogram.model import Ideogram4

    model = Ideogram4.__new__(Ideogram4)
    model.config = mock.Mock()
    model.config.pretrained_transformer_model_name_or_path = None
    model.config.pretrained_model_name_or_path = "ideogram-ai/ideogram-4-fp8"
    model.config.ideogram_fp8_base_upcast = upcast
    model.config.ideogram_load_unconditional_transformer = load_unconditional
    model.config.ideogram_uncond_ramtorch = uncond_ramtorch
    model.config.base_model_precision = base_model_precision
    model.config.weight_dtype = torch.bfloat16
    model.accelerator = mock.Mock()
    model.accelerator.device = torch.device("cpu")
    model.apply_gradient_checkpointing_settings = mock.Mock()
    model.apply_model_specific_freeze = mock.Mock()
    return model


def _fp8_state_dict():
    weight, scale = quantize_weight_to_fp8(torch.randn(3, 4))
    return {"proj.weight": weight, "proj.weight_scale": scale}


class IdeogramUnconditionalLoadTests(unittest.TestCase):
    def test_flag_off_leaves_unconditional_transformer_none(self):
        from simpletuner.helpers.models.ideogram import model as ideogram_model

        model = _make_model(load_unconditional=False)
        with mock.patch.object(ideogram_model, "_load_indexed_or_single_state_dict", return_value=_fp8_state_dict()):
            with mock.patch.object(ideogram_model, "_build_transformer", return_value=mock.Mock()) as build:
                model.load_model(move_to_device=False)

        self.assertIsNone(model.unconditional_transformer)
        self.assertEqual(build.call_count, 1)

    def test_flag_on_loads_frozen_unconditional_transformer(self):
        from simpletuner.helpers.models.ideogram import model as ideogram_model

        model = _make_model(load_unconditional=True)
        uncond = torch.nn.Linear(2, 2)
        uncond.train()
        with mock.patch.object(
            ideogram_model, "_load_indexed_or_single_state_dict", return_value=_fp8_state_dict()
        ) as load_sd:
            with mock.patch.object(ideogram_model, "_build_transformer", side_effect=[mock.Mock(), uncond]) as build:
                model.load_model(move_to_device=False)

        self.assertEqual(build.call_count, 2)
        requested = [call.args[1] for call in load_sd.call_args_list]
        self.assertEqual(requested, [CONDITIONAL_INDEX, UNCONDITIONAL_INDEX])
        self.assertIs(model.unconditional_transformer, uncond)
        self.assertFalse(uncond.training)
        self.assertTrue(all(not param.requires_grad for param in uncond.parameters()))

    def test_uncond_load_respects_fp8_base_upcast(self):
        from simpletuner.helpers.models.ideogram import model as ideogram_model

        model = _make_model(load_unconditional=True, upcast=True)
        with mock.patch.object(
            ideogram_model, "_load_indexed_or_single_state_dict", side_effect=lambda *_: _fp8_state_dict()
        ):
            with mock.patch.object(
                ideogram_model, "_build_transformer", side_effect=[mock.Mock(), torch.nn.Linear(2, 2)]
            ) as build:
                model.load_model(move_to_device=False)

        uncond_state_dict = build.call_args_list[1].args[1]
        self.assertFalse(is_fp8_state_dict(uncond_state_dict))
        self.assertEqual(uncond_state_dict["proj.weight"].dtype, torch.bfloat16)

    def test_uncond_load_default_keeps_fp8_state_dict(self):
        from simpletuner.helpers.models.ideogram import model as ideogram_model

        model = _make_model(load_unconditional=True)
        state_dict = _fp8_state_dict()
        with mock.patch.object(ideogram_model, "_load_indexed_or_single_state_dict", return_value=state_dict):
            with mock.patch.object(
                ideogram_model, "_build_transformer", side_effect=[mock.Mock(), torch.nn.Linear(2, 2)]
            ) as build:
                model.load_model(move_to_device=False)

        self.assertIs(build.call_args_list[1].args[1], state_dict)

    def test_uncond_ramtorch_builds_on_cpu_and_applies_ramtorch(self):
        from simpletuner.helpers.models.ideogram import model as ideogram_model

        model = _make_model(load_unconditional=True, uncond_ramtorch=True)
        model.accelerator.device = torch.device("meta")
        model._apply_ramtorch_layers = mock.Mock(return_value=1)
        model._ramtorch_transformer_percent = mock.Mock(return_value=None)
        uncond = torch.nn.Linear(2, 2)
        with mock.patch.object(ideogram_model, "_load_indexed_or_single_state_dict", return_value=_fp8_state_dict()):
            with mock.patch.object(ideogram_model, "_build_transformer", side_effect=[mock.Mock(), uncond]) as build:
                model.load_model(move_to_device=False)

        self.assertEqual(build.call_args_list[1].args[2], torch.device("cpu"))
        model._apply_ramtorch_layers.assert_called_once()
        self.assertIs(model._apply_ramtorch_layers.call_args.args[0], uncond)
        self.assertTrue(model._apply_ramtorch_layers.call_args.kwargs["force"])
        self.assertTrue(model._apply_ramtorch_layers.call_args.kwargs["full_ramtorch"])

    def test_uncond_ramtorch_requires_load_flag(self):
        from simpletuner.helpers.models.common import ImageModelFoundation

        model = _make_model(load_unconditional=False, uncond_ramtorch=True)
        with mock.patch.object(ImageModelFoundation, "check_user_config"):
            with self.assertRaises(ValueError):
                model.check_user_config()


class IdeogramUnconditionalDispatchTests(unittest.TestCase):
    def _make_predict_model(self, with_uncond: bool = True):
        from simpletuner.helpers.models.ideogram.model import Ideogram4

        model = Ideogram4.__new__(Ideogram4)
        model.config = mock.Mock()
        model.config.weight_dtype = torch.float32
        model.accelerator = mock.Mock()
        model.accelerator.device = torch.device("cpu")
        model.model = mock.Mock(side_effect=lambda **kwargs: torch.zeros_like(kwargs["x"]))
        model.unconditional_transformer = (
            mock.Mock(side_effect=lambda **kwargs: torch.zeros_like(kwargs["x"])) if with_uncond else None
        )
        return model

    def _prepared_batch(self, unconditional: bool = False):
        batch = {
            "noisy_latents": torch.randn(1, 128, 2, 2),
            "timesteps": torch.tensor([500.0]),
            "encoder_hidden_states": torch.randn(1, 5, 16),
        }
        if unconditional:
            batch["is_unconditional_pass"] = True
        return batch

    def test_use_unconditional_transformer_decision(self):
        model = self._make_predict_model(with_uncond=True)
        self.assertTrue(model._use_unconditional_transformer({"is_unconditional_pass": True}))
        self.assertFalse(model._use_unconditional_transformer({}))

        model.unconditional_transformer = None
        self.assertFalse(model._use_unconditional_transformer({"is_unconditional_pass": True}))

    def test_marked_batch_dispatches_to_unconditional_transformer(self):
        model = self._make_predict_model(with_uncond=True)
        batch = self._prepared_batch(unconditional=True)
        batch["flowmap_r_timesteps"] = torch.tensor([250.0])

        output = model.model_predict(batch)

        model.unconditional_transformer.assert_called_once()
        model.model.assert_not_called()
        call_kwargs = model.unconditional_transformer.call_args.kwargs
        self.assertEqual(tuple(call_kwargs["x"].shape), (1, 4, 128))
        self.assertEqual(tuple(call_kwargs["llm_features"].shape), (1, 4, 16))
        self.assertTrue(torch.all(call_kwargs["llm_features"] == 0))
        self.assertEqual(tuple(call_kwargs["position_ids"].shape), (1, 4, 3))
        self.assertTrue(torch.all(call_kwargs["segment_ids"] == 1))
        self.assertTrue(torch.all(call_kwargs["indicator"] == OUTPUT_IMAGE_INDICATOR))
        self.assertNotIn("r_timestep", call_kwargs)
        self.assertEqual(tuple(output["model_prediction"].shape), (1, 128, 2, 2))

    def test_unmarked_batch_uses_conditional_transformer(self):
        model = self._make_predict_model(with_uncond=True)

        output = model.model_predict(self._prepared_batch(unconditional=False))

        model.model.assert_called_once()
        model.unconditional_transformer.assert_not_called()
        self.assertEqual(tuple(output["model_prediction"].shape), (1, 128, 2, 2))

    def test_marked_batch_without_unconditional_transformer_uses_proxy_path(self):
        model = self._make_predict_model(with_uncond=False)

        output = model.model_predict(self._prepared_batch(unconditional=True))

        model.model.assert_called_once()
        self.assertEqual(tuple(output["model_prediction"].shape), (1, 128, 2, 2))


if __name__ == "__main__":
    unittest.main()
