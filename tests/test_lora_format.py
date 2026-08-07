import unittest
from unittest.mock import patch

import torch

from simpletuner.helpers.training.lora_format import (
    PEFTLoRAFormat,
    convert_diffusers_to_comfyui,
    convert_diffusers_to_comfyui_sd_lora,
    detect_state_dict_format,
    get_peft_kwargs,
    peft_lora_config_kwargs_from_state_dict,
    synthesize_missing_lora_alphas_from_ranks,
)

DOWN = torch.full((4, 8), 1.0)
UP = torch.full((8, 4), 2.0)


def _diffusers_named(module_key):
    return {
        f"{module_key}.lora.down.weight": DOWN,
        f"{module_key}.lora.up.weight": UP,
    }


def _peft_named(module_key):
    return {
        f"{module_key}.lora_A.weight": DOWN,
        f"{module_key}.lora_B.weight": UP,
    }


SPELLINGS = (("diffusers", _diffusers_named), ("peft", _peft_named))


class ConvertDiffusersToComfyUITests(unittest.TestCase):
    MODULE = "transformer.blocks.0.attn.to_q"
    CONVERTED = "diffusion_model.blocks.0.attn.to_q"
    ALPHA_KEY = "diffusion_model.blocks.0.attn.to_q.alpha"
    METADATA = {"lora_alpha": 16}

    def _convert(self, state_dict):
        return convert_diffusers_to_comfyui(state_dict, adapter_metadata=self.METADATA)

    def test_both_spellings_convert_to_the_same_keys(self):
        converted = {name: set(self._convert(build(self.MODULE))) for name, build in SPELLINGS}
        self.assertEqual(converted["diffusers"], converted["peft"])

    def test_alpha_is_emitted_for_both_spellings(self):
        for name, build in SPELLINGS:
            with self.subTest(spelling=name):
                converted = self._convert(build(self.MODULE))
                self.assertIn(self.ALPHA_KEY, converted)
                self.assertEqual(float(converted[self.ALPHA_KEY]), float(self.METADATA["lora_alpha"]))

    def test_down_and_up_weights_keep_their_roles(self):
        for name, build in SPELLINGS:
            with self.subTest(spelling=name):
                converted = self._convert(build(self.MODULE))
                self.assertTrue(torch.equal(converted[f"{self.CONVERTED}.lora_A.weight"], DOWN))
                self.assertTrue(torch.equal(converted[f"{self.CONVERTED}.lora_B.weight"], UP))


class ConvertDiffusersToComfyUISDLoraTests(unittest.TestCase):
    MODULE = "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k"
    ALPHA_KEY = "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.alpha"
    METADATA = {"lora_alpha": 8}

    def _convert(self, state_dict):
        return convert_diffusers_to_comfyui_sd_lora(state_dict, adapter_metadata=self.METADATA)

    def test_both_spellings_convert_to_the_same_keys(self):
        converted = {name: set(self._convert(build(self.MODULE))) for name, build in SPELLINGS}
        self.assertEqual(converted["diffusers"], converted["peft"])

    def test_alpha_is_emitted_for_both_spellings(self):
        for name, build in SPELLINGS:
            with self.subTest(spelling=name):
                converted = self._convert(build(self.MODULE))
                self.assertIn(self.ALPHA_KEY, converted)
                self.assertEqual(float(converted[self.ALPHA_KEY]), float(self.METADATA["lora_alpha"]))


class DetectStateDictFormatTests(unittest.TestCase):
    MODULE = "transformer.blocks.0.attn.to_q"

    def test_peft_named_dict_is_reported_as_diffusers(self):
        self.assertEqual(detect_state_dict_format(_peft_named(self.MODULE)), PEFTLoRAFormat.DIFFUSERS)

    def test_diffusers_named_dict_is_reported_as_diffusers(self):
        self.assertEqual(detect_state_dict_format(_diffusers_named(self.MODULE)), PEFTLoRAFormat.DIFFUSERS)

    def test_diffusion_model_peft_named_dict_is_reported_as_comfyui(self):
        self.assertEqual(
            detect_state_dict_format(_peft_named("diffusion_model.blocks.0.attn.to_q")),
            PEFTLoRAFormat.COMFYUI,
        )

    def test_model_diffusion_model_peft_named_dict_is_reported_as_comfyui(self):
        self.assertEqual(
            detect_state_dict_format(_peft_named("model.diffusion_model.blocks.0.attn.to_q")),
            PEFTLoRAFormat.COMFYUI,
        )


class MixedRankAlphaInferenceTests(unittest.TestCase):
    def _ranked(self, ranks):
        state_dict = {}
        for module_key, rank in ranks:
            state_dict[f"{module_key}.lora_A.weight"] = torch.zeros(rank, 8)
            state_dict[f"{module_key}.lora_B.weight"] = torch.zeros(16, rank)
        return state_dict

    def test_mixed_rank_without_alphas_synthesizes_rank_alphas(self):
        state_dict = self._ranked(
            [
                ("transformer.blocks.0.attn.to_q", 64),
                ("transformer.blocks.0.adaln", 16),
            ]
        )

        alphas = synthesize_missing_lora_alphas_from_ranks(state_dict)

        self.assertEqual(alphas["transformer.blocks.0.attn.to_q.alpha"], 64.0)
        self.assertEqual(alphas["transformer.blocks.0.adaln.alpha"], 16.0)

    def test_uniform_rank_without_alphas_keeps_global_scale(self):
        state_dict = self._ranked(
            [
                ("transformer.blocks.0.attn.to_q", 64),
                ("transformer.blocks.0.attn.to_k", 64),
            ]
        )

        self.assertEqual(synthesize_missing_lora_alphas_from_ranks(state_dict), {})

    def test_explicit_alpha_suppresses_synthesis(self):
        state_dict = self._ranked(
            [
                ("transformer.blocks.0.attn.to_q", 64),
                ("transformer.blocks.0.adaln", 16),
            ]
        )
        state_dict["transformer.blocks.0.adaln.alpha"] = torch.tensor(8.0)

        self.assertEqual(synthesize_missing_lora_alphas_from_ranks(state_dict), {})

    def test_init_lora_config_kwargs_use_rank_patterns_for_mixed_ranks(self):
        state_dict = self._ranked(
            [
                ("unet.to_q", 64),
                ("unet.to_k", 64),
                ("unet.adaln", 16),
            ]
        )

        kwargs = peft_lora_config_kwargs_from_state_dict(state_dict, prefix_to_strip="unet.")

        self.assertEqual(kwargs["r"], 64)
        self.assertEqual(kwargs["lora_alpha"], 64.0)
        self.assertEqual(kwargs["rank_pattern"], {"adaln": 16})
        self.assertEqual(kwargs["alpha_pattern"], {"adaln": 16.0})

    def test_explicit_init_lora_alpha_pattern_wins(self):
        state_dict = self._ranked(
            [
                ("unet.to_q", 64),
                ("unet.to_k", 16),
            ]
        )
        state_dict["unet.to_q.alpha"] = torch.tensor(32.0)
        state_dict["unet.to_k.alpha"] = torch.tensor(8.0)

        kwargs = peft_lora_config_kwargs_from_state_dict(state_dict, prefix_to_strip="unet.")

        self.assertEqual(kwargs["r"], 64)
        self.assertEqual(kwargs["lora_alpha"], 32.0)
        self.assertEqual(kwargs["rank_pattern"], {"to_k": 16})
        self.assertEqual(kwargs["alpha_pattern"], {"to_k": 8.0})

    def test_get_peft_kwargs_wrapper_passes_inferred_alphas(self):
        state_dict = self._ranked(
            [
                ("transformer.to_q", 64),
                ("transformer.adaln", 16),
            ]
        )

        with patch("diffusers.utils.get_peft_kwargs", return_value={"ok": True}) as diffusers_get_peft_kwargs:
            result = get_peft_kwargs({"rank": 64}, network_alpha_dict=None, peft_state_dict=state_dict)

        self.assertEqual(result, {"ok": True})
        _, network_alphas, peft_state_dict = diffusers_get_peft_kwargs.call_args.args[:3]
        self.assertIs(peft_state_dict, state_dict)
        self.assertEqual(network_alphas["transformer.to_q.alpha"], 64.0)
        self.assertEqual(network_alphas["transformer.adaln.alpha"], 16.0)

    def test_get_peft_kwargs_wrapper_promotes_explicit_state_dict_alphas(self):
        state_dict = self._ranked(
            [
                ("transformer.to_q", 64),
                ("transformer.adaln", 16),
            ]
        )
        state_dict["transformer.to_q.alpha"] = torch.tensor(32.0)
        state_dict["transformer.adaln.alpha"] = torch.tensor(8.0)

        with patch("diffusers.utils.get_peft_kwargs", return_value={"ok": True}) as diffusers_get_peft_kwargs:
            result = get_peft_kwargs({"rank": 64}, network_alpha_dict=None, peft_state_dict=state_dict)

        self.assertEqual(result, {"ok": True})
        _, network_alphas, _ = diffusers_get_peft_kwargs.call_args.args[:3]
        self.assertEqual(network_alphas["transformer.to_q.alpha"], 32.0)
        self.assertEqual(network_alphas["transformer.adaln.alpha"], 8.0)


if __name__ == "__main__":
    unittest.main()
