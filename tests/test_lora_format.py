import unittest

import torch

from simpletuner.helpers.training.lora_format import (
    PEFTLoRAFormat,
    convert_diffusers_to_comfyui,
    convert_diffusers_to_comfyui_sd_lora,
    detect_state_dict_format,
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


if __name__ == "__main__":
    unittest.main()
