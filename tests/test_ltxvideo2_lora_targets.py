import unittest
from types import SimpleNamespace

import torch

from simpletuner.helpers.models.ltxvideo2.model import LTXVideo2


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(1, 1)
        self.to_k = torch.nn.Linear(1, 1)
        self.to_v = torch.nn.Linear(1, 1)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(1, 1)])


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attention()
        self.video_to_audio_attn = _Attention()


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_Block()])


class LTXVideo2LoraTargetTests(unittest.TestCase):
    def _model_foundation(self, *, has_audio: bool, manual_targets=None):
        model = object.__new__(LTXVideo2)
        model.config = SimpleNamespace(lora_type="standard", peft_lora_target_modules=manual_targets)
        model.model = _Transformer()
        model._data_has_audio = has_audio
        model._data_has_video = True
        return model

    def test_video_only_lora_targets_exclude_video_to_audio_attention(self):
        model = self._model_foundation(has_audio=False)

        targets = model.get_lora_target_layers()

        self.assertIn("transformer_blocks.0.attn.to_q", targets)
        self.assertNotIn("to_q", targets)
        self.assertFalse(any("video_to_audio_attn" in target for target in targets))

    def test_audio_lora_targets_keep_generic_attention_targets(self):
        model = self._model_foundation(has_audio=True)

        targets = model.get_lora_target_layers()

        self.assertIn("to_q", targets)
        self.assertIn("audio_proj_in", targets)

    def test_manual_lora_targets_are_not_filtered(self):
        model = self._model_foundation(has_audio=False, manual_targets=["video_to_audio_attn.to_q"])

        self.assertEqual(model.get_lora_target_layers(), ["video_to_audio_attn.to_q"])


if __name__ == "__main__":
    unittest.main()
