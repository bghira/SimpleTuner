from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from simpletuner.helpers.models.common import ModelFoundation


class _LenHostileModule:
    def __len__(self):
        raise TypeError("_LenHostileModule does not support len()")


class _ConcreteModelFoundation(ModelFoundation):
    def _encode_prompts(self, prompts, is_negative_prompt=False):
        raise NotImplementedError

    def convert_text_embed_for_pipeline(self, prompt_embeds):
        raise NotImplementedError

    def convert_negative_text_embed_for_pipeline(self, prompt_embeds):
        raise NotImplementedError

    def model_predict(self, prepared_batch, custom_timesteps=None):
        raise NotImplementedError


class ModelFoundationCompileTests(TestCase):
    def test_unwrap_model_does_not_truth_test_explicit_model(self):
        foundation = _ConcreteModelFoundation.__new__(_ConcreteModelFoundation)
        foundation.config = SimpleNamespace(controlnet=False)
        foundation.accelerator = object()
        foundation.model = object()
        foundation.controlnet = None

        compiled_like_model = _LenHostileModule()

        with patch("simpletuner.helpers.models.common.unwrap_model", side_effect=lambda _accel, model, **_: model):
            self.assertIs(foundation.unwrap_model(model=compiled_like_model), compiled_like_model)
