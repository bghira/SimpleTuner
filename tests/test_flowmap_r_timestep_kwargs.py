import inspect
import unittest
from unittest.mock import patch

import torch

from simpletuner.helpers.models.common import ModelFoundation


class _DummyFoundation(ModelFoundation):
    NAME = "dummy"

    def model_predict(self, prepared_batch, custom_timesteps: list | None = None):
        return prepared_batch

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        return prompts

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        return {"prompt_embeds": text_embedding}

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        return {"negative_prompt_embeds": text_embedding}


class _AcceptsRTimestep:
    def forward(self, hidden_states=None, r_timestep=None):
        return hidden_states, r_timestep


class _AcceptsTimestepR:
    def forward(self, hidden_states=None, timestep_r=None):
        return hidden_states, timestep_r


class _RejectsRTimestep:
    def forward(self, hidden_states=None, timestep=None):
        return hidden_states, timestep


class _AcceptsArbitraryKwargs:
    def forward(self, hidden_states=None, **kwargs):
        return hidden_states, kwargs


class FlowMapRTimestepKwargsTests(unittest.TestCase):
    def setUp(self):
        self.model = _DummyFoundation.__new__(_DummyFoundation)

    def test_absent_key_leaves_kwargs_unchanged(self):
        call_kwargs = {"timestep": torch.tensor([1.0])}

        applied = self.model._apply_flowmap_r_timestep_kwargs(
            call_kwargs,
            {},
            target=_RejectsRTimestep().forward,
        )

        self.assertFalse(applied)
        self.assertEqual(set(call_kwargs), {"timestep"})

    def test_none_key_leaves_kwargs_unchanged(self):
        call_kwargs = {"timestep": torch.tensor([1.0])}

        applied = self.model._apply_flowmap_r_timestep_kwargs(
            call_kwargs,
            {self.model.FLOWMAP_R_TIMESTEP_BATCH_KEY: None},
            target=_RejectsRTimestep().forward,
        )

        self.assertFalse(applied)
        self.assertEqual(set(call_kwargs), {"timestep"})

    def test_supported_r_timestep_is_forwarded(self):
        r_timesteps = torch.tensor([0.25, 0.5])
        call_kwargs = {}

        applied = self.model._apply_flowmap_r_timestep_kwargs(
            call_kwargs,
            {self.model.FLOWMAP_R_TIMESTEP_BATCH_KEY: r_timesteps},
            target=_AcceptsRTimestep().forward,
        )

        self.assertTrue(applied)
        self.assertIs(call_kwargs["r_timestep"], r_timesteps)

    def test_supported_model_specific_kwarg_is_forwarded(self):
        r_timesteps = torch.tensor([0.25, 0.5])
        call_kwargs = {}

        applied = self.model._apply_flowmap_r_timestep_kwargs(
            call_kwargs,
            {self.model.FLOWMAP_R_TIMESTEP_BATCH_KEY: r_timesteps},
            target=_AcceptsTimestepR().forward,
            kwarg_name="timestep_r",
        )

        self.assertTrue(applied)
        self.assertIs(call_kwargs["timestep_r"], r_timesteps)

    def test_forward_signature_support_is_cached(self):
        r_timesteps = torch.tensor([0.25, 0.5])
        target = _AcceptsRTimestep()

        with patch(
            "simpletuner.helpers.models.common.inspect.signature",
            wraps=inspect.signature,
        ) as signature:
            self.model._apply_flowmap_r_timestep_kwargs(
                {},
                {self.model.FLOWMAP_R_TIMESTEP_BATCH_KEY: r_timesteps},
                target=target.forward,
            )
            self.model._apply_flowmap_r_timestep_kwargs(
                {},
                {self.model.FLOWMAP_R_TIMESTEP_BATCH_KEY: r_timesteps},
                target=target.forward,
            )

        self.assertEqual(signature.call_count, 1)

    def test_unsupported_forward_raises(self):
        with self.assertRaisesRegex(ValueError, "does not accept `r_timestep`"):
            self.model._apply_flowmap_r_timestep_kwargs(
                {},
                {self.model.FLOWMAP_R_TIMESTEP_BATCH_KEY: torch.tensor([0.5])},
                target=_RejectsRTimestep().forward,
            )

    def test_arbitrary_kwargs_do_not_count_as_model_support(self):
        with self.assertRaisesRegex(ValueError, "does not accept `r_timestep`"):
            self.model._apply_flowmap_r_timestep_kwargs(
                {},
                {self.model.FLOWMAP_R_TIMESTEP_BATCH_KEY: torch.tensor([0.5])},
                target=_AcceptsArbitraryKwargs().forward,
            )

    def test_duplicate_kwarg_raises(self):
        with self.assertRaisesRegex(ValueError, "duplicate `r_timestep`"):
            self.model._apply_flowmap_r_timestep_kwargs(
                {"r_timestep": torch.tensor([0.25])},
                {self.model.FLOWMAP_R_TIMESTEP_BATCH_KEY: torch.tensor([0.5])},
                target=_AcceptsRTimestep().forward,
            )


if __name__ == "__main__":
    unittest.main()
