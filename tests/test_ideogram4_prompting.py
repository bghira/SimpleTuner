import json
import types
import unittest

import torch

from simpletuner.helpers.models.ideogram.model import Ideogram4
from simpletuner.helpers.models.ideogram.pipeline import Ideogram4Pipeline
from simpletuner.helpers.models.ideogram.prompting import maybe_convert_prompt_to_ideogram_json


class Ideogram4PromptingTests(unittest.TestCase):
    def test_plain_prompt_is_wrapped_as_schema_json(self):
        converted = maybe_convert_prompt_to_ideogram_json("35mm photo of a blue boat at sunset #1b3a5c")
        parsed = json.loads(converted)

        self.assertEqual(parsed["high_level_description"], "35mm photo of a blue boat at sunset #1b3a5c")
        self.assertEqual(parsed["style_description"]["medium"], "photograph")
        self.assertEqual(parsed["style_description"]["color_palette"], ["#1B3A5C"])
        self.assertIn("compositional_deconstruction", parsed)
        self.assertEqual(parsed["compositional_deconstruction"]["elements"][0]["type"], "obj")

    def test_existing_json_caption_is_canonicalized(self):
        raw = json.dumps(
            {
                "high_level_description": "A poster.",
                "style_description": {
                    "medium": "graphic_design",
                    "art_style": "flat vector",
                    "colour_palette": ["#ffffff"],
                },
                "compositional_deconstruction": {
                    "elements": [{"desc": "Logo text", "type": "text", "text": "ACME"}],
                    "background": "White card.",
                },
            }
        )

        parsed = json.loads(maybe_convert_prompt_to_ideogram_json(raw))

        self.assertEqual(
            list(parsed["style_description"].keys()),
            ["aesthetics", "lighting", "medium", "art_style", "color_palette"],
        )
        self.assertEqual(parsed["style_description"]["color_palette"], ["#FFFFFF"])
        self.assertEqual(list(parsed["compositional_deconstruction"].keys()), ["background", "elements"])
        self.assertEqual(list(parsed["compositional_deconstruction"]["elements"][0].keys()), ["type", "text", "desc"])

    def test_model_pipeline_kwargs_map_to_upstream_names(self):
        model = Ideogram4.__new__(Ideogram4)
        model.config = types.SimpleNamespace(ideogram_auto_json=True, weight_dtype=torch.bfloat16)

        mapped = model.update_pipeline_call_kwargs(
            {
                "prompt": "a sign that says hello",
                "negative_prompt": "",
                "num_images_per_prompt": 1,
                "num_inference_steps": 12,
                "guidance_scale": 4.0,
            }
        )

        self.assertIn("prompts", mapped)
        self.assertIn("negative_prompts", mapped)
        self.assertNotIn("prompt", mapped)
        self.assertNotIn("negative_prompt", mapped)
        self.assertEqual(mapped["num_steps"], 12)
        self.assertFalse(mapped["raise_on_caption_issues"])

    def test_pipeline_cfg_fallback_uses_negative_prompt_with_conditional_transformer(self):
        class DummyTransformer:
            config = types.SimpleNamespace(in_channels=4)

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return torch.ones_like(kwargs["x"])

        transformer = DummyTransformer()
        pipe = Ideogram4Pipeline.__new__(Ideogram4Pipeline)
        pipe.conditional_transformer = transformer
        pipe.unconditional_transformer = None
        pipe.device = torch.device("cpu")
        pipe.dtype = torch.float32
        pipe.config = types.SimpleNamespace(patch_size=2, ae_scale_factor=8, max_text_tokens=2048)
        pipe._verify_prompts = lambda *args, **kwargs: None
        pipe._decode = lambda z, **kwargs: [z]

        def build_inputs(prompts, height, width):
            text_tokens = 2 if prompts[0] == "positive" else 1
            seq_len = text_tokens + 1
            return {
                "token_ids": torch.zeros(1, seq_len, dtype=torch.long),
                "text_position_ids": torch.zeros(1, seq_len, 3, dtype=torch.long),
                "position_ids": torch.zeros(1, seq_len, 3, dtype=torch.long),
                "segment_ids": torch.ones(1, seq_len, dtype=torch.long),
                "indicator": torch.ones(1, seq_len, dtype=torch.long),
                "num_image_tokens": 1,
                "grid_h": 1,
                "grid_w": 1,
                "max_text_tokens": text_tokens,
            }

        pipe._build_inputs = build_inputs
        pipe._encode_text = lambda token_ids, text_position_ids, indicator: torch.zeros(
            token_ids.shape[0], token_ids.shape[1], 8
        )

        pipe("positive", negative_prompts="negative", height=16, width=16, num_steps=1, guidance_scale=2.0)

        self.assertEqual(len(transformer.calls), 2)
        self.assertEqual(transformer.calls[0]["x"].shape[1], 3)
        self.assertEqual(transformer.calls[1]["x"].shape[1], 2)

    def test_model_predict_packs_and_unpacks_conditional_sequence(self):
        class DummyAccelerator:
            device = torch.device("cpu")

            def unwrap_model(self, model):
                return model

        class DummyTransformer:
            def __call__(self, **kwargs):
                x = kwargs["x"]
                self.last_kwargs = kwargs
                return torch.zeros_like(x)

        model = Ideogram4.__new__(Ideogram4)
        model.accelerator = DummyAccelerator()
        model.config = types.SimpleNamespace(weight_dtype=torch.float32)
        model.model = DummyTransformer()

        result = model.model_predict(
            {
                "noisy_latents": torch.randn(2, 32, 4, 4),
                "prompt_embeds": torch.randn(2, 3, 4096),
                "attention_mask": torch.ones(2, 3, dtype=torch.bool),
                "timesteps": torch.tensor([100.0, 500.0]),
            }
        )

        self.assertEqual(result["model_prediction"].shape, (2, 32, 4, 4))
        self.assertEqual(model.model.last_kwargs["x"].shape, (2, 7, 128))
        self.assertEqual(model.model.last_kwargs["position_ids"].shape, (2, 7, 3))


if __name__ == "__main__":
    unittest.main()
