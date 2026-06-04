import json
import os
import tempfile
import types
import unittest
from unittest import mock

import torch
import torch.nn as nn
from safetensors.torch import load_file

from simpletuner.helpers.models.ideogram.model import Ideogram4
from simpletuner.helpers.models.ideogram.pipeline import Ideogram4Pipeline
from simpletuner.helpers.models.ideogram.prompting import maybe_convert_prompt_to_ideogram_json
from simpletuner.helpers.models.ideogram.scheduler import get_schedule_for_resolution
from simpletuner.helpers.models.ideogram.transformer import Ideogram4Config, Ideogram4Transformer


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
        self.assertNotIn("negative_prompts", mapped)
        self.assertNotIn("prompt", mapped)
        self.assertNotIn("negative_prompt", mapped)
        self.assertEqual(mapped["num_steps"], 12)
        self.assertFalse(mapped["raise_on_caption_issues"])

    def test_text_embed_conversion_and_collation_pad_at_batch_time(self):
        model = Ideogram4.__new__(Ideogram4)
        first = {"prompt_embeds": torch.ones(2, 4), "attention_mask": torch.ones(2, dtype=torch.bool)}
        second = {"prompt_embeds": torch.ones(1, 4) * 2, "attention_mask": torch.ones(1, dtype=torch.bool)}

        collated = model.collate_prompt_embeds([first, second])

        self.assertEqual(collated["prompt_embeds"].shape, (2, 2, 4))
        self.assertTrue(torch.equal(collated["attention_masks"], torch.tensor([[True, True], [True, False]])))

        converted = model.convert_text_embed_for_pipeline(first)
        negative = model.convert_negative_text_embed_for_pipeline(first)
        self.assertEqual(converted["prompt_embeds"].shape, (1, 2, 4))
        self.assertEqual(negative["negative_prompt_embeds"].shape, (1, 2, 4))

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

        pipe(
            None,
            prompt_embeds=torch.zeros(1, 2, 8),
            prompt_attention_mask=torch.ones(1, 2, dtype=torch.bool),
            negative_prompt_embeds=torch.zeros(1, 1, 8),
            negative_prompt_attention_mask=torch.ones(1, 1, dtype=torch.bool),
            height=16,
            width=16,
            num_steps=1,
            guidance_scale=2.0,
        )

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
                return torch.ones_like(x)

        model = Ideogram4.__new__(Ideogram4)
        model.accelerator = DummyAccelerator()
        model.config = types.SimpleNamespace(weight_dtype=torch.float32)
        model.model = DummyTransformer()

        result = model.model_predict(
            {
                "noisy_latents": torch.randn(2, 128, 4, 4),
                "prompt_embeds": torch.randn(2, 3, 4096),
                "attention_mask": torch.ones(2, 3, dtype=torch.bool),
                "timesteps": torch.tensor([100.0, 500.0]),
            }
        )

        self.assertEqual(result["model_prediction"].shape, (2, 128, 4, 4))
        self.assertEqual(model.model.last_kwargs["x"].shape, (2, 19, 128))
        self.assertEqual(model.model.last_kwargs["position_ids"].shape, (2, 19, 3))
        self.assertTrue(torch.equal(result["model_prediction"], torch.full((2, 128, 4, 4), -1.0)))
        self.assertTrue(torch.allclose(model.model.last_kwargs["t"], torch.tensor([0.9, 0.5])))

    def test_vae_latent_norm_matches_flux2_batch_norm_layout(self):
        model = Ideogram4.__new__(Ideogram4)
        latents = torch.arange(1 * 32 * 4 * 4, dtype=torch.float32).view(1, 32, 4, 4) / 100.0
        bn = nn.BatchNorm2d(128, eps=1e-4, affine=False, track_running_stats=True)
        bn.running_mean.copy_(torch.linspace(-1.0, 1.0, 128))
        bn.running_var.copy_(torch.linspace(0.25, 2.0, 128))
        model.vae = types.SimpleNamespace(bn=bn)
        model.get_vae = lambda: model.vae

        normalized = model.post_vae_encode_transform_sample(latents)
        packed = model._patchify_vae_latents(latents)
        expected = (packed - bn.running_mean.view(1, -1, 1, 1)) / torch.sqrt(
            bn.running_var.view(1, -1, 1, 1) + bn.eps
        )

        self.assertEqual(normalized.shape, (1, 128, 2, 2))
        self.assertTrue(torch.allclose(normalized, expected))

    def test_sample_flow_sigmas_uses_ideogram_schedule_as_complement(self):
        model = Ideogram4.__new__(Ideogram4)
        model.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
        model.config = types.SimpleNamespace(ideogram_schedule_mu=0.0, ideogram_schedule_std=1.5)
        batch = {"latents": torch.zeros(2, 128, 64, 64)}
        schedule_u = torch.tensor([0.25, 0.75], dtype=torch.float32)

        with mock.patch("torch.rand", return_value=schedule_u):
            sigmas, timesteps = model.sample_flow_sigmas(batch=batch, state={})

        model_t = get_schedule_for_resolution((1024, 1024), known_mean=0.0, std=1.5)(schedule_u)
        expected_sigmas = 1.0 - model_t
        self.assertTrue(torch.allclose(sigmas, expected_sigmas))
        self.assertTrue(torch.allclose(timesteps, expected_sigmas * 1000.0))

    def test_negative_prompt_encoding_is_not_rejected(self):
        class DummyTokenizer:
            def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
                return messages[0]["content"][0]["text"]

            def __call__(self, text, return_tensors="pt", add_special_tokens=False):
                return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

        class DummyPipeline:
            def __init__(self, **kwargs):
                pass

            def _encode_text(self, token_ids, text_position_ids, indicator):
                return torch.zeros(token_ids.shape[0], token_ids.shape[1], 8)

        model = Ideogram4.__new__(Ideogram4)
        model.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
        model.config = types.SimpleNamespace(ideogram_auto_json=True, weight_dtype=torch.float32)
        model.tokenizers = [DummyTokenizer()]
        model.text_encoders = [object()]
        original_pipeline = __import__("simpletuner.helpers.models.ideogram.model", fromlist=["Ideogram4Pipeline"]).Ideogram4Pipeline
        try:
            __import__("simpletuner.helpers.models.ideogram.model", fromlist=["Ideogram4Pipeline"]).Ideogram4Pipeline = DummyPipeline
            encoded = model._encode_prompts(["not blurry"], is_negative_prompt=True)
        finally:
            __import__("simpletuner.helpers.models.ideogram.model", fromlist=["Ideogram4Pipeline"]).Ideogram4Pipeline = original_pipeline

        self.assertEqual(encoded["prompt_embeds"].shape, (1, 2, 8))
        self.assertTrue(torch.equal(encoded["attention_mask"], torch.ones(1, 2, dtype=torch.bool)))

    def test_prompt_upsample_runs_before_json_conversion(self):
        class DummyTokenizer:
            captured_text = None

            def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
                self.captured_text = messages[0]["content"][0]["text"]
                return self.captured_text

            def __call__(self, text, return_tensors="pt", add_special_tokens=False):
                return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

        class DummyPipeline:
            last_upsample_args = None

            def __init__(self, **kwargs):
                pass

            def upsample_prompt(self, prompt, height, width, device=None):
                self.__class__.last_upsample_args = (prompt, height, width)
                return json.dumps(
                    {
                        "high_level_description": f"upsampled {prompt}",
                        "style_description": {"medium": "photograph"},
                        "compositional_deconstruction": {"background": "studio", "elements": []},
                    }
                )

            def _encode_text(self, token_ids, text_position_ids, indicator):
                return torch.zeros(token_ids.shape[0], token_ids.shape[1], 8)

        model = Ideogram4.__new__(Ideogram4)
        model.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
        model.config = types.SimpleNamespace(
            ideogram_auto_json=True,
            ideogram_prompt_upsample=True,
            resolution="768x1024",
            weight_dtype=torch.float32,
        )
        tokenizer = DummyTokenizer()
        model.prompt_enhancer_head = object()
        model.tokenizers = [tokenizer]
        model.text_encoders = [object()]
        module = __import__("simpletuner.helpers.models.ideogram.model", fromlist=["Ideogram4Pipeline"])
        original_pipeline = module.Ideogram4Pipeline
        try:
            module.Ideogram4Pipeline = DummyPipeline
            encoded = model._encode_prompts(["plain prompt"])
        finally:
            module.Ideogram4Pipeline = original_pipeline

        self.assertEqual(DummyPipeline.last_upsample_args, ("plain prompt", 768, 1024))
        self.assertIn("upsampled plain prompt", tokenizer.captured_text)
        self.assertEqual(encoded["prompt_embeds"].shape, (1, 2, 8))

    def test_prompt_enhancer_head_uses_configured_repo(self):
        class DummyHead:
            loaded_repo = None

            @classmethod
            def from_pretrained(cls, repo_id):
                cls.loaded_repo = repo_id
                return cls()

            def to(self, *args, **kwargs):
                return self

            def eval(self):
                return self

        model = Ideogram4.__new__(Ideogram4)
        model.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
        model.config = types.SimpleNamespace(
            ideogram_prompt_enhancer_head_id="example/head",
            weight_dtype=torch.float32,
        )
        module = __import__("simpletuner.helpers.models.ideogram.model", fromlist=["Ideogram4PromptEnhancerHead"])
        original_head = module.Ideogram4PromptEnhancerHead
        try:
            module.Ideogram4PromptEnhancerHead = DummyHead
            loaded = model.load_prompt_enhancer_head(move_to_device=True)
        finally:
            module.Ideogram4PromptEnhancerHead = original_head

        self.assertIs(loaded, model.prompt_enhancer_head)
        self.assertEqual(DummyHead.loaded_repo, "example/head")

    def test_pipeline_saves_lora_weights_with_transformer_prefix(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            Ideogram4Pipeline.save_lora_weights(
                save_directory=tmp_dir,
                transformer_lora_layers={"layers.0.attention.qkv.lora_A.weight": torch.ones(1, 2)},
                weight_name="ideogram-test.safetensors",
                safe_serialization=True,
            )

            saved_path = os.path.join(tmp_dir, "ideogram-test.safetensors")
            self.assertTrue(os.path.exists(saved_path))
            saved = load_file(saved_path)

        self.assertIn("transformer.layers.0.attention.qkv.lora_A.weight", saved)
        self.assertTrue(torch.equal(saved["transformer.layers.0.attention.qkv.lora_A.weight"], torch.ones(1, 2)))

    def test_transformer_gradient_checkpointing_forward_backward(self):
        model = Ideogram4Transformer(
            Ideogram4Config(
                emb_dim=192,
                num_layers=2,
                num_heads=1,
                intermediate_size=384,
                adanln_dim=32,
                llm_features_dim=64,
            )
        )
        model.train()
        model.enable_gradient_checkpointing()

        x = torch.randn(1, 6, 128, requires_grad=True)
        out = model(
            llm_features=torch.randn(1, 6, 64),
            x=x,
            t=torch.rand(1),
            position_ids=torch.zeros(1, 6, 3, dtype=torch.long),
            segment_ids=torch.zeros(1, 6, dtype=torch.long),
            indicator=torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.long),
        )
        loss = out.square().mean()
        loss.backward()

        self.assertEqual(out.shape, (1, 6, 128))
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
