import unittest
from types import SimpleNamespace

import torch

from simpletuner.helpers.models.ace_step.pipeline import ACEStepPipeline


class DummyTokenizer:
    pad_token_id = 0

    def __call__(self, texts, padding="max_length", truncation=True, max_length=256, return_tensors="pt"):
        del padding, truncation, max_length
        if isinstance(texts, str):
            texts = [texts]
        batch = len(texts)
        input_ids = torch.tensor([[1, 2, 0, 0]] * batch, dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 0, 0]] * batch, dtype=torch.long)
        return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)


class DummyTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)

    def forward(self, input_ids=None, **kwargs):
        del kwargs
        return SimpleNamespace(last_hidden_state=self.embed_tokens(input_ids) + 1.0)


class DummyV15Model:
    def __init__(self):
        self.kwargs = None

    def generate_audio(self, **kwargs):
        self.kwargs = kwargs
        src_latents = kwargs["src_latents"]
        return {
            "target_latents": torch.ones(
                src_latents.shape[0],
                src_latents.shape[1],
                src_latents.shape[2],
                dtype=src_latents.dtype,
                device=src_latents.device,
            ),
            "time_costs": {},
        }


class DummyVAE:
    dtype = torch.float32

    def __init__(self):
        self.decode_input = None

    def decode(self, latents):
        self.decode_input = latents
        batch = latents.shape[0]
        sample = torch.full((batch, 2, 3200), 0.25, dtype=torch.float32)
        return SimpleNamespace(sample=sample)


class TestACEStepPipeline(unittest.TestCase):
    def test_v15_validation_call_uses_generation_api_and_decodes_audio(self):
        pipeline = ACEStepPipeline()
        pipeline.loaded = True
        pipeline.is_v15_pipeline = True
        pipeline.device = torch.device("cpu")
        pipeline.dtype = torch.float32
        pipeline.text_tokenizer = DummyTokenizer()
        pipeline.text_encoder_model = DummyTextEncoder()
        pipeline.v15_model = DummyV15Model()
        pipeline.music_dcae = DummyVAE()
        pipeline.silence_latent = torch.zeros(1, 4, 64)

        result = pipeline(
            prompt="test prompt",
            lyrics="la la la",
            audio_duration=1.2,
            guidance_scale=3.5,
            num_inference_steps=7,
            num_images_per_prompt=2,
        )

        self.assertEqual(len(result.audios), 2)
        self.assertEqual(pipeline.v15_model.kwargs["infer_steps"], 7)
        self.assertEqual(pipeline.v15_model.kwargs["diffusion_guidance_scale"], 3.5)
        self.assertEqual(pipeline.v15_model.kwargs["src_latents"].shape, (2, 30, 64))
        self.assertEqual(pipeline.v15_model.kwargs["chunk_masks"].shape, (2, 30, 64))
        self.assertEqual(pipeline.v15_model.kwargs["text_hidden_states"].shape[0], 2)
        self.assertEqual(pipeline.v15_model.kwargs["lyric_hidden_states"].shape[0], 2)
        self.assertEqual(pipeline.music_dcae.decode_input.shape, (2, 64, 30))
        self.assertEqual(result.params["sample_rate"], 48000)


if __name__ == "__main__":
    unittest.main()
