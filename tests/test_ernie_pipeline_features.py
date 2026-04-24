import types
import unittest

import torch

from simpletuner.helpers.models.ernie.pipeline import ErnieImagePipeline


class FakeScheduler:
    def __init__(self):
        self.timesteps = None
        self.step_calls = []

    def set_timesteps(self, sigmas, device=None, **kwargs):
        del kwargs
        self.timesteps = sigmas.to(device=device)

    def step(self, noise_pred, t, latents):
        del t
        self.step_calls.append(noise_pred.detach().clone())
        return types.SimpleNamespace(prev_sample=latents.clone())


class FakeTransformer(torch.nn.Module):
    def __init__(self, pos_value=1.0, neg_value=0.0, skip_value=0.0):
        super().__init__()
        self.config = types.SimpleNamespace(in_channels=4, text_in_dim=3)
        self.pos_value = pos_value
        self.neg_value = neg_value
        self.skip_value = skip_value
        self.calls = []
        self.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(1), requires_grad=False))

    @property
    def dtype(self):
        return torch.bfloat16

    def __call__(self, hidden_states, timestep, text_bth, text_lens, return_dict=False, skip_layers=None, **kwargs):
        del text_lens, return_dict, kwargs
        self.calls.append(
            {
                "hidden_shape": tuple(hidden_states.shape),
                "timestep_dtype": timestep.dtype,
                "text_batch": text_bth.shape[0],
                "skip_layers": skip_layers,
            }
        )
        if skip_layers is not None:
            return (torch.full_like(hidden_states, self.skip_value),)
        if hidden_states.shape[0] % 2 == 0:
            half = hidden_states.shape[0] // 2
            neg = torch.full_like(hidden_states[:half], self.neg_value)
            pos = torch.full_like(hidden_states[half:], self.pos_value)
            return (torch.cat([neg, pos], dim=0),)
        return (torch.full_like(hidden_states, self.pos_value),)


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(block_out_channels=[1, 1, 1, 1])
        self.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(1), requires_grad=False))


class ErniePipelineFeatureTests(unittest.TestCase):
    def _build_pipeline(self, transformer=None):
        original_exec = getattr(ErnieImagePipeline, "_execution_device", None)
        ErnieImagePipeline._execution_device = property(lambda self: torch.device("cpu"))
        if original_exec is not None:
            self.addCleanup(setattr, ErnieImagePipeline, "_execution_device", original_exec)

        return ErnieImagePipeline(
            scheduler=FakeScheduler(),
            vae=FakeVAE(),
            text_encoder=None,
            tokenizer=None,
            transformer=transformer or FakeTransformer(),
        )

    def test_prompt_embeds_repeat_for_num_images_per_prompt(self):
        transformer = FakeTransformer(pos_value=1.0, neg_value=0.0)
        pipe = self._build_pipeline(transformer)
        latents = torch.ones(4, 4, 1, 1, dtype=torch.bfloat16)
        prompt_embeds = [torch.zeros(2, 3), torch.ones(3, 3)]
        negative_prompt_embeds = [torch.zeros(1, 3), torch.ones(1, 3)]

        pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_images_per_prompt=2,
            guidance_scale=2.0,
            num_inference_steps=1,
            height=16,
            width=16,
            latents=latents,
            output_type="latent",
            use_zero_init=False,
        )

        self.assertEqual(transformer.calls[0]["hidden_shape"], (8, 4, 1, 1))
        self.assertEqual(transformer.calls[0]["text_batch"], 8)
        self.assertEqual(transformer.calls[0]["timestep_dtype"], torch.float32)

    def test_negative_prompt_embeds_must_match_prompt_batch(self):
        pipe = self._build_pipeline()

        with self.assertRaisesRegex(ValueError, "negative_prompt_embeds"):
            pipe(
                prompt_embeds=[torch.zeros(2, 3), torch.ones(3, 3)],
                negative_prompt_embeds=[torch.zeros(1, 3)],
                guidance_scale=2.0,
                num_inference_steps=1,
                height=16,
                width=16,
                latents=torch.ones(2, 4, 1, 1, dtype=torch.bfloat16),
                output_type="latent",
            )

    def test_callback_tensor_inputs_are_validated(self):
        pipe = self._build_pipeline()

        with self.assertRaisesRegex(ValueError, "callback_on_step_end_tensor_inputs"):
            pipe(
                prompt_embeds=[torch.zeros(2, 3)],
                negative_prompt_embeds=[torch.zeros(1, 3)],
                guidance_scale=2.0,
                num_inference_steps=1,
                height=16,
                width=16,
                latents=torch.ones(1, 4, 1, 1, dtype=torch.bfloat16),
                output_type="latent",
                callback_on_step_end_tensor_inputs=["not_latents"],
            )


if __name__ == "__main__":
    unittest.main()
