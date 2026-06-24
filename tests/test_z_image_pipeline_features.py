import types
import unittest

import torch

from simpletuner.helpers.models.z_image.pipeline import ZImagePipeline
from simpletuner.helpers.models.z_image.transformer import ZImageTransformer2DModel


class FakeScheduler:
    def __init__(self):
        self.order = 1
        self.timesteps = None
        self.step_calls = []
        self.config = {
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.15,
        }
        self.sigma_min = 0.0

    def set_timesteps(self, num_inference_steps, device=None, **kwargs):
        self.timesteps = torch.arange(num_inference_steps, dtype=torch.float32, device=device)

    def step(self, noise_pred, t, latents, return_dict=False):
        self.step_calls.append(noise_pred.detach().clone())
        return (latents.clone(),)


class FakeTransformer(torch.nn.Module):
    def __init__(self, pos_value=1.0, neg_value=0.0, single_value=1.0, skip_value=0.0):
        super().__init__()
        self.in_channels = 1
        self.config = types.SimpleNamespace(all_patch_size=(2,), all_f_patch_size=(1,))
        self.pos_value = pos_value
        self.neg_value = neg_value
        self.single_value = single_value
        self.skip_value = skip_value
        self.last_skip_layers = None
        self.calls = []
        self.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(1), requires_grad=False))

    @property
    def dtype(self):
        return torch.bfloat16

    def __call__(self, latent_list, timestep, prompt_embeds, skip_layers=None, **kwargs):
        self.last_skip_layers = skip_layers
        self.calls.append(
            {
                "latent_shapes": [latent.shape for latent in latent_list],
                "timestep": timestep.detach().clone(),
                "skip_layers": skip_layers,
            }
        )

        def _fill_like(tensor, value):
            return torch.full_like(tensor, value)

        if skip_layers is not None:
            outputs = [_fill_like(latent_list[0], self.skip_value)]
            return outputs, {}

        if len(latent_list) == 2:
            outputs = [
                _fill_like(latent_list[0], self.pos_value),
                _fill_like(latent_list[1], self.neg_value),
            ]
        else:
            outputs = [_fill_like(latent, self.single_value) for latent in latent_list]

        return outputs, {}


class FakeVAE(torch.nn.Module):
    class Config(types.SimpleNamespace):
        pass

    def __init__(self):
        super().__init__()
        self.config = self.Config(
            block_out_channels=[1],
            scaling_factor=1.0,
            shift_factor=0.0,
        )
        self.dtype = torch.float32
        self.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(1), requires_grad=False))

    def decode(self, latents, return_dict=False):
        return (latents.clone(),)

    def encode(self, image):
        latents = image[:, :1].clone()
        return types.SimpleNamespace(
            latent_dist=types.SimpleNamespace(sample=lambda generator=None: latents),
        )


class ZImagePipelineFeatureTests(unittest.TestCase):
    def _build_pipeline(self, transformer):
        scheduler = FakeScheduler()
        vae = FakeVAE()
        # text encoder / tokenizer unused when prompt_embeds provided
        original_exec = getattr(ZImagePipeline, "_execution_device", None)
        ZImagePipeline._execution_device = property(lambda self: torch.device("cpu"))
        if original_exec is not None:
            self.addCleanup(setattr, ZImagePipeline, "_execution_device", original_exec)

        pipe = ZImagePipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            transformer=transformer,
        )

        def preprocess(images, height=None, width=None):
            if torch.is_tensor(images):
                return images
            if not isinstance(images, list):
                images = [images]
            tensors = []
            for image in images:
                if torch.is_tensor(image):
                    tensor = image
                    if tensor.ndim == 3:
                        tensor = tensor.unsqueeze(0)
                    tensors.append(tensor.squeeze(0).to(dtype=torch.float32))
                else:
                    tensors.append(torch.ones(3, height, width, dtype=torch.float32))
            return torch.stack(tensors)

        pipe.image_processor = types.SimpleNamespace(
            preprocess=preprocess,
            postprocess=lambda x, output_type=None: x,
        )
        pipe = pipe.to(torch_device="cpu", torch_dtype=torch.bfloat16)
        pipe._dtype = torch.bfloat16
        return pipe

    def test_cfg_zero_star_guidance_applied(self):
        transformer = FakeTransformer(pos_value=1.0, neg_value=0.0)
        pipe = self._build_pipeline(transformer)

        latents = torch.ones(1, 1, 2, 2)
        prompt_embeds = torch.zeros(1, 1, 1, dtype=torch.bfloat16)
        neg_embeds = torch.zeros(1, 1, 1, dtype=torch.bfloat16)

        pipe(
            prompt="",
            negative_prompt="",
            num_inference_steps=1,
            guidance_scale=2.0,
            height=2,
            width=2,
            latents=latents.clone(),
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            output_type="latent",
            cfg_truncation=None,
            use_zero_init=False,
            zero_steps=0,
        )

        self.assertEqual(len(pipe.scheduler.step_calls), 1)
        expected = torch.full_like(latents, -2.0)
        self.assertTrue(torch.allclose(pipe.scheduler.step_calls[0], expected))

    def test_no_cfg_until_skips_first_step(self):
        transformer = FakeTransformer(pos_value=1.0, neg_value=0.0, single_value=1.0)
        pipe = self._build_pipeline(transformer)

        latents = torch.ones(1, 1, 2, 2)
        prompt_embeds = torch.zeros(1, 1, 1, dtype=torch.bfloat16)
        neg_embeds = torch.zeros(1, 1, 1, dtype=torch.bfloat16)

        pipe(
            prompt="",
            negative_prompt="",
            num_inference_steps=2,
            guidance_scale=3.0,
            height=2,
            width=2,
            latents=latents.clone(),
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            output_type="latent",
            no_cfg_until_timestep=1,
            cfg_truncation=None,
            use_zero_init=False,
            zero_steps=0,
        )

        self.assertEqual(len(pipe.scheduler.step_calls), 2)
        self.assertTrue(torch.allclose(pipe.scheduler.step_calls[0], torch.full_like(latents, -1.0)))
        self.assertTrue(torch.allclose(pipe.scheduler.step_calls[1], torch.full_like(latents, -3.0)))

    def test_skip_layer_guidance_adjusts_noise(self):
        transformer = FakeTransformer(pos_value=1.0, neg_value=0.0, skip_value=0.0)
        pipe = self._build_pipeline(transformer)

        latents = torch.ones(1, 1, 2, 2)
        prompt_embeds = torch.zeros(1, 1, 1, dtype=torch.bfloat16)
        neg_embeds = torch.zeros(1, 1, 1, dtype=torch.bfloat16)

        pipe(
            prompt="",
            negative_prompt="",
            num_inference_steps=1,
            guidance_scale=2.0,
            height=2,
            width=2,
            latents=latents.clone(),
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            output_type="latent",
            skip_guidance_layers=[1],
            skip_layer_guidance_start=-1.0,
            skip_layer_guidance_stop=1.0,
            cfg_truncation=None,
            use_zero_init=False,
            zero_steps=0,
        )

        self.assertEqual(transformer.last_skip_layers, [1])
        # Base CFG Zero* gives -2, skip correction adds 2.8 -> 0.8
        self.assertTrue(torch.allclose(pipe.scheduler.step_calls[0], torch.full_like(latents, 0.8), atol=1e-5))

    def test_reference_image_concats_reference_latents_and_crops_prediction(self):
        transformer = FakeTransformer(single_value=1.0)
        pipe = self._build_pipeline(transformer)

        latents = torch.ones(1, 1, 2, 4)
        prompt_embeds = torch.zeros(1, 1, 1, dtype=torch.bfloat16)
        reference_image = torch.ones(1, 3, 2, 4)

        pipe(
            prompt="",
            num_inference_steps=1,
            guidance_scale=1.0,
            height=2,
            width=4,
            latents=latents.clone(),
            prompt_embeds=prompt_embeds,
            reference_image=reference_image,
            output_type="latent",
        )

        self.assertEqual(transformer.calls[0]["latent_shapes"], [torch.Size([1, 1, 2, 8])])
        self.assertEqual(transformer.calls[0]["timestep"].shape, torch.Size([1, 4]))
        torch.testing.assert_close(transformer.calls[0]["timestep"][:, :2], torch.zeros(1, 2))
        torch.testing.assert_close(transformer.calls[0]["timestep"][:, 2:], torch.ones(1, 2))
        self.assertEqual(pipe.scheduler.step_calls[0].shape, latents.shape)
        self.assertTrue(torch.allclose(pipe.scheduler.step_calls[0], torch.full_like(latents, -1.0)))

    def test_reference_image_cfg_repeats_reference_latents_and_crops_prediction(self):
        transformer = FakeTransformer(pos_value=1.0, neg_value=0.0)
        pipe = self._build_pipeline(transformer)

        latents = torch.ones(1, 1, 2, 4)
        prompt_embeds = torch.zeros(1, 1, 1, dtype=torch.bfloat16)
        neg_embeds = torch.zeros(1, 1, 1, dtype=torch.bfloat16)
        reference_image = torch.ones(1, 3, 2, 4)

        pipe(
            prompt="",
            negative_prompt="",
            num_inference_steps=1,
            guidance_scale=2.0,
            height=2,
            width=4,
            latents=latents.clone(),
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            reference_image=reference_image,
            output_type="latent",
            cfg_truncation=None,
            use_zero_init=False,
            zero_steps=0,
        )

        self.assertEqual(transformer.calls[0]["latent_shapes"], [torch.Size([1, 1, 2, 8]), torch.Size([1, 1, 2, 8])])
        self.assertEqual(transformer.calls[0]["timestep"].shape, torch.Size([2, 4]))
        torch.testing.assert_close(transformer.calls[0]["timestep"][:, :2], torch.zeros(2, 2))
        torch.testing.assert_close(transformer.calls[0]["timestep"][:, 2:], torch.ones(2, 2))
        self.assertEqual(pipe.scheduler.step_calls[0].shape, latents.shape)
        self.assertTrue(torch.allclose(pipe.scheduler.step_calls[0], torch.full_like(latents, -2.0)))


if __name__ == "__main__":
    unittest.main()
