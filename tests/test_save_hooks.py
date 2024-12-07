import unittest
from helpers.training.save_hooks import SaveHookManager
from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from diffusers import (
    UNet2DConditionModel,
    SD3Transformer2DModel,
    FluxTransformer2DModel,
)
from argparse import Namespace
from helpers.training.state_tracker import StateTracker

import logging

hf_logger = logging.getLogger("diffusers.models.modeling_utils")
hf_logger.disabled = True


class TestSaveHookManager(unittest.TestCase):
    def setUp(self):
        self.accelerator = Accelerator(device_placement=False)

    def _initialize_models(
        self, model_type, ckpt_id, unet=None, transformer=None, text_encoder_2=None
    ):
        self.ckpt_id = ckpt_id
        self.unet = unet
        self.transformer = transformer
        self.text_encoder_one = CLIPTextModel.from_pretrained(
            self.ckpt_id, subfolder="text_encoder"
        )
        self.text_encoder_two = text_encoder_2

        args = Namespace(
            controlnet=None,
            sd3=False,
            flux=False,
            pixart_sigma=False,
            flux_attention_masked_training=False,
        )
        if model_type == "sd3":
            args.model_family = "sd3"
        elif model_type == "flux":
            args.model_family = "flux"

        self.args = args
        StateTracker.set_model_family(model_type)

    def _test_hook_manager(self, expected_denoiser_class, expected_pipeline_class):
        model_hooks = SaveHookManager(
            args=self.args,
            unet=self.unet,
            transformer=self.transformer,
            ema_model=None,
            accelerator=self.accelerator,
            text_encoder_1=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            use_deepspeed_optimizer=False,
        )
        self.assertIsNotNone(model_hooks)
        self.assertEqual(model_hooks.denoiser_class.__name__, expected_denoiser_class)
        self.assertEqual(model_hooks.pipeline_class.__name__, expected_pipeline_class)

    def test_hook_manager_sd(self):
        self._initialize_models(
            model_type="legacy",
            ckpt_id="hf-internal-testing/tiny-stable-diffusion-pipe",
            unet=UNet2DConditionModel.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-pipe", subfolder="unet"
            ),
        )
        self._test_hook_manager("UNet2DConditionModel", "StableDiffusionPipeline")

    def test_hook_manager_sdxl(self):
        self._initialize_models(
            model_type="sdxl",
            ckpt_id="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            unet=UNet2DConditionModel.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-xl-pipe", subfolder="unet"
            ),
            text_encoder_2=CLIPTextModelWithProjection.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
                subfolder="text_encoder_2",
            ),
        )
        self._test_hook_manager("UNet2DConditionModel", "StableDiffusionXLPipeline")

    def test_hook_manager_sd3(self):
        self._initialize_models(
            model_type="sd3",
            ckpt_id="hf-internal-testing/tiny-sd3-pipe",
            transformer=SD3Transformer2DModel.from_pretrained(
                "hf-internal-testing/tiny-sd3-pipe", subfolder="transformer"
            ),
            text_encoder_2=CLIPTextModelWithProjection.from_pretrained(
                "hf-internal-testing/tiny-sd3-pipe", subfolder="text_encoder_2"
            ),
        )
        self._test_hook_manager("SD3Transformer2DModel", "StableDiffusion3Pipeline")

    def test_hook_manager_flux(self):
        self._initialize_models(
            model_type="flux",
            ckpt_id="hf-internal-testing/tiny-flux-pipe",
            transformer=FluxTransformer2DModel.from_pretrained(
                "hf-internal-testing/tiny-flux-pipe", subfolder="transformer"
            ),
        )
        self._test_hook_manager("FluxTransformer2DModel", "FluxPipeline")


if __name__ == "__main__":
    unittest.main()
