import unittest
from helpers.training.save_hooks import SaveHookManager
from accelerate import Accelerator
from transformers import CLIPTextModel
from diffusers import UNet2DConditionModel
from argparse import Namespace
from helpers.training.state_tracker import StateTracker


class TestSaveHookManagerSD(unittest.TestCase):

    def setUp(self):
        self.accelerator = Accelerator(device_placement=False)
        self.ckpt_id = "hf-internal-testing/tiny-stable-diffusion-pipe"
        self.unet = UNet2DConditionModel.from_pretrained(self.ckpt_id, subfolder="unet")
        self.text_encoder_one = CLIPTextModel.from_pretrained(self.ckpt_id, subfolder="text_encoder")

        args = Namespace(controlnet=None, sd3=False, flux=False, pixart=False)
        self.args = args
        StateTracker.set_model_type("legacy")

    def test_hook_manager(self):
        model_hooks = SaveHookManager(
            args=self.args,
            unet=self.unet,
            transformer=None,
            ema_model=None,
            accelerator=self.accelerator,
            text_encoder_1=self.text_encoder_one,
            text_encoder_2=None,
            use_deepspeed_optimizer=False,
        )
        assert model_hooks

        assert model_hooks.denoiser_class.__name__ == "UNet2DConditionModel"
        assert model_hooks.pipeline_class.__name__ == "StableDiffusionPipeline"


if __name__ == "__main__":
    unittest.main()
