from diffusers.training_utils import EMAModel
from diffusers import UNet2DConditionModel
import os


class SDXLSaveHook:
    def __init__(self, args, ema_unet, accelerator):
        self.args = args
        self.ema_unet = ema_unet
        self.accelerator = accelerator

    def save_model_hook(self, models, weights, output_dir):
        if self.args.use_ema:
            self.ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

        for i, model in enumerate(models):
            model.save_pretrained(os.path.join(output_dir, "unet"))

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(self, models, input_dir):
        if self.args.use_ema:
            load_model = EMAModel.from_pretrained(
                os.path.join(input_dir, "unet_ema"), UNet2DConditionModel
            )
            self.ema_unet.load_state_dict(load_model.state_dict())
            self.ema_unet.to(self.accelerator.device)
            del load_model

        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet"
            )
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model
