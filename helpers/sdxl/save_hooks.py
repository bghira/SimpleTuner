from diffusers.training_utils import EMAModel
from diffusers import UNet2DConditionModel
from helpers.training.state_tracker import StateTracker
import os, logging, shutil, json

logger = logging.getLogger("SDXLSaveHook")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL") or "INFO")


class SDXLSaveHook:
    def __init__(self, args, ema_unet, accelerator):
        self.args = args
        self.ema_unet = ema_unet
        self.accelerator = accelerator

    def save_model_hook(self, models, weights, output_dir):
        # Create a temporary directory for atomic saves
        temporary_dir = output_dir.replace("checkpoint", "temporary")
        os.makedirs(temporary_dir, exist_ok=True)

        if self.args.use_ema:
            self.ema_unet.save_pretrained(os.path.join(temporary_dir, "unet_ema"))

        for model in models:
            model.save_pretrained(os.path.join(temporary_dir, "unet"))
            weights.pop()  # Pop the last weight

        # Copy contents of temporary directory to output directory
        for item in os.listdir(temporary_dir):
            s = os.path.join(temporary_dir, item)
            d = os.path.join(output_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)  # Python 3.8+
            else:
                shutil.copy2(s, d)

        # Remove the temporary directory
        shutil.rmtree(temporary_dir)

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

        # Check the checkpoint dir for a "training_state.json" file to load
        training_state_path = os.path.join(input_dir, "training_state.json")
        if os.path.exists(training_state_path):
            with open(training_state_path, "r") as f:
                training_state = json.load(f)
            StateTracker.set_global_step(training_state["global_step"])
            StateTracker.set_epoch_step(training_state["epoch_step"])
            StateTracker.set_epoch(training_state["epoch"])
            self.has_training_started = training_state["has_training_started"]
            self.calculate_luminance = training_state["calculate_luminance"]
            self.all_image_files = training_state["all_image_files"]
            self.all_vae_cache_files = training_state["all_vae_cache_files"]
            self.all_caption_files = training_state["all_caption_files"]
        else:
            logger.warning(
                f"Could not find training_state.json in checkpoint dir {input_dir}"
            )
