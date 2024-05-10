from diffusers.training_utils import EMAModel, _set_state_dict_into_text_encoder
from helpers.training.wrappers import unwrap_model
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
)
from peft import set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

from diffusers import UNet2DConditionModel
from helpers.sdxl.pipeline import StableDiffusionXLPipeline
from helpers.training.state_tracker import StateTracker
import os, logging, shutil, torch

logger = logging.getLogger("SDXLSaveHook")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL") or "INFO")


class SDXLSaveHook:
    def __init__(
        self,
        args,
        unet,
        ema_unet,
        text_encoder_1,
        text_encoder_2,
        accelerator,
        use_deepspeed_optimizer,
    ):
        self.args = args
        self.unet = unet
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.ema_unet = ema_unet
        self.accelerator = accelerator
        self.use_deepspeed_optimizer = use_deepspeed_optimizer

    def save_model_hook(self, models, weights, output_dir):
        # Write "training_state.json" to the output directory containing the training state
        StateTracker.save_training_state(
            os.path.join(output_dir, "training_state.json")
        )
        if "lora" in self.args.model_type:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_1_lora_layers_to_save = None
            text_encoder_2_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(self.accelerator, self.unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                elif isinstance(
                    model, type(self.accelerator.unwrap_model(self.text_encoder_1))
                ):
                    text_encoder_1_lora_layers_to_save = (
                        convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                    )
                elif isinstance(
                    model, type(self.accelerator.unwrap_model(self.text_encoder_2))
                ):
                    text_encoder_2_lora_layers_to_save = (
                        convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                    )
                elif not self.use_deepspeed_optimizer:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_1_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_2_lora_layers_to_save,
            )
            return

        # Create a temporary directory for atomic saves
        temporary_dir = output_dir.replace("checkpoint", "temporary")
        os.makedirs(temporary_dir, exist_ok=True)

        if self.args.use_ema:
            self.ema_unet.save_pretrained(os.path.join(temporary_dir, "unet_ema"))

        for model in models:
            model.save_pretrained(os.path.join(temporary_dir, "unet"))
            if weights:
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
        # Check the checkpoint dir for a "training_state.json" file to load
        training_state_path = os.path.join(input_dir, "training_state.json")
        if os.path.exists(training_state_path):
            StateTracker.load_training_state(training_state_path)
        else:
            logger.warning(
                f"Could not find training_state.json in checkpoint dir {input_dir}"
            )

        if "lora" in self.args.model_type:
            logger.info(f"Loading LoRA weights from Path: {input_dir}")
            unet_ = None
            text_encoder_one_ = None
            text_encoder_two_ = None

            while len(models) > 0:
                model = models.pop()

                if isinstance(model, type(unwrap_model(self.accelerator, self.unet))):
                    unet_ = model
                elif isinstance(
                    model, type(unwrap_model(self.accelerator, self.text_encoder_one))
                ):
                    text_encoder_one_ = model
                elif isinstance(
                    model, type(unwrap_model(self.accelerator, self.text_encoder_two))
                ):
                    text_encoder_two_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

            lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)

            unet_state_dict = {
                f'{k.replace("unet.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("unet.")
            }
            unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
            incompatible_keys = set_peft_model_state_dict(
                unet_ or self.unet, unet_state_dict, adapter_name="default"
            )
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

            if self.args.train_text_encoder:
                # Do we need to call `scale_lora_layers()` here?
                _set_state_dict_into_text_encoder(
                    lora_state_dict,
                    prefix="text_encoder.",
                    text_encoder=text_encoder_one_,
                )

                _set_state_dict_into_text_encoder(
                    lora_state_dict,
                    prefix="text_encoder_2.",
                    text_encoder=text_encoder_one_,
                )

            logger.info("Completed loading LoRA weights.")

        if self.args.use_ema:
            load_model = EMAModel.from_pretrained(
                os.path.join(input_dir, "unet_ema"), UNet2DConditionModel
            )
            self.ema_unet.load_state_dict(load_model.state_dict())
            self.ema_unet.to(self.accelerator.device)
            del load_model
        if self.args.model_type == "full":
            return_exception = False
            for i in range(len(models)):
                try:
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = UNet2DConditionModel.from_pretrained(
                        input_dir, subfolder="unet"
                    )
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model
                except Exception as e:
                    return_exception = e

            if return_exception:
                raise Exception("Could not load model: {}".format(return_exception))
