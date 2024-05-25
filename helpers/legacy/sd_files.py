from diffusers.training_utils import EMAModel, _set_state_dict_into_text_encoder
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
)

from transformers import PretrainedConfig
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from peft import set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from helpers.training.state_tracker import StateTracker
from helpers.training.wrappers import unwrap_model
import os, logging, shutil, torch

logger = logging.getLogger("legacy.sd_files")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def register_file_hooks(
    args,
    accelerator,
    unet,
    text_encoder,
    text_encoder_cls,
    use_deepspeed_optimizer,
    ema_unet=None,
    controlnet=None,
):
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        StateTracker.save_training_state(
            os.path.join(output_dir, "training_state.json")
        )
        if "lora" in StateTracker.get_args().model_type:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(accelerator, unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                    text_encoder_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                elif not use_deepspeed_optimizer:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            StableDiffusionPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_lora_layers_to_save,
            )
            return
        for model in models:
            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
            if args.checkpoints_total_limit is not None:
                checkpoints = os.listdir(args.output_dir)
                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                if len(checkpoints) >= args.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]

                    logger.debug(
                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                    )
                    logger.debug(
                        f"removing checkpoints: {', '.join(removing_checkpoints)}"
                    )

                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(
                            args.output_dir, removing_checkpoint
                        )
                        shutil.rmtree(removing_checkpoint)
            if isinstance(model, type(unwrap_model(accelerator, text_encoder))):
                sub_dir = "text_encoder"
            elif isinstance(model, type(unwrap_model(accelerator, controlnet))):
                sub_dir = "controlnet"
            elif isinstance(model, type(unwrap_model(accelerator, unet))):
                sub_dir = "unet"
            elif not use_deepspeed_optimizer:
                raise ValueError(f"unexpected save model: {model.__class__}")

            model.save_pretrained(os.path.join(output_dir, sub_dir))

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

        if args.use_ema:
            ema_unet.save_pretrained(os.path.join(output_dir, "ema_unet"))

    def load_model_hook(models, input_dir):
        training_state_path = os.path.join(input_dir, "training_state.json")
        if os.path.exists(training_state_path):
            StateTracker.load_training_state(training_state_path)
        else:
            logger.warning(
                f"Could not find training_state.json in checkpoint dir {input_dir}"
            )

        if "lora" in args.model_type:
            unet_ = None
            text_encoder_ = None
            while len(models) > 0:
                model = models.pop()

                if isinstance(model, type(unwrap_model(accelerator, unet))):
                    unet_ = model
                elif isinstance(model, type(unwrap_model(accelerator, text_encoder))):
                    text_encoder_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
            logger.info(f"Loading LoRA weights from Path: {input_dir}")

            lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)

            unet_state_dict = {
                f'{k.replace("unet.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("unet.")
            }
            unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
            incompatible_keys = set_peft_model_state_dict(
                unet_, unet_state_dict, adapter_name="default"
            )
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

            if args.train_text_encoder:
                # Do we need to call `scale_lora_layers()` here?
                _set_state_dict_into_text_encoder(
                    lora_state_dict,
                    prefix="text_encoder.",
                    text_encoder=text_encoder_,
                )

            logger.info("Completed loading LoRA weights.")

        if args.use_ema:
            load_model = EMAModel.from_pretrained(
                os.path.join(input_dir, "ema_unet"), UNet2DConditionModel
            )
            ema_unet.load_state_dict(load_model.state_dict())
            ema_unet.to(accelerator.device)
            del load_model
        if args.model_type == "full":
            return_exception = False

            while len(models) > 0:
                try:
                    # pop models so that they are not loaded again
                    model = models.pop()
                    if isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                        # load transformers style into model
                        load_model = text_encoder_cls.from_pretrained(
                            input_dir, subfolder="text_encoder"
                        )
                    elif isinstance(model, type(accelerator.unwrap_model(controlnet))):
                        # load controlnet into model
                        from diffusers import ControlNetModel

                        load_model = ControlNetModel.from_pretrained(
                            input_dir, subfolder="controlnet"
                        )
                    else:
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

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
