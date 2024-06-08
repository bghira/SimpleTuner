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
import os, logging, shutil, torch, json
from safetensors import safe_open
from safetensors.torch import save_file


logger = logging.getLogger("SDXLSaveHook")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL") or "INFO")


def merge_safetensors_files(directory):
    json_file_name = "diffusion_pytorch_model.safetensors.index.json"
    json_file_path = os.path.join(directory, json_file_name)
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"{json_file_name} not found in the directory.")

    # Step 2: Load the JSON file and extract the weight map
    with open(json_file_path, "r") as file:
        data = json.load(file)
        weight_map = data.get("weight_map")
        if weight_map is None:
            raise KeyError("'weight_map' key not found in the JSON file.")

    # Collect all unique safetensors files from weight_map
    files_to_load = set(weight_map.values())
    all_tensors = {}

    # Load tensors from each unique file
    for file_name in files_to_load:
        part_file_path = os.path.join(directory, file_name)
        if not os.path.exists(part_file_path):
            raise FileNotFoundError(f"Part file {file_name} not found.")

        with safe_open(part_file_path, framework="pt", device="cpu") as f:
            for tensor_key in f.keys():
                if tensor_key in weight_map:
                    all_tensors[tensor_key] = f.get_tensor(tensor_key)

    # Step 4: Save all loaded tensors into a single new safetensors file
    output_file_path = os.path.join(directory, "diffusion_model_pytorch.safetensors")
    save_file(all_tensors, output_file_path)

    logger.info(f"All tensors have been merged and saved into {output_file_path}")


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

        sub_dir = "unet" if not self.args.controlnet else "controlnet"
        for model in models:
            model.save_pretrained(os.path.join(temporary_dir, sub_dir))
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
                    if self.args.controlnet:
                        from diffusers import ControlNetModel

                        load_model = ControlNetModel.from_pretrained(
                            input_dir, subfolder="controlnet"
                        )
                    else:
                        merge_safetensors_files(os.path.join(input_dir, "unet"))
                        load_model = UNet2DConditionModel.from_pretrained(
                            input_dir, subfolder="unet"
                        )
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model
                except Exception as e:
                    import traceback

                    return_exception = f"Could not load model: {e}, traceback: {traceback.format_exc()}"

            if return_exception:
                raise Exception(return_exception)
