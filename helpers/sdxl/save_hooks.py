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
from tqdm import tqdm

logger = logging.getLogger("SDXLSaveHook")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL") or "INFO")

try:
    from diffusers import StableDiffusion3Pipeline
except ImportError:
    logger.error("This release requires the latest version of Diffusers.")

try:
    from diffusers.models import PixArtTransformer2DModel
except Exception as e:
    logger.error(
        f"Can not load Pixart Sigma model class. This release requires the latest version of Diffusers: {e}"
    )
    raise e


def merge_safetensors_files(directory):
    json_file_name = "diffusion_pytorch_model.safetensors.index.json"
    json_file_path = os.path.join(directory, json_file_name)
    if not os.path.exists(json_file_path):
        return

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
    output_file_path = os.path.join(directory, "diffusion_pytorch_model.safetensors")
    save_file(all_tensors, output_file_path)
    # Step 5: If the file now exists, remove the index and part files
    if os.path.exists(output_file_path):
        os.remove(json_file_path)
        for file_name in files_to_load:
            os.remove(os.path.join(directory, file_name))

    logger.info(f"All tensors have been merged and saved into {output_file_path}")


class SDXLSaveHook:
    def __init__(
        self,
        args,
        unet,
        transformer,
        ema_model,
        text_encoder_1,
        text_encoder_2,
        text_encoder_3,
        accelerator,
        use_deepspeed_optimizer,
    ):
        self.args = args
        self.unet = unet
        self.transformer = transformer
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.text_encoder_3 = text_encoder_3
        self.ema_model = ema_model
        self.accelerator = accelerator
        self.use_deepspeed_optimizer = use_deepspeed_optimizer
        self.ema_model_cls = None
        self.ema_model_subdir = None
        if unet is not None:
            self.ema_model_subdir = "unet_ema"
            self.ema_model_cls = UNet2DConditionModel
        if transformer is not None:
            self.ema_model_subdir = "transformer_ema"
            if self.args.sd3:
                self.ema_model_cls = SD3Transformer2DModel
            elif self.args.pixart_sigma:
                self.ema_model_cls = PixArtTransformer2DModel

    def save_model_hook(self, models, weights, output_dir):
        # Write "training_state.json" to the output directory containing the training state
        StateTracker.save_training_state(
            os.path.join(output_dir, "training_state.json")
        )
        if "lora" in self.args.model_type:
            # for SDXL/others, there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers.
            unet_lora_layers_to_save = None
            transformer_lora_layers_to_save = None
            text_encoder_1_lora_layers_to_save = None
            text_encoder_2_lora_layers_to_save = None
            text_encoder_3_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(self.accelerator, self.unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                elif isinstance(
                    model, type(unwrap_model(self.accelerator, self.text_encoder_1))
                ):
                    text_encoder_1_lora_layers_to_save = (
                        convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                    )
                elif isinstance(
                    model, type(unwrap_model(self.accelerator, self.text_encoder_2))
                ):
                    text_encoder_2_lora_layers_to_save = (
                        convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                    )
                elif isinstance(
                    model, type(unwrap_model(self.accelerator, self.text_encoder_3))
                ):
                    text_encoder_3_lora_layers_to_save = (
                        convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                    )

                elif isinstance(
                    model, type(unwrap_model(self.accelerator, self.transformer))
                ):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)

                elif not self.use_deepspeed_optimizer:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            if self.args.sd3:
                StableDiffusion3Pipeline.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                    # SD3 doesn't support text encoder training.
                    # text_encoder_1_lora_layers_to_save=text_encoder_1_lora_layers_to_save,
                    # text_encoder_2_lora_layers_to_save=text_encoder_2_lora_layers_to_save,
                    # text_encoder_3_lora_layers_to_save=text_encoder_3_lora_layers_to_save,
                )
            else:
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
            tqdm.write("Saving EMA model")
            self.ema_model.save_pretrained(
                os.path.join(temporary_dir, self.ema_model_subdir),
                max_shard_size="10GB",
            )

        if self.unet is not None:
            sub_dir = "unet"
        if self.transformer is not None:
            sub_dir = "transformer"
        if self.args.controlnet:
            sub_dir = "controlnet"
        for model in models:
            model.save_pretrained(
                os.path.join(temporary_dir, sub_dir), max_shard_size="10GB"
            )
            merge_safetensors_files(os.path.join(temporary_dir, sub_dir))
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
            transformer_ = None
            text_encoder_one_ = None
            text_encoder_two_ = None
            text_encoder_three_ = None

            while len(models) > 0:
                model = models.pop()

                if isinstance(model, type(unwrap_model(self.accelerator, self.unet))):
                    unet_ = model
                elif isinstance(
                    model, type(unwrap_model(self.accelerator, self.transformer))
                ):
                    transformer_ = model
                elif isinstance(
                    model, type(unwrap_model(self.accelerator, self.text_encoder_1))
                ):
                    text_encoder_one_ = model
                elif isinstance(
                    model, type(unwrap_model(self.accelerator, self.text_encoder_2))
                ):
                    text_encoder_two_ = model
                elif isinstance(
                    model, type(unwrap_model(self.accelerator, self.text_encoder_3))
                ):
                    text_encoder_three_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

            if self.args.sd3:
                lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir)
                transformer_state_dict = {
                    f'{k.replace("transformer.", "")}': v
                    for k, v in lora_state_dict.items()
                    if k.startswith("unet.")
                }
                transformer_state_dict = convert_unet_state_dict_to_peft(
                    transformer_state_dict
                )
                incompatible_keys = set_peft_model_state_dict(
                    transformer_, transformer_state_dict, adapter_name="default"
                )

            else:
                lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(
                    input_dir
                )

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
                    text_encoder=text_encoder_two_,
                )
                if self.args.sd3:
                    _set_state_dict_into_text_encoder(
                        lora_state_dict,
                        prefix="text_encoder_3.",
                        text_encoder=text_encoder_three_,
                    )
            logger.info("Completed loading LoRA weights.")

        if self.args.use_ema:
            load_model = EMAModel.from_pretrained(
                os.path.join(input_dir, self.ema_model_subdir), self.ema_model_cls
            )
            self.ema_model.load_state_dict(load_model.state_dict())
            self.ema_model.to(self.accelerator.device)
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

                        merge_safetensors_files(os.path.join(input_dir, "controlnet"))
                        load_model = ControlNetModel.from_pretrained(
                            input_dir, subfolder="controlnet"
                        )
                    elif self.args.sd3:
                        # Load a stable diffusion 3 checkpoint
                        try:
                            from diffusers import SD3Transformer2DModel
                        except Exception as e:
                            logger.error(
                                f"Can not load SD3 model class. This release requires the latest version of Diffusers: {e}"
                            )
                            raise e
                        if not self.args.train_text_encoder:
                            logger.info(
                                f"Unloading text encoders for full SD3 training without --train_text_encoder"
                            )
                            (
                                self.text_encoder_1,
                                self.text_encoder_2,
                                self.text_encoder_3,
                            ) = (None, None, None)
                        load_model = SD3Transformer2DModel.from_pretrained(
                            input_dir, subfolder="transformer"
                        )
                    elif self.args.pixart_sigma:
                        # load pixart sigma checkpoint
                        load_model = PixArtTransformer2DModel.from_pretrained(
                            input_dir, subfolder="transformer"
                        )
                    elif self.unet is not None:
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
