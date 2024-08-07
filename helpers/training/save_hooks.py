from diffusers.training_utils import EMAModel, _set_state_dict_into_text_encoder
from helpers.training.wrappers import unwrap_model
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
)
from peft import set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

from helpers.sdxl.pipeline import StableDiffusionXLPipeline
from helpers.training.state_tracker import StateTracker
from helpers.models.smoldit import SmolDiT2DModel, SmolDiTPipeline
import os
import logging
import shutil
import json
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

logger = logging.getLogger("SaveHookManager")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL") or "INFO")

try:
    from diffusers import (
        UNet2DConditionModel,
        StableDiffusion3Pipeline,
        SD3Transformer2DModel,
        StableDiffusionPipeline,
        FluxPipeline,
        PixArtSigmaPipeline,
        ControlNetModel,
        HunyuanDiTPipeline,
    )
except ImportError:
    logger.error("This release requires the latest version of Diffusers.")

try:
    from diffusers.models import PixArtTransformer2DModel
except Exception as e:
    logger.error(
        f"Can not load Pixart Sigma model class. This release requires the latest version of Diffusers: {e}"
    )
    raise e

try:
    from diffusers.models import FluxTransformer2DModel
except Exception as e:
    logger.error(
        f"Can not load FluxTransformer2DModel model class. This release requires the latest version of Diffusers: {e}"
    )
    raise e

try:
    from diffusers.models import HunyuanDiT2DModel
except Exception as e:
    logger.error(
        f"Can not load Hunyuan DiT model class. This release requires the latest version of Diffusers: {e}"
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


class SaveHookManager:
    def __init__(
        self,
        args,
        unet,
        transformer,
        ema_model,
        text_encoder_1,
        text_encoder_2,
        accelerator,
        use_deepspeed_optimizer,
    ):

        self.args = args
        self.unet = unet
        self.transformer = transformer
        if self.unet is not None and self.transformer is not None:
            raise ValueError("Both `unet` and `transformer` cannot be set.")
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.ema_model = ema_model
        self.accelerator = accelerator
        self.use_deepspeed_optimizer = use_deepspeed_optimizer

        self.denoiser_class = None
        self.denoiser_subdir = None
        self.pipeline_class = None
        if self.unet is not None:
            self.denoiser_class = UNet2DConditionModel
            self.denoiser_subdir = "unet"
            self.pipeline_class = (
                StableDiffusionXLPipeline
                if StateTracker.get_model_type() == "sdxl"
                else StableDiffusionPipeline
            )
        elif self.transformer is not None:
            if args.sd3:
                self.denoiser_class = SD3Transformer2DModel
                self.pipeline_class = StableDiffusion3Pipeline
            elif args.flux:
                self.denoiser_class = FluxTransformer2DModel
                self.pipeline_class = FluxPipeline
            elif args.hunyuan_dit:
                self.denoiser_class = HunyuanDiT2DModel
                self.pipeline_class = HunyuanDiTPipeline
            elif args.pixart:
                self.denoiser_class = PixArtTransformer2DModel
                self.pipeline_class = PixArtSigmaPipeline
            elif args.smoldit:
                self.denoiser_class = SmolDiT2DModel
                self.pipeline_class = SmolDiTPipeline
            self.denoiser_subdir = "transformer"

        if args.controlnet is not None:
            self.denoiser_class = ControlNetModel
            self.denoiser_subdir = "controlnet"
        logger.info(f"Denoiser class set to: {self.denoiser_class.__name__}.")
        logger.info(f"Pipeline class set to: {self.pipeline_class.__name__}.")

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

    def _save_lora(self, models, weights, output_dir):
        # for SDXL/others, there are only two options here. Either are just the unet attn processor layers
        # or there are the unet and text encoder atten layers.
        unet_lora_layers_to_save = None
        transformer_lora_layers_to_save = None
        text_encoder_1_lora_layers_to_save = None
        text_encoder_2_lora_layers_to_save = None
        # Diffusers does not train the third text encoder.
        # text_encoder_3_lora_layers_to_save = None

        for model in models:
            if isinstance(model, type(unwrap_model(self.accelerator, self.unet))):
                unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(model)
                )
            elif isinstance(
                model, type(unwrap_model(self.accelerator, self.text_encoder_1))
            ):
                text_encoder_1_lora_layers_to_save = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(model)
                )
            elif isinstance(
                model, type(unwrap_model(self.accelerator, self.text_encoder_2))
            ):
                text_encoder_2_lora_layers_to_save = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(model)
                )

            elif not isinstance(
                model, type(unwrap_model(self.accelerator, HunyuanDiT2DModel))
            ):
                if isinstance(
                    model, type(unwrap_model(self.accelerator, self.transformer))
                ):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)

            elif not self.use_deepspeed_optimizer:
                raise ValueError(f"unexpected save model: {model.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            if weights:
                weights.pop()

        if self.args.flux:
            self.pipeline_class.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_1_lora_layers_to_save,
            )
        elif self.args.sd3:
            self.pipeline_class.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_1_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_2_lora_layers_to_save,
            )
        elif self.args.legacy:
            self.pipeline_class.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_1_lora_layers_to_save,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )
        else:
            self.pipeline_class.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_1_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_2_lora_layers_to_save,
            )

    def _save_full_model(self, models, weights, output_dir):
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

    def save_model_hook(self, models, weights, output_dir):
        # Write "training_state.json" to the output directory containing the training state
        StateTracker.save_training_state(
            os.path.join(output_dir, "training_state.json")
        )
        if "lora" in self.args.model_type:
            self._save_lora(models=models, weights=weights, output_dir=output_dir)
            return
        else:
            self._save_full_model(models=models, weights=weights, output_dir=output_dir)

    def _load_lora(self, models, input_dir):
        logger.info(f"Loading LoRA weights from Path: {input_dir}")
        unet_ = None
        transformer_ = None
        denoiser = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(self.accelerator, self.unet))):
                unet_ = model
                denoiser = unet_
            elif isinstance(
                model, type(unwrap_model(self.accelerator, self.transformer))
            ):
                transformer_ = model
                denoiser = transformer_
            elif isinstance(
                model, type(unwrap_model(self.accelerator, self.text_encoder_1))
            ):
                text_encoder_one_ = model
            elif isinstance(
                model, type(unwrap_model(self.accelerator, self.text_encoder_2))
            ):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        if self.args.sd3 or self.args.flux:
            key_to_replace = "transformer"
            lora_state_dict = self.pipeline_class.lora_state_dict(input_dir)
        else:
            key_to_replace = "unet"
            lora_state_dict, _ = self.pipeline_class.lora_state_dict(input_dir)

        denoiser_state_dict = {
            f'{k.replace(f"{key_to_replace}.", "")}': v
            for k, v in lora_state_dict.items()
            if k.startswith(f"{key_to_replace}.")
        }
        denoiser_state_dict = convert_unet_state_dict_to_peft(denoiser_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            denoiser, denoiser_state_dict, adapter_name="default"
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

        logger.info("Completed loading LoRA weights.")

    def _load_full_model(self, models, input_dir):
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
                    if self.args.controlnet or self.args.unet:
                        merge_safetensors_files(
                            os.path.join(input_dir, self.denoiser_subdir)
                        )

                    load_model = self.denoiser_class.from_pretrained(
                        input_dir, subfolder=self.denoiser_subdir
                    )
                    if self.args.sd3 and not self.args.train_text_encoder:
                        logger.info(
                            "Unloading text encoders for full SD3 training without --train_text_encoder"
                        )
                        (self.text_encoder_1, self.text_encoder_2) = (None, None)

                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                except Exception as e:
                    import traceback

                    return_exception = f"Could not load model: {e}, traceback: {traceback.format_exc()}"

            if return_exception:
                raise Exception(return_exception)

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
            self._load_lora(models=models, input_dir=input_dir)
        else:
            self._load_full_model(models=models, input_dir=input_dir)
