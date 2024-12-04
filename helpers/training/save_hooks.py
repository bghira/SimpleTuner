from helpers.training.ema import EMAModel
from helpers.training.wrappers import unwrap_model
from helpers.training.multi_process import _get_rank as get_rank
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
)
from peft import set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

from helpers.models.sdxl.pipeline import StableDiffusionXLPipeline
from helpers.training.state_tracker import StateTracker
from helpers.models.smoldit import SmolDiT2DModel, SmolDiTPipeline
from helpers.models.sd3.transformer import SD3Transformer2DModel
import os
import logging
import shutil
import json
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


logger = logging.getLogger("SaveHookManager")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "WARNING"))

try:
    from diffusers import (
        UNet2DConditionModel,
        StableDiffusion3Pipeline,
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
            self.pipeline_class = StableDiffusionXLPipeline
            if StateTracker.get_model_family() == "legacy":
                self.pipeline_class = StableDiffusionPipeline
        elif self.transformer is not None:
            if args.model_family == "sd3":
                self.denoiser_class = SD3Transformer2DModel
                self.pipeline_class = StableDiffusion3Pipeline
            elif (
                args.model_family.lower() == "flux"
                and not args.flux_attention_masked_training
            ):
                self.denoiser_class = FluxTransformer2DModel
                self.pipeline_class = FluxPipeline
            elif (
                args.model_family.lower() == "flux"
                and args.flux_attention_masked_training
            ):
                from helpers.models.flux.transformer import (
                    FluxTransformer2DModelWithMasking,
                )

                self.denoiser_class = FluxTransformer2DModelWithMasking
                self.pipeline_class = FluxPipeline
            elif hasattr(args, "hunyuan_dit") and args.hunyuan_dit:
                self.denoiser_class = HunyuanDiT2DModel
                self.pipeline_class = HunyuanDiTPipeline
            elif args.model_family == "pixart_sigma":
                self.denoiser_class = PixArtTransformer2DModel
                self.pipeline_class = PixArtSigmaPipeline
            elif args.model_family == "smoldit":
                self.denoiser_class = SmolDiT2DModel
                self.pipeline_class = SmolDiTPipeline
            elif args.model_family == "sana":
                from diffusers import SanaPipeline, SanaTransformer2DModel

                self.denoiser_class = SanaTransformer2DModel
                self.pipeline_class = SanaPipeline
            self.denoiser_subdir = "transformer"

        if args.controlnet:
            self.denoiser_class = ControlNetModel
            self.denoiser_subdir = "controlnet"
        logger.debug(f"Denoiser class set to: {self.denoiser_class.__name__}.")
        logger.debug(f"Pipeline class set to: {self.pipeline_class.__name__}.")

        self.ema_model_cls = None
        self.ema_model_subdir = None
        if unet is not None:
            self.ema_model_subdir = "unet_ema"
            self.ema_model_cls = unet.__class__
        if transformer is not None:
            self.ema_model_subdir = "transformer_ema"
            self.ema_model_cls = transformer.__class__
        self.training_state_path = "training_state.json"
        if self.accelerator is not None:
            rank = get_rank()
            if rank > 0:
                self.training_state_path = f"training_state-rank{rank}.json"

    def _primary_model(self):
        if self.args.controlnet:
            return self.controlnet
        if self.unet is not None:
            return self.unet
        if self.transformer is not None:
            return self.transformer

    def _save_lora(self, models, weights, output_dir):
        # for SDXL/others, there are only two options here. Either are just the unet attn processor layers
        # or there are the unet and text encoder atten layers.
        unet_lora_layers_to_save = None
        transformer_lora_layers_to_save = None
        text_encoder_1_lora_layers_to_save = None
        text_encoder_2_lora_layers_to_save = None
        # Diffusers does not train the third text encoder.
        # text_encoder_3_lora_layers_to_save = None

        if self.args.use_ema:
            # we'll temporarily overwrite teh LoRA parameters with the EMA parameters to save it.
            logger.info("Saving EMA model to disk.")
            trainable_parameters = [
                p for p in self._primary_model().parameters() if p.requires_grad
            ]
            self.ema_model.store(trainable_parameters)
            self.ema_model.copy_to(trainable_parameters)
            if self.transformer is not None:
                self.pipeline_class.save_lora_weights(
                    os.path.join(output_dir, "ema"),
                    transformer_lora_layers=convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(self._primary_model())
                    ),
                )
            elif self.unet is not None:
                self.pipeline_class.save_lora_weights(
                    os.path.join(output_dir, "ema"),
                    unet_lora_layers=convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(self._primary_model())
                    ),
                )
            self.ema_model.restore(trainable_parameters)

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

        if self.args.model_family == "flux":
            self.pipeline_class.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_1_lora_layers_to_save,
            )
        elif self.args.model_family == "sd3":
            self.pipeline_class.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_1_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_2_lora_layers_to_save,
            )
        elif self.args.model_family == "legacy":
            self.pipeline_class.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_1_lora_layers_to_save,
            )
        elif self.args.model_family == "sdxl" or self.args.model_family == "kolors":
            self.pipeline_class.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_1_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_2_lora_layers_to_save,
            )
        else:
            raise ValueError(f"unexpected model family: {self.args.model_family}")

    def _save_lycoris(self, models, weights, output_dir):
        """
        save wrappers for lycoris. For now, text encoders are not trainable
        via lycoris.
        """
        from helpers.publishing.huggingface import (
            LORA_SAFETENSORS_FILENAME,
            EMA_SAFETENSORS_FILENAME,
        )

        for _ in models:
            if weights:
                weights.pop()

        lycoris_config = None
        with open(self.args.lycoris_config, "r") as f:
            lycoris_config = json.load(f)

        self.accelerator._lycoris_wrapped_network.save_weights(
            os.path.join(output_dir, LORA_SAFETENSORS_FILENAME),
            list(self.accelerator._lycoris_wrapped_network.parameters())[0].dtype,
            {"lycoris_config": json.dumps(lycoris_config)},  # metadata
        )
        if self.args.use_ema:
            # we'll store lycoris weights.
            self.ema_model.store(self.accelerator._lycoris_wrapped_network.parameters())
            # we'll write EMA to the lycoris adapter temporarily.
            self.ema_model.copy_to(
                self.accelerator._lycoris_wrapped_network.parameters()
            )
            # now we can write the lycoris weights using the EMA_SAFETENSORS_FILENAME instead.
            os.makedirs(os.path.join(output_dir, "ema"), exist_ok=True)
            self.accelerator._lycoris_wrapped_network.save_weights(
                os.path.join(output_dir, "ema", EMA_SAFETENSORS_FILENAME),
                list(self.accelerator._lycoris_wrapped_network.parameters())[0].dtype,
                {"lycoris_config": json.dumps(lycoris_config)},  # metadata
            )
            self.ema_model.restore(
                self.accelerator._lycoris_wrapped_network.parameters()
            )

        # copy the config into the repo
        shutil.copy2(
            self.args.lycoris_config, os.path.join(output_dir, "lycoris_config.json")
        )

        logger.info("LyCORIS weights have been saved to disk")

    def _save_full_model(self, models, weights, output_dir):
        # Create a temporary directory for atomic saves
        temporary_dir = output_dir.replace("checkpoint", "temporary")
        os.makedirs(temporary_dir, exist_ok=True)

        if self.args.use_ema and self.accelerator.is_main_process:
            # even with deepspeed, EMA should only save on the main process.
            ema_model_path = os.path.join(
                temporary_dir, self.ema_model_subdir, "ema_model.pt"
            )
            logger.info(f"Saving EMA model to {ema_model_path}")
            try:
                self.ema_model.save_state_dict(ema_model_path)
            except Exception as e:
                logger.error(f"Error saving EMA model: {e}")
            logger.info(f"Saving EMA safetensors variant.")
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
        shutil.rmtree(temporary_dir, ignore_errors=True)

    def save_model_hook(self, models, weights, output_dir):
        # Write "training_state.json" to the output directory containing the training state
        StateTracker.save_training_state(
            os.path.join(output_dir, self.training_state_path)
        )
        if not self.accelerator.is_main_process:
            return
        if self.args.use_ema:
            # we'll save this EMA checkpoint for restoring the state easier.
            ema_model_path = os.path.join(
                output_dir, self.ema_model_subdir, "ema_model.pt"
            )
            logger.info(f"Saving EMA model to {ema_model_path}")
            try:
                self.ema_model.save_state_dict(ema_model_path)
            except Exception as e:
                logger.error(f"Error saving EMA model: {e}")
        if "lora" in self.args.model_type and self.args.lora_type == "standard":
            self._save_lora(models=models, weights=weights, output_dir=output_dir)
            return
        elif "lora" in self.args.model_type and self.args.lora_type == "lycoris":
            self._save_lycoris(models=models, weights=weights, output_dir=output_dir)
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

            if isinstance(
                unwrap_model(self.accelerator, model),
                type(unwrap_model(self.accelerator, self.unet)),
            ):
                unet_ = model
                denoiser = unet_
            elif isinstance(
                unwrap_model(self.accelerator, model),
                type(unwrap_model(self.accelerator, self.transformer)),
            ):
                transformer_ = model
                denoiser = transformer_
            elif isinstance(
                unwrap_model(self.accelerator, model),
                type(unwrap_model(self.accelerator, self.text_encoder_1)),
            ):
                text_encoder_one_ = model
            elif isinstance(
                unwrap_model(self.accelerator, model),
                type(unwrap_model(self.accelerator, self.text_encoder_2)),
            ):
                text_encoder_two_ = model
            else:
                raise ValueError(
                    f"unexpected save model: {model.__class__}"
                    f"\nunwrapped: {unwrap_model(self.accelerator, model).__class__}"
                    f"\nunet: {unwrap_model(self.accelerator, self.unet).__class__}"
                )

        if self.args.model_family in ["sd3", "flux", "pixart_sigma"]:
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
            from diffusers.training_utils import _set_state_dict_into_text_encoder

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

    def _load_lycoris(self, models, input_dir):
        from helpers.publishing.huggingface import LORA_SAFETENSORS_FILENAME

        while len(models) > 0:
            model = models.pop()

        state = self.accelerator._lycoris_wrapped_network.load_weights(
            os.path.join(input_dir, LORA_SAFETENSORS_FILENAME)
        )
        if len(state.keys()) > 0:
            logging.error(f"LyCORIS failed to load: {state}")
            raise RuntimeError("Loading of LyCORIS model failed")
        weight_dtype = StateTracker.get_weight_dtype()
        if self.transformer is not None:
            self.accelerator._lycoris_wrapped_network.to(
                device=self.accelerator.device, dtype=weight_dtype
            )
        elif self.unet is not None:
            self.accelerator._lycoris_wrapped_network.to(
                device=self.accelerator.device, dtype=weight_dtype
            )
        else:
            raise ValueError("No model found to load LyCORIS weights into.")

        logger.info("LyCORIS weights have been loaded from disk")
        # disable LyCORIS spam logging
        lycoris_logger = logging.getLogger("LyCORIS")
        lycoris_logger.setLevel(logging.ERROR)

    def _load_full_model(self, models, input_dir):
        if self.args.model_type == "full":
            return_exception = False
            for i in range(len(models)):
                try:
                    # pop models so that they are not loaded again
                    model = models.pop()
                    load_model = self.denoiser_class.from_pretrained(
                        input_dir, subfolder=self.denoiser_subdir
                    )
                    if (
                        self.args.model_family == "sd3"
                        and not self.args.train_text_encoder
                    ):
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
        training_state_path = os.path.join(input_dir, self.training_state_path)
        if (
            not os.path.exists(training_state_path)
            and self.training_state_path != "training_state.json"
        ):
            logger.warning(
                f"Could not find {training_state_path} in checkpoint dir {input_dir}. Trying the default path."
            )
            training_state_path = os.path.join(input_dir, "training_state.json")
        if os.path.exists(training_state_path):
            StateTracker.load_training_state(training_state_path)
        else:
            logger.warning(
                f"Could not find {training_state_path} in checkpoint dir {input_dir}"
            )
        if self.args.use_ema and self.accelerator.is_main_process:
            try:
                self.ema_model.load_state_dict(
                    os.path.join(input_dir, self.ema_model_subdir, "ema_model.pt")
                )
                # self.ema_model.to(self.accelerator.device)
            except Exception as e:
                logger.error(f"Could not load EMA model: {e}")
        if "lora" in self.args.model_type and self.args.lora_type == "standard":
            self._load_lora(models=models, input_dir=input_dir)
        elif "lora" in self.args.model_type and self.args.lora_type == "lycoris":
            self._load_lycoris(models=models, input_dir=input_dir)
        else:
            self._load_full_model(models=models, input_dir=input_dir)
