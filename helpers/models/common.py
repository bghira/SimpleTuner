import torch
import inspect
import torch.nn.functional as F
import random
import logging
import inspect
import os
from torchvision import transforms
from diffusers import DiffusionPipeline
from torch.distributions import Beta
from helpers.training.wrappers import unwrap_model
from helpers.training.multi_process import _get_rank
from helpers.training.custom_schedule import (
    apply_flow_schedule_shift,
    generate_timestep_weights,
    segmented_timestep_selection,
)
from helpers.training.min_snr_gamma import compute_snr
from transformers.utils import ContextManagers
from helpers.training.adapter import load_lora_weights
from helpers.training.deepspeed import (
    deepspeed_zero_init_disabled_context_manager,
    prepare_model_for_deepspeed,
)
from abc import ABC, abstractmethod
from enum import Enum
from peft import LoraConfig
import numpy as np
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)
is_primary_process = True
if os.environ.get("RANK") is not None:
    if int(os.environ.get("RANK")) != 0:
        is_primary_process = False
logger.setLevel(
    os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO") if is_primary_process else "ERROR"
)

flow_matching_model_families = ["flux", "sana", "ltxvideo", "wan", "sd3"]
upstream_config_sources = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "kolors": "terminusresearch/kwai-kolors-1.0",
    "sd3": "stabilityai/stable-diffusion-3-large",
    "sana": "terminusresearch/sana-1.6b-1024px",
    "flux": "black-forest-labs/flux.1-dev",
    "sd1x": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd2x": "stabilityai/stable-diffusion-v2-1",
    "ltxvideo": "Lightricks/LTX-Video",
    "wan": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
}


def get_model_config_path(model_family: str, model_path: str):
    if model_path is not None and model_path.endswith(".safetensors"):
        if model_family in upstream_config_sources:
            return upstream_config_sources[model_family]
        else:
            raise ValueError(
                "Cannot find noise schedule config for .safetensors file in architecture {}".format(
                    model_family
                )
            )

    return model_path


class PipelineTypes(Enum):
    IMG2IMG = "img2img"
    TEXT2IMG = "text2img"
    CONTROLNET = "controlnet"
    CONTROL = "control"


class PredictionTypes(Enum):
    EPSILON = "epsilon"
    SAMPLE = "sample"
    V_PREDICTION = "v_prediction"
    FLOW_MATCHING = "flow_matching"

    @staticmethod
    def from_str(label):
        if label in ("eps", "epsilon"):
            return PredictionTypes.EPSILON
        elif label in ("vpred", "v_prediction", "v-prediction"):
            return PredictionTypes.V_PREDICTION
        elif label in ("sample", "x_prediction", "x-prediction"):
            return PredictionTypes.SAMPLE
        elif label in ("flow", "flow_matching", "flow-matching"):
            return PredictionTypes.FLOW_MATCHING
        else:
            raise NotImplementedError


class ModelTypes(Enum):
    UNET = "unet"
    TRANSFORMER = "transformer"
    VAE = "vae"
    TEXT_ENCODER = "text_encoder"


class ModelFoundation(ABC):
    """
    Base class that contains all the universal logic:
      - Noise schedule, prediction target (epsilon, sample, v_prediction, flow-matching)
      - Batch preparation (moving to device, sampling noise, etc.)
      - Loss calculation (including optional SNR weighting)
    """

    MODEL_LICENSE = "other"
    CONTROLNET_LORA_STATE_DICT_PREFIX = "controlnet"

    def __init__(self, config: dict, accelerator):
        self.config = config
        self.accelerator = accelerator
        self.noise_schedule = None
        self.pipelines = {}
        self._qkv_projections_fused = False
        self.setup_model_flavour()
        self.setup_training_noise_schedule()

    def log_model_devices(self):
        """
        Log the devices of the model components.
        """
        if hasattr(self, "model") and self.model is not None:
            logger.debug(f"Model device: {self.model.device}")
        if hasattr(self, "vae") and self.vae is not None:
            logger.debug(f"VAE device: {self.vae.device}")
        if hasattr(self, "text_encoders") and self.text_encoders is not None:
            for i, text_encoder in enumerate(self.text_encoders):
                if text_encoder is None:
                    continue
                logger.debug(f"Text encoder {i} device: {text_encoder.device}")

    def setup_model_flavour(self):
        """
        Sets up the model flavour based on the config.
        This is used to determine the model path if none was provided.
        """
        if getattr(self, "REQUIRES_FLAVOUR", False):
            if getattr(self.config, "model_flavour", None) is None:
                raise ValueError(
                    f"{str(self.__class__)} models require model_flavour to be provided."
                    f" Possible values: {self.HUGGINGFACE_PATHS.keys()}"
                )
        if self.config.pretrained_model_name_or_path is None:
            if self.config.model_flavour is None:
                default_flavour = getattr(self, "DEFAULT_MODEL_FLAVOUR", None)
                if default_flavour is None and len(self.HUGGINGFACE_PATHS) > 0:
                    raise ValueError(
                        f"The current model family {self.config.model_family} requires a model_flavour to be provided. Options: {self.HUGGINGFACE_PATHS.keys()}"
                    )
                elif default_flavour is not None:
                    self.config.model_flavour = default_flavour
            if self.config.model_flavour is not None:
                if self.config.model_flavour not in self.HUGGINGFACE_PATHS:
                    raise ValueError(
                        f"Model flavour {self.config.model_flavour} not found in {self.HUGGINGFACE_PATHS.keys()}"
                    )
                self.config.pretrained_model_name_or_path = self.HUGGINGFACE_PATHS.get(
                    self.config.model_flavour
                )
            else:
                raise ValueError(
                    f"Model flavour {self.config.model_flavour} not found in {self.HUGGINGFACE_PATHS.keys()}"
                )
        if self.config.pretrained_vae_model_name_or_path is None:
            self.config.pretrained_vae_model_name_or_path = (
                self.config.pretrained_model_name_or_path
            )
        if self.config.vae_path is None:
            self.config.vae_path = self.config.pretrained_model_name_or_path

    @abstractmethod
    def model_predict(self, prepared_batch, custom_timesteps: list = None):
        """
        Run a forward pass on the model.
        Must be implemented by the subclass.
        """
        raise NotImplementedError(
            "model_predict must be implemented in the child class."
        )

    def requires_conditioning_dataset(self) -> bool:
        # Most models don't require a conditioning dataset.
        if self.config.controlnet or self.config.control:
            return True
        return False

    def requires_conditioning_latents(self) -> bool:
        # ControlNet requires pixel inputs instead of encoded latents.
        return False

    def requires_validation_edit_captions(self) -> bool:
        """
        Some edit / in-painting models want the *reference* image plus the
        *edited* caption.  Override to return True when that is the case.
        """
        return False

    def requires_conditioning_validation_inputs(self) -> bool:
        # Whether this model / flavour requires conditioning inputs during validation.
        return False

    def conditioning_validation_dataset_type(self) -> bool:
        # Most conditioning inputs (ControlNet) etc require "conditioning" dataset.
        return "conditioning"

    def validation_image_input_edge_length(self):
        # If a model requires a specific input edge length (HiDream E1 -> 768px, DeepFloyd stage2 -> 64px)
        return None

    def control_init(self):
        """
        Initialize the channelwise Control model.
        This is distinct from ControlNet.
        This is a stub and should be implemented in subclasses.
        """
        raise NotImplementedError(
            "control_init must be implemented in the child class."
        )

    def controlnet_init(self):
        """
        Initialize the controlnet model.
        This is a stub and should be implemented in subclasses.
        """
        raise NotImplementedError(
            "controlnet_init must be implemented in the child class."
        )

    def controlnet_predict(self, prepared_batch, custom_timesteps: list = None):
        """
        Run a forward pass on the model.
        Must be implemented by the subclass.
        """
        raise NotImplementedError(
            "model_predict must be implemented in the child class."
        )

    @abstractmethod
    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encodes a batch of text using the text encoder.
        Must be implemented by the subclass.
        """
        raise NotImplementedError(
            "_encode_prompts must be implemented in the child class."
        )

    @abstractmethod
    def convert_text_embed_for_pipeline(
        self, text_embedding: torch.Tensor, prompt: str
    ) -> dict:
        """
        Converts the text embedding to the format expected by the pipeline.
        This is a stub and should be implemented in subclasses.

        Prompt may be useful to inspect if your pipeline requires eg. zeroing empty inputs.
        """
        raise NotImplementedError(
            "convert_text_embed_for_pipeline must be implemented in the child class."
        )

    @abstractmethod
    def convert_negative_text_embed_for_pipeline(
        self, text_embedding: torch.Tensor, prompt: str
    ) -> dict:
        """
        Converts the text embedding to the format expected by the pipeline for negative prompt inputs.
        This is a stub and should be implemented in subclasses.

        Prompt may be useful to inspect if your pipeline requires eg. zeroing empty inputs.
        """
        raise NotImplementedError(
            "convert_text_embed_for_pipeline must be implemented in the child class."
        )

    def collate_prompt_embeds(self, text_encoder_output: dict) -> dict:
        """
        Optional stub method for client classes to do their own text embed collation/stacking.

        Returns a dictionary. If the dictionary is empty, it is ignored and usual collate occurs.
        """
        return {}

    @classmethod
    def get_flavour_choices(cls):
        """
        Returns the available model flavours for this model.
        """
        return list(cls.HUGGINGFACE_PATHS.keys())

    def get_transforms(self, dataset_type: str = "image"):
        """
        Returns nothing, but subclasses can implement different torchvision transforms as needed.

        dataset_type is passed in for models that support transforming videos or images etc.
        """
        if dataset_type in ["video"]:
            raise ValueError(
                f"{dataset_type} transforms are not supported by {self.NAME}."
            )
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def load_lora_weights(self, models, input_dir):
        """
        Generalized LoRA loading method.
        1) Pop models from the 'models' list, detect which is main (denoiser) vs. text encoders.
        2) Pull the relevant LoRA keys out of the pipeline's lora_state_dict() output by prefix.
        3) Convert & load them into the unwrapped PyTorch modules with set_peft_model_state_dict().
        4) Optionally handle text_encoder_x using the diffusers _set_state_dict_into_text_encoder() helper.
        """
        # We'll track whichever sub-model is our 'denoiser' (UNet or Transformer).
        denoiser = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        # 1. Identify models by type comparison
        while len(models) > 0:
            model = models.pop()
            unwrapped_model = self.unwrap_model(model)

            # Compare unwrapped type to main model's unwrapped type
            if isinstance(unwrapped_model, type(self.unwrap_model(self.model))):
                denoiser = model  # e.g., the "transformer" or "unet"
            elif isinstance(unwrapped_model, type(self.unwrap_model(self.controlnet))):
                denoiser = model  # e.g., the "controlnet"
            # If your text_encoders exist:
            elif (
                getattr(self, "text_encoders", None)
                and len(self.text_encoders) > 0
                and isinstance(
                    unwrapped_model, type(self.unwrap_model(self.text_encoders[0]))
                )
            ):
                text_encoder_one_ = model
            elif (
                getattr(self, "text_encoders", None)
                and len(self.text_encoders) > 1
                and isinstance(
                    unwrapped_model, type(self.unwrap_model(self.text_encoders[1]))
                )
            ):
                text_encoder_two_ = model
            else:
                raise ValueError(
                    f"Unexpected model type in load_lora_weights: {model.__class__}\n"
                    f"Unwrapped: {unwrapped_model.__class__}\n"
                    f"Expected main model type {type(self.unwrap_model(self.model))}"
                )

        # 2. Get the LoRA state dict from the pipeline's directory
        #    Usually you call something like self.PIPELINE_CLASSES[...] but
        #    whichever pipeline is relevant. For example:
        pipeline_cls = self.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG]
        lora_state_dict = pipeline_cls.lora_state_dict(input_dir)
        if (
            type(lora_state_dict) is tuple
            and len(lora_state_dict) == 2
            and lora_state_dict[1] is None
        ):
            logger.debug("Overriding ControlNet LoRA state dict with correct structure")
            lora_state_dict = lora_state_dict[0]

        # 3. Extract keys for the main model (which uses self.MODEL_TYPE.value as the prefix)
        #    For example, "transformer." or "unet." is stripped out.
        key_to_replace = (
            self.CONTROLNET_LORA_STATE_DICT_PREFIX
            if self.config.controlnet
            else self.MODEL_TYPE.value
        )
        prefix = f"{key_to_replace}."
        denoiser_sd = {}
        for k, v in lora_state_dict.items():
            if k.startswith(prefix):
                new_key = k.replace(prefix, "")
                denoiser_sd[new_key] = v

        # Convert them to "PEFT" format if needed.
        # (Typically we call convert_unet_state_dict_to_peft on the denoiser's keys.)
        from diffusers.utils import convert_unet_state_dict_to_peft

        denoiser_sd = convert_unet_state_dict_to_peft(denoiser_sd)

        # 4. Load them into the denoiser (the main model)
        from peft.utils import set_peft_model_state_dict

        incompatible_keys = set_peft_model_state_dict(
            denoiser, denoiser_sd, adapter_name="default"
        )

        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"LoRA loading found unexpected keys not in the denoiser model: {unexpected_keys}"
                )

        # 5. If we also train text encoders, apply them
        if getattr(self.config, "train_text_encoder", False):
            from diffusers.training_utils import _set_state_dict_into_text_encoder

            # For text_encoder_1, the prefix in your pipeline's state dict is usually "text_encoder."
            # For text_encoder_2, it might be "text_encoder_2."
            # We'll do them in separate calls:

            if text_encoder_one_ is not None:
                _set_state_dict_into_text_encoder(
                    lora_state_dict,
                    prefix="text_encoder.",  # Must match how your pipeline outputs these
                    text_encoder=text_encoder_one_,
                )

            if text_encoder_two_ is not None:
                _set_state_dict_into_text_encoder(
                    lora_state_dict,
                    prefix="text_encoder_2.",  # Must match how your pipeline organizes these
                    text_encoder=text_encoder_two_,
                )

        logger.info("Finished loading LoRA weights successfully.")

    def save_lora_weights(self, *args, **kwargs):
        self.PIPELINE_CLASSES[
            (
                PipelineTypes.TEXT2IMG
                if not self.config.controlnet
                else PipelineTypes.CONTROLNET
            )
        ].save_lora_weights(*args, **kwargs)

    def pre_ema_creation(self):
        """
        A hook that can be overridden in the subclass to perform actions before EMA creation.
        """
        self.fuse_qkv_projections()

    def post_ema_creation(self):
        """
        A hook that can be overridden in the subclass to perform actions after EMA creation.
        """
        pass

    def check_user_config(self):
        """
        Checks self.config values against important issues. Optionally implemented in child class.
        """
        pass

    def _model_config_path(self):
        return get_model_config_path(
            model_family=self.config.model_family,
            model_path=self.config.pretrained_model_name_or_path,
        )

    def unwrap_model(self, model=None):
        if self.config.controlnet and model is None:
            return unwrap_model(self.accelerator, self.controlnet)
        return unwrap_model(self.accelerator, model or self.model)

    def move_extra_models(self, target_device):
        """
        Move any extra models in the child class.

        This is a stub and can be optionally implemented in subclasses.
        """
        pass

    def move_models(self, target_device):
        """
        Moves the model to the target device.
        """
        if self.model is not None:
            self.model.to(target_device)
        if self.controlnet is not None:
            self.controlnet.to(target_device)
        if self.vae is not None and self.vae.device != "meta":
            self.vae.to(target_device)
        if self.text_encoders is not None:
            for text_encoder in self.text_encoders:
                if text_encoder.device != "meta":
                    text_encoder.to(target_device)
        self.move_extra_models(target_device)

    def get_vae(self):
        """
        Returns the VAE model.
        """
        if not getattr(self, "AUTOENCODER_CLASS", None):
            return
        if (
            not hasattr(self, "vae")
            or self.vae is None
            or getattr(self.vae, "device", None) == "meta"
        ):
            self.load_vae()
        return self.vae

    def load_vae(self, move_to_device: bool = True):
        if not getattr(self, "AUTOENCODER_CLASS", None):
            return

        logger.info(
            f"Loading {self.AUTOENCODER_CLASS.__name__} from {self.config.vae_path}"
        )
        self.vae = None
        self.config.vae_kwargs = {
            "pretrained_model_name_or_path": get_model_config_path(
                self.config.model_family, self.config.vae_path
            ),
            "subfolder": "vae",
            "revision": self.config.revision,
            "force_upcast": False,
            "variant": self.config.variant,
        }
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            try:
                self.vae = self.AUTOENCODER_CLASS.from_pretrained(
                    **self.config.vae_kwargs
                )
            except Exception as e:
                logger.warning(
                    "Couldn't load VAE with default path. Trying without a subfolder.."
                )
                logger.error(e)
                self.config.vae_kwargs["subfolder"] = None
                self.vae = self.AUTOENCODER_CLASS.from_pretrained(
                    **self.config.vae_kwargs
                )
        if self.vae is None:
            raise ValueError(
                "Could not load VAE. Please check the model path and ensure the VAE is compatible."
            )
        if self.config.vae_enable_tiling:
            if hasattr(self.vae, "enable_tiling"):
                logger.info("Enabling VAE tiling.")
                self.vae.enable_tiling()
            else:
                logger.warning(
                    f"VAE tiling is enabled, but not yet supported by {self.config.model_family}."
                )
        if self.config.vae_enable_slicing:
            if hasattr(self.vae, "enable_slicing"):
                logger.info("Enabling VAE slicing.")
                self.vae.enable_slicing()
            else:
                logger.warning(
                    f"VAE slicing is enabled, but not yet supported by {self.config.model_family}."
                )
        if move_to_device and self.vae.device != self.accelerator.device:
            # The VAE is in bfloat16 to avoid NaN losses.
            _vae_dtype = torch.bfloat16
            if hasattr(self.config, "vae_dtype"):
                # Let's use a case-switch for convenience: bf16, fp16, fp32, none/default
                if self.config.vae_dtype == "bf16":
                    _vae_dtype = torch.bfloat16
                elif self.config.vae_dtype == "fp16":
                    raise ValueError(
                        "fp16 is not supported for SDXL's VAE. Please use bf16 or fp32."
                    )
                elif self.config.vae_dtype == "fp32":
                    _vae_dtype = torch.float32
                elif (
                    self.config.vae_dtype == "none"
                    or self.config.vae_dtype == "default"
                ):
                    _vae_dtype = torch.bfloat16
            logger.info(
                f"Moving {self.AUTOENCODER_CLASS.__name__} to accelerator, converting from {self.vae.dtype} to {_vae_dtype}"
            )
            self.vae.to(self.accelerator.device, dtype=_vae_dtype)
        self.AUTOENCODER_SCALING_FACTOR = getattr(
            self.vae.config, "scaling_factor", 1.0
        )
        self.post_vae_load_setup()

    def post_vae_load_setup(self):
        """
        Post VAE load setup.

        This is a stub and can be optionally implemented in subclasses for eg. updating configuration settings
        based on the loaded VAE weights. SDXL uses this to update the user config to reflect refiner training.

        """
        pass

    def pre_vae_encode_transform_sample(self, sample):
        """
        Pre-encode transform for the sample before passing it to the VAE.
        This is a stub and can be optionally implemented in subclasses.
        """
        return sample

    def post_vae_encode_transform_sample(self, sample):
        """
        Post-encode transform for the sample after passing it to the VAE.
        This is a stub and can be optionally implemented in subclasses.
        """
        return sample

    def unload_vae(self):
        if self.vae is not None:
            if hasattr(self.vae, "to"):
                self.vae.to("meta")
            self.vae = None

    def load_text_tokenizer(self):
        if (
            self.TEXT_ENCODER_CONFIGURATION is None
            or len(self.TEXT_ENCODER_CONFIGURATION) == 0
        ):
            return
        self.tokenizers = []
        tokenizer_kwargs = {
            "subfolder": "tokenizer",
            "revision": self.config.revision,
            "use_fast": False,
        }
        tokenizer_idx = 0
        for attr_name, text_encoder_config in self.TEXT_ENCODER_CONFIGURATION.items():
            tokenizer_idx += 1
            tokenizer_cls = text_encoder_config.get("tokenizer")
            tokenizer_kwargs["subfolder"] = text_encoder_config.get(
                "tokenizer_subfolder", "tokenizer"
            )
            tokenizer_kwargs["use_fast"] = text_encoder_config.get("use_fast", False)
            tokenizer_kwargs["pretrained_model_name_or_path"] = get_model_config_path(
                self.config.model_family, self.config.pretrained_model_name_or_path
            )
            if text_encoder_config.get("path", None) is not None:
                tokenizer_kwargs["pretrained_model_name_or_path"] = (
                    text_encoder_config.get("path")
                )
            logger.info(
                f"Loading tokenizer {tokenizer_idx}: {tokenizer_cls.__name__} with args: {tokenizer_kwargs}"
            )
            tokenizer = tokenizer_cls.from_pretrained(**tokenizer_kwargs)
            self.tokenizers.append(tokenizer)
            setattr(self, f"tokenizer_{tokenizer_idx}", tokenizer)

    def load_text_encoder(self, move_to_device: bool = True):
        self.text_encoders = []
        if (
            self.TEXT_ENCODER_CONFIGURATION is None
            or len(self.TEXT_ENCODER_CONFIGURATION) == 0
        ):
            return
        self.load_text_tokenizer()

        text_encoder_idx = 0
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            for (
                attr_name,
                text_encoder_config,
            ) in self.TEXT_ENCODER_CONFIGURATION.items():
                text_encoder_idx += 1
                # load_tes returns a variant and three text encoders
                signature = inspect.signature(text_encoder_config["model"])
                extra_kwargs = {}
                if "torch_dtype" in signature.parameters:
                    extra_kwargs["torch_dtype"] = self.config.weight_dtype
                logger.info(f"Loading {text_encoder_config.get('name')} text encoder")
                text_encoder_path = get_model_config_path(
                    self.config.model_family, self.config.pretrained_model_name_or_path
                )
                if text_encoder_config.get("path", None) is not None:
                    text_encoder_path = text_encoder_config.get("path")
                requires_quant = text_encoder_config.get(
                    "required_quantisation_level", None
                )
                if requires_quant is not None and requires_quant == "int4_weight_only":
                    # we'll use the QuantizationConfig.
                    from torchao.quantization import Int4WeightOnlyConfig
                    from transformers import TorchAoConfig

                    extra_kwargs["device_map"] = "auto"
                    quant_config = Int4WeightOnlyConfig(group_size=128)
                    extra_kwargs["quantization_config"] = TorchAoConfig(
                        quant_type=quant_config
                    )

                text_encoder = text_encoder_config["model"].from_pretrained(
                    text_encoder_path,
                    variant=self.config.variant,
                    revision=self.config.revision,
                    subfolder=text_encoder_config.get("subfolder", "text_encoder")
                    or "",
                    **extra_kwargs,
                )
                if text_encoder.__class__.__name__ in [
                    "UMT5EncoderModel",
                    "T5EncoderModel",
                ]:
                    # maybe the user enabled NovelAI T5.
                    # if self.config.t5_encoder_implementation == "novelai":
                    #     from helpers.models.t5 import NovelAIT5EncoderModel

                    #     logger.info(
                    #         f"Converting {text_encoder.__class__.__name__} to NovelAI T5 implementation.."
                    #     )
                    #     text_encoder = NovelAIT5EncoderModel.from_hf_model(text_encoder)
                    pass

                if move_to_device and getattr(
                    self.config, f"{attr_name}_precision", None
                ) in ["no_change", None]:
                    logger.info(f"Moving {text_encoder_config.get('name')} to GPU")
                    text_encoder.to(
                        self.accelerator.device,
                        dtype=self.config.weight_dtype,
                    )
                setattr(self, f"text_encoder_{text_encoder_idx}", text_encoder)
                self.text_encoders.append(text_encoder)

    def get_text_encoder(self, index: int):
        if self.text_encoders is not None:
            return self.text_encoders[index] if index in self.text_encoders else None

    def unload_text_encoder(self):
        if self.text_encoders is not None:
            for text_encoder in self.text_encoders:
                if hasattr(text_encoder, "to"):
                    text_encoder.to("meta")
            self.text_encoders = None
        if self.tokenizers is not None:
            self.tokenizers = None

    def pretrained_load_args(self, pretrained_load_args: dict) -> dict:
        """
        A stub method for child classes to augment pretrained class load arguments with.
        """
        return pretrained_load_args

    def load_model(self, move_to_device: bool = True):
        pretrained_load_args = {
            "revision": self.config.revision,
            "variant": self.config.variant,
            "torch_dtype": self.config.weight_dtype,
            "use_safetensors": True,
        }
        if "nf4-bnb" == self.config.base_model_precision:
            from diffusers import BitsAndBytesConfig

            pretrained_load_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.config.weight_dtype,
            )
        loader_fn = self.MODEL_CLASS.from_pretrained
        model_path = (
            self.config.pretrained_transformer_model_name_or_path
            if self.MODEL_TYPE is ModelTypes.TRANSFORMER
            else self.config.pretrained_unet_model_name_or_path
        ) or self.config.pretrained_model_name_or_path
        if self.config.pretrained_model_name_or_path.endswith(".safetensors"):
            self.config.pretrained_model_name_or_path = get_model_config_path(
                self.config.model_family, model_path
            )
        if model_path.endswith(".safetensors"):
            loader_fn = self.MODEL_CLASS.from_single_file
        pretrained_load_args = self.pretrained_load_args(pretrained_load_args)
        model_subfolder = self.MODEL_SUBFOLDER
        if (
            self.MODEL_TYPE is ModelTypes.TRANSFORMER
            and self.config.pretrained_transformer_model_name_or_path == model_path
        ):
            # we're using a custom transformer, let's check its subfolder
            if str(self.config.pretrained_transformer_subfolder).lower() == "none":
                model_subfolder = None
            elif str(self.config.pretrained_unet_model_name_or_path).lower() is None:
                model_subfolder = self.MODEL_SUBFOLDER
            else:
                model_subfolder = self.config.pretrained_transformer_subfolder
        elif (
            self.MODEL_TYPE is ModelTypes.UNET
            and self.config.pretrained_unet_model_name_or_path == model_path
        ):
            # we're using a custom transformer, let's check its subfolder
            if str(self.config.pretrained_unet_model_name_or_path).lower() == "none":
                model_subfolder = None
            elif str(self.config.pretrained_unet_model_name_or_path).lower() is None:
                model_subfolder = self.MODEL_SUBFOLDER
            else:
                model_subfolder = self.config.pretrained_unet_subfolder

        logger.info(f"Loading diffusion model from {model_path}")
        self.model = loader_fn(
            model_path,
            subfolder=model_subfolder,
            **pretrained_load_args,
        )
        if move_to_device and self.model is not None:
            self.model.to(self.accelerator.device, dtype=self.config.weight_dtype)
        if (
            self.config.gradient_checkpointing_interval is not None
            and self.config.gradient_checkpointing_interval > 0
            and self.MODEL_TYPE is ModelTypes.UNET
        ):
            logger.warning(
                "Using experimental gradient checkpointing monkeypatch for a checkpoint interval of {}".format(
                    self.config.gradient_checkpointing_interval
                )
            )
            # monkey-patch the gradient checkpointing function for pytorch to run every nth call only.
            # definitely one of the more awful things I've ever done while programming, but it's easier than
            # modifying every one of the unet blocks' forward calls in Diffusers to make it work properly.
            from helpers.training.gradient_checkpointing_interval import (
                set_checkpoint_interval,
            )

            set_checkpoint_interval(int(self.config.gradient_checkpointing_interval))

        if (
            self.config.gradient_checkpointing_interval is not None
            and self.config.gradient_checkpointing_interval > 1
        ):
            if self.model is not None and hasattr(
                self.model, "set_gradient_checkpointing_interval"
            ):
                logger.info("Setting gradient checkpointing interval..")
                self.model.set_gradient_checkpointing_interval(
                    int(self.config.gradient_checkpointing_interval)
                )
        self.fuse_qkv_projections()
        self.post_model_load_setup()

    def post_model_load_setup(self):
        """
        Post model load setup.

        This is a stub and can be optionally implemented in subclasses for eg. updating configuration settings
        based on the loaded model weights. SDXL uses this to update the user config to reflect refiner training.

        """
        pass

    def fuse_qkv_projections(self):
        if self.config.fuse_qkv_projections:
            logger.warning(
                f"{self.__class__.__name__} does not support fused QKV projection yet, please open a feature request on the issue tracker."
            )

    def unfuse_qkv_projections(self):
        """
        Unfuse QKV projections before critical operations like saving.
        This is a no-op by default, but subclasses can override to implement
        proper unfusing when using fused QKV projections.

        Should be called before:
        - Saving LoRA weights
        - Saving full model checkpoints
        - Any operation that expects separate Q, K, V projections
        """
        pass

    def set_prepared_model(self, model, base_model: bool = False):
        # after accelerate prepare, we'll set the model again.
        if self.config.controlnet and not base_model:
            self.controlnet = model
        else:
            self.model = model

    def freeze_components(self):
        # Freeze vae and text_encoders
        if self.vae is not None:
            self.vae.requires_grad_(False)
        if self.text_encoders is not None and len(self.text_encoders) > 0:
            for text_encoder in self.text_encoders:
                text_encoder.requires_grad_(False)
        if "lora" in self.config.model_type:
            if self.model is not None:
                self.model.requires_grad_(False)
        if self.config.controlnet and self.controlnet is not None:
            self.controlnet.train()

    def uses_shared_modules(self):
        # shared modules may be when ControlNet reuses base model layers, eg. HiDream.
        return False

    def get_trained_component(self, base_model: bool = False):
        return self.unwrap_model(model=self.model if base_model else None)

    def _load_pipeline(
        self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True
    ):
        """
        Loads the pipeline class for the model.
        """
        active_pipelines = getattr(self, "pipelines", {})
        if pipeline_type in active_pipelines:
            setattr(
                active_pipelines[pipeline_type],
                self.MODEL_TYPE.value,
                self.unwrap_model(model=self.model),
            )
            if self.config.controlnet:
                setattr(
                    active_pipelines[pipeline_type],
                    "controlnet",
                    self.unwrap_model(self.get_trained_component()),
                )
            return active_pipelines[pipeline_type]

        pipeline_kwargs = {
            "pretrained_model_name_or_path": self._model_config_path(),
        }
        if not hasattr(self, "PIPELINE_CLASSES"):
            raise NotImplementedError("Pipeline class not defined.")
        if pipeline_type not in self.PIPELINE_CLASSES:
            raise NotImplementedError(
                f"Pipeline type {pipeline_type} not defined in {self.__class__.__name__}."
            )
        pipeline_class = self.PIPELINE_CLASSES[pipeline_type]
        if not hasattr(pipeline_class, "from_pretrained"):
            raise NotImplementedError(
                f"Pipeline class {pipeline_class} does not have from_pretrained method."
            )
        signature = inspect.signature(pipeline_class.from_pretrained)
        if "watermarker" in signature.parameters:
            pipeline_kwargs["watermarker"] = None
        if "watermark" in signature.parameters:
            pipeline_kwargs["watermark"] = None
        if load_base_model:
            pipeline_kwargs[self.MODEL_TYPE.value] = self.unwrap_model(model=self.model)
        else:
            pipeline_kwargs[self.MODEL_TYPE.value] = None

        if getattr(self, "vae", None) is not None:
            pipeline_kwargs["vae"] = self.unwrap_model(self.vae)
        elif getattr(self, "AUTOENCODER_CLASS", None) is not None:
            pipeline_kwargs["vae"] = self.get_vae()

        text_encoder_idx = 0
        for (
            text_encoder_attr,
            text_encoder_config,
        ) in self.TEXT_ENCODER_CONFIGURATION.items():
            tokenizer_attr = text_encoder_attr.replace("text_encoder", "tokenizer")
            if (
                self.text_encoders is not None
                and len(self.text_encoders) >= text_encoder_idx
            ):
                pipeline_kwargs[text_encoder_attr] = self.unwrap_model(
                    self.text_encoders[text_encoder_idx]
                )
                logger.info(f"Adding {tokenizer_attr}")
                pipeline_kwargs[tokenizer_attr] = self.tokenizers[text_encoder_idx]
            else:
                pipeline_kwargs[text_encoder_attr] = None
                pipeline_kwargs[tokenizer_attr] = None

            text_encoder_idx += 1

        if self.config.controlnet and pipeline_type is PipelineTypes.CONTROLNET:
            pipeline_kwargs["controlnet"] = self.unwrap_model()

        logger.debug(
            f"Initialising {pipeline_class.__name__} with components: {pipeline_kwargs}"
        )
        self.pipelines[pipeline_type] = pipeline_class.from_pretrained(
            **pipeline_kwargs,
        )

        return self.pipelines[pipeline_type]

    def get_pipeline(
        self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True
    ) -> DiffusionPipeline:
        possibly_cached_pipeline = self._load_pipeline(pipeline_type, load_base_model)
        if (
            self.model is not None
            and getattr(possibly_cached_pipeline, self.MODEL_TYPE.value, None) is None
        ):
            # if the transformer or unet aren't in the cached pipeline, we'll add it.
            setattr(possibly_cached_pipeline, self.MODEL_TYPE.value, self.model)
        # attach the vae to the cached pipeline.
        setattr(possibly_cached_pipeline, "vae", self.get_vae())
        if self.text_encoders is not None:
            for (
                text_encoder_attr,
                text_encoder_config,
            ) in self.TEXT_ENCODER_CONFIGURATION.items():
                if getattr(possibly_cached_pipeline, text_encoder_attr, None) is None:
                    text_encoder_attr_number = 1
                    if "encoder_" in text_encoder_attr:
                        # support multi-encoder model pipelines
                        text_encoder_attr_number = text_encoder_attr.split("_")[-1]
                    setattr(
                        possibly_cached_pipeline,
                        text_encoder_attr,
                        self.text_encoders[int(text_encoder_attr_number) - 1],
                    )
        if self.config.controlnet:
            if getattr(possibly_cached_pipeline, "controlnet", None) is None:
                setattr(possibly_cached_pipeline, "controlnet", self.controlnet)

        return possibly_cached_pipeline

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        """
        When we're running the pipeline, we'll update the kwargs specifically for this model here.
        """

        return pipeline_kwargs

    def setup_training_noise_schedule(self):
        """
        Loads the noise schedule from the config.

        It's important to note, this is the *training* schedule, not inference.
        """
        flow_matching = False
        if self.PREDICTION_TYPE is PredictionTypes.FLOW_MATCHING:
            from diffusers import FlowMatchEulerDiscreteScheduler

            self.noise_schedule = FlowMatchEulerDiscreteScheduler.from_pretrained(
                get_model_config_path(
                    self.config.model_family, self.config.pretrained_model_name_or_path
                ),
                subfolder="scheduler",
                shift=self.config.flow_schedule_shift,
            )
            flow_matching = True
        elif self.PREDICTION_TYPE in [
            PredictionTypes.EPSILON,
            PredictionTypes.V_PREDICTION,
            PredictionTypes.SAMPLE,
        ]:
            from diffusers import DDPMScheduler

            self.noise_schedule = DDPMScheduler.from_pretrained(
                get_model_config_path(
                    self.config.model_family, self.config.pretrained_model_name_or_path
                ),
                subfolder="scheduler",
                rescale_betas_zero_snr=self.config.rescale_betas_zero_snr,
                timestep_spacing=self.config.training_scheduler_timestep_spacing,
            )
            if self.config.prediction_type is None:
                self.config.prediction_type = self.noise_schedule.config.prediction_type
        else:
            raise NotImplementedError(
                f"Unknown prediction type {self.PREDICTION_TYPE}."
            )

        return self.config, self.noise_schedule

    def get_prediction_target(self, prepared_batch: dict):
        """
        Returns the target used in the loss function.
        Depending on the noise schedule prediction type or flow-matching settings,
        the target is computed differently.
        """
        if prepared_batch.get("target") is not None:
            # Parent-student training
            target = prepared_batch["target"]
        elif self.PREDICTION_TYPE is PredictionTypes.FLOW_MATCHING:
            target = prepared_batch["noise"] - prepared_batch["latents"]
        elif self.PREDICTION_TYPE is PredictionTypes.EPSILON:
            target = prepared_batch["noise"]
        elif self.PREDICTION_TYPE is PredictionTypes.V_PREDICTION:
            target = self.noise_schedule.get_velocity(
                prepared_batch["latents"],
                prepared_batch["noise"],
                prepared_batch["timesteps"],
            )
        elif self.PREDICTION_TYPE is PredictionTypes.SAMPLE:
            target = prepared_batch["latents"]
        else:
            raise ValueError(f"Unknown prediction type {self.PREDICTION_TYPE}.")
        return target

    def prepare_batch_conditions(self, batch: dict, state: dict) -> dict:
        # it's a list, but most models will expect it to be a length-1 list containing a tensor, which is what they actually want
        if (
            isinstance(batch.get("conditioning_pixel_values"), list)
            and len(batch["conditioning_pixel_values"]) > 0
        ):
            batch["conditioning_pixel_values"] = batch["conditioning_pixel_values"][0]
        if (
            isinstance(batch.get("conditioning_latents"), list)
            and len(batch["conditioning_latents"]) > 0
        ):
            batch["conditioning_latents"] = batch["conditioning_latents"][0]
        return batch

    def prepare_batch(self, batch: dict, state: dict) -> dict:
        """
        Moves the batch to the proper device/dtype,
        samples noise, timesteps and, if applicable, flow-matching sigmas.
        This code is mostly common across models, but if you'd like to override certain pieces, use prepare_batch_conditions.

        Args:
            batch (dict): The batch to prepare.
            state (dict): The training state.
        Returns:
            dict: The prepared batch.
        """
        if not batch:
            return batch

        target_device_kwargs = {
            "device": self.accelerator.device,
            "dtype": self.config.weight_dtype,
        }

        logger.debug(f"Preparing batch: {batch.keys()}")
        # Ensure the encoder hidden states are on device
        if batch["prompt_embeds"] is not None and hasattr(batch["prompt_embeds"], "to"):
            batch["encoder_hidden_states"] = batch["prompt_embeds"].to(
                **target_device_kwargs
            )

        # Process additional conditioning if provided
        pooled_embeds = batch.get("add_text_embeds")
        time_ids = batch.get("batch_time_ids")
        batch["added_cond_kwargs"] = {}
        if pooled_embeds is not None and hasattr(pooled_embeds, "to"):
            batch["added_cond_kwargs"]["text_embeds"] = pooled_embeds.to(
                **target_device_kwargs
            )
        if time_ids is not None and hasattr(time_ids, "to"):
            batch["added_cond_kwargs"]["time_ids"] = time_ids.to(**target_device_kwargs)

        # Process latents (assumed to be in 'latent_batch')
        latents = batch.get("latent_batch")
        if not hasattr(latents, "to"):
            raise ValueError("Received invalid value for latents.")
        batch["latents"] = latents.to(**target_device_kwargs)

        encoder_attention_mask = batch.get("encoder_attention_mask")
        if encoder_attention_mask is not None and hasattr(encoder_attention_mask, "to"):
            batch["encoder_attention_mask"] = encoder_attention_mask.to(
                **target_device_kwargs
            )

        # Sample noise
        noise = torch.randn_like(batch["latents"])
        bsz = batch["latents"].shape[0]
        # If not flow matching, possibly apply an offset to noise
        if not self.config.flow_matching and self.config.offset_noise:
            if (
                self.config.noise_offset_probability == 1.0
                or random.random() < self.config.noise_offset_probability
            ):
                noise = noise + self.config.noise_offset * torch.randn(
                    latents.shape[0],
                    latents.shape[1],
                    1,
                    1,
                    device=latents.device,
                )
        batch["noise"] = noise

        # Possibly add input perturbation to input noise only
        if self.config.input_perturbation != 0 and (
            not getattr(self.config, "input_perturbation_steps", None)
            or state["global_step"] < self.config.input_perturbation_steps
        ):
            input_perturbation = self.config.input_perturbation
            if getattr(self.config, "input_perturbation_steps", None):
                input_perturbation *= 1.0 - (
                    state["global_step"] / self.config.input_perturbation_steps
                )
            batch["input_noise"] = noise + input_perturbation * torch.randn_like(
                batch["latents"]
            )
        else:
            batch["input_noise"] = noise

        # Flow matching branch: set sigmas and timesteps.
        if self.PREDICTION_TYPE is PredictionTypes.FLOW_MATCHING:
            if not self.config.flux_fast_schedule and not any(
                [
                    self.config.flow_use_beta_schedule,
                    self.config.flow_use_uniform_schedule,
                ]
            ):
                batch["sigmas"] = torch.sigmoid(
                    self.config.flow_sigmoid_scale
                    * torch.randn((bsz,), device=self.accelerator.device)
                )
                batch["sigmas"] = apply_flow_schedule_shift(
                    self.config, self.noise_schedule, batch["sigmas"], batch["noise"]
                )
            elif self.config.flow_use_uniform_schedule:
                batch["sigmas"] = torch.rand((bsz,), device=self.accelerator.device)
                batch["sigmas"] = apply_flow_schedule_shift(
                    self.config, self.noise_schedule, batch["sigmas"], batch["noise"]
                )
            elif self.config.flow_use_beta_schedule:
                alpha = self.config.flow_beta_schedule_alpha
                beta = self.config.flow_beta_schedule_beta
                beta_dist = Beta(alpha, beta)
                batch["sigmas"] = beta_dist.sample((bsz,)).to(
                    device=self.accelerator.device
                )
                batch["sigmas"] = apply_flow_schedule_shift(
                    self.config, self.noise_schedule, batch["sigmas"], batch["noise"]
                )
            else:
                available_sigmas = [1.0] * 7 + [0.75, 0.5, 0.25]
                batch["sigmas"] = torch.tensor(
                    random.choices(available_sigmas, k=bsz),
                    device=self.accelerator.device,
                )
            batch["timesteps"] = batch["sigmas"] * 1000.0
            # Ensure sigmas is reshaped appropriately (default is 4D, may be overriden in video subclass)
            self.expand_sigmas(batch)
            batch["noisy_latents"] = (1 - batch["sigmas"]) * batch["latents"] + batch[
                "sigmas"
            ] * batch["input_noise"]
        else:
            weights = generate_timestep_weights(
                self.config, self.noise_schedule.config.num_train_timesteps
            ).to(self.accelerator.device)
            if bsz > 1 and not self.config.disable_segmented_timestep_sampling:
                batch["timesteps"] = segmented_timestep_selection(
                    actual_num_timesteps=self.noise_schedule.config.num_train_timesteps,
                    bsz=bsz,
                    weights=weights,
                    config=self.config,
                    use_refiner_range=False,  # You can override in subclass if needed.
                ).to(self.accelerator.device)
            else:
                batch["timesteps"] = torch.multinomial(
                    weights, bsz, replacement=True
                ).long()
            batch["noisy_latents"] = self.noise_schedule.add_noise(
                batch["latents"].float(),
                batch["input_noise"].float(),
                batch["timesteps"],
            ).to(device=self.accelerator.device, dtype=self.config.weight_dtype)

        # any model-specific augmentation can occur inside prepare_batch_conditions.
        batch = self.prepare_batch_conditions(batch=batch, state=state)

        return batch

    def encode_text_batch(self, text_batch: list, is_negative_prompt: bool = False):
        """
        Encodes a batch of text using the text encoder.
        """
        if not self.TEXT_ENCODER_CONFIGURATION:
            raise ValueError("No text encoder configuration found.")
        encoded_text = self._encode_prompts(text_batch, is_negative_prompt)
        return self._format_text_embedding(encoded_text)

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        """
        Models can optionally format the stored text embedding, eg. in a dict, or
        filter certain outputs from appearing in the file cache.

        Args:
            text_embedding (torch.Tensor): The embed to adjust.

        Returns:
            torch.Tensor: The adjusted embed. By default, this method does nothing.
        """
        return text_embedding

    def conditional_loss(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
        loss_type: str = "l2",
        huber_c: float = 0.1,
    ):
        """
        Compute loss with support for L2, Huber, and Smooth L1.

        Args:
            model_pred: Model predictions
            target: Target values
            reduction: Reduction type ('mean' or 'sum')
            loss_type: Type of loss ('l2', 'huber', 'smooth_l1')
            huber_c: Huber loss parameter
        """
        if loss_type == "l2":
            loss = F.mse_loss(model_pred, target, reduction=reduction)
        elif loss_type == "huber":
            loss = (
                2
                * huber_c
                * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
            )
            if reduction == "mean":
                loss = torch.mean(loss)
            elif reduction == "sum":
                loss = torch.sum(loss)
        elif loss_type == "smooth_l1":
            loss = 2 * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
            if reduction == "mean":
                loss = torch.mean(loss)
            elif reduction == "sum":
                loss = torch.sum(loss)
        else:
            raise NotImplementedError(f"Unsupported Loss Type {loss_type}")
        return loss

    def compute_scheduled_huber_c(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Compute the scheduled huber_c parameter based on timesteps.

        Args:
            timesteps: Current timesteps in the diffusion process

        Returns:
            Scheduled huber_c values
        """
        if not hasattr(self.config, "loss_type"):
            return torch.tensor(0.1)  # Default value

        if self.config.loss_type not in ["huber", "smooth_l1"]:
            return torch.tensor(0.1)  # Not used for other loss types

        huber_schedule = getattr(self.config, "huber_schedule", "constant")
        base_huber_c = getattr(self.config, "huber_c", 0.1)

        if huber_schedule == "constant":
            return torch.tensor(base_huber_c)

        elif huber_schedule == "exponential":
            # Exponential decay based on timestep
            num_train_timesteps = self.noise_schedule.config.num_train_timesteps
            alpha = -math.log(base_huber_c) / num_train_timesteps

            # Handle batch of timesteps
            huber_c_values = []
            for t in timesteps:
                huber_c = math.exp(-alpha * t.item())
                huber_c_values.append(huber_c)

            return torch.tensor(huber_c_values, device=timesteps.device)

        elif huber_schedule == "snr":
            # SNR-based scheduling
            snr = compute_snr(timesteps, self.noise_schedule)
            sigmas = (
                (1.0 - self.noise_schedule.alphas_cumprod[timesteps])
                / self.noise_schedule.alphas_cumprod[timesteps]
            ) ** 0.5
            huber_c = (1 - base_huber_c) / (1 + sigmas) ** 2 + base_huber_c
            return huber_c

        else:
            raise NotImplementedError(f"Unknown Huber loss schedule {huber_schedule}")

    def loss(
        self, prepared_batch: dict, model_output, apply_conditioning_mask: bool = True
    ):
        """
        Computes the loss between the model prediction and the target.
        Optionally applies SNR weighting and a conditioning mask.
        """
        target = self.get_prediction_target(prepared_batch)
        model_pred = model_output["model_prediction"]
        if target is None:
            raise ValueError("Target is None. Cannot compute loss.")

        # Get loss type from config (default to l2 for backward compatibility)
        loss_type = getattr(self.config, "loss_type", "l2")

        if self.PREDICTION_TYPE == PredictionTypes.FLOW_MATCHING:
            # Flow matching always uses L2 loss
            loss = (model_pred.float() - target.float()) ** 2
        elif self.PREDICTION_TYPE in [
            PredictionTypes.EPSILON,
            PredictionTypes.V_PREDICTION,
        ]:
            # Check if we're using Huber or smooth L1 loss
            if loss_type in ["huber", "smooth_l1"]:
                # Get timesteps for the batch
                timesteps = prepared_batch["timesteps"]

                # For scheduled huber, we compute per-sample then average
                if getattr(self.config, "huber_schedule", "constant") != "constant":
                    batch_size = model_pred.shape[0]
                    losses = []

                    for i in range(batch_size):
                        # Get scheduled huber_c for this timestep
                        huber_c = self.compute_scheduled_huber_c(
                            timesteps[i : i + 1]
                        ).item()

                        # Compute loss for this sample
                        sample_loss = self.conditional_loss(
                            model_pred[i : i + 1].float(),
                            target[i : i + 1].float(),
                            reduction="none",
                            loss_type=loss_type,
                            huber_c=huber_c,
                        )
                        losses.append(sample_loss)

                    loss = torch.cat(losses, dim=0)
                else:
                    # Constant huber_c - can be computed all at once
                    huber_c = getattr(self.config, "huber_c", 0.1)
                    loss = self.conditional_loss(
                        model_pred.float(),
                        target.float(),
                        reduction="none",
                        loss_type=loss_type,
                        huber_c=huber_c,
                    )

                # Apply SNR weighting if configured (for Huber/smooth L1)
                if self.config.snr_gamma is not None and self.config.snr_gamma > 0:
                    snr = compute_snr(prepared_batch["timesteps"], self.noise_schedule)
                    snr_divisor = snr
                    if (
                        self.noise_schedule.config.prediction_type
                        == PredictionTypes.V_PREDICTION.value
                    ):
                        snr_divisor = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [
                                snr,
                                self.config.snr_gamma
                                * torch.ones_like(prepared_batch["timesteps"]),
                            ],
                            dim=1,
                        ).min(dim=1)[0]
                        / snr_divisor
                    )
                    mse_loss_weights = mse_loss_weights.view(-1, 1, 1, 1)
                    loss = loss * mse_loss_weights

            else:
                if self.config.snr_gamma is None or self.config.snr_gamma == 0:
                    loss = self.config.snr_weight * F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                else:
                    snr = compute_snr(prepared_batch["timesteps"], self.noise_schedule)
                    snr_divisor = snr
                    if (
                        self.noise_schedule.config.prediction_type
                        == PredictionTypes.V_PREDICTION.value
                    ):
                        snr_divisor = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [
                                snr,
                                self.config.snr_gamma
                                * torch.ones_like(prepared_batch["timesteps"]),
                            ],
                            dim=1,
                        ).min(dim=1)[0]
                        / snr_divisor
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    mse_loss_weights = mse_loss_weights.view(-1, 1, 1, 1)
                    loss = loss * mse_loss_weights
        else:
            raise NotImplementedError(
                f"Loss calculation not implemented for prediction type {self.PREDICTION_TYPE}."
            )

        # Apply conditioning mask if needed
        conditioning_type = prepared_batch.get("conditioning_type")
        if conditioning_type == "mask" and apply_conditioning_mask:
            logger.debug("Applying conditioning mask to loss.")
            mask_image = (
                prepared_batch["conditioning_pixel_values"]
                .to(dtype=loss.dtype, device=loss.device)[:, 0]
                .unsqueeze(1)
            )
            mask_image = torch.nn.functional.interpolate(
                mask_image, size=loss.shape[2:], mode="area"
            )
            mask_image = mask_image / 2 + 0.5
            loss = loss * mask_image
        elif conditioning_type == "segmentation" and apply_conditioning_mask:
            if random.random() < self.config.masked_loss_probability:
                mask_image = prepared_batch["conditioning_pixel_values"].to(
                    dtype=loss.dtype, device=loss.device
                )
                mask_image = torch.sum(mask_image, dim=1, keepdim=True) / 3
                mask_image = torch.nn.functional.interpolate(
                    mask_image, size=loss.shape[2:], mode="area"
                )
                mask_image = mask_image / 2 + 0.5
                mask_image = (mask_image > 0).to(dtype=loss.dtype, device=loss.device)
                loss = loss * mask_image

        # Average over channels and spatial dims, then over batch.
        loss = loss.mean(dim=list(range(1, len(loss.shape)))).mean()
        return loss

    def auxiliary_loss(self, model_output, prepared_batch: dict, loss: torch.Tensor):
        """
        Computes an auxiliary loss if needed.
        This is a stub and can be optionally implemented in subclasses.
        """
        return loss, None


class ImageModelFoundation(ModelFoundation):
    """
    Implements logic common to image-based diffusion models.
    Handles typical VAE, text encoder loading and a UNet forward pass.
    """

    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    DEFAULT_CONTROLNET_LORA_TARGET = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
        "proj_in",
        "proj_out",
        "conv",
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "linear_1",
        "linear_2",
        "time_emb_proj",
        "controlnet_cond_embedding.conv_in",
        "controlnet_cond_embedding.blocks.0",
        "controlnet_cond_embedding.blocks.1",
        "controlnet_cond_embedding.blocks.2",
        "controlnet_cond_embedding.blocks.3",
        "controlnet_cond_embedding.blocks.4",
        "controlnet_cond_embedding.blocks.5",
        "controlnet_cond_embedding.conv_out",
        "controlnet_down_blocks.0",
        "controlnet_down_blocks.1",
        "controlnet_down_blocks.2",
        "controlnet_down_blocks.3",
        "controlnet_down_blocks.4",
        "controlnet_down_blocks.5",
        "controlnet_down_blocks.6",
        "controlnet_down_blocks.7",
        "controlnet_down_blocks.8",
        "controlnet_mid_block",
    ]
    SHARED_MODULE_PREFIXES = (
        None  # No shared modules by default, but can be overridden in subclasses.
    )
    # A bit more than the default, but some will need to override this to just Attention layers, like SD3.
    DEFAULT_LYCORIS_TARGET = ["Attention", "FeedForward"]
    DEFAULT_PIPELINE_TYPE = PipelineTypes.TEXT2IMG
    VALIDATION_USES_NEGATIVE_PROMPT = True

    def __init__(self, config: dict, accelerator):
        super().__init__(config, accelerator)
        self.has_vae = True
        self.has_text_encoder = True
        # These will be set by your loading methods
        self.vae = None
        self.model = None
        self.controlnet = None
        self.text_encoders = None
        self.tokenizers = None

    def expand_sigmas(self, batch: dict) -> dict:
        batch["sigmas"] = batch["sigmas"].view(-1, 1, 1, 1)

        return batch

    def get_lora_save_layers(self):
        return None

    def get_lora_target_layers(self):
        # Some models, eg. Flux should override this with more complex config-driven logic.
        if self.config.lora_type.lower() == "standard":
            if self.config.controlnet:
                return self.DEFAULT_CONTROLNET_LORA_TARGET
            return self.DEFAULT_LORA_TARGET
        elif self.config.lora_type.lower() == "lycoris":
            return self.DEFAULT_LYCORIS_TARGET
        else:
            raise NotImplementedError(
                f"Unknown LoRA target type {self.config.lora_type}."
            )

    def add_lora_adapter(self):
        target_modules = self.get_lora_target_layers()
        save_modules = self.get_lora_save_layers()
        addkeys, misskeys = [], []

        # Force LyCORIS for ControlNet on UNet models to support Conv2d
        if self.config.controlnet and self.MODEL_TYPE.value == "unet":
            logger.warning(
                "ControlNet with UNet requires Conv2d layer support. "
                "Using LyCORIS (LoHa) adapter instead of standard LoRA."
            )
            from peft import LoHaConfig

            self.lora_config = LoHaConfig(
                r=self.config.lora_rank,
                alpha=(
                    self.config.lora_alpha
                    if self.config.lora_alpha is not None
                    else self.config.lora_rank
                ),
                rank_dropout=self.config.lora_dropout,
                module_dropout=0.0,
                use_effective_conv2d=True,  # Critical for Conv2d support
                target_modules=target_modules,
                modules_to_save=save_modules,
                # init_weights defaults to True, which is what we want
            )
        else:
            # Standard LoRA for everything else
            self.lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=(
                    self.config.lora_alpha
                    if self.config.lora_alpha is not None
                    else self.config.lora_rank
                ),
                lora_dropout=self.config.lora_dropout,
                init_lora_weights=self.config.lora_initialisation_style,
                target_modules=target_modules,
                modules_to_save=save_modules,
                use_dora=self.config.use_dora,
            )

        # Apply adapter
        if self.config.controlnet:
            self.controlnet.add_adapter(self.lora_config)
        else:
            self.model.add_adapter(self.lora_config)

        if self.config.init_lora:
            # Note: use_dora only applies to LoraConfig, not LoHaConfig
            use_dora = (
                self.config.use_dora
                if isinstance(self.lora_config, LoraConfig)
                else False
            )
            addkeys, misskeys = load_lora_weights(
                {
                    self.MODEL_TYPE: (
                        self.controlnet if self.config.controlnet else self.model
                    )
                },
                self.config.init_lora,
                use_dora=use_dora,
            )

        return addkeys, misskeys

    def custom_model_card_schedule_info(self):
        """
        Override this in your subclass to add model-specific info.

        See SD3 or Flux classes for an example.
        """
        return []

    def custom_model_card_code_example(self, repo_id: str = None) -> str:
        """
        Override this to provide custom code examples for model cards.
        Returns None by default to use the standard template.
        """
        return None


class VideoToTensor:
    def __call__(self, video):
        """
        Converts a video (numpy array of shape (num_frames, height, width, channels))
        to a tensor of shape (num_frames, channels, height, width) by applying the
        standard ToTensor conversion to each frame.
        """
        if isinstance(video, np.ndarray):
            frames = []
            for frame in video:
                # Convert frame to PIL Image if not already.
                if not isinstance(frame, Image.Image):
                    frame = Image.fromarray(frame)
                # Apply the standard ToTensor transform.
                frame_tensor = transforms.functional.to_tensor(frame)
                frames.append(frame_tensor)
            return torch.stack(frames)
        elif isinstance(video, list):
            # If video is a list of frames, process similarly.
            frames = []
            for frame in video:
                if not isinstance(frame, Image.Image):
                    frame = Image.fromarray(frame)
                frames.append(transforms.functional.to_tensor(frame))
            return torch.stack(frames)
        else:
            raise TypeError("Input video must be a numpy array or a list of frames.")

    def __repr__(self):
        return self.__class__.__name__ + "()"


class VideoModelFoundation(ImageModelFoundation):
    """
    Base class for video models. Provides default 5D handling and optional
    text encoder instantiation. The actual text encoder classes and their
    attributes can be stored in a hardcoded dict if needed. This base class
    does not do it by default.
    """

    def __init__(self, config, accelerator):
        """
        :param config: The training configuration object/dict.
        """
        super().__init__(config, accelerator)
        self.config = config
        # Optionally, store or initialize text encoders here.
        # But do NOT automatically do it unless your code requires it.

        # For example, if you have a dictionary of text-encoder descriptors:
        # self.text_encoders = {
        #     "encoder1": {"class": MyTextEncoderClass, "attr_name": "text_encoder_1"},
        #     "encoder2": {"class": AnotherTextEncoder, "attr_name": "text_encoder_2"},
        # }
        # The trainer or child class might call self._init_text_encoders() at the right time.

    def get_transforms(self, dataset_type: str = "image"):
        return transforms.Compose(
            [
                VideoToTensor() if dataset_type == "video" else transforms.ToTensor(),
            ]
        )

    def expand_sigmas(self, batch):
        if len(batch["latents"].shape) == 5:
            # ltxvideo and others with 5D tensors need expansion to match dims here i think
            logger.debug(
                f"Latents shape vs sigmas, timesteps: {batch['latents'].shape}, {batch['sigmas'].shape}, {batch['timesteps'].shape}"
            )
            batch["sigmas"] = batch["sigmas"].reshape(
                batch["latents"].shape[0], 1, 1, 1, 1
            )

    def apply_i2v_augmentation(self, batch):
        pass

    def prepare_5d_inputs(self, tensor):
        """
        Example method to handle default 5D shape. The typical shape might be:
        (batch_size, frames, channels, height, width).

        You can reshape or permute as needed for the underlying model.
        """
        # Pseudocode for typical flattening:
        # B, F, C, H, W = tensor.shape
        # return tensor.view(B * F, C, H, W)
        return tensor
