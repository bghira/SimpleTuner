import logging, os
import huggingface_hub
from helpers.training.default_settings.safety_check import safety_check
from helpers.publishing.huggingface import HubManager
from configure import model_labels
from typing import Optional
import shutil
import hashlib
import json
import copy
import random
import math
import sys
import glob
import wandb


from helpers import log_format  # noqa
from helpers.configuration.loader import load_config
from helpers.caching.memory import reclaim_memory
from helpers.training.multi_process import _get_rank as get_rank
from helpers.training.validation import Validation, prepare_validation_prompt_list
from helpers.training.evaluation import ModelEvaluator
from helpers.training.state_tracker import StateTracker
from helpers.training.custom_schedule import get_lr_scheduler
from helpers.training.optimizer_param import (
    determine_optimizer_class_with_config,
    determine_params_to_optimize,
    is_lr_scheduler_disabled,
    is_lr_schedulefree,
    cpu_offload_optimizer,
)
from helpers.data_backend.factory import BatchFetcher
from helpers.training.deepspeed import (
    prepare_model_for_deepspeed,
)
from helpers.training.wrappers import unwrap_model
from helpers.data_backend.factory import configure_multi_databackend
from helpers.data_backend.factory import random_dataloader_iterator
from helpers.training.min_snr_gamma import compute_snr
from helpers.training.peft_init import init_lokr_network_with_perturbed_normal
from accelerate.logging import get_logger
from helpers.models.all import model_families

logger = get_logger(
    "SimpleTuner", log_level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
)

filelock_logger = get_logger("filelock")
connection_logger = get_logger("urllib3.connectionpool")
training_logger = get_logger("training-loop")

# More important logs.
target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
logger.setLevel(target_level)
training_logger_level = os.environ.get("SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL", "INFO")
training_logger.setLevel(training_logger_level)

# Less important logs.
filelock_logger.setLevel("WARNING")
connection_logger.setLevel("WARNING")
import torch
import diffusers
import accelerate
import transformers
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from configure import model_classes
from torch.distributions import Beta

try:
    from lycoris import LycorisNetwork
except:
    print("[ERROR] Lycoris not available. Please install ")
from tqdm.auto import tqdm

from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    DDPMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    UniPCMultistepScheduler,
)

from peft.utils import get_peft_model_state_dict
from helpers.training.ema import EMAModel
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
)
from diffusers.utils.import_utils import is_xformers_available
from helpers.models.common import VideoModelFoundation, ImageModelFoundation

is_optimi_available = False
try:
    from optimi import prepare_for_gradient_release

    is_optimi_available = True
except:
    pass

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

SCHEDULER_NAME_MAP = {
    "euler": EulerDiscreteScheduler,
    "euler-a": EulerAncestralDiscreteScheduler,
    "unipc": UniPCMultistepScheduler,
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
}
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

transformers.utils.logging.set_verbosity_warning()
diffusers.utils.logging.set_verbosity_warning()


class Trainer:
    def __init__(
        self,
        config: dict = None,
        disable_accelerator: bool = False,
        job_id: str = None,
        exit_on_error: bool = False,
    ):
        self.accelerator = None
        self.model = None
        self.job_id = job_id
        StateTracker.set_job_id(job_id)
        self.parse_arguments(
            args=config,
            disable_accelerator=disable_accelerator,
            exit_on_error=exit_on_error,
        )
        if (
            getattr(self, "config", None) is not None
            and self.config.model_family in model_families
        ):
            self.model = model_families[self.config.model_family](
                self.config, self.accelerator
            )
            self.model.check_user_config()
            StateTracker.set_model(self.model)
        self._misc_init()
        self.lycoris_wrapped_network = None
        self.lycoris_config = None
        self.lr_scheduler = None
        self.webhook_handler = None
        self.should_abort = False
        self.ema_model = None
        self.validation = None
        # this updates self.config further, so we will run it here.
        self.init_noise_schedule()

    def _config_to_obj(self, config):
        if not config:
            return None
        return type("Config", (object,), config)

    def parse_arguments(
        self, args=None, disable_accelerator: bool = False, exit_on_error: bool = False
    ):
        self.config = load_config(args, exit_on_error=exit_on_error)
        report_to = (
            None if self.config.report_to.lower() == "none" else self.config.report_to
        )
        if not disable_accelerator:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                mixed_precision=(
                    self.config.mixed_precision
                    if not torch.backends.mps.is_available()
                    else None
                ),
                log_with=report_to,
                project_config=self.config.accelerator_project_config,
                kwargs_handlers=[self.config.process_group_kwargs],
            )
        safety_check(args=self.config, accelerator=self.accelerator)

        if self.config.lr_scale:
            lr_cur = self.config.learning_rate
            lr_scale_bsz = self.config.train_batch_size
            lr_scale_ga = self.config.gradient_accumulation_steps
            lr_scale_np = getattr(self.accelerator, "num_processes", 1)
            lr_scale_mul = lr_scale_ga * lr_scale_bsz * lr_scale_np
            lr_new = lr_cur * (
                math.sqrt(lr_scale_mul) if self.config.lr_scale_sqrt else lr_scale_mul
            )
            logger.info(
                f"Scaling learning rate from {lr_cur:.1e} to {lr_new:.1e}"
                f" due to {'--lr-scale and --lr-scale-sqrt' if self.config.lr_scale_sqrt else '--lr-scale'}"
                f" (bsz: {lr_scale_bsz}, ga: {lr_scale_ga}, nprocs: {lr_scale_np})"
            )
            self.config.learning_rate = lr_new

        StateTracker.set_accelerator(self.accelerator)
        StateTracker.set_args(self.config)
        StateTracker.set_weight_dtype(self.config.weight_dtype)
        self.set_model_family()

    def run(self):
        try:
            # Initialize essential configurations and schedules
            self.configure_webhook()
            self.init_noise_schedule()
            self.init_seed()
            self.init_huggingface_hub()

            # Core initialization steps with signal checks after each step
            self._initialize_components_with_signal_check(
                [
                    self.init_preprocessing_models,
                    self.init_data_backend,
                    self.init_validation_prompts,
                    self.init_unload_text_encoder,
                    self.init_unload_vae,
                    self.init_load_base_model,
                    self.init_precision,
                    self.init_controlnet_model,
                    self.init_freeze_models,
                    self.init_trainable_peft_adapter,
                    self.init_ema_model,
                ]
            )

            # Model movement and validation setup
            self.move_models(destination="accelerator")
            self._exit_on_signal()
            self.init_validations()
            self._exit_on_signal()
            self.enable_sageattention_inference()
            self.init_benchmark_base_model()
            self.disable_sageattention_inference()
            self._exit_on_signal()
            self.resume_and_prepare()
            self._exit_on_signal()
            self.init_trackers()

            # Start the training process
            self.train()

        except Exception as e:
            import traceback

            logger.error(
                f"Failed to run training: {e}, traceback: {traceback.format_exc()}"
            )
            self._send_webhook_msg(
                message=f"Failed to run training: {e}",
            )
            self._send_webhook_raw(
                structured_data={
                    "message": f"Failed to run training: {e}",
                    "status": "error",
                },
                message_type="fatal_error",
            )

            raise e

    def _initialize_components_with_signal_check(self, initializers):
        """
        Runs a list of initializer functions with signal checks after each.

        Args:
            initializers (list): A list of initializer functions to run sequentially.
        """
        for initializer in initializers:
            initializer()
            self._exit_on_signal()

    def _get_noise_schedule(self):
        self.config, scheduler = self.model.setup_training_noise_schedule()

        return scheduler

    def init_noise_schedule(self):
        if self.model is None:
            return
        from helpers.models.common import PredictionTypes

        self.config.flow_matching = (
            True
            if self.model.PREDICTION_TYPE is PredictionTypes.FLOW_MATCHING
            else False
        )
        self.noise_scheduler = self._get_noise_schedule()
        self.lr = 0.0

    def configure_webhook(self, send_startup_message: bool = True):
        self.webhook_handler = None
        if self.config.webhook_config is None:
            return
        from helpers.webhooks.handler import WebhookHandler

        self.webhook_handler = WebhookHandler(
            self.config.webhook_config,
            self.accelerator,
            f"{self.config.tracker_project_name} {self.config.tracker_run_name}",
            send_video=(
                True if isinstance(self.model, VideoModelFoundation) else False
            ),
            args=self.config,
        )
        StateTracker.set_webhook_handler(self.webhook_handler)
        if send_startup_message:
            self._send_webhook_msg(
                message="SimpleTuner has launched. Hold onto your butts!",
                store_response=True,
            )
        self._send_webhook_raw(
            structured_data={
                "message": "Training job has started, configuration has begun."
            },
            message_type="configure_webhook",
        )

    def _misc_init(self):
        """things that do not really need an order."""
        torch.set_num_threads(self.config.torch_num_threads)
        self.state = {}
        self.state["lr"] = 0.0
        # Global step represents the most recently *completed* optimization step, which means it
        #  takes into account the number of gradient_accumulation_steps. If we use 1 gradient_accumulation_step,
        #  then global_step and step will be the same throughout training. However, if we use
        #  2 gradient_accumulation_steps, then global_step will be twice as large as step, and so on.
        self.state["global_step"] = 0
        self.state["global_resume_step"] = 0
        self.state["first_epoch"] = 1
        self.timesteps_buffer = []
        self.guidance_values_list = []
        self.train_loss = 0.0
        self.bf = None
        self.grad_norm = None
        self.extra_lr_scheduler_kwargs = {}
        StateTracker.set_global_step(self.state["global_step"])
        self.config.use_deepspeed_optimizer, self.config.use_deepspeed_scheduler = (
            prepare_model_for_deepspeed(self.accelerator, self.config)
        )
        self.config.base_weight_dtype = self.config.weight_dtype
        self.config.is_quanto = False
        self.config.is_torchao = False
        self.config.is_bnb = False
        if "quanto" in self.config.base_model_precision:
            self.config.is_quanto = True
        elif "torchao" in self.config.base_model_precision:
            self.config.is_torchao = True
        elif "bnb" in self.config.base_model_precision:
            self.config.is_bnb = True
        # if text_encoder_1_precision -> text_encoder_4_precision has quanto we'll mark that as well
        for i in range(1, 5):
            if isinstance(
                getattr(self.config, f"text_encoder_{i}_precision", None), str
            ) and getattr(self.config, f"text_encoder_{i}_precision", None):
                if "quanto" in getattr(self.config, f"text_encoder_{i}_precision"):
                    if self.config.is_torchao:
                        raise ValueError(
                            "Cannot enable Quanto and TorchAO together. One quant engine must be used for all precision levels."
                        )
                    self.config.is_quanto = True
                elif "torchao" in getattr(self.config, f"text_encoder_{i}_precision"):
                    if self.config.is_quanto:
                        raise ValueError(
                            "Cannot enable Quanto and TorchAO together. One quant engine must be used for all precision levels."
                        )
                    self.config.is_torchao = True
                elif "bnb" in getattr(self.config, f"text_encoder_{i}_precision"):
                    self.config.is_bnb = True
        if self.config.is_quanto or self.config.is_torchao:
            from helpers.training.quantisation import quantise_model

            self.quantise_model = quantise_model

    def set_model_family(self, model_family: str = None):
        model_family = getattr(self.config, "model_family", model_family)
        if model_family not in model_classes["full"]:
            raise ValueError(f"Invalid model family specified: {model_family}")

        from helpers.models.all import model_families

        model_implementation = model_families.get(model_family)
        StateTracker.set_model_family(model_family)
        self.config.model_type_label = (
            getattr(model_implementation, "NAME", None)
            or model_labels[model_family.lower()]
        )
        if StateTracker.is_sdxl_refiner():
            self.config.model_type_label = "SDXL Refiner"

    def init_clear_backend_cache(self):
        if self.config.output_dir is not None:
            os.makedirs(self.config.output_dir, exist_ok=True)
        if self.config.preserve_data_backend_cache:
            return
        if not self.accelerator.is_local_main_process:
            return
        StateTracker.delete_cache_files(
            preserve_data_backend_cache=self.config.preserve_data_backend_cache
        )

    def init_seed(self):
        if self.config.seed is not None and self.config.seed != 0:
            set_seed(self.config.seed, self.config.seed_for_each_device)

    def init_huggingface_hub(self, access_token: str = None):
        # Handle the repository creation
        self.hub_manager = None
        if not self.accelerator.is_main_process or not self.config.push_to_hub:
            return
        if access_token:
            huggingface_hub.login(token=access_token)
        self.hub_manager = HubManager(config=self.config, model=self.model)
        try:
            StateTracker.set_hf_user(huggingface_hub.whoami())
            logger.info(
                f"Logged into Hugging Face Hub as '{StateTracker.get_hf_username()}'"
            )
        except Exception as e:
            logger.error(f"Failed to log into Hugging Face Hub: {e}")
            raise e

    def init_preprocessing_models(self, move_to_accelerator: bool = True):
        # image embeddings
        self.init_vae(move_to_accelerator=move_to_accelerator)
        # text embeds
        self.init_text_encoder(move_to_accelerator=move_to_accelerator)

    def init_vae(self, move_to_accelerator: bool = True):
        if getattr(self.model, "AUTOENCODER_CLASS", None) is None:
            logger.debug(f"Model {self.model.NAME} does not have a VAE.")
            return
        logger.info(f"Load VAE: {self.config.pretrained_vae_model_name_or_path}")
        self.model.load_vae(move_to_device=move_to_accelerator)
        StateTracker.set_vae_dtype(self.model.vae.dtype)
        StateTracker.set_vae(self.model.vae)

    def init_text_encoder(self, move_to_accelerator: bool = True):
        self.model.load_text_encoder(move_to_device=move_to_accelerator)

    def init_freeze_models(self):
        self.model.freeze_components()
        self.accelerator.wait_for_everyone()

    def init_load_base_model(self):
        webhook_msg = f"Loading model: `{self.config.pretrained_model_name_or_path}`..."
        self._send_webhook_msg(message=webhook_msg)
        self._send_webhook_raw(
            structured_data={"message": webhook_msg},
            message_type="init_load_base_model_begin",
        )
        self.model.load_model(move_to_device=False)
        self.accelerator.wait_for_everyone()
        self._send_webhook_raw(
            structured_data={"message": "Base model has loaded."},
            message_type="init_load_base_model_completed",
        )

    def init_data_backend(self):
        try:
            self.init_clear_backend_cache()
            self._send_webhook_msg(
                message="Configuring data backends... (this may take a while!)"
            )
            self._send_webhook_raw(
                structured_data={"message": "Configuring data backends."},
                message_type="init_data_backend_begin",
            )
            configure_multi_databackend(
                self.config,
                accelerator=self.accelerator,
                text_encoders=self.model.text_encoders,
                tokenizers=self.model.tokenizers,
                model=self.model,
            )
            self._send_webhook_raw(
                structured_data={"message": "Completed configuring data backends."},
                message_type="init_data_backend_completed",
            )
        except Exception as e:
            import traceback

            logger.error(f"{e}, traceback: {traceback.format_exc()}")
            self._send_webhook_msg(
                message=f"Failed to load data backends: {e}",
                message_level="critical",
            )
            self._send_webhook_raw(
                structured_data={
                    "message": f"Failed to load data backends: {e}",
                    "status": "error",
                },
                message_type="fatal_error",
            )

            raise e

        try:
            self.init_validation_prompts()
        except Exception as e:
            logger.error("Could not generate validation prompts.")
            logger.error(e)
            raise e

        # We calculate the number of steps per epoch by dividing the number of images by the effective batch divisor.
        # Gradient accumulation steps mean that we only update the model weights every /n/ steps.
        collected_data_backend_str = list(StateTracker.get_data_backends().keys())
        if self.config.push_to_hub and self.accelerator.is_main_process:
            self.hub_manager.collected_data_backend_str = collected_data_backend_str
            self.hub_manager.set_validation_prompts(self.validation_prompt_metadata)
            logger.debug(
                f"Collected validation prompts: {self.validation_prompt_metadata}"
            )
        self._recalculate_training_steps()
        logger.info(
            f"Collected the following data backends: {collected_data_backend_str}"
        )
        self._send_webhook_msg(
            message=f"Collected the following data backends: {collected_data_backend_str}"
        )
        self._send_webhook_raw(
            structured_data={
                "message": f"Collected the following data backends: {collected_data_backend_str}"
            },
            message_type="init_data_backend",
        )
        self.accelerator.wait_for_everyone()

    def init_validation_prompts(self):
        if (
            hasattr(self.accelerator, "state")
            and hasattr(self.accelerator.state, "deepspeed_plugin")
            and getattr(self.accelerator.state.deepspeed_plugin, "deepspeed_config", {})
            .get("zero_optimization", {})
            .get("stage")
            == 3
        ):
            logger.error("Cannot run validations with DeepSpeed ZeRO stage 3.")
            return
        if self.accelerator.is_main_process and not self.config.validation_disable:
            self.validation_prompt_metadata = prepare_validation_prompt_list(
                args=self.config,
                embed_cache=StateTracker.get_default_text_embed_cache(),
                model=self.model,
            )
        else:
            self.validation_prompt_metadata = None
            self.validation_shortnames = None
            self.validation_negative_prompt_embeds = None
            self.validation_negative_pooled_embeds = None
        self.accelerator.wait_for_everyone()

    def stats_memory_used(self):
        # Grab GPU memory used:
        if torch.cuda.is_available():
            curent_memory_allocated = torch.cuda.memory_allocated() / 1024**3
        elif torch.backends.mps.is_available():
            curent_memory_allocated = torch.mps.current_allocated_memory() / 1024**3
        else:
            logger.warning(
                "CUDA, ROCm, or Apple MPS not detected here. We cannot report VRAM reductions."
            )
            curent_memory_allocated = 0

        return curent_memory_allocated

    def init_unload_text_encoder(self):
        if self.config.model_type != "full" and self.config.train_text_encoder:
            return
        memory_before_unload = self.stats_memory_used()
        if self.accelerator.is_main_process:
            logger.info("Unloading text encoders, as they are not being trained.")
        self.model.unload_text_encoder()
        for backend_id, backend in StateTracker.get_data_backends().items():
            if "text_embed_cache" in backend:
                backend["text_embed_cache"].text_encoders = None
                backend["text_embed_cache"].pipeline = None
        reclaim_memory()
        memory_after_unload = self.stats_memory_used()
        memory_saved = memory_after_unload - memory_before_unload
        logger.info(
            f"After nuking text encoders from orbit, we freed {abs(round(memory_saved, 2))} GB of VRAM."
        )

    def init_precision(
        self, preprocessing_models_only: bool = False, ema_only: bool = False
    ):
        self.config.enable_adamw_bf16 = (
            True if self.config.weight_dtype == torch.bfloat16 else False
        )
        quantization_device = (
            "cpu" if self.config.quantize_via == "cpu" else self.accelerator.device
        )

        if "bnb" in self.config.base_model_precision:
            # can't cast or move bitsandbytes models
            return

        if not self.config.disable_accelerator and self.config.is_quantized:
            if self.config.base_model_default_dtype == "fp32":
                self.config.base_weight_dtype = torch.float32
                self.config.enable_adamw_bf16 = False
            elif self.config.base_model_default_dtype == "bf16":
                self.config.base_weight_dtype = torch.bfloat16
                self.config.enable_adamw_bf16 = True
            elif self.config.base_model_default_dtype == "fp16":
                raise ValueError("fp16 mixed precision training is not supported.")
            if not preprocessing_models_only:
                logger.info(
                    f"Moving {self.model.MODEL_TYPE.value} to dtype={self.config.base_weight_dtype}, device={quantization_device}"
                )
                self.model.model.to(
                    quantization_device, dtype=self.config.base_weight_dtype
                )
        if self.config.is_quanto:
            with self.accelerator.local_main_process_first():
                if ema_only:
                    self.quantise_model(ema=self.ema_model, args=self.config)

                    return
                self.quantise_model(
                    model=(
                        self.model.get_trained_component()
                        if not preprocessing_models_only
                        else None
                    ),
                    text_encoders=self.model.text_encoders,
                    controlnet=None,
                    ema=self.ema_model,
                    args=self.config,
                )
        elif self.config.is_torchao:
            with self.accelerator.local_main_process_first():
                if ema_only:
                    self.ema_model = self.quantise_model(
                        ema=self.ema_model, args=self.config, return_dict=True
                    )["ema"]

                    return
                (
                    self.model.model,
                    self.model.text_encoders,
                    self.controlnet,
                    self.ema_model,
                ) = self.quantise_model(
                    model=(
                        self.model.get_trained_component()
                        if not preprocessing_models_only
                        else None
                    ),
                    text_encoders=self.model.text_encoders,
                    controlnet=None,
                    ema=self.ema_model,
                    args=self.config,
                )

    def init_controlnet_model(self):
        if not self.config.controlnet:
            return
        self.model.controlnet_init()
        self.accelerator.wait_for_everyone()

    def init_trainable_peft_adapter(self):
        if "lora" not in self.config.model_type:
            return
        if self.config.controlnet:
            raise ValueError("Cannot train LoRA with ControlNet.")
        if "standard" == self.config.lora_type.lower():
            lora_info_msg = f"Using LoRA training mode (rank={self.config.lora_rank})"
            logger.info(lora_info_msg)
            self._send_webhook_msg(message=lora_info_msg)
            addkeys, misskeys = self.model.add_lora_adapter()
            if addkeys:
                logger.warning(
                    "The following keys were found in %s, but are not part of the model and are ignored:\n %s.\nThis is most likely an error"
                    % (self.config.init_lora, str(addkeys))
                )
            if misskeys:
                logger.warning(
                    "The following keys were part of the model but not found in %s:\n %s.\nThese keys will be initialized according to the lora weight initialisation. This could be an error, or intended behaviour in case a lora is finetuned with additional keys."
                    % (self.config.init_lora, str(misskeys))
                )

        elif "lycoris" == self.config.lora_type.lower():
            from lycoris import create_lycoris

            with open(self.config.lycoris_config, "r") as f:
                self.lycoris_config = json.load(f)
            multiplier = int(self.lycoris_config.get("multiplier", 1))
            linear_dim = int(self.lycoris_config.get("linear_dim", 4))
            linear_alpha = int(self.lycoris_config.get("linear_alpha", 1))
            apply_preset = self.lycoris_config.get("apply_preset", None)
            if apply_preset is not None and apply_preset != {}:
                LycorisNetwork.apply_preset(apply_preset)

            # Remove the positional arguments we extracted.
            keys_to_remove = ["multiplier", "linear_dim", "linear_alpha"]
            for key in keys_to_remove:
                if key in self.lycoris_config:
                    del self.lycoris_config[key]

            logger.info("Using lycoris training mode")
            self._send_webhook_msg(message="Using lycoris training mode.")

            if self.config.init_lora is not None:
                from lycoris import create_lycoris_from_weights

                self.lycoris_wrapped_network = create_lycoris_from_weights(
                    multiplier,
                    self.config.init_lora,
                    self.model.get_trained_component(),
                    weights_sd=None,
                    **self.lycoris_config,
                )[0]
            else:
                self.lycoris_wrapped_network = create_lycoris(
                    self.model.get_trained_component(),
                    multiplier,
                    linear_dim,
                    linear_alpha,
                    **self.lycoris_config,
                )

                if self.config.init_lokr_norm is not None:
                    init_lokr_network_with_perturbed_normal(
                        self.lycoris_wrapped_network,
                        scale=self.config.init_lokr_norm,
                    )

            self.lycoris_wrapped_network.apply_to()
            setattr(
                self.accelerator,
                "_lycoris_wrapped_network",
                self.lycoris_wrapped_network,
            )
            lycoris_num_params = sum(
                p.numel() for p in self.lycoris_wrapped_network.parameters()
            )
            logger.info(
                f"LyCORIS network has been initialized with {lycoris_num_params:,} parameters"
            )
        self.accelerator.wait_for_everyone()

    def init_post_load_freeze(self):
        if self.config.layer_freeze_strategy == "bitfit":
            from helpers.training.model_freeze import apply_bitfit_freezing

            if self.model.get_trained_component() is not None:
                logger.info(
                    f"Applying BitFit freezing strategy to the {self.model.MODEL_TYPE.value}."
                )
                self.model.model = apply_bitfit_freezing(
                    unwrap_model(self.accelerator, self.model.model), self.config
                )
        self.enable_gradient_checkpointing()

    def enable_gradient_checkpointing(self):
        if self.config.gradient_checkpointing:
            logger.debug("Enabling gradient checkpointing.")
            if hasattr(
                self.model.get_trained_component(), "enable_gradient_checkpointing"
            ):
                unwrap_model(
                    self.accelerator, self.model.get_trained_component()
                ).enable_gradient_checkpointing()
            if (
                hasattr(self.config, "train_text_encoder")
                and self.config.train_text_encoder
            ):
                for text_encoder in self.model.text_encoders:
                    if text_encoder is not None:
                        unwrap_model(
                            self.accelerator, text_encoder
                        ).gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self):
        if self.config.gradient_checkpointing:
            logger.debug("Disabling gradient checkpointing.")
            if hasattr(
                self.model.get_trained_component(), "disable_gradient_checkpointing"
            ):
                unwrap_model(
                    self.accelerator, self.model.get_trained_component()
                ).disable_gradient_checkpointing()
            if self.config.controlnet:
                unwrap_model(
                    self.accelerator, self.controlnet
                ).disable_gradient_checkpointing()
            if (
                hasattr(self.config, "train_text_encoder")
                and self.config.train_text_encoder
            ):
                unwrap_model(
                    self.accelerator, self.text_encoder_1
                ).gradient_checkpointing_disable()
                unwrap_model(
                    self.accelerator, self.text_encoder_2
                ).gradient_checkpointing_disable()

    def _get_trainable_parameters(self):
        # Return just a list of the currently trainable parameters.
        if self.config.model_type == "lora":
            if self.config.lora_type == "lycoris":
                return self.lycoris_wrapped_network.parameters()
        return [
            param
            for param in self.model.get_trained_component().parameters()
            if param.requires_grad
        ]

    def _recalculate_training_steps(self):
        # Scheduler and math around the number of training steps.
        if not hasattr(self.config, "overrode_max_train_steps"):
            self.config.overrode_max_train_steps = False
        self.config.total_num_batches = sum(
            [
                len(
                    backend["metadata_backend"] if "metadata_backend" in backend else []
                )
                for _, backend in StateTracker.get_data_backends().items()
            ]
        )
        self.config.num_update_steps_per_epoch = math.ceil(
            self.config.total_num_batches / self.config.gradient_accumulation_steps
        )
        if getattr(self.config, "overrode_max_train_steps", False):
            self.config.max_train_steps = (
                self.config.num_train_epochs * self.config.num_update_steps_per_epoch
            )
            # Afterwards we recalculate our number of training epochs
            self.config.num_train_epochs = math.ceil(
                self.config.max_train_steps / self.config.num_update_steps_per_epoch
            )
            logger.info(
                "After removing any undesired samples and updating cache entries, we have settled on"
                f" {self.config.num_train_epochs} epochs and {self.config.num_update_steps_per_epoch} steps per epoch."
            )
        if self.config.max_train_steps is None or self.config.max_train_steps == 0:
            if (
                self.config.num_train_epochs is None
                or self.config.num_train_epochs == 0
            ):
                raise ValueError(
                    "You must specify either --max_train_steps or --num_train_epochs with a value > 0"
                )
            self.config.max_train_steps = (
                self.config.num_train_epochs * self.config.num_update_steps_per_epoch
            )
            logger.info(
                f"Calculated our maximum training steps at {self.config.max_train_steps} because we have"
                f" {self.config.num_train_epochs} epochs and {self.config.num_update_steps_per_epoch} steps per epoch."
            )
            self.config.overrode_max_train_steps = True
        elif self.config.num_train_epochs is None or self.config.num_train_epochs == 0:
            if self.config.max_train_steps is None or self.config.max_train_steps == 0:
                raise ValueError(
                    "You must specify either --max_train_steps or --num_train_epochs with a value > 0"
                )
            self.config.num_train_epochs = math.ceil(
                self.config.max_train_steps
                / max(self.config.num_update_steps_per_epoch, 1)
            )
            logger.info(
                f"Calculated our maximum training steps at {self.config.max_train_steps} because we have"
                f" {self.config.num_train_epochs} epochs and {self.config.num_update_steps_per_epoch} steps per epoch."
            )
        if self.lr_scheduler is not None and hasattr(
            self.lr_scheduler, "num_update_steps_per_epoch"
        ):
            self.lr_scheduler.num_update_steps_per_epoch = (
                self.config.num_update_steps_per_epoch
            )
        self.config.total_batch_size = (
            self.config.train_batch_size
            * self.accelerator.num_processes
            * self.config.gradient_accumulation_steps
        )

    def init_optimizer(self):
        logger.info(f"Learning rate: {self.config.learning_rate}")
        extra_optimizer_args = {"lr": self.config.learning_rate}
        # Initialize the optimizer
        optimizer_args_from_config, optimizer_class = (
            determine_optimizer_class_with_config(
                args=self.config,
                use_deepspeed_optimizer=self.config.use_deepspeed_optimizer,
                is_quantized=self.config.is_quantized,
                enable_adamw_bf16=self.config.enable_adamw_bf16,
            )
        )
        extra_optimizer_args.update(optimizer_args_from_config)

        self.params_to_optimize = determine_params_to_optimize(
            args=self.config,
            model=self.model,
            model_type_label=self.config.model_type_label,
            lycoris_wrapped_network=self.lycoris_wrapped_network,
        )

        if self.config.use_deepspeed_optimizer:
            logger.info(
                f"DeepSpeed Optimizer arguments, weight_decay={self.config.adam_weight_decay} eps={self.config.adam_epsilon}, extra_arguments={extra_optimizer_args}"
            )
            self.optimizer = optimizer_class(self.params_to_optimize)
        else:
            logger.info(f"Optimizer arguments={extra_optimizer_args}")
            if self.config.train_text_encoder and self.config.text_encoder_lr:
                # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
                # --learning_rate
                self.params_to_optimize[1]["lr"] = float(self.config.learning_rate)
                if self.text_encoder_2 is not None:
                    self.params_to_optimize[2]["lr"] = float(self.config.learning_rate)

            self.optimizer = cpu_offload_optimizer(
                params_to_optimize=self.params_to_optimize,
                optimizer_cls=optimizer_class,
                optimizer_parameters=extra_optimizer_args,
                fused=self.config.fuse_optimizer,
                offload_gradients=self.config.optimizer_offload_gradients,
                offload_mechanism=self.config.optimizer_cpu_offload_method,
            )

        if (
            is_optimi_available
            and self.config.optimizer_release_gradients
            and "optimi" in self.config.optimizer
        ):
            logger.warning(
                "Marking model for gradient release. This feature is experimental, and may use more VRAM or not work."
            )
            prepare_for_gradient_release(
                self.model.get_trained_component(), self.optimizer
            )

    def init_lr_scheduler(self):
        self.config.is_schedulefree = is_lr_schedulefree(self.config.optimizer)
        self.config.is_lr_scheduler_disabled = (
            is_lr_scheduler_disabled(self.config.optimizer)
            or self.config.use_deepspeed_scheduler
        )
        if self.config.is_schedulefree:
            logger.info("Using experimental ScheduleFree optimiser..")
        if self.config.is_lr_scheduler_disabled:
            # we don't use LR schedulers with schedulefree optimisers
            logger.info("Optimiser cannot use an LR scheduler, so we are disabling it.")
            lr_scheduler = None
            logger.info(f"Using dummy learning rate scheduler")
            if torch.backends.mps.is_available():
                lr_scheduler = None
            else:
                lr_scheduler = accelerate.utils.DummyScheduler(
                    self.optimizer,
                    total_num_steps=self.config.max_train_steps,
                    warmup_num_steps=self.config.lr_warmup_steps,
                )
            return lr_scheduler

        logger.info(
            f"Loading {self.config.lr_scheduler} learning rate scheduler with {self.config.lr_warmup_steps} warmup steps"
        )
        lr_scheduler = get_lr_scheduler(
            self.config,
            self.optimizer,
            self.accelerator,
            logger,
            global_step=self.state["global_step"],
            use_deepspeed_scheduler=False,
        )
        if hasattr(lr_scheduler, "num_update_steps_per_epoch"):
            lr_scheduler.num_update_steps_per_epoch = (
                self.config.num_update_steps_per_epoch
            )
        if hasattr(lr_scheduler, "last_step"):
            lr_scheduler.last_step = self.state.get("global_resume_step", 0)

        return lr_scheduler

    def init_ema_model(self):
        # Create EMA for the model.
        self.ema_model = None
        if not self.config.use_ema:
            return
        if self.accelerator.is_main_process:
            logger.info("Using EMA. Creating EMAModel.")

            ema_model_cls = None
            ema_model_config = None
            if self.config.controlnet:
                ema_model_cls = self.controlnet.__class__
                ema_model_config = self.controlnet.config
            elif self.model.get_trained_component() is not None:
                ema_model_cls = self.model.get_trained_component().__class__
                ema_model_config = self.model.get_trained_component().config
            else:
                raise ValueError(
                    f"Please open a bug report or disable EMA. Unknown EMA model family: {self.config.model_family}"
                )

            self.ema_model = EMAModel(
                self.config,
                self.accelerator,
                parameters=self._get_trainable_parameters(),
                model_cls=ema_model_cls,
                model_config=ema_model_config,
                decay=self.config.ema_decay,
                foreach=not self.config.ema_foreach_disable,
            )
            logger.info(
                f"EMA model creation completed with {self.ema_model.parameter_count():,} parameters"
            )

        self.accelerator.wait_for_everyone()

    def init_hooks(self):
        from helpers.training.save_hooks import SaveHookManager

        self.model_hooks = SaveHookManager(
            args=self.config,
            model=self.model,
            ema_model=self.ema_model,
            accelerator=self.accelerator,
            use_deepspeed_optimizer=self.config.use_deepspeed_optimizer,
        )
        self.accelerator.register_save_state_pre_hook(self.model_hooks.save_model_hook)
        self.accelerator.register_load_state_pre_hook(self.model_hooks.load_model_hook)

    def init_prepare_models(self, lr_scheduler):
        # Prepare everything with our `accelerator`.
        logger.info("Preparing models..")

        # TODO: Is this still needed? Seems like a hack job from January 2024.
        self.train_dataloaders = []
        for _, backend in StateTracker.get_data_backends().items():
            if "train_dataloader" not in backend:
                continue
            self.train_dataloaders.append(backend["train_dataloader"])
            break
        if len(self.train_dataloaders) == 0:
            logger.error("For some reason, no dataloaders were configured.")
            sys.exit(0)
        if self.config.disable_accelerator:
            logger.warning(
                "Because SIMPLETUNER_DISABLE_ACCELERATOR is set, we will not prepare the accelerator."
            )
            return
        logger.info("Loading our accelerator...")
        if torch.backends.mps.is_available():
            self.accelerator.native_amp = False
        self._send_webhook_msg(message="Moving weights to GPU...")
        self._send_webhook_raw(
            structured_data={"message": "Moving weights to GPU"},
            message_type="init_prepare_models_begin",
        )
        primary_model = self.model.get_trained_component()
        results = self.accelerator.prepare(
            primary_model, lr_scheduler, self.optimizer, self.train_dataloaders[0]
        )
        self.model.set_prepared_model(results[0])

        self.lr_scheduler = results[1]
        self.optimizer = results[2]
        # The rest of the entries are dataloaders:
        self.train_dataloaders = [results[3:]]
        if self.config.use_ema and self.ema_model is not None:
            if self.config.ema_device == "accelerator":
                logger.info("Moving EMA model weights to accelerator...")
            print(f"EMA model: {self.ema_model}")
            self.ema_model.to(
                (
                    self.accelerator.device
                    if self.config.ema_device == "accelerator"
                    else "cpu"
                ),
                dtype=self.config.weight_dtype,
            )

            if self.config.ema_device == "cpu" and not self.config.ema_cpu_only:
                logger.info("Pinning EMA model weights to CPU...")
                try:
                    self.ema_model.pin_memory()
                except Exception as e:
                    self._send_webhook_raw(
                        structured_data={"message": f"Failed to pin EMA to CPU: {e}"},
                        message_type="error",
                    )
                    logger.error(f"Failed to pin EMA model to CPU: {e}")

        idx_count = 0
        for _, backend in StateTracker.get_data_backends().items():
            if idx_count == 0 or "train_dataloader" not in backend:
                continue
            self.train_dataloaders.append(
                self.accelerator.prepare(backend["train_dataloader"])
            )
        idx_count = 0

        if "lora" in self.config.model_type and self.config.train_text_encoder:
            logger.info("Preparing text encoders for training.")
            if self.config.model_family == "sd3":
                logger.info("NOTE: The third text encoder is not trained for SD3.")
            self.text_encoder_1, self.text_encoder_2 = self.accelerator.prepare(
                self.text_encoder_1, self.text_encoder_2
            )
        self._recalculate_training_steps()
        self.accelerator.wait_for_everyone()
        self._send_webhook_raw(
            structured_data={"message": "Completed moving weights to GPU"},
            message_type="init_prepare_models_completed",
        )

    def init_unload_vae(self):
        if self.config.keep_vae_loaded or self.config.vae_cache_ondemand:
            return
        memory_before_unload = self.stats_memory_used()
        self.model.unload_vae()
        for _, backend in StateTracker.get_data_backends().items():
            if "vaecache" in backend:
                backend["vaecache"].vae = None
        reclaim_memory()
        memory_after_unload = self.stats_memory_used()
        memory_saved = memory_after_unload - memory_before_unload
        logger.info(
            f"After nuking the VAE from orbit, we freed {abs(round(memory_saved, 2)) * 1024} MB of VRAM."
        )

    def init_validations(self):
        if (
            hasattr(self.accelerator, "state")
            and hasattr(self.accelerator.state, "deepspeed_plugin")
            and getattr(self.accelerator.state.deepspeed_plugin, "deepspeed_config", {})
            .get("zero_optimization", {})
            .get("stage")
            == 3
        ):
            logger.error("Cannot run validations with DeepSpeed ZeRO stage 3.")
            return
        self.evaluation = None
        if self.config.validation_disable:
            return
        if (
            self.config.eval_steps_interval is not None
            and self.config.eval_steps_interval > 0
        ):
            from helpers.training.validation import Evaluation

            self.evaluation = Evaluation(accelerator=self.accelerator)
        model_evaluator = ModelEvaluator.from_config(args=self.config)
        self.validation = Validation(
            trainable_parameters=self._get_trainable_parameters,
            accelerator=self.accelerator,
            model=self.model,
            args=self.config,
            validation_prompt_metadata=self.validation_prompt_metadata,
            vae_path=self.config.vae_path,
            weight_dtype=self.config.weight_dtype,
            embed_cache=StateTracker.get_default_text_embed_cache(),
            ema_model=self.ema_model,
            model_evaluator=model_evaluator,
            is_deepspeed=self.config.use_deepspeed_optimizer,
        )
        if not self.config.train_text_encoder and self.validation is not None:
            self.validation.clear_text_encoders()
        self.init_benchmark_base_model()
        self.accelerator.wait_for_everyone()

    def init_benchmark_base_model(self):
        if (
            self.config.disable_benchmark
            or self.validation is None
            or self.validation.benchmark_exists("base_model")
        ):
            # if we've disabled it or the benchmark exists, we will not do it again.
            # deepspeed zero3 can't do validations at all.
            return
        if not self.accelerator.is_main_process:
            return
        logger.info(
            "Benchmarking base model for comparison. Supply `--disable_benchmark: true` to disable this behaviour."
        )
        self._send_webhook_raw(
            structured_data={"message": "Base model benchmark begins"},
            message_type="init_benchmark_base_model_begin",
        )
        # we'll run validation on base model if it hasn't already.
        self.validation.run_validations(validation_type="base_model", step=0)
        self.validation.save_benchmark("base_model")
        self._send_webhook_raw(
            structured_data={"message": "Base model benchmark completed"},
            message_type="init_benchmark_base_model_completed",
        )

    def init_resume_checkpoint(self, lr_scheduler):
        # Potentially load in the weights and states from a previous save
        self.config.total_steps_remaining_at_start = self.config.max_train_steps
        self.state["current_epoch"] = self.state["first_epoch"]
        self.state["global_resume_step"] = self.state["global_step"] = (
            StateTracker.get_global_step()
        )
        StateTracker.set_global_resume_step(self.state["global_resume_step"])
        if not self.config.resume_from_checkpoint:
            return lr_scheduler
        if self.config.resume_from_checkpoint != "latest":
            path = os.path.basename(self.config.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            path = self.checkpoint_state_latest(self.config.output_dir)

        if path is None:
            logger.info(
                f"Checkpoint '{self.config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            self._send_webhook_raw(
                structured_data={
                    "message": "No model to resume. Beginning fresh training run."
                },
                message_type="init_resume_checkpoint",
            )

            self.config.resume_from_checkpoint = None
            return lr_scheduler

        logger.info(f"Resuming from checkpoint {path}")
        self.accelerator.load_state(os.path.join(self.config.output_dir, path))
        try:
            if (
                "constant" == self.config.lr_scheduler
                and not self.config.is_schedulefree
            ):
                for g in self.optimizer.param_groups:
                    if "lr" in g:
                        g["lr"] = self.config.learning_rate
                for k, v in lr_scheduler.state_dict().items():
                    if k in ("base_lrs", "_last_lr"):
                        v[0] = self.config.learning_rate
        except Exception as e:
            self._send_webhook_raw(
                structured_data={
                    "message": "Could not update learning rate scheduler LR value."
                },
                message_type="warning",
            )
            logger.error(
                f"Could not update lr_scheduler {self.config.lr_scheduler} learning rate to {self.config.learning_rate} upon resume: {e}"
            )

        self._send_webhook_raw(
            structured_data={"message": f"Resuming model: {path}"},
            message_type="init_resume_checkpoint",
        )
        training_state_filename = f"training_state.json"
        if get_rank() > 0:
            training_state_filename = f"training_state-{get_rank()}.json"
        for _, backend in StateTracker.get_data_backends().items():
            if "sampler" in backend:
                backend["sampler"].load_states(
                    state_path=os.path.join(
                        self.config.output_dir,
                        path,
                        training_state_filename,
                    ),
                )
        self.state["global_resume_step"] = self.state["global_step"] = (
            StateTracker.get_global_step()
        )
        StateTracker.set_global_resume_step(self.state["global_resume_step"])
        training_state_in_ckpt = StateTracker.get_training_state()
        self._send_webhook_raw(
            structured_data=training_state_in_ckpt,
            message_type="init_resume_checkpoint_details",
        )
        logger.debug(f"Training state inside checkpoint: {training_state_in_ckpt}")
        if hasattr(lr_scheduler, "last_step"):
            lr_scheduler.last_step = self.state["global_resume_step"]
        logger.info(f"Resuming from global_step {self.state['global_resume_step']}.")

        # Log the current state of each data backend.
        for _, backend in StateTracker.get_data_backends().items():
            if "sampler" in backend:
                backend["sampler"].log_state()
        # We store the number of dataset resets that have occurred inside the checkpoint.
        self.state["first_epoch"] = StateTracker.get_epoch()
        if self.state["first_epoch"] > 1 or self.state["global_resume_step"] > 1:
            self.config.total_steps_remaining_at_start -= self.state[
                "global_resume_step"
            ]
            logger.debug(
                f"Resuming from epoch {self.state['first_epoch']}, which leaves us with {self.config.total_steps_remaining_at_start}."
            )
        self.state["current_epoch"] = self.state["first_epoch"]
        StateTracker.set_epoch(self.state["current_epoch"])
        if hasattr(lr_scheduler, "last_epoch"):
            lr_scheduler.last_epoch = (
                training_state_in_ckpt.get(
                    "epoch_step", self.state.get("global_resume_step", 1)
                )
                * self.accelerator.num_processes
            )

        if (
            self.state["current_epoch"] > self.config.num_train_epochs + 1
            and not self.config.ignore_final_epochs
        ):
            logger.info(
                f"Reached the end ({self.state['current_epoch']} epochs) of our training run ({self.config.num_train_epochs} epochs). This run will do zero steps."
            )
        self.accelerator.wait_for_everyone()

        if self.optimizer is not None and self.config.optimizer == "prodigy":
            # fix the device assignment for the prodigy optimizer parameters
            for group in (
                self.optimizer.param_groups
                if self.optimizer.optimizer.split_groups
                else self.optimizer.param_groups[:1]
            ):
                p = group["params"][0]
                group["running_d_numerator"] = group["running_d_numerator"].to(p.device)
                group["running_d_denom"] = group["running_d_denom"].to(p.device)
                if "use_focus" not in group:
                    group["use_focus"] = False

        return lr_scheduler

    def init_trackers(self):
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        self.guidance_values_table = None
        if self.accelerator.is_main_process:
            # Copy args into public_args:
            public_args = copy.deepcopy(self.config)
            delattr(public_args, "accelerator_project_config")
            delattr(public_args, "process_group_kwargs")
            delattr(public_args, "weight_dtype")
            delattr(public_args, "base_weight_dtype")
            if hasattr(public_args, "vae_kwargs"):
                delattr(public_args, "vae_kwargs")
            if hasattr(public_args, "sana_complex_human_instruction"):
                delattr(public_args, "sana_complex_human_instruction")

            # Hash the contents of public_args to reflect a deterministic ID for a single set of params:
            public_args_hash = hashlib.md5(
                json.dumps(vars(public_args), sort_keys=True).encode("utf-8")
            ).hexdigest()
            project_name = self.config.tracker_project_name or "simpletuner-training"
            tracker_run_name = (
                self.config.tracker_run_name or "simpletuner-training-run"
            )
            try:
                self.accelerator.init_trackers(
                    project_name,
                    config=vars(public_args),
                    init_kwargs={
                        "wandb": {
                            "name": tracker_run_name,
                            "id": f"{public_args_hash}",
                            "resume": "allow",
                            "allow_val_change": True,
                        }
                    },
                )
            except Exception as e:
                if "Object has no attribute 'disabled'" in repr(e):
                    logger.warning(
                        "WandB is disabled, and Accelerate was not quite happy about it."
                    )
                else:
                    logger.error(f"Could not initialize trackers: {e}")
                    self._send_webhook_raw(
                        structured_data={
                            "message": f"Could not initialize trackers. Continuing without. {e}"
                        },
                        message_type="error",
                    )
            self._send_webhook_raw(
                structured_data=public_args.__dict__,
                message_type="training_config",
            )

    def resume_and_prepare(self):
        self.init_optimizer()
        lr_scheduler = self.init_lr_scheduler()
        self.init_hooks()
        self.init_prepare_models(lr_scheduler=lr_scheduler)
        lr_scheduler = self.init_resume_checkpoint(lr_scheduler=lr_scheduler)
        self.init_post_load_freeze()

    def enable_sageattention_inference(self):
        # if the sageattention is inference-only, we'll enable it.
        # if it's training only, we'll disable it.
        # if it's inference+training, we leave it alone.
        if (
            "sageattention" not in self.config.attention_mechanism
            or self.config.sageattention_usage == "training+inference"
        ):
            return
        if self.config.sageattention_usage == "inference":
            self.enable_sageattention()
        if self.config.sageattention_usage == "training":
            self.disable_sageattention()

    def disable_sageattention_inference(self):
        # if the sageattention is inference-only, we'll disable it.
        # if it's training only, we'll enable it.
        # if it's inference+training, we leave it alone.
        if (
            "sageattention" not in self.config.attention_mechanism
            or self.config.sageattention_usage == "training+inference"
        ):
            return
        if self.config.sageattention_usage == "inference":
            self.disable_sageattention()
        if self.config.sageattention_usage == "training":
            self.enable_sageattention()

    def disable_sageattention(self):
        if "sageattention" not in self.config.attention_mechanism:
            return

        if (
            hasattr(torch.nn.functional, "scaled_dot_product_attention_sdpa")
            and torch.nn.functional
            != torch.nn.functional.scaled_dot_product_attention_sdpa
        ):
            logger.info("Disabling SageAttention.")
            setattr(
                torch.nn.functional,
                "scaled_dot_product_attention",
                torch.nn.functional.scaled_dot_product_attention_sdpa,
            )

    def enable_sageattention(self):
        if "sageattention" not in self.config.attention_mechanism:
            return

        # we'll try and load SageAttention and overload pytorch's sdpa function.
        try:
            logger.info("Enabling SageAttention.")
            from sageattention import (
                sageattn,
                sageattn_qk_int8_pv_fp16_triton,
                sageattn_qk_int8_pv_fp16_cuda,
                sageattn_qk_int8_pv_fp8_cuda,
            )

            sageattn_functions = {
                "sageattention": sageattn,
                "sageattention-int8-fp16-triton": sageattn_qk_int8_pv_fp16_triton,
                "sageattention-int8-fp16-cuda": sageattn_qk_int8_pv_fp16_cuda,
                "sageattention-int8-fp8-cuda": sageattn_qk_int8_pv_fp8_cuda,
            }
            # store the old SDPA for validations to use during VAE decode
            if not hasattr(torch.nn.functional, "scaled_dot_product_attention_sdpa"):
                setattr(
                    torch.nn.functional,
                    "scaled_dot_product_attention_sdpa",
                    torch.nn.functional.scaled_dot_product_attention,
                )

            def sageattn_wrapper_for_torch_sdpa_with_fallback(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=None,
                enable_gqa=False,
            ) -> torch.Tensor:
                try:
                    return sageattn_functions[self.config.attention_mechanism](
                        query, key, value, is_causal=is_causal
                    )
                except:
                    logger.error(
                        f"Could not run SageAttention with {self.config.attention_mechanism}. Falling back to Pytorch SDPA."
                    )
                    return torch.nn.functional.scaled_dot_product_attention_sdpa(
                        query,
                        key,
                        value,
                        attn_mask,
                        dropout_p,
                        is_causal,
                        scale,
                        enable_gqa,
                    )

            torch.nn.functional.scaled_dot_product_attention = (
                sageattn_wrapper_for_torch_sdpa_with_fallback
            )
            if not hasattr(torch.nn.functional, "scaled_dot_product_attention_sage"):
                setattr(
                    torch.nn.functional,
                    "scaled_dot_product_attention_sage",
                    torch.nn.functional.scaled_dot_product_attention,
                )

            if "training" in self.config.sageattention_usage:
                logger.warning(
                    f"Using {self.config.attention_mechanism} for attention calculations during training. Your attention layers will not be trained. To disable SageAttention, remove or set --attention_mechanism to a different value."
                )
        except ImportError as e:
            logger.error(
                "Could not import SageAttention. Please install it to use this --attention_mechanism=sageattention."
            )
            logger.error(repr(e))
            sys.exit(1)

    def move_models(self, destination: str = "accelerator"):
        target_device = "cpu"
        if destination == "accelerator":
            target_device = self.accelerator.device
        logger.info(
            f"Moving the {str(self.model.get_trained_component().__class__)} to GPU in {self.config.weight_dtype if not self.config.is_quantized else self.config.base_model_precision} precision."
        )
        if self.model.get_trained_component() is not None:
            if self.config.is_quantized:
                self.model.get_trained_component().to(target_device)
            else:
                self.model.get_trained_component().to(
                    target_device, dtype=self.config.weight_dtype
                )
        if getattr(self.accelerator, "_lycoris_wrapped_network", None) is not None:
            self.accelerator._lycoris_wrapped_network = (
                self.accelerator._lycoris_wrapped_network.to(
                    target_device, dtype=self.config.weight_dtype
                )
            )

        if (
            "sageattention" in self.config.attention_mechanism
            and "training" in self.config.sageattention_usage
        ):
            logger.info(
                "Using SageAttention for training. This is an unsupported, experimental configuration."
            )
            self.enable_sageattention()
        elif self.config.attention_mechanism == "xformers":
            if is_xformers_available():
                import xformers  # type: ignore # noqa

                if hasattr(
                    self.model.get_trained_component(),
                    "enable_xformers_memory_efficient_attention",
                ):
                    logger.info("Enabling xformers memory-efficient attention.")
                    self.model.get_trained_component().enable_xformers_memory_efficient_attention()
                else:
                    self.config.enable_xformers_memory_efficient_attention = False
                    self.config.attention_mechanism = "diffusers"
                    logger.warning(
                        "xformers is not enabled, as it is incompatible with this model type."
                        " Falling back to diffusers attention mechanism (Pytorch SDPA)."
                        " Alternatively, provide --attention_mechanism=sageattention for a more efficient option on CUDA systems."
                    )
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

        if self.config.controlnet:
            self.model.get_trained_component().train()
            logger.info(
                f"Moving ControlNet to {target_device} in {self.config.weight_dtype} precision."
            )
            self.model.unwrap_model(self.model.model).to(
                device=target_device, dtype=self.config.weight_dtype
            )
            if self.config.train_text_encoder:
                logger.warning(
                    "Unknown results will occur when finetuning the text encoder alongside ControlNet."
                )

    def mark_optimizer_train(self):
        if is_lr_schedulefree(self.config.optimizer) and hasattr(
            self.optimizer, "train"
        ):
            # we typically have to call train() on the optim for schedulefree.
            logger.debug("Setting optimiser into train() mode.")
            self.optimizer.train()

    def mark_optimizer_eval(self):
        if is_lr_schedulefree(self.config.optimizer) and hasattr(
            self.optimizer, "eval"
        ):
            # we typically have to call eval() on the optim for schedulefree before saving or running validations.
            logger.debug("Setting optimiser into eval() mode.")
            self.optimizer.eval()

    def _send_webhook_msg(
        self, message: str, message_level: str = "info", store_response: bool = False
    ):
        if type(message) is not str:
            logger.error(
                f"_send_webhook_msg received {type(message)} type message instead of str."
            )
            return False
        if self.webhook_handler is None or not self.webhook_handler:
            return
        self.webhook_handler.send(
            message=message, message_level=message_level, store_response=store_response
        )

    def _send_webhook_raw(
        self,
        structured_data: dict,
        message_type: str,
        message_level: str = "info",
    ):
        if type(structured_data) is not dict:
            logger.error(
                f"_send_webhook_msg received {type(structured_data)} type message instead of dict."
            )
            return False
        if not self.webhook_handler:
            return
        self.webhook_handler.send_raw(
            structured_data=structured_data,
            message_type=message_type,
            message_level=message_level,
            job_id=self.job_id,
        )

    def _train_initial_msg(self):
        initial_msg = "\n***** Running training *****"
        initial_msg += f"\n-  Num batches = {self.config.total_num_batches}"
        initial_msg += f"\n-  Num Epochs = {self.config.num_train_epochs}"
        initial_msg += f"\n  - Current Epoch = {self.state['first_epoch']}"
        initial_msg += f"\n-  Total train batch size (w. parallel, distributed & accumulation) = {self.config.total_batch_size}"
        initial_msg += f"\n  - Instantaneous batch size per device = {self.config.train_batch_size}"
        initial_msg += f"\n  - Gradient Accumulation steps = {self.config.gradient_accumulation_steps}"
        initial_msg += f"\n-  Total optimization steps = {self.config.max_train_steps}"
        if self.state["global_step"] > 1:
            initial_msg += f"\n  - Steps completed: {self.state['global_step']}"
        initial_msg += f"\n-  Total optimization steps remaining = {max(0, getattr(self.config, 'total_steps_remaining_at_start', self.config.max_train_steps))}"
        logger.info(initial_msg)
        self._send_webhook_msg(message=initial_msg)
        structured_data = {
            "total_num_batches": self.config.total_num_batches,
            "total_num_epochs": self.config.num_train_epochs,
            "total_num_steps": self.config.max_train_steps,
            "current_epoch": self.state["first_epoch"],
            "total_batch_size": self.config.total_batch_size,
            "micro_batch_size": self.config.train_batch_size,
            "current_step": self.state["global_step"],
            "remaining_num_steps": max(
                0,
                getattr(
                    self.config,
                    "total_steps_remaining_at_start",
                    self.config.max_train_steps,
                ),
            ),
        }
        self._send_webhook_raw(
            structured_data=structured_data, message_type="_train_initial_msg"
        )

    def _epoch_rollover(self, epoch):
        if self.state["first_epoch"] == epoch:
            return
        logger.debug(
            f"Just completed epoch {self.state['current_epoch']}. Beginning epoch {epoch}. Starting epoch was {self.state['first_epoch']}. Final epoch will be {self.config.num_train_epochs}"
        )
        for backend_id, backend in StateTracker.get_data_backends().items():
            backend_config = StateTracker.get_data_backend_config(backend_id)
            if (
                backend_config.get("crop")
                and backend_config.get("crop_aspect") == "random"
                and "metadata_backend" in backend
                and not self.config.aspect_bucket_disable_rebuild
            ) or (
                backend_config.get("vae_cache_clear_each_epoch")
                and "vaecache" in backend
            ):
                # when the aspect ratio is random, we need to shuffle the dataset on each epoch.
                if self.accelerator.is_main_process:
                    # we only compute the aspect ratio indices on the main process.
                    # we have to set read_only to False since we're generating a new, un-split list.
                    # otherwise, we can't actually save the new cache to disk.
                    backend["metadata_backend"].read_only = False
                    # this will generate+save the new cache to the storage backend.
                    backend["metadata_backend"].compute_aspect_ratio_bucket_indices(
                        ignore_existing_cache=True
                    )
                self.accelerator.wait_for_everyone()
                logger.info(f"Reloading cache for backend {backend_id}")
                backend["metadata_backend"].reload_cache(set_config=False)
                logger.info("Waiting for other threads to finish..")
                self.accelerator.wait_for_everyone()
                # we'll have to split the buckets between GPUs again now, so that the VAE cache distributes properly.
                logger.info("Splitting buckets across GPUs")
                backend["metadata_backend"].split_buckets_between_processes(
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps
                )
                # we have to rebuild the VAE cache if it exists.
                if "vaecache" in backend:
                    logger.info("Rebuilding VAE cache..")
                    backend["vaecache"].rebuild_cache()
                # no need to manually call metadata_backend.save_cache() here.
        self.state["current_epoch"] = epoch
        StateTracker.set_epoch(epoch)
        if self.config.lr_scheduler == "cosine_with_restarts":
            self.extra_lr_scheduler_kwargs["epoch"] = epoch

    def _exit_on_signal(self):
        if self.should_abort:
            self._send_webhook_raw(
                structured_data={"message": "Aborting training run."},
                message_type="exit",
            )
            raise StopIteration("Training run received abort signal.")

    def abort(self):
        logger.info("Aborting training run.")
        if self.bf is not None:
            self.bf.stop_fetching()
        # we should set should_abort = True on each data backend's vae cache, metadata, and text backend
        for _, backend in StateTracker.get_data_backends().items():
            if "vaecache" in backend:
                logger.debug(f"Aborting VAE cache")
                backend["vaecache"].should_abort = True
            if "metadata_backend" in backend:
                logger.debug(f"Aborting metadata backend")
                backend["metadata_backend"].should_abort = True
            if "text_backend" in backend:
                logger.debug(f"Aborting text backend")
                backend["text_backend"].should_abort = True
            if "sampler" in backend:
                logger.debug(f"Aborting sampler")
                backend["sampler"].should_abort = True
        self.should_abort = True

    def model_predict(
        self,
        prepared_batch,
        custom_timesteps: list = None,
    ):
        if self.config.controlnet:
            training_logger.debug(
                f"Extra conditioning dtype: {prepared_batch['conditioning_pixel_values'].dtype}"
            )
        if custom_timesteps is not None:
            timesteps = custom_timesteps
        if not self.config.disable_accelerator:
            if self.config.controlnet:
                model_pred = self.model.controlnet_predict(
                    prepared_batch=prepared_batch,
                )
            else:
                model_pred = self.model.model_predict(
                    prepared_batch=prepared_batch,
                )

        else:
            # Dummy model prediction for debugging.
            model_pred = torch.randn_like(prepared_batch["noisy_latents"])

        # x-prediction requires that we now subtract the noise residual from the prediction to get the target sample.
        if (
            hasattr(self.noise_scheduler, "config")
            and hasattr(self.noise_scheduler.config, "prediction_type")
            and self.noise_scheduler.config.prediction_type == "sample"
        ):
            model_pred = model_pred - prepared_batch["noise"]

        return model_pred

    def _max_grad_value(self):
        max_grad_value = float("-inf")  # Start with a very small number
        for param in self._get_trainable_parameters():
            if param.grad is not None:
                max_grad_value = max(max_grad_value, param.grad.abs().max().item())

        return max_grad_value

    def prepare_batch(self, batch: dict):
        """
        Prepare a batch for the model prediction.

        Args:
            batch (dict): Batch from iterator_fn.

        Returns:
            batch (dict): Prepared batch.
        """
        if not batch:
            training_logger.debug(
                f"No batch was returned by the iterator_fn, returning {batch}"
            )
            return batch

        return self.model.prepare_batch(batch, state=self.state)

    def get_prediction_target(self, prepared_batch: dict):
        return self.model.get_prediction_target(prepared_batch)

    def _calculate_loss(
        self,
        prepared_batch: dict,
        model_pred,
        target,
        apply_conditioning_mask: bool = True,
    ):
        # Compute the per-pixel loss without reducing over spatial dimensions
        if self.config.flow_matching:
            # For flow matching, compute the per-pixel squared differences
            loss = (
                model_pred.float() - target.float()
            ) ** 2  # Shape: (batch_size, C, H, W)
        elif self.config.snr_gamma is None or self.config.snr_gamma == 0:
            training_logger.debug("Calculating loss")
            loss = self.config.snr_weight * F.mse_loss(
                model_pred.float(), target.float(), reduction="none"
            )  # Shape: (batch_size, C, H, W)
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            training_logger.debug("Using min-SNR loss")
            snr = compute_snr(prepared_batch["timesteps"], self.noise_scheduler)
            snr_divisor = snr
            if self.noise_scheduler.config.prediction_type == "v_prediction":
                snr_divisor = snr + 1

            training_logger.debug("Calculating MSE loss weights using SNR as divisor")
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
            )  # Shape: (batch_size,)

            # Compute the per-pixel MSE loss without reduction
            loss = F.mse_loss(
                model_pred.float(), target.float(), reduction="none"
            )  # Shape: (batch_size, C, H, W)

            # Reshape mse_loss_weights for broadcasting and apply to loss
            mse_loss_weights = mse_loss_weights.view(
                -1, 1, 1, 1
            )  # Shape: (batch_size, 1, 1, 1)
            loss = loss * mse_loss_weights  # Shape: (batch_size, C, H, W)

        # Mask the loss using any conditioning data
        conditioning_type = prepared_batch.get("conditioning_type")
        if conditioning_type == "mask" and apply_conditioning_mask:
            # Adapted from:
            # https://github.com/kohya-ss/sd-scripts/blob/main/library/custom_train_functions.py#L482
            mask_image = (
                prepared_batch["conditioning_pixel_values"]
                .to(dtype=loss.dtype, device=loss.device)[:, 0]
                .unsqueeze(1)
            )  # Shape: (batch_size, 1, H', W')
            mask_image = torch.nn.functional.interpolate(
                mask_image, size=loss.shape[2:], mode="area"
            )  # Resize to match loss spatial dimensions
            mask_image = mask_image / 2 + 0.5  # Normalize to [0,1]
            loss = loss * mask_image  # Element-wise multiplication

        # Reduce the loss by averaging over channels and spatial dimensions
        loss = loss.mean(dim=list(range(1, len(loss.shape))))  # Shape: (batch_size,)

        # Further reduce the loss by averaging over the batch dimension
        loss = loss.mean()  # Scalar value
        return loss

    def checkpoint_state_remove(self, output_dir, checkpoint):
        removing_checkpoint = os.path.join(output_dir, checkpoint)
        try:
            logger.debug(f"Removing {removing_checkpoint}")
            shutil.rmtree(removing_checkpoint, ignore_errors=True)
        except Exception as e:
            logger.error(f"Failed to remove directory: {removing_checkpoint}")
            print(e)

    def checkpoint_state_filter(self, output_dir, suffix=None):
        checkpoints_keep = []
        checkpoints = os.listdir(output_dir)
        for checkpoint in checkpoints:
            cs = checkpoint.split("-")
            base = cs[0]
            sfx = None
            if len(cs) < 2:
                continue
            elif len(cs) > 2:
                sfx = cs[2]

            if base != "checkpoint":
                continue
            if suffix and sfx and suffix != sfx:
                continue
            if (suffix and not sfx) or (sfx and not suffix):
                continue

            checkpoints_keep.append(checkpoint)

        return checkpoints_keep

    def checkpoint_state_cleanup(self, output_dir, limit, suffix=None):
        # remove any left over temp checkpoints (partially written, etc)
        checkpoints = self.checkpoint_state_filter(output_dir, "tmp")
        for removing_checkpoint in checkpoints:
            self.checkpoint_state_remove(output_dir, removing_checkpoint)

        # now remove normal checkpoints past the limit
        checkpoints = self.checkpoint_state_filter(output_dir, suffix)
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `limit - 1` checkpoints
        if len(checkpoints) < limit:
            return

        num_to_remove = len(checkpoints) - limit + 1
        removing_checkpoints = checkpoints[0:num_to_remove]
        logger.debug(
            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
        )
        logger.debug(f"removing checkpoints: {', '.join(removing_checkpoints)}")

        for removing_checkpoint in removing_checkpoints:
            self.checkpoint_state_remove(output_dir, removing_checkpoint)

    def checkpoint_state_save(self, output_dir, suffix=None):
        print("\n")

        save_path = os.path.join(
            output_dir,
            f"checkpoint-{self.state['global_step']}",
        )
        if suffix:
            save_path = f"{save_path}-{suffix}"

        # A temporary directory should be used so that saving state is an atomic operation.
        save_path_tmp = (
            f"{save_path}-tmp" if self.config.checkpointing_use_tempdir else save_path
        )

        # schedulefree optim needs the optimizer to be in eval mode to save the state (and then back to train after)
        self.mark_optimizer_eval()
        self.accelerator.save_state(save_path_tmp)
        self.mark_optimizer_train()
        for _, backend in StateTracker.get_data_backends().items():
            if "sampler" in backend:
                logger.debug(f"Backend: {backend}")
                backend["sampler"].save_state(
                    state_path=os.path.join(
                        save_path_tmp,
                        self.model_hooks.training_state_path,
                    ),
                )
        if save_path != save_path_tmp:
            os.rename(save_path_tmp, save_path)

    def checkpoint_state_latest(self, output_dir):
        # both checkpoint-[0-9]+ and checkpoint-[0-9]-rolling are candidates
        dirs = os.listdir(output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint") and not d.endswith("tmp")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        return dirs[-1] if len(dirs) > 0 else None

    def train(self):
        self.init_trackers()
        self._train_initial_msg()
        self.mark_optimizer_train()

        # Only show the progress bar once on each machine.
        show_progress_bar = True
        if not self.accelerator.is_local_main_process:
            show_progress_bar = False
        progress_bar = tqdm(
            range(0, self.config.max_train_steps),
            disable=not show_progress_bar,
            initial=self.state["global_step"],
            desc=f"Epoch {self.state['first_epoch']}/{self.config.num_train_epochs} Steps",
            ncols=125,
        )
        self.accelerator.wait_for_everyone()

        # Some values that are required to be initialised later.
        step = self.state["global_step"]
        training_luminance_values = []
        current_epoch_step = None
        self.bf, fetch_thread = None, None
        iterator_fn = random_dataloader_iterator
        num_epochs_to_track = self.config.num_train_epochs + 1
        if self.config.ignore_final_epochs:
            num_epochs_to_track += 1000000
        for epoch in range(self.state["first_epoch"], num_epochs_to_track):
            if (
                self.state["current_epoch"] > self.config.num_train_epochs + 1
                and not self.config.ignore_final_epochs
            ):
                # This might immediately end training, but that's useful for simply exporting the model.
                logger.info(
                    f"Training run is complete ({self.config.num_train_epochs}/{self.config.num_train_epochs} epochs, {self.state['global_step']}/{self.config.max_train_steps} steps)."
                )
                break
            self._epoch_rollover(epoch)
            self.model.get_trained_component().train()
            training_models = [self.model.get_trained_component()]
            if (
                "lora" in self.config.model_type
                and self.config.train_text_encoder
                and "standard" in self.config.lora_type.lower()
            ):
                for text_encoder in self.text_encoders:
                    if "t5" in str(text_encoder.__class__):
                        continue
                    text_encoder.train()
                    training_models.append(text_encoder)

            if current_epoch_step is not None:
                # We are resetting to the next epoch, if it is not none.
                current_epoch_step = 0
            else:
                # If it's None, we need to calculate the current epoch step based on the current global step.
                current_epoch_step = (
                    self.state["global_step"] % self.config.num_update_steps_per_epoch
                )
            train_backends = {}
            for backend_id, backend in StateTracker.get_data_backends().items():
                if (
                    StateTracker.backend_status(backend_id)
                    or "train_dataloader" not in backend
                ):
                    # Exclude exhausted backends.
                    logger.debug(
                        f"Excluding backend: {backend_id}, as it is exhausted? {StateTracker.backend_status(backend_id)} or not found {('train_dataloader' not in backend)}"
                    )
                    continue
                train_backends[backend_id] = backend["train_dataloader"]
            # Begin dataloader prefetch, if enabled.
            iterator_args = [train_backends]
            if self.config.dataloader_prefetch:
                iterator_args = []
                if self.bf is not None:
                    self.bf.stop_fetching()
                self.bf = BatchFetcher(
                    datasets=train_backends,
                    max_size=self.config.dataloader_prefetch_qlen,
                    step=step,
                )
                if fetch_thread is not None:
                    fetch_thread.join()
                fetch_thread = self.bf.start_fetching()
                iterator_fn = self.bf.next_response

            while True:
                self._exit_on_signal()
                step += 1
                prepared_batch = self.prepare_batch(iterator_fn(step, *iterator_args))
                training_logger.debug(f"Iterator: {iterator_fn}")
                if self.config.lr_scheduler == "cosine_with_restarts":
                    self.extra_lr_scheduler_kwargs["step"] = self.state["global_step"]

                if self.accelerator.is_main_process:
                    progress_bar.set_description(
                        f"Epoch {self.state['current_epoch']}/{self.config.num_train_epochs}, Steps"
                    )

                # If we receive a False from the enumerator, we know we reached the next epoch.
                if prepared_batch is False:
                    logger.debug(f"Reached the end of epoch {epoch}")
                    break

                if prepared_batch is None:
                    import traceback

                    raise ValueError(
                        f"Received a None batch, which is not a good thing. Traceback: {traceback.format_exc()}"
                    )

                # Add the current batch of training data's avg luminance to a list.
                if "batch_luminance" in prepared_batch:
                    training_luminance_values.append(prepared_batch["batch_luminance"])

                with self.accelerator.accumulate(training_models):
                    bsz = prepared_batch["latents"].shape[0]
                    training_logger.debug("Sending latent batch to GPU.")

                    if int(bsz) != int(self.config.train_batch_size):
                        logger.error(
                            f"Received {bsz} latents, but expected {self.config.train_batch_size}. Processing short batch."
                        )
                    training_logger.debug(f"Working on batch size: {bsz}")
                    # Prepare the data for the scatter plot
                    for timestep in prepared_batch["timesteps"].tolist():
                        self.timesteps_buffer.append(
                            (self.state["global_step"], timestep)
                        )

                    if "encoder_hidden_states" in prepared_batch:
                        encoder_hidden_states = prepared_batch["encoder_hidden_states"]
                        training_logger.debug(
                            f"Encoder hidden states: {encoder_hidden_states.shape}"
                        )

                    if "add_text_embeds" in prepared_batch:
                        add_text_embeds = prepared_batch["add_text_embeds"]
                        training_logger.debug(
                            f"Pooled embeds: {add_text_embeds.shape if add_text_embeds is not None else None}"
                        )

                    # Predict the noise residual and compute loss
                    is_regularisation_data = prepared_batch.get(
                        "is_regularisation_data", False
                    )
                    if is_regularisation_data and self.config.model_type == "lora":
                        training_logger.debug("Predicting parent model residual.")
                        with torch.no_grad():
                            if self.config.lora_type.lower() == "lycoris":
                                training_logger.debug(
                                    "Detaching LyCORIS adapter for parent prediction."
                                )
                                self.accelerator._lycoris_wrapped_network.set_multiplier(
                                    0.0
                                )
                            else:
                                self.model.get_trained_component().disable_lora()
                            prepared_batch["target"] = self.model_predict(
                                prepared_batch=prepared_batch,
                            )["model_prediction"]
                            if self.config.lora_type.lower() == "lycoris":
                                training_logger.debug(
                                    "Attaching LyCORIS adapter for student prediction."
                                )
                                self.accelerator._lycoris_wrapped_network.set_multiplier(
                                    1.0
                                )
                            else:
                                self.model.get_trained_component().enable_lora()

                    training_logger.debug("Predicting noise residual.")
                    model_pred = self.model_predict(
                        prepared_batch=prepared_batch,
                    )
                    loss = self.model.loss(
                        prepared_batch=prepared_batch,
                        model_output=model_pred,
                        apply_conditioning_mask=True,
                    )
                    loss, aux_loss_logs = self.model.auxiliary_loss(
                        prepared_batch=prepared_batch,
                        model_output=model_pred,
                        loss=loss,
                    )

                    parent_loss = None
                    if is_regularisation_data:
                        parent_loss = loss

                    # Gather the losses across all processes for logging (if using distributed training)
                    avg_loss = self.accelerator.gather(
                        loss.repeat(self.config.train_batch_size)
                    ).mean()
                    self.train_loss += (
                        avg_loss.item() / self.config.gradient_accumulation_steps
                    )
                    # Backpropagate
                    self.grad_norm = None
                    if not self.config.disable_accelerator:
                        training_logger.debug("Backwards pass.")
                        self.accelerator.backward(loss)

                        if (
                            self.config.optimizer != "adam_bfloat16"
                            and self.config.gradient_precision == "fp32"
                        ):
                            # After backward, convert gradients to fp32 for stable accumulation
                            for param in self.params_to_optimize:
                                if param.grad is not None:
                                    param.grad.data = param.grad.data.to(torch.float32)

                        self.grad_norm = self._max_grad_value()
                        if (
                            self.accelerator.sync_gradients
                            and self.config.optimizer
                            not in ["optimi-stableadamw", "prodigy"]
                            and self.config.max_grad_norm > 0
                        ):
                            # StableAdamW/Prodigy do not need clipping, similar to Adafactor.
                            if self.config.grad_clip_method == "norm":
                                self.grad_norm = self.accelerator.clip_grad_norm_(
                                    self._get_trainable_parameters(),
                                    self.config.max_grad_norm,
                                )
                            elif self.config.use_deepspeed_optimizer:
                                # deepspeed can only do norm clipping (internally)
                                pass
                            elif self.config.grad_clip_method == "value":
                                self.accelerator.clip_grad_value_(
                                    self._get_trainable_parameters(),
                                    self.config.max_grad_norm,
                                )
                            else:
                                raise ValueError(
                                    f"Unknown grad clip method: {self.config.grad_clip_method}. Supported methods: value, norm"
                                )
                        training_logger.debug("Stepping components forward.")
                        if self.config.optimizer_release_gradients:
                            step_offset = 0  # simpletuner indexes steps from 1.
                            should_not_release_gradients = (
                                step + step_offset
                            ) % self.config.gradient_accumulation_steps != 0
                            training_logger.debug(
                                f"step: {step}, should_not_release_gradients: {should_not_release_gradients}, self.config.optimizer_release_gradients: {self.config.optimizer_release_gradients}"
                            )
                            self.optimizer.optimizer_accumulation = (
                                should_not_release_gradients
                            )
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad(
                            set_to_none=self.config.set_grads_to_none
                        )

                # Checks if the accelerator has performed an optimization step behind the scenes
                wandb_logs = {}
                if self.accelerator.sync_gradients:
                    try:
                        if "prodigy" in self.config.optimizer:
                            self.lr_scheduler.step(**self.extra_lr_scheduler_kwargs)
                            self.lr = self.optimizer.param_groups[0]["d"]
                        elif self.config.is_lr_scheduler_disabled:
                            # hackjob method of retrieving LR from accelerated optims
                            self.lr = StateTracker.get_last_lr()
                        else:
                            self.lr_scheduler.step(**self.extra_lr_scheduler_kwargs)
                            self.lr = self.lr_scheduler.get_last_lr()[0]
                    except Exception as e:
                        logger.error(
                            f"Failed to get the last learning rate from the scheduler. Error: {e}"
                        )
                    wandb_logs.update(
                        {
                            "train_loss": self.train_loss,
                            "optimization_loss": loss,
                            "learning_rate": self.lr,
                            "epoch": epoch,
                        }
                    )
                    if parent_loss is not None:
                        wandb_logs["regularisation_loss"] = parent_loss
                    if aux_loss_logs is not None:
                        for key, value in aux_loss_logs.items():
                            wandb_logs[f"aux_loss/{key}"] = value
                    if self.grad_norm is not None:
                        if self.config.grad_clip_method == "norm":
                            wandb_logs["grad_norm"] = self.grad_norm
                        else:
                            wandb_logs["grad_absmax"] = self.grad_norm
                    if self.validation is not None and hasattr(
                        self.validation, "evaluation_result"
                    ):
                        eval_result = self.validation.get_eval_result()
                        if eval_result is not None and type(eval_result) == dict:
                            # add the dict to wandb_logs
                            self.validation.clear_eval_result()
                            wandb_logs.update(eval_result)

                    progress_bar.update(1)
                    self.state["global_step"] += 1
                    current_epoch_step += 1
                    StateTracker.set_global_step(self.state["global_step"])

                    ema_decay_value = "None (EMA not in use)"
                    if self.config.use_ema:
                        if self.ema_model is not None:
                            self.ema_model.step(
                                parameters=self._get_trainable_parameters(),
                                global_step=self.state["global_step"],
                            )
                            wandb_logs["ema_decay_value"] = self.ema_model.get_decay()
                            ema_decay_value = wandb_logs["ema_decay_value"]
                        self.accelerator.wait_for_everyone()

                    # Log scatter plot to wandb
                    if (
                        self.config.report_to == "wandb"
                        and self.accelerator.is_main_process
                    ):
                        # Prepare the data for the scatter plot
                        data = [
                            [iteration, timestep]
                            for iteration, timestep in self.timesteps_buffer
                        ]
                        table = wandb.Table(
                            data=data, columns=["global_step", "timestep"]
                        )
                        wandb_logs["timesteps_scatter"] = wandb.plot.scatter(
                            table,
                            "global_step",
                            "timestep",
                            title="Timestep distribution by step",
                        )

                    # Clear buffers
                    self.timesteps_buffer = []

                    # Average out the luminance values of each batch, so that we can store that in this step.
                    avg_training_data_luminance = sum(training_luminance_values) / len(
                        training_luminance_values
                    )
                    wandb_logs["train_luminance"] = avg_training_data_luminance

                    logger.debug(
                        f"Step {self.state['global_step']} of {self.config.max_train_steps}: loss {loss.item()}, lr {self.lr}, epoch {epoch}/{self.config.num_train_epochs}, ema_decay_value {ema_decay_value}, train_loss {self.train_loss}"
                    )
                    webhook_pending_msg = f"Step {self.state['global_step']} of {self.config.max_train_steps}: loss {round(loss.item(), 4)}, lr {self.lr}, epoch {epoch}/{self.config.num_train_epochs}, ema_decay_value {ema_decay_value}, train_loss {round(self.train_loss, 4)}"

                    # Reset some values for the next go.
                    training_luminance_values = []
                    self.train_loss = 0.0

                    if (
                        self.config.webhook_reporting_interval is not None
                        and self.state["global_step"]
                        % self.config.webhook_reporting_interval
                        == 0
                    ):
                        structured_data = {
                            "state": self.state,
                            "loss": round(self.train_loss, 4),
                            "parent_loss": parent_loss,
                            "learning_rate": self.lr,
                            "epoch": epoch,
                            "final_epoch": self.config.num_train_epochs,
                        }
                        self._send_webhook_raw(
                            structured_data=structured_data, message_type="train"
                        )

                    if (
                        self.config.checkpointing_steps
                        and self.state["global_step"] % self.config.checkpointing_steps
                        == 0
                    ):
                        self._send_webhook_msg(
                            message=f"Checkpoint: `{webhook_pending_msg}`",
                            message_level="info",
                        )
                        if (
                            self.accelerator.is_main_process
                            and self.config.checkpoints_total_limit is not None
                        ):
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            self.checkpoint_state_cleanup(
                                self.config.output_dir,
                                self.config.checkpoints_total_limit,
                            )

                        if (
                            self.accelerator.is_main_process
                            or self.config.use_deepspeed_optimizer
                        ):
                            self.checkpoint_state_save(self.config.output_dir)
                    elif (
                        self.config.checkpointing_rolling_steps
                        and self.state["global_step"]
                        % self.config.checkpointing_rolling_steps
                        == 0
                    ):
                        self._send_webhook_msg(
                            message=f"Checkpoint: `{webhook_pending_msg}`",
                            message_level="info",
                        )
                        if (
                            self.accelerator.is_main_process
                            and self.config.checkpoints_rolling_total_limit is not None
                        ):
                            # _before_ saving state, check if this save would set us over the `checkpoints_rolling_total_limit`
                            self.checkpoint_state_cleanup(
                                self.config.output_dir,
                                self.config.checkpoints_rolling_total_limit,
                                "rolling",
                            )

                        if (
                            self.accelerator.is_main_process
                            or self.config.use_deepspeed_optimizer
                        ):
                            self.checkpoint_state_save(
                                self.config.output_dir, "rolling"
                            )

                    if (
                        self.config.accelerator_cache_clear_interval is not None
                        and self.state["global_step"]
                        % self.config.accelerator_cache_clear_interval
                        == 0
                    ):
                        reclaim_memory()

                    # here we might run eval loss calculations.
                    if self.evaluation is not None and self.evaluation.would_evaluate(
                        self.state
                    ):
                        self.mark_optimizer_eval()
                        all_accumulated_losses = self.evaluation.execute_eval(
                            prepare_batch=self.prepare_batch,
                            model_predict=self.model_predict,
                            calculate_loss=self.model.loss,
                            get_prediction_target=self.get_prediction_target,
                            noise_scheduler=self._get_noise_schedule(),
                        )
                        if all_accumulated_losses:
                            tracker_table = self.evaluation.generate_tracker_table(
                                all_accumulated_losses=all_accumulated_losses
                            )
                            logger.debug(f"Tracking information: {tracker_table}")
                            wandb_logs.update(tracker_table)
                        self.mark_optimizer_train()

                    self.accelerator.log(
                        wandb_logs,
                        step=self.state["global_step"],
                    )

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": float(self.lr),
                }
                if aux_loss_logs is not None:
                    logs_to_print = {}
                    for key, value in aux_loss_logs.items():
                        logs_to_print[f"aux_loss/{key}"] = value
                    training_logger.debug(f"Aux loss: {logs_to_print}")
                if self.grad_norm is not None:
                    if self.config.grad_clip_method == "norm":
                        logs["grad_norm"] = float(self.grad_norm.clone().detach())
                    elif self.config.grad_clip_method == "value":
                        logs["grad_absmax"] = self.grad_norm

                progress_bar.set_postfix(**logs)

                if self.validation is not None:
                    if self.validation.would_validate():
                        self.mark_optimizer_eval()
                        self.enable_sageattention_inference()
                        self.disable_gradient_checkpointing()
                    self.validation.run_validations(
                        validation_type="intermediary", step=step
                    )
                    if self.validation.would_validate():
                        self.disable_sageattention_inference()
                        self.enable_gradient_checkpointing()
                        self.mark_optimizer_train()
                if (
                    self.config.push_to_hub
                    and self.config.push_checkpoints_to_hub
                    and self.state["global_step"] % self.config.checkpointing_steps == 0
                    and step % self.config.gradient_accumulation_steps == 0
                    and self.state["global_step"] > self.state["global_resume_step"]
                ):
                    if self.accelerator.is_main_process:
                        try:
                            self.hub_manager.upload_latest_checkpoint(
                                validation_images=(
                                    getattr(self.validation, "validation_images")
                                    if self.validation is not None
                                    else None
                                ),
                                webhook_handler=self.webhook_handler,
                            )
                        except Exception as e:
                            logger.error(
                                f"Error uploading to hub: {e}, continuing training."
                            )
                self.accelerator.wait_for_everyone()

                if self.state["global_step"] >= self.config.max_train_steps or (
                    epoch > self.config.num_train_epochs
                    and not self.config.ignore_final_epochs
                ):
                    logger.info(
                        f"Training has completed."
                        f"\n -> global_step = {self.state['global_step']}, max_train_steps = {self.config.max_train_steps}, epoch = {epoch}, num_train_epochs = {self.config.num_train_epochs}",
                    )
                    break
            if self.state["global_step"] >= self.config.max_train_steps or (
                epoch > self.config.num_train_epochs
                and not self.config.ignore_final_epochs
            ):
                logger.info(
                    f"Exiting training loop. Beginning model unwind at epoch {epoch}, step {self.state['global_step']}"
                )
                break

        # Create the pipeline using the trained modules and save it.
        self.accelerator.wait_for_everyone()
        validation_images = None
        if self.accelerator.is_main_process:
            self.mark_optimizer_eval()
            if self.validation is not None:
                self.enable_sageattention_inference()
                self.disable_gradient_checkpointing()
                validation_images = self.validation.run_validations(
                    validation_type="final",
                    step=self.state["global_step"],
                    force_evaluation=True,
                    skip_execution=True,
                ).validation_images
                # we don't have to do this but we will anyway.
                self.disable_sageattention_inference()
            if self.model.get_trained_component() is not None:
                self.model.model = unwrap_model(self.accelerator, self.model.model)
            if (
                "lora" in self.config.model_type
                and "standard" == self.config.lora_type.lower()
            ):
                lora_save_kwargs = {
                    "save_directory": self.config.output_dir,
                    f"{self.model.MODEL_TYPE.value}_lora_layers": get_peft_model_state_dict(
                        self.model.get_trained_component()
                    ),
                }

                if self.config.train_text_encoder:
                    if self.model.get_text_encoder(0) is not None:
                        self.text_encoder_1 = self.accelerator.unwrap_model(
                            self.model.get_text_encoder(0)
                        )
                        lora_save_kwargs["text_encoder_lora_layers"] = (
                            convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(self.text_encoder_1)
                            )
                        )
                    if self.model.get_text_encoder(1) is not None:
                        self.text_encoder_2 = self.accelerator.unwrap_model(
                            self.model.get_text_encoder(1)
                        )
                        lora_save_kwargs["text_encoder_2_lora_layers"] = (
                            convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(self.text_encoder_2)
                            )
                        )
                        if self.model.get_text_encoder(2) is not None:
                            self.text_encoder_3 = self.accelerator.unwrap_model(
                                self.model.get_text_encoder(2)
                            )
                else:
                    text_encoder_lora_layers = None
                    text_encoder_2_lora_layers = None

                from helpers.models.common import PipelineTypes

                self.model.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG].save_lora_weights(
                    **lora_save_kwargs,
                )
                del text_encoder_lora_layers
                del text_encoder_2_lora_layers
                reclaim_memory()
            elif (
                "lora" in self.config.model_type
                and "lycoris" == self.config.lora_type.lower()
            ):
                if (
                    self.accelerator.is_main_process
                    or self.config.use_deepspeed_optimizer
                ):
                    logger.info(
                        f"Saving final LyCORIS checkpoint to {self.config.output_dir}"
                    )
                    # Save final LyCORIS checkpoint.
                    if (
                        getattr(self.accelerator, "_lycoris_wrapped_network", None)
                        is not None
                    ):
                        from helpers.publishing.huggingface import (
                            LORA_SAFETENSORS_FILENAME,
                        )

                        self.accelerator._lycoris_wrapped_network.save_weights(
                            os.path.join(
                                self.config.output_dir, LORA_SAFETENSORS_FILENAME
                            ),
                            list(
                                self.accelerator._lycoris_wrapped_network.parameters()
                            )[0].dtype,
                            {
                                "lycoris_config": json.dumps(self.lycoris_config)
                            },  # metadata
                        )
                        shutil.copy2(
                            self.config.lycoris_config,
                            os.path.join(self.config.output_dir, "lycoris_config.json"),
                        )

            elif self.config.use_ema:
                if self.model.get_trained_component() is not None:
                    self.ema_model.copy_to(
                        self.model.get_trained_component().parameters()
                    )

            if self.config.model_type == "full":
                if self.config.save_text_encoder:
                    self.model.load_text_encoder()
                self.model.load_vae()
                pipeline = self.model.get_pipeline()
                pipeline.save_pretrained(
                    os.path.join(self.config.output_dir, "pipeline"),
                    safe_serialization=True,
                )
                logger.info(
                    f"Wrote pipeline to disk: {self.config.output_dir}/pipeline"
                )

            if self.config.push_to_hub and self.accelerator.is_main_process:
                self.hub_manager.upload_model(validation_images, self.webhook_handler)
        self.accelerator.end_training()
