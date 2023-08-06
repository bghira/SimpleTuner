#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 pseudoterminalX, CaptnSeraph and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging


# Quiet down, you.
logger = logging.getLogger('accelerate')
logger.setLevel(logging.WARNING)
os.environ.set("ACCELERATE_LOG_LEVEL", "WARNING")

import math
import os
import shutil
import random
import gc
import warnings
from pathlib import Path
from urllib.parse import urlparse
from helpers.aspect_bucket import BalancedBucketSampler
from helpers.dreambooth_dataset import DreamBoothDataset
from helpers.state_tracker import StateTracker
from helpers.sdxl_embeds import TextEmbeddingCache
from helpers.vae_cache import VAECache
from helpers.custom_schedule import enforce_zero_terminal_snr

logger = logging.getLogger()
filelock_logger = logging.getLogger("filelock")
connection_logger = logging.getLogger("urllib3.connectionpool")
training_logger = logging.getLogger("training-loop")

# More important logs.
logger.setLevel("DEBUG")
training_logger.setLevel("DEBUG")

# Less important logs.
filelock_logger.setLevel("WARNING")
connection_logger.setLevel("WARNING")

logger.info("Import accelerate")
import accelerate

logger.info("Import datasets")
import datasets

logger.info("Import numpy")
import numpy as np
import PIL

logger.info("Import pytorch")
import torch

logger.info("Import torch.nn")
import torch.nn as nn

logger.info("Import torch.nn.functional")
import torch.nn.functional as F

logger.info("Import torch.utils.checkpoint")
import torch.utils.checkpoint

logger.info("Import transformers")
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

logger.info("Import diffusers")
import diffusers

logger.info("Import pooplines.")
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
from torchvision.transforms import ToTensor

# Convert PIL Image to PyTorch Tensor
to_tensor = ToTensor()


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0.dev0")

logger = get_logger(__name__, log_level="DEBUG")

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}
WANDB_TABLE_COL_NAMES = ["step", "image", "text"]


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for general fine-tuning."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default="madebyollin/sdxl-vae-fp16-fix",
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--print_filenames",
        action="store_true",
        help=(
            "If any image files are stopping the process eg. due to corruption or truncation, this will help identify which is at fault."
        ),
    )
    parser.add_argument(
        "--debug_aspect_buckets",
        action="store_true",
        help="If set, will print excessive debugging for aspect bucket operations.",
    )
    parser.add_argument(
        "--debug_dataset_loader",
        action="store_true",
        help="If set, will print excessive debugging for data loader operations.",
    )
    parser.add_argument(
        "--use_original_images",
        type=str,
        default="false",
        help="When this option is provided, image cropping and processing will be disabled. It is a good idea to use this with caution, for training multiple aspect ratios.",
    )
    parser.add_argument(
        "--prepend_instance_prompt",
        action="store_true",
        help=(
            "When determining the captions from the filename, prepend the instance prompt as an enforced keyword."
        ),
    )
    parser.add_argument(
        "--seen_state_path",
        type=str,
        default=None,
        help="Where the JSON document containing the state of the seen images is stored. This helps ensure we do not repeat images too many times.",
    )
    parser.add_argument(
        "--state_path",
        type=str,
        default=None,
        help="A JSON document containing the current state of training, will be placed here.",
    )

    parser.add_argument(
        "--only_instance_prompt",
        action="store_true",
        help=("Use the instance prompt instead of the caption from filename."),
    )
    parser.add_argument(
        "--caption_dropout_interval",
        type=int,
        default=0,
        help=(
            "Every X steps, we will drop the caption from the input to assist in classifier-free guidance training."
            "When StabilityAI trained Stable Diffusion, a value of 10 was used."
            "Very high values might be useful to do some sort of enforced style training."
            "Default value is zero, maximum value is 100."
        ),
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    # TODO: Fix Huggingface dataset handling.
    parser.add_argument(
        "--image_column",
        type=str,
        default="input_image",
        help="The column of the dataset containing the original image on which edits where made.",
    )
    parser.add_argument(
        "--image_prompt_column",
        type=str,
        default="image_prompt",
        help="The column of the dataset containing the caption.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is sampled during training for inference.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_resolution",
        type=int,
        default=256,
        help="Square resolution images will be output at this resolution (256x256).",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run fine-tuning validation every X steps. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="simpletuner-sdxl",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution."
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=(
            "Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=(
            "Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--dadaptation_learning_rate",
        type=float,
        default=1.0,
        help="Learning rate for the discriminator adaptation. Default: 1.0",
    )
    parser.add_argument(
        "--use_dadapt_optimizer",
        action="store_true",
        help="Whether or not to use the discriminator adaptation optimizer.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--vae_dtype",
        type=str,
        default="bf16",
        required=False,
        help=(
            "The dtype of the VAE model. Choose between ['default', 'fp16', 'fp32', 'bf16']."
            "The default VAE dtype is float32, due to NaN issues in SDXL 1.0."
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_run_name",
        type=str,
        default="simpletuner-sdxl-test",
        help="The name of the run to track with the tracker.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()
    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    # Make one log on every process with the configuration for debugging.
    logger.info(accelerator.state, main_process_only=False)
    # if accelerator.is_local_main_process:
    #     datasets.utils.logging.set_verbosity_warning()
    #     transformers.utils.logging.set_verbosity_warning()
    #     diffusers.utils.logging.set_verbosity_info()
    # else:
    #     datasets.utils.logging.set_verbosity_error()
    #     transformers.utils.logging.set_verbosity_error()
    #     diffusers.utils.logging.set_verbosity_error()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG,
    )
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        force_upcast=False,
    )
    vae.enable_slicing()
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # Create EMA for the unet.
    if args.use_ema:
        logger.info("Using EMA. Creating EMAModel.")
        ema_unet = EMAModel(
            unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config
        )
        logger.info("EMA model creation complete.")

    if args.enable_xformers_memory_efficient_attention:
        logger.info("Enabling xformers memory-efficient attention.")
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DConditionModel
                )
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
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

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if hasattr(args, "train_text_encoder") and args.train_text_encoder:
            text_encoder_1.gradient_checkpointing_enable()
            text_encoder_2.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        logger.info("Enabling tf32 precision boost for NVIDIA devices.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        logging.warning(
            "If using an Ada or Ampere NVIDIA device, --allow_tf32 could add a bit more performance."
        )

    if args.scale_lr:
        logger.info(f"Scaling learning rate ({args.learning_rate}), due to --scale_lr")
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )
    logger.info(f"Learning rate: {args.learning_rate}")
    extra_optimizer_args = {}
    # Initialize the optimizer
    if args.use_8bit_adam:
        logger.info("Using 8bit AdamW optimizer.")
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif hasattr(args, "use_dadapt_optimizer") and args.use_dadapt_optimizer:
        logger.info("Using D-Adaptation optimizer.")
        try:
            from dadaptation import DAdaptAdam
        except ImportError:
            raise ImportError(
                "Please install the dadaptation library to make use of DaDapt optimizer."
                "You can do so by running `pip install dadaptation`"
            )

        optimizer_cls = DAdaptAdam
        if (
            hasattr(args, "dadaptation_learning_rate")
            and args.dadaptation_learning_rate is not None
        ):
            logger.debug(
                f"Overriding learning rate {args.learning_rate} with {args.dadaptation_learning_rate} for D-Adaptation optimizer."
            )
            args.learning_rate = args.dadaptation_learning_rate
            extra_optimizer_args["decouple"] = True
    else:
        logger.info("Using AdamW optimizer.")
        optimizer_cls = torch.optim.AdamW
    logger.info(
        f"Optimizer arguments, weight_decay={args.adam_weight_decay} eps={args.adam_epsilon}"
    )
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
        **extra_optimizer_args,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        logger.info(f"Loading Huggingface Hub dataset: {args.dataset_name}")
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )

    # 6. Get the column names for input/target.
    if hasattr(args, "dataset_name") and args.dataset_name is not None:
        raise ValueError('Huggingface datasets are not currently supported.')
        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset["train"].column_names
        dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
        if args.image_column is None:
            image_column = (
                dataset_columns[0] if dataset_columns is not None else column_names[0]
            )
        else:
            image_column = args.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
                )
        if args.image_prompt_column is None:
            image_prompt_column = (
                dataset_columns[1] if dataset_columns is not None else column_names[1]
            )
        else:
            image_prompt_column = args.image_prompt_column
            if image_prompt_column not in column_names:
                raise ValueError(
                    f"--image_prompt_column' value '{args.image_prompt_column}' needs to be one of: {', '.join(column_names)}"
                )
        if args.edited_image_column is None:
            edited_image_column = (
                dataset_columns[2] if dataset_columns is not None else column_names[2]
            )
        else:
            edited_image_column = args.edited_image_column
            if edited_image_column not in column_names:
                raise ValueError(
                    f"--edited_image_column' value '{args.edited_image_column}' needs to be one of: {', '.join(column_names)}"
                )
    else:
        logging.info(
            "Using SimpleTuner dataset layout, instead of huggingface --dataset layout."
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        logger.warning(f'Using "--fp16" with mixed precision training should be done with a custom VAE. Make sure you understand how this works.')
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        logger.warning(f'Using "--fp16" with mixed precision training should be done with a custom VAE. Make sure you understand how this works.')

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions, tokenizer):
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    # Load scheduler, tokenizer and models.
    tokenizer_1 = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )
    text_encoder_cls_1 = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_2 = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    betas_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        prediction_type="v_prediction",
        rescale_betas_zero_snr=True
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        prediction_type="v_prediction",
    )
    noise_scheduler.betas = betas_scheduler.betas
    text_encoder_1 = text_encoder_cls_1.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    text_encoder_2 = text_encoder_cls_2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
    )

    # We ALWAYS pre-compute the additional condition embeddings needed for SDXL
    # UNet as the model is already big and it uses two text encoders.
    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    tokenizers = [tokenizer_1, tokenizer_2]
    text_encoders = [text_encoder_1, text_encoder_2]

    # Freeze vae and text_encoders
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Get null conditioning
    def compute_null_conditioning():
        null_conditioning_list = []
        for a_tokenizer, a_text_encoder in zip(tokenizers, text_encoders):
            null_conditioning_list.append(
                a_text_encoder(
                    tokenize_captions([""], tokenizer=a_tokenizer).to(
                        accelerator.device
                    ),
                    output_hidden_states=True,
                ).hidden_states[-2]
            )
        return torch.concat(null_conditioning_list, dim=-1)

    null_conditioning = compute_null_conditioning()

    def compute_time_ids(width=None, height=None):
        if width is None:
            width = args.resolution
        if height is None:
            height = args.resolution
        crops_coords_top_left = (
            args.crops_coords_top_left_h,
            args.crops_coords_top_left_w,
        )
        original_size = target_size = (width, height)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=weight_dtype)
        return add_time_ids.to(accelerator.device).repeat(args.train_batch_size, 1)

    def collate_fn(examples):
        logger.debug(f"Running collate_fn")
        if not StateTracker.status_training():
            logger.debug(f"Not training, returning nothing from collate_fn")
            return
        logger.debug(f"Examples: {examples}")

        # Initialize the VAE Cache if it doesn't exist
        global vaecache
        if "vaecache" not in globals():
            vaecache = VAECache(vae, accelerator)

        pixel_values = []
        filepaths = []  # we will store the file paths here
        for example in examples:
            image_data = example["instance_images"]
            width = image_data.width
            height = image_data.height
            pixel_values.append(
                to_tensor(image_data).to(
                    memory_format=torch.contiguous_format, dtype=vae_dtype
                )
            )
            filepaths.append(example["instance_images_path"])  # store the file path

        # Compute the VAE embeddings for individual images
        latents = [vaecache.encode_image(pv, fp) for pv, fp in zip(pixel_values, filepaths)]
        pixel_values = torch.stack(latents)

        # Extract the captions from the examples.
        captions = [example["instance_prompt_text"] for example in examples]

        # Compute the embeddings using the captions.
        (
            prompt_embeds_all,
            add_text_embeds_all,
        ) = embed_cache.compute_embeddings_for_prompts(captions)
        prompt_embeds_all = torch.concat([prompt_embeds_all for _ in range(1)], dim=0)
        add_text_embeds_all = torch.concat(
            [add_text_embeds_all for _ in range(1)], dim=0
        )

        logger.debug(f"Returning collate_fn results.")
        return {
            "pixel_values": pixel_values,
            "prompt_embeds": prompt_embeds_all,
            "add_text_embeds": add_text_embeds_all,
            "add_time_ids": compute_time_ids(width, height),
        }

    # DataLoaders creation:
    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        accelerator=accelerator,
        size=args.resolution,
        center_crop=args.center_crop,
        print_names=args.print_filenames or False,
        use_original_images=bool(args.use_original_images),
        prepend_instance_prompt=args.prepend_instance_prompt or False,
        use_captions=not args.only_instance_prompt or False,
        caption_dropout_interval=args.caption_dropout_interval,
        use_precomputed_token_ids=True,
        debug_dataset_loader=args.debug_dataset_loader,
    )
    custom_balanced_sampler = BalancedBucketSampler(
        train_dataset.aspect_ratio_bucket_indices,
        batch_size=args.train_batch_size,
        seen_images_path=args.seen_state_path,
        state_path=args.state_path,
        debug_aspect_buckets=args.debug_aspect_buckets,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,  # The sampler handles shuffling
        sampler=custom_balanced_sampler,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )
    embed_cache = TextEmbeddingCache(
        text_encoders=text_encoders, tokenizers=tokenizers, accelerator=accelerator
    )
    logger.info(f"Pre-computing text embeds / updating cache.")
    embed_cache.precompute_embeddings_for_prompts(train_dataset.get_all_captions())

    if args.validation_prompt is not None:
        (
            validation_prompt_embeds,
            validation_pooled_embeds,
        ) = embed_cache.compute_embeddings_for_prompts([args.validation_prompt])
        (
            validation_negative_prompt_embeds,
            validation_negative_pooled_embeds,
        ) = embed_cache.compute_embeddings_for_prompts(["blurry, cropped, ugly"])
    # Grab GPU memory used:
    if accelerator.is_main_process:
        logger.info(
            f"Moving text encoders back to CPU, to save VRAM. Currently, we cannot completely unload the text encoder."
        )
    memory_before_unload = torch.cuda.memory_allocated() / 1024**3
    text_encoder_1.to("cpu")
    text_encoder_2.to("cpu")
    memory_after_unload = torch.cuda.memory_allocated() / 1024**3
    memory_saved = memory_after_unload - memory_before_unload
    gc.collect()
    torch.cuda.empty_cache()
    if accelerator.is_main_process:
        logger.info(
            f"After nuking text encoders from orbit, we freed {abs(round(memory_saved, 2))} GB of VRAM."
            "This number might be massively understated, because of how CUDA memory management works."
            "The real memories were the friends we trained a model on along the way."
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    logger.info(f'Loading noise scheduler...')
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    logger.info(f'Loading our accelerator...')
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        logger.info('Moving EMA model weights to accelerator...')
        ema_unet.to(accelerator.device)

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae_dtype = torch.float32
    if hasattr(args, "vae_dtype"):
        logger.info(
            f"Initialising VAE in {args.vae_dtype} precision, you may specify a different value if preferred: bf16, fp16, fp32, default"
        )
        # Let's use a case-switch for convenience: bf16, fp16, fp32, none/default
        if args.vae_dtype == "bf16":
            vae_dtype = torch.bfloat16
        elif args.vae_dtype == "fp16":
            vae_dtype = torch.float16
        elif args.vae_dtype == "fp32":
            vae_dtype = torch.float32
        elif args.vae_dtype == "none" or args.vae_dtype == "default":
            vae_dtype = torch.float32
    if args.pretrained_vae_model_name_or_path is not None:
        logger.debug(f"Initialising VAE with weight dtype {vae_dtype}")
        vae.to(accelerator.device, dtype=vae_dtype)
    else:
        logger.debug(f"Initialising VAE with custom dtype {vae_dtype}")
        vae.to(accelerator.device, dtype=vae_dtype)
    logger.info(f"Loaded VAE into VRAM.")
    if accelerator.is_main_process:
        logger.info(f"Pre-computing VAE latent space.")
        vaecache = VAECache(vae, accelerator)
        vaecache.process_directory(args.instance_data_dir)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_run_name, config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    resume_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            StateTracker.start_training()
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )
    else:
        StateTracker.start_training()

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        logger.debug(f"Starting into epoch: {epoch}")
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                if step + 1 == resume_step:
                    # We want to trigger the batch to be properly generated when we start.
                    if not StateTracker.status_training():
                        logging.info(
                            f"Starting training, as resume_step has been reached."
                        )
                        StateTracker.start_training()
                logger.warning("Skipping step.")
                continue

            if batch is None:
                logging.warning(f"Burning a None size batch.")
                continue
            with accelerator.accumulate(unet):
                training_logger.debug(f"Beginning another step.")
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                training_logger.debug("Moved pixels to accelerator.")
                latents = pixel_values
                # Sample noise that we'll add to the latents
                training_logger.debug(f"Sampling random noise")
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                training_logger.debug(f"Working on batch size: {bsz}")
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()
                training_logger.debug(f"Generated and converted timesteps to float64.")

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                training_logger.debug(
                    f"Generated noisy latent frame from latents and noise."
                )

                # SDXL additional inputs
                encoder_hidden_states = batch["prompt_embeds"]
                add_text_embeds = batch["add_text_embeds"]
                training_logger.debug(
                    f"Encoder hidden states: {encoder_hidden_states.shape}"
                )
                training_logger.debug(f"Added text embeds: {add_text_embeds.shape}")
                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                if args.pretrained_vae_model_name_or_path is not None:
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # Predict the noise residual and compute loss
                # training_logger.debug(f'add_text_embeds: {add_text_embeds.shape}, time_ids: {add_time_ids}')
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": batch["add_time_ids"],
                }
                training_logger.debug("Predicting noise residual.")
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                training_logger.debug(f"Calculating loss")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                training_logger.debug(f"Backwards pass.")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    training_logger.debug(f"Accelerator is syncing gradients")
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                training_logger.debug(f"Stepping components forward.")
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    training_logger.debug(f"Stepping EMA unet forward")
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            ### BEGIN: Perform validation every `validation_epochs` steps
            if accelerator.is_main_process:
                if global_step % args.validation_steps == 0 and global_step > 1:
                    pass
                    if (
                        args.validation_prompt is None
                        or args.num_validation_images is None
                        or args.num_validation_images <= 0
                    ):
                        logging.warning(
                            f"Not generating any validation images for this checkpoint. Live dangerously and prosper, pal!"
                        )
                        continue

                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                        f" {args.validation_prompt}."
                    )
                    # create pipeline
                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())

                    # The models need unwrapping because for compatibility in distributed training mode.
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        text_encoder=text_encoder_1,
                        text_encoder_2=text_encoder_2,
                        tokenizer=None,
                        tokenizer_2=None,
                        vae=vae,
                        revision=args.revision,
                        torch_dtype=weight_dtype,
                    )
                    pipeline.scheduler.config.prediction_type = "v_prediction"
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    # run inference
                    # Save validation images
                    val_save_dir = os.path.join(args.output_dir, "validation_images")
                    if not os.path.exists(val_save_dir):
                        os.makedirs(val_save_dir)

                    with torch.autocast(
                        str(accelerator.device).replace(":0", ""),
                        enabled=(
                            accelerator.mixed_precision == "fp16"
                            or accelerator.mixed_precision == "bf16"
                        ),
                    ):
                        validation_generator = torch.Generator(
                            device=accelerator.device
                        ).manual_seed(args.seed or 0)
                        edited_images = pipeline(
                            prompt_embeds=validation_prompt_embeds,
                            pooled_prompt_embeds=validation_pooled_embeds,
                            negative_prompt_embeds=validation_negative_prompt_embeds,
                            negative_pooled_prompt_embeds=validation_negative_pooled_embeds,
                            num_images_per_prompt=args.num_validation_images,
                            num_inference_steps=20,
                            guidance_scale=7,
                            generator=validation_generator,
                            height=args.validation_resolution,
                            width=args.validation_resolution,
                        ).images
                        val_img_idx = 0
                        for a_val_img in edited_images:
                            a_val_img.save(
                                os.path.join(
                                    val_save_dir,
                                    f"step_{global_step}_val_img_{val_img_idx}.png",
                                )
                            )
                            val_img_idx += 1

                    for tracker in accelerator.trackers:
                        if tracker.name == "wandb":
                            wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
                            idx = 0
                            for edited_image in edited_images:
                                tracker.log(
                                    {f"image-{idx}": wandb.Image(edited_images[idx])}
                                )
                                idx += 1
                    if args.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_unet.restore(unet.parameters())

                    del pipeline
                    torch.cuda.empty_cache()
                ### END: Perform validation every `validation_epochs` steps

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            vae=vae,
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        if args.validation_prompt is not None:
            edited_images = []
            pipeline = pipeline.to(accelerator.device)
            with torch.autocast(str(accelerator.device).replace(":0", "")):
                for _ in range(args.num_validation_images):
                    edited_images.append(
                        pipeline(
                            args.validation_prompt,
                            image=original_image,
                            num_inference_steps=20,
                            image_guidance_scale=1.5,
                            guidance_scale=7,
                            generator=generator,
                        ).images[0]
                    )

            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
                    for edited_image in edited_images:
                        wandb_table.add_data(
                            wandb.Image(original_image),
                            wandb.Image(edited_image),
                            args.validation_prompt,
                        )
                    tracker.log({"test": wandb_table})

    accelerator.end_training()


if __name__ == "__main__":
    main()
