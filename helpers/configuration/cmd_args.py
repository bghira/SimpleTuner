from datetime import timedelta
from accelerate.utils import ProjectConfiguration
from accelerate import InitProcessGroupKwargs
import argparse
import os
from typing import Dict, List, Optional, Tuple
import random
import time
import json
import logging
import sys
import torch
from helpers.training import quantised_precision_levels
from helpers.training.optimizer_param import (
    is_optimizer_deprecated,
    is_optimizer_grad_fp32,
    map_deprecated_optimizer_parameter,
    optimizer_choices,
)
from helpers.models.all import (
    model_families,
    get_model_flavour_choices,
    get_all_model_flavours,
)

model_family_choices = list(model_families.keys())

logger = logging.getLogger("ArgsParser")
# Are we the primary process?
is_primary_process = True
if os.environ.get("RANK") is not None:
    if int(os.environ.get("RANK")) != 0:
        is_primary_process = False
logger.setLevel(
    os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO" if is_primary_process else "ERROR")
)

if torch.cuda.is_available():
    os.environ["NCCL_SOCKET_NTIMEO"] = "2000000"


def print_on_main_thread(message):
    if is_primary_process:
        print(message)


def info_log(message):
    if is_primary_process:
        logger.info(message)


def warning_log(message):
    if is_primary_process:
        logger.warning(message)


def error_log(message):
    if is_primary_process:
        logger.error(message)


def get_argument_parser():
    parser = argparse.ArgumentParser(
        description="The following SimpleTuner command-line options are available:",
        exit_on_error=False,
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help=(
            "SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0."
            " More details here: https://arxiv.org/abs/2303.09556."
        ),
    )
    parser.add_argument(
        "--use_soft_min_snr",
        action="store_true",
        help=(
            "If set, will use the soft min SNR calculation method. This method uses the sigma_data parameter."
            " If not provided, the method will raise an error."
        ),
    )
    parser.add_argument(
        "--soft_min_snr_sigma_data",
        default=None,
        type=float,
        help=(
            "The standard deviation of the data used in the soft min weighting method."
            " This is required when using the soft min SNR calculation method."
        ),
    )
    parser.add_argument(
        "--model_family",
        choices=model_family_choices,
        default=None,
        required=True,
        help=("The model family to train. This option is required."),
    )
    parser.add_argument(
        "--model_flavour",
        default=None,
        required=False,
        choices=get_all_model_flavours(),
        type=str,
        help=(
            "Certain models require designating a given flavour to reference configurations from."
            " The value for this depends on the model that is selected."
            f" Currently supported values:{get_model_flavour_choices()}"
        ),
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=[
            "full",
            "lora",
        ],
        default="full",
        help=(
            "The training type to use. 'full' will train the full model, while 'lora' will train the LoRA model."
            " LoRA is a smaller model that can be used for faster training."
        ),
    )
    parser.add_argument(
        "--hidream_use_load_balancing_loss",
        action="store_true",
        default=False,
        help=(
            "When set, will use the load balancing loss for HiDream training. This is an experimental implementation."
        ),
    )
    parser.add_argument(
        "--hidream_load_balancing_loss_weight",
        type=float,
        default=None,
        help=(
            "When set, will use augment the load balancing loss for HiDream training. This is an experimental implementation."
        ),
    )
    parser.add_argument(
        "--flux_lora_target",
        type=str,
        choices=[
            "mmdit",
            "context",
            "context+ffs",
            "all",
            "all+ffs",
            "ai-toolkit",
            "tiny",
            "nano",
        ],
        default="all",
        help=(
            "This option only applies to Standard LoRA, not Lycoris. Flux has single and joint attention blocks."
            " By default, all attention layers are trained, but not the feed-forward layers"
            " If 'mmdit' is provided, the text input layers will not be trained."
            " If 'context' is provided, then ONLY the text attention layers are trained"
            " If 'context+ffs' is provided, then text attention and text feed-forward layers are trained. This is somewhat similar to text-encoder-only training in earlier SD versions."
            " If 'all' is provided, all layers will be trained, minus feed-forward."
            " If 'all+ffs' is provided, all layers will be trained including feed-forward."
            " If 'ai-toolkit' is provided, all layers will be trained including feed-forward and norms (based on ostris/ai-toolkit)."
            " If 'tiny' is provided, only two layers will be trained."
            " If 'nano' is provided, only one layers will be trained."
        ),
    )
    parser.add_argument(
        "--flow_sigmoid_scale",
        type=float,
        default=1.0,
        help="Scale factor for sigmoid timestep sampling for flow-matching models.",
    )
    parser.add_argument(
        "--flux_fast_schedule",
        action="store_true",
        help=(
            "An experimental feature to train Flux.1S using a noise schedule closer to what it was trained with,"
            " which has improved results in short experiments. Thanks to @mhirki for the contribution."
        ),
    )
    parser.add_argument(
        "--flow_use_uniform_schedule",
        default=False,
        action="store_true",
        help=(
            "Whether or not to use a uniform schedule instead of sigmoid for flow-matching noise schedule."
            " Using uniform sampling may cause a bias toward dark images, and should be used with caution."
        ),
    )
    parser.add_argument(
        "--flow_use_beta_schedule",
        action="store_true",
        default=False,
        help=(
            "Whether or not to use a beta schedule instead of sigmoid for flow-matching."
            " The default values of alpha and beta approximate a sigmoid."
        ),
    )
    parser.add_argument(
        "--flow_beta_schedule_alpha",
        type=float,
        default=2.0,
        help=("The alpha value of the flow-matching beta schedule. Default is 2.0"),
    )
    parser.add_argument(
        "--flow_beta_schedule_beta",
        type=float,
        default=2.0,
        help=("The beta value of the flow-matching beta schedule. Default is 2.0"),
    )
    parser.add_argument(
        "--flow_schedule_shift",
        type=float,
        default=3,
        help=(
            "Shift the noise schedule. This is a value between 0 and ~4.0, where 0 disables the timestep-dependent shift,"
            " and anything greater than 0 will shift the timestep sampling accordingly. Sana and SD3 were trained with"
            " a shift value of 3. This value can change how contrast/brightness are learnt by the model, and whether fine"
            " details are ignored or accentuated. A higher value will focus more on large compositional features,"
            " and a lower value will focus on the high frequency fine details."
        ),
    )
    parser.add_argument(
        "--flow_schedule_auto_shift",
        action="store_true",
        default=False,
        help=(
            "Shift the noise schedule depending on image resolution. The shift value calculation is taken from the official"
            " Flux inference code. Shift value is math.exp(1.15) = 3.1581 for a pixel count of 1024px * 1024px. The shift"
            " value grows exponentially with higher pixel counts. It is a good idea to train on a mix of different resolutions"
            " when this option is enabled. You may need to lower your learning rate with this enabled."
        ),
    )
    parser.add_argument(
        "--flux_guidance_mode",
        type=str,
        choices=["constant", "random-range"],
        default="constant",
        help=(
            "Flux has a 'guidance' value used during training time that reflects the CFG range of your training samples."
            " The default mode 'constant' will use a single value for every sample."
            " The mode 'random-range' will randomly select a value from the range of the CFG for each sample."
            " Set the range using --flux_guidance_min and --flux_guidance_max."
        ),
    )
    parser.add_argument(
        "--flux_guidance_value",
        type=float,
        default=1.0,
        help=(
            "When using --flux_guidance_mode=constant, this value will be used for every input sample."
            " Using a value of 1.0 seems to preserve the CFG distillation for the Dev model,"
            " and using any other value will result in the resulting LoRA requiring CFG at inference time."
        ),
    )
    parser.add_argument(
        "--flux_guidance_min",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--flux_guidance_max",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--flux_attention_masked_training",
        action="store_true",
        default=False,
        help=(
            "Use attention masking while training flux. This can be a destructive operation,"
            " unless finetuning a model which was already trained with it."
        ),
    )
    parser.add_argument(
        "--ltx_train_mode",
        choices=["t2v", "i2v"],
        default="i2v",
        help=(
            "This value will be the default for all video datasets that do not have their own i2v settings defined."
            " By default, we enable i2v mode, but it can be switched to t2v for your convenience."
        ),
    )
    parser.add_argument(
        "--ltx_i2v_prob",
        type=float,
        default=0.1,
        help=(
            "Probability in [0,1] of applying i2v (image-to-video) style training. "
            "If random.random() < i2v_prob during training, partial or complete first-frame protection "
            "will be triggered (depending on --ltx_protect_first_frame). "
            "If set to 0.0, no i2v logic is applied (pure t2v). Default: 0.1 (from finetuners project)"
        ),
    )
    parser.add_argument(
        "--ltx_protect_first_frame",
        action="store_true",
        help=(
            "If specified, fully protect the first frame whenever i2v logic is triggered (see --ltx_i2v_prob). "
            "This means the first frame is never noised or denoised, effectively pinned to the original content."
        ),
    )
    parser.add_argument(
        "--ltx_partial_noise_fraction",
        type=float,
        default=0.05,
        help=(
            "Maximum fraction of noise to introduce into the first frame when i2v is triggered and "
            "the first frame is not fully protected. For instance, a value of 0.05 means the first frame "
            "can have up to 5 percent random noise mixed in, preserving 95 percent of the original content. "
            "Ignored if --ltx_protect_first_frame is set."
        ),
    )

    parser.add_argument(
        "--t5_padding",
        choices=["zero", "unmodified"],
        default="unmodified",
        help=(
            "The padding behaviour for Flux and SD3. 'zero' will pad the input with zeros."
            " The default is 'unmodified', which will not pad the input."
        ),
    )
    parser.add_argument(
        "--sd3_clip_uncond_behaviour",
        type=str,
        choices=["empty_string", "zero"],
        default="empty_string",
        help=(
            "SD3 can be trained using zeroed prompt embeds during unconditional dropout,"
            " or an encoded empty string may be used instead (the default). Changing this value may stabilise or"
            " destabilise training. The default is 'empty_string'."
        ),
    )
    parser.add_argument(
        "--sd3_t5_uncond_behaviour",
        type=str,
        choices=["empty_string", "zero"],
        default=None,
        help=(
            "Override the value of unconditional prompts from T5 embeds."
            " The default is to follow the value of --sd3_clip_uncond_behaviour."
        ),
    )
    parser.add_argument(
        "--lora_type",
        type=str.lower,
        choices=["standard", "lycoris"],
        default="standard",
        help=(
            "When training using --model_type=lora, you may specify a different type of LoRA to train here."
            " standard refers to training a vanilla LoRA via PEFT, lycoris refers to training with KohakuBlueleaf's library of the same name."
        ),
    )
    parser.add_argument(
        "--lora_init_type",
        type=str,
        choices=["default", "gaussian", "loftq", "olora", "pissa"],
        default="default",
        help=(
            "The initialization type for the LoRA model. 'default' will use Microsoft's initialization method,"
            " 'gaussian' will use a Gaussian scaled distribution, and 'loftq' will use LoftQ initialization."
            " In short experiments, 'default' produced accurate results earlier in training, 'gaussian' had slightly more"
            " creative outputs, and LoftQ produces an entirely different result with worse quality at first, taking"
            " potentially longer to converge than the other methods."
        ),
    )
    parser.add_argument(
        "--init_lora",
        type=str,
        default=None,
        help="Specify an existing LoRA or LyCORIS safetensors file to initialize the adapter and continue training, if a full checkpoint is not available.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        required=False,
        default=None,
        help=(
            "The alpha value for the LoRA model. This is the learning rate for the LoRA update matrices."
        ),
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help=(
            "LoRA dropout randomly ignores neurons during training. This can help prevent overfitting."
        ),
    )
    parser.add_argument(
        "--lycoris_config",
        type=str,
        default="config/lycoris_config.json",
        help=("The location for the JSON file of the Lycoris configuration."),
    )
    parser.add_argument(
        "--init_lokr_norm",
        type=float,
        required=False,
        default=None,
        help=(
            "Setting this turns on perturbed normal initialization of the LyCORIS LoKr PEFT layers. A good value is between 1e-4 and 1e-2."
        ),
    )
    parser.add_argument(
        "--controlnet",
        action="store_true",
        default=False,
        help=(
            "If set, ControlNet style training will be used, where a conditioning input image is required alongside the training data."
        ),
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        action="store_true",
        default=None,
        help=(
            "When provided alongside --controlnet, this will specify ControlNet model weights to preload from the hub."
        ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help=(
            "Path to pretrained model or model identifier from huggingface.co/models."
            " Some model architectures support loading single-file .safetensors directly."
            " Note that when using single-file safetensors, the tokeniser and noise schedule configs"
            " will be used from the vanilla upstream Huggingface repository, which requires"
            " network access. If you are training on a machine without network access, you should"
            " pre-download the entire Huggingface model repository instead of using single-file loader."
        ),
    )
    parser.add_argument(
        "--pretrained_transformer_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained transformer model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_transformer_subfolder",
        type=str,
        default="transformer",
        help="The subfolder to load the transformer model from. Use 'none' for a flat directory.",
    )
    parser.add_argument(
        "--pretrained_unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_unet_subfolder",
        type=str,
        default="unet",
        help="The subfolder to load the unet model from. Use 'none' for a flat directory.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default="madebyollin/sdxl-vae-fp16-fix",
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--pretrained_t5_model_name_or_path",
        type=str,
        default=None,
        help=(
            "T5-XXL is a huge model, and starting from many different models will download a separate one each time."
            " This option allows you to specify a specific location to retrieve T5-XXL v1.1 from, so that it only downloads once.."
        ),
    )

    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        choices=["epsilon", "v_prediction", "sample", "flow_matching"],
        help=(
            "For models which support it, you can supply this value to override the prediction type. Choose between ['epsilon', 'v_prediction', 'sample', 'flow_matching']."
            " This may be needed for some SDXL derivatives that are trained using v_prediction or flow_matching."
        ),
    )
    parser.add_argument(
        "--snr_weight",
        type=float,
        default=1.0,
        help=(
            "When training a model using `--prediction_type=sample`, one can supply an SNR weight value to augment the loss with."
            " If a value of 0.5 is provided here, the loss is taken half from the SNR and half from the MSE."
        ),
    )
    parser.add_argument(
        "--training_scheduler_timestep_spacing",
        type=str,
        default="trailing",
        choices=["leading", "linspace", "trailing"],
        help=(
            "(SDXL Only) Spacing timesteps can fundamentally alter the course of history. Er, I mean, your model weights."
            " For all training, including epsilon, it would seem that 'trailing' is the right choice. SD 2.x always uses 'trailing',"
            " but SDXL may do better in its default state when using 'leading'."
        ),
    )
    parser.add_argument(
        "--inference_scheduler_timestep_spacing",
        type=str,
        default="trailing",
        choices=["leading", "linspace", "trailing"],
        help=(
            "(SDXL Only) The Bytedance paper on zero terminal SNR recommends inference using 'trailing'. SD 2.x always uses 'trailing',"
            " but SDXL may do better in its default state when using 'leading'."
        ),
    )
    parser.add_argument(
        "--refiner_training",
        action="store_true",
        default=False,
        help=(
            "When training or adapting a model into a mixture-of-experts 2nd stage / refiner model, this option should be set."
            " This will slice the timestep schedule defined by --refiner_training_strength proportion value (default 0.2)"
        ),
    )
    parser.add_argument(
        "--refiner_training_invert_schedule",
        action="store_true",
        default=False,
        help=(
            "While the refiner training strength is applied to the end of the schedule, this option will invert the result"
            " for training a **base** model, eg. the first model in a mixture-of-experts series."
            " A --refiner_training_strength of 0.35 will result in the refiner learning timesteps 349-0."
            " Setting --refiner_training_invert_schedule then would result in the base model learning timesteps 999-350."
        ),
    )
    parser.add_argument(
        "--refiner_training_strength",
        default=0.2,
        type=float,
        help=(
            "When training a refiner / 2nd stage mixture of experts model, the refiner training strength"
            " indicates how much of the *end* of the schedule it will be trained on. A value of 0.2 means"
            " timesteps 199-0 will be the focus of this model, and 0.3 would be 299-0 and so on."
            " The default value is 0.2, in line with the SDXL refiner pretraining."
        ),
    )
    parser.add_argument(
        "--timestep_bias_strategy",
        type=str,
        default="none",
        choices=["earlier", "later", "range", "none"],
        help=(
            "The timestep bias strategy, which may help direct the model toward learning low or frequency details."
            " Choices: ['earlier', 'later', 'none']."
            " The default is 'none', which means no bias is applied, and training proceeds normally."
            " The value of 'later' will prefer to generate samples for later timesteps."
        ),
    )
    parser.add_argument(
        "--timestep_bias_multiplier",
        type=float,
        default=1.0,
        help=(
            "The multiplier for the bias. Defaults to 1.0, which means no bias is applied."
            " A value of 2.0 will double the weight of the bias, and a value of 0.5 will halve it."
        ),
    )
    parser.add_argument(
        "--timestep_bias_begin",
        type=int,
        default=0,
        help=(
            "When using `--timestep_bias_strategy=range`, the beginning timestep to bias."
            " Defaults to zero, which equates to having no specific bias."
        ),
    )
    parser.add_argument(
        "--timestep_bias_end",
        type=int,
        default=1000,
        help=(
            "When using `--timestep_bias_strategy=range`, the final timestep to bias."
            " Defaults to 1000, which is the number of timesteps that SDXL Base and SD 2.x were trained on."
            " Just to throw a wrench into the works, Kolors was trained on 1100 timesteps."
        ),
    )
    parser.add_argument(
        "--timestep_bias_portion",
        type=float,
        default=0.25,
        help=(
            "The portion of timesteps to bias. Defaults to 0.25, which 25 percent of timesteps will be biased."
            " A value of 0.5 will bias one half of the timesteps. The value provided for `--timestep_bias_strategy` determines"
            " whether the biased portions are in the earlier or later timesteps."
        ),
    )
    parser.add_argument(
        "--disable_segmented_timestep_sampling",
        action="store_true",
        help=(
            "By default, the timestep schedule is divided into roughly `train_batch_size` number of segments, and then"
            " each of those are sampled from separately. This improves the selection distribution, but may not"
            " be desired in certain training scenarios, eg. when limiting the timestep selection range."
        ),
    )
    parser.add_argument(
        "--rescale_betas_zero_snr",
        action="store_true",
        help=(
            "If set, will rescale the betas to zero terminal SNR. This is recommended for training with v_prediction."
            " For epsilon, this might help with fine details, but will not result in contrast improvements."
        ),
    )
    parser.add_argument(
        "--vae_dtype",
        type=str,
        default="bf16",
        choices=["default", "fp16", "fp32", "bf16"],
        required=False,
        help=(
            "The dtype of the VAE model. Choose between ['default', 'fp16', 'fp32', 'bf16']."
            " The default VAE dtype is bfloat16, due to NaN issues in SDXL 1.0."
            " Using fp16 is not recommended."
        ),
    )
    parser.add_argument(
        "--vae_batch_size",
        type=int,
        default=4,
        help=(
            "When pre-caching latent vectors, this is the batch size to use. Decreasing this may help with VRAM issues,"
            " but if you are at that point of contention, it's possible that your GPU has too little RAM. Default: 4."
        ),
    )
    parser.add_argument(
        "--vae_enable_tiling",
        action="store_true",
        default=False,
        help=(
            "If set, will enable tiling for VAE caching. This is useful for very large images when VRAM is limited."
            " This may be required for 2048px VAE caching on 24G accelerators, in addition to reducing --vae_batch_size."
        ),
    )
    parser.add_argument(
        "--vae_enable_slicing",
        action="store_true",
        default=False,
        help=(
            "If set, will enable slicing for VAE caching. This is useful for video models."
        ),
    )
    parser.add_argument(
        "--vae_cache_scan_behaviour",
        type=str,
        choices=["recreate", "sync"],
        default="recreate",
        help=(
            "When a mismatched latent vector is detected, a scan will be initiated to locate inconsistencies and resolve them."
            " The default setting 'recreate' will delete any inconsistent cache entries and rebuild it."
            " Alternatively, 'sync' will update the bucket configuration so that the image is in a bucket that matches its latent size."
            " The recommended behaviour is to use the default value and allow the cache to be recreated."
        ),
    )
    parser.add_argument(
        "--vae_cache_ondemand",
        action="store_true",
        default=False,
        help=(
            "By default, will batch-encode images before training. For some situations, ondemand may be desired, but it greatly slows training and increases memory pressure."
        ),
    )
    parser.add_argument(
        "--compress_disk_cache",
        action="store_true",
        default=False,
        help=(
            "If set, will gzip-compress the disk cache for Pytorch files. This will save substantial disk space, but may slow down the training process."
        ),
    )
    parser.add_argument(
        "--aspect_bucket_disable_rebuild",
        action="store_true",
        default=False,
        help=(
            "When using a randomised aspect bucket list, the VAE and aspect cache are rebuilt on each epoch."
            " With a large and diverse enough dataset, rebuilding the aspect list may take a long time, and this may be undesirable."
            " This option will not override vae_cache_clear_each_epoch. If both options are provided, only the VAE cache will be rebuilt."
        ),
    )
    parser.add_argument(
        "--keep_vae_loaded",
        action="store_true",
        default=False,
        help="If set, will keep the VAE loaded in memory. This can reduce disk churn, but consumes VRAM during the forward pass.",
    )
    parser.add_argument(
        "--skip_file_discovery",
        type=str,
        default="",
        help=(
            "Comma-separated values of which stages to skip discovery for. Skipping any stage will speed up resumption,"
            " but will increase the risk of errors, as missing images or incorrectly bucketed images may not be caught."
            " 'vae' will skip the VAE cache process, 'aspect' will not build any aspect buckets, and 'text' will avoid text embed management."
            " Valid options: aspect, vae, text, metadata."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " at least bfloat16 precision."
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        required=False,
        help=(
            "Variant of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " at least bfloat16 precision."
        ),
    )
    parser.add_argument(
        "--preserve_data_backend_cache",
        action="store_true",
        default=False,
        help=(
            "For very large cloud storage buckets that will never change, enabling this option will prevent the trainer"
            " from scanning it at startup, by preserving the cache files that we generate. Be careful when using this,"
            " as, switching datasets can result in the preserved cache being used, which would be problematic."
            " Currently, cache is not stored in the dataset itself but rather, locally. This may change in a future release."
        ),
    )
    parser.add_argument(
        "--use_dora",
        action="store_true",
        default=False,
        help=(
            "If set, will use the DoRA-enhanced LoRA training. This is an experimental feature, may slow down training,"
            " and is not recommended for general use."
        ),
    )
    parser.add_argument(
        "--override_dataset_config",
        action="store_true",
        default=False,
        help=(
            "When provided, the dataset's config will not be checked against the live backend config."
            " This is useful if you want to simply update the behaviour of an existing dataset,"
            " but the recommendation is to not change the dataset configuration after caching has begun,"
            " as most options cannot be changed without unexpected behaviour later on. Additionally, it prevents"
            " accidentally loading an SDXL configuration on a SD 2.x model and vice versa."
        ),
    )
    parser.add_argument(
        "--cache_dir_text",
        type=str,
        default="cache",
        help=(
            "This is the path to a local directory that will contain your text embed cache."
        ),
    )
    parser.add_argument(
        "--cache_dir_vae",
        type=str,
        default="",
        help=(
            "This is the path to a local directory that will contain your VAE outputs."
            " Unlike the text embed cache, your VAE latents will be stored in the AWS data backend."
            " Each backend can have its own value, but if that is not provided, this will be the default value."
        ),
    )
    parser.add_argument(
        "--data_backend_config",
        type=str,
        default=None,
        help=(
            "The relative or fully-qualified path for your data backend config."
            " See multidatabackend.json.example for an example."
        ),
    )
    parser.add_argument(
        "--data_backend_sampling",
        type=str,
        choices=["uniform", "auto-weighting"],
        default="auto-weighting",
        help=(
            "When using multiple data backends, the sampling weighting can be set to 'uniform' or 'auto-weighting'."
            " The default value is 'auto-weighting', which will automatically adjust the sampling weights based on the"
            " number of images in each backend. 'uniform' will sample from each backend equally."
        ),
    )
    parser.add_argument(
        "--ignore_missing_files",
        action="store_true",
        help=(
            "This option will disable the check for files that have been deleted or removed from your data directory."
            " This would allow training on large datasets without keeping the associated images on disk, though it's"
            " not recommended and is not a supported feature. Use with caution, as it mostly exists for experimentation."
        ),
    )
    parser.add_argument(
        "--write_batch_size",
        type=int,
        default=128,
        help=(
            "When using certain storage backends, it is better to batch smaller writes rather than continuous dispatching."
            " In SimpleTuner, write batching is currently applied during VAE caching, when many small objects are written."
            " This mostly applies to S3, but some shared server filesystems may benefit as well, eg. Ceph. Default: 64."
        ),
    )
    parser.add_argument(
        "--read_batch_size",
        type=int,
        default=25,
        help=(
            "Used by the VAE cache to prefetch image data. This is the number of images to read ahead."
        ),
    )
    parser.add_argument(
        "--image_processing_batch_size",
        type=int,
        default=32,
        help=(
            "When resizing and cropping images, we do it in parallel using processes or threads."
            " This defines how many images will be read into the queue before they are processed."
        ),
    )
    parser.add_argument(
        "--enable_multiprocessing",
        default=False,
        action="store_true",
        help=(
            "If set, will use processes instead of threads during metadata caching operations."
            " For some systems, multiprocessing may be faster than threading, but will consume a lot more memory."
            " Use this option with caution, and monitor your system's memory usage."
        ),
    )
    parser.add_argument(
        "--max_workers",
        default=32,
        type=int,
        help=("How many active threads or processes to run during VAE caching."),
    )
    parser.add_argument(
        "--aws_max_pool_connections",
        type=int,
        default=128,
        help=(
            "When using AWS backends, the maximum number of connections to keep open to the S3 bucket at a single time."
            " This should be greater or equal to the max_workers and aspect bucket worker count values."
        ),
    )
    parser.add_argument(
        "--torch_num_threads",
        type=int,
        default=8,
        help=(
            "The number of threads to use for PyTorch operations. This is not the same as the number of workers."
            " Default: 8."
        ),
    )
    parser.add_argument(
        "--dataloader_prefetch",
        action="store_true",
        default=False,
        help=(
            "When provided, the dataloader will read-ahead and attempt to retrieve latents, text embeds, and other metadata"
            " ahead of the time when the batch is required, so that it can be immediately available."
        ),
    )
    parser.add_argument(
        "--dataloader_prefetch_qlen",
        type=int,
        default=10,
        help=("Set the number of prefetched batches."),
    )
    parser.add_argument(
        "--aspect_bucket_worker_count",
        type=int,
        default=12,
        help=(
            "The number of workers to use for aspect bucketing. This is a CPU-bound task, so the number of workers"
            " should be set to the number of CPU threads available. If you use an I/O bound backend, an even higher"
            " value may make sense. Default: 12."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--cache_clear_validation_prompts",
        action="store_true",
        help=(
            "When provided, any validation prompt entries in the text embed cache will be recreated."
            " This is useful if you've modified any of the existing prompts."
        ),
    )
    parser.add_argument(
        "--caption_strategy",
        type=str,
        default="filename",
        choices=["filename", "textfile", "instance_prompt", "parquet"],
        help=(
            "The default captioning strategy, 'filename', will use the filename as the caption, after stripping some characters like underscores."
            " The 'textfile' strategy will use the contents of a text file with the same name as the image."
            " The 'parquet' strategy requires a parquet file with the same name as the image, containing a 'caption' column."
        ),
    )
    parser.add_argument(
        "--parquet_caption_column",
        type=str,
        default=None,
        help=(
            "When using caption_strategy=parquet, this option will allow you to globally set the default caption field across all datasets"
            " that do not have an override set."
        ),
    )
    parser.add_argument(
        "--parquet_filename_column",
        type=str,
        default=None,
        help=(
            "When using caption_strategy=parquet, this option will allow you to globally set the default filename field across all datasets"
            " that do not have an override set."
        ),
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="This is unused. Filenames will be the captions instead.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="simpletuner-results",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--seed_for_each_device",
        type=bool,
        default=True,
        help=(
            "By default, a unique seed will be used for each GPU."
            " This is done deterministically, so that each GPU will receive the same seed across invocations."
            " If --seed_for_each_device=false is provided, then we will use the same seed across all GPUs,"
            " which will almost certainly result in the over-sampling of inputs on larger datasets."
        ),
    )
    parser.add_argument(
        "--framerate",
        default=None,
        help=(
            "By default, SimpleTuner will use a framerate of 25 for training and inference on video models."
            " You are on your own if you modify this value, but it is provided for your convenience."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution. If using --resolution_type=area, this float value represents megapixels."
        ),
    )
    parser.add_argument(
        "--resolution_type",
        type=str,
        default="pixel_area",
        choices=["pixel", "area", "pixel_area"],
        help=(
            "Resizing images maintains aspect ratio. This defines the resizing strategy."
            " If 'pixel', the images will be resized to the resolution by the shortest pixel edge, if the target size does not match the current size."
            " If 'area', the images will be resized so the pixel area is this many megapixels. Common rounded values such as `0.5` and `1.0` will be implicitly adjusted to their squared size equivalents."
            " If 'pixel_area', the pixel value (eg. 1024) will be converted to the proper value for 'area', and then calculate everything the same as 'area' would."
        ),
    )
    parser.add_argument(
        "--aspect_bucket_rounding",
        type=int,
        default=None,
        choices=range(1, 10),
        help=(
            "The number of decimal places to round the aspect ratio to. This is used to create buckets for aspect ratios."
            " For higher precision, ensure the image sizes remain compatible. Higher precision levels result in a"
            " greater number of buckets, which may not be a desirable outcome."
        ),
    )
    parser.add_argument(
        "--aspect_bucket_alignment",
        type=int,
        choices=[8, 64],
        default=64,
        help=(
            "When training diffusion models, the image sizes generally must align to a 64 pixel interval."
            " This is an exception when training models like DeepFloyd that use a base resolution of 64 pixels,"
            " as aligning to 64 pixels would result in a 1:1 or 2:1 aspect ratio, overly distorting images."
            " For DeepFloyd, this value is set to 32, but all other training defaults to 64. You may experiment"
            " with this value, but it is not recommended."
        ),
    )
    parser.add_argument(
        "--minimum_image_size",
        type=float,
        default=None,
        help=(
            "The minimum resolution for both sides of input images."
            " If --delete_unwanted_images is set, images smaller than this will be DELETED."
            " The default value is None, which means no minimum resolution is enforced."
            " If this option is not provided, it is possible that images will be destructively upsampled, harming model performance."
        ),
    )
    parser.add_argument(
        "--maximum_image_size",
        type=float,
        default=None,
        help=(
            "When cropping images that are excessively large, the entire scene context may be lost, eg. the crop might just"
            " end up being a portion of the background. To avoid this, a maximum image size may be provided, which will"
            " result in very-large images being downsampled before cropping them. This value uses --resolution_type to determine"
            " whether it is a pixel edge or megapixel value."
        ),
    )
    parser.add_argument(
        "--target_downsample_size",
        type=float,
        default=None,
        help=(
            "When using --maximum_image_size, very-large images exceeding that value will be downsampled to this target"
            " size before cropping. If --resolution_type=area and --maximum_image_size=4.0, --target_downsample_size=2.0"
            " would result in a 4 megapixel image being resized to 2 megapixel before cropping to 1 megapixel."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="(SD 2.x only) Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=None,
        required=False,
        help=(
            "The maximum sequence length of the tokenizer output, which defines the sequence length of text embed outputs."
            " If not set, will default to the tokenizer's max length. Unfortunately, this option only applies to T5 models, and"
            " due to the biases inducted by sequence length, changing it will result in potentially catastrophic model collapse."
            " This option causes poor training results. This is normal, and can be expected from changing this value."
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--ignore_final_epochs",
        action="store_true",
        default=False,
        help=(
            "When provided, the max epoch counter will not determine the end of the training run."
            " Instead, it will end when it hits --max_train_steps."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
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
        "--gradient_checkpointing_interval",
        default=None,
        type=int,
        help=(
            "Some models (Flux, SDXL, SD1.x/2.x, SD3) can have their gradient checkpointing limited to every nth block."
            " This can speed up training but will use more memory with larger intervals."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4e-7,
        help=(
            "Initial learning rate (after the potential warmup period) to use."
            " When using a cosine or sine schedule, --learning_rate defines the maximum learning rate."
        ),
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=None,
        help="Learning rate for the text encoder. If not provided, the value of --learning_rate will be used.",
    )
    parser.add_argument(
        "--lr_scale",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scale_sqrt",
        action="store_true",
        default=False,
        help="If using --lr-scale, use the square root of (number of GPUs * gradient accumulation steps * batch size).",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="sine",
        choices=[
            "linear",
            "sine",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        help=("The scheduler type to use. Default: sine"),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=0.8,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--distillation_method",
        default=None,
        choices=["dcm"],
        help=(
            "The distillation method to use. Currently, only 'dcm' is supported via LoRA."
            " This will apply the selected distillation method to the model."
        ),
    )
    parser.add_argument(
        "--distillation_config",
        default=None,
        type=str,
        help=(
            "The config for your selected distillation method."
            " If passing it via config.json, simply provide the JSON object directly."
        ),
    )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use EMA (exponential moving average) model. Works with LoRA, Lycoris, and full training.",
    )
    parser.add_argument(
        "--ema_device",
        choices=["cpu", "accelerator"],
        default="cpu",
        help=(
            "The device to use for the EMA model. If set to 'accelerator', the EMA model will be placed on the accelerator."
            " This provides the fastest EMA update times, but is not ultimately necessary for EMA to function."
        ),
    )
    parser.add_argument(
        "--ema_validation",
        choices=["none", "ema_only", "comparison"],
        default="comparison",
        help=(
            "When 'none' is set, no EMA validation will be done."
            " When using 'ema_only', the validations will rely mostly on the EMA weights."
            " When using 'comparison' (default) mode, the validations will first run on the checkpoint before also running for"
            " the EMA weights. In comparison mode, the resulting images will be provided side-by-side."
        ),
    )
    parser.add_argument(
        "--ema_cpu_only",
        action="store_true",
        default=False,
        help=(
            "When using EMA, the shadow model is moved to the accelerator before we update its parameters."
            " When provided, this option will disable the moving of the EMA model to the accelerator."
            " This will save a lot of VRAM at the cost of a lot of time for updates. It is recommended to also supply"
            " --ema_update_interval to reduce the number of updates to eg. every 100 steps."
        ),
    )
    parser.add_argument(
        "--ema_foreach_disable",
        action="store_true",
        default=True,
        help=(
            "By default, we use torch._foreach functions for updating the shadow parameters, which should be fast."
            " When provided, this option will disable the foreach methods and use vanilla EMA updates."
        ),
    )
    parser.add_argument(
        "--ema_update_interval",
        type=int,
        default=None,
        help=(
            "The number of optimization steps between EMA updates. If not provided, EMA network will update on every step."
        ),
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.995,
        help=(
            "The closer to 0.9999 this gets, the less updates will occur over time. Setting it to a lower value, such as 0.990,"
            " will allow greater influence of later updates."
        ),
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
        "--offload_during_startup",
        action="store_true",
        help=(
            "When set, text encoders, the VAE, or other models will be moved to and from the CPU as needed, which can slow"
            " down startup, but saves VRAM. This is useful for video models or high-resolution pre-caching of latent embeds."
        ),
    )
    parser.add_argument(
        "--offload_param_path",
        type=str,
        default=None,
        help=(
            "When using DeepSpeed ZeRo stage 2 or 3 with NVMe offload, this may be specified to provide a path for the offload."
        ),
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=optimizer_choices.keys(),
        required=True,
        default=None,
    )
    parser.add_argument(
        "--optimizer_config",
        type=str,
        default=None,
        help=(
            "When setting a given optimizer, this allows a comma-separated list of key-value pairs to be provided that will override the optimizer defaults."
            " For example, `--optimizer_config=decouple_lr=True,weight_decay=0.01`."
        ),
    )
    parser.add_argument(
        "--optimizer_cpu_offload_method",
        choices=["none"],  # , "torchao"],
        default="none",
        help=(
            "This option is a placeholder. In the future, it will allow for the selection of different CPU offload methods."
        ),
    )
    parser.add_argument(
        "--optimizer_offload_gradients",
        action="store_true",
        default=False,
        help=(
            "When creating a CPU-offloaded optimiser, the gradients can be offloaded to the CPU to save more memory."
        ),
    )
    parser.add_argument(
        "--fuse_optimizer",
        action="store_true",
        default=False,
        help=(
            "When creating a CPU-offloaded optimiser, the fused optimiser could be used to save on memory, while running slightly slower."
        ),
    )
    parser.add_argument(
        "--optimizer_beta1",
        type=float,
        default=None,
        help="The value to use for the first beta value in the optimiser, which is used for the first moment estimate. A range of 0.8-0.9 is common.",
    )
    parser.add_argument(
        "--optimizer_beta2",
        type=float,
        default=None,
        help="The value to use for the second beta value in the optimiser, which is used for the second moment estimate. A range of 0.999-0.9999 is common.",
    )
    parser.add_argument(
        "--optimizer_release_gradients",
        action="store_true",
        help=(
            "When using Optimi optimizers, this option will release the gradients after the optimizer step."
            " This can save memory, but may slow down training. With Quanto, there may be no benefit."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and other optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and other optimizers.",
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
        "--prodigy_steps",
        type=int,
        default=None,
        help=(
            "When training with Prodigy, this defines how many steps it should be adjusting its learning rate for."
            " It seems to be that Diffusion models benefit from a capping off of the adjustments after 25 percent"
            " of the training run (dependent on batch size, repeats, and epochs)."
            " It this value is not supplied, it will be calculated at 25 percent of your training steps."
        ),
    )
    parser.add_argument(
        "--max_grad_norm",
        default=2.0,
        type=float,
        help=(
            "Clipping the max gradient norm can help prevent exploding gradients, but"
            " may also harm training by introducing artifacts or making it hard to train artifacts away."
        ),
    )
    parser.add_argument(
        "--grad_clip_method",
        default="value",
        choices=["value", "norm"],
        help=(
            "When applying --max_grad_norm, the method to use for clipping the gradients."
            " The previous default option 'norm' will scale ALL gradient values when any outliers in the gradient are encountered, which can reduce training precision."
            " The new default option 'value' will clip individual gradient values using this value as a maximum, which may preserve precision while avoiding outliers, enhancing convergence."
            " In simple terms, the default will help the model learn faster without blowing up (SD3.5 Medium was the main test model)."
            " Use 'norm' to return to the old behaviour."
        ),
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--push_checkpoints_to_hub",
        action="store_true",
        help=(
            "When set along with --push_to_hub, all intermediary checkpoints will be pushed to the hub as if they were a final checkpoint."
        ),
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--model_card_note",
        type=str,
        default=None,
        help=(
            "Add a string to the top of your model card to provide users with some additional context."
        ),
    )
    parser.add_argument(
        "--model_card_safe_for_work",
        action="store_true",
        default=False,
        help=(
            "Hugging Face Hub requires a warning to be added to models that may generate NSFW content."
            " This is done by default in SimpleTuner for safety purposes, but can be disabled with this option."
            " Additionally, removing the not-for-all-audiences tag from the README.md in the repo will also disable this warning"
            " on previously-uploaded models."
        ),
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
        "--disable_benchmark",
        action="store_true",
        default=False,
        help=(
            "By default, the model will be benchmarked on the first batch of the first epoch."
            " This can be disabled with this option."
        ),
    )
    parser.add_argument(
        "--evaluation_type",
        type=str,
        default=None,
        choices=["clip", "none"],
        help=(
            "Validations must be enabled for model evaluation to function. The default is to use no evaluator,"
            " and 'clip' will use a CLIP model to evaluate the resulting model's performance during validations."
        ),
    )
    parser.add_argument(
        "--eval_dataset_pooling",
        action="store_true",
        default=False,
        help=(
            "When provided, only the pooled evaluation results will be returned in a single chart from all eval sets."
            " Without this option, all eval sets will have separate charts."
        ),
    )
    parser.add_argument(
        "--pretrained_evaluation_model_name_or_path",
        type=str,
        default="openai/clip-vit-large-patch14-336",
        help=(
            "Optionally provide a custom model to use for ViT evaluations."
            " The default is currently clip-vit-large-patch14-336, allowing for lower patch sizes (greater accuracy)"
            " and an input resolution of 336x336."
        ),
    )
    parser.add_argument(
        "--validation_on_startup",
        action="store_true",
        default=False,
        help=(
            "When training begins, the starting model will have validation prompts run through it, for later comparison."
        ),
    )
    parser.add_argument(
        "--validation_seed_source",
        type=str,
        default="cpu",
        choices=["gpu", "cpu"],
        help=(
            "Some systems may benefit from using CPU-based seeds for reproducibility. On other systems, this may cause a TypeError."
            " Setting this option to 'cpu' may cause validation errors. If so, please set SIMPLETUNER_LOG_LEVEL=DEBUG"
            " and submit debug.log to a new Github issue report."
        ),
    )
    parser.add_argument(
        "--validation_lycoris_strength",
        type=float,
        default=1.0,
        help=(
            "When inferencing for validations, the Lycoris model will by default be run at its training strength, 1.0."
            " However, this value can be increased to a value of around 1.3 or 1.5 to get a stronger effect from the model."
        ),
    )
    parser.add_argument(
        "--validation_torch_compile",
        action="store_true",
        default=False,
        help=(
            "Supply `--validation_torch_compile=true` to enable the use of torch.compile() on the validation pipeline."
            " For some setups, torch.compile() may error out. This is dependent on PyTorch version, phase of the moon,"
            " but if it works, you should leave it enabled for a great speed-up."
        ),
    )
    parser.add_argument(
        "--validation_torch_compile_mode",
        type=str,
        default="max-autotune",
        choices=["max-autotune", "reduce-overhead", "default"],
        help=(
            "PyTorch provides different modes for the Torch Inductor when compiling graphs. max-autotune,"
            " the default mode, provides the most benefit."
        ),
    )
    parser.add_argument(
        "--validation_guidance_skip_layers",
        type=str,
        default=None,
        help=(
            "StabilityAI recommends a value of [7, 8, 9] for Stable Diffusion 3.5 Medium."
            " For Wan 2.1, a value of [9], [10], or, [9, 10] was found to work well."
        ),
    )
    parser.add_argument(
        "--validation_guidance_skip_layers_start",
        type=float,
        default=0.01,
        help=("StabilityAI recommends a value of 0.01 for SLG start."),
    )
    parser.add_argument(
        "--validation_guidance_skip_layers_stop",
        type=float,
        default=0.2,
        help=("StabilityAI recommends a value of 0.2 for SLG stop."),
    )
    parser.add_argument(
        "--validation_guidance_skip_scale",
        type=float,
        default=2.8,
        help=(
            "StabilityAI recommends a value of 2.8 for SLG guidance skip scaling."
            " When adding more layers, you must increase the scale, eg. adding one more layer requires doubling"
            " the value given."
        ),
    )
    parser.add_argument(
        "--sana_complex_human_instruction",
        type=str,
        # this is diverged from upstream SANA implementation.
        # while Gemma won't censor prompts with this configuration, it also won't enhance them.
        # therefore, it's recommended anything unsavoury to the Gemma encoder should already be in a suitable caption format like CogVLM would generate.
        # think: "a lot of flowery text that relates to the concept" rather than very simple and short tags/prompts.
        default=[
            "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
            "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
            "- Do not add detail that is not in the original prompt! Do not mention colors if the User Prompt does not!",
            "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
            "- If the prompt contains explicit material, do not remove this information from the output, which would damage the performance of downstream classifiers.",
            "Here are examples of how to transform or refine prompts:",
            "- User Prompt: A cat sleeping -> Enhanced: A cat sleeping peacefully, showcasing the joy of pet ownership. Cute floof kitty cat gatto.",
            "- User Prompt: A busy city street -> Enhanced: A bustling city street scene featuring a crowd of people.",
            "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
            "User Prompt: ",
        ],
        help=(
            "When generating embeds for Sana, a complex human instruction will be attached to your prompt by default."
            " This is required for the Gemma model to produce meaningful image caption embeds."
        ),
    )
    parser.add_argument(
        "--disable_tf32",
        action="store_true",
        help=(
            "Previous defaults were to disable TF32 on Ampere GPUs. This option is provided to explicitly disable TF32,"
            " after default configuration was updated to enable TF32 on Ampere GPUs."
        ),
    )
    parser.add_argument(
        "--validation_using_datasets",
        action="store_true",
        default=None,
        help=(
            "When set, validation will use images sampled randomly from each dataset for validation."
            " Be mindful of privacy issues when publishing training data to the internet."
        ),
    )
    parser.add_argument(
        "--webhook_config",
        type=str,
        default=None,
        help=(
            "The path to the webhook configuration file. This file should be a JSON file with the following format:"
            ' {"url": "https://your.webhook.url", "webhook_type": "discord"}}'
        ),
    )
    parser.add_argument(
        "--webhook_reporting_interval",
        type=int,
        default=None,
        help=(
            "When using 'raw' webhooks that receive structured data, you can specify a reporting interval here for"
            " training progress updates to be sent at. This does not impact 'discord' webhook types."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations,'
            ' or `"none"` to disable logging.'
        ),
    )
    parser.add_argument(
        "--tracker_run_name",
        type=str,
        default="simpletuner-testing",
        help="The name of the run to track with the tracker.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="simpletuner",
        help="The name of the project for WandB or Tensorboard.",
    )
    parser.add_argument(
        "--tracker_image_layout",
        choices=["gallery", "table"],
        default="gallery",
        help=(
            "When running validations with multiple images, you may want them all placed together in a table, row-wise."
            " Gallery mode, the default, will allow use of a slider to view the historical images easily."
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_prompt_library",
        action="store_true",
        help="If this is provided, the SimpleTuner prompt library will be used to generate multiple images.",
    )
    parser.add_argument(
        "--user_prompt_library",
        type=str,
        default=None,
        help="This should be a path to the JSON file containing your prompt library. See user_prompt_library.json.example.",
    )
    parser.add_argument(
        "--validation_negative_prompt",
        type=str,
        default="blurry, cropped, ugly",
        help=(
            "When validating images, a negative prompt may be used to guide the model away from certain features."
            " When this value is set to --validation_negative_prompt='', no negative guidance will be applied."
            " Default: blurry, cropped, ugly"
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_disable",
        action="store_true",
        help="Enable to completely disable the generation of validation images.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--eval_steps_interval",
        type=int,
        default=None,
        help=(
            "When set, the model will be evaluated every X steps. This is useful for"
            " monitoring the model's progress during training, but it requires an eval set"
            " configured in your dataloader."
        ),
    )
    parser.add_argument(
        "--eval_timesteps",
        type=int,
        default=28,
        help=(
            "Defines how many timesteps to sample during eval."
            " You can emulate inference by setting this to the value of --validation_num_inference_steps."
        ),
    )
    parser.add_argument(
        "--num_eval_images",
        type=int,
        default=4,
        help=(
            "If possible, this many eval images will be selected from each dataset."
            " This is used when training super-resolution models such as DeepFloyd Stage II,"
            " which will upscale input images from the training set during validation."
            " If using --eval_steps_interval, this will be the number of batches sampled"
            " for loss calculations."
        ),
    )
    parser.add_argument(
        "--eval_dataset_id",
        type=str,
        default=None,
        help=(
            "When provided, only this dataset's images will be used as the eval set, to keep"
            " the training and eval images split. This option only applies for img2img validations,"
            " not validation loss calculations."
        ),
    )
    parser.add_argument(
        "--validation_num_inference_steps",
        type=int,
        default=30,
        help=(
            "The default scheduler, DDIM, benefits from more steps. UniPC can do well with just 10-15."
            " For more speed during validations, reduce this value. For better quality, increase it."
            " For model distilation, you will likely want to keep this low."
        ),
    )
    parser.add_argument(
        "--validation_num_video_frames",
        type=int,
        default=None,
        help=(
            "When this is set, you can reduce the number of frames from the default model value (but not go beyond that)."
        ),
    )
    parser.add_argument(
        "--validation_resolution",
        type=str,
        default=256,
        help="Square resolution images will be output at this resolution (256x256).",
    )
    parser.add_argument(
        "--validation_noise_scheduler",
        type=str,
        choices=["ddim", "ddpm", "euler", "euler-a", "unipc", "dpm++"],
        default=None,
        help=(
            "When validating the model at inference time, a different scheduler may be chosen."
            " UniPC can offer better speed, and Euler A can put up with instabilities a bit better."
            " For zero-terminal SNR models, DDIM is the best choice. Choices: ['ddim', 'ddpm', 'euler', 'euler-a', 'unipc', 'dpm++'],"
            " Default: None (use the model default)"
        ),
    )
    parser.add_argument(
        "--validation_disable_unconditional",
        action="store_true",
        help=(
            "When set, the validation pipeline will not generate unconditional samples."
            " This is useful to speed up validations with a single prompt on slower systems, or if you are not"
            " interested in unconditional space generations."
        ),
    )
    parser.add_argument(
        "--enable_watermark",
        default=False,
        action="store_true",
        help=(
            "The SDXL 0.9 and 1.0 licenses both require a watermark be used to identify any images created to be shared."
            " Since the images created during validation typically are not shared, and we want the most accurate results,"
            " this watermarker is disabled by default. If you are sharing the validation images, it is up to you"
            " to ensure that you are complying with the license, whether that is through this watermarker, or another."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "no"],
        help=(
            "SimpleTuner only supports bf16 training. Bf16 requires PyTorch >="
            " 1.10. on an Nvidia Ampere or later GPU, and PyTorch 2.3 or newer for Apple Silicon."
            " Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
            " fp16 is offered as an experimental option, but is not recommended as it is less-tested and you will likely encounter errors."
        ),
    )
    parser.add_argument(
        "--gradient_precision",
        type=str,
        choices=["unmodified", "fp32"],
        default=None,
        help=(
            "One of the hallmark discoveries of the Llama 3.1 paper is numeric instability when calculating"
            " gradients in bf16 precision. The default behaviour when gradient accumulation steps are enabled"
            " is now to use fp32 gradients, which is slower, but provides more accurate updates."
        ),
    )
    parser.add_argument(
        "--quantize_via",
        type=str,
        choices=["cpu", "accelerator"],
        default="accelerator",
        help=(
            "When quantising the model, the quantisation process can be done on the CPU or the accelerator."
            " When done on the accelerator (default), slightly more VRAM is required, but the process completes in milliseconds."
            " When done on the CPU, the process may take upwards of 60 seconds, but can complete without OOM on 16G cards."
        ),
    )
    parser.add_argument(
        "--base_model_precision",
        type=str,
        default="no_change",
        choices=quantised_precision_levels,
        help=(
            "When training a LoRA, you might want to quantise the base model to a lower precision to save more VRAM."
            " The default value, 'no_change', does not quantise any weights."
            " Using 'fp4-bnb' or 'fp8-bnb' will require Bits n Bytes for quantisation (NVIDIA, maybe AMD)."
            " Using 'fp8-quanto' will require Quanto for quantisation (Apple Silicon, NVIDIA, AMD)."
        ),
    )
    parser.add_argument(
        "--quantize_activations",
        action="store_true",
        help=(
            "(EXPERIMENTAL) This option is currently unsupported, and exists solely for development purposes."
        ),
    )
    parser.add_argument(
        "--base_model_default_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help=(
            "Unlike --mixed_precision, this value applies specifically for the default weights of your quantised base model."
            " When quantised, not every parameter can or should be quantised down to the target precision."
            " By default, we use bf16 weights for the base model - but this can be changed to fp32 to enable"
            " the use of other optimizers than adamw_bf16. However, this uses marginally more memory,"
            " and may not be necessary for your use case."
        ),
    )
    for i in range(1, 5):
        parser.add_argument(
            f"--text_encoder_{i}_precision",
            type=str,
            default="no_change",
            choices=quantised_precision_levels,
            help=(
                f"When training a LoRA, you might want to quantise text encoder {i} to a lower precision to save more VRAM."
                " The default value is to follow base_model_precision (no_change)."
                " Using 'fp4-bnb' or 'fp8-bnb' will require Bits n Bytes for quantisation (NVIDIA, maybe AMD)."
                " Using 'fp8-quanto' will require Quanto for quantisation (Apple Silicon, NVIDIA, AMD)."
            ),
        )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--attention_mechanism",
        type=str,
        choices=[
            "diffusers",
            "xformers",
            "sageattention",
            "sageattention-int8-fp16-triton",
            "sageattention-int8-fp16-cuda",
            "sageattention-int8-fp8-cuda",
        ],
        default="diffusers",
        help=(
            "On NVIDIA CUDA devices, alternative flash attention implementations are offered, with the default being native pytorch SDPA."
            " SageAttention has multiple backends to select from."
            " The recommended value, 'sageattention', guesses what would be the 'best' option for SageAttention on your hardware"
            " (usually this is the int8-fp16-cuda backend). However, manually setting this value to int8-fp16-triton"
            " may provide better averages for per-step training and inference performance while the cuda backend"
            " may provide the highest maximum speed (with also a lower minimum speed). NOTE: SageAttention training quality"
            " has not been validated."
        ),
    )
    parser.add_argument(
        "--sageattention_usage",
        type=str,
        choices=["training", "inference", "training+inference"],
        default="inference",
        help=(
            "SageAttention breaks gradient tracking through the backward pass, leading to untrained QKV layers."
            " This can result in substantial problems for training, so it is recommended to use SageAttention only for inference (default behaviour)."
            " If you are confident in your training setup or do not wish to train QKV layers, you may use 'training' to enable SageAttention for training."
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=0.1,
        help="The scale of noise offset. Default: 0.1",
    )
    parser.add_argument(
        "--noise_offset_probability",
        type=float,
        default=0.25,
        help=(
            "When training with --offset_noise, the value of --noise_offset will only be applied probabilistically."
            " The default behaviour is for offset noise (if enabled) to be applied 25 percent of the time."
        ),
    )
    parser.add_argument(
        "--validation_guidance",
        type=float,
        default=7.5,
        help="CFG value for validation images. Default: 7.5",
    )
    parser.add_argument(
        "--validation_guidance_real",
        type=float,
        default=1.0,
        help="Use real CFG sampling for distilled models. Default: 1.0 (no CFG)",
    )
    parser.add_argument(
        "--validation_no_cfg_until_timestep",
        type=int,
        default=2,
        help="When using real CFG sampling for Flux validation images, skip doing CFG on these timesteps. Default: 2",
    )
    parser.add_argument(
        "--validation_guidance_rescale",
        type=float,
        default=0.0,
        help="CFG rescale value for validation images. Default: 0.0, max 1.0",
    )
    parser.add_argument(
        "--validation_randomize",
        action="store_true",
        default=False,
        help="If supplied, validations will be random, ignoring any seeds.",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=None,
        help=(
            "If not supplied, the value for --seed will be used."
            " If neither those nor --validation_randomize are supplied, a seed of zero is used."
        ),
    )
    parser.add_argument(
        "--fully_unload_text_encoder",
        action="store_true",
        help=(
            "If set, will fully unload the text_encoder from memory when not in use."
            " This currently has the side effect of crashing validations, but it is useful"
            " for initiating VAE caching on GPUs that would otherwise be too small."
        ),
    )
    parser.add_argument(
        "--freeze_encoder_before",
        type=int,
        default=12,
        help="When using 'before' strategy, we will freeze layers earlier than this.",
    )
    parser.add_argument(
        "--freeze_encoder_after",
        type=int,
        default=17,
        help="When using 'after' strategy, we will freeze layers later than this.",
    )
    parser.add_argument(
        "--freeze_encoder_strategy",
        type=str,
        default="after",
        help=(
            "When freezing the text_encoder, we can use the 'before', 'between', or 'after' strategy."
            " The 'between' strategy will freeze layers between those two values, leaving the outer layers unfrozen."
            " The default strategy is to freeze all layers from 17 up."
            " This can be helpful when fine-tuning Stable Diffusion 2.1 on a new style."
        ),
    )
    parser.add_argument(
        "--layer_freeze_strategy",
        type=str,
        choices=["none", "bitfit"],
        default="none",
        help=(
            "When freezing parameters, we can use the 'none' or 'bitfit' strategy."
            " The 'bitfit' strategy will freeze all weights, and leave bias in a trainable state."
            " The default strategy is to leave all parameters in a trainable state."
            " Freezing the weights can improve convergence for finetuning."
            " Using bitfit only moderately reduces VRAM consumption, but substantially reduces the count of trainable parameters."
        ),
    )
    parser.add_argument(
        "--unet_attention_slice",
        action="store_true",
        default=False,
        help=(
            "If set, will use attention slicing for the SDXL UNet. This is an experimental feature and is not recommended for general use."
            " SD 2.x makes use of attention slicing on Apple MPS platform to avoid a NDArray size crash, but SDXL does not"
            " seem to require attention slicing on MPS. If memory constrained, try enabling it anyway."
        ),
    )
    parser.add_argument(
        "--print_filenames",
        action="store_true",
        help=(
            "If any image files are stopping the process eg. due to corruption or truncation, this will help identify which is at fault."
        ),
    )
    parser.add_argument(
        "--print_sampler_statistics",
        action="store_true",
        help=(
            "If provided, will print statistics about the dataset sampler. This is useful for debugging."
            " The default behaviour is to not print sampler statistics."
        ),
    )
    parser.add_argument(
        "--metadata_update_interval",
        type=int,
        default=3600,
        help=(
            "When generating the aspect bucket indicies, we want to save it every X seconds."
            " The default is to save it every 1 hour, such that progress is not lost on clusters"
            " where runtime is limited to 6-hour increments (e.g. the JUWELS Supercomputer)."
            " The minimum value is 60 seconds."
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
        "--freeze_encoder",
        type=bool,
        default=True,
        help="Whether or not to freeze the text_encoder. The default is true.",
    )
    parser.add_argument(
        "--save_text_encoder",
        action="store_true",
        default=False,
        help=(
            "If set, will save the text_encoder after training."
            " This is useful if you're using --push_to_hub so that the final pipeline contains all necessary components to run."
        ),
    )
    parser.add_argument(
        "--text_encoder_limit",
        type=int,
        default=25,
        help=(
            "When training the text_encoder, we want to limit how long it trains for to avoid catastrophic loss."
        ),
    )
    parser.add_argument(
        "--prepend_instance_prompt",
        action="store_true",
        help=(
            "When determining the captions from the filename, prepend the instance prompt as an enforced keyword."
        ),
    )
    parser.add_argument(
        "--only_instance_prompt",
        action="store_true",
        help="Use the instance prompt instead of the caption from filename.",
    )
    parser.add_argument(
        "--data_aesthetic_score",
        type=float,
        default=7.0,
        help=(
            "Since currently we do not calculate aesthetic scores for data, we will statically set it to one value. This is only used by the SDXL Refiner."
        ),
    )
    parser.add_argument(
        "--sdxl_refiner_uses_full_range",
        action="store_true",
        default=False,
        help=(
            "If set, the SDXL Refiner will use the full range of the model, rather than the design value of 20 percent."
            " This is useful for training models that will be used for inference from end-to-end of the noise schedule."
            " You may use this for example, to turn the SDXL refiner into a full text-to-image model."
        ),
    )
    parser.add_argument(
        "--caption_dropout_probability",
        type=float,
        default=None,
        help=(
            "Caption dropout will randomly drop captions and, for SDXL, size conditioning inputs based on this probability."
            " When set to a value of 0.1, it will drop approximately 10 percent of the inputs."
            " Maximum recommended value is probably less than 0.5, or 50 percent of the inputs. Maximum technical value is 1.0."
            " The default is to use zero caption dropout, though for better generalisation, a value of 0.1 is recommended."
        ),
    )
    parser.add_argument(
        "--delete_unwanted_images",
        action="store_true",
        help=(
            "If set, will delete images that are not of a minimum size to save on disk space for large training runs."
            " Default behaviour: Unset, remove images from bucket only."
        ),
    )
    parser.add_argument(
        "--delete_problematic_images",
        action="store_true",
        help=(
            "If set, any images that error out during load will be removed from the underlying storage medium."
            " This is useful to prevent repeatedly attempting to cache bad files on a cloud bucket."
        ),
    )
    parser.add_argument(
        "--disable_bucket_pruning",
        action="store_true",
        help=(
            "When training on very small datasets, you might not care that the batch sizes will outpace your image count."
            " Setting this option will prevent SimpleTuner from deleting your bucket lists that do not meet"
            " the minimum image count requirements. Use at your own risk, it may end up throwing off your statistics or epoch tracking."
        ),
    )
    parser.add_argument(
        "--offset_noise",
        action="store_true",
        default=False,
        help=(
            "Fine-tuning against a modified noise"
            " See: https://www.crosslabs.org//blog/diffusion-with-offset-noise for more information."
        ),
    )
    parser.add_argument(
        "--input_perturbation",
        type=float,
        default=0.0,
        help=(
            "Add additional noise only to the inputs fed to the model during training."
            " This will make the training converge faster. A value of 0.1 is suggested if you want to enable this."
            " Input perturbation seems to also work with flow-matching (e.g. SD3 and Flux)."
        ),
    )
    parser.add_argument(
        "--input_perturbation_steps",
        type=float,
        default=0,
        help=(
            "Only apply input perturbation over the first N steps with linear decay."
            " This should prevent artifacts from showing up in longer training runs."
        ),
    )
    parser.add_argument(
        "--lr_end",
        type=str,
        default="4e-7",
        help=(
            "A polynomial learning rate will end up at this value after the specified number of warmup steps."
            " A sine or cosine wave will use this value as its lower bound for the learning rate."
        ),
    )
    parser.add_argument(
        "--i_know_what_i_am_doing",
        action="store_true",
        help=(
            "This flag allows you to override some safety checks."
            " It's not recommended to use this unless you are developing the platform."
            " Generally speaking, issue reports submitted with this flag enabled will go to the bottom of the queue."
        ),
    )
    parser.add_argument(
        "--accelerator_cache_clear_interval",
        default=None,
        type=int,
        help=(
            "Clear the cache from VRAM every X steps. This can help prevent memory leaks, but may slow down training."
        ),
    )

    return parser


def get_default_config():
    parser = get_argument_parser()
    default_config = {}
    for action in parser.__dict__["_actions"]:
        if action.dest:
            default_config[action.dest] = action.default

    return default_config


def parse_cmdline_args(input_args=None, exit_on_error: bool = False):
    parser = get_argument_parser()
    args = None
    if input_args is not None:
        for key_val in input_args:
            print_on_main_thread(f"{key_val}")
        try:
            args = parser.parse_args(input_args)
        except:
            logger.error(f"Could not parse input: {input_args}")
            import traceback

            logger.error(traceback.format_exc())
    else:
        args = parser.parse_args()

    if args is None and exit_on_error:
        sys.exit(1)

    if args.optimizer == "adam_bfloat16" and args.mixed_precision != "bf16":
        if not torch.backends.mps.is_available():
            logging.error(
                "You cannot use --adam_bfloat16 without --mixed_precision=bf16."
            )
            sys.exit(1)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.seed is not None:
        if args.seed == 0:
            # the current time should be used if value is zero, providing a rolling seed.
            args.seed = int(time.time())
        elif args.seed == -1:
            # more random seed if value is -1, it will be very different on each startup.
            args.seed = int(random.randint(0, 2**30))

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    if args.cache_dir is None or args.cache_dir == "":
        args.cache_dir = os.path.join(args.output_dir, "cache")

    if args.maximum_image_size is not None and not args.target_downsample_size:
        raise ValueError(
            "When providing --maximum_image_size, you must also provide a value for --target_downsample_size."
        )
    if (
        args.maximum_image_size is not None
        and args.resolution_type == "area"
        and args.maximum_image_size > 5
        and not os.environ.get("SIMPLETUNER_MAXIMUM_IMAGE_SIZE_OVERRIDE", False)
    ):
        raise ValueError(
            f"When using --resolution_type=area, --maximum_image_size must be less than 5 megapixels. You may have accidentally entered {args.maximum_image_size} pixels, instead of megapixels."
        )
    elif (
        args.maximum_image_size is not None
        and args.resolution_type == "pixel"
        and args.maximum_image_size < 512
    ):
        raise ValueError(
            f"When using --resolution_type=pixel, --maximum_image_size must be at least 512 pixels. You may have accidentally entered {args.maximum_image_size} megapixels, instead of pixels."
        )
    if (
        args.target_downsample_size is not None
        and args.resolution_type == "area"
        and args.target_downsample_size > 5
        and not os.environ.get("SIMPLETUNER_MAXIMUM_IMAGE_SIZE_OVERRIDE", False)
    ):
        raise ValueError(
            f"When using --resolution_type=area, --target_downsample_size must be less than 5 megapixels. You may have accidentally entered {args.target_downsample_size} pixels, instead of megapixels."
        )
    elif (
        args.target_downsample_size is not None
        and args.resolution_type == "pixel"
        and args.target_downsample_size < 512
    ):
        raise ValueError(
            f"When using --resolution_type=pixel, --target_downsample_size must be at least 512 pixels. You may have accidentally entered {args.target_downsample_size} megapixels, instead of pixels."
        )

    model_is_bf16 = (
        args.base_model_precision == "no_change"
        and (args.mixed_precision == "bf16" or torch.backends.mps.is_available())
    ) or (
        args.base_model_precision != "no_change"
        and args.base_model_default_dtype == "bf16"
    )
    model_is_quantized = args.base_model_precision != "no_change"
    # check optimiser validity
    chosen_optimizer = args.optimizer
    is_optimizer_deprecated(chosen_optimizer)
    from helpers.training.optimizer_param import optimizer_parameters

    optimizer_cls, optimizer_details = optimizer_parameters(chosen_optimizer, args)
    using_bf16_optimizer = optimizer_details.get("default_settings", {}).get(
        "precision"
    ) in ["any", "bf16"]
    if using_bf16_optimizer and not model_is_bf16:
        raise ValueError(
            f"Model is not using bf16 precision, but the optimizer {chosen_optimizer} requires it."
        )
    if is_optimizer_grad_fp32(args.optimizer):
        warning_log(
            "Using an optimizer that requires fp32 gradients. Training will potentially run more slowly."
        )
        if args.gradient_precision != "fp32":
            args.gradient_precision = "fp32"
    else:
        if args.gradient_precision == "fp32":
            args.gradient_precision = "unmodified"

    if torch.backends.mps.is_available():
        if (
            args.model_family.lower() not in ["sd3", "flux", "legacy"]
            and not args.unet_attention_slice
        ):
            warning_log(
                "MPS may benefit from the use of --unet_attention_slice for memory savings at the cost of speed."
            )
        if args.train_batch_size > 16:
            error_log(
                "An M3 Max 128G will use 12 seconds per step at a batch size of 1 and 65 seconds per step at a batch size of 12."
                " Any higher values will result in NDArray size errors or other unstable training results and crashes."
                "\nPlease reduce the batch size to 12 or lower."
            )
            sys.exit(1)

        if args.quantize_via == "accelerator":
            error_log(
                "MPS does not benefit from models being quantized on the accelerator device. Overriding --quantize_via to 'cpu'."
            )
            args.quantize_via = "cpu"

    if (
        args.max_train_steps is not None
        and args.max_train_steps > 0
        and args.num_train_epochs > 0
    ):
        error_log(
            "When using --max_train_steps (MAX_NUM_STEPS), you must set --num_train_epochs (NUM_EPOCHS) to 0."
        )
        sys.exit(1)

    if (
        args.pretrained_vae_model_name_or_path is not None
        # currently these are the only models we have using the SDXL VAE.
        and args.model_family not in ["sdxl", "pixart_sigma", "kolors"]
        and "sdxl" in args.pretrained_vae_model_name_or_path
        and "deepfloyd" not in args.model_type
    ):
        warning_log(
            f"The VAE model {args.pretrained_vae_model_name_or_path} is not compatible. Please use a compatible VAE to eliminate this warning. The baked-in VAE will be used, instead."
        )
        args.pretrained_vae_model_name_or_path = None
    if (
        args.pretrained_vae_model_name_or_path == ""
        or args.pretrained_vae_model_name_or_path == "''"
    ):
        args.pretrained_vae_model_name_or_path = None

    if "deepfloyd" not in args.model_type:
        info_log(
            f"VAE Model: {args.pretrained_vae_model_name_or_path or args.pretrained_model_name_or_path}"
        )
        info_log(f"Default VAE Cache location: {args.cache_dir_vae}")
        info_log(f"Text Cache location: {args.cache_dir_text}")

    elif "deepfloyd" in args.model_type:
        deepfloyd_pixel_alignment = 8
        if args.aspect_bucket_alignment != deepfloyd_pixel_alignment:
            warning_log(
                f"Overriding aspect bucket alignment pixel interval to {deepfloyd_pixel_alignment}px instead of {args.aspect_bucket_alignment}px."
            )
            args.aspect_bucket_alignment = deepfloyd_pixel_alignment

    if "deepfloyd-stage2" in args.model_type and args.resolution < 256:
        warning_log(
            "DeepFloyd Stage II requires a resolution of at least 256. Setting to 256."
        )
        args.resolution = 256
        args.aspect_bucket_alignment = 64
        args.resolution_type = "pixel"

    validation_resolution_is_float = False
    if "." in str(args.validation_resolution):
        try:
            # this makes handling for int() conversion easier later.
            args.validation_resolution = float(args.validation_resolution)
            validation_resolution_is_float = True
        except ValueError:
            pass
    validation_resolution_is_digit = False
    try:
        int(args.validation_resolution)
        validation_resolution_is_digit = True
    except ValueError:
        pass

    if (
        (validation_resolution_is_digit or validation_resolution_is_float)
        and int(args.validation_resolution) < 128
        and "deepfloyd" not in args.model_type
    ):
        # Convert from megapixels to pixels:
        log_msg = f"It seems that --validation_resolution was given in megapixels ({args.validation_resolution}). Converting to pixel measurement:"
        if int(args.validation_resolution) == 1:
            args.validation_resolution = 1024
        else:
            args.validation_resolution = int(int(args.validation_resolution) * 1e3)
            # Make it divisible by 8:
            args.validation_resolution = int(int(args.validation_resolution) / 8) * 8
        info_log(f"{log_msg} {int(args.validation_resolution)}px")
    if args.timestep_bias_portion < 0.0 or args.timestep_bias_portion > 1.0:
        raise ValueError("Timestep bias portion must be between 0.0 and 1.0.")

    if args.controlnet and "lora" in args.model_type:
        raise ValueError("ControlNet is not supported for LoRA models.")

    if args.metadata_update_interval < 60:
        raise ValueError("Metadata update interval must be at least 60 seconds.")

    args.vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )

    if args.use_ema and args.ema_cpu_only:
        args.ema_device = "cpu"

    if (args.optimizer_beta1 is not None and args.optimizer_beta2 is None) or (
        args.optimizer_beta1 is None and args.optimizer_beta2 is not None
    ):
        error_log("Both --optimizer_beta1 and --optimizer_beta2 should be provided.")
        sys.exit(1)

    if args.gradient_checkpointing:
        # enable torch compile w/ activation checkpointing :[ slows us down.
        torch._dynamo.config.optimize_ddp = False

    args.logging_dir = os.path.join(args.output_dir, args.logging_dir)
    args.accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=args.logging_dir
    )
    # Create the custom configuration
    args.process_group_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=5400)
    )  # 1.5 hours

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if args.disable_tf32:
            warning_log(
                "--disable_tf32 is provided, not enabling. Training will potentially be much slower."
            )
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        else:
            info_log(
                "Enabled NVIDIA TF32 for faster training on Ampere GPUs. Use --disable_tf32 if this causes any problems."
            )

    args.is_quantized = (
        False
        if (args.base_model_precision == "no_change" or "lora" not in args.model_type)
        else True
    )
    args.weight_dtype = (
        torch.bfloat16
        if (
            args.mixed_precision == "bf16"
            or (args.base_model_default_dtype == "bf16" and args.is_quantized)
        )
        else torch.float16 if args.mixed_precision == "fp16" else torch.float32
    )
    args.disable_accelerator = os.environ.get("SIMPLETUNER_DISABLE_ACCELERATOR", False)

    if "lycoris" == args.lora_type.lower():
        from lycoris import create_lycoris

        if args.lycoris_config is None:
            raise ValueError(
                "--lora_type=lycoris requires you to add a JSON "
                + "configuration file location with --lycoris_config"
            )
        # is it readable?
        if not os.path.isfile(args.lycoris_config) or not os.access(
            args.lycoris_config, os.R_OK
        ):
            raise ValueError(
                f"Could not find the JSON configuration file at '{args.lycoris_config}'"
            )
        import json

        with open(args.lycoris_config, "r") as f:
            lycoris_config = json.load(f)
        assert "algo" in lycoris_config, "lycoris_config JSON must contain algo key"
        assert (
            "multiplier" in lycoris_config
        ), "lycoris_config JSON must contain multiplier key"
        if (
            "full_matrix" not in lycoris_config
            or lycoris_config.get("full_matrix") is not True
        ):
            assert (
                "linear_dim" in lycoris_config
            ), "lycoris_config JSON must contain linear_dim key if full_matrix is not set."
        assert (
            "linear_alpha" in lycoris_config
        ), "lycoris_config JSON must contain linear_alpha key"

    elif "standard" == args.lora_type.lower():
        if hasattr(args, "lora_init_type") and args.lora_init_type is not None:
            if torch.backends.mps.is_available() and args.lora_init_type == "loftq":
                error_log(
                    "Apple MPS cannot make use of LoftQ initialisation. Overriding to 'default'."
                )
            elif args.is_quantized and args.lora_init_type == "loftq":
                error_log(
                    "LoftQ initialisation is not supported with quantised models. Overriding to 'default'."
                )
            else:
                args.lora_initialisation_style = (
                    args.lora_init_type if args.lora_init_type != "default" else True
                )
        if args.use_dora:
            if "quanto" in args.base_model_precision:
                error_log(
                    "Quanto does not yet support DoRA training in PEFT. Disabling DoRA. 😴"
                )
                args.use_dora = False
            else:
                warning_log(
                    "DoRA support is experimental and not very thoroughly tested."
                )
                args.lora_initialisation_style = "default"

    if args.distillation_config is not None:
        if args.distillation_config.startswith("{"):
            try:
                import ast

                args.distillation_config = ast.literal_eval(args.distillation_config)
            except Exception as e:
                logger.error(f"Could not load distillation_config: {e}")
                raise

    if not args.data_backend_config:
        from helpers.training.state_tracker import StateTracker

        args.data_backend_config = os.path.join(
            StateTracker.get_config_path(), "multidatabackend.json"
        )
        warning_log(
            f"No data backend config provided. Using default config at {args.data_backend_config}."
        )

    if (
        args.validation_num_video_frames is not None
        and args.validation_num_video_frames < 1
    ):
        raise ValueError("validation_num_video_frames must be at least 1.")

    # Check if we have a valid gradient accumulation steps.
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, should be >= 1"
        )

    if args.validation_guidance_skip_layers is not None:
        if args.model_family not in ["sd3", "wan"]:
            raise ValueError(
                "Currently, skip-layer guidance is not supported for {}".format(
                    args.model_family
                )
            )
        try:
            import json

            args.validation_guidance_skip_layers = json.loads(
                args.validation_guidance_skip_layers
            )
        except Exception as e:
            logger.error(f"Could not load validation_guidance_skip_layers: {e}")
            raise

    if (
        args.sana_complex_human_instruction is not None
        and type(args.sana_complex_human_instruction) is str
        and args.sana_complex_human_instruction not in ["", "None"]
    ):
        try:
            import json

            args.sana_complex_human_instruction = json.loads(
                args.sana_complex_human_instruction
            )
        except Exception as e:
            logger.error(
                f"Could not load complex human instruction ({args.sana_complex_human_instruction}): {e}"
            )
            raise
    elif args.sana_complex_human_instruction == "None":
        args.sana_complex_human_instruction = None

    if args.attention_mechanism != "diffusers" and not torch.cuda.is_available():
        warning_log(
            "For non-CUDA systems, only Diffusers attention mechanism is officially supported."
        )

    deprecated_options = {
        # how to deprecate options:
        # "flux_beta_schedule_alpha": "flow_beta_schedule_alpha",
    }

    for deprecated_option, replacement_option in deprecated_options.items():
        if (
            getattr(args, replacement_option) is not None
            and getattr(args, deprecated_option) is not None
            and type(getattr(args, deprecated_option)) is not object
        ):
            warning_log(
                f"The option --{deprecated_option} has been replaced with --{replacement_option}."
            )
            setattr(args, replacement_option, getattr(args, deprecated_option))
        elif getattr(args, deprecated_option) is not None:
            error_log(
                f"The option {deprecated_option} has been deprecated without a replacement option. Please remove it from your configuration."
            )
            sys.exit(1)

    return args
