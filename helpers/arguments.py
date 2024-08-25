import argparse
import os
import random
import time
import logging
import sys
import torch
from helpers.models.smoldit import SmolDiTConfigurationNames
from helpers.training import quantised_precision_levels
from helpers.training.optimizer_param import (
    map_args_to_optimizer,
    is_optimizer_deprecated,
    is_optimizer_bf16,
    map_deprecated_optimizer_parameter,
    optimizer_choices,
)

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


def info_log(message):
    if is_primary_process:
        logger.info(message)


def warning_log(message):
    if is_primary_process:
        logger.warning(message)


def error_log(message):
    if is_primary_process:
        logger.error(message)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="The following SimpleTuner command-line options are available:"
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
        "--model_type",
        type=str,
        choices=[
            "full",
            "lora",
            "deepfloyd-full",
            "deepfloyd-lora",
            "deepfloyd-stage2",
            "deepfloyd-stage2-lora",
        ],
        default="full",
        help=(
            "The training type to use. 'full' will train the full model, while 'lora' will train the LoRA model."
            " LoRA is a smaller model that can be used for faster training."
        ),
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        default=False,
        help=(
            "This option must be provided when training a Stable Diffusion 1.x or 2.x model."
        ),
    )
    parser.add_argument(
        "--kolors",
        action="store_true",
        default=False,
        help=("This option must be provided when training a Kolors model."),
    )
    parser.add_argument(
        "--flux",
        action="store_true",
        default=False,
        help=("This option must be provided when training a Flux model."),
    )
    parser.add_argument(
        "--flux_lora_target",
        type=str,
        choices=["mmdit", "context", "context+ffs", "all", "all+ffs", "ai-toolkit"],
        default="all",
        help=(
            "Flux has single and joint attention blocks."
            " By default, all attention layers are trained, but not the feed-forward layers"
            " If 'mmdit' is provided, the text input layers will not be trained."
            " If 'context' is provided, then ONLY the text attention layers are trained"
            " If 'context+ffs' is provided, then text attention and text feed-forward layers are trained. This is somewhat similar to text-encoder-only training in earlier SD versions."
            " If 'all' is provided, all layers will be trained, minus feed-forward."
            " If 'all+ffs' is provided, all layers will be trained including feed-forward."
        ),
    )
    parser.add_argument(
        "--flow_matching_sigmoid_scale",
        type=float,
        default=1.0,
        help="Scale factor for sigmoid timestep sampling for flow-matching models..",
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
        help="Use attention masking while training flux.",
    )
    parser.add_argument(
        "--smoldit",
        action="store_true",
        default=False,
        help=("Use the experimental SmolDiT model architecture."),
    )
    parser.add_argument(
        "--smoldit_config",
        type=str,
        choices=SmolDiTConfigurationNames,
        default="smoldit-base",
        help=(
            "The SmolDiT configuration to use. This is a list of pre-configured models."
            " The default is 'smoldit-base'."
        ),
    )
    parser.add_argument(
        "--flow_matching_loss",
        type=str,
        choices=["diffusers", "compatible", "diffusion"],
        default="compatible",
        help=(
            "A discrepancy exists between the Diffusers implementation of flow matching and the minimal implementation provided"
            " by StabilityAI. This experimental option allows switching loss calculations to be compatible with those."
            " Additionally, 'diffusion' is offered as an option to reparameterise a model to v_prediction loss."
        ),
    )
    parser.add_argument(
        "--pixart_sigma",
        action="store_true",
        default=False,
        help=("This must be set when training a PixArt Sigma model."),
    )
    parser.add_argument(
        "--sd3",
        action="store_true",
        default=False,
        help=("This option must be provided when training a Stable Diffusion 3 model."),
    )
    parser.add_argument(
        "--sd3_t5_mask_behaviour",
        type=str,
        choices=["do-nothing", "mask"],
        default="mask",
        help=(
            "StabilityAI did not correctly implement their attention masking on T5 inputs for SD3 Medium."
            " This option enables you to switch between their broken implementation or the corrected mask"
            " implementation. Although, the corrected masking is still applied via hackish workaround,"
            " manually applying the mask to the prompt embeds so that the padded positions are zero."
            " This improves the results for short captions, but does not change the behaviour for long captions."
            " It is important to note that this limitation currently prevents expansion of SD3 Medium's"
            " prompt length, as it will unnecessarily attend to every token in the prompt embed,"
            " even masked positions."
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
        help="Specify an existing LoRA safetensors file to initialize the LoRA and continue training or finetune an existing LoRA.",
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
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
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
        default="epsilon",
        choices=["epsilon", "v_prediction", "sample"],
        help=(
            "The type of prediction to use for the u-net. Choose between ['epsilon', 'v_prediction', 'sample']."
            " For SD 2.1-v, this is v_prediction. For 2.1-base, it is epsilon. SDXL is generally epsilon."
            " SD 1.5 is epsilon."
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
        "--vae_cache_preprocess",
        action="store_true",
        default=True,
        help=(
            "This option is deprecated and will be removed in a future release. Use --vae_cache_ondemand instead."
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
        required=True,
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
            " number of images in each backend. 'uniform' will sample from each backend equally, which may be"
            " more desirable for DreamBooth training with eg. ignore_epochs=True on your regularisation dataset."
        ),
    )
    parser.add_argument(
        "--write_batch_size",
        type=int,
        default=64,
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
            " This is useful if you've modified any of the existing prompts, or, disabled/enabled Compel,"
            " via `--disable_compel`"
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
            " For DeepFloyd, this value is set to 8, but all other training defaults to 64. You may experiment"
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
    # DeepFloyd
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=None,
        required=False,
        help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )
    # End DeepFloyd-specific settings
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
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
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
        "--use_ema",
        action="store_true",
        help="Whether to use EMA (exponential moving average) model.",
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
        "--use_8bit_adam",
        action="store_true",
        help="Deprecated in favour of --optimizer=optimi-adamw.",
    )
    parser.add_argument(
        "--use_adafactor_optimizer",
        action="store_true",
        help="Deprecated in favour of --optimizer=stableadamw.",
    )
    parser.add_argument(
        "--use_prodigy_optimizer",
        action="store_true",
        help="Deprecated and removed.",
    )
    parser.add_argument(
        "--use_dadapt_optimizer",
        action="store_true",
        help="Deprecated and removed.",
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
        "--adam_bfloat16",
        action="store_true",
        help="Deprecated in favour of --optimizer=adamw_bf16.",
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
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--benchmark_base_model",
        action="store_true",
        default=False,
        help=(
            "If set, before training, the base model images will be sampled via the same prompts and saved to the output directory."
            " These samples will be stitched to each validation output. Note that currently this cannot be enabled after training begins."
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
        "--validation_torch_compile",
        type=str,
        default="false",
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
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
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
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
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
        "--num_eval_images",
        type=int,
        default=4,
        help=(
            "If possible, this many eval images will be selected from each dataset."
            " This is used when training super-resolution models such as DeepFloyd Stage II,"
            " which will upscale input images from the training set."
        ),
    )
    parser.add_argument(
        "--eval_dataset_id",
        type=str,
        default=None,
        help=(
            "When provided, only this dataset's images will be used as the eval set, to keep"
            " the training and eval images split."
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
        "--validation_resolution",
        type=str,
        default=256,
        help="Square resolution images will be output at this resolution (256x256).",
    )
    parser.add_argument(
        "--validation_noise_scheduler",
        type=str,
        choices=["ddim", "ddpm", "euler", "euler-a", "unipc"],
        default=None,
        help=(
            "When validating the model at inference time, a different scheduler may be chosen."
            " UniPC can offer better speed, and Euler A can put up with instabilities a bit better."
            " For zero-terminal SNR models, DDIM is the best choice. Choices: ['ddim', 'ddpm', 'euler', 'euler-a', 'unipc'],"
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
        "--disable_compel",
        action="store_true",
        help=(
            "This option does nothing. It is deprecated and will be removed in a future release."
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
        choices=["bf16", "no"],
        help=(
            "SimpleTuner only supports bf16 training. Bf16 requires PyTorch >="
            " 1.10. on an Nvidia Ampere or later GPU, and PyTorch 2.3 or newer for Apple Silicon."
            " Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
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
    for i in range(1, 4):
        parser.add_argument(
            f"--text_encoder_{i}_precision",
            type=str,
            default=None,
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
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
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
        help="Use real CFG sampling for Flux validation images. Default: 1.0 (no CFG)",
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

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.adam_bfloat16 and args.mixed_precision != "bf16":
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

    if "int4" in args.base_model_precision and torch.cuda.is_available():
        print(
            "WARNING: int4 precision is ONLY supported on A100 and H100 or newer devices. Waiting 10 seconds to continue.."
        )
        time.sleep(10)

    model_is_bf16 = (
        args.base_model_precision == "no_change"
        and (args.mixed_precision == "bf16" or torch.backends.mps.is_available())
    ) or (
        args.base_model_precision != "no_change"
        and args.base_model_default_dtype == "bf16"
    )
    model_is_quantized = args.base_model_precision != "no_change"
    # check optimiser validity
    chosen_optimizer = map_args_to_optimizer(args)
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
    print(f"optimizer: {optimizer_details}")

    if torch.backends.mps.is_available():
        if (
            not args.flux
            and not args.sd3
            and not args.unet_attention_slice
            and not args.legacy
        ):
            warning_log(
                "MPS may benefit from the use of --unet_attention_slice for memory savings at the cost of speed."
            )
        if not args.smoldit and args.train_batch_size > 16:
            error_log(
                "An M3 Max 128G will use 12 seconds per step at a batch size of 1 and 65 seconds per step at a batch size of 12."
                " Any higher values will result in NDArray size errors or other unstable training results and crashes."
                "\nPlease reduce the batch size to 12 or lower."
            )
            sys.exit(1)

    if args.max_train_steps is not None and args.num_train_epochs > 0:
        error_log(
            "When using --max_train_steps (MAX_NUM_STEPS), you must set --num_train_epochs (NUM_EPOCHS) to 0."
        )
        sys.exit(1)

    if (
        args.pretrained_vae_model_name_or_path is not None
        and any([args.legacy, args.flux])
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
    if args.sd3:
        warning_log(
            "MM-DiT requires an alignment value of 64px. Overriding the value of --aspect_bucket_alignment."
        )
        args.aspect_bucket_alignment = 64
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
    if "." in args.validation_resolution:
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
    if args.validation_torch_compile == "true":
        args.validation_torch_compile = True
    else:
        args.validation_torch_compile = False

    if args.sd3:
        args.pretrained_vae_model_name_or_path = None
        args.disable_compel = True

    t5_max_length = 77
    if args.sd3 and (
        args.tokenizer_max_length is None
        or int(args.tokenizer_max_length) > t5_max_length
    ):
        if not args.i_know_what_i_am_doing:
            warning_log(
                f"Updating T5 XXL tokeniser max length to {t5_max_length} for SD3."
            )
            args.tokenizer_max_length = t5_max_length
        else:
            warning_log(
                f"-!- SD3 supports a max length of {t5_max_length} tokens, but you have supplied `--i_know_what_i_am_doing`, so this limit will not be enforced. -!-"
            )
            warning_log(
                f"The model will begin to collapse after a short period of time, if the model you are continuing from has not been tuned beyond {t5_max_length} tokens."
            )
    flux_version = "dev"
    model_max_seq_length = 512
    if (
        "schnell" in args.pretrained_model_name_or_path.lower()
        or args.flux_fast_schedule
    ):
        if not args.flux_fast_schedule:
            logger.error("Schnell requires --flux_fast_schedule.")
            sys.exit(1)
        flux_version = "schnell"
        model_max_seq_length = 256

    if args.flux:
        if (
            args.tokenizer_max_length is None
            or int(args.tokenizer_max_length) > model_max_seq_length
        ):
            if not args.i_know_what_i_am_doing:
                warning_log(
                    f"Updating T5 XXL tokeniser max length to {model_max_seq_length} for Flux."
                )
                args.tokenizer_max_length = model_max_seq_length
            else:
                warning_log(
                    f"-!- Flux supports a max length of {model_max_seq_length} tokens, but you have supplied `--i_know_what_i_am_doing`, so this limit will not be enforced. -!-"
                )
                warning_log(
                    f"The model will begin to collapse after a short period of time, if the model you are continuing from has not been tuned beyond 256 tokens."
                )
        if flux_version == "dev":
            if args.validation_num_inference_steps > 28:
                warning_log(
                    "Flux Dev expects around 28 or fewer inference steps. Consider limiting --validation_num_inference_steps to 28."
                )
            if args.validation_num_inference_steps < 15:
                warning_log(
                    "Flux Dev expects around 15 or more inference steps. Consider increasing --validation_num_inference_steps to 15."
                )
        if flux_version == "schnell" and args.validation_num_inference_steps > 4:
            warning_log(
                "Flux Schnell requires fewer inference steps. Consider reducing --validation_num_inference_steps to 4."
            )

    if args.use_ema and args.ema_cpu_only:
        args.ema_device = "cpu"

    if (args.optimizer_beta1 is not None and args.optimizer_beta2 is None) or (
        args.optimizer_beta1 is None and args.optimizer_beta2 is not None
    ):
        error_log("Both --optimizer_beta1 and --optimizer_beta2 should be provided.")
        sys.exit(1)

    if not args.i_know_what_i_am_doing:
        if args.pixart_sigma or args.sd3:
            if args.max_grad_norm is None or float(args.max_grad_norm) > 0.01:
                warning_log(
                    f"{'PixArt Sigma' if args.pixart_sigma else 'Stable Diffusion 3'} requires --max_grad_norm=0.01 to prevent model collapse. Overriding value. Set this value manually to disable this warning."
                )
                args.max_grad_norm = 0.01

    if args.gradient_accumulation_steps > 1:
        if args.gradient_precision == "unmodified" or args.gradient_precision is None:
            warning_log(
                "Gradient accumulation steps are enabled, but gradient precision is set to 'unmodified'."
                " This may lead to numeric instability. Consider disabling gradient accumulation steps. Continuing in 10 seconds.."
            )
            time.sleep(10)
        elif args.gradient_precision == "fp32":
            info_log(
                "Gradient accumulation steps are enabled, and gradient precision is set to 'fp32'."
            )
            args.gradient_precision = "fp32"

    if args.use_ema:
        if args.sd3:
            raise ValueError(
                "Using EMA is not currently supported for Stable Diffusion 3 training."
            )
        if "lora" in args.model_type:
            raise ValueError("Using EMA is not currently supported for LoRA training.")

    return args
