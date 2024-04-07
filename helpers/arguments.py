import argparse, os, random, time, json, logging, sys, torch
from pathlib import Path
from helpers.training.state_tracker import StateTracker

logger = logging.getLogger("ArgsParser")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


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
        "--model_type",
        type=str,
        choices=["full", "lora"],
        default="full",
        help=(
            "The training type to use. 'full' will train the full model, while 'lora' will train the LoRA model."
            " LoRA is a smaller model that can be used for faster training."
        ),
    )
    parser.add_argument(
        "--lora_type",
        type=str,
        choices=["Standard"],
        help=(
            "When training using --model_type=lora, you may specify a different type of LoRA to train here."
            " Currently, only 'Standard' type is supported. This option exists for compatibility with Kohya configuration files."
        ),
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
        default=16,
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
        default=False,
        help=(
            "By default, will encode images during training. For some situations, pre-processing may be desired."
            " To revert to the old behaviour, supply --vae_cache_preprocess=false."
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
            " float32 precision."
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
        default="area",
        choices=["pixel", "area"],
        help=(
            "Resizing images maintains aspect ratio. This defines the resizing strategy."
            " If 'pixel', the images will be resized to the resolution by pixel edge."
            " If 'area', the images will be resized so the pixel area is this many megapixels."
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
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--use_adafactor_optimizer",
        action="store_true",
        help="Whether or not to use the Adafactor optimizer.",
    )
    parser.add_argument(
        "--adafactor_relative_step",
        type=bool,
        default=False,
        help=(
            "When set, will use the experimental Adafactor mode for relative step computations instead of the value set by --learning_rate."
            " This is an experimental feature, and you are on your own for support."
        ),
    )
    parser.add_argument(
        "--use_prodigy_optimizer",
        action="store_true",
        help="Whether or not to use the Prodigy optimizer.",
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_decouple",
        type=bool,
        default=True,
        help="Use AdamW style decoupled weight decay",
    )
    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_learning_rate",
        type=float,
        default=0.5,
        help=(
            "Though this is called the prodigy learning rate, it corresponds to the d_coef parameter in the Prodigy optimizer."
            " This acts as a coefficient in the expression for the estimate of d. Default for this trainer is 0.5, but the Prodigy"
            " default is 1.0, which ends up over-cooking models."
        ),
    )
    parser.add_argument(
        "--prodigy_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use. Prodigy default is 0, but SimpleTuner uses 1e-2.",
    )
    parser.add_argument(
        "--prodigy_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--use_dadapt_optimizer",
        action="store_true",
        help="Whether or not to use the discriminator adaptation optimizer.",
    )
    parser.add_argument(
        "--dadaptation_learning_rate",
        type=float,
        default=1.0,
        help="Learning rate for the discriminator adaptation. Default: 1.0",
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
        help="Whether or not to use stochastic bf16 in Adam. Currently the only supported optimizer.",
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
        help=(
            "The token to use to push to the Model Hub. Do not use in combination with --report_to=wandb,"
            " as this value will be exposed in the logs. Instead, use `huggingface-cli login` on the command line."
        ),
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
        type=float,
        default=256,
        help="Square resolution images will be output at this resolution (256x256).",
    )
    parser.add_argument(
        "--validation_noise_scheduler",
        type=str,
        choices=["ddim", "ddpm", "euler", "euler-a", "unipc"],
        default="euler",
        help=(
            "When validating the model at inference time, a different scheduler may be chosen."
            " UniPC can offer better speed, and Euler A can put up with instabilities a bit better."
            " For zero-terminal SNR models, DDIM is the best choice. Choices: ['ddim', 'ddpm', 'euler', 'euler-a', 'unipc'],"
            " Default: ddim"
        ),
    )
    parser.add_argument(
        "--disable_compel",
        action="store_true",
        help=(
            "If provided, prompts will be handled using the typical prompt encoding strategy."
            " Otherwise, the default behaviour is to use Compel for prompt embed generation."
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
        choices=["bf16"],
        help=(
            "SimpleTuner only supports bf16 training. Bf16 requires PyTorch >="
            " 1.10. on an Nvidia Ampere or later GPU, and PyTorch 2.3 or newer for Apple Silicon."
            " Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
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
        "--freeze_unet_strategy",
        type=str,
        choices=["none", "bitfit"],
        default="none",
        help=(
            "When freezing the UNet, we can use the 'none' or 'bitfit' strategy."
            " The 'bitfit' strategy will freeze all weights, and leave bias thawed."
            " The default strategy is to leave the full u-net thawed."
            " Freezing the weights can improve convergence for finetuning."
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
        "--input_perturbation",
        type=float,
        default=0,
        help="The scale of input pretubation. Recommended 0.1.",
    )
    parser.add_argument(
        "--input_perturbation_probability",
        type=float,
        default=0.25,
        help=(
            "While input perturbation can help with training convergence, having it applied all the time is likely damaging."
            " When this value is less than 1.0, any perturbed noise will be applied probabilistically. Default: 0.25"
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
            "If you are using an optimizer other than AdamW, you must set this flag to continue."
            " This is a safety feature to prevent accidental use of an unsupported optimizer, as weights are stored in bfloat16."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.adam_bfloat16 and args.mixed_precision != "bf16":
        logging.error("You cannot use --adam_bfloat16 without --mixed_precision=bf16.")
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

    if not args.adamw_bfloat16 and not args.i_know_what_i_am_doing:
        raise ValueError(
            "Currently, only the AdamW optimizer supports bfloat16 training. Please set --adamw_bfloat16 to true."
        )

    if not args.i_know_what_i_am_doing and (
        args.use_prodigy_optimizer
        or args.use_dadapt_optimizer
        or args.use_adafactor_optimizer
        or args.use_8bit_adam
    ):
        raise ValueError(
            "Currently, only the AdamW optimizer supports bfloat16 training. Please set --adamw_bfloat16 to true, or set --i_know_what_i_am_doing."
        )

    if torch.backends.mps.is_available():
        if not args.unet_attention_slice and StateTracker.get_model_type() != "legacy":
            logger.warning(
                "MPS may benefit from the use of --unet_attention_slice for memory savings at the cost of speed."
            )
        if args.train_batch_size > 12:
            logger.error(
                "An M3 Max 128G will use 12 seconds per step at a batch size of 1 and 65 seconds per step at a batch size of 12."
                " Any higher values will result in NDArray size errors or other unstable training results and crashes."
                "\nPlease reduce the batch size to 12 or lower."
            )
            sys.exit(1)

    if args.cache_dir_vae is None or args.cache_dir_vae == "":
        args.cache_dir_vae = os.path.join(args.output_dir, "cache_vae")
    if args.cache_dir_text is None or args.cache_dir_text == "":
        args.cache_dir_text = os.path.join(args.output_dir, "cache_text")
    for target_dir in [
        Path(args.cache_dir),
        Path(args.cache_dir_vae),
        Path(args.cache_dir_text),
    ]:
        os.makedirs(target_dir, exist_ok=True)

    if (
        args.pretrained_vae_model_name_or_path is not None
        and StateTracker.get_model_type() == "legacy"
        and "sdxl" in args.pretrained_vae_model_name_or_path
    ):
        logger.error(
            f"The VAE model {args.pretrained_vae_model_name_or_path} is not compatible with SD 2.x. Please use a 2.x VAE to eliminate this error."
        )
        args.pretrained_vae_model_name_or_path = None
    logger.info(
        f"VAE Model: {args.pretrained_vae_model_name_or_path or args.pretrained_model_name_or_path}"
    )
    logger.info(f"Default VAE Cache location: {args.cache_dir_vae}")
    logger.info(f"Text Cache location: {args.cache_dir_text}")

    if args.validation_resolution < 128:
        # Convert from megapixels to pixels:
        log_msg = f"It seems that --validation_resolution was given in megapixels ({args.validation_resolution}). Converting to pixel measurement:"
        if args.validation_resolution == 1:
            args.validation_resolution = 1024
        else:
            args.validation_resolution = int(args.validation_resolution * 1e3)
            # Make it divisible by 8:
            args.validation_resolution = int(args.validation_resolution / 8) * 8
        logger.info(f"{log_msg} {args.validation_resolution}px")
    if args.timestep_bias_portion < 0.0 or args.timestep_bias_portion > 1.0:
        raise ValueError("Timestep bias portion must be between 0.0 and 1.0.")

    if args.metadata_update_interval < 60:
        raise ValueError("Metadata update interval must be at least 60 seconds.")
    if args.validation_torch_compile == "true":
        args.validation_torch_compile = True
    else:
        args.validation_torch_compile = False
    return args
