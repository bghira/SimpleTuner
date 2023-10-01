import argparse, os, random, time, json, logging

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
            "Spacing timesteps can fundamentally alter the course of history. Er, I mean, your model weights."
            " For all training, including epsilon, it would seem that 'trailing' is the right choice."
        ),
    )
    parser.add_argument(
        "--inference_scheduler_timestep_spacing",
        type=str,
        default="trailing",
        choices=["leading", "linspace", "trailing"],
        help=(
            "The Bytedance paper on zero terminal SNR recommends inference using 'trailing'."
        ),
    )
    parser.add_argument(
        "--timestep_bias_strategy",
        type=str,
        default="none",
        choices=["earlier", "later", "none"],
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
        required=False,
        help=(
            "The dtype of the VAE model. Choose between ['default', 'fp16', 'fp32', 'bf16']."
            "The default VAE dtype is float32, due to NaN issues in SDXL 1.0."
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
            " Valid options: aspect, vae, text."
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
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help=(
            "A folder containing the training data. Folder contents must either follow the structure described in"
            " the SimpleTuner documentation (https://github.com/bghira/SimpleTuner), or the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. For 🤗 Datasets in particular,"
            " a `metadata.jsonl` file must exist to provide the captions for the images. For SimpleTuner layout,"
            " the images can be in subfolders. No particular config is required. Ignored if `dataset_name` is specified."
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
            " If the AWS backend is in use, this will be a prefix for the bucket's VAE cache entries."
        ),
    )
    parser.add_argument(
        "--data_backend",
        type=str,
        default="local",
        choices=["local", "aws"],
        help=(
            "The data backend to use. Choose between ['local', 'aws']. Default: local."
            " If using AWS, you must set the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
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
        "--apply_dataset_padding",
        action="store_true",
        default=False,
        help=(
            "If set, will apply padding to the dataset to ensure that the number of images is divisible by the batch."
            " This has some side-effects (especially on smaller datasets) of over-sampling and overly repeating images."
        ),
    )
    parser.add_argument(
        "--aws_config_file",
        type=str,
        default=None,
        help=(
            "Path to the AWS configuration file in JSON format."
            " Config key names are the same as SimpleTuner option counterparts."
        ),
    )
    parser.add_argument(
        "--aws_bucket_name",
        type=str,
        default=None,
        help="The AWS bucket name to use.",
    )
    parser.add_argument(
        "--aws_bucket_image_prefix",
        type=str,
        default="",
        help=(
            "Instead of using --instance_data_dir, AWS S3 relies on aws_bucket_*_prefix parameters."
            " When provided, this parameter will be prepended to the image path."
        ),
    )
    parser.add_argument(
        "--aws_endpoint_url",
        type=str,
        default=None,
        help=(
            "The AWS server to use. If not specified, will use the default server for the region specified."
            " For Wasabi, use https://s3.wasabisys.com."
        ),
    )
    parser.add_argument(
        "--aws_region_name",
        type=str,
        default="us-east-1",
        help=(
            "The AWS region to use. If not specified, will use the default region for the server specified."
            " For example, if you specify 's3.amazonaws.com', the default region will be 'us-east-1'."
        ),
    )
    parser.add_argument(
        "--aws_access_key_id",
        type=str,
        default=None,
        help="The AWS access key ID.",
    )
    parser.add_argument(
        "--aws_secret_access_key",
        type=str,
        default=None,
        help="The AWS secret access key.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
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
        "--seen_state_path",
        type=str,
        default="seen_images.json",
        help="Where the JSON document containing the state of the seen images is stored. This helps ensure we do not repeat images too many times.",
    )
    parser.add_argument(
        "--state_path",
        type=str,
        default="training_state.json",
        help="A JSON document containing the current state of training, will be placed here.",
    )
    parser.add_argument(
        "--caption_strategy",
        type=str,
        default="filename",
        choices=["filename", "textfile", "instance_prompt"],
        help=(
            "The default captioning strategy, 'filename', will use the filename as the caption, after stripping some characters like underscores."
            "The 'textfile' strategy will use the contents of a text file with the same name as the image."
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
        default="pixel",
        choices=["pixel", "area"],
        help=(
            "Resizing images maintains aspect ratio. This defines the resizing strategy."
            " If 'pixel', the images will be resized to the resolution by pixel edge."
            " If 'area', the images will be resized so the pixel area is this many megapixels."
        ),
    )
    parser.add_argument(
        "--minimum_image_size",
        type=int,
        default=768,
        help=(
            "The minimum resolution for both sides of input images."
            " If --delete_unwanted_images is set, images smaller than this will be DELETED."
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
            " cropped. The images will be resized to the resolution first before cropping. If training SDXL,"
            " the VAE cache and aspect bucket cache will need to be (re)built so they include crop coordinates."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set. Currently untested."
        ),
    )
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
        default="polynomial",
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
        "--track_luminance",
        action="store_true",
        help=(
            "When provided, the luminance of the images will be tracked during training."
            " This has a pretty substantial compute cost for higher resolution images,"
            " though it is easily justified when training with offset noise or some other"
            " noise modification technique that could bias the model toward very-dark images."
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
        "--num_validation_images",
        type=int,
        default=4,
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
        type=int,
        default=256,
        help="Square resolution images will be output at this resolution (256x256).",
    )
    parser.add_argument(
        "--validation_noise_scheduler",
        type=str,
        choices=["ddim", "ddpm", "euler", "euler-a", "unipc"],
        default="ddim",
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
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
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
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
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
        "--freeze_encoder",
        action="store_true",
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
        "--caption_dropout_interval",
        type=int,
        default=0,
        help=(
            "Every X steps, we will drop the caption from the input to assist in classifier-free guidance training."
            " When StabilityAI trained Stable Diffusion, a value of 10 was used."
            " Very high values might be useful to do some sort of enforced style training."
            " Default value is zero, maximum value is 100."
        ),
    )
    parser.add_argument(
        "--conditioning_dropout_probability",
        type=float,
        default=None,
        help="Conditioning dropout probability. Experimental. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--caption_dropout_probability",
        type=float,
        default=None,
        help="Caption dropout probability. Same as caption_dropout_interval, but this is for SDXL.",
    )
    parser.add_argument(
        "--input_pertubation",
        type=float,
        default=0,
        help="The scale of input pretubation. Recommended 0.1.",
    )
    parser.add_argument(
        "--use_original_images",
        type=str,
        default="false",
        help=(
            "When this option is provided, image cropping and processing will be disabled."
            " It is a good idea to use this with caution, for training multiple aspect ratios."
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
        "--learning_rate_end",
        type=str,
        default="4e-7",
        help="A polynomial learning rate will end up at this value after the specified number of warmup steps.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

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

    if args.aws_config_file is not None:
        try:
            with open(args.aws_config_file, "r") as f:
                aws_config = json.load(f)
        except Exception as e:
            raise ValueError(f"Could not load AWS config file: {e}")
        if not isinstance(aws_config, dict):
            raise ValueError("AWS config file must be a JSON object.")
        args.aws_bucket_name = aws_config.get("aws_bucket_name", args.aws_bucket_name)
        args.aws_bucket_image_prefix = aws_config.get("aws_bucket_image_prefix", "")
        args.aws_endpoint_url = aws_config.get(
            "aws_endpoint_url", args.aws_endpoint_url
        )
        args.aws_region_name = aws_config.get("aws_region_name", args.aws_region_name)
        args.aws_access_key_id = aws_config.get(
            "aws_access_key_id", args.aws_access_key_id
        )
        args.aws_secret_access_key = aws_config.get(
            "aws_secret_access_key", args.aws_secret_access_key
        )
    if args.data_backend == "aws":
        if args.aws_bucket_name is None:
            raise ValueError("Must specify an AWS bucket name.")
        if args.aws_endpoint_url is None and args.aws_region_name is None:
            raise ValueError("Must specify an AWS endpoint URL or region name.")
        if args.aws_access_key_id is None:
            raise ValueError("Must specify an AWS access key ID.")
        if args.aws_secret_access_key is None:
            raise ValueError("Must specify an AWS secret access key.")
        # Override the instance data dir with the bucket image prefix.
        args.instance_data_dir = args.aws_bucket_image_prefix
    else:
        if args.cache_dir_vae is None:
            args.cache_dir_vae = os.path.join(args.output_dir, "cache_vae")
        if args.cache_dir_text is None:
            args.cache_dir_text = os.path.join(args.output_dir, "cache_text")
        os.makedirs([args.cache_dir_vae, args.cache_dir_text], exist_ok=True)
    logger.info(f"VAE Cache location: {args.cache_dir_vae}")
    logger.info(f"Text Cache location: {args.cache_dir_text}")

    if args.validation_resolution < 128:
        raise ValueError(
            "It seems that the value for --validation_resolution is less than 128 pixels, which is invalid."
            f" You might have accidentally set it in megapixels: {args.validation_resolution}"
        )
    if args.seen_state_path is None:
        raise ValueError(
            'Please specify a path to a "seen" state dict via the --seen_state_path parameter.'
        )
    if args.state_path is None:
        raise ValueError(
            "Please specify a location of your training state status file via the --state_path parameter."
        )
    if args.caption_dropout_interval > 100:
        raise ValueError(
            "Please specify a caption dropout interval equal to or less than 100 via the --caption_dropout_interval parameter."
        )
    if args.timestep_bias_portion < 0.0 or args.timestep_bias_portion > 1.0:
        raise ValueError("Timestep bias portion must be between 0.0 and 1.0.")

    return args
