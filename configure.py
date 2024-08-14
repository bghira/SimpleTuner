import os
import huggingface_hub
import torch
from helpers.training import quantised_precision_levels

model_classes = {
    "full": [
        "flux",
        "sdxl",
        "pixart_sigma",
        "kolors",
        "stable_diffusion_3",
        "stable_diffusion_legacy",
    ],
    "lora": ["flux", "sdxl", "kolors", "stable_diffusion_3", "stable_diffusion_legacy"],
    "controlnet": ["sdxl", "stable_diffusion_legacy"],
}

default_models = {
    "flux": "black-forest-labs/FLUX.1-dev",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "pixart_sigma": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    "kolors": "kwai-kolors/kolors-diffusers",
    "terminus": "ptx0/terminus-xl-velocity-v2",
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
}

lora_ranks = [1, 16, 64, 128, 256]
learning_rates_by_rank = {
    1: "3e-4",
    16: "1e-4",
    64: "8e-5",
    128: "6e-5",
    256: "5.09e-5",
}

optim_compatibility = {
    "bf16": ["adamw_bf16"],
    "no": ["adamw", "adafactor", "prodigy", "dadaptation"],
}


def print_config(env_contents: dict, extra_args: list):
    # env_contents["TRAINER_EXTRA_ARGS"] = " ".join(extra_args)
    # output = json.dumps(env_contents, indent=4)
    # print(output)
    pass


def prompt_user(prompt, default=None):
    if default:
        prompt = f"{prompt} (default: {default})"
    user_input = input(f"{prompt}: ")
    return user_input.strip() or default


def configure_env():
    print("Welcome to SimpleTuner!")
    print("This script will guide you through setting up your config.env file.\n")
    env_contents = {
        "RESUME_CHECKPOINT": "latest",
        "DATALOADER_CONFIG": "config/multidatabackend.json",
        "ASPECT_BUCKET_ROUNDING": 2,
        "TRAINING_SEED": 42,
        "USE_EMA": "false",
        "USE_XFORMERS": "false",
        "MINIMUM_RESOLUTION": 0,
    }
    extra_args = []

    output_dir = prompt_user(
        "Enter the directory where you want to store your outputs", "output/models"
    )
    while not os.path.exists(output_dir):
        should_create = (
            prompt_user(
                "That directory did not exist. Should I create it? Answer 'n' to select a new location. ([y]/n)",
                "y",
            )
            == "y"
        )
        if should_create:
            os.makedirs(output_dir, exist_ok=True)
        else:
            print(
                f"Directory {output_dir} does not exist. Please create it and try again."
            )
            output_dir = prompt_user(
                "Enter the directory where you want to store your outputs",
                "output/models",
            )
    env_contents["OUTPUT_DIR"] = output_dir

    # Start with the basic options
    model_type = prompt_user(
        "What type of model are you training? (Options: [lora], full)", "lora"
    ).lower()
    env_contents["USE_DORA"] = "false"
    env_contents["USE_BITFIT"] = "false"
    if model_type == "lora":
        use_dora = prompt_user(
            "Would you like to train a DoRA model? (y/[n])", "n"
        ).lower()
        if use_dora == "y":
            env_contents["USE_DORA"] = "true"
        lora_rank = None
        while lora_rank not in lora_ranks:
            if lora_rank is not None:
                print(f"Invalid LoRA rank: {lora_rank}")
            lora_rank = int(
                prompt_user(
                    f"Set the LoRA rank (Options: {', '.join([str(x) for x in lora_ranks])})",
                    "64",
                )
            )
            extra_args.append(f"--lora_rank={lora_rank}")
    elif model_type == "full":
        use_bitfit = prompt_user(
            "Would you like to train a BitFit model? (y/[n])", "n"
        ).lower()
        if use_bitfit == "y":
            env_contents["USE_BITFIT"] = "true"
        use_ema = prompt_user(
            "Would you like to use EMA for training? (y/[n])", "n"
        ).lower()
        env_contents["USE_EMA"] = "false"
        if use_ema == "y":
            env_contents["USE_EMA"] = "true"

    print("We'll try and login to Hugging Face Hub..")
    whoami = None
    try:
        whoami = huggingface_hub.whoami()
    except:
        pass
    should_retry = True
    env_contents["PUSH_TO_HUB"] = "false"
    env_contents["PUSH_CHECKPOINTS"] = "false"
    while not whoami and should_retry:
        should_retry = (
            prompt_user(
                "You are not currently logged into Hugging Face Hub. Would you like to login? (y/n)",
                "y",
            ).lower()
            == "y"
        )
        if not should_retry:
            whoami = None
            print("Will not be logged into Hugging Face Hub.")
            break
        huggingface_hub.login()
        whoami = huggingface_hub.whoami()

    finishing_count_type = prompt_user(
        "Should we schedule the end of training by epochs, or steps?", "steps"
    ).lower()
    while finishing_count_type not in ["steps", "epochs"]:
        print(f"Invalid finishing count type: {finishing_count_type}")
        finishing_count_type = prompt_user(
            "Should we schedule the end of training by epochs, or steps?", "steps"
        ).lower()
    if finishing_count_type == "steps":
        env_contents["MAX_NUM_STEPS"] = prompt_user(
            "Set the maximum number of steps", 10000
        )
        env_contents["NUM_EPOCHS"] = 0
    else:
        env_contents["NUM_EPOCHS"] = prompt_user(
            "Set the maximum number of epochs", 100
        )
        env_contents["MAX_NUM_STEPS"] = 0

    checkpointing_interval = prompt_user(
        "Set the checkpointing interval (in steps)", 500
    )
    env_contents["CHECKPOINTING_STEPS"] = int(checkpointing_interval)
    checkpointing_limit = prompt_user(
        "How many checkpoints do you want to keep? LoRA are small, and you can keep more than a full finetune.",
        5,
    )
    env_contents["CHECKPOINTING_LIMIT"] = int(checkpointing_limit)
    if whoami is not None:
        print("Connected to Hugging Face Hub as:", whoami["name"])
        should_push_to_hub = (
            prompt_user(
                "Do you want to push your model to Hugging Face Hub when it is completed uploading? (y/n)",
                "y",
            ).lower()
            == "y"
        )
        env_contents["HUB_MODEL_NAME"] = prompt_user(
            f"What do you want the name of your Hugging Face Hub model to be? This will be accessible as https://huggingface.co/{whoami['name']}/your-model-name-here",
            f"simpletuner-{model_type}",
        )
        should_push_checkpoints = False
        if should_push_to_hub:
            env_contents["PUSH_TO_HUB"] = "true"
            should_push_checkpoints = (
                prompt_user(
                    "Do you want to push intermediary checkpoints to Hugging Face Hub? ([y]/n)",
                    "y",
                ).lower()
                == "y"
            )
            if should_push_checkpoints:
                env_contents["PUSH_CHECKPOINTS"] = "true"
    report_to_wandb = (
        prompt_user(
            "Would you like to report training statistics to Weights & Biases? ([y]/n)",
            "y",
        ).lower()
        == "y"
    )
    report_to_tensorboard = (
        prompt_user(
            "Would you like to report training statistics to TensorBoard? (y/[n])", "n"
        ).lower()
        == "y"
    )
    report_to_str = ""
    if report_to_wandb or report_to_tensorboard:
        tracker_project_name = prompt_user(
            "Enter the name of your Weights & Biases project", f"{model_type}-training"
        )
        env_contents["TRACKER_PROJECT_NAME"] = tracker_project_name
        tracker_run_name = prompt_user(
            "Enter the name of your Weights & Biases runs. This can use shell commands, which can be used to dynamically set the run name.",
            "$(date +%s)",
        )
        env_contents["TRACKER_RUN_NAME"] = tracker_run_name
        report_to_str = "--report_to="
        if report_to_wandb:
            report_to_str += "wandb"
        if report_to_tensorboard:
            if report_to_wandb:
                report_to_str += ","
            report_to_str += "tensorboard"
    env_contents["DEBUG_EXTRA_ARGS"] = report_to_str

    print_config(env_contents, extra_args)

    model_class = None
    while model_class not in model_classes[model_type]:
        if model_class is not None:
            print(f"Invalid model class: {model_class}")
        model_class = prompt_user(
            f"Which model family are you training? ({'/'.join(model_classes[model_type])})",
            "flux",
        ).lower()

    can_load_model = False
    model_name = None
    while not can_load_model:
        if model_name is not None:
            print(
                "For some reason, we can not load that model. Can you check your Hugging Face login and try again?"
            )
        model_name = prompt_user(
            "Enter the model name from Hugging Face Hub", default_models[model_class]
        )
        try:
            model_info = huggingface_hub.model_info(model_name)
            if hasattr(model_info, "id"):
                can_load_model = True
        except:
            continue
    env_contents["MODEL_TYPE"] = model_type
    env_contents["MODEL_NAME"] = model_name
    for cls in model_classes[model_type]:
        if cls == "sdxl":
            continue
        env_contents[cls.upper()] = "false"
    env_contents[model_class.upper()] = "true"
    # Flux-specific options
    if "FLUX" in env_contents and env_contents["FLUX"] == "true":
        if env_contents["MODEL_TYPE"].lower() == "lora":
            flux_targets = ["mmdit", "context", "all", "all+ffs"]
            flux_target_layers = None
            while flux_target_layers not in flux_targets:
                if flux_target_layers:
                    print(f"Invalid Flux target layers: {flux_target_layers}")
                flux_target_layers = prompt_user(
                    f"Set Flux target layers (Options: {'/'.join(flux_targets)})",
                    "all+ffs",
                )
            env_contents["FLUX_LORA_TARGET"] = flux_target_layers

    print_config(env_contents, extra_args)

    # Additional settings
    env_contents["TRAIN_BATCH_SIZE"] = int(
        prompt_user(
            "Set the training batch size. Larger values will require larger datasets, more VRAM, and slow things down.",
            1,
        )
    )
    env_contents["USE_GRADIENT_CHECKPOINTING"] = (
        "true"
        if prompt_user(
            "Would you like to use gradient checkpointing? This saves boatloads of memory. ([y]/n)",
            "y",
        ).lower()
        == "y"
        else "false"
    )
    env_contents["GRADIENT_ACCUMULATION_STEPS"] = int(
        prompt_user(
            "Set your number of gradient accumulation steps, or 1 to disable. This linearly increases training time with higher values.",
            2,
        )
    )
    if env_contents["GRADIENT_ACCUMULATION_STEPS"] < 1:
        env_contents["GRADIENT_ACCUMULATION_STEPS"] = 1

    env_contents["CAPTION_DROPOUT_PROBABILITY"] = float(
        prompt_user("Set the caption dropout rate, or use 0.0 to disable it.", "0.1")
    )

    resolution_types = ["pixel", "area", "pixel_area"]
    env_contents["RESOLUTION_TYPE"] = None
    while env_contents["RESOLUTION_TYPE"] not in resolution_types:
        if env_contents["RESOLUTION_TYPE"]:
            print(f"Invalid resolution type: {env_contents['RESOLUTION_TYPE']}")
        env_contents["RESOLUTION_TYPE"] = prompt_user(
            "How do you want to measure dataset resolutions? 'pixel' will size images with the shorter edge, 'area' will measure in megapixels, and is great for aspect-bucketing. 'pixel_area' is a combination of these two ideas, which lets you set your area using pixels instead of megapixels.",
            "pixel_area",
        ).lower()
    if (
        env_contents["RESOLUTION_TYPE"] == "pixel"
        or env_contents["RESOLUTION_TYPE"] == "pixel_area"
    ):
        default_resolution = 1024
        resolution_unit = "pixel"
    else:
        default_resolution = 1.0
        resolution_unit = "megapixel"
    env_contents["RESOLUTION"] = prompt_user(
        f"What would you like the default resolution of your datasets to be? The default for is {env_contents['RESOLUTION_TYPE']} is {default_resolution} {resolution_unit}s.",
        default_resolution,
    )

    # remove spaces from validation resolution, ensure it's a single WxH or a comma-separated list of WxH
    env_contents["VALIDATION_SEED"] = prompt_user("Set the seed for validation", 42)
    env_contents["VALIDATION_STEPS"] = prompt_user(
        "How many steps in between validation outputs?",
        env_contents["CHECKPOINTING_STEPS"],
    )
    env_contents["VALIDATION_RESOLUTION"] = None
    while (
        env_contents["VALIDATION_RESOLUTION"] is None
        or "x" not in env_contents["VALIDATION_RESOLUTION"]
    ):
        if env_contents["VALIDATION_RESOLUTION"] is not None:
            print(
                "Invalid resolution format. Please enter a single resolution, or a comma-separated list. Example: 1024x1024,1280x768"
            )
        env_contents["VALIDATION_RESOLUTION"] = prompt_user(
            "Set the validation resolution. Format could be a single resolution, or comma-separated.",
            "1024x1024",
        )
        env_contents["VALIDATION_RESOLUTION"] = ",".join(
            [x.strip() for x in env_contents["VALIDATION_RESOLUTION"].split(",")]
        )
    env_contents["VALIDATION_GUIDANCE"] = prompt_user(
        "Set the guidance scale for validation", "7.5"
    )
    env_contents["VALIDATION_GUIDANCE_RESCALE"] = prompt_user(
        "Set the guidance re-scale for validation - this is called dynamic thresholding and is used mostly for zero-terminal SNR models.",
        "0.0",
    )
    env_contents["VALIDATION_NUM_INFERENCE_STEPS"] = prompt_user(
        "Set the number of inference steps for validation", "20"
    )
    env_contents["VALIDATION_PROMPT"] = prompt_user(
        "Set the validation prompt", "A photo-realistic image of a cat"
    )
    print_config(env_contents, extra_args)

    # Advanced options
    env_contents["ALLOW_TF32"] = "false"
    if torch.cuda.is_available():
        use_tf32 = (
            prompt_user("Would you like to enable TF32 mode? (y/n)", "n").lower() == "y"
        )
        if use_tf32:
            env_contents["ALLOW_TF32"] = "true"
    mixed_precision_options = ["bf16", "no"]
    env_contents["MIXED_PRECISION"] = None
    while (
        not env_contents["MIXED_PRECISION"]
        or env_contents["MIXED_PRECISION"] not in mixed_precision_options
    ):
        if env_contents["MIXED_PRECISION"]:
            print(f"Invalid mixed precision option: {env_contents['MIXED_PRECISION']}")
        env_contents["MIXED_PRECISION"] = prompt_user(
            "Set mixed precision mode (Options: bf16, no (fp32))", "bf16"
        )
    compatible_optims = optim_compatibility[env_contents["MIXED_PRECISION"]]
    env_contents["OPTIMIZER"] = None
    while (
        not env_contents["OPTIMIZER"]
        or env_contents["OPTIMIZER"] not in compatible_optims
    ):
        if env_contents["OPTIMIZER"]:
            print(f"Invalid optimizer: {env_contents['OPTIMIZER']}")
        env_contents["OPTIMIZER"] = prompt_user(
            f"Choose an optimizer (Options: {'/'.join(compatible_optims)})",
            compatible_optims[0],
        )

    lr_schedulers = ["polynomial", "constant"]
    lr_scheduler = None
    while lr_scheduler not in lr_schedulers:
        if lr_scheduler:
            print(f"Invalid learning rate scheduler: {lr_scheduler}")
        lr_scheduler = prompt_user(
            f"Set the learning rate scheduler. Options: {'/'.join(lr_schedulers)}",
            lr_schedulers[0],
        )
    learning_rate = prompt_user(
        "Set the learning rate",
        (
            learning_rates_by_rank[lora_rank]
            if model_type == "lora"
            else 1.0 if env_contents["OPTIMIZER"] == "prodigy" else "1e-6"
        ),
    )
    lr_warmup_steps = prompt_user(
        "Set the number of warmup steps before the learning rate reaches its peak. This is set to 10 percent of the total runtime by default, or 100 steps, whichever is higher.",
        min(100, int(env_contents["MAX_NUM_STEPS"]) // 10),
    )
    env_contents["LEARNING_RATE"] = learning_rate
    env_contents["LR_SCHEDULE"] = lr_scheduler
    if lr_scheduler == "polynomial":
        extra_args.append("--lr_end=1e-8")
    env_contents["LR_WARMUP_STEPS"] = lr_warmup_steps

    quantization = (
        prompt_user(
            "Would you like to enable model quantization? NOTE: Currently, a bug prevents resuming training once it has crashed, when quantisation is enabled. (y/n)",
            "n",
        ).lower()
        == "y"
    )
    if quantization:
        if env_contents["USE_DORA"] == "true":
            print("DoRA will be disabled for quantisation.")
            env_contents["USE_DORA"] = "false"
        quantization_type = None
        while (
            not quantization_type or quantization_type not in quantised_precision_levels
        ):
            if quantization_type:
                print(f"Invalid quantization type: {quantization_type}")
            quantization_type = prompt_user(
                f"Choose quantization type (Options: {'/'.join(quantised_precision_levels)})",
                "int8-quanto",
            )
        extra_args.append(f"--base_model_precision={quantization_type}")
    print_config(env_contents, extra_args)
    gradient_precision_levels = ["unmodified", "fp32"]
    gradient_precision = None
    while gradient_precision not in gradient_precision_levels:
        if gradient_precision:
            print(f"Invalid gradient precision: {gradient_precision}")
        gradient_precision = prompt_user(
            "Set gradient precision, which might be required if you have accumulation steps enabled (Options: fp32, unmodified)",
            "unmodified",
        )
    extra_args.append(f"--gradient_precision={gradient_precision}")
    compress_disk_cache = (
        prompt_user("Would you like to compress the disk cache? (y/n)", "y").lower()
        == "y"
    )
    if compress_disk_cache:
        extra_args.append("--compress_disk_cache")

    # multi-gpu training
    env_contents["ACCELERATE_EXTRA_ARGS"] = ""
    env_contents["TRAINING_NUM_PROCESSES"] = prompt_user(
        "How many GPUs will you be training on?", 1
    )
    if env_contents["TRAINING_NUM_PROCESSES"] > 1:
        env_contents["ACCELERATE_EXTRA_ARGS"] = "--multi_gpu"
    env_contents["TRAINING_NUM_MACHINES"] = 1

    # torch compile
    torch_compile = (
        prompt_user("Would you like to use torch compile? (y/n)", "n").lower() == "y"
    )
    env_contents["VALIDATION_TORCH_COMPILE"] = "false"
    env_contents["TRAINER_DYNAMO_BACKEND"] = "no"
    if torch_compile:
        env_contents["VALIDATION_TORCH_COMPILE"] = "true"
        env_contents["TRAINER_DYNAMO_BACKEND"] = (
            "inductor" if torch.cuda.is_available() else "aot_eager"
        )

    # Summary and confirmation
    print_config(env_contents, extra_args)
    confirm = prompt_user("Does this look correct? (y/n)", "y").lower() == "y"

    if confirm:
        # Write to .env file
        trainer_extra_args_str = " ".join(extra_args)
        env_contents["TRAINER_EXTRA_ARGS"] = trainer_extra_args_str
        with open("config/config.env", "w") as env_file:
            for key, value in env_contents.items():
                env_file.write(f"{key}='{value}'\n")

        print("\nConfiguration file created successfully!")
    else:
        print("\nConfiguration aborted. No changes were made.")


if __name__ == "__main__":
    configure_env()
