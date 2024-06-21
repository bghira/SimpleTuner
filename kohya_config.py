"""khoya_config.py

This script is meant to convert a Kohya configuration file to a SimpleTuner command-line arguments.

Execute it as follows:

    python kohya_config.py --config_path /path/to/kohya_config.json [--pretty]

If --pretty is specified, the output will be formatted with line breaks and trailing slashes.

"""

import json
import argparse
import logging
from helpers import log_format

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format=log_format)
logger = logging.getLogger("kohya_config")

# Default script name assumption
DEFAULT_SCRIPT_NAME = "train_sd2x.py"

ARG_MAPPING = {
    "LoRA_type": "lora_type",
    "adaptive_noise_scale": {
        "warning": "--adaptive_noise_scale is not currently supported."
    },
    "additional_parameters": {"warning": "--additional_parameters is not supported."},
    "block_alphas": None,
    "block_dims": None,
    "block_lr_zero_threshold": None,
    "bucket_no_upscale": {
        "info": "--bucket_no_upscale: use multidatabackend.json to configure these values."
    },
    "bucket_reso_steps": {
        "info": "--bucket_reso_steps may be approximated through max_image_size and target_downsample_size in multidatabackend.json."
    },
    "cache_latents": {"info": "--cache_latents: latents are always cached."},
    "cache_latents_to_disk": None,
    "caption_dropout_every_n_epochs": {
        "info": "--caption_dropout_every_n_epochs will do nothing. caption dropout is all the time or none of the time."
    },
    "caption_dropout_rate": {
        "parameter": "caption_dropout_probability",
        "range": [0, 1],
    },
    "caption_extension": {
        "info": "--caption_extension: use caption_strategy value inside multidatabackend.json. custom file extensions are not supported."
    },
    "clip_skip": {"warning": "--clip_skip is not currently supported."},
    "color_aug": {"warning": "--color_aug is not currently supported"},
    "conv_alpha": {"warning": "--conv_alpha is not currently supported"},
    "conv_block_alphas": {"warning": "--conv_block_alphas is not currently supported"},
    "conv_block_dims": {"warning": "--conv_block_dims is not currently supported"},
    "conv_dim": {"warning": "--conv_dim is not currently supported"},
    "decompose_both": {"warning": "--decompose_both is not currently supported"},
    "dim_from_weights": {"warning": "--dim_from_weights is not currently supported."},
    "down_lr_weight": {"warning": "--down_lr_weight is not currently supported."},
    "enable_bucket": {
        "info": "--enable_bucket: use multidatabackend.json to configure these values per-dataset."
    },
    "epoch": {
        "info": "--epoch is not used in SimpleTuner, use --num_train_epochs instead."
    },
    "factor": {"warning": "--factor is not supported."},
    "flip_aug": {"warning": "--flip_aug is not supported."},
    "full_bf16": [
        {"parameter": "mixed_precision", "value": "bf16", "ignore_if_value": False},
        {"parameter": "vae_dtype", "value": "bf16", "ignore_if_value": False},
    ],
    "full_fp16": {
        "warning": "full_fp16 is not supported in SimpleTuner, use `--pure_bf16` instead."
    },
    "gradient_accumulation_steps": "gradient_accumulation_steps",
    "gradient_checkpointing": "gradient_checkpointing",
    "keep_tokens": {"warning": "--keep_tokens is not currently supported."},
    "learning_rate": "learning_rate",
    "logging_dir": "logging_dir",
    "lora_network_weights": {
        "warning": "--lora_network_weights is not currently supported."
    },
    "lr_scheduler": "lr_scheduler",
    "lr_scheduler_num_cycles": {"parameter": "lr_num_cycles", "ignore_if_value": ""},
    "lr_scheduler_power": {"parameter": "lr_power", "ignore_if_value": ""},
    "lr_warmup": "lr_warmup_steps",
    "max_bucket_reso": {
        "info": "--max_bucket_reso may be approximated through max_image_size in multidatabackend.json."
    },
    "max_data_loader_n_workers": {
        "warning": "--max_data_loader_n_workers is not currently supported."
    },
    "max_resolution": {
        "info": "--max_resolution may be approximated through max_image_size in multidatabackend.json."
    },
    "max_timestep": [
        {
            "parameter": "timestep_bias_strategy",
            "value": "range",
        },
        {"parameter": "timestep_bias_end", "value": "value_passthrough"},
    ],
    "max_token_length": {"warning": "--max_token_length is not currently supported."},
    "max_train_epochs": "num_train_epochs",
    "mem_eff_attn": {
        "warning": "--mem_eff_attn: use --enable_xformers_memory_efficient_attention instead."
    },
    "mid_lr_weight": {"warning": "--mid_lr_weight is not currently supported."},
    "min_bucket_reso": {
        "warning": "--min_bucket_reso: use multidatabackend.json to configure these values."
    },
    "min_snr_gamma": "snr_gamma",
    "min_timestep": [
        {
            "parameter": "timestep_bias_strategy",
            "value": "range",
        },
        {"parameter": "timestep_bias_begin", "value": "value_passthrough"},
    ],
    "mixed_precision": {"parameter": "mixed_precision", "ignore_if_value": False},
    "model_list": {
        "warning": "--model_list is not currently supported. use --pretrained_model_name_or_path instead."
    },
    "module_dropout": {"warning": "--module_dropout is not currently supported."},
    "multires_noise_discount": {
        "warning": "--multires_noise_discount is not currently supported."
    },
    "multires_noise_iterations": {
        "warning": "--multires_noise_iterations is not currently supported."
    },
    "network_alpha": {"warning": "--network_alpha is not currently supported."},
    "network_dim": {"warning": "--network_dim is not currently supported."},
    "network_dropout": {"warning": "--network_dropout is not currently supported."},
    "no_token_padding": {"warning": "--no_token_padding is not currently supported."},
    "noise_offset": "noise_offset",
    "noise_offset_type": {
        "warning": "--noise_offset_type is not currently supported other than 'Original'",
        "warn_if_not_value": "Original",
    },
    "num_cpu_threads_per_process": None,
    "optimizer": {
        "mapping": {
            "AdamW": "",
            "AdamW8bit": "use_8bit_adam",
            "Dadapt": "use_dadapt_optimizer",
            # SGD is not supported.
            "SGDNesterov": None,
            "Adafactor": "use_adafactor_optimizer",
        }
    },
    "optimizer_args": {
        "warning": "--optimizer_args: due to the complexity, this is not supported by this tool."
    },
    "output_dir": "output_dir",
    "output_name": None,
    "persistent_data_loader_workers": {
        "info": "--persistent_data_loader_workers: workers are always persistent."
    },
    "pretrained_model_name_or_path": "pretrained_model_name_or_path",
    "prior_loss_weight": {"warning": "prior_loss_weight is not currently supported."},
    "random_crop": {"parameter": "crop_style", "value": "random"},
    "rank_dropout": "lora_dropout",
    "reg_data_dir": {
        "warning": "--reg_data_dir: use multidatabackend.json to configure datasets. Regularization data is recommended to be added as a separate dataset with a low probability value."
    },
    "repeats": {
        "warning": "In SimpleTuner, repeats is a dataset-specific configuration with a notable difference that Kohya's scripts treat repeats as how many times the whole dataset is seen, but SimpleTuner treats this value as how many *additional* times the dataset will be seen. Subtract 1 from your Kohya repeats value to determine the correct value for SimpleTuner."
    },
    "resume": {"warning": "SimpleTuner can not resume from Kohya training states."},
    "sample_every_n_epochs": {
        "warning": "sample_every_n_epochs is not currently supported."
    },
    "sample_every_n_steps": {"parameter": "validation_steps", "ignore_if_value": 0},
    "sample_prompts": "validation_prompt",
    "sample_sampler": "validation_noise_scheduler",
    "save_every_n_epochs": {
        "warning": "save_every_n_epochs is not currently supported.",
        "ignore_if_value": 0,
    },
    "save_every_n_steps": {
        "parameter": "checkpointing_steps",
        "ignore_if_value": 0,
        "warning": "--save_every_n_steps should be supplied with a value greater than zero",
        "warn_if_value": 0,
    },
    "save_last_n_steps": None,
    "save_last_n_steps_state": None,
    "save_model_as": {
        "warning": "All models are saved as safetensors files in SimpleTuner. --save_model_as will do nothing.",
        "ignore_if_value": "safetensors",
    },
    "save_precision": {
        "info": "All weights are always in float32 for SimpleTuner. --save_precision will do nothing."
    },
    "save_state": {"info": "--save_state: states are always saved."},
    "scale_v_pred_loss_like_noise_pred": {
        "warning": "scale_v_pred_loss_like_noise_pred is not currently supported."
    },
    "scale_weight_norms": {
        "parameter": "max_grad_norm",
        "warn_if_value": 0,
        "warning": "In SimpleTuner, max_grad_norm is set to 2 by default. Please change this if you see issues.",
    },
    "sdxl": {"script_name": "train_sdxl.py"},
    "sdxl_cache_text_encoder_outputs": {
        "warning": "--sdxl_cache_text_encoder_outputs is not currently supported. Text encoder outputs are always cached."
    },
    "sdxl_no_half_vae": {
        "parameter": "vae_dtype",
        "value": "fp32",
        "ignore_if_value": False,
    },
    "seed": "seed",
    "shuffle_caption": {"warning": "shuffle_caption is not currently supported."},
    "stop_text_encoder_training": [
        {"parameter": "freeze_encoder", "value": True, "ignore_if_value": 0.0},
        {
            "parameter": "text_encoder_limit",
            "value": "value_passthrough",
            "ignore_if_value": 0.0,
        },
    ],
    "text_encoder_lr": {"parameter": "text_encoder_lr", "ignore_if_value": 0.0},
    "train_batch_size": "train_batch_size",
    "train_data_dir": {
        "error": "This parameter is not used in SimpleTuner. You must configure multidatabackend.json, and use that instead."
    },
    "train_on_input": None,
    "training_comment": None,
    "unet_lr": "learning_rate",
    "unit": None,
    "up_lr_weight": None,
    "use_cp": None,
    "use_wandb": {"parameter": "report_to", "value": "wandb", "ignore_if_value": False},
    "v2": {"script_name": "train_sd2x.py", "ignore_if_value": False},
    "v_parameterization": {"parameter": "prediction_type", "value": "v_prediction"},
    "vae_batch_size": {"parameter": "vae_batch_size", "range": [1, 128]},
    "wandb_api_key": {
        "error": "Using --wandb_api_key in Kohya or SimpleTuner is considered insecure. Use `wandb login` instead."
    },
    "weighted_captions": None,
    "xformers": {
        "parameter": "enable_xformers_memory_efficient_attention",
        "value": True,
        "ignore_if_value": False,
    },
}


class KoyhaConfigToSimpleTunerArgs:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """Load the Koyha configuration from the specified JSON file."""
        with open(self.config_path, "r") as file:
            return json.load(file)

    def process_mapping(self, pretty: bool = False):
        """Process the ARG_MAPPING to generate SimpleTuner command-line arguments."""
        args = []
        for key, value in self.config.items():
            mapping = ARG_MAPPING.get(
                key, {"warning": f"Key {key} is not currently supported."}
            )
            if mapping is None:
                continue
            if isinstance(mapping, dict):
                # Conditional logic or warnings/infos
                if "warning" in mapping and (
                    (value != 0 and value is not None and value != "")
                    or (
                        "warn_if_not_value" in mapping
                        and value != mapping["warn_if_not_value"]
                    )
                    or (
                        "warn_if_value" in mapping and value == mapping["warn_if_value"]
                    )
                ):
                    logger.warning(f"{key}: {mapping['warning']}")
                if "ignore_if_value" in mapping and value == mapping["ignore_if_value"]:
                    continue

                if "info" in mapping:
                    logger.info(f"{key}: {mapping['info']}")
                elif (
                    "error" in mapping
                    and value != 0
                    and value is not None
                    and value != ""
                ):
                    logger.error(f"{key}: {mapping['error']}")
                elif "parameter" in mapping:
                    # Direct mapping with a parameter change
                    if "range" in mapping and (
                        value < mapping["range"][0] or value > mapping["range"][1]
                    ):
                        raise Exception(
                            f"{key}: Value {value} is out of range {mapping['range']}."
                        )
                    if "value" in mapping:
                        value = mapping["value"]
                    args.append(
                        self.format_arg(mapping["parameter"], value, pretty=pretty)
                    )
                elif "script_name" in mapping:
                    # Change the script name based on the key
                    global DEFAULT_SCRIPT_NAME
                    DEFAULT_SCRIPT_NAME = mapping["script_name"]
            elif isinstance(mapping, str) and type(value) is int:
                args.append(self.format_arg(mapping, value, pretty=pretty))
            elif (
                isinstance(mapping, str) and type(value) is str and not value.isdigit()
            ):
                # Direct mapping
                args.append(self.format_arg(mapping, f"'{value}'", pretty=pretty))
            elif isinstance(mapping, str):
                # Direct mapping
                args.append(self.format_arg(mapping, value, pretty=pretty))
            elif isinstance(mapping, list):
                # Handling complex mappings like full_bf16
                for item in mapping:
                    if "ignore_if_value" in item and value == item["ignore_if_value"]:
                        continue
                    if "value" in item and item["value"] == "value_passthrough":
                        args.append(
                            self.format_arg(item["parameter"], value, pretty=pretty)
                        )
                    elif "value" not in item:
                        args.append(
                            self.format_arg(item["parameter"], value, pretty=pretty)
                        )
                    else:
                        args.append(
                            self.format_arg(
                                item["parameter"], item["value"], pretty=pretty
                            )
                        )
            elif mapping is None:
                # Explicitly ignored keys
                continue
        # Remove duplicate keys
        args = list(set(args))

        # Sort the keys
        args.sort()

        # Remove the trailing "/"
        args[-1] = args[-1].replace(" \\ \n", "")
        return args

    def format_arg(self, arg, value, pretty: bool = False):
        """Format the argument for command-line usage."""
        if pretty:
            return f"--{arg}={value} \\\n"
        return f"--{arg}={value}"

    def generate_command(self, pretty: bool = False):
        """Generate the SimpleTuner command line based on the processed Koyha config."""
        args = self.process_mapping(pretty=pretty)
        cmd = f"python {DEFAULT_SCRIPT_NAME} " + " ".join(args)
        return cmd


def parse_args():
    """Parse command-line arguments to specify the Koyha configuration file path."""
    parser = argparse.ArgumentParser(
        description="Convert Koyha config to SimpleTuner args."
    )
    parser.add_argument(
        "--config_path", type=str, help="Path to the Koyha configuration JSON file."
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty print the output."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    converter = KoyhaConfigToSimpleTunerArgs(args.config_path)
    command = converter.generate_command(pretty=args.pretty)
    print(f"\n{command}")
