import os
import logging
import json
import torch
from helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

licenses = {
    "flux": "flux-1-dev-non-commercial-license",
    "sdxl": "creativeml-openrail-m",
    "legacy": "openrail++",
    "pixart_sigma": "openrail++",
    "kolors": "apache-2.0",
    "smoldit": "apache-2.0",
    "sd3": "stabilityai-ai-community",
}
allowed_licenses = [
    "apache-2.0",
    "mit",
    "openrail",
    "bigscience-openrail-m",
    "creativeml-openrail-m",
    "bigscience-bloom-rail-1.0",
    "bigcode-openrail-m",
    "afl-3.0",
    "artistic-2.0",
    "bsl-1.0",
    "bsd",
    "bsd-2-clause",
    "bsd-3-clause",
    "bsd-3-clause-clear",
    "c-uda",
    "cc",
    "cc0-1.0",
    "cc-by-2.0",
    "cc-by-2.5",
    "cc-by-3.0",
    "cc-by-4.0",
    "cc-by-sa-3.0",
    "cc-by-sa-4.0",
    "cc-by-nc-2.0",
    "cc-by-nc-3.0",
    "cc-by-nc-4.0",
    "cc-by-nd-4.0",
    "cc-by-nc-nd-3.0",
    "cc-by-nc-nd-4.0",
    "cc-by-nc-sa-2.0",
    "cc-by-nc-sa-3.0",
    "cc-by-nc-sa-4.0",
    "cdla-sharing-1.0",
    "cdla-permissive-1.0",
    "cdla-permissive-2.0",
    "wtfpl",
    "ecl-2.0",
    "epl-1.0",
    "epl-2.0",
    "etalab-2.0",
    "eupl-1.1",
    "agpl-3.0",
    "gfdl",
    "gpl",
    "gpl-2.0",
    "gpl-3.0",
    "lgpl",
    "lgpl-2.1",
    "lgpl-3.0",
    "isc",
    "lppl-1.3c",
    "ms-pl",
    "apple-ascl",
    "mpl-2.0",
    "odc-by",
    "odbl",
    "openrail++",
    "osl-3.0",
    "postgresql",
    "ofl-1.1",
    "ncsa",
    "unlicense",
    "zlib",
    "pddl",
    "lgpl-lr",
    "deepfloyd-if-license",
    "llama2",
    "llama3",
    "llama3.1",
    "gemma",
    "unknown",
    "other",
    "array",
]
for _model, _license in licenses.items():
    if _license not in allowed_licenses:
        licenses[_model] = "other"


def _model_imports(args):
    output = "import torch\n"
    output += "from diffusers import DiffusionPipeline"
    if "lycoris" == args.lora_type.lower() and "lora" in args.model_type:
        output += "\nfrom lycoris import create_lycoris_from_weights"

    return f"{output}"


def ema_info(args):
    if args.use_ema:
        ema_information = """
## Exponential Moving Average (EMA)

SimpleTuner generates a safetensors variant of the EMA weights and a pt file.

The safetensors file is intended to be used for inference, and the pt file is for continuing finetuning.

The EMA model may provide a more well-rounded result, but typically will feel undertrained compared to the full model as it is a running decayed average of the model weights.
"""
        return ema_information
    return ""


def lycoris_download_info():
    """output a function to download the adapter"""
    output_fn = """
def download_adapter(repo_id: str):
    import os
    from huggingface_hub import hf_hub_download
    adapter_filename = "pytorch_lora_weights.safetensors"
    cache_dir = os.environ.get('HF_PATH', os.path.expanduser('~/.cache/huggingface/hub/models'))
    cleaned_adapter_path = repo_id.replace("/", "_").replace("\\\\", "_").replace(":", "_")
    path_to_adapter = os.path.join(cache_dir, cleaned_adapter_path)
    path_to_adapter_file = os.path.join(path_to_adapter, adapter_filename)
    os.makedirs(path_to_adapter, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id, filename=adapter_filename, local_dir=path_to_adapter
    )

    return path_to_adapter_file
    """

    return output_fn


def _model_component_name(args):
    model_component_name = "pipeline.transformer"
    if args.model_family in ["sdxl", "kolors", "legacy", "deepfloyd"]:
        model_component_name = "pipeline.unet"

    return model_component_name


def _model_load(args, repo_id: str = None):
    model_component_name = _model_component_name(args)
    hf_user_name = StateTracker.get_hf_username()
    if hf_user_name is not None:
        repo_id = f"{hf_user_name}/{repo_id}" if hf_user_name else repo_id
    if "lora" in args.model_type:
        if args.lora_type.lower() == "standard":
            output = (
                f"model_id = '{args.pretrained_model_name_or_path}'"
                f"\nadapter_id = '{repo_id if repo_id is not None else args.output_dir}'"
                f"\npipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype={StateTracker.get_weight_dtype()}) # loading directly in bf16"
                f"\npipeline.load_lora_weights(adapter_id)"
            )
        elif args.lora_type.lower() == "lycoris":
            output = (
                f"{lycoris_download_info()}"
                f"\nmodel_id = '{args.pretrained_model_name_or_path}'"
                f"\nadapter_repo_id = '{repo_id if repo_id is not None else args.output_dir}'"
                f"\nadapter_filename = 'pytorch_lora_weights.safetensors'"
                f"\nadapter_file_path = download_adapter(repo_id=adapter_repo_id)"
                f"\npipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype={StateTracker.get_weight_dtype()}) # loading directly in bf16"
                "\nlora_scale = 1.0"
            )
    else:
        output = (
            f"model_id = '{repo_id if repo_id else os.path.join(args.output_dir, 'pipeline')}'"
            f"\npipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype={StateTracker.get_weight_dtype()}) # loading directly in bf16"
        )
    if args.model_type == "lora" and args.lora_type.lower() == "lycoris":
        output += f"\nwrapper, _ = create_lycoris_from_weights(lora_scale, adapter_file_path, {model_component_name})"
        output += "\nwrapper.merge_to()"

    return output


def _torch_device():
    return """'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'"""


def _negative_prompt(args, in_call: bool = False):
    if args.model_family.lower() == "flux":
        return ""
    if not in_call:
        return f"negative_prompt = '{args.validation_negative_prompt}'"
    return "\n    negative_prompt=negative_prompt,"


def _guidance_rescale(args):
    if args.model_family.lower() in ["sd3", "flux", "pixart_sigma"]:
        return ""
    return f"\n    guidance_rescale={args.validation_guidance_rescale},"


def _skip_layers(args):
    if (
        args.model_family.lower() not in ["sd3"]
        or args.validation_guidance_skip_layers is None
    ):
        return ""
    return f"\n    skip_guidance_layers={args.validation_guidance_skip_layers},"


def _pipeline_move_to(args):
    output = f"pipeline.to({_torch_device()}) # the pipeline is already in its target precision level"

    return output


def _pipeline_quanto(args):
    # return some optional lines to run Quanto on the model pipeline
    if args.model_type == "full":
        return ""
    model_component_name = _model_component_name(args)
    comment_character = ""
    was_quantised = "The model was quantised during training, and so it is recommended to do the same during inference time."
    if args.base_model_precision == "no_change":
        comment_character = "#"
        was_quantised = "The model was not quantised during training, so it is not necessary to quantise it during inference time."
    output = f"""
## Optional: quantise the model to save on vram.
## Note: {was_quantised}
{comment_character}from optimum.quanto import quantize, freeze, qint8
{comment_character}quantize({model_component_name}, weights=qint8)
{comment_character}freeze({model_component_name})
    """

    return output


def _validation_resolution(args):
    if args.validation_resolution == "" or args.validation_resolution is None:
        return f"width=1024,\n" f"    height=1024,"
    resolutions = [args.validation_resolution]
    if "," in args.validation_resolution:
        # split the resolution into a list of resolutions
        resolutions = args.validation_resolution.split(",")
    for resolution in resolutions:
        if "x" in resolution:
            return (
                f"width={resolution.split('x')[0]},\n"
                f"    height={resolution.split('x')[1]},"
            )
        return f"width={resolution},\n" f"    height={resolution},"


def code_example(args, repo_id: str = None):
    """Return a string with the code example."""
    code_example = f"""
```python
{_model_imports(args)}

{_model_load(args, repo_id)}

prompt = "{args.validation_prompt if args.validation_prompt else 'An astronaut is riding a horse through the jungles of Thailand.'}"
{_negative_prompt(args)}
{_pipeline_quanto(args)}
{_pipeline_move_to(args)}
image = pipeline(
    prompt=prompt,{_negative_prompt(args, in_call=True) if args.model_family.lower() != 'flux' else ''}
    num_inference_steps={args.validation_num_inference_steps},
    generator=torch.Generator(device={_torch_device()}).manual_seed({args.validation_seed or args.seed or 42}),
    {_validation_resolution(args)}
    guidance_scale={args.validation_guidance},{_guidance_rescale(args)}{_skip_layers(args)}
).images[0]
image.save("output.png", format="PNG")
```
"""
    return code_example


def model_type(args):
    if "lora" in args.model_type:
        if "standard" == args.lora_type.lower():
            return "standard PEFT LoRA"
        if "lycoris" == args.lora_type.lower():
            return "LyCORIS adapter"
    else:
        return "full rank finetune"


def lora_info(args):
    """Return a string with the LORA information."""
    if "lora" not in args.model_type:
        return ""
    if args.lora_type.lower() == "standard":
        return f"""- LoRA Rank: {args.lora_rank}
- LoRA Alpha: {args.lora_alpha}
- LoRA Dropout: {args.lora_dropout}
- LoRA initialisation style: {args.lora_init_type}
    """
    if args.lora_type.lower() == "lycoris":
        lycoris_config_file = args.lycoris_config
        # read the json file
        with open(lycoris_config_file, "r") as file:
            try:
                lycoris_config = json.load(file)
            except:
                lycoris_config = {"error": "could not locate or load LyCORIS config."}
        return f"""### LyCORIS Config:\n```json\n{json.dumps(lycoris_config, indent=4)}\n```"""


def model_card_note(args):
    """Return a string with the model card note."""
    note_contents = args.model_card_note if args.model_card_note else ""
    if note_contents is None or note_contents == "":
        return ""
    return f"\n**Note:** {note_contents}\n"


def flux_schedule_info(args):
    if args.model_family.lower() != "flux":
        return ""
    output_args = []
    if args.flux_fast_schedule:
        output_args.append("flux_fast_schedule")
    if args.flow_schedule_auto_shift:
        output_args.append("flow_schedule_auto_shift")
    if args.flow_schedule_shift is not None:
        output_args.append(f"shift={args.flow_schedule_shift}")
    output_args.append(f"flux_guidance_mode={args.flux_guidance_mode}")
    if args.flux_guidance_value:
        output_args.append(f"flux_guidance_value={args.flux_guidance_value}")
    if args.flux_guidance_min:
        output_args.append(f"flux_guidance_min={args.flux_guidance_min}")
    if args.flux_guidance_mode == "random-range":
        output_args.append(f"flux_guidance_max={args.flux_guidance_max}")
        output_args.append(f"flux_guidance_min={args.flux_guidance_min}")
    if args.flow_use_beta_schedule:
        output_args.append(f"flow_beta_schedule_alpha={args.flow_beta_schedule_alpha}")
        output_args.append(f"flow_beta_schedule_beta={args.flow_beta_schedule_beta}")
    if args.flux_attention_masked_training:
        output_args.append("flux_attention_masked_training")
    if args.t5_padding != "unmodified":
        output_args.append(f"t5_padding={args.t5_padding}")
    output_args.append(f"flow_matching_loss={args.flow_matching_loss}")
    if (
        args.model_type == "lora"
        and args.lora_type == "standard"
        and args.flux_lora_target is not None
    ):
        output_args.append(f"flux_lora_target={args.flux_lora_target}")
    output_str = (
        f" (extra parameters={output_args})"
        if output_args
        else " (no special parameters set)"
    )

    return output_str


def sd3_schedule_info(args):
    if args.model_family.lower() != "sd3":
        return ""
    output_args = []
    if args.flow_schedule_auto_shift:
        output_args.append("flow_schedule_auto_shift")
    if args.flow_schedule_shift is not None:
        output_args.append(f"shift={args.flow_schedule_shift}")
    if args.flow_use_beta_schedule:
        output_args.append(f"flow_beta_schedule_alpha={args.flow_beta_schedule_alpha}")
        output_args.append(f"flow_beta_schedule_beta={args.flow_beta_schedule_beta}")
    if args.flow_use_uniform_schedule:
        output_args.append(f"flow_use_uniform_schedule")
    # if args.model_type == "lora" and args.lora_type == "standard":
    #     output_args.append(f"flux_lora_target={args.flux_lora_target}")
    output_str = (
        f" (extra parameters={output_args})"
        if output_args
        else " (no special parameters set)"
    )

    return output_str


def ddpm_schedule_info(args):
    """Information about DDPM schedules, eg. rescaled betas or offset noise"""
    output_args = []
    if args.snr_gamma:
        output_args.append(f"snr_gamma={args.snr_gamma}")
    if args.use_soft_min_snr:
        output_args.append(f"use_soft_min_snr")
        if args.soft_min_snr_sigma_data:
            output_args.append(
                f"soft_min_snr_sigma_data={args.soft_min_snr_sigma_data}"
            )
    if args.rescale_betas_zero_snr:
        output_args.append(f"rescale_betas_zero_snr")
    if args.offset_noise:
        output_args.append(f"offset_noise")
        output_args.append(f"noise_offset={args.noise_offset}")
        output_args.append(f"noise_offset_probability={args.noise_offset_probability}")
    output_args.append(
        f"training_scheduler_timestep_spacing={args.training_scheduler_timestep_spacing}"
    )
    output_args.append(
        f"inference_scheduler_timestep_spacing={args.inference_scheduler_timestep_spacing}"
    )
    output_str = (
        f" (extra parameters={output_args})"
        if output_args
        else " (no special parameters set)"
    )

    return output_str


def model_schedule_info(args):
    if args.model_family == "flux":
        return flux_schedule_info(args)
    if args.model_family == "sd3":
        return sd3_schedule_info(args)
    else:
        return ddpm_schedule_info(args)


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = "",
    train_text_encoder: bool = False,
    prompt: str = "",
    validation_prompts: list = None,
    validation_shortnames: list = None,
    repo_folder: str = None,
):
    if repo_folder is None:
        raise ValueError("The repo_folder must be specified and not be None.")
    if type(validation_prompts) is not list:
        raise ValueError(
            f"The validation_prompts must be a list. Received {validation_prompts}"
        )
    # if we have more than one '/' in the base_model, we will turn it into unknown/model
    model_family = StateTracker.get_model_family()
    if base_model.count("/") > 1:
        base_model = f"{model_family}/unknown-model"
    logger.debug(f"Validating from prompts: {validation_prompts}")
    assets_folder = os.path.join(repo_folder, "assets")
    optimizer_config = StateTracker.get_args().optimizer_config
    if optimizer_config is None:
        optimizer_config = ""
    os.makedirs(assets_folder, exist_ok=True)
    datasets_str = ""
    for dataset in StateTracker.get_data_backends().keys():
        if "sampler" in StateTracker.get_data_backends()[dataset]:
            datasets_str += f"### {dataset}\n"
            datasets_str += f"{StateTracker.get_data_backends()[dataset]['sampler'].log_state(show_rank=False, alt_stats=True)}"
    widget_str = ""
    idx = 0
    shortname_idx = 0
    negative_prompt_text = str(StateTracker.get_args().validation_negative_prompt)
    if negative_prompt_text == "":
        negative_prompt_text = "''"
    if images is not None and len(images) > 0:
        widget_str = "widget:"
        for image_list in images.values() if isinstance(images, dict) else images:
            if not isinstance(image_list, list):
                image_list = [image_list]
            sub_idx = 0
            for image in image_list:
                image_path = os.path.join(assets_folder, f"image_{idx}_{sub_idx}.png")
                image.save(image_path, format="PNG")
                validation_prompt = "no prompt available"
                if validation_prompts is not None:
                    try:
                        validation_prompt = validation_prompts[shortname_idx]
                    except IndexError:
                        validation_prompt = f"prompt not found ({validation_shortnames[shortname_idx] if validation_shortnames is not None and shortname_idx in validation_shortnames else shortname_idx})"
                if validation_prompt == "":
                    validation_prompt = "unconditional (blank prompt)"
                else:
                    # Escape anything that YAML won't like
                    validation_prompt = validation_prompt.replace("'", "''")
                widget_str += f"\n- text: '{validation_prompt}'"
                widget_str += "\n  parameters:"
                widget_str += f"\n    negative_prompt: '{negative_prompt_text}'"
                widget_str += "\n  output:"
                widget_str += f"\n    url: ./assets/image_{idx}_{sub_idx}.png"
                idx += 1
                sub_idx += 1

            shortname_idx += 1
    args = StateTracker.get_args()
    yaml_content = f"""---
license: {licenses.get(model_family, "other")}
base_model: "{base_model}"
tags:
  - {model_family}
  - {f'{model_family}-diffusers' if 'deepfloyd' not in args.model_type else 'deepfloyd-if-diffusers'}
  - text-to-image
  - diffusers
  - simpletuner
  - {'not-for-all-audiences' if not args.model_card_safe_for_work else 'safe-for-work'}
  - {args.model_type}
{'  - template:sd-lora' if 'lora' in args.model_type else ''}
{f'  - {args.lora_type}' if 'lora' in args.model_type else ''}
inference: true
{widget_str}
---

"""
    model_card_content = f"""# {repo_id}

This is a {model_type(args)} derived from [{base_model}](https://huggingface.co/{base_model}).

{'This is a **diffusion** model trained using DDPM objective instead of Flow matching. **Be sure to set the appropriate scheduler configuration.**' if args.model_family == "sd3" and args.flow_matching_loss == "diffusion" else ''}
{'The main validation prompt used during training was:' if prompt else 'Validation used ground-truth images as an input for partial denoising (img2img).' if args.validation_using_datasets else 'No validation prompt was used during training.'}
{'```' if prompt else ''}
{prompt}
{'```' if prompt else ''}

{model_card_note(args)}
## Validation settings
- CFG: `{StateTracker.get_args().validation_guidance}`
- CFG Rescale: `{StateTracker.get_args().validation_guidance_rescale}`
- Steps: `{StateTracker.get_args().validation_num_inference_steps}`
- Sampler: `{'FlowMatchEulerDiscreteScheduler' if args.model_family in ['sd3', 'flux'] else StateTracker.get_args().validation_noise_scheduler}`
- Seed: `{StateTracker.get_args().validation_seed}`
- Resolution{'s' if ',' in StateTracker.get_args().validation_resolution else ''}: `{StateTracker.get_args().validation_resolution}`
{f"- Skip-layer guidance: {_skip_layers(args)}" if args.model_family in ['sd3', 'flux'] else ''}

Note: The validation settings are not necessarily the same as the [training settings](#training-settings).

{'You can find some example images in the following gallery:' if images is not None else ''}\n

<Gallery />

The text encoder {'**was**' if train_text_encoder else '**was not**'} trained.
{'You may reuse the base model text encoder for inference.' if not train_text_encoder else 'If the text encoder from this repository is not used at inference time, unexpected or bad results could occur.'}


## Training settings

- Training epochs: {StateTracker.get_epoch() - 1}
- Training steps: {StateTracker.get_global_step()}
- Learning rate: {StateTracker.get_args().learning_rate}
  - Learning rate schedule: {StateTracker.get_args().lr_scheduler}
  - Warmup steps: {StateTracker.get_args().lr_warmup_steps}
- Max grad norm: {StateTracker.get_args().max_grad_norm}
- Effective batch size: {StateTracker.get_args().train_batch_size * StateTracker.get_args().gradient_accumulation_steps * StateTracker.get_accelerator().num_processes}
  - Micro-batch size: {StateTracker.get_args().train_batch_size}
  - Gradient accumulation steps: {StateTracker.get_args().gradient_accumulation_steps}
  - Number of GPUs: {StateTracker.get_accelerator().num_processes}
- Gradient checkpointing: {StateTracker.get_args().gradient_checkpointing}
- Prediction type: {'flow-matching' if (StateTracker.get_args().model_family in ["sd3", "flux"]) else StateTracker.get_args().prediction_type}{model_schedule_info(args=StateTracker.get_args())}
- Optimizer: {StateTracker.get_args().optimizer}{optimizer_config if optimizer_config is not None else ''}
- Trainable parameter precision: {'Pure BF16' if torch.backends.mps.is_available() or StateTracker.get_args().mixed_precision == "bf16" else 'FP32'}
- Caption dropout probability: {StateTracker.get_args().caption_dropout_probability * 100}%
{'- Xformers: Enabled' if StateTracker.get_args().attention_mechanism == 'xformers' else ''}
{f'- SageAttention: Enabled {StateTracker.get_args().sageattention_usage}' if StateTracker.get_args().attention_mechanism == 'sageattention' else ''}
{lora_info(args=StateTracker.get_args())}

## Datasets

{datasets_str}

## Inference

{code_example(args=StateTracker.get_args(), repo_id=repo_id)}

{ema_info(args=StateTracker.get_args())}
"""

    logger.debug(f"YAML:\n{yaml_content}")
    logger.debug(f"Model Card:\n{model_card_content}")
    with open(os.path.join(repo_folder, "README.md"), "w", encoding="utf-8") as f:
        f.write(yaml_content + model_card_content)
