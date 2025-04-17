import os
import logging
import json
import torch
from helpers.training.state_tracker import StateTracker
from typing import Union
import numpy as np
from PIL import Image
from diffusers.utils.export_utils import export_to_gif
from helpers.models.common import ModelFoundation

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

licenses = {
    "sd1x": "openrail++",
    "sd2x": "openrail++",
    "sd3": "stabilityai-ai-community",
    "sdxl": "creativeml-openrail-m",
    "flux": "flux-1-dev-non-commercial-license",
    "pixart_sigma": "openrail++",
    "kolors": "apache-2.0",
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


def _model_component_name(model):
    model_component_name = f"pipeline.{model.MODEL_TYPE.value}"

    return model_component_name


def _model_load(args, repo_id: str = None, model=None):
    model_component_name = _model_component_name(model)
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


def _guidance_rescale(model):
    # only these model families support zsnr sampling
    if model.MODEL_TYPE.value != "unet":
        return ""
    return f"\n    guidance_rescale={model.config.validation_guidance_rescale},"


def _skip_layers(args):
    if args.validation_guidance_skip_layers is None:
        return ""
    return f"\n    skip_guidance_layers={args.validation_guidance_skip_layers},"


def _pipeline_move_to(args):
    output = f"pipeline.to({_torch_device()}) # the pipeline is already in its target precision level"

    return output


def _pipeline_quanto(args, model):
    # return some optional lines to run Quanto on the model pipeline
    if args.model_type == "full":
        return ""
    model_component_name = _model_component_name(model)
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


def _output_attribute(args, model):
    if "Video" in str(type(model)):
        return "frames[0]"
    return "images[0]"


def _output_save_call(args):
    if args.model_family in ["ltxvideo", "wan"]:
        return f"""
from diffusers.utils.export_utils import export_to_gif
export_to_gif(model_output, "output.gif", fps={args.framerate})
"""
    return f"""
model_output.save("output.png", format="PNG")
"""


def code_example(args, repo_id: str = None, model=None):
    """Return a string with the code example."""
    code_example = f"""
```python
{_model_imports(args)}

{_model_load(args, repo_id, model=model)}

prompt = "{args.validation_prompt if args.validation_prompt else 'An astronaut is riding a horse through the jungles of Thailand.'}"
{_negative_prompt(args)}
{_pipeline_quanto(args, model)}
{_pipeline_move_to(args)}
model_output = pipeline(
    prompt=prompt,{_negative_prompt(args, in_call=True) if args.model_family.lower() != 'flux' else ''}
    num_inference_steps={args.validation_num_inference_steps},
    generator=torch.Generator(device={_torch_device()}).manual_seed({args.validation_seed or args.seed or 42}),
    {_validation_resolution(args)}
    guidance_scale={args.validation_guidance},{_guidance_rescale(model)}{_skip_layers(args)}
).{_output_attribute(args, model)}
{_output_save_call(args)}
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
        return f"""
### LyCORIS Config:\n```json\n{json.dumps(lycoris_config, indent=4)}\n```
"""


def model_card_note(args):
    """Return a string with the model card note."""
    note_contents = args.model_card_note if args.model_card_note else ""
    if note_contents is None or note_contents == "":
        return ""
    return f"\n**Note:** {note_contents}\n"


def save_metadata_sample(
    image_path: str,
    image: Union[Image.Image, np.ndarray, list],
):
    if isinstance(image, list):
        file_extension = "gif"
        output_path = f"{image_path}.{file_extension}"
        export_to_gif(
            image=image,
            output_gif_path=output_path,
            fps=StateTracker.get_args().framerate,
        )
    elif isinstance(image, Image.Image):
        file_extension = "png"
        output_path = f"{image_path}.{file_extension}"
        image.save(output_path, format="PNG")
    else:
        raise ValueError(f"Cannot export sample type {type(image)} yet.")

    return output_path, file_extension


def _model_card_family_tag(model_family: str):
    if model_family == "ltxvideo":
        # the hub has a hyphen.
        return "ltx-video"
    if model_family == "wan":
        return "WanPipeline"
    return model_family


def _pipeline_tag(args):
    return "text-to-image" if args.model_family not in ["ltxvideo"] else "text-to-video"


def save_model_card(
    model: ModelFoundation,
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
    datasettypes = ["image", "video"]
    for dataset in StateTracker.get_data_backends(_types=datasettypes).keys():
        if "sampler" in StateTracker.get_data_backends(_types=datasettypes)[dataset]:
            datasets_str += f"### {dataset}\n"
            datasets_str += f"{StateTracker.get_data_backends(_types=datasettypes)[dataset]['sampler'].log_state(show_rank=False, alt_stats=True)}"
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
                image_path, image_extension = save_metadata_sample(
                    image_path=os.path.join(assets_folder, f"image_{idx}_{sub_idx}"),
                    image=image,
                )
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
                widget_str += (
                    f"\n    url: ./assets/image_{idx}_{sub_idx}.{image_extension}"
                )
                idx += 1
                sub_idx += 1

            shortname_idx += 1
    args = StateTracker.get_args()
    yaml_content = f"""---
license: {model.MODEL_LICENSE}
base_model: "{base_model}"
tags:
  - {_model_card_family_tag(model_family)}
  - {f'{_model_card_family_tag(model_family)}-diffusers' if 'deepfloyd' not in args.model_type else 'deepfloyd-if-diffusers'}
  - {_pipeline_tag(args)}
  - {'image-to-image' if args.model_family not in ["ltxvideo"] else 'image-to-video'}
  - diffusers
  - simpletuner
  - {'not-for-all-audiences' if not args.model_card_safe_for_work else 'safe-for-work'}
  - {args.model_type}
{'  - template:sd-lora' if 'lora' in args.model_type else ''}
{f'  - {args.lora_type}' if 'lora' in args.model_type else ''}
pipeline_tag: {_pipeline_tag(args)}
inference: true
{widget_str}
---

"""
    model_card_content = f"""# {repo_id}

This is a {model_type(args)} derived from [{base_model}](https://huggingface.co/{base_model}).

{'The main validation prompt used during training was:' if prompt else 'Validation used ground-truth images as an input for partial denoising (img2img).' if args.validation_using_datasets else 'No validation prompt was used during training.'}
{'```' if prompt else ''}
{prompt}
{'```' if prompt else ''}

{model_card_note(args)}
## Validation settings
- CFG: `{StateTracker.get_args().validation_guidance}`
- CFG Rescale: `{StateTracker.get_args().validation_guidance_rescale}`
- Steps: `{StateTracker.get_args().validation_num_inference_steps}`
- Sampler: `{'FlowMatchEulerDiscreteScheduler' if model.PREDICTION_TYPE.value == "flow_matching" else StateTracker.get_args().validation_noise_scheduler}`
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
- Max grad {StateTracker.get_args().grad_clip_method}: {StateTracker.get_args().max_grad_norm}
- Effective batch size: {StateTracker.get_args().train_batch_size * StateTracker.get_args().gradient_accumulation_steps * StateTracker.get_accelerator().num_processes}
  - Micro-batch size: {StateTracker.get_args().train_batch_size}
  - Gradient accumulation steps: {StateTracker.get_args().gradient_accumulation_steps}
  - Number of GPUs: {StateTracker.get_accelerator().num_processes}
- Gradient checkpointing: {StateTracker.get_args().gradient_checkpointing}
- Prediction type: {model.PREDICTION_TYPE.value}{model.custom_model_card_schedule_info()}
- Optimizer: {StateTracker.get_args().optimizer}{optimizer_config if optimizer_config is not None else ''}
- Trainable parameter precision: {'Pure BF16' if torch.backends.mps.is_available() or StateTracker.get_args().mixed_precision == "bf16" else 'FP32'}
- Base model precision: `{args.base_model_precision}`
- Caption dropout probability: {StateTracker.get_args().caption_dropout_probability or 0.0 * 100}%
{'- Xformers: Enabled' if StateTracker.get_args().attention_mechanism == 'xformers' else ''}
{f'- SageAttention: Enabled {StateTracker.get_args().sageattention_usage}' if StateTracker.get_args().attention_mechanism == 'sageattention' else ''}
{lora_info(args=StateTracker.get_args())}

## Datasets

{datasets_str}

## Inference

{code_example(args=StateTracker.get_args(), repo_id=repo_id, model=model)}

{ema_info(args=StateTracker.get_args())}
"""

    logger.debug(f"YAML:\n{yaml_content}")
    logger.debug(f"Model Card:\n{model_card_content}")
    with open(os.path.join(repo_folder, "README.md"), "w", encoding="utf-8") as f:
        f.write(yaml_content + model_card_content)
