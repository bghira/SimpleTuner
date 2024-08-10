import os
import logging
from helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


def _model_imports(args):
    output = "import torch\n"
    output += "from diffusers import DiffusionPipeline"

    return f"{output}"


def _model_load(args, repo_id: str = None):
    hf_user_name = StateTracker.get_hf_username()
    if hf_user_name is None:
        repo_id = f"{hf_user_name}/{repo_id}" if hf_user_name else repo_id
    if "lora" in args.model_type:
        output = (
            f"model_id = '{args.pretrained_model_name_or_path}'"
            f"\nadapter_id = '{repo_id if repo_id is not None else args.output_dir}'"
            f"\npipeline = DiffusionPipeline.from_pretrained(model_id)"
            f"\npipeline.load_lora_weights(adapter_id)"
        )
    else:
        output = (
            f"model_id = '{repo_id if repo_id else os.path.join(args.output_dir, 'pipeline')}'"
            f"\npipeline = DiffusionPipeline.from_pretrained(model_id)"
        )

    return output


def _torch_device():
    return """'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'"""


def _negative_prompt(args, in_call: bool = False):
    if args.flux:
        return ""
    if not in_call:
        return f"negative_prompt = '{args.validation_negative_prompt}'"
    return "\n    negative_prompt=negative_prompt,"


def _guidance_rescale(args):
    if any([args.sd3, args.flux, args.pixart_sigma]):
        return ""
    return f"\n    guidance_rescale={args.validation_guidance_rescale},"


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

pipeline.to({_torch_device()})
image = pipeline(
    prompt=prompt,{_negative_prompt(args, in_call=True) if not args.flux else ''}
    num_inference_steps={args.validation_num_inference_steps},
    generator=torch.Generator(device={_torch_device()}).manual_seed(1641421826),
    {_validation_resolution(args)}
    guidance_scale={args.validation_guidance},{_guidance_rescale(args)}
).images[0]
image.save("output.png", format="PNG")
```
"""
    return code_example


def lora_info(args):
    """Return a string with the LORA information."""
    if "lora" not in args.model_type:
        return ""
    return f"""- LoRA Rank: {args.lora_rank}
- LoRA Alpha: {args.lora_alpha}
- LoRA Dropout: {args.lora_dropout}
- LoRA initialisation style: {args.lora_init_type}
"""


def model_card_note(args):
    """Return a string with the model card note."""
    note_contents = args.model_card_note if args.model_card_note else ""
    return f"\n{note_contents}\n"


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
    logger.debug(f"Validating from prompts: {validation_prompts}")
    assets_folder = os.path.join(repo_folder, "assets")
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
    yaml_content = f"""---
license: creativeml-openrail-m
base_model: "{base_model}"
tags:
  - {'stable-diffusion' if 'deepfloyd' not in StateTracker.get_args().model_type else 'deepfloyd-if'}
  - {'stable-diffusion-diffusers' if 'deepfloyd' not in StateTracker.get_args().model_type else 'deepfloyd-if-diffusers'}
  - text-to-image
  - diffusers
  - simpletuner
  - {StateTracker.get_args().model_type}
{'  - template:sd-lora' if 'lora' in StateTracker.get_args().model_type else ''}
inference: true
{widget_str}
---

"""
    model_card_content = f"""# {repo_id}

This is a {'LoRA' if 'lora' in StateTracker.get_args().model_type else 'full rank finetune'} derived from [{base_model}](https://huggingface.co/{base_model}).

{'This is a **diffusion** model trained using DDPM objective instead of Flow matching. **Be sure to set the appropriate scheduler configuration.**' if StateTracker.get_args().sd3 and StateTracker.get_args().flow_matching_loss == "diffusion" else ''}

{'The main validation prompt used during training was:' if prompt else 'Validation used ground-truth images as an input for partial denoising (img2img).' if StateTracker.get_args().validation_using_datasets else 'No validation prompt was used during training.'}
{model_card_note(args)}
{'```' if prompt else ''}
{prompt}
{'```' if prompt else ''}

## Validation settings
- CFG: `{StateTracker.get_args().validation_guidance}`
- CFG Rescale: `{StateTracker.get_args().validation_guidance_rescale}`
- Steps: `{StateTracker.get_args().validation_num_inference_steps}`
- Sampler: `{StateTracker.get_args().validation_noise_scheduler}`
- Seed: `{StateTracker.get_args().validation_seed}`
- Resolution{'s' if ',' in StateTracker.get_args().validation_resolution else ''}: `{StateTracker.get_args().validation_resolution}`

Note: The validation settings are not necessarily the same as the [training settings](#training-settings).

{'You can find some example images in the following gallery:' if images is not None else ''}\n

<Gallery />

The text encoder {'**was**' if train_text_encoder else '**was not**'} trained.
{'You may reuse the base model text encoder for inference.' if not train_text_encoder else 'If the text encoder from this repository is not used at inference time, unexpected or bad results could occur.'}


## Training settings

- Training epochs: {StateTracker.get_epoch() - 1}
- Training steps: {StateTracker.get_global_step()}
- Learning rate: {StateTracker.get_args().learning_rate}
- Effective batch size: {StateTracker.get_args().train_batch_size * StateTracker.get_args().gradient_accumulation_steps * StateTracker.get_accelerator().num_processes}
  - Micro-batch size: {StateTracker.get_args().train_batch_size}
  - Gradient accumulation steps: {StateTracker.get_args().gradient_accumulation_steps}
  - Number of GPUs: {StateTracker.get_accelerator().num_processes}
- Prediction type: {'flow-matching' if (StateTracker.get_args().sd3 or StateTracker.get_args().flux) else StateTracker.get_args().prediction_type}
- Rescaled betas zero SNR: {StateTracker.get_args().rescale_betas_zero_snr}
- Optimizer: {'AdamW, stochastic bf16' if StateTracker.get_args().adam_bfloat16 else 'AdamW8Bit' if StateTracker.get_args().use_8bit_adam else 'Adafactor' if StateTracker.get_args().use_adafactor_optimizer else 'Prodigy' if StateTracker.get_args().use_prodigy_optimizer else 'AdamW'}
- Precision: {'Pure BF16' if StateTracker.get_args().adam_bfloat16 else StateTracker.get_args().mixed_precision}
- Xformers: {'Enabled' if StateTracker.get_args().enable_xformers_memory_efficient_attention else 'Not used'}
{lora_info(args=StateTracker.get_args())}

## Datasets

{datasets_str}

## Inference

{code_example(args=StateTracker.get_args(), repo_id=repo_id)}
"""

    logger.debug(f"YAML:\n{yaml_content}")
    logger.debug(f"Model Card:\n{model_card_content}")
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml_content + model_card_content)
