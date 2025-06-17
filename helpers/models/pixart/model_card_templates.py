from helpers.training.state_tracker import StateTracker


def pixart_sigma_controlnet_code_example(args, repo_id: str, model) -> str:
    """Generate ControlNet-specific code example for PixArt Sigma"""
    hf_user_name = StateTracker.get_hf_username()
    if hf_user_name and repo_id:
        repo_id = f"{hf_user_name}/{repo_id}"

    return f"""
```python
import torch
from diffusers import PixArtSigmaPipeline, PixArtSigmaControlNetPipeline
# if you're not in the SimpleTuner environment, this import will fail.
from helpers.models.pixart.controlnet import PixArtSigmaControlNetAdapterModel

# Load base model
base_model_id = "{args.pretrained_model_name_or_path}"
controlnet_id = "{repo_id if repo_id else args.output_dir}"

# Load ControlNet adapter
controlnet = PixArtSigmaControlNetAdapterModel.from_pretrained(
    f"{{controlnet_id}}/controlnet"
)

# Create pipeline
pipeline = PixArtSigmaControlNetPipeline.from_pretrained(
    base_model_id,
    controlnet=controlnet,
    torch_dtype=torch.bfloat16
)
pipeline.to('cuda' if torch.cuda.is_available() else 'cpu')

# Load your control image
from PIL import Image
control_image = Image.open("path/to/control/image.png")

# Generate
prompt = "{args.validation_prompt if args.validation_prompt else 'An astronaut riding a horse'}"
image = pipeline(
    prompt=prompt,
    image=control_image,
    num_inference_steps={args.validation_num_inference_steps},
    guidance_scale={args.validation_guidance},
    generator=torch.Generator(device='cuda').manual_seed({args.validation_seed or 42}),
    controlnet_conditioning_scale={getattr(args, 'controlnet_conditioning_scale', 1.0)},
).images[0]

image.save("output.png")
"""
