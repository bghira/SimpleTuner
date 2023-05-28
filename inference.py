from accelerate import Accelerator
from diffusers import DiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel
import torch, os

# Load the pipeline with the same arguments (model, revision) that were used for training
model_id = "stabilityai/stable-diffusion-2-1"
base_dir = "/notebooks/datasets"
model_path = os.path.join(base_dir, 'models')
output_test_dir = os.path.join(base_dir, 'test_results')
save_pretrained = False
torch_seed = 420420420

# Find the latest checkpoint
import os
checkpoints = [ int(x.split('-')[1]) for x in os.listdir(model_path) if x.startswith('checkpoint-') ]
checkpoints.sort()
range_begin = 0
range_step = 100
try:
    range_end = checkpoints[-1]
except Exception as e:
    range_end = range_begin
print(f'Highest checkpoint found so far: {range_end}')

# Convert numeric range to an array of string numerics:
checkpoints = [ str(x) for x in range(range_begin, range_end + range_step, range_step) ]
checkpoints.reverse()

torch.manual_seed(torch_seed)
torch.set_float32_matmul_precision('high')
def enforce_zero_terminal_snr(betas):
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

     # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    # Shift so last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    # Scale so first timestep is back to old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
    alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
     # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas

def patch_scheduler_betas(scheduler):
    scheduler.betas = enforce_zero_terminal_snr(scheduler.betas)
    return scheduler


for checkpoint in checkpoints:
    if len(checkpoints) > 1 and os.path.isfile(f'{output_test_dir}/target-{checkpoint}.png'):
        continue
    try:
        print(f'Loading checkpoint: {model_path}/checkpoint-{checkpoint}')
        if checkpoint != "0":
            unet = UNet2DConditionModel.from_pretrained(f"{model_path}/checkpoint-{checkpoint}/unet")
            unet = torch.compile(unet)
            # if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
            text_encoder = CLIPTextModel.from_pretrained(f"{model_path}/checkpoint-{checkpoint}/text_encoder")
            pipeline = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder)
        else:
            pipeline = DiffusionPipeline.from_pretrained(model_id)
            pipeline.unet = torch.compile(pipeline.unet)
        pipeline.scheduler = DDPMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            use_karras_sigmas=True
        )
        patch_scheduler_betas(pipeline.scheduler)
        pipeline.to("cuda")
    except Exception as e:
        print(f'Could not generate pipeline for checkpoint {checkpoint}: {e}')
        continue
    # Does the file exist already?
    import os
    negative = "low quality, low res, oorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, synthetic, rendering"
    prompts = {
        "woman": "a woman, hanging out on the beach",
        "man": "a man playing guitar in a park",
        "child": "a child flying a kite on a sunny day",
        "alien": "an alien exploring the Mars surface",
        "robot": "a robot serving coffee in a cafe",
        "knight": "a knight protecting a castle",
        "menn": "a group of men",
        "bicycle": "a bicycle, on a mountainside, on a sunny day",
        "cosmic": "cosmic entity, sitting in an impossible position, quantum reality, colours",
        "wizard": "a mage wizard, bearded and gray hair, blue  star hat with wand and mystical haze",
        "wizarddd": "digital art, fantasy, portrait of an old wizard, detailed",
        "macro": "a dramatic city-scape at sunset or sunrise",
        "micro": "RNA and other molecular machinery of life",
        "gecko": "a leopard gecko stalking a cricket"
    }

    for shortname, prompt in prompts.items():
        if not os.path.isfile(f'{output_test_dir}/{shortname}-{checkpoint}.png'):
            print(f'Generating {shortname} at {checkpoint}')
            output = pipeline(negative_prompt=negative, prompt=prompt, num_inference_steps=50).images[0]
            output.save(f'{output_test_dir}/{shortname}-{checkpoint}.png')
        
    if save_pretrained and not os.path.exists(f'{model_path}pipeline'):
        print(f'Saving pretrained pipeline.')
        pipeline.save_pretrained('{model_path}beta-v3')
    elif save_pretrained:
        raise Exception('Can not save pretrained model, path already exists.')
