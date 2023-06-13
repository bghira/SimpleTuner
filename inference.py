from accelerate import Accelerator
from diffusers import DiffusionPipeline, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel
from prompts import prompts
import torch, os

# Load the pipeline with the same arguments (model, revision) that were used for training
model_id = "stabilityai/stable-diffusion-2-1"
base_dir = "/notebooks/datasets"
model_path = os.path.join(base_dir, 'models')
#output_test_dir = os.path.join(base_dir, 'test_results')
output_test_dir = os.path.join(base_dir, 'random_test')
save_pretrained = False
torch_seed = 4202420420

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
#checkpoints = [ str(x) for x in range(range_begin, range_end + range_step, range_step) ]
checkpoints.reverse()

torch.manual_seed(torch_seed)
torch.set_float32_matmul_precision('high')

for checkpoint in checkpoints:
    if len(checkpoints) > 1 and os.path.isfile(f'{output_test_dir}/last_dinosaur-{checkpoint}.png'):
        continue
    try:
        print(f'Loading checkpoint: {model_path}/checkpoint-{checkpoint}')
        # Does the checkpoint path exist?
        if not os.path.exists(f'{model_path}/checkpoint-{checkpoint}'):
            print(f'Checkpoint {checkpoint} does not exist.')
            continue
        
        if checkpoint != "0":
            unet = UNet2DConditionModel.from_pretrained(f"{model_path}/checkpoint-{checkpoint}/unet")
            unet = torch.compile(unet)
            # if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
            text_encoder = CLIPTextModel.from_pretrained(f"{model_path}/checkpoint-{checkpoint}/text_encoder")
            pipeline = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder)
        else:
            pipeline = DiffusionPipeline.from_pretrained(model_id)
            pipeline.unet = torch.compile(pipeline.unet)
        pipeline.scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            rescale_betas_zero_snr=True,
            guidance_rescale=0.3,
            timestep_scaling="trailing"
        )
        pipeline.to("cuda")
    except Exception as e:
        print(f'Could not generate pipeline for checkpoint {checkpoint}: {e}')
        continue
    # Does the file exist already?
    import os
    negative = "watermark, cropped, out-of-frame, low quality, low res, oorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, synthetic, rendering"
    for shortname, prompt in prompts.items():
        if not os.path.isfile(f'{output_test_dir}/{shortname}-{checkpoint}.png'):
            print(f'Generating {shortname} at {checkpoint}')
            output = pipeline(negative_prompt=negative, prompt=prompt, guidance_scale=9.2, guidance_rescale=0.3, width=1152, height=768, num_inference_steps=15).images[0]
            output.save(f'{output_test_dir}/{shortname}-{checkpoint}.png')
        
    if save_pretrained and not os.path.exists(f'{model_path}pipeline'):
        print(f'Saving pretrained pipeline.')
        pipeline.save_pretrained(f'{model_path}/pseudo-realism', safe_serialization=True)
    elif save_pretrained:
        raise Exception('Can not save pretrained model, path already exists.')
