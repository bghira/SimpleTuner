import torch
from pathlib import Path
from helpers.models.hidream.pipeline import HiDreamImagePipeline
from helpers.models.hidream.transformer import HiDreamImageTransformer2DModel
from lycoris import create_lycoris_from_weights
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from optimum.quanto import quantize, freeze, qint8
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
import diffusers
# ==========================
# Config Block (EDIT HERE)
# ==========================
config = {
    "char_prompt": (
        "John, Indian Middle-aged, Long Flowing Black Hair, Brown Eyes, Lean Muscular Build, Structured M-shaped Hairline, Thick Upward-Curled Handlebar Mustache, Tripundra Tilak with Central Red Teardrop"
    ),
    "adapter_file_path": "/mnt/data/om/SimpleTuner/output/ravana_cfg1/checkpoint-3000/pytorch_lora_weights.safetensors",
    "output_dir": "generations/ravana_cfg1_guidance=5",  # Will be created if not exist
    "prompts": [
        "sipping coffee in a cafe",
        "running in a marathon, with the shot taking the full body from right side",
        "dancing in a disco with friends",
        "cleaning their house after waking up in the morning",
    ],
    "negative_prompt": "ugly, cropped, blurry, low-quality, mediocre average",
    "lora_scale": 0.9,
    "guidance_scale": 5,
    "seed": 42,
    "width": 1024,
    "height": 1024,
    "inference_steps": 50,
}
# ==========================

# Setup output dir
Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
llama_repo = "unsloth/Meta-Llama-3.1-8B-Instruct"
model_id = "HiDream-ai/HiDream-I1-Full"

tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(llama_repo)
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    llama_repo,
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16,
)
diffusers.HiDreamImageTransformer2DModel = HiDreamImageTransformer2DModel

transformer = HiDreamImageTransformer2DModel.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    subfolder="transformer"
)

pipeline = HiDreamImagePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    transformer=transformer,
)
pipeline.to(device)

# Apply LoRA
wrapper, _ = create_lycoris_from_weights(config["lora_scale"], config["adapter_file_path"], pipeline.transformer)
wrapper.merge_to()

# Optionally quantize
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

# Loop through prompts
for idx, prompt_suffix in enumerate(config["prompts"]):
    full_prompt = f"{config['char_prompt']} {prompt_suffix}"

    # Encode prompt
    t5_embeds, llama_embeds, neg_t5_embeds, neg_llama_embeds, pooled_embeds, neg_pooled_embeds = pipeline.encode_prompt(
        prompt=full_prompt,
        prompt_2=full_prompt,
        prompt_3=full_prompt,
        prompt_4=full_prompt,
        num_images_per_prompt=1
    )

    # Generate
    image = pipeline(
        t5_prompt_embeds=t5_embeds,
        llama_prompt_embeds=llama_embeds,
        pooled_prompt_embeds=pooled_embeds,
        negative_t5_prompt_embeds=neg_t5_embeds,
        negative_llama_prompt_embeds=neg_llama_embeds,
        negative_pooled_prompt_embeds=neg_pooled_embeds,
        num_inference_steps=config["inference_steps"],
        generator=torch.Generator(device=device).manual_seed(config["seed"]),
        width=config["width"],
        height=config["height"],
        guidance_scale=config["guidance_scale"],
    ).images[0]

    # Save
    save_path = Path(config["output_dir"]) / f"prompt{idx+1}.png"
    image.save(save_path)
    print(f"Saved: {save_path}")
