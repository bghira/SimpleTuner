import autoroot
import autorootcwd
import argparse
import torch
from pathlib import Path
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from lycoris import create_lycoris_from_weights
from optimum.quanto import quantize, freeze, qint8
import diffusers

def parse_args():
    parser = argparse.ArgumentParser(description="HiDream Full Inference with optional LoRA")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt or path to .txt file with prompts (one per line)")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA .safetensors file (optional)")
    parser.add_argument("--output_dir", type=str, default="generations")
    parser.add_argument("--lora_scale", type=float, default=0.9)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--negative_prompt", type=str, default="ugly, cropped, blurry, low-quality, mediocre average")
    parser.add_argument("--cfg_zero", action="store_true", help="Use cfg_zero pipeline variant")
    return parser.parse_args()

def load_prompts(prompt_input):
    if prompt_input.endswith(".txt"):
        return Path(prompt_input).read_text().strip().splitlines()
    return [prompt_input]

def main():
    args = parse_args()
    prompts = load_prompts(args.prompt)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llama_repo = "unsloth/Meta-Llama-3.1-8B-Instruct"
    model_id = "HiDream-ai/HiDream-I1-Full"

    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(llama_repo)
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        llama_repo,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16,
    )

    from helpers.models.hidream.transformer import HiDreamImageTransformer2DModel
    diffusers.HiDreamImageTransformer2DModel = HiDreamImageTransformer2DModel
    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        subfolder="transformer"
    )

    if args.cfg_zero:
        from helpers.models.hidream.pipeline_cfg_zero import HiDreamImagePipeline
    else:
        from helpers.models.hidream.pipeline import HiDreamImagePipeline

    pipeline = HiDreamImagePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        transformer=transformer,
    ).to(device)

    # Apply LoRA if adapter is present
    if args.adapter_path and Path(args.adapter_path).exists():
        print(f"[✓] Applying LoRA from {args.adapter_path}")
        wrapper, _ = create_lycoris_from_weights(args.lora_scale, args.adapter_path, pipeline.transformer)
        wrapper.merge_to()
    else:
        if args.adapter_path:
            print(f"[!] LoRA path provided but not found: {args.adapter_path} — skipping.")
        else:
            print("[i] No LoRA adapter path provided — proceeding without LoRA.")

    quantize(pipeline.transformer, weights=qint8)
    freeze(pipeline.transformer)

    for idx, prompt in enumerate(prompts):
        t5_embeds, llama_embeds, neg_t5_embeds, neg_llama_embeds, pooled_embeds, neg_pooled_embeds = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            prompt_4=prompt,
            num_images_per_prompt=1,
        )

        image = pipeline(
            t5_prompt_embeds=t5_embeds,
            llama_prompt_embeds=llama_embeds,
            pooled_prompt_embeds=pooled_embeds,
            negative_t5_prompt_embeds=neg_t5_embeds,
            negative_llama_prompt_embeds=neg_llama_embeds,
            negative_pooled_prompt_embeds=neg_pooled_embeds,
            num_inference_steps=args.inference_steps,
            generator=torch.Generator(device=device).manual_seed(args.seed),
            width=args.width,
            height=args.height,
            guidance_scale=args.guidance_scale,
        ).images[0]

        save_path = Path(args.output_dir) / f"image_{idx+1}.png"
        image.save(save_path)
        print(f"[✓] Saved: {save_path}")

if __name__ == "__main__":
    main()
