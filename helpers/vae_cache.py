import os, torch, hashlib, logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class VAECache:
    def __init__(self, vae, accelerator, cache_dir="vae_cache"):
        self.vae = vae
        self.accelerator = accelerator
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def create_hash(self, caption):
        return hashlib.md5(caption.encode()).hexdigest()

    def save_to_cache(self, filename, embeddings):
        torch.save(embeddings, filename)

    def load_from_cache(self, filename):
        return torch.load(filename)

    def _encode_image(self, pixel_values):
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor


    def precompute_vae_results(self, image_paths):
        with torch.no_grad():
            logger.debug(f"Beginning compute_embeddings_for_prompts: {prompts}")

            for prompt in prompts:
                filename = os.path.join(
                    self.cache_dir, self.create_hash(prompt) + ".pt"
                )

                if os.path.exists(filename):
                    prompt_embeds, add_text_embeds = self.load_from_cache(filename)
                else:
                    prompt_embeds, pooled_prompt_embeds = self.encode_prompts(
                        self.text_encoders, self.tokenizers, [prompt]
                    )
                    add_text_embeds = pooled_prompt_embeds
                    prompt_embeds = prompt_embeds.to(self.accelerator.device)
                    add_text_embeds = add_text_embeds.to(self.accelerator.device)
                    self.save_to_cache(filename, (prompt_embeds, add_text_embeds))
