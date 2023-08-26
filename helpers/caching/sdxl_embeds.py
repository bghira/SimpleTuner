import os, torch, hashlib, logging
from tqdm import tqdm

logger = logging.getLogger("TextEmbeddingCache")
logger.setLevel(logging.INFO)


class TextEmbeddingCache:
    def __init__(self, text_encoders, tokenizers, accelerator, cache_dir="cache"):
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.accelerator = accelerator
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _generate_filename(self, filepath: str):
        """Get the cache filename for a given image filepath."""
        # Extract the base name from the filepath and replace the image extension with .pt
        return os.path.join(
            self.cache_dir, os.path.splitext(os.path.basename(filepath))[0] + ".pt"
        )

    def save_to_cache(self, filename, embeddings):
        logger.debug(f'Saving to cache: {filename}')
        torch.save(embeddings, filename)

    def load_from_cache(self, filename):
        return torch.load(filename)

    # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(self, text_encoders, tokenizers, prompt):
        prompt_embeds_list = []

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_prompts(self, text_encoders, tokenizers, prompts):
        prompt_embeds_all = []
        pooled_prompt_embeds_all = []

        for prompt in prompts:
            prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
                text_encoders, tokenizers, prompt
            )
            prompt_embeds_all.append(prompt_embeds)
            pooled_prompt_embeds_all.append(pooled_prompt_embeds)

        return torch.stack(prompt_embeds_all).squeeze(dim=1), torch.stack(
            pooled_prompt_embeds_all
        ).squeeze(dim=1)

    def compute_embeddings_for_prompts(self, prompts, return_concat: bool = True):
        prompt_embeds_all = []
        add_text_embeds_all = []

        with torch.no_grad():
            for prompt in tqdm(
                prompts, desc="Processing prompts", disable=return_concat
            ):
                filename = os.path.join(
                    self.cache_dir, self._generate_filename(prompt) + ".pt"
                )

                if os.path.exists(filename) and not return_concat:
                    logger.debug(
                        f"Not loading from cache, since we are only precomputing the embeds."
                    )
                    continue
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

                prompt_embeds_all.append(prompt_embeds)
                add_text_embeds_all.append(add_text_embeds)

            if not return_concat:
                logger.info(
                    "Not returning embeds, since we just concatenated a whackload of them."
                )
                del prompt_embeds_all
                del add_text_embeds_all
                return

            prompt_embeds_all = torch.cat(prompt_embeds_all, dim=0)
            add_text_embeds_all = torch.cat(add_text_embeds_all, dim=0)

        return prompt_embeds_all, add_text_embeds_all

    def precompute_embeddings_for_prompts(self, prompts):
        """This is the same function as compute_embeddings_for_prompts, but it specifically runs with the Accelerate context manager.

        Args:
            prompts (list[str]): All of the prompts.
        """
        self.compute_embeddings_for_prompts(prompts, return_concat=False)
