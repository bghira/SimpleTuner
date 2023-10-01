import os, torch, hashlib, logging
from tqdm import tqdm
from helpers.training.state_tracker import StateTracker
from helpers.prompts import PromptHandler

logger = logging.getLogger("TextEmbeddingCache")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL") or "INFO")


class TextEmbeddingCache:
    prompts = None

    def __init__(
        self,
        text_encoders,
        tokenizers,
        accelerator,
        cache_dir: str = "cache",
        model_type: str = "sdxl",
        prompt_handler: PromptHandler = None,
    ):
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.accelerator = accelerator
        self.cache_dir = cache_dir
        self.model_type = model_type
        self.prompt_handler = prompt_handler
        os.makedirs(self.cache_dir, exist_ok=True)

    def create_hash(self, caption):
        return hashlib.md5(caption.encode()).hexdigest()

    def save_to_cache(self, filename, embeddings):
        torch.save(embeddings, filename)

    def load_from_cache(self, filename):
        return torch.load(filename, map_location=self.accelerator.device)

    def encode_legacy_prompt(self, text_encoder, tokenizer, prompt):
        input_tokens = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        output = text_encoder(input_tokens)[0]
        logger.debug(f"Legacy prompt shape: {output.shape}")
        logger.debug(f"Legacy prompt encoded: {output}")
        return output

    # Adapted from pipelines.StableDiffusionXLPipeline.encode_sdxl_prompt
    def encode_sdxl_prompt(
        self,
        text_encoders,
        tokenizers,
        prompt,
        is_validation: bool = False,
        negative_prompt: str = "",
    ):
        prompt_embeds_list = []

        emitted_warning = False
        # If prompt_handler (Compel) is available, use it for all prompts
        if self.prompt_handler and is_validation:
            positive_prompt, negative_prompt = self.prompt_handler.process_long_prompt(
                prompt
            )
            return positive_prompt, negative_prompt
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt, padding="max_length", truncation=True, return_tensors="pt"
            )
            text_input_ids = text_inputs.input_ids

            untruncated_ids = tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[
                -1
            ] > tokenizer.model_max_length and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                )
                if not emitted_warning:
                    # Only print this once. It's a bit spammy otherwise.
                    emitted_warning = True
                    logger.warning(
                        f"The following part of your input was truncated because CLIP can only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}"
                    )

            prompt_embeds_output = text_encoder(
                text_input_ids.to(text_encoder.device), output_hidden_states=True
            )
            # We are always interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds_output[0]
            prompt_embeds = prompt_embeds_output.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)

            # Clear out anything we moved to the text encoder device
            del text_input_ids

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_sdxl_prompts(
        self,
        text_encoders,
        tokenizers,
        prompts,
        is_validation: bool = False,
        negative_prompt: str = "",
    ):
        prompt_embeds_all = []
        pooled_prompt_embeds_all = []

        for prompt in prompts:
            prompt_embeds, pooled_prompt_embeds = self.encode_sdxl_prompt(
                text_encoders, tokenizers, prompt, is_validation
            )
            prompt_embeds_all.append(prompt_embeds)
            pooled_prompt_embeds_all.append(pooled_prompt_embeds)

        return torch.stack(prompt_embeds_all).squeeze(dim=1), torch.stack(
            pooled_prompt_embeds_all
        ).squeeze(dim=1)

    def encode_prompt(self, prompt: str, is_validation: bool = False):
        if self.model_type == "sdxl":
            return self.encode_sdxl_prompt(
                self.text_encoders, self.tokenizers, prompt, is_validation
            )
        else:
            return self.encode_legacy_prompt(
                self.text_encoders[0], self.tokenizers[0], prompt
            )

    def compute_embeddings_for_prompts(
        self,
        prompts,
        return_concat: bool = True,
        is_validation: bool = False,
        negative_prompt: str = "",
    ):
        if self.model_type == "sdxl":
            return self.compute_embeddings_for_sdxl_prompts(
                prompts,
                return_concat=return_concat,
                is_validation=is_validation,
                negative_prompt=negative_prompt,
            )
        elif self.model_type == "legacy":
            return self.compute_embeddings_for_legacy_prompts(
                prompts, return_concat=return_concat
            )

    def compute_embeddings_for_sdxl_prompts(
        self,
        prompts: list = None,
        return_concat: bool = True,
        is_validation: bool = False,
        negative_prompt: str = "",
    ):
        prompt_embeds_all = []
        add_text_embeds_all = []

        with torch.no_grad():
            for prompt in tqdm(
                prompts or self.prompts,
                desc="Processing prompts",
                disable=return_concat,
            ):
                filename = os.path.join(
                    self.cache_dir, self.create_hash(prompt) + ".pt"
                )
                if os.path.exists(filename) and not return_concat:
                    continue
                if os.path.exists(filename):
                    logger.debug(f"Loading from cache: {filename}")
                    prompt_embeds, add_text_embeds = self.load_from_cache(filename)
                else:
                    logger.debug(f"Encoding prompt: {prompt}")
                    prompt_embeds, pooled_prompt_embeds = self.encode_sdxl_prompts(
                        self.text_encoders,
                        self.tokenizers,
                        [prompt],
                        is_validation,
                        negative_prompt=negative_prompt,
                    )
                    add_text_embeds = pooled_prompt_embeds
                    if return_concat:
                        prompt_embeds = prompt_embeds.to(self.accelerator.device)
                        add_text_embeds = add_text_embeds.to(self.accelerator.device)
                    self.save_to_cache(filename, (prompt_embeds, add_text_embeds))

                prompt_embeds_all.append(prompt_embeds)
                add_text_embeds_all.append(add_text_embeds)

            if not return_concat:
                del prompt_embeds_all
                del add_text_embeds_all
                return

            prompt_embeds_all = torch.cat(prompt_embeds_all, dim=0)
            add_text_embeds_all = torch.cat(add_text_embeds_all, dim=0)

        return prompt_embeds_all, add_text_embeds_all

    def compute_embeddings_for_legacy_prompts(
        self, prompts: list = None, return_concat: bool = True
    ):
        prompt_embeds_all = []

        with torch.no_grad():
            for prompt in tqdm(
                prompts or self.prompts,
                desc="Processing prompts",
                disable=return_concat,
            ):
                filename = os.path.join(
                    self.cache_dir, self.create_hash(prompt) + ".pt"
                )
                if os.path.exists(filename) and not return_concat:
                    continue
                if os.path.exists(filename):
                    logger.debug(f"Loading from cache: {filename}")
                    prompt_embeds = self.load_from_cache(filename)
                else:
                    logger.debug(f"Encoding prompt: {prompt}")
                    prompt_embeds = self.encode_legacy_prompt(
                        self.text_encoders[0], self.tokenizers[0], [prompt]
                    )
                    prompt_embeds = prompt_embeds.to(self.accelerator.device)
                    self.save_to_cache(filename, prompt_embeds)

                prompt_embeds_all.append(prompt_embeds)

            if not return_concat:
                logger.info(
                    "Not returning embeds, since we just concatenated a whackload of them."
                )
                del prompt_embeds_all
                return

        return prompt_embeds_all

    def split_cache_between_processes(self, prompts: list):
        # Use the accelerator to split the data
        with self.accelerator.split_between_processes(prompts) as split_files:
            self.prompts = split_files
