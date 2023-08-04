import os, torch, hashlib, logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Function to create a hash for a given caption
def create_hash(caption):
    return hashlib.md5(caption.encode()).hexdigest()

# Function to save the embeddings to a cache file
def save_to_cache(filename, embeddings):
    torch.save(embeddings, filename)

# Function to load the embeddings from a cache file
def load_from_cache(filename):
    return torch.load(filename)
# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt):
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
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
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
def encode_prompts(text_encoders, tokenizers, prompts):
    prompt_embeds_all = []
    pooled_prompt_embeds_all = []

    for prompt in prompts:
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
        prompt_embeds_all.append(prompt_embeds)
        pooled_prompt_embeds_all.append(pooled_prompt_embeds)

    return torch.stack(prompt_embeds_all).squeeze(dim=1), torch.stack(pooled_prompt_embeds_all).squeeze(dim=1)
# Modified function to compute and cache embeddings
def compute_embeddings_for_prompts(prompts, text_encoders, tokenizers, accelerator, cache_dir='cache'):
    prompt_embeds_all = []
    add_text_embeds_all = []

    with torch.no_grad():
        logger.debug(f'Beginning compute_embeddings_for_prompts: {prompts}')
        
        for prompt in prompts:
            # Create a unique filename based on the hash of the caption
            filename = os.path.join(cache_dir, create_hash(prompt) + '.pt')

            # Check if the embeddings are cached
            if os.path.exists(filename):
                prompt_embeds, add_text_embeds = load_from_cache(filename)
            else:
                prompt_embeds, pooled_prompt_embeds = encode_prompts(text_encoders, tokenizers, [prompt])
                add_text_embeds = pooled_prompt_embeds
                prompt_embeds = prompt_embeds.to(accelerator.device)
                add_text_embeds = add_text_embeds.to(accelerator.device)
                # Save the computed embeddings to the cache
                save_to_cache(filename, (prompt_embeds, add_text_embeds))

            prompt_embeds_all.append(prompt_embeds)
            add_text_embeds_all.append(add_text_embeds)

        prompt_embeds_all = torch.cat(prompt_embeds_all, dim=0)
        add_text_embeds_all = torch.cat(add_text_embeds_all, dim=0)

    logger.debug(f'Returning computed embeddings: {prompt_embeds_all}, {add_text_embeds_all}')
    return prompt_embeds_all, add_text_embeds_all