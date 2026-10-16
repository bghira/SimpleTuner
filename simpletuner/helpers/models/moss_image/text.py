"""Unpadded Qwen3.5 text conditioning for MOSS-Image."""

import torch

from simpletuner.helpers.models.text import collate_embeddings

MAX_TEXT_LENGTH = 4096


@torch.no_grad()
def encode_prompts(tokenizer, text_encoder, prompts, device, max_length=MAX_TEXT_LENGTH):
    if not 1 <= max_length <= MAX_TEXT_LENGTH:
        raise ValueError("MOSS-Image tokenizer length must be between 1 and 4096.")
    entries = []
    for prompt in prompts:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        output = text_encoder(**inputs, use_cache=False, return_dict=True)
        entries.append(
            {
                "prompt_embeds": output.last_hidden_state[0],
                "attention_masks": inputs.attention_mask[0].bool(),
            }
        )
    result = collate_embeddings(entries)
    return result["prompt_embeds"], result["attention_masks"]
