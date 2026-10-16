"""Qwen3-VL text conditioning shared by training and image sampling."""

import torch
import torch.nn.functional as F

TEXT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
MAX_TEXT_LENGTH = 8192
TEXT_PADDING_MULTIPLE = 80


def collate_embeddings(entries):
    embeddings, masks = [], []
    for entry in entries:
        value = entry["prompt_embeds"]
        mask = entry["attention_masks"]
        if value.ndim == 2:
            value, mask = value.unsqueeze(0), mask.reshape(1, -1)
        embeddings.extend(value.unbind(0))
        masks.extend(mask.bool().unbind(0))
    length = max(value.shape[0] for value in embeddings)
    return {
        "prompt_embeds": torch.stack([F.pad(value, (0, 0, 0, length - value.shape[0])) for value in embeddings]),
        "attention_masks": torch.stack([F.pad(value, (0, length - value.shape[0])) for value in masks]),
    }


@torch.no_grad()
def encode_prompts(tokenizer, text_encoder, prompts, device, max_length=MAX_TEXT_LENGTH):
    entries = []
    for prompt in prompts:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokens = tokenizer(formatted, truncation=True, max_length=max_length, return_tensors="pt")
        length = tokens.input_ids.shape[1]
        padding = min(length + (-length) % TEXT_PADDING_MULTIPLE, max_length) - length
        ids = F.pad(tokens.input_ids, (0, padding), value=tokenizer.pad_token_id).to(device)
        mask = F.pad(tokens.attention_mask, (0, padding)).to(device)
        positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
        positions = positions.unsqueeze(0).expand(4, -1, -1)
        output = text_encoder(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        embedding = torch.cat([output.hidden_states[index] for index in TEXT_LAYERS], dim=-1)
        entries.append({"prompt_embeds": embedding[0], "attention_masks": torch.ones_like(mask[0], dtype=torch.bool)})
    result = collate_embeddings(entries)
    return result["prompt_embeds"], result["attention_masks"]
