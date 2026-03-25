# Vendored from diffusers-anima: /src/diffusers-anima/src/diffusers_anima/pipelines/anima/prompt_utils.py
# Adapted for SimpleTuner local imports.

"""Prompt normalization and batching utilities for AnimaPipeline."""

from __future__ import annotations

from .validation import PromptInput


def _normalize_prompt_list(prompt: PromptInput, *, input_name: str) -> list[str]:
    if isinstance(prompt, str):
        prompts = [prompt]
    elif isinstance(prompt, (list, tuple)):
        prompts = list(prompt)
    else:
        raise ValueError(f"`{input_name}` must be a string or a list/tuple of strings.")

    if len(prompts) == 0:
        raise ValueError(f"`{input_name}` must not be empty.")
    for index, text in enumerate(prompts):
        if not isinstance(text, str):
            raise ValueError(f"`{input_name}`[{index}] must be a string.")
        if input_name == "prompt" and len(text.strip()) == 0:
            raise ValueError("`prompt` entries must be non-empty strings.")
    return prompts


def _resolve_prompt_batches(
    *,
    prompt: PromptInput,
    negative_prompt: PromptInput | None,
    num_images_per_prompt: int,
) -> tuple[list[str], list[str]]:
    """Expand prompt/negative_prompt inputs into per-image prompt batches."""
    prompts = _normalize_prompt_list(prompt, input_name="prompt")
    if num_images_per_prompt < 1:
        raise ValueError("`num_images_per_prompt` must be >= 1.")

    if negative_prompt is None:
        negative_prompts = [""] * len(prompts)
    elif isinstance(negative_prompt, str):
        negative_prompts = [negative_prompt] * len(prompts)
    else:
        negative_prompts = _normalize_prompt_list(negative_prompt, input_name="negative_prompt")
        if len(negative_prompts) != len(prompts):
            raise ValueError(
                "`negative_prompt` list length must match `prompt` list length. "
                f"Got {len(negative_prompts)} and {len(prompts)}."
            )

    batched_prompts: list[str] = []
    batched_negative_prompts: list[str] = []
    for text, neg_text in zip(prompts, negative_prompts, strict=True):
        for _ in range(num_images_per_prompt):
            batched_prompts.append(text)
            batched_negative_prompts.append(neg_text)
    return batched_prompts, batched_negative_prompts
