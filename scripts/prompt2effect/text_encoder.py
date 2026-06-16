from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EncodedPrompts:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor | None
    hidden_dim: int


class FrozenTransformersTextEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer, T5EncoderModel
        except ImportError as exc:
            raise ImportError("Prompt2Effect text encoding requires the transformers package.") from exc

        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if "t5" in model_name_or_path.lower():
            self.text_encoder = T5EncoderModel.from_pretrained(model_name_or_path, torch_dtype=dtype)
        else:
            self.text_encoder = AutoModel.from_pretrained(model_name_or_path, torch_dtype=dtype)
        self.text_encoder.to(device)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def encode(self, prompts: list[str], *, max_length: int) -> EncodedPrompts:
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
        output = self.text_encoder(**tokenized)
        hidden_states = getattr(output, "last_hidden_state", None)
        if hidden_states is None:
            raise ValueError(f"Text encoder `{self.model_name_or_path}` did not return last_hidden_state.")
        attention_mask = tokenized.get("attention_mask")
        return EncodedPrompts(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            hidden_dim=int(hidden_states.shape[-1]),
        )


def resolve_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype `{name}`. Use fp32, fp16, or bf16.")


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
