# Copyright 2026 SimpleTuner contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reference-audio constrained generation for MiniMax Music 3.

This file is self-contained. It loads a released SimpleTuner RVQ encoder,
encodes 44.1 kHz audio with the original DAV encoder, predicts eight RVQ codes
per 25 Hz frame, and periodically constrains the official MiniMax Music 3
language model to the encoder's top-5 semantic candidates.
"""

from __future__ import annotations

import dataclasses
import math
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm

COLLECTION_REPO_ID = "SimpleTuner/open-rvq-encoder-minimax-music3"
OFFICIAL_MODEL_REPO_ID = "MiniMaxAI/MiniMax-Music3"
DEFAULT_ENCODER_FILE = "encoders/minimax_music3_rvq_encoder_v4_169m_autoregressive_depth_recommended.safetensors"
DAV_FILE = "dav.pth"
SAMPLE_RATE = 44_100
DAV_HOP_SAMPLES = 512
DAV_CHUNK_OVERLAP_SAMPLES = 8_192
FRAME_RATE = 25
LATENT_RATE_NUM = 441
LATENT_RATE_DEN = 128
AUDIO_CODE_OFFSET = 151_675
AUDIO_END_TOKEN_ID = 151_670
AUDIO_CFG_TOKEN_ID = 151_654
SEMANTIC_VOCAB_SIZE = 16_384
AR_CFG_SCALE = 1.5
AR_TOP_K = 50
REFERENCE_CANDIDATE_COUNT = 5
DEFAULT_REFERENCE_INTERVAL = 5

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


@dataclasses.dataclass(frozen=True)
class ReferenceRolloutTrace:
    frame_hiddens: torch.Tensor
    generated_codes: torch.Tensor
    reference_feedback: torch.Tensor
    warmup_feedback_codes: torch.Tensor
    target_semantic_log_probs: torch.Tensor | None = None
    target_semantic_top1: torch.Tensor | None = None


@dataclasses.dataclass(frozen=True)
class RVQEncoderConfig:
    latent_channels: int = 128
    codebook_vocab_sizes: tuple[int, ...] = (16_384, 1024, 1024, 1024, 1024, 1024, 1024, 1024)
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    max_position_embeddings: int = 128
    conv_dilations: tuple[int, ...] = (1, 3, 9)
    mup: bool = False
    mup_output_mult: float = 1.0
    mup_readout_zero_init: bool = False
    mup_attention_multiplier: float = 8.0
    depth_decoder: bool = False
    depth_decoder_dim: int = 512
    depth_decoder_layers: int = 2
    depth_decoder_heads: int = 8
    depth_decoder_ff_mult: int = 4
    depth_decoder_dropout: float = 0.1

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RVQEncoderConfig":
        normalized = dict(values)
        for key in ("codebook_vocab_sizes", "conv_dilations"):
            if key in normalized:
                normalized[key] = tuple(normalized[key])
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(normalized) - known)
        if unknown:
            raise ValueError(f"Unknown RVQ encoder configuration fields: {unknown}")
        return cls(**normalized)


class RVQResBlock(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(F.gelu(self.norm(hidden_states)))
        return hidden_states + self.conv2(F.gelu(residual))


class RVQMuTransformerEncoderLayer(nn.Module):
    def __init__(self, config: RVQEncoderConfig):
        super().__init__()
        if config.d_model % config.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.attention_multiplier = config.mup_attention_multiplier
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.linear1 = nn.Linear(config.d_model, config.d_model * config.ff_mult)
        self.linear2 = nn.Linear(config.d_model * config.ff_mult, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.attn_dropout = nn.Dropout(config.dropout)

    def _split_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, frames, _ = hidden_states.shape
        return hidden_states.view(batch, frames, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(hidden_states)
        query = self._split_heads(self.q_proj(normalized))
        key = self._split_heads(self.k_proj(normalized))
        value = self._split_heads(self.v_proj(normalized))
        scores = torch.matmul(query, key.transpose(-2, -1)) * (self.attention_multiplier / self.head_dim)
        probabilities = self.attn_dropout(F.softmax(scores.float(), dim=-1).to(query.dtype))
        attended = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
        attended = attended.view(hidden_states.shape)
        hidden_states = hidden_states + self.dropout(self.out_proj(attended))
        feedforward = self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(hidden_states)))))
        return hidden_states + self.dropout(feedforward)


class RVQDepthDecoderLayer(nn.Module):
    def __init__(self, config: RVQEncoderConfig):
        super().__init__()
        dim = config.depth_decoder_dim
        if dim % config.depth_decoder_heads:
            raise ValueError("depth_decoder_dim must be divisible by depth_decoder_heads")
        self.num_heads = config.depth_decoder_heads
        self.head_dim = dim // config.depth_decoder_heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.linear1 = nn.Linear(dim, dim * config.depth_decoder_ff_mult)
        self.linear2 = nn.Linear(dim * config.depth_decoder_ff_mult, dim)
        self.dropout = nn.Dropout(config.depth_decoder_dropout)
        self.attn_dropout = nn.Dropout(config.depth_decoder_dropout)

    def _split_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, depth, _ = hidden_states.shape
        return hidden_states.view(batch, depth, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(hidden_states)
        query = self._split_heads(self.q_proj(normalized))
        key = self._split_heads(self.k_proj(normalized))
        value = self._split_heads(self.v_proj(normalized))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        depth = hidden_states.shape[1]
        scores = scores.masked_fill(torch.ones((depth, depth), dtype=torch.bool, device=scores.device).triu(1), -torch.inf)
        probabilities = self.attn_dropout(F.softmax(scores.float(), dim=-1).to(query.dtype))
        attended = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
        attended = attended.view(hidden_states.shape)
        hidden_states = hidden_states + self.dropout(self.out_proj(attended))
        feedforward = self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(hidden_states)))))
        return hidden_states + self.dropout(feedforward)


class MiniMaxMusicRVQDepthDecoder(nn.Module):
    def __init__(self, config: RVQEncoderConfig):
        super().__init__()
        self.config = config
        self.context_projection = nn.Linear(config.d_model, config.depth_decoder_dim, bias=False)
        self.prior_embeddings = nn.ModuleList(
            nn.Embedding(vocab_size, config.depth_decoder_dim) for vocab_size in config.codebook_vocab_sizes[:-1]
        )
        self.position = nn.Parameter(torch.zeros(1, len(config.codebook_vocab_sizes), config.depth_decoder_dim))
        self.layers = nn.ModuleList(RVQDepthDecoderLayer(config) for _ in range(config.depth_decoder_layers))
        self.norm = nn.LayerNorm(config.depth_decoder_dim)
        self.heads = nn.ModuleList(
            nn.Linear(config.depth_decoder_dim, vocab_size) for vocab_size in config.codebook_vocab_sizes[1:]
        )

    def _decode(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden_states = sequence + self.position[:, : sequence.shape[1]].to(sequence.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)

    def forward(self, frame_context: torch.Tensor, semantic_codes: torch.Tensor) -> list[torch.Tensor]:
        batch, frames, _ = frame_context.shape
        sequence = torch.cat(
            (
                self.context_projection(frame_context).flatten(0, 1).unsqueeze(1),
                self.prior_embeddings[0](semantic_codes).flatten(0, 1).unsqueeze(1),
            ),
            dim=1,
        )
        logits = []
        for acoustic_index, head in enumerate(self.heads):
            head_logits = head(self._decode(sequence)[:, -1]).view(batch, frames, -1)
            logits.append(head_logits)
            if acoustic_index + 1 < len(self.heads):
                selected = head_logits.argmax(dim=-1)
                prior = self.prior_embeddings[acoustic_index + 1](selected).flatten(0, 1).unsqueeze(1)
                sequence = torch.cat((sequence, prior), dim=1)
        return logits


class MiniMaxMusicRVQEncoder(nn.Module):
    def __init__(self, config: RVQEncoderConfig):
        super().__init__()
        self.config = config
        self.conv_in = nn.Conv1d(config.latent_channels, config.d_model, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList(RVQResBlock(config.d_model, dilation) for dilation in config.conv_dilations)
        self.position = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.d_model))
        if config.mup:
            self.transformer = nn.ModuleList(RVQMuTransformerEncoderLayer(config) for _ in range(config.num_layers))
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.d_model * config.ff_mult,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, config.num_layers)
        self.norm_out = nn.LayerNorm(config.d_model)
        readout_sizes = config.codebook_vocab_sizes[:1] if config.depth_decoder else config.codebook_vocab_sizes
        self.heads = nn.ModuleList(nn.Linear(config.d_model, vocab_size) for vocab_size in readout_sizes)
        self.depth_decoder = MiniMaxMusicRVQDepthDecoder(config) if config.depth_decoder else None

    def forward(self, latents: torch.Tensor, pool: torch.Tensor) -> list[torch.Tensor]:
        hidden_states = self.conv_in(latents.transpose(1, 2))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = torch.bmm(pool.to(hidden_states.dtype), hidden_states.transpose(1, 2))
        hidden_states = hidden_states + self.position[:, : pool.shape[1]].to(hidden_states.dtype)
        layers = self.transformer if isinstance(self.transformer, nn.ModuleList) else self.transformer.layers
        for layer in layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        semantic = self.heads[0](hidden_states)
        if self.depth_decoder is None:
            return [head(hidden_states) for head in self.heads]
        semantic_codes = semantic.argmax(dim=-1)
        return [semantic, *self.depth_decoder(hidden_states, semantic_codes)]


class MiniMaxMusic3Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * hidden_states).pow(2)


class DAVResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            MiniMaxMusic3Snake1d(dim),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=3 * dilation)),
            MiniMaxMusic3Snake1d(dim),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=1)),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.block(hidden_states)
        if residual.shape[-1] != hidden_states.shape[-1]:
            padding = (hidden_states.shape[-1] - residual.shape[-1]) // 2
            hidden_states = hidden_states[..., padding : hidden_states.shape[-1] - padding]
        return hidden_states + residual


class DAVEncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            DAVResidualUnit(dim // 2, 1),
            DAVResidualUnit(dim // 2, 3),
            DAVResidualUnit(dim // 2, 9),
            MiniMaxMusic3Snake1d(dim // 2),
            weight_norm(nn.Conv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class DAVEncoder(nn.Module):
    def __init__(self, encoder_dim: int = 64, rates: tuple[int, ...] = (2, 4, 8, 8), latent_dim: int = 1024):
        super().__init__()
        layers: list[nn.Module] = [weight_norm(nn.Conv1d(1, encoder_dim, kernel_size=7, padding=3))]
        for stride in rates:
            encoder_dim *= 2
            layers.append(DAVEncoderBlock(encoder_dim, stride))
        layers.extend((MiniMaxMusic3Snake1d(encoder_dim), weight_norm(nn.Conv1d(encoder_dim, latent_dim, 3, padding=1))))
        self.block = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class DAVEncoderOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DAVEncoder()
        self.mean_proj = nn.Conv1d(1024, 64, kernel_size=1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3 or waveform.shape[1] not in (1, 2):
            raise ValueError("waveform must have shape [channels, samples] or [batch, channels, samples]")
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        remainder = waveform.shape[-1] % DAV_HOP_SAMPLES
        if remainder:
            waveform = F.pad(waveform, (0, DAV_HOP_SAMPLES - remainder))
        batch, _, samples = waveform.shape
        hidden_states = self.encoder(waveform.reshape(batch * 2, 1, samples))
        return self.mean_proj(hidden_states).reshape(batch, 128, -1)


def frame_latent_starts(frame_count: int) -> list[int]:
    """Map continuous 25 Hz frames to the 44.1 kHz / 512 DAV timeline."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    return [(index * LATENT_RATE_NUM) // LATENT_RATE_DEN for index in range(frame_count + 1)]


def build_pool_matrix(bounds: list[int]) -> torch.Tensor:
    if len(bounds) < 2:
        raise ValueError("at least two frame boundaries are required")
    origin = bounds[0]
    local = [value - origin for value in bounds]
    pool = torch.zeros((len(local) - 1, local[-1]), dtype=torch.float32)
    for index, (start, end) in enumerate(zip(local[:-1], local[1:])):
        if end <= start:
            raise ValueError(f"invalid DAV span [{start}, {end}) for frame {index}")
        pool[index, start:end] = 1.0 / (end - start)
    return pool


def _clean_caption(caption: str) -> str:
    def rewrite(match: re.Match) -> str:
        parts = match.group(1).strip().split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else parts[0]

    text = _SPECIAL_TAG_RE.sub(rewrite, caption)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        lines.append(re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line).rstrip())
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", "\n".join(lines), flags=re.MULTILINE)
    return re.sub(r"\n{2,}", "\n", text.replace("• ", "").replace("    ", ""))


def _normalize_lyrics(lyrics: str) -> str:
    lines = []
    for line in lyrics.split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        lines.append(match.group(1).strip() if match else line)
    text = "\n".join(lines).replace("] ", "]\n").replace(" [", "\n[").replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)
    return f"[start]\n{text}"


def build_text_ids(tokenizer, prompt: str, lyrics: str, device: torch.device) -> torch.Tensor:
    if not prompt.strip() or not lyrics.strip():
        raise ValueError("prompt and lyrics must be non-empty")
    text = (
        f"<|im_start|><|caption_start|>{_clean_caption(prompt)}<|caption_end|>"
        f"<|lyrics_start|>{_normalize_lyrics(lyrics)}<|lyrics_end|><|im_end|><|audio_start|>"
    )
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    unconditional = input_ids.clone()
    unconditional[:, 1:-2] = AUDIO_CFG_TOKEN_ID
    return torch.cat((input_ids, unconditional), dim=0).to(device)


def _sample_top_k(
    logits: torch.Tensor,
    generator: torch.Generator | None,
    top_k: int = AR_TOP_K,
) -> torch.Tensor:
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    threshold = torch.topk(values, min(top_k, values.shape[-1]), dim=-1).values[..., -1, None]
    probabilities = torch.softmax(values.masked_fill(values < threshold, -torch.inf), dim=-1)
    sample_device = generator.device if generator is not None else probabilities.device
    return torch.multinomial(probabilities.to(sample_device), 1, generator=generator).squeeze(-1).to(values.device)


def _sample_official_depth_codes(
    language_model,
    depth_decoder,
    hidden: torch.Tensor,
    semantic_codes: torch.Tensor,
    generator: torch.Generator | None,
    cfg_scale: float = AR_CFG_SCALE,
    top_k: int = AR_TOP_K,
    greedy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = [depth_decoder.projection(hidden).unsqueeze(1)]
    semantic = language_model.model.embed_tokens(semantic_codes + AUDIO_CODE_OFFSET)
    sequence.append(depth_decoder.projection(semantic).unsqueeze(1))
    codes = [semantic_codes]
    hidden_parts = []
    for index in range(1, 8):
        depth_hidden = depth_decoder(torch.cat(sequence, dim=1))[:, -1]
        hidden_parts.append(depth_hidden[:1])
        logits = depth_decoder.audio_heads[index - 1](depth_hidden).float()
        guided = logits[1:2] + (logits[:1] - logits[1:2]) * cfg_scale
        code = (guided.argmax(dim=-1) if greedy else _sample_top_k(guided, generator, top_k)).repeat(2)
        codes.append(code)
        if index < 7:
            embedding = depth_decoder.audio_embeddings(code + (index - 1) * 1024)
            sequence.append(depth_decoder.projection(embedding).unsqueeze(1))
    return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)


def _embed_official_codes(language_model, depth_decoder, codes: torch.Tensor) -> torch.Tensor:
    semantic = language_model.model.embed_tokens(codes[..., 0] + AUDIO_CODE_OFFSET)
    offsets = torch.arange(codes.shape[-1] - 1, device=codes.device) * 1024
    acoustic = depth_decoder.audio_embeddings(codes[..., 1:] + offsets).sum(dim=-2)
    return (semantic + acoustic.to(semantic.dtype)) * (codes.shape[-1] ** -0.5)


def _sample_semantic_candidates_diffusers(
    language_model,
    hidden: torch.Tensor,
    semantic_candidates: torch.Tensor,
    cfg_scale: float,
    top_k: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    token_candidates = semantic_candidates.to(device=hidden.device, dtype=torch.long) + AUDIO_CODE_OFFSET
    token_candidates = token_candidates.unsqueeze(0)
    logits = language_model.lm_head(hidden).float()
    conditioned = logits[:1].gather(-1, token_candidates)
    unconditioned = logits[1:2].gather(-1, token_candidates)
    guided = unconditioned + (conditioned - unconditioned) * cfg_scale
    selected_index = _sample_top_k(guided, generator, min(top_k, guided.shape[-1])).view(1, 1)
    return token_candidates.gather(-1, selected_index).squeeze(-1) - AUDIO_CODE_OFFSET


@torch.inference_mode()
def rollout_codes_diffusers(
    pipeline,
    codes: torch.Tensor,
    semantic_candidates: torch.Tensor,
    *,
    prompt: str,
    lyrics: str,
    generator: torch.Generator | None = None,
    feedback_generator: torch.Generator | None = None,
    reference_interval: int | None = DEFAULT_REFERENCE_INTERVAL,
    reference_strength: float = 0.0,
    reference_feedback_mask: torch.Tensor | None = None,
    reference_warmup_codes: torch.Tensor | None = None,
    cfg_scale: float = AR_CFG_SCALE,
    top_k: int = AR_TOP_K,
) -> ReferenceRolloutTrace:
    """Generate a rollout and optionally feed reference frames back into the LM context."""
    if codes.ndim != 2 or codes.shape[1] != 8 or codes.shape[0] == 0:
        raise ValueError("codes must have shape [frames, 8]")
    if semantic_candidates.ndim != 2 or semantic_candidates.shape != (codes.shape[0], REFERENCE_CANDIDATE_COUNT):
        raise ValueError(f"semantic_candidates must have shape [frames, {REFERENCE_CANDIDATE_COUNT}]")
    if reference_interval is not None and not 1 <= reference_interval <= 10:
        raise ValueError("reference_interval must be between 1 and 10")
    if not 0.0 <= reference_strength <= 1.0:
        raise ValueError("reference_strength must be between 0 and 1")
    if reference_feedback_mask is not None:
        if reference_feedback_mask.shape != (codes.shape[0],):
            raise ValueError("reference_feedback_mask must have shape [frames]")
        if reference_strength != 0.0:
            raise ValueError("reference_strength and reference_feedback_mask are mutually exclusive")
    if 0.0 < reference_strength < 1.0 and feedback_generator is None:
        raise ValueError("feedback_generator is required for probabilistic reference feedback")
    if reference_warmup_codes is not None and reference_warmup_codes.shape != (8,):
        raise ValueError("reference_warmup_codes must have shape [8]")
    if not 1 <= top_k <= SEMANTIC_VOCAB_SIZE:
        raise ValueError(f"top_k must be between 1 and {SEMANTIC_VOCAB_SIZE}")
    language_model = pipeline.language_model
    depth_decoder = pipeline.rvq_depth_decoder
    device = next(language_model.parameters()).device
    codes = codes.to(device=device, dtype=torch.long)
    semantic_candidates = semantic_candidates.to(device=device, dtype=torch.long)
    text_ids = build_text_ids(pipeline.tokenizer, prompt, lyrics, device)
    text_output = language_model.model(inputs_embeds=language_model.model.embed_tokens(text_ids), use_cache=True)
    past = text_output.past_key_values
    last_hidden = text_output.last_hidden_state[:, -1]

    vocab_mask = torch.ones(language_model.config.vocab_size, dtype=torch.bool, device=device)
    vocab_mask[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE] = False
    vocab_mask[AUDIO_END_TOKEN_ID] = False
    if reference_warmup_codes is not None:
        warmup_feedback = reference_warmup_codes.to(device=device, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    else:
        logits = language_model.lm_head(last_hidden).float().masked_fill(vocab_mask, -torch.inf)
        conditioned, unconditioned = logits[:1], logits[1:2]
        guided = unconditioned + (conditioned - unconditioned) * cfg_scale
        threshold = torch.topk(conditioned, top_k, dim=-1).values[..., -1, None]
        warmup_token = _sample_top_k(guided.masked_fill(conditioned < threshold, -torch.inf), generator, top_k)
        if int(warmup_token.item()) == AUDIO_END_TOKEN_ID:
            raise ValueError("the selected seed ended during the required AR warm-up frame")
        warmup_semantic = (warmup_token - AUDIO_CODE_OFFSET).repeat(2)
        warmup_feedback, _ = _sample_official_depth_codes(
            language_model,
            depth_decoder,
            last_hidden,
            warmup_semantic,
            generator,
            cfg_scale=cfg_scale,
            top_k=top_k,
        )
    output = language_model.model(
        inputs_embeds=_embed_official_codes(language_model, depth_decoder, warmup_feedback).unsqueeze(1),
        past_key_values=past,
        use_cache=True,
    )
    past = output.past_key_values
    last_hidden = output.last_hidden_state[:, -1]
    all_semantic_candidates = torch.arange(SEMANTIC_VOCAB_SIZE, device=device, dtype=torch.long)
    hidden_frames = []
    generated_frames = []
    reference_feedback = []
    for frame_index in range(codes.shape[0]):
        frame_candidates = (
            semantic_candidates[frame_index]
            if reference_interval is not None and frame_index % reference_interval == 0
            else all_semantic_candidates
        )
        semantic_code = _sample_semantic_candidates_diffusers(
            language_model,
            last_hidden,
            frame_candidates,
            cfg_scale,
            top_k,
            generator,
        ).repeat(2)
        sampled_codes, depth_hidden = _sample_official_depth_codes(
            language_model,
            depth_decoder,
            last_hidden,
            semantic_code,
            generator,
            cfg_scale=cfg_scale,
            top_k=top_k,
        )
        hidden_frames.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1).cpu())
        generated_frames.append(sampled_codes[:1].cpu())
        use_reference = frame_index + 1 < codes.shape[0] and bool(
            reference_feedback_mask[frame_index]
            if reference_feedback_mask is not None
            else (
                reference_strength >= 1.0
                or (reference_strength > 0.0 and torch.rand((), generator=feedback_generator).item() < reference_strength)
            )
        )
        reference_feedback.append(use_reference)
        if frame_index + 1 < codes.shape[0]:
            feedback_codes = codes[frame_index].unsqueeze(0).repeat(2, 1) if use_reference else sampled_codes
            output = language_model.model(
                inputs_embeds=_embed_official_codes(language_model, depth_decoder, feedback_codes).unsqueeze(1),
                past_key_values=past,
                use_cache=True,
            )
            past = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1]
    return ReferenceRolloutTrace(
        frame_hiddens=torch.cat(hidden_frames, dim=0).unsqueeze(0),
        generated_codes=torch.cat(generated_frames, dim=0),
        reference_feedback=torch.tensor(reference_feedback, dtype=torch.bool),
        warmup_feedback_codes=warmup_feedback[:1].cpu(),
    )


def replay_codes_diffusers(
    pipeline,
    codes: torch.Tensor,
    semantic_candidates: torch.Tensor,
    *,
    prompt: str,
    lyrics: str,
    generator: torch.Generator | None = None,
    reference_interval: int = DEFAULT_REFERENCE_INTERVAL,
    cfg_scale: float = AR_CFG_SCALE,
    top_k: int = AR_TOP_K,
) -> torch.Tensor:
    """Generate a fixed-length rollout with periodic top-5 semantic constraints."""
    return rollout_codes_diffusers(
        pipeline,
        codes,
        semantic_candidates,
        prompt=prompt,
        lyrics=lyrics,
        generator=generator,
        reference_interval=reference_interval,
        cfg_scale=cfg_scale,
        top_k=top_k,
    ).frame_hiddens


class MiniMaxMusic3ReferenceAdapter:
    def __init__(self, dav_encoder: DAVEncoderOnly, rvq_encoder: MiniMaxMusicRVQEncoder):
        self.dav_encoder = dav_encoder.eval()
        self.rvq_encoder = rvq_encoder.eval()

    @classmethod
    def from_files(cls, encoder_file: str | Path, config_file: str | Path, dav_file: str | Path):
        import json

        from safetensors.torch import load_file

        config = RVQEncoderConfig.from_dict(json.loads(Path(config_file).read_text(encoding="utf-8")))
        rvq_encoder = MiniMaxMusicRVQEncoder(config)
        rvq_encoder.load_state_dict(load_file(str(encoder_file)), strict=True)
        dav_encoder = DAVEncoderOnly()
        checkpoint = torch.load(dav_file, map_location="cpu", weights_only=True)
        dav_state = {key: value for key, value in checkpoint.items() if key.startswith(("encoder.", "mean_proj."))}
        dav_encoder.load_state_dict(dav_state, strict=True)
        return cls(dav_encoder, rvq_encoder)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = COLLECTION_REPO_ID,
        *,
        encoder_file: str = DEFAULT_ENCODER_FILE,
        revision: str | None = None,
        cache_dir: str | None = None,
        official_model_revision: str | None = None,
    ):
        from huggingface_hub import hf_hub_download

        encoder_path = hf_hub_download(repo_id, encoder_file, revision=revision, cache_dir=cache_dir)
        config_path = hf_hub_download(
            repo_id,
            str(Path(encoder_file).with_suffix(".json")),
            revision=revision,
            cache_dir=cache_dir,
        )
        dav_path = hf_hub_download(
            OFFICIAL_MODEL_REPO_ID,
            DAV_FILE,
            revision=official_model_revision,
            cache_dir=cache_dir,
        )
        return cls.from_files(encoder_path, config_path, dav_path)

    @staticmethod
    def _resample(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim == 3:
            if waveform.shape[0] != 1:
                raise ValueError("reference audio batch size must be one")
            waveform = waveform[0]
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [samples], [channels, samples], or [1, channels, samples]")
        if sample_rate != SAMPLE_RATE:
            import torchaudio

            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
        return waveform.float()

    def _encode_dav(
        self,
        waveform: torch.Tensor,
        device: torch.device,
        chunk_seconds: float,
    ) -> torch.Tensor:
        if chunk_seconds <= 0:
            raise ValueError("dav_chunk_seconds must be positive")
        remainder = waveform.shape[-1] % DAV_HOP_SAMPLES
        if remainder:
            waveform = F.pad(waveform, (0, DAV_HOP_SAMPLES - remainder))
        core_samples = max(
            DAV_HOP_SAMPLES,
            int(chunk_seconds * SAMPLE_RATE) // DAV_HOP_SAMPLES * DAV_HOP_SAMPLES,
        )
        if waveform.shape[-1] <= core_samples:
            return self.dav_encoder(waveform.unsqueeze(0).to(device=device, dtype=torch.float32))[0]

        latent_chunks = []
        for core_start in range(0, waveform.shape[-1], core_samples):
            core_end = min(core_start + core_samples, waveform.shape[-1])
            read_start = max(0, core_start - DAV_CHUNK_OVERLAP_SAMPLES)
            read_end = min(waveform.shape[-1], core_end + DAV_CHUNK_OVERLAP_SAMPLES)
            chunk = waveform[:, read_start:read_end].unsqueeze(0).to(device=device, dtype=torch.float32)
            chunk_latents = self.dav_encoder(chunk)[0]
            keep_start = (core_start - read_start) // DAV_HOP_SAMPLES
            keep_length = (core_end - core_start) // DAV_HOP_SAMPLES
            latent_chunks.append(chunk_latents[:, keep_start : keep_start + keep_length])
        return torch.cat(latent_chunks, dim=-1)

    @torch.inference_mode()
    def _predict_codes(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        device: str | torch.device | None = None,
        encoder_dtype: torch.dtype | None = None,
        semantic_top_k: int | None = None,
        offload_after: bool = True,
        dav_chunk_seconds: float = 30.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if semantic_top_k is not None and not 1 <= semantic_top_k <= self.rvq_encoder.config.codebook_vocab_sizes[0]:
            raise ValueError(f"semantic_top_k must be between 1 and {self.rvq_encoder.config.codebook_vocab_sizes[0]}")
        waveform = self._resample(waveform, sample_rate)
        original_samples = waveform.shape[-1]
        if original_samples < SAMPLE_RATE // FRAME_RATE:
            raise ValueError("reference audio must contain at least one 25 Hz frame")
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        encoder_dtype = encoder_dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)

        self.dav_encoder.to(device=device, dtype=torch.float32)
        latents = self._encode_dav(waveform, device, dav_chunk_seconds)
        if offload_after:
            latents = latents.cpu()
            self.dav_encoder.to("cpu")

        frame_count = int(original_samples * FRAME_RATE // SAMPLE_RATE)
        bounds = frame_latent_starts(frame_count)
        while frame_count and bounds[-1] > latents.shape[-1]:
            frame_count -= 1
            bounds = frame_latent_starts(frame_count)
        if not frame_count:
            raise ValueError("DAV encoding produced no complete reference frames")

        window_size = self.rvq_encoder.config.max_position_embeddings
        regular_starts = list(range(0, max(frame_count - window_size + 1, 0), window_size))
        if frame_count >= window_size:
            regular_starts.append(frame_count - window_size if not regular_starts else regular_starts[-1])
            regular_starts = sorted(set(regular_starts))
            tail = frame_count - window_size
            if tail not in regular_starts:
                regular_starts.append(tail)
        else:
            regular_starts = [0]

        predictions = torch.empty((frame_count, 8), dtype=torch.long)
        semantic_candidates = (
            torch.empty((frame_count, semantic_top_k), dtype=torch.long) if semantic_top_k is not None else None
        )
        assigned = torch.zeros(frame_count, dtype=torch.bool)
        self.rvq_encoder.to(device=device, dtype=encoder_dtype)
        autocast = torch.autocast(device.type, dtype=encoder_dtype) if device.type == "cuda" else nullcontext()
        with autocast:
            for frame_start in regular_starts:
                frame_end = min(frame_start + window_size, frame_count)
                local_bounds = bounds[frame_start : frame_end + 1]
                latent_start, latent_end = local_bounds[0], local_bounds[-1]
                window_latents = latents[:, latent_start:latent_end].transpose(0, 1).to(device=device, dtype=encoder_dtype)
                pool = build_pool_matrix(local_bounds).to(device)
                logits = self.rvq_encoder(window_latents.unsqueeze(0), pool.unsqueeze(0))
                predicted = torch.stack([head.argmax(dim=-1)[0] for head in logits], dim=-1).cpu()
                window_semantic_candidates = (
                    logits[0].topk(semantic_top_k, dim=-1).indices[0].cpu() if semantic_top_k is not None else None
                )
                take = ~assigned[frame_start:frame_end]
                predictions[frame_start:frame_end][take] = predicted[take]
                if semantic_candidates is not None:
                    semantic_candidates[frame_start:frame_end][take] = window_semantic_candidates[take]
                assigned[frame_start:frame_end][take] = True
        if offload_after:
            self.rvq_encoder.to("cpu")
        if not assigned.all():
            raise RuntimeError("RVQ window inference did not cover every reference frame")
        return predictions, semantic_candidates

    def predict_codes(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        device: str | torch.device | None = None,
        encoder_dtype: torch.dtype | None = None,
        offload_after: bool = True,
        dav_chunk_seconds: float = 30.0,
    ) -> torch.Tensor:
        predictions, _ = self._predict_codes(
            waveform,
            sample_rate,
            device=device,
            encoder_dtype=encoder_dtype,
            offload_after=offload_after,
            dav_chunk_seconds=dav_chunk_seconds,
        )
        return predictions

    def predict_codes_with_semantic_candidates(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        semantic_top_k: int = 5,
        device: str | torch.device | None = None,
        encoder_dtype: torch.dtype | None = None,
        offload_after: bool = True,
        dav_chunk_seconds: float = 30.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predictions, semantic_candidates = self._predict_codes(
            waveform,
            sample_rate,
            device=device,
            encoder_dtype=encoder_dtype,
            semantic_top_k=semantic_top_k,
            offload_after=offload_after,
            dav_chunk_seconds=dav_chunk_seconds,
        )
        assert semantic_candidates is not None
        return predictions, semantic_candidates

    def encode_reference(
        self,
        pipeline,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        prompt: str,
        lyrics: str,
        generator: torch.Generator | None = None,
        device: str | torch.device | None = None,
        reference_interval: int = DEFAULT_REFERENCE_INTERVAL,
        cfg_scale: float = AR_CFG_SCALE,
        top_k: int = AR_TOP_K,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codes, semantic_candidates = self.predict_codes_with_semantic_candidates(
            waveform,
            sample_rate,
            semantic_top_k=REFERENCE_CANDIDATE_COUNT,
            device=device,
        )
        frame_hiddens = replay_codes_diffusers(
            pipeline,
            codes,
            semantic_candidates,
            prompt=prompt,
            lyrics=lyrics,
            generator=generator,
            reference_interval=reference_interval,
            cfg_scale=cfg_scale,
            top_k=top_k,
        )
        return frame_hiddens, codes


def install_diffusers_reference_adapter() -> None:
    """Allow the official modular pipeline to accept precomputed frame_hiddens."""
    from diffusers.modular_pipelines.minimax_music3.encoders import (
        MiniMaxMusic3AutoregressiveStep,
        MiniMaxMusic3TokenizeStep,
    )
    from diffusers.modular_pipelines.modular_pipeline_utils import InputParam

    if getattr(MiniMaxMusic3TokenizeStep, "_simpletuner_reference_patch", False):
        return

    original_tokenize_call = MiniMaxMusic3TokenizeStep.__call__
    original_generate_call = MiniMaxMusic3AutoregressiveStep.__call__
    original_tokenize_inputs = MiniMaxMusic3TokenizeStep.inputs.fget
    original_generate_inputs = MiniMaxMusic3AutoregressiveStep.inputs.fget

    def tokenize_inputs(self):
        inputs = original_tokenize_inputs(self)
        for value in inputs:
            if value.name in {"prompt", "lyrics"}:
                value.required = False
        inputs.append(InputParam("frame_hiddens", default=None, type_hint=torch.Tensor))
        return inputs

    def generate_inputs(self):
        inputs = original_generate_inputs(self)
        for value in inputs:
            if value.name == "text_ids":
                value.required = False
        inputs.append(InputParam("frame_hiddens", default=None, type_hint=torch.Tensor))
        return inputs

    def tokenize_call(self, components, state):
        block_state = self.get_block_state(state)
        if getattr(block_state, "frame_hiddens", None) is not None:
            return components, state
        return original_tokenize_call(self, components, state)

    def generate_call(self, components, state):
        block_state = self.get_block_state(state)
        if getattr(block_state, "frame_hiddens", None) is not None:
            return components, state
        return original_generate_call(self, components, state)

    MiniMaxMusic3TokenizeStep.inputs = property(tokenize_inputs)
    MiniMaxMusic3AutoregressiveStep.inputs = property(generate_inputs)
    MiniMaxMusic3TokenizeStep.__call__ = tokenize_call
    MiniMaxMusic3AutoregressiveStep.__call__ = generate_call
    MiniMaxMusic3TokenizeStep._simpletuner_reference_patch = True
    MiniMaxMusic3AutoregressiveStep._simpletuner_reference_patch = True
