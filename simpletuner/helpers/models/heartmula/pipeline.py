# Copyright 2025 SimpleTuner contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from .codec import HeartCodec
from .gen_config import HeartMuLaGenConfig
from .modeling_heartmula import HeartMuLaModel

logger = logging.getLogger(__name__)


@dataclass
class HeartMuLaPipelineOutput:
    audios: list[torch.Tensor]


class HeartMuLaGenPipeline:
    def __init__(
        self,
        transformer: HeartMuLaModel,
        audio_codec: HeartCodec,
        tokenizer: Tokenizer,
        gen_config: HeartMuLaGenConfig,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if transformer is None:
            raise ValueError("HeartMuLaGenPipeline requires a transformer model.")
        if audio_codec is None:
            raise ValueError("HeartMuLaGenPipeline requires a HeartCodec instance.")
        self.transformer = transformer
        self.audio_codec = audio_codec
        self.tokenizer = tokenizer
        self.gen_config = gen_config
        self.device = device or next(transformer.parameters()).device
        self.dtype = dtype or next(transformer.parameters()).dtype

        model_config = getattr(transformer, "config", None)
        if model_config is None and hasattr(transformer, "base_model"):
            model_config = getattr(getattr(transformer.base_model, "model", None), "config", None)
        if model_config is None:
            raise ValueError("HeartMuLaGenPipeline could not resolve model config for codebook sizing.")

        self.model_config = model_config
        num_codebooks = model_config.audio_num_codebooks
        if audio_codec.config.num_quantizers != num_codebooks:
            raise ValueError(
                "HeartMuLa/HeartCodec codebook mismatch: "
                f"model has {num_codebooks}, codec has {audio_codec.config.num_quantizers}."
            )
        self._parallel_number = num_codebooks + 1

        codec_device = next(audio_codec.parameters()).device
        if codec_device != self.device:
            audio_codec.to(self.device)

    def _read_text(self, text: str) -> str:
        if os.path.isfile(text):
            with open(text, encoding="utf-8") as handle:
                return handle.read()
        return text

    def _encode_text(self, text: str) -> list[int]:
        token_ids = self.tokenizer.encode(text).ids
        if not token_ids or token_ids[0] != self.gen_config.text_bos_id:
            token_ids = [self.gen_config.text_bos_id] + token_ids
        if token_ids[-1] != self.gen_config.text_eos_id:
            token_ids = token_ids + [self.gen_config.text_eos_id]
        return token_ids

    def _build_prompt_tokens(self, tags: str, lyrics: str, guidance_scale: float):
        tags = self._read_text(tags).strip().lower()
        if not tags.startswith("<tag>"):
            tags = f"<tag>{tags}"
        if not tags.endswith("</tag>"):
            tags = f"{tags}</tag>"
        lyrics = self._read_text(lyrics).strip().lower()

        tag_ids = self._encode_text(tags)
        lyric_ids = self._encode_text(lyrics)

        prompt_len = len(tag_ids) + 1 + len(lyric_ids)
        tokens = torch.full(
            (prompt_len, self._parallel_number),
            fill_value=self.gen_config.empty_id,
            dtype=torch.long,
        )
        tokens[: len(tag_ids), -1] = torch.tensor(tag_ids, dtype=torch.long)
        tokens[len(tag_ids) + 1 :, -1] = torch.tensor(lyric_ids, dtype=torch.long)

        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens_mask[:, -1] = True

        if guidance_scale != 1.0:
            tokens = torch.cat([tokens.unsqueeze(0), tokens.unsqueeze(0)], dim=0)
            tokens_mask = torch.cat([tokens_mask.unsqueeze(0), tokens_mask.unsqueeze(0)], dim=0)
            uncond_mask = torch.cat(
                [
                    torch.zeros(1, dtype=torch.bool),
                    torch.ones(1, dtype=torch.bool),
                ],
                dim=0,
            )
        else:
            tokens = tokens.unsqueeze(0)
            tokens_mask = tokens_mask.unsqueeze(0)
            uncond_mask = None

        return tokens, tokens_mask, uncond_mask, prompt_len

    def _past_length(self, past_key_values) -> int:
        if past_key_values is None:
            return 0
        try:
            return past_key_values[0][0].shape[-2]
        except Exception:
            return 0

    def _attention_mask(self, batch_size: int, total_length: int, device: torch.device) -> torch.Tensor:
        return torch.ones((batch_size, total_length), device=device, dtype=torch.long)

    def _sample_topk(
        self,
        logits: torch.Tensor,
        topk: int,
        temperature: float,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        logits = logits / temperature
        vocab = logits.shape[-1]
        if topk is None or topk <= 0 or topk >= vocab:
            probs = torch.softmax(logits, dim=-1)
        else:
            topk_vals, _ = torch.topk(logits, topk, dim=-1)
            cutoff = topk_vals[..., -1, None]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs.float(), num_samples=1, generator=generator)

    def _apply_cfg(self, logits: torch.Tensor, guidance_scale: float) -> torch.Tensor:
        if guidance_scale <= 1.0 or logits.shape[0] % 2 != 0:
            return logits, None
        half = logits.shape[0] // 2
        cond = logits[:half]
        uncond = logits[half:]
        guided = uncond + (cond - uncond) * guidance_scale
        return guided, half

    def _sample_audio_frame(
        self,
        hidden_state: torch.Tensor,
        guidance_scale: float,
        topk: int,
        temperature: float,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        logits0 = self.transformer.codebook0_head(hidden_state)
        guided_logits, half = self._apply_cfg(logits0, guidance_scale)
        sample0 = self._sample_topk(guided_logits, topk, temperature, generator)
        if half is not None:
            sample0 = sample0.repeat(2, 1)

        num_codebooks = self.model_config.audio_num_codebooks
        samples = [sample0]

        decoder_inputs = torch.cat([hidden_state.unsqueeze(1), self.transformer._embed_audio(0, sample0)], dim=1)
        decoder_inputs = self.transformer.projection(decoder_inputs)
        batch = decoder_inputs.shape[0]
        positions = torch.arange(decoder_inputs.shape[1], device=decoder_inputs.device).unsqueeze(0).repeat(batch, 1)
        attn_mask = self._attention_mask(batch, decoder_inputs.shape[1], decoder_inputs.device)
        decoder_out = self.transformer.forward_decoder(
            inputs_embeds=decoder_inputs,
            attention_mask=attn_mask,
            position_ids=positions,
            use_cache=True,
        )
        past = decoder_out.past_key_values
        last_hidden = decoder_out.last_hidden_state[:, -1, :]

        for idx in range(1, num_codebooks):
            logits = torch.matmul(last_hidden, self.transformer.audio_head[idx - 1])
            guided_logits, half = self._apply_cfg(logits, guidance_scale)
            sample = self._sample_topk(guided_logits, topk, temperature, generator)
            if half is not None:
                sample = sample.repeat(2, 1)
            samples.append(sample)

            if idx == num_codebooks - 1:
                break
            embed = self.transformer._embed_audio(idx, sample)
            embed = self.transformer.projection(embed)
            past_len = self._past_length(past)
            pos = torch.full((batch, 1), past_len, device=embed.device, dtype=torch.long)
            attn_mask = self._attention_mask(batch, past_len + 1, embed.device)
            decoder_out = self.transformer.forward_decoder(
                inputs_embeds=embed,
                attention_mask=attn_mask,
                position_ids=pos,
                past_key_values=past,
                use_cache=True,
            )
            past = decoder_out.past_key_values
            last_hidden = decoder_out.last_hidden_state[:, -1, :]

        return torch.cat(samples, dim=1)

    @torch.inference_mode()
    def __call__(
        self,
        tags: str,
        lyrics: str = "",
        *,
        guidance_scale: float = 1.5,
        audio_duration: float = 30.0,
        max_audio_length_ms: Optional[int] = None,
        temperature: float = 1.0,
        topk: int = 50,
        cfg_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> HeartMuLaPipelineOutput:
        del kwargs
        if cfg_scale is not None:
            guidance_scale = float(cfg_scale)
        if max_audio_length_ms is None:
            max_audio_length_ms = int(audio_duration * 1000)
        else:
            audio_duration = max_audio_length_ms / 1000.0
        max_audio_frames = max(int(max_audio_length_ms // 80), 1)

        tokens, tokens_mask, uncond_mask, prompt_len = self._build_prompt_tokens(tags, lyrics, guidance_scale)
        tokens = tokens.to(self.device)
        tokens_mask = tokens_mask.to(self.device)
        if uncond_mask is not None:
            uncond_mask = uncond_mask.to(self.device)

        attention_mask = tokens_mask.any(dim=-1).to(dtype=torch.long)
        positions = torch.arange(prompt_len, device=self.device).unsqueeze(0).repeat(tokens.shape[0], 1)

        self.transformer.eval()
        self.audio_codec.eval()
        autocast_ctx = (
            torch.autocast(device_type=self.device.type, dtype=self.dtype) if self.device.type != "cpu" else nullcontext()
        )
        with autocast_ctx:
            hidden = self.transformer._build_backbone_inputs(tokens, tokens_mask, uncond_mask=uncond_mask)
            backbone_out = self.transformer.forward_backbone(
                inputs_embeds=hidden,
                attention_mask=attention_mask,
                position_ids=positions,
                use_cache=True,
            )
            past_key_values = backbone_out.past_key_values
            current_hidden = backbone_out.last_hidden_state[:, -1, :]

            frames = []
            position = prompt_len
            for _ in range(max_audio_frames):
                frame_tokens = self._sample_audio_frame(
                    current_hidden,
                    guidance_scale,
                    topk,
                    temperature,
                    generator,
                )
                if guidance_scale > 1.0 and frame_tokens.shape[0] % 2 == 0:
                    cond_tokens = frame_tokens[: frame_tokens.shape[0] // 2]
                else:
                    cond_tokens = frame_tokens

                frames.append(cond_tokens)
                if torch.any(cond_tokens >= self.gen_config.audio_eos_id):
                    break

                frame_inputs = torch.full(
                    (frame_tokens.shape[0], 1, self._parallel_number),
                    fill_value=self.gen_config.empty_id,
                    dtype=torch.long,
                    device=self.device,
                )
                frame_inputs[:, 0, : self.model_config.audio_num_codebooks] = frame_tokens
                frame_mask = torch.zeros_like(frame_inputs, dtype=torch.bool)
                frame_mask[:, 0, : self.model_config.audio_num_codebooks] = True

                frame_embeds = self.transformer._build_backbone_inputs(frame_inputs, frame_mask, uncond_mask=uncond_mask)
                attn_mask = self._attention_mask(frame_inputs.shape[0], position + 1, self.device)
                pos = torch.full((frame_inputs.shape[0], 1), position, device=self.device, dtype=torch.long)
                backbone_out = self.transformer.forward_backbone(
                    inputs_embeds=frame_embeds,
                    attention_mask=attn_mask,
                    position_ids=pos,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = backbone_out.past_key_values
                current_hidden = backbone_out.last_hidden_state[:, -1, :]
                position += 1

        if not frames:
            raise ValueError("HeartMuLa produced no audio frames.")

        frame_tensor = torch.stack(frames, dim=1)
        frame_tensor = frame_tensor.transpose(1, 2)
        audios: list[torch.Tensor] = []
        for idx in range(frame_tensor.shape[0]):
            decoded = self.audio_codec.detokenize(
                frame_tensor[idx].to(self.device),
                duration=audio_duration,
                guidance_scale=1.25,
                disable_progress=True,
                device=self.device,
            )
            audios.append(decoded)

        return HeartMuLaPipelineOutput(audios=audios)

    @classmethod
    def _resolve_repo_path(cls, repo_or_path: str) -> str:
        if repo_or_path and os.path.exists(repo_or_path):
            return repo_or_path
        logger.info("Downloading HeartMuLa assets from %s", repo_or_path)
        return snapshot_download(repo_or_path)

    @classmethod
    def _resolve_gen_assets(cls, model_path: str) -> str:
        candidates = []
        if model_path:
            path = Path(model_path)
            candidates.extend([path, path.parent, path / "HeartMuLaGen", path.parent / "HeartMuLaGen"])
        for candidate in candidates:
            if candidate is None:
                continue
            tokenizer_path = candidate / "tokenizer.json"
            gen_path = candidate / "gen_config.json"
            if tokenizer_path.is_file() and gen_path.is_file():
                return str(candidate)
        return cls._resolve_repo_path("HeartMuLa/HeartMuLaGen")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        transformer: Optional[HeartMuLaModel] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "HeartMuLaGenPipeline":
        del kwargs
        model_path = cls._resolve_repo_path(pretrained_model_name_or_path)

        if transformer is None:
            transformer = HeartMuLaModel.from_pretrained(model_path, torch_dtype=torch_dtype)
        transformer.eval()

        codec_path = None
        if os.path.isdir(model_path):
            candidate = os.path.join(model_path, "HeartCodec-oss")
            if os.path.isdir(candidate):
                codec_path = candidate
        if codec_path is None:
            codec_path = cls._resolve_repo_path("HeartMuLa/HeartCodec-oss")
        audio_codec = HeartCodec.from_pretrained(codec_path)

        assets_path = cls._resolve_gen_assets(model_path)
        tokenizer_path = os.path.join(assets_path, "tokenizer.json")
        gen_config_path = os.path.join(assets_path, "gen_config.json")
        if not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(f"HeartMuLa tokenizer.json not found at {tokenizer_path}.")
        if not os.path.isfile(gen_config_path):
            raise FileNotFoundError(f"HeartMuLa gen_config.json not found at {gen_config_path}.")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        gen_config = HeartMuLaGenConfig.from_file(gen_config_path)

        device = next(transformer.parameters()).device
        dtype = next(transformer.parameters()).dtype
        return cls(
            transformer=transformer,
            audio_codec=audio_codec,
            tokenizer=tokenizer,
            gen_config=gen_config,
            device=device,
            dtype=dtype,
        )
