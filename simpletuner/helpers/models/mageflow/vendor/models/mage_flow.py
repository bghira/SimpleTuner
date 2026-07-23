from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange, repeat
from loguru import logger
from pydantic import BaseModel, Field
from torch import Tensor

from .modules._attn_backend import set_attn_backend
from .modules.mage_layers import (
    AdaLayerNormContinuous,
    MageFlowEmbedRope,
    MageFlowTimestepProjEmbeddings,
    MageFlowTransformerBlock,
    RMSNorm,
)
from .modules.text_encoder import TextEncoder, qwen3_patch_forward


class ModelConfig(BaseModel):
    static_shift: float = Field(
        default=6.0,
        description="Static shift value for the z-image time-shift schedule (the only " "supported schedule). Default: 6.0.",
    )
    vae_path: str = Field(...)
    model_structure: dict = Field(default_factory=dict)
    txt_enc_path: str = Field(...)
    txt_max_length: int = Field(default=4096)
    pretrained_model_name_or_path: str | None = Field(default=None)
    pretrained_full_model_path: str | None = Field(default=None)  # Load full model weights (DiT + txt_enc + vae)
    packing: bool = Field(default=False)
    vae_sample_posterior: bool = Field(default=True)  # Sample (vs mode) from VAE posterior at encode time (Flux2 + CoD)
    vae_encoder_only: bool = Field(default=False)  # Skip loading VAE decoder to save GPU memory (training only, MageVAE)
    compile_vae_encoder: bool = Field(default=False)  # torch.compile VAE encoder to reduce CUDA kernel launch overhead
    attn_type: str = Field(
        default="flash2",
        description="Flash-attn backend used by both the DiT (mage_layers) "
        "and the HF text encoder (text_encoder). One of: 'flash2' (default) or 'flash4'.",
    )


@dataclass
class MageFlowParams:
    in_channels: int
    out_channels: int
    context_in_dim: int
    hidden_size: int
    num_heads: int
    depth: int
    axes_dim: list[int]
    checkpoint: bool
    patch_size: int = 1


class MageFlow(nn.Module):
    def __init__(self, params: MageFlowParams):
        super().__init__()
        self.params = params
        self.checkpoint = params.checkpoint
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.inner_dim = params.hidden_size  # num_attention_heads * attention_head_dim
        self.axes_dim = params.axes_dim
        self.num_attention_heads = params.num_heads
        self.attention_head_dim = self.inner_dim // self.num_attention_heads
        self.patch_size = params.patch_size
        assert sum(self.axes_dim) == self.attention_head_dim

        self.pos_embed = MageFlowEmbedRope(theta=10000, axes_dim=self.axes_dim, scale_rope=True)
        self.img_in = nn.Linear(self.in_channels, self.inner_dim)
        self.txt_norm = RMSNorm(params.context_in_dim, eps=1e-6)
        self.txt_in = nn.Linear(params.context_in_dim, self.inner_dim)

        self.time_text_embed = MageFlowTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                MageFlowTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                )
                for _ in range(params.depth)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, self.patch_size * self.patch_size * self.out_channels, bias=True)

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        timesteps: Tensor,
        img_shapes=None,
        img_cu_seqlens: Tensor | None = None,
        txt_cu_seqlens: Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # Prepare vision RoPE (msrope); text tokens are not rotated.
        ms_pe = self.pos_embed(img_shapes, device=img.device)

        img = self.img_in(img)
        txt = self.txt_norm(txt)

        timesteps = timesteps.to(img.dtype)
        temb = self.time_text_embed(timesteps, img)

        txt = self.txt_in(txt)
        txt_vec = torch.zeros(txt.shape[0], self.inner_dim, dtype=txt.dtype, device=txt.device)

        temb = temb + txt_vec

        attention_kwargs = attention_kwargs or {}

        for _index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.checkpoint:
                txt, img = torch.utils.checkpoint.checkpoint(
                    block,
                    img,  # hidden_states
                    txt,  # encoder_hidden_states
                    temb,  # temb
                    ms_pe,  # image_rotary_emb
                    txt_cu_seqlens,  # txt_cu_lens
                    img_cu_seqlens,  # img_cu_lens
                    use_reentrant=False,
                )

            else:
                txt, img = block(
                    hidden_states=img,
                    encoder_hidden_states=txt,
                    txt_cu_lens=txt_cu_seqlens,
                    img_cu_lens=img_cu_seqlens,
                    temb=temb,
                    image_rotary_emb=ms_pe,
                    joint_attention_kwargs=attention_kwargs,
                )

        # Use only the image part (hidden_states) from the dual-stream blocks
        img = self.norm_out(
            img,
            temb,
            cu_seqlens=img_cu_seqlens,
        )
        img = self.proj_out(img)
        return img


class MageFlowModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        set_attn_backend(getattr(config, "attn_type", "flash2"))
        self.patch_text_encoder_forward()
        self.vae = self.load_vae()
        self.transformer = self.load_transformer()
        self.txt_enc = self.load_text_enc()

        # Optionally override all components from a full model checkpoint (e.g. ema.pt)
        full_path = getattr(self.config, "pretrained_full_model_path", None)
        if full_path is not None:
            import os

            if os.path.exists(full_path):
                logger.info(f"Loading full model weights from {full_path}")
                sd = torch.load(full_path, map_location="cpu")
                # Handle wrapped EMA format: {'ema_state_dict': ..., ...}
                if isinstance(sd, dict) and "ema_state_dict" in sd:
                    sd = sd["ema_state_dict"]
                missing, unexpected = self.load_state_dict(sd, strict=False)
                if missing:
                    logger.warning(f"Full model load missing keys ({len(missing)}): {missing[:5]}...")
                if unexpected:
                    logger.warning(f"Full model load unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
                logger.info("Full model weights loaded successfully.")
            else:
                logger.warning(f"pretrained_full_model_path not found: {full_path}")

        # Freeze VAE and Text Encoder
        self.vae.requires_grad_(False)

        # Drop VAE decoder to save GPU memory (training only, decoder unused during training)
        if self.config.vae_encoder_only:
            from .modules.mage_vae import MageVAE

            if isinstance(self.vae, MageVAE):
                decoder_params = sum(p.numel() for p in self.vae.decoder_model.parameters()) / 1e6
                self.vae.decoder_model = None
            elif hasattr(self.vae, "decoder"):
                decoder_params = sum(p.numel() for p in self.vae.decoder.parameters()) / 1e6
                self.vae.decoder = None
            else:
                decoder_params = 0
            logger.info(f"vae_encoder_only=True: dropped VAE decoder ({decoder_params:.1f}M params) to save memory")

        # NOTE: VAE encoder torch.compile() is deferred to
        # maybe_compile_vae_encoder(), called after checkpoint load. Reason:
        # avoid wasted compile work before load_checkpoint overwrites weights.
        # The save-side _unwrap_compiled_submodules guard in DeepSpeedTrainer
        # is a belt-and-suspenders defense against any future code that
        # re-introduces the function-style ``module = torch.compile(module)``
        # pattern (which does pollute state_dict with ``_orig_mod.``).

        # Text encoder is always frozen (inference only).
        self.txt_enc.requires_grad_(False)
        logger.info(f"{sum([p.numel() for p in self.transformer.parameters() if p.requires_grad]) / 1000000} M parameters")

    def patch_text_encoder_forward(self):
        qwen3_patch_forward()
        logger.info("Patched Qwen3-VL text encoder forward methods")

    def maybe_compile_vae_encoder(self) -> None:
        """Compile the VAE encoder with torch.compile() to fuse small ops and
        reduce CUDA kernel launch overhead.

        Uses ``nn.Module.compile()`` (in-place) for the encoder so the module
        hierarchy and parameter names are unchanged — ``state_dict()`` keeps
        clean keys (no ``_orig_mod.`` prefix), and checkpoints stay
        interchangeable with the non-compiled path.

        For the MageVAE branch we still assign ``torch.compile(...)`` to a
        method (``_encode_moments``); methods aren't ``nn.Module``s so this
        does not pollute ``state_dict()``.

        Idempotent: safe to call multiple times; already-compiled modules are
        detected and skipped.
        """
        if not getattr(self.config, "compile_vae_encoder", False):
            return
        torch.set_float32_matmul_precision("high")
        from .modules.mage_vae import MageVAE

        if isinstance(self.vae, MageVAE):
            fn = self.vae._encode_moments
            if hasattr(fn, "_torchdynamo_orig_callable") or hasattr(fn, "_orig_mod"):
                return  # already compiled
            self.vae._encode_moments = torch.compile(fn, dynamic=True)
            logger.info("compile_vae_encoder=True: compiled MageVAE._encode_moments")
        elif hasattr(self.vae, "encoder"):
            if getattr(self.vae.encoder, "_compiled_call_impl", None) is not None:
                return  # already compiled
            self.vae.encoder.compile()
            logger.info("compile_vae_encoder=True: compiled VAE encoder (in-place)")

    def load_text_enc(self):
        return TextEncoder(
            model_name=self.config.txt_enc_path,
            version=self.config.txt_enc_path,
            tokenizer_max_length=self.config.txt_max_length,
            torch_dtype=torch.bfloat16,
            prompt_template=None,
            dit_structure=self.config.model_structure,
            use_packed_text_infer=self.config.packing,
            attn_type=getattr(self.config, "attn_type", "flash2"),
        )

    def load_vae(self):
        from .modules.mage_vae import MageVAE

        return MageVAE(
            ckpt_path=self.config.vae_path,
            sample_posterior=self.config.vae_sample_posterior,
        )

    def load_transformer(self):
        # Imported lazily to avoid a circular import: ``utils`` imports MageFlow /
        # MageFlowParams from this module.
        from .utils import load_model

        return load_model(
            dit_structure=self.config.model_structure,
            pretrain_path=self.config.pretrained_model_name_or_path,
        )

    def compile(self):
        self.transformer.compile()

    def compute_vae_encodings(
        self,
        pixel_values: torch.Tensor | list[torch.Tensor],
        with_ids: bool = True,
    ):
        if isinstance(pixel_values, list):
            # All same resolution → batch encode via the tensor path
            if len(pixel_values) > 1 and len({img.shape for img in pixel_values}) == 1:
                stacked = torch.stack(pixel_values, dim=0)
                result = self.compute_vae_encodings(stacked, with_ids=with_ids)
                # Repack from [N, L, C] batch format to [1, N*L, C] packed format
                if with_ids:
                    model_input, img_shapes, img_ids = result
                    model_input = model_input.reshape(1, -1, model_input.shape[-1])
                    img_ids = img_ids.reshape(1, -1, img_ids.shape[-1])
                    return model_input, img_shapes, img_ids
                model_input, img_shapes = result
                model_input = model_input.reshape(1, -1, model_input.shape[-1])
                return model_input, img_shapes

            # Packed / variable-size images
            model_inputs = []
            img_shapes = []
            img_ids_list = []

            def _append(latents):
                _, _, h, w = latents.shape
                img_shapes.append([(1, h, w)])
                model_inputs.append(rearrange(latents, "b c h w -> b (h w) c").squeeze(0))
                if with_ids:
                    ids = torch.zeros(h, w, 3, device=latents.device)
                    ids[..., 1] = ids[..., 1] + torch.arange(h, device=latents.device)[:, None]
                    ids[..., 2] = ids[..., 2] + torch.arange(w, device=latents.device)[None, :]
                    img_ids_list.append(rearrange(ids, "h w c -> (h w) c"))

            # MageVAE encoder is launch-bound on B=1; group same-shape images
            # in the pack into one batched encode call.
            if len(pixel_values) > 1:
                groups: dict[tuple[int, int], list[int]] = {}
                for i, img in enumerate(pixel_values):
                    key = (int(img.shape[-2]), int(img.shape[-1]))
                    groups.setdefault(key, []).append(i)
                latents_per_idx = [None] * len(pixel_values)
                for (h, w), idxs in groups.items():
                    batch = torch.stack([pixel_values[i] for i in idxs], dim=0)
                    batch = batch.to(memory_format=torch.contiguous_format).float()
                    batch = batch.to(self.vae.device, dtype=self.vae.dtype)
                    with torch.no_grad():
                        lat = self.vae.encode(batch)  # [B, 128, H/16, W/16]
                    for j, i in enumerate(idxs):
                        latents_per_idx[i] = lat[j : j + 1]
                for latents in latents_per_idx:
                    _append(latents)
            else:
                for img in pixel_values:
                    img = img.unsqueeze(0).to(memory_format=torch.contiguous_format).float()
                    img = img.to(self.vae.device, dtype=self.vae.dtype)
                    with torch.no_grad():
                        latents = self.vae.encode(img)  # [1, 128, H/16, W/16]
                    _append(latents)

            model_input = torch.cat(model_inputs, dim=0).unsqueeze(0)
            if with_ids:
                img_ids = torch.cat(img_ids_list, dim=0).unsqueeze(0)
                return model_input, img_shapes, img_ids
            return model_input, img_shapes

        # Tensor (padded batch)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        pixel_values = pixel_values.to(self.vae.device, dtype=self.vae.dtype)
        with torch.no_grad():
            model_input = self.vae.encode(pixel_values)  # [B, 128, H/16, W/16]
        bs, c, h, w = model_input.shape
        img_shapes = [[(1, h, w)]] * bs
        model_input = rearrange(model_input, "b c h w -> b (h w) c")
        if with_ids:
            img_ids = torch.zeros(h, w, 3, device=model_input.device)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(h, device=model_input.device)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(w, device=model_input.device)[None, :]
            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
            return model_input, img_shapes, img_ids
        return model_input, img_shapes
