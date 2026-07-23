import json
import logging
import os
from typing import Any, Optional

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from simpletuner.helpers.models.mageflow.vendor.models.mage_flow import MageFlow, MageFlowParams
from simpletuner.helpers.models.mageflow.vendor.models.modules._attn_backend import set_attn_backend
from simpletuner.helpers.musubi_block_swap import MusubiBlockSwapManager
from simpletuner.helpers.training.gradient_checkpointing_interval import (
    get_checkpoint_backend,
    get_checkpoint_function,
    set_checkpoint_backend,
)

logger = logging.getLogger(__name__)


def _store_hidden_state(buffer, key: str, hidden_states: torch.Tensor):
    if buffer is None:
        return
    buffer[key] = hidden_states


def _resolve_repo_dir(repo_id_or_path: str, *, revision: Optional[str] = None, local_files_only: bool = False) -> str:
    if os.path.isdir(repo_id_or_path):
        return os.path.abspath(repo_id_or_path)
    return snapshot_download(repo_id=repo_id_or_path, revision=revision, local_files_only=local_files_only)


class MageFlowTransformer2DModel(MageFlow, ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["MageFlowTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        vec_in_dim: int = 0,
        context_in_dim: int = 2560,
        hidden_size: int = 3072,
        mlp_ratio: float = 4.0,
        num_heads: int = 24,
        depth: int = 12,
        depth_single_blocks: int = 0,
        axes_dim: Optional[list[int]] = None,
        theta: int = 10000,
        patch_size: int = 1,
        qkv_bias: bool = True,
        guidance_embed: bool = False,
        checkpoint: bool = False,
        rope_type: str = "msrope",
        time_type: str = "qwen_proj",
        double_block_type: str = "double_stream",
        vec_type: Optional[str] = None,
        apply_text_rotary_emb: bool = False,
        txt_max_length: int = 2048,
        max_sequence_length: int = 2048,
        param_dtype: str = "bfloat16",
        packing: bool = True,
        schedule_mode: str = "z-image",
        static_shift: float = 6.0,
        use_time_shift: bool = False,
        attn_type: str = "sdpa",
        musubi_blocks_to_swap: int = 0,
        musubi_block_swap_device: str = "cpu",
        enable_time_sign_embed: bool = False,
    ):
        del (
            vec_in_dim,
            mlp_ratio,
            depth_single_blocks,
            theta,
            qkv_bias,
            guidance_embed,
            rope_type,
            time_type,
            double_block_type,
            vec_type,
            apply_text_rotary_emb,
            txt_max_length,
            max_sequence_length,
            param_dtype,
            packing,
            schedule_mode,
            static_shift,
            use_time_shift,
        )
        set_attn_backend(attn_type)
        params = MageFlowParams(
            in_channels=in_channels,
            out_channels=out_channels,
            context_in_dim=context_in_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            depth=depth,
            axes_dim=axes_dim or [16, 56, 56],
            checkpoint=checkpoint,
            patch_size=patch_size,
        )
        MageFlow.__init__(self, params)
        self.time_sign_embed: torch.nn.Embedding | None = None
        if enable_time_sign_embed:
            self.time_sign_embed = torch.nn.Embedding(2, self.inner_dim)
            torch.nn.init.zeros_(self.time_sign_embed.weight)
        self.gradient_checkpointing_backend = get_checkpoint_backend()
        self._gradient_checkpointing_func = get_checkpoint_function()
        self._musubi_block_swap = MusubiBlockSwapManager.build(
            depth=depth,
            blocks_to_swap=musubi_blocks_to_swap,
            swap_device=musubi_block_swap_device,
            logger=logger,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, subfolder: Optional[str] = None, **kwargs):
        revision = kwargs.pop("revision", None)
        local_files_only = bool(kwargs.pop("local_files_only", False))
        torch_dtype = kwargs.pop("torch_dtype", None)
        attn_type = kwargs.pop("attn_type", None)
        musubi_blocks_to_swap = kwargs.pop("musubi_blocks_to_swap", 0)
        musubi_block_swap_device = kwargs.pop("musubi_block_swap_device", "cpu")
        enable_time_sign_embed = kwargs.pop("enable_time_sign_embed", False)
        use_safetensors = kwargs.pop("use_safetensors", True)
        if not use_safetensors:
            raise ValueError("Mage-Flow transformer checkpoints are expected to be safetensors.")

        repo_dir = _resolve_repo_dir(
            pretrained_model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
        )
        model_dir = os.path.join(repo_dir, subfolder) if subfolder else repo_dir
        with open(os.path.join(model_dir, "config.json"), "r") as handle:
            config = json.load(handle)
        config.pop("_class_name", None)
        if attn_type is not None:
            config["attn_type"] = attn_type
        config["musubi_blocks_to_swap"] = musubi_blocks_to_swap
        config["musubi_block_swap_device"] = musubi_block_swap_device
        config["enable_time_sign_embed"] = enable_time_sign_embed
        model = cls(**config)
        state_dict = load_file(os.path.join(model_dir, "diffusion_pytorch_model.safetensors"), device="cpu")
        model.load_state_dict(state_dict, strict=False, assign=True)
        if torch_dtype is not None:
            model.to(dtype=torch_dtype)
        return model

    def set_gradient_checkpointing_backend(self, backend: str):
        set_checkpoint_backend(backend)
        self.gradient_checkpointing_backend = backend
        self._gradient_checkpointing_func = get_checkpoint_function()

    def enable_gradient_checkpointing(self, gradient_checkpointing_func=None):
        self.checkpoint = True
        self.config.checkpoint = True
        if gradient_checkpointing_func is not None:
            self._gradient_checkpointing_func = gradient_checkpointing_func

    def disable_gradient_checkpointing(self):
        self.checkpoint = False
        self.config.checkpoint = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        timesteps: torch.Tensor,
        img_shapes=None,
        img_cu_seqlens: torch.Tensor | None = None,
        txt_cu_seqlens: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        timestep_sign: torch.Tensor | None = None,
        skip_layers: list[int] | None = None,
        hidden_states_buffer: dict | None = None,
        return_dict: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        if timesteps.ndim != 1:
            raise ValueError("Mage-Flow expects batchwise 1D timesteps; tokenwise timesteps are not supported.")

        ms_pe = self.pos_embed(img_shapes, device=img.device)

        img = self.img_in(img)
        txt = self.txt_norm(txt)

        timesteps = timesteps.to(img.dtype)
        temb = self.time_text_embed(timesteps, img)
        if timestep_sign is not None:
            if self.time_sign_embed is None:
                raise ValueError(
                    "timestep_sign was provided but the model was loaded without `enable_time_sign_embed=True`. "
                    "Enable TwinFlow or load a TwinFlow-compatible checkpoint."
                )
            sign_tensor = timestep_sign.to(device=img.device)
            if sign_tensor.ndim == 0:
                sign_tensor = sign_tensor.expand(temb.shape[0])
            elif sign_tensor.ndim == 1:
                if sign_tensor.shape[0] == 1 and temb.shape[0] != 1:
                    sign_tensor = sign_tensor.expand(temb.shape[0])
                elif sign_tensor.shape[0] != temb.shape[0]:
                    raise ValueError(
                        f"Mage-Flow timestep_sign expected 1 or {temb.shape[0]} batch values, "
                        f"got {sign_tensor.shape[0]}."
                    )
            else:
                raise ValueError(
                    f"Mage-Flow timestep_sign expected scalar or 1D batch tensor, got shape {tuple(sign_tensor.shape)}."
                )
            sign_idx = (sign_tensor < 0).long()
            temb = temb + self.time_sign_embed(sign_idx).to(dtype=temb.dtype, device=temb.device)

        txt = self.txt_in(txt)
        txt_vec = torch.zeros(txt.shape[0], self.inner_dim, dtype=txt.dtype, device=txt.device)
        temb = temb + txt_vec

        attention_kwargs = attention_kwargs or {}

        musubi_manager = self._musubi_block_swap
        musubi_offload_active = False
        if musubi_manager is not None:
            musubi_offload_active = musubi_manager.activate(
                self.transformer_blocks,
                img.device,
                torch.is_grad_enabled(),
            )

        skip_layers_set = set(skip_layers) if skip_layers is not None else set()
        for index_block, block in enumerate(self.transformer_blocks):
            if musubi_offload_active and musubi_manager.is_managed_block(index_block):
                musubi_manager.stream_in(block, img.device)
            if index_block in skip_layers_set:
                pass
            elif self.training and self.checkpoint:
                txt, img = self._gradient_checkpointing_func(
                    block,
                    img,
                    txt,
                    temb,
                    ms_pe,
                    txt_cu_seqlens,
                    img_cu_seqlens,
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
            if musubi_offload_active and musubi_manager.is_managed_block(index_block):
                musubi_manager.stream_out(block)
            _store_hidden_state(hidden_states_buffer, f"layer_{index_block}", img)

        img = self.norm_out(
            img,
            temb,
            cu_seqlens=img_cu_seqlens,
        )
        img = self.proj_out(img)
        if return_dict is False:
            return (img,)
        return img
