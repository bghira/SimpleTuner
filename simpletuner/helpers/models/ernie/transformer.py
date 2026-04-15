import logging
from typing import Any, Dict, List, Optional

import torch
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.loaders import peft as diffusers_peft

from simpletuner.helpers.models.ernie.transformer_diffusers import (
    ErnieImageTransformer2DModel as DiffusersErnieImageTransformer2DModel,
)
from simpletuner.helpers.models.ernie.transformer_diffusers import ErnieImageTransformer2DModelOutput
from simpletuner.helpers.training.tread import TREADRouter

logger = logging.getLogger(__name__)


def _store_hidden_state(buffer, key: str, hidden_states: torch.Tensor):
    if buffer is None:
        return
    buffer[key] = hidden_states


def _start_route_rotary(rotary_emb: torch.Tensor, mask_info) -> torch.Tensor:
    ids = mask_info.ids_shuffle.view(mask_info.ids_shuffle.shape[0], mask_info.ids_shuffle.shape[1], 1, 1)
    routed = torch.take_along_dim(rotary_emb, ids.expand_as(rotary_emb), dim=1)
    return routed[:, : mask_info.ids_keep.size(1), :, :]


def _start_route_sequence_first(sequence_tensor: torch.Tensor, mask_info) -> torch.Tensor:
    batch_first = sequence_tensor.permute(1, 0, 2).contiguous()
    routed = torch.take_along_dim(
        batch_first,
        mask_info.ids_shuffle.unsqueeze(-1).expand_as(batch_first),
        dim=1,
    )
    return routed[:, : mask_info.ids_keep.size(1), :].permute(1, 0, 2).contiguous()


def _start_route_temb(temb: List[torch.Tensor], mask_info) -> List[torch.Tensor]:
    return [_start_route_sequence_first(value, mask_info) for value in temb]


if "ErnieImageTransformer2DModel" not in diffusers_peft._SET_ADAPTER_SCALE_FN_MAPPING:
    diffusers_peft._SET_ADAPTER_SCALE_FN_MAPPING["ErnieImageTransformer2DModel"] = lambda model_cls, weights: weights


class ErnieImageTransformer2DModel(
    DiffusersErnieImageTransformer2DModel,
    PeftAdapterMixin,
    FromOriginalModelMixin,
):
    _tread_router: Optional[TREADRouter] = None
    _tread_routes: Optional[List[Dict[str, Any]]] = None

    def set_router(self, router: TREADRouter, routes: Optional[List[Dict[str, Any]]] = None):
        self._tread_router = router
        self._tread_routes = routes

    def set_gradient_checkpointing_interval(self, interval: int):
        self.gradient_checkpointing_interval = interval

    def set_gradient_checkpointing_backend(self, backend: str):
        self.gradient_checkpointing_backend = backend

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_bth: torch.Tensor,
        text_lens: torch.Tensor,
        return_dict: bool = True,
        force_keep_mask: Optional[torch.Tensor] = None,
        hidden_states_buffer: Optional[dict] = None,
        timestep_sign: Optional[torch.Tensor] = None,
        skip_layers: Optional[List[int]] = None,
    ):
        device, dtype = hidden_states.device, hidden_states.dtype
        batch_size, channels, height, width = hidden_states.shape
        patch_size = self.patch_size
        height_patches = height // patch_size
        width_patches = width // patch_size
        image_token_count = height_patches * width_patches

        image_hidden_states = self.x_embedder(hidden_states).transpose(0, 1).contiguous()
        if self.text_proj is not None and text_bth.numel() > 0:
            text_bth = self.text_proj(text_bth)
        max_text_tokens = text_bth.shape[1]
        text_hidden_states = text_bth.transpose(0, 1).contiguous()

        hidden_states = torch.cat([image_hidden_states, text_hidden_states], dim=0)
        total_tokens = hidden_states.shape[0]

        text_ids = (
            torch.cat(
                [
                    torch.arange(max_text_tokens, device=device, dtype=torch.float32)
                    .view(1, max_text_tokens, 1)
                    .expand(batch_size, -1, -1),
                    torch.zeros((batch_size, max_text_tokens, 2), device=device),
                ],
                dim=-1,
            )
            if max_text_tokens > 0
            else torch.zeros((batch_size, 0, 3), device=device)
        )
        grid_yx = torch.stack(
            torch.meshgrid(
                torch.arange(height_patches, device=device, dtype=torch.float32),
                torch.arange(width_patches, device=device, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        image_ids = torch.cat(
            [
                text_lens.float().view(batch_size, 1, 1).expand(-1, image_token_count, -1),
                grid_yx.view(1, image_token_count, 2).expand(batch_size, -1, -1),
            ],
            dim=-1,
        )
        rotary_pos_emb = self.pos_embed(torch.cat([image_ids, text_ids], dim=1))

        valid_text = (
            torch.arange(max_text_tokens, device=device).view(1, max_text_tokens) < text_lens.view(batch_size, 1)
            if max_text_tokens > 0
            else torch.zeros((batch_size, 0), device=device, dtype=torch.bool)
        )
        attention_mask = torch.cat(
            [torch.ones((batch_size, image_token_count), device=device, dtype=torch.bool), valid_text],
            dim=1,
        )[:, None, None, :]

        timestep_proj = self.time_proj(timestep).to(dtype=dtype)
        conditioning = self.time_embedding(timestep_proj)
        if timestep_sign is not None:
            if self.time_sign_embed is None:
                raise ValueError(
                    "timestep_sign was provided but the model was loaded without `enable_time_sign_embed=True`. "
                    "Enable `twinflow_enabled` before loading the ERNIE transformer."
                )
            sign_idx = (timestep_sign.view(-1) < 0).long().to(device=device)
            conditioning = conditioning + self.time_sign_embed(sign_idx).to(device=device, dtype=conditioning.dtype)
        temb = [
            value.unsqueeze(0).expand(total_tokens, -1, -1).contiguous()
            for value in self.adaLN_modulation(conditioning).chunk(6, dim=-1)
        ]

        hidden_states = hidden_states.permute(1, 0, 2).contiguous()
        current_rotary = rotary_pos_emb
        current_attention_mask = attention_mask
        current_temb = temb

        routes = self._tread_routes or []
        router = self._tread_router
        use_routing = self.training and len(routes) > 0 and torch.is_grad_enabled()
        if use_routing and router is None:
            raise ValueError("TREAD routing requested but no router has been configured. Call set_router before training.")

        if force_keep_mask is not None:
            if force_keep_mask.shape != (batch_size, image_token_count):
                raise ValueError(
                    f"force_keep_mask must have shape {(batch_size, image_token_count)}, got {tuple(force_keep_mask.shape)}"
                )
            force_keep_mask = force_keep_mask.to(device=device, dtype=torch.bool)

        if routes:
            total_layers = len(self.layers)

            def _to_pos(idx):
                return idx if idx >= 0 else total_layers + idx

            routes = [
                {
                    **route,
                    "start_layer_idx": _to_pos(route["start_layer_idx"]),
                    "end_layer_idx": _to_pos(route["end_layer_idx"]),
                }
                for route in routes
            ]

        route_ptr = 0
        routing_now = False
        tread_mask_info = None
        saved_hidden_states = None
        saved_rotary = None
        saved_attention_mask = None
        saved_temb = None
        capture_idx = 0

        def apply_layer(layer_idx, layer, batch_first_hidden_states, rotary_emb, attn_mask, temb_values):
            sequence_first_hidden_states = batch_first_hidden_states.permute(1, 0, 2).contiguous()
            if (
                torch.is_grad_enabled()
                and self.gradient_checkpointing
                and (self.gradient_checkpointing_interval is None or layer_idx % self.gradient_checkpointing_interval == 0)
            ):
                if self.gradient_checkpointing_backend == "unsloth":
                    from simpletuner.helpers.training.offloaded_gradient_checkpointer import offloaded_checkpoint

                    sequence_first_hidden_states = offloaded_checkpoint(
                        layer,
                        sequence_first_hidden_states,
                        rotary_emb,
                        temb_values,
                        attn_mask,
                        use_reentrant=False,
                    )
                else:
                    sequence_first_hidden_states = self._gradient_checkpointing_func(
                        layer,
                        sequence_first_hidden_states,
                        rotary_emb,
                        temb_values,
                        attn_mask,
                    )
            else:
                sequence_first_hidden_states = layer(sequence_first_hidden_states, rotary_emb, temb_values, attn_mask)
            return sequence_first_hidden_states.permute(1, 0, 2).contiguous()

        skip_set = set(skip_layers) if skip_layers is not None else set()
        combined_blocks = list(self.layers)
        musubi_manager = self._musubi_block_swap
        musubi_offload_active = False
        grad_enabled = torch.is_grad_enabled()
        if musubi_manager is not None:
            musubi_offload_active = musubi_manager.activate(combined_blocks, hidden_states.device, grad_enabled)

        for layer_idx, layer in enumerate(self.layers):
            if musubi_offload_active and musubi_manager.is_managed_block(layer_idx):
                musubi_manager.stream_in(layer, hidden_states.device)
            if use_routing and route_ptr < len(routes) and layer_idx == routes[route_ptr]["start_layer_idx"]:
                keep_mask = torch.zeros(
                    (batch_size, hidden_states.shape[1]),
                    device=hidden_states.device,
                    dtype=torch.bool,
                )
                keep_mask[:, image_token_count:] = True
                if force_keep_mask is not None:
                    keep_mask[:, :image_token_count] |= force_keep_mask

                tread_mask_info = router.get_mask(
                    hidden_states,
                    mask_ratio=routes[route_ptr]["selection_ratio"],
                    force_keep=keep_mask,
                )
                saved_hidden_states = hidden_states.clone()
                saved_rotary = current_rotary.clone()
                saved_attention_mask = current_attention_mask.clone()
                saved_temb = current_temb
                hidden_states = router.start_route(hidden_states, tread_mask_info)
                current_rotary = _start_route_rotary(current_rotary, tread_mask_info)
                current_temb = _start_route_temb(current_temb, tread_mask_info)
                current_attention_mask = torch.ones(
                    (batch_size, 1, 1, hidden_states.shape[1]),
                    device=hidden_states.device,
                    dtype=torch.bool,
                )
                routing_now = True

            if layer_idx not in skip_set:
                hidden_states = apply_layer(
                    layer_idx,
                    layer,
                    hidden_states,
                    current_rotary,
                    current_attention_mask,
                    current_temb,
                )
            capture_hidden_states = hidden_states
            if routing_now:
                capture_hidden_states = router.end_route(
                    hidden_states,
                    tread_mask_info,
                    original_x=saved_hidden_states,
                )
            _store_hidden_state(
                hidden_states_buffer,
                f"layer_{capture_idx}",
                capture_hidden_states[:, :image_token_count, ...],
            )
            capture_idx += 1

            if routing_now and route_ptr < len(routes) and layer_idx == routes[route_ptr]["end_layer_idx"]:
                hidden_states = router.end_route(hidden_states, tread_mask_info, original_x=saved_hidden_states)
                current_rotary = saved_rotary
                current_attention_mask = saved_attention_mask
                current_temb = saved_temb
                routing_now = False
                route_ptr += 1
            if musubi_offload_active and musubi_manager.is_managed_block(layer_idx):
                musubi_manager.stream_out(layer)

        hidden_states = hidden_states.permute(1, 0, 2).contiguous()
        hidden_states = self.final_norm(hidden_states, conditioning).type_as(hidden_states)
        patches = self.final_linear(hidden_states)[:image_token_count].transpose(0, 1).contiguous()
        output = (
            patches.view(batch_size, height_patches, width_patches, patch_size, patch_size, self.out_channels)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .view(batch_size, self.out_channels, height, width)
        )

        if return_dict:
            return ErnieImageTransformer2DModelOutput(sample=output)
        return (output,)
