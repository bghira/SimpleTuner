import inspect
from typing import Optional, Union

import torch
from diffusers import UNet2DConditionModel

from simpletuner.helpers.models.flowmap import (
    blend_flowmap_embeddings,
    clone_flowmap_embedder,
    prepare_flowmap_delta_timestep,
    register_flowmap_config,
    set_flowmap_gate,
    validate_flowmap_deltatime_type,
)


class FlowMapUNet2DConditionModel(UNet2DConditionModel):
    def __init__(
        self,
        *args,
        gate_value: Optional[float] = None,
        deltatime_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.delta_time_embedding: Optional[torch.nn.Module] = None
        self.flowmap_deltatime_type: Optional[str] = None
        self.register_buffer("flowmap_delta_emb_gate", torch.tensor([0.25], dtype=torch.float32), persistent=False)
        if deltatime_type is not None:
            self.enable_flowmap_time_conditioning(
                gate_value=0.25 if gate_value is None else float(gate_value),
                deltatime_type=deltatime_type,
            )

    def enable_flowmap_time_conditioning(self, gate_value: float = 0.25, deltatime_type: str = "r") -> None:
        self.flowmap_deltatime_type = validate_flowmap_deltatime_type(deltatime_type, model_name="UNet")
        if self.delta_time_embedding is None:
            self.delta_time_embedding = clone_flowmap_embedder(self.time_embedding)
        set_flowmap_gate(self, gate_value)
        register_flowmap_config(self, gate_value, deltatime_type)

    def _normalize_unet_timestep(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
    ) -> torch.Tensor:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        else:
            timesteps = timesteps.to(sample.device)
        return timesteps.expand(sample.shape[0])

    def _flowmap_time_embedding_forward(
        self,
        *,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        r_timestep: torch.Tensor,
        base_forward,
        t_emb: torch.Tensor,
        timestep_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.delta_time_embedding is None or self.flowmap_deltatime_type is None:
            raise ValueError("UNet FlowMap conditioning requires `enable_flowmap_time_conditioning()` before training.")

        base_embedding = base_forward(t_emb, timestep_cond)
        timestep = self._normalize_unet_timestep(sample, timestep)
        delta_timestep = prepare_flowmap_delta_timestep(
            timestep,
            r_timestep,
            self.flowmap_deltatime_type,
            model_name="UNet",
        )
        delta_t_emb = self.time_proj(delta_timestep)
        delta_t_emb = delta_t_emb.to(dtype=sample.dtype)
        delta_embedding = self.delta_time_embedding(delta_t_emb, timestep_cond)
        return blend_flowmap_embeddings(base_embedding, delta_embedding, self.flowmap_delta_emb_gate)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        *args,
        r_timestep: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if r_timestep is None:
            return super().forward(sample, timestep, encoder_hidden_states, *args, **kwargs)

        base_forward = self.time_embedding.forward

        def flowmap_forward(t_emb: torch.Tensor, timestep_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
            return self._flowmap_time_embedding_forward(
                sample=sample,
                timestep=timestep,
                r_timestep=r_timestep,
                base_forward=base_forward,
                t_emb=t_emb,
                timestep_cond=timestep_cond,
            )

        self.time_embedding.forward = flowmap_forward
        try:
            return super().forward(sample, timestep, encoder_hidden_states, *args, **kwargs)
        finally:
            self.time_embedding.forward = base_forward


def _flowmap_unet_init_signature() -> inspect.Signature:
    base_signature = inspect.signature(UNet2DConditionModel.__init__)
    parameters = list(base_signature.parameters.values())
    parameters.extend(
        [
            inspect.Parameter(
                "gate_value",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[float],
            ),
            inspect.Parameter(
                "deltatime_type",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[str],
            ),
        ]
    )
    return base_signature.replace(parameters=parameters)


FlowMapUNet2DConditionModel.__init__.__signature__ = _flowmap_unet_init_signature()
