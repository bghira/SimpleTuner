# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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

import torch
from diffusers.modular_pipelines.modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from diffusers.utils import logging

from .modular_pipeline import MiniMaxH3ModularPipeline, MiniMaxH3Ref2VAModularPipeline
from .scheduler import MiniMaxH3Scheduler
from .transformer import MiniMaxH3Transformer3DModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def optimized_scale(positive_flat: torch.Tensor, negative_flat: torch.Tensor) -> torch.Tensor:
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    return dot_product / squared_norm


def _denoiser_inputs() -> list[InputParam]:
    r"""Everything one MiniMax-H3 forward reads, beyond the transformer itself."""
    return [
        InputParam(
            name="latents",
            type_hint=torch.Tensor,
            required=True,
            description="The video rows of the packed sequence, conditioning rows first.",
        ),
        InputParam(
            name="audio_latents",
            type_hint=torch.Tensor,
            required=True,
            description="The channel-major audio rows of the packed sequence, reference rows first.",
        ),
        InputParam.template("prompt_embeds"),
        InputParam(
            name="negative_prompt_embeds",
            type_hint=torch.Tensor,
            description="Optional negative prompt embeddings for real CFG.",
        ),
        InputParam(
            name="row_timestep_plan",
            type_hint=list,
            required=True,
            description="One `(timestep, timestep_indices)` pair per step.",
        ),
        InputParam(name="token_tags", type_hint=torch.Tensor, required=True, description="The modality tag of every row."),
        InputParam(
            name="position_ids",
            type_hint=torch.Tensor,
            required=True,
            description="The `(t, h, w)` rotary coordinate of every row.",
        ),
        InputParam(
            name="video_indices",
            type_hint=torch.Tensor,
            required=True,
            description="Sequence positions of the video rows.",
        ),
        InputParam(
            name="audio_indices",
            type_hint=torch.Tensor,
            required=True,
            description="Sequence positions of the audio rows.",
        ),
        InputParam(
            name="text_indices",
            type_hint=torch.Tensor,
            required=True,
            description="Sequence positions of the text rows.",
        ),
        InputParam(
            name="negative_row_timestep_plan",
            type_hint=list,
            description="Optional negative-branch row-to-timestep plan for real CFG.",
        ),
        InputParam(
            name="negative_token_tags",
            type_hint=torch.Tensor,
            description="Optional negative-branch modality tag of every row.",
        ),
        InputParam(
            name="negative_position_ids",
            type_hint=torch.Tensor,
            description="Optional negative-branch `(t, h, w)` rotary coordinate of every row.",
        ),
        InputParam(
            name="negative_video_indices",
            type_hint=torch.Tensor,
            description="Optional negative-branch sequence positions of the video rows.",
        ),
        InputParam(
            name="negative_audio_indices",
            type_hint=torch.Tensor,
            description="Optional negative-branch sequence positions of the audio rows.",
        ),
        InputParam(
            name="negative_text_indices",
            type_hint=torch.Tensor,
            description="Optional negative-branch sequence positions of the text rows.",
        ),
        InputParam(name="guidance_scale", type_hint=float, default=1.0, description="Real CFG scale."),
        InputParam(
            name="guidance_scale_real",
            type_hint=float,
            description="Alias for `guidance_scale`, matching SimpleTuner validation configuration.",
        ),
        InputParam(
            name="guidance_rescale",
            type_hint=float,
            description="Optional standard-deviation guidance rescale factor.",
        ),
        InputParam(
            name="skip_guidance_layers",
            type_hint=list,
            description="Layer indices to skip when computing skipped-layer guidance.",
        ),
        InputParam(
            name="skip_layer_guidance_scale",
            type_hint=float,
            default=2.8,
            description="Scale for skipped-layer guidance.",
        ),
        InputParam(
            name="skip_layer_guidance_start",
            type_hint=float,
            default=0.01,
            description="Fraction of denoising steps after which skipped-layer guidance starts.",
        ),
        InputParam(
            name="skip_layer_guidance_stop",
            type_hint=float,
            default=0.2,
            description="Fraction of denoising steps before which skipped-layer guidance stops.",
        ),
        InputParam(
            name="use_cfg_zero_star",
            type_hint=bool,
            default=True,
            description="Whether to apply CFG Zero* when real CFG is active.",
        ),
        InputParam(
            name="use_zero_init",
            type_hint=bool,
            default=True,
            description="Whether CFG Zero* should zero the first guided steps.",
        ),
        InputParam(
            name="zero_steps",
            type_hint=int,
            default=0,
            description="Number of initial guided steps to zero when `use_zero_init` is enabled.",
        ),
        InputParam(
            name="no_cfg_until_timestep",
            type_hint=int,
            default=0,
            description="Step index before which real CFG is disabled.",
        ),
        InputParam(
            name="cfg_end_timestep",
            type_hint=int,
            description="Last step index that may use real CFG.",
        ),
        InputParam(
            name="minimax_h3_reference_mode",
            type_hint=str,
            default="vanilla",
            description="MiniMax-H3 static reference handling mode: vanilla or cached_kv.",
        ),
        InputParam.template("attention_kwargs"),
    ]


def _denoiser_outputs() -> list[OutputParam]:
    return [
        OutputParam(
            "noise_pred", type_hint=torch.Tensor, description="Predicted velocity of the video rows of the sequence."
        ),
        OutputParam(
            "audio_noise_pred",
            type_hint=torch.Tensor,
            description="Predicted velocity of the audio rows of the sequence.",
        ),
    ]


def _state_attr(block_state: BlockState, name: str, prefix: str = ""):
    value = getattr(block_state, f"{prefix}{name}", None) if prefix else getattr(block_state, name)
    if value is None and prefix:
        return getattr(block_state, name)
    return value


def _state_attr_or(block_state: BlockState, name: str, default, prefix: str = ""):
    value = getattr(block_state, f"{prefix}{name}", None) if prefix else getattr(block_state, name, None)
    if value is None and prefix:
        value = getattr(block_state, name, None)
    return default if value is None else value


def _predict_velocity(
    transformer: MiniMaxH3Transformer3DModel,
    block_state: BlockState,
    i: int,
    *,
    prefix: str = "",
    prompt_embeds: torch.Tensor | None = None,
    skip_layers: list[int] | None = None,
):
    r"""One MiniMax-H3 forward pass: every row of the packed sequence, at its own noise level, at once."""
    row_timestep_plan = _state_attr(block_state, "row_timestep_plan", prefix)
    unique_timesteps, timestep_indices = row_timestep_plan[i]
    prompt_embeds = block_state.prompt_embeds if prompt_embeds is None else prompt_embeds
    return transformer(
        hidden_states=block_state.latents[None],
        audio_hidden_states=block_state.audio_latents[None],
        encoder_hidden_states=prompt_embeds,
        timestep=unique_timesteps,
        timestep_indices=timestep_indices,
        token_tags=_state_attr(block_state, "token_tags", prefix),
        position_ids=_state_attr(block_state, "position_ids", prefix),
        video_indices=_state_attr(block_state, "video_indices", prefix),
        audio_indices=_state_attr(block_state, "audio_indices", prefix),
        text_indices=_state_attr(block_state, "text_indices", prefix),
        attention_kwargs=getattr(block_state, "attention_kwargs", None),
        skip_layers=skip_layers,
        num_condition_video_rows=_state_attr_or(block_state, "num_condition_video_rows", 0, prefix),
        num_condition_audio_rows=_state_attr_or(block_state, "num_condition_audio_rows", 0, prefix),
        minimax_h3_reference_mode=getattr(block_state, "minimax_h3_reference_mode", "vanilla") or "vanilla",
        return_dict=False,
    )


def _resolve_guidance_scale(block_state: BlockState) -> float:
    guidance_scale_real = getattr(block_state, "guidance_scale_real", None)
    if guidance_scale_real is not None:
        return float(guidance_scale_real)
    return float(getattr(block_state, "guidance_scale", 1.0) or 1.0)


def _within_cfg_window(block_state: BlockState, i: int) -> bool:
    start = int(getattr(block_state, "no_cfg_until_timestep", 0) or 0)
    stop = getattr(block_state, "cfg_end_timestep", None)
    return i >= start and (stop is None or i <= int(stop))


def _within_skip_window(block_state: BlockState, i: int, num_steps: int) -> bool:
    start = float(getattr(block_state, "skip_layer_guidance_start", 0.01) or 0.0)
    stop = float(getattr(block_state, "skip_layer_guidance_stop", 0.2) or 0.0)
    return i > num_steps * start and i < num_steps * stop


def _apply_cfg(
    positive: torch.Tensor,
    negative: torch.Tensor,
    scale: float,
    block_state: BlockState,
    i: int,
) -> torch.Tensor:
    if bool(getattr(block_state, "use_cfg_zero_star", True)):
        positive_flat = positive.reshape(positive.shape[0], -1).float()
        negative_flat = negative.reshape(negative.shape[0], -1).float()
        alpha = optimized_scale(positive_flat, negative_flat).view(positive.shape[0], *([1] * (positive.ndim - 1)))
        alpha = alpha.to(device=positive.device, dtype=positive.dtype)
        if i <= int(getattr(block_state, "zero_steps", 0) or 0) and bool(getattr(block_state, "use_zero_init", True)):
            return positive * 0.0
        return negative * alpha + scale * (positive - negative * alpha)
    return positive + (scale - 1.0) * (positive - negative)


def _rescale_guidance(guided: torch.Tensor, positive: torch.Tensor, guidance_rescale: float | None) -> torch.Tensor:
    if guidance_rescale is None or float(guidance_rescale) <= 0.0:
        return guided
    axes = tuple(range(1, guided.ndim))
    std_positive = torch.std(positive.float(), dim=axes, keepdim=True)
    std_guided = torch.std(guided.float(), dim=axes, keepdim=True)
    factor = (std_positive / (std_guided + 1e-8)).to(device=guided.device, dtype=guided.dtype)
    rescale = float(guidance_rescale)
    return guided * (1.0 - rescale + rescale * factor)


def _validate_negative_branch(block_state: BlockState) -> torch.Tensor:
    negative_prompt_embeds = getattr(block_state, "negative_prompt_embeds", None)
    if negative_prompt_embeds is None:
        raise ValueError("MiniMax-H3 real CFG requires `negative_prompt` or `negative_prompt_embeds`.")
    negative_text_indices = _state_attr(block_state, "text_indices", "negative_")
    expected_text_rows = int(negative_text_indices.shape[0])
    if negative_prompt_embeds.shape[1] != expected_text_rows:
        raise ValueError(
            "MiniMax-H3 negative prompt embeds must match their packed text layout: "
            f"got {negative_prompt_embeds.shape[1]} embeds for {expected_text_rows} text rows."
        )
    return negative_prompt_embeds


def _predict_guided_velocity(
    transformer: MiniMaxH3Transformer3DModel,
    block_state: BlockState,
    i: int,
    num_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive_video, positive_audio = _predict_velocity(transformer, block_state, i)
    video_velocity, audio_velocity = positive_video, positive_audio

    guidance_scale = _resolve_guidance_scale(block_state)
    apply_cfg = guidance_scale > 1.0 and _within_cfg_window(block_state, i)
    if apply_cfg:
        negative_prompt_embeds = _validate_negative_branch(block_state)
        negative_video, negative_audio = _predict_velocity(
            transformer,
            block_state,
            i,
            prefix="negative_",
            prompt_embeds=negative_prompt_embeds,
        )
        video_velocity = _apply_cfg(positive_video, negative_video, guidance_scale, block_state, i)
        audio_velocity = _apply_cfg(positive_audio, negative_audio, guidance_scale, block_state, i)
        guidance_rescale = getattr(block_state, "guidance_rescale", None)
        video_velocity = _rescale_guidance(video_velocity, positive_video, guidance_rescale)
        audio_velocity = _rescale_guidance(audio_velocity, positive_audio, guidance_rescale)

    skip_guidance_layers = getattr(block_state, "skip_guidance_layers", None)
    if skip_guidance_layers is not None:
        skip_guidance_layers = list(skip_guidance_layers)
    if skip_guidance_layers and _within_skip_window(block_state, i, num_steps):
        skip_video, skip_audio = _predict_velocity(
            transformer,
            block_state,
            i,
            skip_layers=skip_guidance_layers,
        )
        scale = float(getattr(block_state, "skip_layer_guidance_scale", 2.8) or 0.0)
        video_velocity = video_velocity + (positive_video - skip_video) * scale
        audio_velocity = audio_velocity + (positive_audio - skip_audio) * scale

    return video_velocity, audio_velocity


class MiniMaxH3LoopDenoiser(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Runs the one MiniMax-H3 forward pass of a denoising iteration, which predicts the velocity of every row "
            "of the packed sequence at once. The checkpoint is guidance-distilled, so there is no unconditional pass "
            "and no guider."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", MiniMaxH3Transformer3DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return _denoiser_inputs()

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return _denoiser_outputs()

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.noise_pred, block_state.audio_noise_pred = _predict_guided_velocity(
            components.transformer, block_state, i, len(block_state.timesteps)
        )
        return components, block_state


class MiniMaxH3Ref2VALoopDenoiser(ModularPipelineBlocks):
    model_name = "minimax-h3-ref2va"

    @property
    def description(self) -> str:
        return (
            "Runs the one MiniMax-H3 forward pass of a `ref2va` denoising iteration, against the `transformer_ref` "
            "partition of the checkpoint."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer_ref", MiniMaxH3Transformer3DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return _denoiser_inputs()

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return _denoiser_outputs()

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3Ref2VAModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.noise_pred, block_state.audio_noise_pred = _predict_guided_velocity(
            components.transformer_ref, block_state, i, len(block_state.timesteps)
        )
        return components, block_state


class MiniMaxH3LoopSchedulerStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Steps the generated video and audio rows down their own schedule. The conditioning rows are re-imposed "
            "by construction: only the generated rows are ever written, so the anchors survive the whole loop."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", MiniMaxH3Scheduler),
            ComponentSpec("audio_scheduler", MiniMaxH3Scheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latents",
                type_hint=torch.Tensor,
                required=True,
                description="The video rows of the packed sequence, conditioning rows first.",
            ),
            InputParam(
                name="audio_latents",
                type_hint=torch.Tensor,
                required=True,
                description="The channel-major audio rows of the packed sequence, reference rows first.",
            ),
            InputParam(
                name="noise_pred",
                type_hint=torch.Tensor,
                required=True,
                description="Predicted velocity of the video rows.",
            ),
            InputParam(
                name="audio_noise_pred",
                type_hint=torch.Tensor,
                required=True,
                description="Predicted velocity of the audio rows.",
            ),
            InputParam(
                name="audio_timesteps",
                type_hint=torch.Tensor,
                required=True,
                description="Timesteps of the audio schedule.",
            ),
            InputParam(
                name="num_condition_video_rows",
                type_hint=int,
                default=0,
                description="How many leading video rows are conditioning rows.",
            ),
            InputParam(
                name="num_condition_audio_rows",
                type_hint=int,
                default=0,
                description="How many leading audio rows are reference rows.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The video rows of the packed sequence after one step.",
            ),
            OutputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                description="The audio rows of the packed sequence after one step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        num_condition_video_rows = block_state.num_condition_video_rows
        num_condition_audio_rows = block_state.num_condition_audio_rows

        block_state.latents[num_condition_video_rows:] = components.scheduler.step(
            block_state.noise_pred[0, num_condition_video_rows:].float(),
            t,
            block_state.latents[num_condition_video_rows:],
            return_dict=False,
        )[0]
        block_state.audio_latents[num_condition_audio_rows:] = components.audio_scheduler.step(
            block_state.audio_noise_pred[0, num_condition_audio_rows:].float(),
            block_state.audio_timesteps[i],
            block_state.audio_latents[num_condition_audio_rows:],
            return_dict=False,
        )[0]
        return components, block_state


class MiniMaxH3DenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return "Iteratively denoises the packed MiniMax-H3 sequence over the two schedules."

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", MiniMaxH3Scheduler),
            ComponentSpec("audio_scheduler", MiniMaxH3Scheduler),
        ]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam.template("timesteps", required=True, description="Timesteps of the video schedule."),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        with self.progress_bar(total=len(block_state.timesteps)) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                progress_bar.update()
        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3DenoiseStep(MiniMaxH3DenoiseLoopWrapper):
    block_classes = [MiniMaxH3LoopDenoiser, MiniMaxH3LoopSchedulerStep]
    block_names = ["denoiser", "update"]

    @property
    def description(self) -> str:
        return "Runs the `t2va` / `fl2va` MiniMax-H3 denoising loop, one forward pass per step."


class MiniMaxH3Ref2VADenoiseStep(MiniMaxH3DenoiseLoopWrapper):
    model_name = "minimax-h3-ref2va"
    block_classes = [MiniMaxH3Ref2VALoopDenoiser, MiniMaxH3LoopSchedulerStep]
    block_names = ["denoiser", "update"]

    @property
    def description(self) -> str:
        return "Runs the `ref2va` MiniMax-H3 denoising loop, one forward pass per step."
