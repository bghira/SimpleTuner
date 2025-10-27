from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .scheduler import FlowMatchingSchedulerAdapter
from .wrappers import FoundationModelWrapper


def _flatten_video(tensor: torch.Tensor) -> torch.Tensor:
    b, c, t, h, w = tensor.shape
    return tensor.reshape(b * t, c, h, w)


def _unflatten_video(tensor: torch.Tensor, shape: Tuple[int, int, int, int, int]) -> torch.Tensor:
    b, c, t, h, w = shape
    return tensor.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


@dataclass
class SelfForcingTrainingPipeline:
    denoising_step_list: Sequence[int]
    scheduler: FlowMatchingSchedulerAdapter
    generator: FoundationModelWrapper
    num_frame_per_block: int = 3
    independent_first_frame: bool = False
    same_step_across_blocks: bool = False
    last_step_only: bool = False
    num_max_frames: int = 21
    context_noise: int = 0
    device: Optional[torch.device] = None
    config: Optional[object] = None

    def __post_init__(self) -> None:
        if not self.denoising_step_list:
            raise ValueError("`denoising_step_list` must contain at least one timestep.")
        steps = list(int(step) for step in self.denoising_step_list)
        if steps[-1] == 0:
            steps = steps[:-1]
        self.denoising_step_list = steps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sync_indices(self, num_blocks: int, num_denoising_steps: int, device: torch.device) -> List[int]:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                indices = torch.randint(
                    low=0,
                    high=num_denoising_steps,
                    size=(num_blocks,),
                    device=device,
                )
                if self.last_step_only:
                    indices.fill_(num_denoising_steps - 1)
            else:
                indices = torch.empty(num_blocks, dtype=torch.long, device=device)
            dist.broadcast(indices, src=0)
            return indices.tolist()

        indices = torch.randint(
            low=0,
            high=num_denoising_steps,
            size=(num_blocks,),
            device=device,
        )
        if self.last_step_only:
            indices.fill_(num_denoising_steps - 1)
        return indices.tolist()

    def _compute_block_layout(self, total_frames: int, initial_latent: Optional[torch.Tensor]) -> List[int]:
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            if total_frames % self.num_frame_per_block != 0:
                raise ValueError("Noise tensor frames must be divisible by num_frame_per_block.")
            num_blocks = total_frames // self.num_frame_per_block
            return [self.num_frame_per_block] * num_blocks

        if (total_frames - 1) % self.num_frame_per_block != 0:
            raise ValueError("Noise tensor does not align with block size for independent first frame setting.")
        blocks = [1]
        blocks.extend([self.num_frame_per_block] * ((total_frames - 1) // self.num_frame_per_block))
        return blocks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        *,
        conditional_dict: Dict[str, torch.Tensor],
        initial_latent: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]] | Tuple[torch.Tensor, Optional[int], Optional[int], int]:
        """
        Simulates the WAN generator trajectory, returning denoised latents alongside
        bookkeeping about the selected training timestep.
        """
        if noise.ndim != 5:
            raise ValueError(f"Noise tensor must be 5D (B, C, T, H, W); received {tuple(noise.shape)}.")

        device = noise.device
        dtype = noise.dtype
        batch_size, channels, total_frames, height, width = noise.shape
        block_layout = self._compute_block_layout(total_frames, initial_latent)
        num_blocks = len(block_layout)
        num_denoising_steps = len(self.denoising_step_list)
        exit_indices = self._sync_indices(num_blocks, num_denoising_steps, device=device)

        if initial_latent is not None:
            if initial_latent.ndim != 5:
                raise ValueError("Initial latents must be shaped (B, C, T, H, W).")
            if initial_latent.shape[0] != batch_size:
                raise ValueError("Initial latent batch size must match noise batch size.")
            num_input_frames = initial_latent.shape[2]
        else:
            num_input_frames = 0

        total_output_frames = total_frames + num_input_frames
        output = torch.zeros(
            (batch_size, channels, total_output_frames, height, width),
            device=device,
            dtype=dtype,
        )

        current_start = 0
        if initial_latent is not None:
            output[:, :, :num_input_frames] = initial_latent
            current_start = num_input_frames

        denoised_flag = exit_indices[0] if exit_indices else 0
        for block_index, frames_in_block in enumerate(block_layout):
            block_slice = slice(current_start - num_input_frames, current_start - num_input_frames + frames_in_block)
            noisy_block = noise[:, :, block_slice, :, :].contiguous()
            denoised_prediction = noisy_block  # fallback

            for step_idx, timestep_value in enumerate(self.denoising_step_list):
                timestep_tensor = torch.full(
                    (batch_size, frames_in_block),
                    timestep_value,
                    device=device,
                    dtype=torch.long,
                )

                flow_pred, pred_x0 = self.generator.forward(
                    noisy_block,
                    timestep_tensor,
                    conditional_dict,
                )
                denoised_prediction = pred_x0

                selected_index = exit_indices[0] if self.same_step_across_blocks else exit_indices[block_index]
                if step_idx == selected_index:
                    denoised_flag = step_idx
                    break

                if step_idx + 1 >= len(self.denoising_step_list):
                    break

                next_timestep_value = self.denoising_step_list[step_idx + 1]
                next_timestep = torch.full_like(timestep_tensor, next_timestep_value)

                flat_clean = _flatten_video(pred_x0)
                flat_noise = torch.randn_like(flat_clean)
                noisy_block = _unflatten_video(
                    self.scheduler.add_noise(flat_clean, flat_noise, next_timestep.reshape(-1)),
                    (batch_size, channels, frames_in_block, height, width),
                )

            output[:, :, current_start : current_start + frames_in_block] = denoised_prediction
            current_start += frames_in_block

        denoised_from = self.denoising_step_list[min(denoised_flag, len(self.denoising_step_list) - 1)]
        if denoised_flag + 1 < len(self.denoising_step_list):
            denoised_to: Optional[int] = self.denoising_step_list[denoised_flag + 1]
        else:
            denoised_to = 0

        if return_sim_step:
            return output, denoised_from, denoised_to, denoised_flag + 1
        return output, denoised_from, denoised_to
