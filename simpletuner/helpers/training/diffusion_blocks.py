from __future__ import annotations

import builtins
import copy
import inspect
import math
import random
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

_PREFERRED_BLOCK_LIST_NAMES = (
    "transformer_blocks",
    "single_transformer_blocks",
    "joint_transformer_blocks",
    "double_stream_blocks",
    "single_stream_blocks",
    "double_stream_layers",
    "single_stream_layers",
    "text_transformer_blocks",
    "visual_transformer_blocks",
    "blocks",
    "layers",
)


class _SelectiveModuleListMixin:
    """Filter iteration while preserving physical indices and state-dict layout."""

    def __iter__(self):
        active_slice = getattr(self, "_diffusion_blocks_active_slice", None)
        if active_slice is None:
            return super().__iter__()
        start, end = active_slice
        base_getitem = super().__getitem__
        return iter([base_getitem(index) for index in range(start, end)])


_SELECTIVE_MODULE_LIST_TYPES: dict[type[nn.ModuleList], type[nn.ModuleList]] = {}


def _enumerate_diffusion_blocks(iterable, start=0):
    active_slice = getattr(iterable, "_diffusion_blocks_active_slice", None)
    if active_slice is not None:
        start += active_slice[0]
    return builtins.enumerate(iterable, start)


def _select_diffusion_block_from_timestep(module: nn.Module, args: tuple, kwargs: dict) -> None:
    controller = module._diffusion_blocks_controller
    if module.training and controller.training_block is not None:
        controller.activate(controller.training_block)
        return
    timesteps = kwargs.get(controller.timestep_name)
    if timesteps is None and controller.timestep_position < len(args):
        timesteps = args[controller.timestep_position]
    if not torch.is_tensor(timesteps):
        raise ValueError("DiffusionBlocks could not read tensor timesteps from the transformer forward call.")
    controller.activate_for_sigmas(normalize_model_timesteps(timesteps))


def _resolve_module(root: nn.Module, path: str) -> nn.Module:
    current = root
    for part in path.split("."):
        if not hasattr(current, part):
            raise ValueError(f"DiffusionBlocks block path {path!r} does not exist (missing {part!r}).")
        current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise ValueError(f"DiffusionBlocks block path {path!r} does not resolve to a torch module.")
    return current


def discover_block_paths(model: nn.Module) -> list[str]:
    candidates: dict[str, list[str]] = {name: [] for name in _PREFERRED_BLOCK_LIST_NAMES}
    for path, module in model.named_modules():
        if not path or not isinstance(module, nn.ModuleList) or len(module) < 2:
            continue
        leaf = path.rsplit(".", 1)[-1]
        if leaf in candidates:
            candidates[leaf].append(path)

    def shallowest(name: str) -> list[str]:
        paths = candidates[name]
        if not paths:
            return []
        minimum_depth = min(path.count(".") for path in paths)
        return [path for path in paths if path.count(".") == minimum_depth]

    def unique_shallowest(name: str) -> list[str]:
        paths = shallowest(name)
        return paths if len(paths) <= 1 else []

    for name in _PREFERRED_BLOCK_LIST_NAMES:
        primary = unique_shallowest(name)
        if primary:
            if name == "transformer_blocks":
                return primary + unique_shallowest("single_transformer_blocks")
            if name == "joint_transformer_blocks":
                return primary + unique_shallowest("single_transformer_blocks")
            if name == "double_stream_blocks":
                return primary + unique_shallowest("single_stream_blocks")
            if name == "double_stream_layers":
                return primary + unique_shallowest("single_stream_layers")
            if name == "text_transformer_blocks":
                return primary + unique_shallowest("visual_transformer_blocks")
            return primary
    return []


@dataclass(frozen=True)
class DiffusionBlocksConfig:
    layers_per_block: int
    blocks_to_train: tuple[int, ...] | None = None
    overlap: float = 0.05
    block_paths: tuple[str, ...] | None = None
    timestep_boundaries: tuple[float, ...] | None = None

    @classmethod
    def from_dict(cls, value: dict) -> "DiffusionBlocksConfig":
        if not isinstance(value, dict):
            raise ValueError("diffusion_blocks_config must be a JSON object.")
        layers_per_block = int(value.get("layers_per_block", 0))
        if layers_per_block < 1:
            raise ValueError("diffusion_blocks_config.layers_per_block must be at least 1.")

        raw_blocks = value.get("blocks_to_train")
        blocks_to_train = None
        if raw_blocks not in (None, "all"):
            if not isinstance(raw_blocks, (list, tuple)) or not raw_blocks:
                raise ValueError("diffusion_blocks_config.blocks_to_train must be 'all' or a non-empty list.")
            blocks_to_train = tuple(int(index) for index in raw_blocks)

        overlap = float(value.get("overlap", 0.05))
        if not 0.0 <= overlap <= 0.5:
            raise ValueError("diffusion_blocks_config.overlap must be between 0.0 and 0.5.")

        raw_paths = value.get("block_paths")
        block_paths = None
        if raw_paths is not None:
            if not isinstance(raw_paths, (list, tuple)) or not raw_paths:
                raise ValueError("diffusion_blocks_config.block_paths must be a non-empty list.")
            block_paths = tuple(str(path) for path in raw_paths)

        raw_boundaries = value.get("timestep_boundaries")
        timestep_boundaries = None
        if raw_boundaries is not None:
            if not isinstance(raw_boundaries, (list, tuple)):
                raise ValueError("diffusion_blocks_config.timestep_boundaries must be a list.")
            timestep_boundaries = tuple(float(boundary) for boundary in raw_boundaries)

        return cls(
            layers_per_block=layers_per_block,
            blocks_to_train=blocks_to_train,
            overlap=overlap,
            block_paths=block_paths,
            timestep_boundaries=timestep_boundaries,
        )


class DiffusionBlocksController:
    def __init__(self, model: nn.Module, config: DiffusionBlocksConfig):
        self.model = model
        self.config = config
        self.block_paths = list(config.block_paths or discover_block_paths(model))
        if not self.block_paths:
            raise ValueError(
                "DiffusionBlocks could not find a homogeneous transformer block list. "
                "Set diffusion_blocks_config.block_paths explicitly for a compatible transformer."
            )

        self.block_lists: list[nn.ModuleList] = []
        for path in self.block_paths:
            module = _resolve_module(model, path)
            if not isinstance(module, nn.ModuleList):
                raise ValueError(f"DiffusionBlocks block path {path!r} must resolve to torch.nn.ModuleList.")
            self.block_lists.append(module)

        self.stage_offsets = []
        total_depth = 0
        for blocks in self.block_lists:
            self.stage_offsets.append(total_depth)
            total_depth += len(blocks)
        self.total_depth = total_depth
        self.num_blocks = math.ceil(total_depth / config.layers_per_block)
        if self.num_blocks < 2:
            raise ValueError(
                "DiffusionBlocks requires at least two layer groups; reduce layers_per_block below the model depth."
            )
        self.boundaries = self._validate_boundaries(config.timestep_boundaries)
        self.boundaries_are_explicit = config.timestep_boundaries is not None
        self.trainable_blocks = self._validate_trainable_blocks(config.blocks_to_train)
        self.training_block: int | None = None
        self.block_owners = [
            _resolve_module(model, path.rsplit(".", 1)[0]) if "." in path else model for path in self.block_paths
        ]
        self._tread_original_routes = copy.deepcopy(getattr(model, "_tread_routes", None))
        self._install_selective_iteration()
        self.activate(None)
        self._install_timestep_hook()

    def _validate_boundaries(self, boundaries: tuple[float, ...] | None) -> tuple[float, ...]:
        if boundaries is None:
            return tuple(index / self.num_blocks for index in range(self.num_blocks + 1))
        if len(boundaries) != self.num_blocks + 1:
            raise ValueError(
                "diffusion_blocks_config.timestep_boundaries must contain num_blocks + 1 values "
                f"({self.num_blocks + 1} for this model)."
            )
        if boundaries[0] != 0.0 or boundaries[-1] != 1.0:
            raise ValueError("DiffusionBlocks timestep boundaries must start at 0.0 and end at 1.0.")
        if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError("DiffusionBlocks timestep boundaries must be strictly increasing.")
        return boundaries

    def _validate_trainable_blocks(self, blocks: tuple[int, ...] | None) -> tuple[int, ...]:
        if blocks is None:
            return tuple(range(self.num_blocks))
        if len(set(blocks)) != len(blocks):
            raise ValueError("diffusion_blocks_config.blocks_to_train contains duplicate block indices.")
        if any(index < 0 or index >= self.num_blocks for index in blocks):
            raise ValueError(f"DiffusionBlocks block indices must be between 0 and {self.num_blocks - 1}.")
        return blocks

    def set_timestep_boundaries(self, boundaries: Iterable[float]) -> None:
        self.boundaries = self._validate_boundaries(tuple(float(value) for value in boundaries))

    def _group_slice(self, stage_index: int, block_index: int) -> tuple[int, int]:
        depth = len(self.block_lists[stage_index])
        stage_start = self.stage_offsets[stage_index]
        stage_end = stage_start + depth
        group_start = block_index * self.config.layers_per_block
        group_end = min(group_start + self.config.layers_per_block, self.total_depth)
        intersection_start = max(group_start, stage_start)
        intersection_end = min(group_end, stage_end)
        if intersection_start >= intersection_end:
            return 0, 0
        return intersection_start - stage_start, intersection_end - stage_start

    def layer_slices(self, block_index: int) -> dict[str, tuple[int, int]]:
        self._check_block_index(block_index)
        return {path: self._group_slice(stage_index, block_index) for stage_index, path in enumerate(self.block_paths)}

    def _install_selective_iteration(self) -> None:
        for owner, blocks in zip(self.block_owners, self.block_lists):
            if hasattr(blocks, "_diffusion_blocks_active_slice"):
                raise ValueError("A transformer block list is already managed by DiffusionBlocks.")
            original_type = type(blocks)
            selective_type = _SELECTIVE_MODULE_LIST_TYPES.get(original_type)
            if selective_type is None:
                selective_type = type(
                    f"DiffusionBlocks{original_type.__name__}",
                    (_SelectiveModuleListMixin, original_type),
                    {},
                )
                _SELECTIVE_MODULE_LIST_TYPES[original_type] = selective_type
            blocks.__class__ = selective_type
            blocks._diffusion_blocks_active_slice = None
            forward = getattr(owner, "forward", None)
            forward_globals = getattr(getattr(forward, "__func__", forward), "__globals__", None)
            if forward_globals is not None:
                # Several transformers use enumerate indices for routing tables and per-layer state.
                forward_globals["enumerate"] = _enumerate_diffusion_blocks

    def activate(self, block_index: int | None) -> None:
        if block_index is not None:
            self._check_block_index(block_index)
        for stage_index, blocks in enumerate(self.block_lists):
            blocks._diffusion_blocks_active_slice = (
                None if block_index is None else self._group_slice(stage_index, block_index)
            )
        global_start = 0 if block_index is None else block_index * self.config.layers_per_block
        for owner in self.block_owners:
            owner._diffusion_blocks_active_global_start = global_start
        self._activate_tread_routes(block_index)
        self.active_block = block_index

    def _activate_tread_routes(self, block_index: int | None) -> None:
        if self._tread_original_routes is None:
            return
        if block_index is None:
            self.model._tread_routes = copy.deepcopy(self._tread_original_routes)
            return

        group_start = block_index * self.config.layers_per_block
        group_end = min(group_start + self.config.layers_per_block, self.total_depth) - 1
        active_routes = []
        for route in self._tread_original_routes:
            start = int(route["start_layer_idx"])
            end = int(route["end_layer_idx"])
            start = start if start >= 0 else self.total_depth + start
            end = end if end >= 0 else self.total_depth + end
            clipped_start = max(start, group_start)
            clipped_end = min(end, group_end)
            if clipped_start <= clipped_end:
                active_routes.append(
                    {
                        **route,
                        "start_layer_idx": clipped_start,
                        "end_layer_idx": clipped_end,
                    }
                )
        self.model._tread_routes = active_routes

    def set_training_block(self, block_index: int) -> None:
        self._check_block_index(block_index)
        self.training_block = block_index
        self.activate(block_index)

    def _install_timestep_hook(self) -> None:
        signature = inspect.signature(self.model.forward)
        parameter_names = [name for name in signature.parameters if name != "self"]
        timestep_name = next((name for name in ("timestep", "timesteps") if name in parameter_names), None)
        if timestep_name is None:
            raise ValueError("DiffusionBlocks requires the transformer forward method to accept timestep or timesteps.")
        self.timestep_name = timestep_name
        self.timestep_position = parameter_names.index(timestep_name)
        self.model._diffusion_blocks_controller = self
        self.model.register_forward_pre_hook(_select_diffusion_block_from_timestep, with_kwargs=True)

    def _check_block_index(self, block_index: int) -> None:
        if block_index < 0 or block_index >= self.num_blocks:
            raise ValueError(f"DiffusionBlocks block index must be between 0 and {self.num_blocks - 1}.")

    def choose_training_block(self) -> int:
        return random.choice(self.trainable_blocks)

    def freeze_unselected_blocks(self) -> int:
        if len(self.trainable_blocks) == self.num_blocks:
            return 0
        trainable = set(self.trainable_blocks)
        frozen_parameters = 0
        for stage_index, blocks in enumerate(self.block_lists):
            stage_offset = self.stage_offsets[stage_index]
            for layer_index, layer in enumerate(blocks):
                block_index = (stage_offset + layer_index) // self.config.layers_per_block
                if block_index in trainable:
                    continue
                for parameter in layer.parameters():
                    if parameter.requires_grad:
                        parameter.requires_grad_(False)
                        frozen_parameters += parameter.numel()
        return frozen_parameters

    def sigma_range(self, block_index: int, *, include_overlap: bool) -> tuple[float, float]:
        self._check_block_index(block_index)
        reversed_index = self.num_blocks - 1 - block_index
        low = self.boundaries[reversed_index]
        high = self.boundaries[reversed_index + 1]
        if include_overlap and self.config.overlap:
            width = high - low
            low = max(0.0, low - width * self.config.overlap)
            high = min(1.0, high + width * self.config.overlap)
        return low, high

    def block_for_sigmas(self, sigmas: torch.Tensor) -> int:
        values = sigmas.detach().float().reshape(-1).clamp(0.0, 1.0)
        ascending = torch.bucketize(values, values.new_tensor(self.boundaries[1:-1]), right=False)
        indices = (self.num_blocks - 1) - ascending
        unique = torch.unique(indices)
        if unique.numel() != 1:
            raise ValueError(
                "DiffusionBlocks requires every sample/token in one model forward to use the same noise-range block."
            )
        return int(unique.item())

    def activate_for_sigmas(self, sigmas: torch.Tensor) -> int:
        block_index = self.block_for_sigmas(sigmas)
        self.activate(block_index)
        return block_index

    def accepts_training_sigmas(self, sigmas: torch.Tensor, block_index: int) -> torch.Tensor:
        low, high = self.sigma_range(block_index, include_overlap=True)
        values = sigmas.detach().float().reshape(sigmas.shape[0], -1)[:, 0]
        if block_index == 0:
            return (values >= low) & (values <= high)
        return (values >= low) & (values < high)


def normalize_model_timesteps(timesteps: torch.Tensor, num_train_timesteps: int = 1000) -> torch.Tensor:
    values = timesteps.detach().float()
    if values.numel() and torch.max(values) > 1.0:
        values = values / float(num_train_timesteps)
    return values.clamp(0.0, 1.0)
