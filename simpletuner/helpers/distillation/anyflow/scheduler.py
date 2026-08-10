from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, Iterable, Optional, Sequence

import torch

_TIMESTEP_PARAMETER_NAMES = ("timestep", "timesteps", "t")
_FLOWMAP_PARAMETER_NAMES = ("r_timestep", "timestep_r")


class _AnyFlowValidationComponent(torch.nn.Module):
    def __init__(self, component: Any, scheduler: "AnyFlowValidationScheduler", kwarg_name: str):
        super().__init__()
        self._anyflow_component = component
        self._anyflow_scheduler = scheduler
        self._anyflow_kwarg_name = kwarg_name
        self._anyflow_timestep_name, self._anyflow_timestep_index = self._resolve_timestep_parameter(component)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._anyflow_component, name)

    def forward(self, *args, **kwargs):
        if kwargs.get(self._anyflow_kwarg_name) is None:
            timestep = self._extract_timestep(args, kwargs)
            timestep, r_timestep, timestep_indices = self._anyflow_scheduler.component_timesteps(timestep, kwargs)
            if self._anyflow_timestep_name in kwargs:
                kwargs[self._anyflow_timestep_name] = timestep
            elif timestep_indices is not None and self._anyflow_timestep_index is not None:
                positional_args = list(args)
                positional_args[self._anyflow_timestep_index] = timestep
                args = tuple(positional_args)
            if timestep_indices is not None:
                kwargs["timestep_indices"] = timestep_indices
            kwargs[self._anyflow_kwarg_name] = r_timestep
        return self._anyflow_component(*args, **kwargs)

    def _extract_timestep(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if self._anyflow_timestep_name in kwargs:
            return kwargs[self._anyflow_timestep_name]
        if self._anyflow_timestep_index is not None and self._anyflow_timestep_index < len(args):
            return args[self._anyflow_timestep_index]
        raise ValueError("AnyFlow validation could not find the model timestep argument needed to derive `r_timestep`.")

    @staticmethod
    def _resolve_timestep_parameter(component: Any) -> tuple[str, Optional[int]]:
        forward = getattr(component, "forward", component)
        signature = inspect.signature(forward)
        positional_index = 0
        for parameter in signature.parameters.values():
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                if parameter.name in _TIMESTEP_PARAMETER_NAMES:
                    index = positional_index if parameter.kind is not inspect.Parameter.KEYWORD_ONLY else None
                    return parameter.name, index
                if parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
                    positional_index += 1
            elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                positional_index = None
            elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
                return "timestep", None
        raise ValueError(
            "AnyFlow validation requires a FlowMap-capable component with a `timestep`, `timesteps`, or `t` "
            "forward parameter."
        )


class AnyFlowValidationScheduler:
    """Scheduler proxy that injects FlowMap interval endpoints during validation."""

    def __init__(self, scheduler: Any, *, num_train_timesteps: Optional[float] = None):
        self.scheduler = scheduler
        self.num_train_timesteps = num_train_timesteps
        self._timestep_transform_name: Optional[str] = None
        self._audio_scheduler: Optional[AnyFlowValidationScheduler] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.scheduler, name)

    def install_pipeline_hooks(
        self,
        pipeline: Any,
        *,
        component_names: Sequence[str] = ("transformer", "unet"),
    ) -> None:
        audio_scheduler = getattr(pipeline, "audio_scheduler", None)
        if audio_scheduler is not None:
            self._audio_scheduler = AnyFlowValidationScheduler(
                audio_scheduler,
                num_train_timesteps=self.num_train_timesteps,
            )
        for component_name in component_names:
            component = getattr(pipeline, component_name, None)
            if component is None:
                continue
            if getattr(component, "_anyflow_validation_wrapper", False):
                component._anyflow_scheduler = self
                return

            kwarg_name = self._flowmap_kwarg_name(component)
            if kwarg_name is None:
                continue

            wrapped = _AnyFlowValidationComponent(component, self, kwarg_name)
            wrapped._anyflow_validation_wrapper = True
            setattr(pipeline, component_name, wrapped)
            return

        names = ", ".join(component_names)
        raise ValueError(
            "AnyFlow validation could not find a pipeline component accepting `r_timestep` or `timestep_r` "
            f"among: {names}."
        )

    def component_timesteps(
        self,
        timestep: Any,
        component_kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        timestep_tensor = self._to_tensor(timestep)
        timestep_indices = component_kwargs.get("timestep_indices")
        audio_indices = component_kwargs.get("audio_indices")
        if self._audio_scheduler is None or timestep_indices is None or audio_indices is None:
            return timestep_tensor, self.r_timestep_for(timestep_tensor), None

        timestep_indices = timestep_indices.to(device=timestep_tensor.device, dtype=torch.long)
        row_timesteps = timestep_tensor[timestep_indices]
        row_r_timesteps = self.r_timestep_for(row_timesteps)

        audio_indices = audio_indices.to(device=timestep_tensor.device, dtype=torch.long)
        num_condition_audio_rows = int(component_kwargs.get("num_condition_audio_rows", 0) or 0)
        generated_audio_indices = audio_indices[num_condition_audio_rows:]
        if generated_audio_indices.numel() > 0:
            row_r_timesteps[generated_audio_indices] = self._audio_scheduler.r_timestep_for(
                row_timesteps[generated_audio_indices]
            )

        video_indices = component_kwargs.get("video_indices")
        if video_indices is not None:
            video_indices = video_indices.to(device=timestep_tensor.device, dtype=torch.long)
            num_condition_video_rows = int(component_kwargs.get("num_condition_video_rows", 0) or 0)
            condition_video_indices = video_indices[:num_condition_video_rows]
            row_r_timesteps[condition_video_indices] = row_timesteps[condition_video_indices]
        condition_audio_indices = audio_indices[:num_condition_audio_rows]
        row_r_timesteps[condition_audio_indices] = row_timesteps[condition_audio_indices]

        timestep_pairs = torch.stack((row_timesteps.float(), row_r_timesteps.float()), dim=-1)
        unique_pairs, remapped_indices = torch.unique(timestep_pairs, dim=0, sorted=True, return_inverse=True)
        return (
            unique_pairs[:, 0].to(dtype=timestep_tensor.dtype),
            unique_pairs[:, 1].to(dtype=timestep_tensor.dtype),
            remapped_indices.to(device=timestep_indices.device, dtype=timestep_indices.dtype),
        )

    def r_timestep_for(self, timestep: Any) -> torch.Tensor:
        timestep_tensor = self._to_tensor(timestep)
        if timestep_tensor.numel() == 0:
            return timestep_tensor

        schedule = self._scheduler_timesteps(timestep_tensor.device)
        if schedule.numel() == 0:
            return torch.zeros_like(timestep_tensor)

        train_scale = self._train_timestep_scale(schedule)
        transformed_schedule, transform = self._transform_schedule_for_component(schedule, timestep_tensor, train_scale)
        endpoints = self._endpoint_schedule(schedule, train_scale, timestep_tensor.device)
        transformed_endpoints = transform(endpoints)

        flat_timestep = timestep_tensor.detach().to(dtype=torch.float32).reshape(-1)
        flat_result = []
        flat_schedule = transformed_schedule.detach().to(dtype=torch.float32).reshape(-1)
        for value in flat_timestep:
            index = int(torch.argmin(torch.abs(flat_schedule - value)).item())
            flat_result.append(transformed_endpoints[index])
        result = torch.stack(flat_result).reshape(timestep_tensor.shape)
        return result.to(device=timestep_tensor.device, dtype=timestep_tensor.dtype)

    @staticmethod
    def _flowmap_kwarg_name(component: Any) -> Optional[str]:
        forward = getattr(component, "forward", component)
        signature = inspect.signature(forward)
        accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
        parameter_names = set(signature.parameters)
        for kwarg_name in _FLOWMAP_PARAMETER_NAMES:
            if kwarg_name in parameter_names:
                return kwarg_name
        return "r_timestep" if accepts_kwargs else None

    @staticmethod
    def _to_tensor(timestep: Any) -> torch.Tensor:
        if torch.is_tensor(timestep):
            return timestep
        return torch.tensor(timestep, dtype=torch.float32)

    def _scheduler_timesteps(self, device: torch.device) -> torch.Tensor:
        timesteps = getattr(self.scheduler, "timesteps", None)
        if timesteps is None:
            return torch.empty(0, device=device, dtype=torch.float32)
        if torch.is_tensor(timesteps):
            return timesteps.detach().to(device=device, dtype=torch.float32).reshape(-1)
        return torch.tensor(list(timesteps), device=device, dtype=torch.float32).reshape(-1)

    def _train_timestep_scale(self, schedule: torch.Tensor) -> float:
        if self.num_train_timesteps is not None:
            return float(self.num_train_timesteps)
        scheduler_config = getattr(self.scheduler, "config", SimpleNamespace())
        configured = getattr(scheduler_config, "num_train_timesteps", None)
        if configured is None and isinstance(scheduler_config, dict):
            configured = scheduler_config.get("num_train_timesteps")
        if configured is not None:
            return float(configured)
        return float(max(torch.max(torch.abs(schedule)).item(), 1.0))

    def _endpoint_schedule(
        self,
        schedule: torch.Tensor,
        train_scale: float,
        device: torch.device,
    ) -> torch.Tensor:
        schedule_scale = 1.0 if torch.max(torch.abs(schedule)).item() <= 1.5 else train_scale
        final_raw_timestep = 0.0 if schedule[0] >= schedule[-1] else schedule_scale
        final = torch.tensor([final_raw_timestep], device=device, dtype=torch.float32)
        return torch.cat([schedule[1:], final])

    def _transform_schedule_for_component(
        self,
        schedule: torch.Tensor,
        timestep: torch.Tensor,
        train_scale: float,
    ) -> tuple[torch.Tensor, Any]:
        flat_timestep = timestep.detach().to(dtype=torch.float32).reshape(-1)
        schedule_abs_max = torch.max(torch.abs(schedule)).item()
        timestep_abs_max = torch.max(torch.abs(flat_timestep)).item()

        if self._timestep_transform_name is not None:
            transform = self._schedule_transform(self._timestep_transform_name, train_scale)
            return transform(schedule), transform

        current = flat_timestep[0]
        candidates = self._schedule_transform_candidates(schedule, train_scale, schedule_abs_max, timestep_abs_max)
        best_name, best_schedule = min(
            candidates,
            key=lambda item: torch.min(torch.abs(item[1].detach().to(dtype=torch.float32).reshape(-1) - current)).item(),
        )
        self._timestep_transform_name = best_name
        return best_schedule, self._schedule_transform(best_name, train_scale)

    def _schedule_transform_candidates(
        self,
        schedule: torch.Tensor,
        train_scale: float,
        schedule_abs_max: float,
        timestep_abs_max: float,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        yield "identity", schedule
        if schedule_abs_max > 1.5 and timestep_abs_max <= 1.5:
            yield "normalized", schedule / train_scale
            yield "inverted_normalized", (train_scale - schedule) / train_scale
        elif schedule_abs_max <= 1.5 and timestep_abs_max > 1.5:
            yield "scaled", schedule * train_scale
            yield "inverted_scaled", (1.0 - schedule) * train_scale
        elif schedule_abs_max <= 1.5 and timestep_abs_max <= 1.5:
            yield "inverted_unit", 1.0 - schedule

    @staticmethod
    def _schedule_transform(name: str, train_scale: float):
        if name == "normalized":
            return lambda values: values / train_scale
        if name == "inverted_normalized":
            return lambda values: (train_scale - values) / train_scale
        if name == "scaled":
            return lambda values: values * train_scale
        if name == "inverted_scaled":
            return lambda values: (1.0 - values) * train_scale
        if name == "inverted_unit":
            return lambda values: 1.0 - values
        return lambda values: values
