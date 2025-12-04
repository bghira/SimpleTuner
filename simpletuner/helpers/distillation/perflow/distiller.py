from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from simpletuner.diff2flow import DiffusionToFlowBridge
from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.distillation.common import DistillationBase
from simpletuner.helpers.distillation.perflow.generator import PerFlowODEGenerator
from simpletuner.helpers.distillation.registry import DistillationRegistry
from simpletuner.helpers.training.custom_schedule import PeRFlowScheduler
from simpletuner.helpers.training.multi_process import rank_info
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)


class PerFlowDistiller(DistillationBase):
    """PerFlow/ReFlow-style distillation that trains from cached ODE endpoints."""

    _DEFAULTS: Dict[str, Any] = {
        "distillation_type": "perflow",
        "timestep_sampler": "u_shaped",
        "loss_type": "l2",
        "huber_c": 0.01,
        "loss_weight": 1.0,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "set_alpha_to_one": False,
        "t_noise": 1.0,
        "t_clean": 0.0,
        "num_time_windows": 4,
        "num_inference_steps": 30,
        "ode_generator": {},
    }

    def __init__(
        self,
        teacher_model,
        student_model=None,
        *,
        noise_scheduler=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        merged_config: Dict[str, Any] = dict(self._DEFAULTS)
        if config:
            merged_config.update(config)

        super().__init__(teacher_model, student_model, merged_config)
        self.noise_scheduler = noise_scheduler
        self.rank_info = rank_info()

        self._required_cache_type = self.config.get("distillation_type", "perflow")
        self._scheduler_config = self._build_scheduler_config(noise_scheduler)
        self._distillation_scheduler = self._build_scheduler()
        self._flow_bridge = DiffusionToFlowBridge(alphas_cumprod=self._distillation_scheduler.alphas_cumprod)

        ode_config = dict(self.config.get("ode_generator", {}))
        ode_config.setdefault("distillation_type", self._required_cache_type)
        self._ode_generator = PerFlowODEGenerator(
            teacher_model=self.teacher_model,
            scheduler_provider=self._build_scheduler,
            config=ode_config,
        )

        self._distillation_caches: List[Any] = []
        self._cache_cycle = None
        self._scheduler_metadata = self._describe_scheduler(self._distillation_scheduler)

    def _build_scheduler_config(self, noise_scheduler) -> Dict[str, Any]:
        source_cfg = getattr(noise_scheduler, "config", None)
        return {
            "num_train_timesteps": int(
                self.config.get("num_train_timesteps", getattr(source_cfg, "num_train_timesteps", 1000))
            ),
            "beta_start": float(self.config.get("beta_start", getattr(source_cfg, "beta_start", 0.00085))),
            "beta_end": float(self.config.get("beta_end", getattr(source_cfg, "beta_end", 0.012))),
            "beta_schedule": self.config.get("beta_schedule", getattr(source_cfg, "beta_schedule", "scaled_linear")),
            "trained_betas": getattr(source_cfg, "trained_betas", None),
            "set_alpha_to_one": bool(self.config.get("set_alpha_to_one", getattr(source_cfg, "set_alpha_to_one", False))),
            "prediction_type": "flow_matching",
            "t_noise": float(self.config.get("t_noise", getattr(source_cfg, "t_noise", 1.0))),
            "t_clean": float(self.config.get("t_clean", getattr(source_cfg, "t_clean", 0.0))),
            "num_time_windows": int(self.config.get("num_time_windows", getattr(source_cfg, "num_time_windows", 4))),
        }

    def _build_scheduler(self) -> PeRFlowScheduler:
        scheduler = PeRFlowScheduler(**self._scheduler_config)
        return scheduler

    def requires_distillation_cache(self) -> bool:
        return True

    def get_required_distillation_cache_type(self) -> Optional[str]:
        return self._required_cache_type

    def get_ode_generator_provider(self):
        return self._ode_generator

    def get_scheduler(self, *_):
        return self._build_scheduler()

    def _ensure_cache_cycle(self) -> None:
        if self._cache_cycle is not None and self._distillation_caches:
            return

        try:
            backend_map = StateTracker.get_data_backends(
                _type=DatasetType.DISTILLATION_CACHE,
                _types=[DatasetType.DISTILLATION_CACHE],
            )
        except Exception:
            backend_map = {}

        caches = []
        for backend in backend_map.values():
            cache = backend.get("distillation_cache")
            if cache is not None:
                caches.append(cache)

        if not caches:
            raise ValueError("PerFlow distillation requires at least one distillation_cache backend.")

        self._distillation_caches = caches
        self._cache_cycle = itertools.cycle(self._distillation_caches)

    def _next_cache_payload(self) -> Dict[str, Any]:
        self._ensure_cache_cycle()
        if not self._distillation_caches:
            raise ValueError("No distillation caches were configured for PerFlow distillation.")

        for _ in range(len(self._distillation_caches)):
            cache = next(self._cache_cycle)
            payload, artifact_path = cache.load_next_pair()
            if payload is None:
                continue

            entry = payload if isinstance(payload, dict) else {"value": payload}
            metadata = entry.get("metadata") or {}
            metadata.setdefault("artifact_path", artifact_path)
            metadata.setdefault("distillation_type", cache.distillation_type)
            self._validate_cache_metadata(metadata)
            entry["metadata"] = metadata
            return entry

        raise ValueError("Distillation caches did not yield any artifacts for training.")

    def _validate_cache_metadata(self, metadata: Dict[str, Any]) -> None:
        cache_type = metadata.get("distillation_type")
        if cache_type and cache_type != self._required_cache_type:
            raise ValueError(
                f"{self.rank_info}PerFlow distiller expected cache type '{self._required_cache_type}' "
                f"but received '{cache_type}'."
            )

        scheduler_meta = metadata.get("scheduler") or {}
        expected = self._scheduler_metadata
        for key, expected_value in expected.items():
            value = scheduler_meta.get(key)
            if value is None or expected_value is None:
                continue
            if expected_value != value:
                raise ValueError(
                    f"{self.rank_info}Cache scheduler mismatch for key '{key}': expected {expected_value}, got {value}."
                )

    def _describe_scheduler(self, scheduler: PeRFlowScheduler) -> Dict[str, Any]:
        cfg = getattr(scheduler, "config", None)
        return {
            "name": scheduler.__class__.__name__,
            "num_train_timesteps": getattr(cfg, "num_train_timesteps", None),
            "prediction_type": getattr(cfg, "prediction_type", None),
            "beta_schedule": getattr(cfg, "beta_schedule", None),
            "beta_start": getattr(cfg, "beta_start", None),
            "beta_end": getattr(cfg, "beta_end", None),
            "t_noise": getattr(cfg, "t_noise", None),
            "t_clean": getattr(cfg, "t_clean", None),
            "num_time_windows": getattr(cfg, "num_time_windows", None),
            "set_alpha_to_one": getattr(cfg, "set_alpha_to_one", None),
        }

    def _stack_cached_tensor(
        self,
        cache_entries: Sequence[Dict[str, Any]],
        key: str,
        expected_dim: int,
    ) -> torch.Tensor:
        tensors: List[torch.Tensor] = []
        for entry in cache_entries:
            value = entry.get(key)
            if value is None:
                raise ValueError(f"Distillation cache entry missing '{key}' tensor.")
            tensor = value if torch.is_tensor(value) else torch.tensor(value, dtype=torch.float32)
            if tensor.ndim == expected_dim + 1 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim != expected_dim:
                raise ValueError(
                    f"Expected cached '{key}' tensor to have {expected_dim} dimensions; " f"got shape {tuple(tensor.shape)}"
                )
            tensors.append(tensor.clone().detach())

        return torch.stack(tensors, dim=0)

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        num_train = int(self._scheduler_config["num_train_timesteps"])
        sampler = str(self.config.get("timestep_sampler", "uniform")).lower()
        if sampler == "u_shaped":
            dist = torch.distributions.Beta(torch.tensor(0.5), torch.tensor(0.5))
            samples = dist.sample((batch_size,)).to(device=device)
        else:
            samples = torch.rand((batch_size,), device=device)
        timesteps = torch.clamp((samples * (num_train - 1)).long(), min=1, max=num_train - 1)
        return timesteps

    def _resolve_batch_size(self, prepared_batch: Dict[str, Any]) -> int:
        for key in ("latents", "noisy_latents", "prompt_embeds", "pixel_values", "input_ids"):
            value = prepared_batch.get(key)
            if torch.is_tensor(value):
                return int(value.shape[0])
            if isinstance(value, (list, tuple)) and value and torch.is_tensor(value[0]):
                return int(value[0].shape[0])
        return int(self.config.get("train_batch_size", 1))

    def prepare_batch(self, batch: Dict[str, Any], *_):
        batch_size = self._resolve_batch_size(batch)
        cache_entries = [self._next_cache_payload() for _ in range(batch_size)]

        latents = self._stack_cached_tensor(cache_entries, "latents", expected_dim=3)
        noise = self._stack_cached_tensor(cache_entries, "noise", expected_dim=3)

        target_device = batch.get("latents", batch.get("noisy_latents", None))
        if torch.is_tensor(target_device):
            device = target_device.device
            dtype = target_device.dtype
        else:
            accelerator = getattr(self.teacher_model, "accelerator", None)
            device = getattr(accelerator, "device", torch.device("cpu"))
            dtype = getattr(self.teacher_model, "weight_dtype", torch.float32)

        latents = latents.to(device=device, dtype=dtype)
        noise = noise.to(device=device, dtype=dtype)

        timesteps = self._sample_timesteps(batch_size, device=device)
        noisy_latents = self._distillation_scheduler.add_noise(latents.float(), noise.float(), timesteps).to(
            device=device, dtype=dtype
        )
        flow_target = noise - latents

        batch["latents"] = latents
        batch["clean_latents"] = latents
        batch["noise"] = noise
        batch["input_noise"] = noise
        batch["timesteps"] = timesteps
        batch["noisy_latents"] = noisy_latents
        batch["flow_target"] = flow_target
        batch["distillation_metadata"] = [entry.get("metadata", {}) for entry in cache_entries]
        batch["distillation_cache_entries"] = cache_entries
        return batch

    def compute_distill_loss(
        self,
        prepared_batch: Dict[str, Any],
        model_output: Dict[str, Any],
        _original_loss: torch.Tensor,
    ):
        prediction_type = getattr(getattr(self.student_model, "PREDICTION_TYPE", None), "value", None)
        model_pred = model_output["model_prediction"]

        self._flow_bridge.to(device=model_pred.device, dtype=model_pred.dtype)
        if prediction_type in ("flow_matching", "flow"):
            pred_flow = model_pred
        else:
            pred_flow = self._flow_bridge.prediction_to_flow(
                model_pred.float(),
                prepared_batch["noisy_latents"].float(),
                prepared_batch["timesteps"],
                prediction_type=prediction_type or "epsilon",
            )

        flow_target = prepared_batch.get("flow_target")
        if flow_target is None:
            flow_target = prepared_batch["noise"] - prepared_batch["latents"]

        loss_type = str(self.config.get("loss_type", "l2")).lower()
        if loss_type in ("huber", "smooth_l1"):
            beta = float(self.config.get("huber_c", 0.01))
            loss = F.smooth_l1_loss(pred_flow.float(), flow_target.float(), beta=beta)
        else:
            loss = F.mse_loss(pred_flow.float(), flow_target.float())

        weight = float(self.config.get("loss_weight", 1.0))
        loss = loss * weight

        logs = {
            "perflow_loss": float(loss.detach()),
            "perflow_timestep": float(torch.mean(prepared_batch["timesteps"].float()).detach()),
            "total": float(loss.detach()),
        }
        return loss, logs


DistillationRegistry.register(
    "perflow",
    PerFlowDistiller,
    requires_distillation_cache=True,
    distillation_type="perflow",
    data_requirements=[[DatasetType.IMAGE, DatasetType.VIDEO]],
    requirement_notes="Requires distillation_cache backend matching distillation_type.",
)
