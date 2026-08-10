from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

FLOWMAP_KEYS = (
    "flowmap_r_timesteps",
    "anyflow_r_timesteps",
    "anyflow_timestep_interval",
)


@dataclass(frozen=True)
class GeometryPoint:
    timestep: float
    sigma: float
    model_timestep: float
    r_timestep: float
    r_sigma: float
    anyflow_weight: float
    drift_weight: float
    sft_weight: float
    normal_target: torch.Tensor
    anyflow_target: torch.Tensor
    flowmap_objective_target: torch.Tensor
    base_prediction: torch.Tensor
    drift_reference_prediction: torch.Tensor
    normal_batch: dict[str, Any]
    prepared_batch: dict[str, Any]
    branch: str = "arbitrary"
    guidance_scale: float = 1.0


def _flat_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().reshape(-1)


def tensor_norm(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(_flat_float(tensor)).cpu())


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = _flat_float(left)
    right_flat = _flat_float(right)
    denominator = torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat)
    if float(denominator) == 0.0:
        return float("nan")
    return float(torch.dot(left_flat, right_flat).div(denominator).cpu())


def norm_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    denominator_norm = tensor_norm(denominator)
    if denominator_norm == 0.0:
        return float("nan")
    return tensor_norm(numerator) / denominator_norm


def mean_squared_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.mean((_flat_float(left) - _flat_float(right)).square()).cpu())


def trajectory_metrics(
    *,
    adapter_label: str,
    point: GeometryPoint,
    adapter_prediction: torch.Tensor,
    normal_adapter_prediction: torch.Tensor,
) -> dict[str, Any]:
    adapter_residual = adapter_prediction - point.drift_reference_prediction
    anyflow_correction = point.anyflow_target - point.drift_reference_prediction
    objective_correction = point.flowmap_objective_target - point.drift_reference_prediction
    normal_adapter_residual = normal_adapter_prediction - point.base_prediction
    target_delta = point.anyflow_target - point.normal_target
    flowmap_base_residual = point.drift_reference_prediction - point.base_prediction
    return {
        "adapter": adapter_label,
        "branch": point.branch,
        "guidance_scale": point.guidance_scale,
        "timestep": point.timestep,
        "sigma": point.sigma,
        "model_timestep": point.model_timestep,
        "r_timestep": point.r_timestep,
        "r_sigma": point.r_sigma,
        "interval": point.sigma - point.r_sigma,
        "anyflow_weight": point.anyflow_weight,
        "drift_weight": point.drift_weight,
        "sft_weight": point.sft_weight,
        "cos_adapter_base": cosine_similarity(adapter_prediction, point.base_prediction),
        "adapter_base_norm_ratio": norm_ratio(adapter_prediction, point.base_prediction),
        "cos_adapter_drift_reference": cosine_similarity(adapter_prediction, point.drift_reference_prediction),
        "adapter_drift_reference_norm_ratio": norm_ratio(adapter_prediction, point.drift_reference_prediction),
        "adapter_residual_norm": tensor_norm(adapter_residual),
        "adapter_residual_base_norm_ratio": norm_ratio(adapter_residual, point.base_prediction),
        "cos_adapter_residual_anyflow_correction": cosine_similarity(adapter_residual, anyflow_correction),
        "adapter_residual_anyflow_correction_norm_ratio": norm_ratio(adapter_residual, anyflow_correction),
        "cos_adapter_flowmap_objective": cosine_similarity(adapter_prediction, point.flowmap_objective_target),
        "adapter_flowmap_objective_norm_ratio": norm_ratio(adapter_prediction, point.flowmap_objective_target),
        "adapter_flowmap_objective_mse": mean_squared_error(adapter_prediction, point.flowmap_objective_target),
        "cos_adapter_residual_objective_correction": cosine_similarity(adapter_residual, objective_correction),
        "adapter_residual_objective_correction_norm_ratio": norm_ratio(adapter_residual, objective_correction),
        "normal_adapter_residual_norm": tensor_norm(normal_adapter_residual),
        "normal_adapter_residual_base_norm_ratio": norm_ratio(normal_adapter_residual, point.base_prediction),
        "cos_normal_adapter_base": cosine_similarity(normal_adapter_prediction, point.base_prediction),
        "normal_adapter_base_norm_ratio": norm_ratio(normal_adapter_prediction, point.base_prediction),
        "cos_normal_adapter_normal_target": cosine_similarity(normal_adapter_prediction, point.normal_target),
        "normal_adapter_normal_target_norm_ratio": norm_ratio(normal_adapter_prediction, point.normal_target),
        "normal_adapter_normal_target_mse": mean_squared_error(normal_adapter_prediction, point.normal_target),
        "cos_anyflow_normal_target": cosine_similarity(point.anyflow_target, point.normal_target),
        "anyflow_normal_target_norm_ratio": norm_ratio(point.anyflow_target, point.normal_target),
        "anyflow_target_delta_norm": tensor_norm(target_delta),
        "anyflow_target_delta_normal_norm_ratio": norm_ratio(target_delta, point.normal_target),
        "cos_base_normal_target": cosine_similarity(point.base_prediction, point.normal_target),
        "base_normal_target_norm_ratio": norm_ratio(point.base_prediction, point.normal_target),
        "base_normal_target_mse": mean_squared_error(point.base_prediction, point.normal_target),
        "cos_drift_reference_anyflow_target": cosine_similarity(point.drift_reference_prediction, point.anyflow_target),
        "drift_reference_anyflow_target_norm_ratio": norm_ratio(point.drift_reference_prediction, point.anyflow_target),
        "drift_reference_anyflow_target_mse": mean_squared_error(point.drift_reference_prediction, point.anyflow_target),
        "cos_drift_reference_base": cosine_similarity(point.drift_reference_prediction, point.base_prediction),
        "flowmap_base_residual_norm": tensor_norm(flowmap_base_residual),
        "flowmap_base_residual_base_norm_ratio": norm_ratio(flowmap_base_residual, point.base_prediction),
        "adapter_prediction_norm": tensor_norm(adapter_prediction),
        "base_prediction_norm": tensor_norm(point.base_prediction),
        "drift_reference_prediction_norm": tensor_norm(point.drift_reference_prediction),
        "normal_target_norm": tensor_norm(point.normal_target),
        "anyflow_target_norm": tensor_norm(point.anyflow_target),
        "flowmap_objective_target_norm": tensor_norm(point.flowmap_objective_target),
        "normal_adapter_prediction_norm": tensor_norm(normal_adapter_prediction),
    }


def sample_vector(tensor: torch.Tensor, max_elements: int) -> np.ndarray:
    flattened = _flat_float(tensor).cpu().numpy()
    if max_elements <= 0 or flattened.size <= max_elements:
        return flattened
    indices = np.linspace(0, flattened.size - 1, num=max_elements, dtype=np.int64)
    return flattened[indices]


def pca_coordinates(vectors: Iterable[np.ndarray]) -> np.ndarray:
    matrix = np.stack(list(vectors)).astype(np.float64, copy=False)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if matrix.shape[0] < 2 or not np.any(centered):
        return np.zeros((matrix.shape[0], 2), dtype=np.float64)
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    coordinates = left[:, :2] * singular_values[:2]
    if coordinates.shape[1] == 1:
        coordinates = np.pad(coordinates, ((0, 0), (0, 1)))
    return coordinates


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "value"


def _prediction_tensor(output: dict[str, Any]) -> torch.Tensor:
    prediction = output.get("model_prediction")
    if not torch.is_tensor(prediction):
        raise ValueError("MiniMax-H3 diagnostic expected a tensor model_prediction.")
    hidden_states_buffer = output.get("hidden_states_buffer")
    if isinstance(hidden_states_buffer, dict):
        hidden_states_buffer.clear()
    return prediction.detach()


def _without_flowmap_conditioning(batch: dict[str, Any]) -> dict[str, Any]:
    normal_batch = dict(batch)
    for key in FLOWMAP_KEYS:
        normal_batch.pop(key, None)
    normal_batch.pop("target", None)
    normal_batch.pop("flow_target", None)
    return normal_batch


def _adapter_prediction(
    model,
    distiller,
    batch: dict[str, Any],
    *,
    enabled: bool,
    objective_space: bool = False,
) -> torch.Tensor:
    distiller.toggle_adapter(enable=enabled)
    try:
        with torch.no_grad():
            prediction = _prediction_tensor(model.model_predict(batch))
            if objective_space:
                anyflow_distiller = (
                    distiller
                    if distiller.__class__.__name__ == "AnyFlowDistiller"
                    else getattr(distiller, "inner_distiller", None)
                )
                if anyflow_distiller is None or anyflow_distiller.__class__.__name__ != "AnyFlowDistiller":
                    raise ValueError("Objective-space prediction requires an AnyFlow distiller.")
                prediction = anyflow_distiller._fuse_guidance_prediction(batch, prediction)
            return prediction.float().cpu()
    finally:
        distiller.toggle_adapter(enable=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scalar_batch_value(batch: dict[str, Any], key: str, fallback: float) -> float:
    value = batch.get(key)
    if not torch.is_tensor(value):
        return fallback
    return float(value.detach().float().reshape(-1)[0].cpu())


def _explicit_anyflow_batch(
    inner_distiller,
    normal_batch: dict[str, Any],
    model,
    branch: str,
    t_sigmas: torch.Tensor | None = None,
) -> dict[str, Any]:
    anyflow_batch = dict(normal_batch)
    if t_sigmas is None:
        t_sigmas = inner_distiller._scalar_sigmas(anyflow_batch)
    else:
        t_sigmas = t_sigmas.to(device=anyflow_batch["latents"].device, dtype=torch.float32)
    if branch == "diffusion":
        r_sigmas = t_sigmas.clone()
    elif branch == "consistency":
        r_sigmas = torch.zeros_like(t_sigmas)
    elif branch == "midpoint":
        r_sigmas = t_sigmas * 0.5
    else:
        raise ValueError(f"Unknown AnyFlow diagnostic branch: {branch!r}.")

    inner_distiller._set_batch_sigma_path(anyflow_batch, t_sigmas)
    r_timesteps = inner_distiller._timesteps_from_sigmas(r_sigmas, anyflow_batch["timesteps"]).to(
        device=anyflow_batch["timesteps"].device,
        dtype=anyflow_batch["timesteps"].dtype,
    )
    flowmap_key = getattr(model, "FLOWMAP_R_TIMESTEP_BATCH_KEY", inner_distiller.FLOWMAP_R_TIMESTEP_BATCH_KEY)
    anyflow_batch[flowmap_key] = r_timesteps
    anyflow_batch["anyflow_r_timesteps"] = r_timesteps
    anyflow_batch["anyflow_timestep_interval"] = (anyflow_batch["timesteps"] - r_timesteps).abs()
    anyflow_batch["anyflow_t_sigmas"] = t_sigmas
    anyflow_batch["anyflow_r_sigmas"] = r_sigmas
    for candidate in ("diffusion", "consistency", "arbitrary"):
        selected = candidate == branch or (candidate == "arbitrary" and branch == "midpoint")
        anyflow_batch[f"anyflow_{candidate}_mask"] = torch.full_like(t_sigmas, selected, dtype=torch.bool)

    base_target = inner_distiller._base_flow_target(anyflow_batch, model=model)
    anyflow_batch["target"] = inner_distiller._meanflow_target(
        prepared_batch=anyflow_batch,
        model=model,
        t_sigmas=t_sigmas,
        r_sigmas=r_sigmas,
        base_target=base_target,
    ).detach()
    anyflow_batch["flow_target"] = anyflow_batch["target"]
    return anyflow_batch


def prepare_geometry_points(
    trainer,
    prepared_batch: dict[str, Any],
    timesteps: list[float],
    branches: list[str],
    seed: int,
):
    distiller = trainer.distiller
    if distiller.__class__.__name__ == "AnyFlowDistiller":
        inner_distiller = distiller
        drift_weight = 0.0
        sft_weight = 0.0
    else:
        inner_distiller = getattr(distiller, "inner_distiller", None)
        if inner_distiller is None or inner_distiller.__class__.__name__ != "AnyFlowDistiller":
            raise ValueError("H3 objective geometry requires AnyFlow directly or wrapped by H3_DRIFT.")
        drift_weight = float(distiller.config.get("loss_weight", 1.0))
        sft_weight = float(distiller.config.get("sft_loss_weight", 1.0))

    points = []
    anyflow_weight = float(inner_distiller.config.get("loss_weight", 1.0))
    guidance_scale = float(inner_distiller.config.get("fuse_guidance_scale", 1.0))
    flowmap_weight = anyflow_weight + drift_weight
    if flowmap_weight <= 0.0:
        raise ValueError("H3 objective geometry requires a positive AnyFlow or H3_DRIFT loss weight.")
    for index, timestep in enumerate(timesteps):
        normal_batch = trainer._prepare_custom_timestep_batch(prepared_batch, [timestep])
        normal_batch = _without_flowmap_conditioning(normal_batch)
        timestep_scale = float(getattr(inner_distiller, "num_train_timesteps", 1000.0) or 1000.0)
        requested_sigma = float(timestep) / timestep_scale if float(timestep) > 1.0 else float(timestep)
        explicit_t_sigmas = torch.full(
            (normal_batch["latents"].shape[0],),
            requested_sigma,
            device=normal_batch["latents"].device,
            dtype=torch.float32,
        )
        if hasattr(distiller, "_normal_video_target"):
            normal_target = distiller._normal_video_target(normal_batch).float().cpu()
        else:
            normal_target = (
                trainer.model.get_flow_matching_target(
                    normal_batch,
                    prefer_explicit_target=False,
                )
                .float()
                .cpu()
            )

        base_prediction = _adapter_prediction(
            trainer.model,
            distiller,
            normal_batch,
            enabled=False,
            objective_space=True,
        )
        sigma = requested_sigma
        model_timestep = _scalar_batch_value(normal_batch, "timesteps", 1.0 - sigma)
        for branch_index, branch in enumerate(branches):
            _seed_everything(seed + index * len(branches) + branch_index)
            anyflow_batch = _explicit_anyflow_batch(
                inner_distiller,
                normal_batch,
                trainer.model,
                branch,
                t_sigmas=explicit_t_sigmas,
            )
            anyflow_target = anyflow_batch["target"].detach().float().cpu()
            drift_reference = _adapter_prediction(
                trainer.model,
                distiller,
                anyflow_batch,
                enabled=False,
                objective_space=True,
            )
            flowmap_objective_target = (anyflow_weight * anyflow_target + drift_weight * drift_reference) / flowmap_weight
            r_timestep = _scalar_batch_value(anyflow_batch, "anyflow_r_timesteps", float("nan"))
            r_sigma = _scalar_batch_value(anyflow_batch, "anyflow_r_sigmas", 1.0 - r_timestep)
            points.append(
                GeometryPoint(
                    timestep=float(timestep),
                    sigma=sigma,
                    model_timestep=model_timestep,
                    r_timestep=r_timestep,
                    r_sigma=r_sigma,
                    anyflow_weight=anyflow_weight,
                    drift_weight=drift_weight,
                    sft_weight=sft_weight,
                    normal_target=normal_target,
                    anyflow_target=anyflow_target,
                    flowmap_objective_target=flowmap_objective_target,
                    base_prediction=base_prediction,
                    drift_reference_prediction=drift_reference,
                    normal_batch=normal_batch,
                    prepared_batch=anyflow_batch,
                    branch=branch,
                    guidance_scale=guidance_scale,
                )
            )
    return points


def _load_adapter(trainer, checkpoint_dir: Path) -> None:
    component = trainer.model.get_trained_component(unwrap_model=False)
    trainer.model.load_lora_weights([component], str(checkpoint_dir))


def _first_raw_batch():
    from simpletuner.helpers.data_backend.runtime import random_dataloader_iterator
    from simpletuner.helpers.training.state_tracker import StateTracker

    backends = {
        backend_id: backend["train_dataloader"]
        for backend_id, backend in StateTracker.get_data_backends().items()
        if "train_dataloader" in backend and not StateTracker.backend_status(backend_id)
    }
    raw_batch = random_dataloader_iterator(1, backends)
    if raw_batch is False or not isinstance(raw_batch, dict):
        raise RuntimeError("Unable to fetch a diagnostic batch from the configured training data backend.")
    return raw_batch


def initialize_trainer(config: dict[str, Any]):
    from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase
    from simpletuner.helpers.training.trainer import Trainer

    trainer = Trainer(config=config, exit_on_error=True)
    trainer.init_noise_schedule()
    trainer.init_seed()
    trainer.init_huggingface_hub()
    trainer.init_preprocessing_models()
    trainer.init_precision(preprocessing_models_only=True)
    trainer.init_data_backend()
    trainer.init_unload_text_encoder()
    trainer.init_unload_vae()
    trainer.init_load_base_model()
    trainer.init_delete_model_caches()
    trainer.init_controlnet_model()
    trainer.init_tread_model()
    trainer.init_precision()
    trainer.init_freeze_models()
    trainer.init_distillation_adapter_modules()
    trainer.init_trainable_peft_adapter()
    trainer.move_models(destination="accelerator")
    if (getattr(trainer.config, "musubi_blocks_to_swap", 0) or 0) > 0:
        trainer._move_model_with_block_swap(trainer.model.get_trained_component(unwrap_model=False))
    trainer.init_distillation()
    AttentionBackendController.apply(trainer.config, AttentionPhase.TRAIN)
    trainer.model.get_trained_component(unwrap_model=False).eval()
    return trainer


def _diagnostic_config(config_path: Path, output_dir: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config.update(
        {
            "checkpoint_step_interval": 0,
            "dataloader_prefetch": False,
            "max_train_steps": 1,
            "output_dir": str(output_dir / "trainer-output"),
            "push_checkpoints_to_hub": False,
            "push_to_hub": False,
            "report_to": "none",
            "resume_from_checkpoint": None,
            "validation_on_startup": False,
            "validation_step_interval": 0,
            "validation_steps": 0,
        }
    )
    return config


def _parse_adapter(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Adapters must use LABEL=/path/to/checkpoint syntax.")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Adapters must have a non-empty label and path.")
    return label.strip(), Path(path).expanduser()


def _parse_timesteps(value: str) -> list[float]:
    try:
        timesteps = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Timesteps must be comma-separated numbers.") from exc
    if not timesteps or any(timestep <= 0.0 or timestep > 1000.0 for timestep in timesteps):
        raise argparse.ArgumentTypeError("Timesteps must be in (0, 1000].")
    return timesteps


def _parse_branches(value: str) -> list[str]:
    branches = [item.strip().lower() for item in value.split(",") if item.strip()]
    supported = {"diffusion", "consistency", "midpoint"}
    unsupported = sorted(set(branches) - supported)
    if not branches or unsupported:
        choices = ", ".join(sorted(supported))
        raise argparse.ArgumentTypeError(f"Branches must be selected from {choices}; got {unsupported or branches}.")
    return branches


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_npz(
    path: Path,
    points: list[GeometryPoint],
    adapter_vectors: dict[tuple[str, float, str, str], torch.Tensor],
    max_vector_elements: int,
) -> tuple[list[str], list[np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    pca_labels: list[str] = []
    pca_vectors: list[np.ndarray] = []
    for point in points:
        timestep_name = f"{point.branch}:t{point.timestep:g}"
        for kind, tensor in (
            ("normal_target", point.normal_target),
            ("anyflow_target", point.anyflow_target),
            ("flowmap_objective_target", point.flowmap_objective_target),
            ("base_prediction", point.base_prediction),
            ("drift_reference", point.drift_reference_prediction),
        ):
            label = f"{kind}:{timestep_name}"
            vector = sample_vector(tensor, max_vector_elements)
            arrays[_safe_name(label)] = vector
            pca_labels.append(label)
            pca_vectors.append(vector)
    for (adapter_label, timestep, interval_branch, prediction_branch), tensor in adapter_vectors.items():
        label = f"adapter_prediction:{prediction_branch}:{interval_branch}:{adapter_label}:t{timestep:g}"
        vector = sample_vector(tensor, max_vector_elements)
        arrays[_safe_name(label)] = vector
        pca_labels.append(label)
        pca_vectors.append(vector)
    arrays["metadata_json"] = np.asarray(json.dumps({"pca_labels": pca_labels, "max_vector_elements": max_vector_elements}))
    np.savez_compressed(path, **arrays)
    return pca_labels, pca_vectors


def _write_plots(
    output_dir: Path,
    rows: list[dict[str, Any]],
    pca_labels: list[str],
    pca_vectors: list[np.ndarray],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to write H3 geometry plots.") from exc

    coordinates = pca_coordinates(pca_vectors)
    figure, axis = plt.subplots(figsize=(11, 8))
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(pca_labels):
        group = label.split(":", 1)[0]
        groups.setdefault(group, []).append(index)
    for group, indices in groups.items():
        axis.scatter(coordinates[indices, 0], coordinates[indices, 1], label=group, s=24)
    axis.set_title("MiniMax-H3 objective trajectory PCA")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "pca_trajectory.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    adapter_groups = list(dict.fromkeys((str(row["adapter"]), str(row["branch"])) for row in rows))
    for adapter_label, branch in adapter_groups:
        adapter_rows = sorted(
            (row for row in rows if row["adapter"] == adapter_label and row["branch"] == branch),
            key=lambda row: float(row["timestep"]),
        )
        timesteps = [float(row["timestep"]) for row in adapter_rows]
        axes[0].plot(
            timesteps,
            [float(row["cos_adapter_flowmap_objective"]) for row in adapter_rows],
            marker="o",
            label=f"{adapter_label}/{branch}: objective cosine",
        )
        axes[1].plot(
            timesteps,
            [float(row["adapter_residual_base_norm_ratio"]) for row in adapter_rows],
            marker="o",
            label=f"{adapter_label}/{branch}: residual/base norm",
        )
        axes[1].plot(
            timesteps,
            [float(row["normal_adapter_residual_base_norm_ratio"]) for row in adapter_rows],
            marker="x",
            linestyle=":",
            label=f"{adapter_label}/{branch}: normal residual/base norm",
        )
    first_by_timestep = {}
    for row in rows:
        first_by_timestep.setdefault((str(row["branch"]), float(row["timestep"])), row)
    target_branches = list(dict.fromkeys(branch for branch, _ in first_by_timestep))
    for branch in target_branches:
        target_rows = [
            first_by_timestep[key]
            for key in sorted((key for key in first_by_timestep if key[0] == branch), key=lambda key: key[1])
        ]
        target_timesteps = [float(row["timestep"]) for row in target_rows]
        axes[0].plot(
            target_timesteps,
            [float(row["cos_anyflow_normal_target"]) for row in target_rows],
            linestyle="--",
            label=f"{branch}: AnyFlow/normal cosine",
        )
        axes[1].plot(
            target_timesteps,
            [float(row["anyflow_normal_target_norm_ratio"]) for row in target_rows],
            linestyle="--",
            label=f"{branch}: AnyFlow/normal norm",
        )
    axes[0].set_ylabel("Cosine similarity")
    axes[1].set_ylabel("Norm ratio")
    axes[1].set_xlabel("Timestep")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("MiniMax-H3 norm and cosine geometry")
    figure.tight_layout()
    figure.savefig(output_dir / "norm_cosine_by_timestep.png", dpi=160)
    plt.close(figure)


def run_diagnostic(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = _diagnostic_config(args.config, args.output_dir)
    trainer = initialize_trainer(config)
    try:
        _seed_everything(args.seed)
        raw_batch = _first_raw_batch()
        prepared_batch = trainer.model.prepare_batch(raw_batch, state=trainer.state)
        points = prepare_geometry_points(trainer, prepared_batch, args.timesteps, args.branches, args.seed)

        rows = []
        adapter_vectors: dict[tuple[str, float, str, str], torch.Tensor] = {}
        normal_prediction_cache: dict[tuple[str, float], torch.Tensor] = {}
        adapters = [("base", None), *args.adapter]
        for adapter_label, checkpoint_dir in adapters:
            if checkpoint_dir is not None:
                _load_adapter(trainer, checkpoint_dir)
            for point in points:
                adapter_prediction = _adapter_prediction(
                    trainer.model,
                    trainer.distiller,
                    point.prepared_batch,
                    enabled=checkpoint_dir is not None,
                    objective_space=True,
                )
                normal_cache_key = (adapter_label, point.timestep)
                if normal_cache_key not in normal_prediction_cache:
                    normal_prediction_cache[normal_cache_key] = _adapter_prediction(
                        trainer.model,
                        trainer.distiller,
                        point.normal_batch,
                        enabled=checkpoint_dir is not None,
                        objective_space=True,
                    )
                normal_adapter_prediction = normal_prediction_cache[normal_cache_key]
                rows.append(
                    trajectory_metrics(
                        adapter_label=adapter_label,
                        point=point,
                        adapter_prediction=adapter_prediction,
                        normal_adapter_prediction=normal_adapter_prediction,
                    )
                )
                adapter_vectors[(adapter_label, point.timestep, point.branch, "flowmap")] = adapter_prediction
                adapter_vectors[(adapter_label, point.timestep, point.branch, "normal")] = normal_adapter_prediction

        _write_csv(args.output_dir / "trajectory_metrics.csv", rows)
        pca_labels, pca_vectors = _write_npz(
            args.output_dir / "trajectory_vectors.npz",
            points,
            adapter_vectors,
            args.max_vector_elements,
        )
        _write_plots(args.output_dir, rows, pca_labels, pca_vectors)
        with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": str(args.config),
                    "adapters": [(label, None if path is None else str(path)) for label, path in adapters],
                    "timesteps": args.timesteps,
                    "branches": args.branches,
                    "seed": args.seed,
                    "max_vector_elements": args.max_vector_elements,
                    "batch": {
                        "data_backend_id": prepared_batch.get("data_backend_id"),
                        "filepaths": [str(path) for path in prepared_batch.get("filepaths", [])],
                    },
                },
                handle,
                indent=2,
            )
    finally:
        trainer.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect MiniMax-H3 AnyFlow/H3_DRIFT objective geometry.")
    parser.add_argument("--config", type=Path, required=True, help="SimpleTuner config.json path.")
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        type=_parse_adapter,
        metavar="LABEL=CHECKPOINT_DIR",
        help="Checkpoint adapter to compare; may be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--timesteps",
        type=_parse_timesteps,
        default=_parse_timesteps("50,100,250,500,750,900,975"),
    )
    parser.add_argument(
        "--branches",
        type=_parse_branches,
        default=_parse_branches("diffusion,consistency,midpoint"),
        help="Comma-separated explicit AnyFlow intervals to inspect.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-vector-elements", type=int, default=65536)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_diagnostic(args)


if __name__ == "__main__":
    main()
