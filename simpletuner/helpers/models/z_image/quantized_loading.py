from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlparse

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors import safe_open

PATCH_KEY = "2-1"


def _resolve_zimage_single_file_path(
    pretrained_model_link_or_path: str,
    *,
    filename: str | None = None,
    subfolder: str | None = None,
    revision: str | None = None,
) -> str:
    if pretrained_model_link_or_path is None:
        raise ValueError("pretrained_model_link_or_path is required")
    if os.path.isfile(pretrained_model_link_or_path):
        return pretrained_model_link_or_path
    if os.path.isdir(pretrained_model_link_or_path):
        if filename is None:
            raise ValueError("filename is required when loading a Z-Image single-file checkpoint from a directory")
        base = os.path.join(pretrained_model_link_or_path, subfolder) if subfolder else pretrained_model_link_or_path
        return os.path.join(base, filename)

    parsed = urlparse(pretrained_model_link_or_path)
    if parsed.scheme in {"http", "https"}:
        if parsed.netloc != "huggingface.co":
            raise ValueError("Z-Image single-file loading only supports local files or Hugging Face Hub URLs")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 5 or parts[2] not in {"resolve", "blob"}:
            raise ValueError(f"Unsupported Hugging Face single-file URL: {pretrained_model_link_or_path}")
        repo_id = "/".join(parts[:2])
        url_revision = parts[3]
        url_filename = "/".join(parts[4:])
        return hf_hub_download(repo_id=repo_id, filename=url_filename, revision=revision or url_revision)

    if filename is None:
        raise ValueError("filename is required when loading a Z-Image single-file checkpoint from a Hub repo")
    relative = os.path.join(subfolder, filename) if subfolder else filename
    return hf_hub_download(repo_id=pretrained_model_link_or_path, filename=relative, revision=revision)


def _map_zimage_comfy_key_to_diffusers(key: str) -> list[str]:
    if key.endswith(".weight_scale") or key.endswith(".comfy_quant"):
        return []
    if key == "x_embedder.weight":
        return [f"all_x_embedder.{PATCH_KEY}.weight"]
    if key == "x_embedder.bias":
        return [f"all_x_embedder.{PATCH_KEY}.bias"]
    if key.startswith("final_layer."):
        return [key.replace("final_layer.", f"all_final_layer.{PATCH_KEY}.", 1)]
    if key.endswith(".attention.qkv.weight"):
        base = key.removesuffix(".attention.qkv.weight")
        return [
            f"{base}.attention.to_q.weight",
            f"{base}.attention.to_k.weight",
            f"{base}.attention.to_v.weight",
        ]
    key = key.replace(".attention.q_norm.weight", ".attention.norm_q.weight")
    key = key.replace(".attention.k_norm.weight", ".attention.norm_k.weight")
    key = key.replace(".attention.out.weight", ".attention.to_out.0.weight")
    return [key]


def _decode_comfy_quant(value: torch.Tensor) -> dict[str, Any]:
    if value.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 comfy_quant metadata, got {value.dtype}")
    raw = bytes(value.cpu().tolist()).decode("utf-8").rstrip("\x00")
    return json.loads(raw)


def _get_module(root: nn.Module, module_name: str) -> nn.Module:
    module = root
    for part in module_name.split("."):
        if isinstance(module, nn.ModuleDict):
            module = module[part]
        elif isinstance(module, nn.ModuleList):
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _set_module(root: nn.Module, module_name: str, value: nn.Module) -> None:
    parent_name, child_name = module_name.rsplit(".", 1)
    parent = _get_module(root, parent_name)
    if isinstance(parent, nn.ModuleDict):
        parent[child_name] = value
    elif isinstance(parent, nn.ModuleList):
        parent[int(child_name)] = value
    else:
        setattr(parent, child_name, value)


def _set_buffer(root: nn.Module, buffer_name: str, value: torch.Tensor) -> None:
    parent_name, child_name = buffer_name.rsplit(".", 1)
    parent = _get_module(root, parent_name)
    parent._buffers[child_name] = value


def _materialize_zimage_meta_buffers(model: nn.Module) -> None:
    for name, buffer in model.named_buffers():
        if not buffer.is_meta:
            continue
        if name == "t_embedder.flowmap_delta_emb_gate":
            _set_buffer(model, name, torch.tensor([0.25], dtype=torch.float32))
            continue
        raise RuntimeError(f"Z-Image ConvRot loader left unexpected meta buffer: {name}")


def _load_sdnq_training_symbols():
    from simpletuner.helpers.training.sdnq_compile import configure_sdnq_compile_mode

    configure_sdnq_compile_mode()
    try:
        from sdnq.dequantizer import SDNQDequantizer
        from sdnq.layers import get_sdnq_wrapper_class
        from sdnq.training.forward import get_forward_func
        from sdnq.training.tensor import SDNQTensor
    except ImportError as exc:
        raise ImportError("Z-Image ConvRot INT8 weights require SDNQ. Install it with `pip install sdnq`.") from exc
    return SDNQDequantizer, SDNQTensor, get_sdnq_wrapper_class, get_forward_func


def _validate_shape(key: str, tensor: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    if tensor.shape != expected.shape:
        raise RuntimeError(f"Z-Image ConvRot tensor {key} has shape {tuple(tensor.shape)}, expected {tuple(expected.shape)}")
    return tensor


def _wrap_convrot_linear(
    model: nn.Module,
    module_name: str,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    result_dtype: torch.dtype,
    hadamard_group_size: int,
) -> None:
    SDNQDequantizer, SDNQTensor, get_sdnq_wrapper_class, get_forward_func = _load_sdnq_training_symbols()

    module = _get_module(model, module_name)
    if not isinstance(module, nn.Linear):
        raise RuntimeError(f"Z-Image ConvRot target {module_name} is {module.__class__.__name__}, expected Linear")
    if tuple(weight.shape) != tuple(module.weight.shape):
        raise RuntimeError(
            f"Z-Image ConvRot tensor {module_name}.weight has shape {tuple(weight.shape)}, "
            f"expected {tuple(module.weight.shape)}"
        )
    if tuple(scale.shape) != (weight.shape[0], 1):
        raise RuntimeError(
            f"Z-Image ConvRot tensor {module_name}.weight_scale has shape {tuple(scale.shape)}, "
            f"expected {(weight.shape[0], 1)}"
        )

    dequantizer = SDNQDequantizer(
        result_dtype=result_dtype,
        result_shape=None,
        original_shape=weight.shape,
        original_stride=list(weight.stride()),
        quantized_weight_shape=weight.shape,
        weights_dtype="int8",
        quantized_matmul_dtype="int8",
        hadamard_group_size=hadamard_group_size,
        group_size=-1,
        svd_rank=32,
        svd_steps=8,
        use_quantized_matmul=False,
        re_quantize_for_matmul=False,
        use_stochastic_rounding=False,
        use_hadamard=True,
        layer_class_name="Linear",
    )
    module.weight = nn.Parameter(
        SDNQTensor(
            weight.contiguous(),
            scale.to(torch.float32).contiguous(),
            None,
            None,
            None,
            dequantizer,
        ),
        requires_grad=module.weight.requires_grad,
    )
    module.weight._is_hf_initialized = True

    forward = get_forward_func(
        "int8",
        "int8",
        True,
        True,
        True,
        -1,
    )
    _set_module(model, module_name, get_sdnq_wrapper_class(module, forward))


def _validate_quant_metadata(checkpoint, key: str) -> int:
    quant_key = f"{key.removesuffix('.weight')}.comfy_quant"
    if quant_key not in checkpoint.keys():
        raise RuntimeError(f"Z-Image ConvRot tensor {key} is missing comfy_quant metadata")
    quant_metadata = _decode_comfy_quant(checkpoint.get_tensor(quant_key))
    if not quant_metadata.get("convrot", False) or not quant_metadata.get("per_row", False):
        raise RuntimeError(f"Z-Image INT8 tensor {key} is not marked as per-row ConvRot")
    hadamard_group_size = int(quant_metadata.get("convrot_groupsize", 0))
    if hadamard_group_size <= 0:
        raise RuntimeError(f"Z-Image ConvRot tensor {key} has invalid convrot_groupsize")
    return hadamard_group_size


def load_zimage_comfy_convrot_checkpoint(
    model_cls,
    pretrained_model_link_or_path: str,
    *,
    filename: str | None = None,
    subfolder: str | None = None,
    revision: str | None = None,
    torch_dtype: torch.dtype | None = None,
) -> nn.Module:
    checkpoint_path = _resolve_zimage_single_file_path(
        pretrained_model_link_or_path,
        filename=filename,
        subfolder=subfolder,
        revision=revision,
    )
    result_dtype = torch_dtype or torch.bfloat16

    non_quantized_state_dict: dict[str, torch.Tensor] = {}
    quantized_weights: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}

    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        ordered_checkpoint_keys = checkpoint.keys()
        checkpoint_keys = set(ordered_checkpoint_keys)
        for key in ordered_checkpoint_keys:
            if key.endswith(".weight_scale") or key.endswith(".comfy_quant"):
                continue

            diffusers_keys = _map_zimage_comfy_key_to_diffusers(key)
            tensor = checkpoint.get_tensor(key)
            if tensor.dtype == torch.int8:
                scale_key = f"{key}_scale"
                if scale_key not in checkpoint_keys:
                    raise RuntimeError(f"Z-Image ConvRot tensor {key} is missing weight_scale")
                hadamard_group_size = _validate_quant_metadata(checkpoint, key)
                scale = checkpoint.get_tensor(scale_key)
                if key.endswith(".attention.qkv.weight"):
                    if len(diffusers_keys) != 3:
                        raise RuntimeError(f"Z-Image ConvRot tensor {key} did not map to q/k/v tensors")
                    if tensor.shape[0] % 3 != 0 or scale.shape[0] % 3 != 0:
                        raise RuntimeError(f"Z-Image ConvRot tensor {key} cannot be split into q/k/v tensors")
                    qkv_tensors = tensor.split(tensor.shape[0] // 3, dim=0)
                    qkv_scales = scale.split(scale.shape[0] // 3, dim=0)
                    for diffusers_key, qkv_tensor, qkv_scale in zip(diffusers_keys, qkv_tensors, qkv_scales):
                        quantized_weights[diffusers_key] = (
                            qkv_tensor.contiguous(),
                            qkv_scale.contiguous(),
                            hadamard_group_size,
                        )
                else:
                    if len(diffusers_keys) != 1:
                        raise RuntimeError(f"Z-Image ConvRot tensor {key} maps to multiple targets unexpectedly")
                    quantized_weights[diffusers_keys[0]] = (tensor, scale, hadamard_group_size)
            else:
                if len(diffusers_keys) != 1:
                    raise RuntimeError(f"Z-Image non-quantized tensor {key} maps to multiple targets unexpectedly")
                non_quantized_state_dict[diffusers_keys[0]] = tensor

    with torch.device("meta"):
        model = model_cls()
    expected_state_dict = model.state_dict()
    for key, tensor in list(non_quantized_state_dict.items()):
        if key not in expected_state_dict:
            raise RuntimeError(f"Z-Image ConvRot checkpoint has unexpected tensor: {key}")
        tensor = _validate_shape(key, tensor, expected_state_dict[key])
        if torch.is_floating_point(tensor) and torch_dtype is not None:
            tensor = tensor.to(torch_dtype)
        non_quantized_state_dict[key] = tensor

    expected_quantized_keys = set(quantized_weights)
    missing, unexpected = model.load_state_dict(non_quantized_state_dict, strict=False, assign=True)
    unexpected = list(unexpected)
    real_missing = [key for key in missing if key not in expected_quantized_keys]
    if real_missing or unexpected:
        raise RuntimeError(
            "Z-Image ConvRot checkpoint does not match transformer architecture. "
            f"Missing: {len(real_missing)}, Unexpected: {len(unexpected)}"
        )

    hadamard_group_sizes: set[int] = set()
    for weight_key, (weight, scale, hadamard_group_size) in quantized_weights.items():
        if weight_key not in expected_state_dict:
            raise RuntimeError(f"Z-Image ConvRot checkpoint has unexpected tensor: {weight_key}")
        _validate_shape(weight_key, weight, expected_state_dict[weight_key])
        hadamard_group_sizes.add(hadamard_group_size)
        _wrap_convrot_linear(
            model,
            weight_key.removesuffix(".weight"),
            weight,
            scale,
            result_dtype=result_dtype,
            hadamard_group_size=hadamard_group_size,
        )

    if len(hadamard_group_sizes) != 1:
        raise RuntimeError(f"Z-Image ConvRot checkpoint uses multiple Hadamard group sizes: {sorted(hadamard_group_sizes)}")
    _materialize_zimage_meta_buffers(model)
    model.quantization_method = "zimage_comfy_convrot_sdnq"
    quantization_config = {
        "quant_method": "sdnq_training",
        "weights_dtype": "int8",
        "quantized_matmul_dtype": "int8",
        "use_hadamard": True,
        "hadamard_group_size": hadamard_group_sizes.pop(),
        "group_size": -1,
        "source_format": "comfy_zimage_convrot",
    }
    model.quantization_config = quantization_config
    return model
