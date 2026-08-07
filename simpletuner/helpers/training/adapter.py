import math

import peft
import safetensors.torch
import torch

ANYFLOW_SIDECAR_PREFIXES = ("condition_embedder.delta_embedder.",)


def _alpha_value(value):
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def _set_lora_alpha(layer, adapter_name: str, alpha) -> None:
    alpha = _alpha_value(alpha)
    layer.lora_alpha[adapter_name] = alpha
    rank_map = getattr(layer, "r", {})
    rank = rank_map.get(adapter_name) if isinstance(rank_map, dict) else rank_map
    if not rank:
        return
    scaling = alpha / math.sqrt(rank) if getattr(layer, "use_rslora", False) else alpha / rank
    layer.scaling[adapter_name] = scaling


def determine_adapter_target_modules(args, unet, transformer):
    if unet is not None:
        return ["to_k", "to_q", "to_v", "to_out.0"]
    elif transformer is not None:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

        if args.model_family.lower() == "flux" and args.flux_lora_target == "all":
            # target_modules = mmdit layers here
            target_modules = [
                "to_k",
                "to_q",
                "to_v",
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "to_out.0",
                "to_add_out",
            ]
        elif args.flux_lora_target == "context":
            # i think these are the text input layers.
            target_modules = [
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "to_add_out",
            ]
        elif args.flux_lora_target == "context+ffs":
            # i think these are the text input layers.
            target_modules = [
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "to_add_out",
                "ff_context.net.0.proj",
                "ff_context.net.2",
            ]
        elif args.flux_lora_target == "all+ffs":
            target_modules = [
                "to_k",
                "to_q",
                "to_v",
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "to_out.0",
                "to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
                "proj_mlp",
                "proj_out",
            ]
        elif args.flux_lora_target == "ai-toolkit":
            # from ostris' ai-toolkit, possibly required to continue finetuning one.
            target_modules = [
                "to_q",
                "to_k",
                "to_v",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_out.0",
                "to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
                "norm.linear",
                "norm1.linear",
                "norm1_context.linear",
                "proj_mlp",
                "proj_out",
            ]
        elif args.flux_lora_target == "tiny":
            # From TheLastBen
            # https://www.reddit.com/r/StableDiffusion/comments/1f523bd/good_flux_loras_can_be_less_than_45mb_128_dim/
            target_modules = [
                "single_transformer_blocks.7.proj_out",
                "single_transformer_blocks.20.proj_out",
            ]
        elif args.flux_lora_target == "nano":
            # From TheLastBen
            # https://www.reddit.com/r/StableDiffusion/comments/1f523bd/good_flux_loras_can_be_less_than_45mb_128_dim/
            target_modules = [
                "single_transformer_blocks.7.proj_out",
            ]

        return target_modules


@torch.no_grad()
def load_lora_weights(dictionary, filename, loraKey="default", use_dora=False, state_dict=None):
    additional_keys = set()
    if state_dict is None:
        state_dict = safetensors.torch.load_file(filename)
    for prefix, model in dictionary.items():
        lora_layers = {
            (prefix + "." + x): y for (x, y) in model.named_modules() if isinstance(y, peft.tuners.lora.layer.Linear)
        }
        model_state = model.state_dict()
        sidecar_prefix = f"{prefix}."
        for key, tensor in state_dict.items():
            if not key.startswith(sidecar_prefix):
                continue
            model_key = key.removeprefix(sidecar_prefix)
            if not model_key.startswith(ANYFLOW_SIDECAR_PREFIXES):
                continue
            try:
                destination = model_state[model_key]
            except KeyError as exc:
                raise ValueError(
                    f"LoRA file contains AnyFlow sidecar tensor `{key}`, but `{prefix}` does not have `{model_key}`. "
                    "Call enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type='r') before loading this LoRA."
                ) from exc
            if destination.shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch for AnyFlow sidecar tensor `{key}`: "
                    f"model {tuple(destination.shape)} vs file {tuple(tensor.shape)}."
                )
            destination.copy_(tensor.to(device=destination.device, dtype=destination.dtype))
    missing_keys = set(
        [x + ".lora_A.weight" for x in lora_layers.keys()]
        + [x + ".lora_B.weight" for x in lora_layers.keys()]
        + ([x + ".lora_magnitude_vector.weight" for x in lora_layers.keys()] if use_dora else [])
    )
    loaded_ranks = {}
    explicit_alpha_keys = False
    for k, v in state_dict.items():
        if "lora_A" in k:
            kk = k.replace(".lora_A.weight", "")
            if kk in lora_layers:
                lora_layers[kk].lora_A[loraKey].weight.copy_(v)
                rank = int(v.shape[0])
                existing_rank = loaded_ranks.get(kk)
                if existing_rank is not None and existing_rank != rank:
                    raise ValueError(f"LoRA checkpoint has conflicting ranks for `{kk}`: {existing_rank} and {rank}.")
                loaded_ranks[kk] = rank
                missing_keys.remove(k)
            else:
                additional_keys.add(k)
        elif "lora_B" in k:
            kk = k.replace(".lora_B.weight", "")
            if kk in lora_layers:
                lora_layers[kk].lora_B[loraKey].weight.copy_(v)
                rank = int(v.shape[1])
                existing_rank = loaded_ranks.get(kk)
                if existing_rank is not None and existing_rank != rank:
                    raise ValueError(f"LoRA checkpoint has conflicting ranks for `{kk}`: {existing_rank} and {rank}.")
                loaded_ranks[kk] = rank
                missing_keys.remove(k)
            else:
                additional_keys.add(k)
        elif ".alpha" in k or ".lora_alpha" in k:
            kk = k.replace(".lora_alpha", "").replace(".alpha", "")
            if kk in lora_layers:
                explicit_alpha_keys = True
                _set_lora_alpha(lora_layers[kk], loraKey, v)
        elif ".lora_magnitude_vector" in k:
            kk = k.replace(".lora_magnitude_vector.weight", "")
            if kk in lora_layers:
                lora_layers[kk].lora_magnitude_vector[loraKey].weight.copy_(v)
                missing_keys.remove(k)
            else:
                additional_keys.add(k)
    if not explicit_alpha_keys and len(set(loaded_ranks.values())) > 1:
        for kk, rank in loaded_ranks.items():
            if kk in lora_layers:
                _set_lora_alpha(lora_layers[kk], loraKey, rank)
    return (additional_keys, missing_keys)
