# AnyFlow Continuation Quickstart

This guide is for continuing the AnyFlow training objective on a downstream Wan dataset. For the implementation overview, see [AnyFlow](/documentation/experimental/ANYFLOW.md).

The public NVIDIA AnyFlow checkpoints are full Diffusers pipelines with full transformer weights, not LoRA adapters. Do not point `init_lora` at those repositories. Use `init_lora` only when you have an actual SimpleTuner-compatible LoRA file or repository.

## Which Checkpoint To Use

Use the bidirectional T2V AnyFlow checkpoints as the pretrained transformer:

- `nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers`
- `nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers`

Keep the original Wan checkpoint as the source for the text encoder, tokenizer, VAE, and scheduler:

- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- `Wan-AI/Wan2.1-T2V-14B-Diffusers`

The FAR checkpoints (`nvidia/AnyFlow-FAR-*`) use a causal AnyFlow transformer architecture and are not the target of this SimpleTuner quickstart.

## Example Config

Start from the normal Wan quickstart config, then change the model and distillation fields:

```json
{
  "model_family": "wan",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_vae_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_transformer_model_name_or_path": "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_transformer_subfolder": "transformer",
  "data_backend_config": "config/wan/multidatabackend.json",
  "output_dir": "output/wan-anyflow-lora",
  "lora_rank": 32,
  "lora_alpha": 32,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "learning_rate": 0.0001,
  "max_train_steps": 1000,
  "distillation_method": "anyflow",
  "distillation_config": {
    "anyflow": {
      "target_mode": "online_teacher",
      "teacher_rollout_steps": 1,
      "r_timestep_sampler": "uniform",
      "min_interval_ratio": 0.02,
      "gate_value": 0.25,
      "deltatime_type": "r",
      "loss_weight": 1.0
    }
  }
}
```

Run training from the SimpleTuner directory:

```bash
simpletuner train
```

The resulting LoRA continues from the AnyFlow distilled transformer while keeping the AnyFlow objective active during downstream fine-tuning.

## If You Have An AnyFlow LoRA

If an AnyFlow LoRA has been extracted and published separately, then use the original Wan base checkpoint and load the adapter with `init_lora`:

```json
{
  "model_family": "wan",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_vae_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "init_lora": "your-org/anyflow-wan21-1.3b-lora",
  "lora_rank": 32,
  "lora_alpha": 32,
  "distillation_method": "anyflow"
}
```

The LoRA rank and target modules must match the published adapter. A full transformer checkpoint is not a valid `init_lora` value.

## About Extracting A LoRA

Extracting a LoRA from a full AnyFlow transformer is possible in principle, but it is a conversion project rather than a training option. SimpleTuner includes experimental extraction scripts:

```bash
python scripts/extract_peft_lora.py \
  Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers \
  output/anyflow-wan21-1.3b-r32.safetensors \
  --rank 32
```

For a LyCORIS/LoCon adapter, use `scripts/extract_lycoris_adapter.py` with the same arguments plus `--algo locon`.

The conversion does this:

1. Load the matching Wan base transformer and the AnyFlow transformer.
2. Diff matching linear-layer weights.
3. Factorize each delta into low-rank LoRA matrices.
4. Save a PEFT-compatible adapter with the exact target module list and rank.
5. Validate that generation and downstream continuation still behave like the full AnyFlow checkpoint.

This is approximate and rank-dependent. The default script target set matches SimpleTuner's Wan PEFT defaults (`to_q,to_k,to_v,to_out.0`). Use `--target-modules all-linear` only if the downstream config also targets the same modules.

## Current Limits

- The public NVIDIA AnyFlow model license is noncommercial; check the upstream model card before publishing derived adapters.
- AnyFlow validation is wired through the distiller scheduler hook for registered FlowMap-capable pipelines. Custom or external validation paths still need to pass `r_timestep` or `timestep_r` into the model component.
- Full-rank online-teacher continuation still needs separate student and teacher wiring. LoRA continuation is the supported path for now.
