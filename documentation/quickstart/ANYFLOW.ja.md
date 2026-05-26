# AnyFlow 継続学習クイックスタート

このガイドは downstream Wan dataset で AnyFlow training objective を継続するためのものです。実装概要は [AnyFlow](/documentation/experimental/ANYFLOW.ja.md) を参照してください。

NVIDIA AnyFlow の公開 checkpoints は full transformer weights を含む Diffusers pipeline であり、LoRA adapter ではありません。これらの repository を `init_lora` に指定しないでください。`init_lora` は SimpleTuner-compatible な LoRA file または repository がある場合だけ使います。

## 使う checkpoint

pretrained transformer には bidirectional T2V AnyFlow checkpoint を使います:

- `nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers`
- `nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers`

text encoder、tokenizer、VAE、scheduler には original Wan checkpoint を使います:

- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- `Wan-AI/Wan2.1-T2V-14B-Diffusers`

FAR checkpoints (`nvidia/AnyFlow-FAR-*`) は causal AnyFlow transformer architecture を使うため、この SimpleTuner quickstart の対象ではありません。

## 設定例

通常の Wan quickstart config から始め、model と distillation の fields を変更します:

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

SimpleTuner directory から training を実行します:

```bash
simpletuner train
```

出力される LoRA は distilled AnyFlow transformer から継続し、downstream fine-tuning 中も AnyFlow objective を維持します。

## AnyFlow LoRA がある場合

extracted AnyFlow LoRA が別途公開されている場合は、original Wan base checkpoint を使い、adapter を `init_lora` で読み込みます:

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

LoRA rank と target modules は published adapter と一致している必要があります。full transformer checkpoint は `init_lora` の有効な値ではありません。

## LoRA extraction について

full AnyFlow transformer から LoRA を抽出することは原理的には可能ですが、training option ではなく conversion project です。SimpleTuner には experimental scripts があります:

```bash
python scripts/extract_peft_lora.py \
  Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers \
  output/anyflow-wan21-1.3b-r32.safetensors \
  --rank 32
```

LyCORIS/LoCon の場合は、同じ arguments と `--algo locon` で `scripts/extract_lycoris_adapter.py` を使います。

conversion は matching Wan base transformer と AnyFlow transformer を読み込み、matching linear-layer weights の delta を計算し、low-rank LoRA matrices に factorize し、compatible adapter として保存して検証します。

これは approximate で rank-dependent です。default target は SimpleTuner の Wan PEFT defaults (`to_q,to_k,to_v,to_out.0`) と一致します。`--target-modules all-linear` は downstream config も同じ modules を target する場合だけ使ってください。

## 現在の制限

- public NVIDIA AnyFlow model license は noncommercial です。derived adapters を公開する前に upstream model card を確認してください。
- standard validation は実行できますが、AnyFlow-style few-step validation には `r_timestep` を渡す sampler または pipeline support がまだ必要です。
- full-rank online-teacher continuation には separate student/teacher wiring が必要です。現時点の supported path は LoRA continuation です。
