# AnyFlow continuation quickstart

यह guide downstream Wan dataset पर AnyFlow training objective जारी रखने के लिए है। implementation overview के लिए [AnyFlow](/documentation/experimental/ANYFLOW.hi.md) देखें।

NVIDIA AnyFlow के public checkpoints full Diffusers pipelines हैं जिनमें full transformer weights हैं, LoRA adapters नहीं। उन repositories को `init_lora` में न दें। `init_lora` सिर्फ तब इस्तेमाल करें जब आपके पास SimpleTuner-compatible LoRA file या repository हो।

## कौन सा checkpoint इस्तेमाल करें

pretrained transformer के लिए bidirectional T2V AnyFlow checkpoints इस्तेमाल करें:

- `nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers`
- `nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers`

text encoder, tokenizer, VAE, और scheduler के लिए original Wan checkpoint रखें:

- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- `Wan-AI/Wan2.1-T2V-14B-Diffusers`

FAR checkpoints (`nvidia/AnyFlow-FAR-*`) causal AnyFlow transformer architecture इस्तेमाल करते हैं और इस SimpleTuner quickstart का target नहीं हैं।

## Example config

normal Wan quickstart config से शुरू करें और model तथा distillation fields बदलें:

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

SimpleTuner directory से training चलाएं:

```bash
simpletuner train
```

resulting LoRA distilled AnyFlow transformer से continue करता है और downstream fine-tuning के दौरान AnyFlow objective active रखता है।

## अगर आपके पास AnyFlow LoRA है

अगर extracted AnyFlow LoRA अलग से published है, तो original Wan base checkpoint इस्तेमाल करें और adapter को `init_lora` से load करें:

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

LoRA rank और target modules published adapter से match होने चाहिए। full transformer checkpoint `init_lora` के लिए valid नहीं है।

## LoRA extraction के बारे में

full AnyFlow transformer से LoRA extract करना theoretically possible है, लेकिन यह conversion project है। SimpleTuner में experimental scripts हैं:

```bash
python scripts/extract_peft_lora.py \
  Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers \
  output/anyflow-wan21-1.3b-r32.safetensors \
  --rank 32
```

LyCORIS/LoCon के लिए same arguments और `--algo locon` के साथ `scripts/extract_lycoris_adapter.py` इस्तेमाल करें।

conversion matching Wan base transformer और AnyFlow transformer load करता है, matching linear weights के deltas निकालता है, deltas को low-rank LoRA matrices में factorize करता है, compatible adapter save करता है, और result validate करता है।

यह approximate और rank-dependent है। default target SimpleTuner के Wan PEFT defaults (`to_q,to_k,to_v,to_out.0`) से match करता है। `--target-modules all-linear` तभी इस्तेमाल करें जब downstream config भी वही modules target करे।

## मौजूदा limits

- public NVIDIA AnyFlow model license noncommercial है; derived adapters publish करने से पहले upstream model card देखें।
- AnyFlow validation registered FlowMap-capable pipelines के लिए distiller scheduler hook से wired है। custom या external validation paths को model component में `r_timestep` या `timestep_r` pass करना होगा।
- full-rank online-teacher continuation के लिए अलग student और teacher wiring चाहिए। अभी supported path LoRA continuation है।
