# Guía rápida de continuación AnyFlow

Esta guía muestra cómo continuar el objetivo de entrenamiento AnyFlow en un dataset Wan downstream. Para la explicación general, consulta [AnyFlow](/documentation/experimental/ANYFLOW.es.md).

Los checkpoints públicos de NVIDIA AnyFlow son pipelines Diffusers completos con pesos completos del transformer, no adaptadores LoRA. No apuntes `init_lora` a esos repositorios. Usa `init_lora` solo cuando tengas un archivo o repositorio LoRA compatible con SimpleTuner.

## Qué checkpoint usar

Usa los checkpoints AnyFlow T2V bidireccionales como transformer preentrenado:

- `nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers`
- `nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers`

Mantén el checkpoint Wan original para text encoder, tokenizer, VAE y scheduler:

- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- `Wan-AI/Wan2.1-T2V-14B-Diffusers`

Los checkpoints FAR (`nvidia/AnyFlow-FAR-*`) usan una arquitectura causal AnyFlow y no son el objetivo de esta guía de SimpleTuner.

## Config de ejemplo

Empieza con la configuración normal de Wan y cambia los campos de modelo y destilación:

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

Ejecuta el entrenamiento desde el directorio de SimpleTuner:

```bash
simpletuner train
```

El LoRA resultante continúa desde el transformer AnyFlow destilado y mantiene activo el objetivo AnyFlow durante el fine-tuning.

## Si tienes un LoRA AnyFlow

Si se publica por separado un LoRA AnyFlow extraído, usa el checkpoint Wan original y carga el adaptador con `init_lora`:

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

El rank y los módulos target deben coincidir con el adaptador publicado. Un checkpoint transformer completo no es un valor válido para `init_lora`.

## Sobre extraer un LoRA

Extraer un LoRA desde un transformer AnyFlow completo es posible en principio, pero es un proyecto de conversión. SimpleTuner incluye scripts experimentales:

```bash
python scripts/extract_peft_lora.py \
  Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers \
  output/anyflow-wan21-1.3b-r32.safetensors \
  --rank 32
```

Para LyCORIS/LoCon, usa `scripts/extract_lycoris_adapter.py` con los mismos argumentos y `--algo locon`.

La conversión carga el transformer Wan base y el AnyFlow, calcula deltas de pesos compatibles, factoriza esos deltas en matrices LoRA de bajo rango, guarda un adaptador compatible y valida el resultado.

Es una aproximación dependiente del rank. El target por defecto coincide con los defaults PEFT de Wan en SimpleTuner (`to_q,to_k,to_v,to_out.0`). Usa `--target-modules all-linear` solo si la configuración downstream apunta a los mismos módulos.

## Límites actuales

- La licencia pública de los modelos NVIDIA AnyFlow es no comercial; revisa la model card antes de publicar adaptadores derivados.
- La validación estándar puede ejecutarse, pero la validación AnyFlow de pocos pasos aún necesita soporte de sampler o pipeline que pase `r_timestep`.
- La continuación full-rank con online teacher aún necesita cableado separado de student y teacher. Por ahora, LoRA es la ruta soportada.
