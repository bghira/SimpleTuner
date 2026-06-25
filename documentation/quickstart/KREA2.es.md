# Guía rápida de Krea2

Esta guía cubre el entrenamiento LoRA de Krea2 en SimpleTuner. Krea2 es un transformer de imágenes grande con flow matching, acondicionamiento de texto estilo Qwen y el VAE de Qwen Image. Funciona mejor en GPUs NVIDIA con mucha memoria.

El ejemplo inicial está en:

```bash
simpletuner/examples/krea2.peft-lora/config.json
```

## Punto de partida recomendado

Para la primera ejecución, usa la configuración de ejemplo y mantén el modelo conservador:

```json
{
  "model_family": "krea2",
  "model_flavour": "raw",
  "model_type": "lora",
  "pretrained_model_name_or_path": "krea/Krea-2-Raw",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "fuse_qkv_projections": true,
  "train_batch_size": 1,
  "base_model_precision": "no_change"
}
```

Krea2 es nativo de 1024px, pero 512px y 768px son útiles para iterar rápido y revisar datasets. Usa un dataloader de 1024px cuando la ejecución ya sea estable.

## Notas de hardware

Krea2 puede entrenar en bf16 en una H100 de 80GB con batch 1 a 1024px. En nuestras pruebas, batches más grandes caben sin compile, pero compile agrega suficiente memoria de grafo/cudagraph como para provocar OOM en muchas configuraciones grandes.

TorchAO int8 weight-only reduce mucho la VRAM, pero no fue más rápido que bf16 en la ruta de entrenamiento probada. Úsalo cuando la capacidad de memoria sea más importante que el tiempo por paso.

Recomendaciones:

- Usa `bf16` cuando el modelo quepa.
- Usa `int8-torchao` cuando necesites margen de memoria.
- Mantén `gradient_checkpointing=true`.
- Mantén `fuse_qkv_projections=true`.
- Usa `dynamo_backend=inductor`, `dynamo_mode=reduce-overhead` y `dynamo_use_regional_compilation=true` solo después de confirmar que el batch/resolución cabe.

## Valores de configuración clave

```json
{
  "model_family": "krea2",
  "model_flavour": "raw",
  "model_type": "lora",
  "pretrained_model_name_or_path": "krea/Krea-2-Raw",
  "base_model_precision": "no_change",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "fuse_qkv_projections": true,
  "optimizer": "optimi-lion",
  "learning_rate": 1e-4,
  "lora_rank": 64,
  "train_batch_size": 1,
  "resolution": 1024,
  "validation_resolution": "1024x1024"
}
```

Para TorchAO int8:

```json
{
  "base_model_precision": "int8-torchao",
  "quantize_via": "cpu"
}
```

Para compile reduce-overhead:

```json
{
  "dynamo_backend": "inductor",
  "dynamo_mode": "reduce-overhead",
  "dynamo_use_regional_compilation": true
}
```

## Entrenamiento con imagen de referencia

Krea2 soporta acondicionamiento opcional con latentes de referencia para datasets de edición. Actívalo cuando tu dataloader proporcione imágenes de referencia emparejadas o latentes de referencia en caché:

```json
{
  "krea2_reference_latents": true
}
```

Los latentes de referencia deben coincidir con la forma de los latentes objetivo.

## Configuración del dataloader

Krea2 usa la estructura general de dataloader de imagen de otros modelos transformer. La resolución real de entrenamiento viene del JSON del dataloader, no solo de `resolution` en el config principal. Para entrenar a 1024px, asegúrate de que `resolution`, `maximum_image_size` y `target_downsample_size` también sean 1024 en el dataloader.

Un dataset de 512px es útil para pruebas rápidas, revisar captions y detectar crops rotos. El run final suele necesitar 1024px para dar una señal real de calidad.

Para datasets locales, usa `type: local`, define `instance_data_dir`, y elige una estrategia de caption. Para un sujeto pequeño, `caption_strategy=instanceprompt` suele ser suficiente al inicio. Para estilos, filenames o captions completos suelen funcionar mejor.

## Validación

La validación de Krea2 es costosa, así que empieza con pocos prompts. Un prompt único puede ocultar overfitting o memorización; cuando el run sea estable, añade una pequeña prompt library.

Ejemplo:

```json
{
  "validation_prompt": "a studio portrait of <token>, soft directional light, detailed fabric texture",
  "validation_negative_prompt": "ugly, cropped, blurry, low-quality, mediocre average",
  "validation_num_inference_steps": 28,
  "validation_guidance": 4.5,
  "validation_resolution": "1024x1024"
}
```

## Notas de cuantización

`int8-torchao` almacena los pesos base del transformer en int8 y entrena pesos LoRA bf16 encima. En H100 redujo mucho la VRAM, pero fue más lento que bf16 en esta ruta de entrenamiento. Es una opción de capacidad, no una garantía de throughput.

## Resultados de benchmark

Estas mediciones se tomaron en una NVIDIA H100 de 80GB usando el trainer real de SimpleTuner, Krea2 LoRA, QKV fusionado, gradient checkpointing y un dataset pequeño de Domokun. La VRAM se muestreó externamente con `nvidia-smi`. Tómalas solo como guía comparativa; versiones distintas de PyTorch, CUDA, drivers, datasets, rank LoRA, optimizadores, backends de atención y GPUs pueden cambiar los resultados.

### QKV fusionado + checkpointing, compile desactivado

| Precisión | Resolución | Batch | s/paso estable | Pico VRAM |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 512 | 1 | 0.353 | 31.10 GiB |
| bf16 | 512 | 4 | 1.230 | 39.31 GiB |
| bf16 | 512 | 8 | 2.430 | 50.32 GiB |
| bf16 | 1024 | 1 | 0.990 | 33.28 GiB |
| bf16 | 1024 | 4 | 3.850 | 48.35 GiB |
| bf16 | 1024 | 8 | 7.690 | 67.88 GiB |
| int8-torchao | 512 | 1 | 0.535 | 18.10 GiB |
| int8-torchao | 512 | 4 | 1.690 | 27.46 GiB |
| int8-torchao | 512 | 8 | 3.220 | 40.52 GiB |
| int8-torchao | 1024 | 1 | 1.330 | 20.35 GiB |
| int8-torchao | 1024 | 4 | 4.850 | 36.99 GiB |
| int8-torchao | 1024 | 8 | 9.520 | 58.84 GiB |

### QKV fusionado + checkpointing + compile reduce-overhead

| Precisión | Resolución | Batch | Estado | s/paso estable | Pico VRAM |
| --- | ---: | ---: | --- | ---: | ---: |
| bf16 | 512 | 1 | ok | 0.260 | 41.20 GiB |
| bf16 | 512 | 4 | OOM | - | 79.07 GiB |
| bf16 | 512 | 8 | OOM | - | 79.10 GiB |
| bf16 | 1024 | 1 | ok | 0.704 | 63.71 GiB |
| bf16 | 1024 | 4 | OOM | - | 79.11 GiB |
| bf16 | 1024 | 8 | OOM | - | 78.40 GiB |
| int8-torchao | 512 | 1 | ok | 0.410 | 30.93 GiB |
| int8-torchao | 512 | 4 | ok | 1.300 | 78.60 GiB |
| int8-torchao | 512 | 8 | OOM | - | 79.12 GiB |
| int8-torchao | 1024 | 1 | ok | 0.990 | 58.68 GiB |
| int8-torchao | 1024 | 4 | OOM | - | 78.92 GiB |
| int8-torchao | 1024 | 8 | OOM | - | 78.09 GiB |

## Guía práctica

- Para iterar más rápido en una H100, usa bf16, QKV fusionado, checkpointing, compile activado y batch 1.
- Para batches efectivos más grandes, usa bf16 sin compile y sube `train_batch_size` hasta que la VRAM sea el límite.
- Para ejecuciones con poca memoria, usa `int8-torchao`; espera menos VRAM pero pasos más lentos.
- Compile ayuda en batch 1, pero puede consumir suficiente VRAM para hacer fallar batches mayores.

## Problemas comunes

- Si esperabas 1024px pero el log muestra 512px, revisa el dataloader JSON.
- Si compile OOM pero el run sin compile entra, baja batch size o desactiva compile.
- Si int8 usa menos VRAM pero es más lento, coincide con nuestras pruebas H100.
- Si la imagen de referencia no influye en validación, confirma que `krea2_reference_latents=true` y que el dataset de validación usa pares de referencia.
- Si overfitea rápido, baja learning rate, reduce pasos o amplía la variedad del dataset.
