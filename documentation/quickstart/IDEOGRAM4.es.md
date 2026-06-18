# Ideogram 4 Quickstart

Esta guía cubre el entrenamiento LoRA de Ideogram 4 en SimpleTuner. Ideogram 4 es un modelo de imagen flow-matching de unos 9B parámetros, fuerte en tipografía, composición y prompts complejos. El checkpoint público se distribuye en FP8; SimpleTuner usa esa versión FP8 por defecto.

Configuración inicial:

```bash
simpletuner/examples/ideogram-fp8.peft-lora/config.json
```

## Hardware y precisión

Puntos de partida recomendados:

- **Predeterminado:** pesos base FP8, pesos LoRA entrenables en bf16, rank 16-32.
- **Baja VRAM:** NF4 para el modelo base.
- **Alta VRAM:** pesos bf16-upcast si tienes suficiente VRAM y quieres evitar la carga cuantizada.

Medido en H100 80GB con FP8 nativo (`base_model_precision=fp8-torchao`, `quantize_via=pipeline`), LoRA rank 32, mixed precision bf16, gradient checkpointing activado, entrenamiento square 1024px y validación desactivada:

| Batch size | Pico de VRAM |
| --- | ---: |
| 1 | 15,999 MiB / 15.6 GiB |
| 2 | 20,095 MiB / 19.6 GiB |
| 4 | 28,603 MiB / 27.9 GiB |

La validación tiene un pico de generación separado, así que deja margen adicional con `ideogram_validation=true`. En GPUs más pequeñas, empieza con FP8 o NF4, rank 8-16, gradient checkpointing y offload. Apple Silicon (MPS) es compatible con el entrenamiento de Ideogram 4: el checkpoint FP8 se descuantiza a bf16 al cargar. Para reducir memoria, usa `base_model_precision=int8-sdnq` junto con `quantize_via=cpu` (FP8/NF4 son solo para CUDA).

### Torch compile

Para `torch.compile`, prefiere regional compilation con pesos FP8 nativos:

```json
{
  "dynamo_backend": "inductor",
  "dynamo_use_regional_compilation": true
}
```

`dynamo_backend="inductor"` puro también funciona, pero el compile del primer step del modelo completo es lento. Por ahora, evita `dynamo_mode="reduce-overhead"` y `dynamo_fullgraph=true` para Ideogram 4 LoRA; las capas PEFT LoRA pueden activar CUDA graph output reuse en la segunda invocación compilada.

## Configuración

Copia la configuración y el dataloader de ejemplo:

```bash
mkdir -p config/examples
cp simpletuner/examples/ideogram-fp8.peft-lora/config.json config/config.json
cp simpletuner/examples/multidatabackend-ideogram-dreambooth-1024px.json config/examples/multidatabackend-ideogram-dreambooth-1024px.json
```

Campos importantes:

```json
{
  "model_type": "lora",
  "model_family": "ideogram",
  "model_flavour": "fp8",
  "base_model_precision": "no_change",
  "quantize_via": "cpu",
  "mixed_precision": "bf16",
  "train_batch_size": 1,
  "resolution": 1024,
  "resolution_type": "pixel_area",
  "gradient_checkpointing": true,
  "ideogram_auto_json": true,
  "ideogram_validation": true,
  "ideogram_schedule_mu": 0.0,
  "ideogram_schedule_std": 1.5
}
```

FP8 es la primera recomendación:

```json
{
  "model_flavour": "fp8",
  "base_model_precision": "no_change",
  "quantize_via": "cpu"
}
```

Para poca VRAM, usa NF4:

```json
{
  "base_model_precision": "nf4-bnb",
  "base_model_default_dtype": "bf16",
  "quantize_via": "cpu"
}
```

En Apple Silicon (MPS), usa SDNQ int8 en su lugar:

```json
{
  "base_model_precision": "int8-sdnq",
  "quantize_via": "cpu"
}
```

## Cache de text embeds

La salida del text encoder de Ideogram 4 concatena 13 capas hidden-state de Qwen. Por defecto, SimpleTuner proyecta esas features raw con las capas congeladas `llm_cond_norm` y `llm_cond_proj` del transformer antes de escribir los archivos de cache de text embeds. Esto reduce mucho el tamaño del cache y conserva el tensor de conditioning que consume el transformer.

Las capas de proyección están congeladas tanto en LoRA como en entrenamiento full del transformer. Para entrenamiento del text encoder, LoRA no estándar, o targets LoRA que incluyan explícitamente `llm_cond_norm` o `llm_cond_proj`, SimpleTuner mantiene la salida raw del text encoder en el cache.

El coste grande del cache viene del ancho de las features, no de padding guardado. El precompute de text embeds escribe un archivo por prompt con la longitud real de tokens de ese prompt; el padding de batch ocurre después en memoria. El tensor raw de 13 capas tiene `13 * 4096 = 53,248` valores float32 por token, unos 0.203 MiB por token antes del overhead de serialización. Una caption de 512 tokens ocupa alrededor de 104 MiB en raw, mientras que el cache proyectado en bf16 ocupa unos 4.5 MiB.

Si adaptas esta ruta para entrenar desde cero un modelo comparable estilo Ideogram y la proyección de texto no es un componente preentrenado fijo, desactiva el cache proyectado y planifica el almacenamiento mucho mayor de text embeds raw.

Usa el cache completo solo si necesitas explícitamente las features raw del text encoder o estás depurando compatibilidad de cache:

```json
{
  "text_embed_full_cache": true
}
```

Esto desactiva la optimización de cache proyectado de Ideogram 4 y guarda la salida completa de 13 capas del text encoder.

## Validación

La validación de Ideogram está desactivada hasta que la habilites:

```json
{
  "ideogram_validation": true
}
```

Este flag es temporal. La ruta CFG upstream de Ideogram espera un transformer unconditional separado, mientras que SimpleTuner actualmente entrena solo el transformer conditional por defecto. Con el flag activo, la validación usa el transformer conditional también para el negative/unconditional pass, así que todavía puedes revisar prompts y negative prompts.

## Formato de captions

Ideogram 4 funciona mejor con captions JSON estructuradas. Campos recomendados:

- `high_level_description`
- `style_description`
- `style_description.color_palette` con colores hex
- `compositional_deconstruction.background`
- `compositional_deconstruction.elements`
- `bbox` opcional como `[x1, y1, x2, y2]`

Si tu dataset mezcla texto normal y JSON, mantén:

```json
{
  "ideogram_auto_json": true
}
```

Los prompts de texto se envuelven en el schema JSON de Ideogram; las captions JSON existentes se normalizan y conservan. Las captions JSON escritas a mano siguen siendo mejores, especialmente si describen composición, fondo, elementos y colores.

## Prompt upsampling

Opcionalmente:

```json
{
  "ideogram_prompt_upsample": true,
  "ideogram_prompt_enhancer_head_id": "diffusers/qwen3-vl-8b-instruct-lm-head"
}
```

Esto reescribe prompts con el prompt upsampler de Ideogram antes de convertirlos a JSON. Es más lento; déjalo desactivado hasta confirmar que el entrenamiento base funciona.

## LoRA y LyCORIS

El PEFT LoRA estándar apunta a projections de attention:

```json
{
  "lora_type": "standard",
  "lora_rank": 32
}
```

LyCORIS/LoKr puede apuntar a las clases `Attention` y `FeedForward` expuestas por Ideogram. Full-matrix LoKr puede producir adaptadores enormes; usa LoRA estándar para iterar rápido.

## Expectativa de loss

El loss de Ideogram puede parecer alto comparado con otros modelos. Valores cerca o por encima de `1.0` no significan automáticamente que el modelo esté roto ni que las imágenes de validación sean incoherentes.

En pruebas, Ideogram generó imágenes coherentes aunque el step loss fluctuara alrededor de `0.3-1.3`, con spikes ocasionales. Evalúa por coherencia de validación, fidelidad al prompt y si el loss explota, no por esperar un scalar muy bajo.

## Entrenamiento

```bash
simpletuner train
```

Checkout de desarrollo:

```bash
CONFIG_BACKEND=json CONFIG_PATH=config/config.json .venv/bin/python simpletuner/train.py
```
