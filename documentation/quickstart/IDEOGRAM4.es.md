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

En GPUs NVIDIA de 80G, el entrenamiento LoRA FP8 o bf16-upcast a 1024px con batch size 1 debería caber normalmente. En GPUs más pequeñas, empieza con FP8 o NF4, rank 8-16, gradient checkpointing y offload. Las GPUs Apple no se recomiendan para entrenar Ideogram 4.

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
