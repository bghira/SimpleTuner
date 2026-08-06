# Guía rápida de MiniMax H3

MiniMax H3 es un modelo de video/audio flow-matching de 33B. SimpleTuner soporta training de adapters con la familia `minimaxh3`, incluyendo conditioning FL2VA de primer/último frame y flavours ConvRot cuantizados.

## Configs iniciales

Empieza desde uno de estos ejemplos:

- `simpletuner/examples/minimaxh3-fl2va-convrot-int8.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-24g.peft-lora+ramtorch`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-32g.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-48g.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-80g.peft-lora`

Usa el preset más cercano a tu VRAM y ajusta resolución, frames, attention backend y checkpointing después de un smoke test.

## Ajustes principales

```json
{
  "model_family": "minimaxh3",
  "model_flavour": "convrot-int8",
  "model_type": "lora",
  "mixed_precision": "bf16",
  "base_model_precision": "no_change",
  "text_encoder_1_precision": "int8-quanto",
  "validation_disable_unconditional": true,
  "validation_guidance": 1.0,
  "validation_guidance_real": 1.0
}
```

Los ejemplos usan `convrot-int8`. Puedes usar `convrot-int4` con la misma familia si quieres el checkpoint de menor precisión.

## Mantener la destilación

MiniMax H3 es CFG-distilled. El checkpoint base está pensado para funcionar sin rama unconditional, así que los ejemplos validan con guidance `1.0` y `validation_disable_unconditional: true`.

Los adapters pueden alejarse del comportamiento destilado. Por eso los ejemplos activan `h3_drift` por defecto:

```json
{
  "distillation_method": "h3_drift",
  "distillation_config": {
    "h3_drift": {
      "loss_weight": 0.5,
      "sft_loss_weight": 1.0,
      "balance": "token",
      "video_weight": 1.0,
      "audio_weight": 1.0
    }
  }
}
```

El distiller ejecuta una referencia de la base congelada con el adapter desactivado y penaliza la deriva de la predicción video/audio. Mantenlo activo para LoRAs normales. Baja `loss_weight` si el concepto no aprende; súbelo si la validación pierde el comportamiento base. Más detalles: [MiniMax H3 Drift Distillation](../distillation/MINIMAX_H3_DRIFT.es.md).

Negative prompting no forma parte del contrato base de H3. SimpleTuner mantiene CFG real y negative prompts para checkpoints de-destilados, pero `h3_drift` preserva el comportamiento condicional original.

## Modo de audio

`minimax_h3_target_mode: "auto"` se resuelve como video-only y evita trabajo de audio VAE:

```json
{
  "minimax_h3_target_mode": "video"
}
```

Usa `"av"` solo si el dataset tiene latentes de audio y quieres training conjunto audio/video. También puedes configurarlo por data backend con `h3_target_mode` o `minimax_h3_target_mode`.

## Memoria

- Usa el ejemplo 24G con RamTorch si la VRAM es limitada.
- Prueba `musubi_blocks_to_swap` antes de subir mucho el checkpointing.
- Mantén VAE tiling, slicing y temporal roll activados.
- Benchmarkea `attention_mechanism` en la GPU real.
- Repite el smoke test si cambias `torch.compile`, porque las cachés pueden aumentar VRAM.

## Ejecutar

```bash
simpletuner train example=minimaxh3-fl2va-convrot-int8.peft-lora
```

Haz un smoke test corto y revisa que `h3_drift_loss`, la pérdida normal y las validaciones se comporten de forma coherente.
