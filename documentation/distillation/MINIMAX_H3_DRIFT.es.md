# Destilación de deriva de MiniMax H3

MiniMax H3 es un modelo de video/audio de flow matching ya destilado. En un entrenamiento LoRA o LyCORIS normal, el adaptador aprende el objetivo del dataset, pero puede mover demasiado el comportamiento destilado del checkpoint base: guidance, equilibrio entre modalidades y la secuencia empaquetada de video/audio.

`h3_drift` evita esa deriva comparando la predicción del adaptador con la predicción del mismo modelo cuando el adaptador está desactivado. No carga otro teacher ni usa una caché de destilación. En cada batch SimpleTuner:

1. calcula la pérdida normal de MiniMax H3 con el adaptador activo;
2. desactiva temporalmente el adaptador;
3. ejecuta la base congelada con `torch.no_grad()` y el mismo batch preparado;
4. calcula MSE entre las predicciones de video/audio;
5. reactiva el adaptador y retropropaga la pérdida combinada.

```text
total = sft_loss_weight * normal_h3_loss + loss_weight * frozen_base_prediction_mse
```

Con un distiller interno:

```text
total = inner_distiller_loss + sft_loss_weight * normal_h3_loss + loss_weight * frozen_base_prediction_mse
```

## Cuándo usarlo

Úsalo para LoRA o LyCORIS de MiniMax H3 salvo que quieras quitar o reemplazar la destilación original. Es útil para LoRAs de estilo o concepto, FL2VA/Ref2VA, entrenamiento conjunto audio/video y flavours cuantizados como `convrot-int8` o `convrot-int4`.

No se admite full-rank: si el transformer completo se actualiza, ya no existe una ruta base congelada fiable para comparar.

## Configuración rápida

```json
{
  "model_family": "minimaxh3",
  "model_flavour": "convrot-int8",
  "model_type": "lora",
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

Los ejemplos incluidos lo activan por defecto con `loss_weight: 0.5`. Es un punto medio: el objetivo del dataset sigue siendo principal, pero la referencia base tiene peso suficiente para limitar la deriva.

## Claves

- `loss_weight`: peso de la pérdida contra la base congelada. Empieza con `0.25` a `0.5`; usa `1.0` si la validación pierde el comportamiento base.
- `sft_loss_weight`: peso de la pérdida normal. Normalmente debe quedar en `1.0`.
- `balance`: `token` promedia por elementos válidos; `modality` promedia por modalidad después de aplicar pesos.
- `video_weight`: peso de la deriva de video.
- `audio_weight`: peso de la deriva de audio.
- `inner_distillation_method`: distiller opcional que se ejecuta dentro de `h3_drift`, por ejemplo `anyflow`, `dmd`, `perflow`, `flow_dpo` o `self_forcing`.
- `inner_distillation_config`: configuración que se pasa al distiller interno.

## Componer otro distiller

`h3_drift` puede envolver otro distiller para usar step distillation o un objetivo de preferencia sin dejar de conservar el comportamiento base de MiniMax H3:

```json
{
  "distillation_method": "h3_drift",
  "distillation_config": {
    "h3_drift": {
      "loss_weight": 0.5,
      "sft_loss_weight": 1.0,
      "inner_distillation_method": "anyflow",
      "inner_distillation_config": {
        "target_mode": "linear",
        "r_timestep_sampler": "zero",
        "loss_weight": 1.0
      }
    }
  }
}
```

El wrapper delega preparación de batches, scheduler de validación, caché de distillation, batches de captions y hooks de ciclo de vida al distiller interno. El distiller interno mantiene sus propias comprobaciones de compatibilidad.

`sft_loss_weight` sigue siendo el objetivo normal de MiniMax H3 incluso si el distiller interno reescribe `target`. Si el distiller interno agrega conditioning de timestep FlowMap/AnyFlow, H3 drift calcula ese término SFT con otro forward del adaptador, quitando antes las claves internas de timestep. Así la ruta normal de inferencia H3 a 30 steps sigue anclada mientras se entrena la ruta step-distilled.

## Audio y video

`minimax_h3_target_mode: "auto"` se resuelve como video-only. Usa `"video"` para desactivar audio o `"av"` para entrenar filas objetivo de audio junto con video. También puedes definir `h3_target_mode` o `minimax_h3_target_mode` por data backend.

El distiller sigue el batch preparado: compara solo video en batches video-only, compara video y audio en batches `av`, respeta `audio_latent_mask`, `sample_weight` y las máscaras visuales.

## Mantener la destilación CFG

MiniMax H3 es CFG-distilled. El checkpoint base se valida normalmente con `validation_guidance: 1.0`, `validation_guidance_real: 1.0` y `validation_disable_unconditional: true`. Negative prompting no forma parte del contrato base.

SimpleTuner soporta CFG real y negative prompt para checkpoints que la comunidad pueda reentrenar fuera de esa destilación. `h3_drift` empuja en sentido contrario: mantiene la predicción condicional cerca de la base. Si quieres de-destilar H3 o enseñar negative prompts, reduce `loss_weight` o desactiva este distiller.

## Logs y coste

Los logs importantes son `h3_drift_loss`, `h3_drift_video_loss`, `h3_drift_audio_loss`, los contadores de elementos, `h3_drift_weighted_loss`, `h3_drift_sft_loss`, `h3_drift_inner_total` cuando hay distiller interno, y `total`.

El coste es una pasada forward adicional por step, sin cargar un segundo transformer. Si envuelve un distiller FlowMap/AnyFlow interno con `sft_loss_weight` activo, también ejecuta un forward normal del adaptador para el ancla SFT. Sigue siendo compatible con ConvRot, RamTorch, musubi block swap, gradient checkpointing y attention offload, pero cada preset debe medirse porque los forwards extra pueden cambiar el backend más rápido.

## Problemas comunes

- Error de low-rank: usa `model_type: "lora"`.
- Audio loss cero: el batch es video-only, el target mode no es `av`, o `audio_latent_mask` excluye todo.
- El adaptador aprende poco: baja `loss_weight`, sube rank o entrena más.
- El audio deriva: prueba `balance: "modality"` o sube `audio_weight`.
