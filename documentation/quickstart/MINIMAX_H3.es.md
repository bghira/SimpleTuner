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
  "flow_schedule_shift": 12.0,
  "audio_flow_schedule_shift": 3.0,
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

Para training solo de audio, `dataset_type: "audio"` es suficiente. Como H3 declara soporte para fake video,
SimpleTuner registra `audio.audio_only: true` en la configuración normalizada, construye el video placeholder y excluye
su loss. La opción explícita `audio_only` sigue siendo válida, pero no es necesaria.

## Paralelismo de contexto

El context parallelism de H3 usa Ulysses con `context_parallel_strategy: "alltoall"`. La secuencia packed puede incluir
padding hasta el grado CP, por lo que el backend de atención local debe aceptar una máscara. `native` y `cudnn` son
compatibles; SimpleTuner reemplaza otros backends por `native` cuando CP está habilitado.

Con unos 8k audio tokens, CP principalmente intercambia comunicación por menos memoria de activaciones y checkpointing
más ligero. CP no divide los pesos por sí solo; compáralo con DDP salvo que la secuencia sea mayor o se combine con FSDP.

## Atención sparse experimental

MiniMax indica que H3 usó atención sparse 3D tipo MoBA durante su etapa final de entrenamiento. La versión pública inicial usa atención densa, y MiniMax no ha publicado la forma exacta de los bloques, el presupuesto de retención, el calendario de capas ni el kernel de producción. Por eso SimpleTuner mantiene esta aproximación experimental desactivada por defecto.

```json
{
  "minimax_h3_sparse_attention": "moba3d",
  "minimax_h3_sparse_block_shape": "1,8,16",
  "minimax_h3_sparse_video_kv_fraction": 0.25,
  "minimax_h3_sparse_share_heads": false,
  "minimax_h3_sparse_start_layer": 0
}
```

La implementación promedia bloques 3D de query/key para routing top-k sin parámetros. Las queries de video objetivo conservan acceso denso a texto, audio y contexto de referencia; las queries que no son objetivo permanecen densas. Las dimensiones de bloque deben multiplicar 128. Una fracción KV de video de `1.0` es el control numérico de conectividad densa mediante FlexAttention.

Este modo requiere CUDA y agrega un límite de grafo Dynamo alrededor de FlexAttention. Ulysses context parallelism funciona con `context_parallel_strategy=alltoall`; ring context parallelism y TREAD no son compatibles. A 480px, sparse routing puede usar más memoria que FlashAttention porque debe rellenar y reordenar la lattice objetivo y el contexto empaquetado. Trátalo como una ablación de fine-tuning hasta que MiniMax publique su implementación de referencia.

## Memoria

- Usa el ejemplo 24G con RamTorch si la VRAM es limitada.
- Prueba `musubi_blocks_to_swap` antes de subir mucho el checkpointing.
- Mantén `flow_schedule_shift` de video en `12.0` y `audio_flow_schedule_shift` en `3.0`. El helper H3 corrige el default global heredado `3.0` para video porque no coincide con el schedule de MiniMax H3.
- SimpleTuner fuerza VAE tiling y temporal roll/chunking para el video VAE de H3. La geometría usa el upstream `256` tile size con `64` overlap; poner esas opciones en false se ignora porque el decode sin tiling puede producir cambios fuertes de color y patrones halftone.
- Benchmarkea `attention_mechanism` en la GPU real.
- Repite el smoke test si cambias `torch.compile`, porque las cachés pueden aumentar VRAM.

## Ejecutar

```bash
simpletuner train example=minimaxh3-fl2va-convrot-int8.peft-lora
```

Haz un smoke test corto y revisa que `h3_drift_loss`, la pérdida normal y las validaciones se comporten de forma coherente.
