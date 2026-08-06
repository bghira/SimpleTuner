# Guia rápido do MiniMax H3

MiniMax H3 é um modelo de video/audio flow-matching de 33B. SimpleTuner suporta adapter training pela family `minimaxh3`, incluindo conditioning FL2VA de primeiro/último frame e flavours ConvRot quantizados.

## Configs iniciais

Comece por um destes examples:

- `simpletuner/examples/minimaxh3-fl2va-convrot-int8.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-24g.peft-lora+ramtorch`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-32g.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-48g.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-80g.peft-lora`

Use o preset mais próximo da sua VRAM e ajuste resolução, frames, attention backend e checkpointing depois de um smoke test.

## Ajustes principais

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

Os examples usam `convrot-int8`. Use `convrot-int4` na mesma family se quiser o checkpoint de menor precision.

## Manter a destilação

MiniMax H3 é CFG-distilled. O checkpoint base foi feito para rodar sem branch unconditional, então os examples validam com guidance `1.0` e `validation_disable_unconditional: true`.

Adapter training ainda pode driftar para longe do comportamento destilado. Por isso os examples ativam `h3_drift` por padrão:

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

O distiller roda uma referência da base congelada com o adapter desativado e penaliza drift na prediction de video/audio. Mantenha ativo para LoRAs H3 normais. Reduza `loss_weight` se o concept não aprende; aumente se a validação perde o comportamento base. Veja [MiniMax H3 Drift Distillation](../distillation/MINIMAX_H3_DRIFT.pt-BR.md).

Negative prompting não faz parte do contrato base do H3. SimpleTuner mantém real CFG e negative prompts para checkpoints de-destilados, mas `h3_drift` preserva o comportamento condicional original.

## Modo de audio

`minimax_h3_target_mode: "auto"` vira video-only e evita trabalho de audio VAE:

```json
{
  "minimax_h3_target_mode": "video"
}
```

Use `"av"` só quando o dataset tiver target audio latents e você quiser joint audio/video training. Também pode configurar por backend com `h3_target_mode` ou `minimax_h3_target_mode`.

## Memória

- Use o example 24G com RamTorch quando VRAM estiver apertada.
- Teste `musubi_blocks_to_swap` antes de aumentar muito checkpointing.
- Mantenha VAE tiling, slicing e temporal roll ligados.
- Faça benchmark de `attention_mechanism` na GPU real.
- Refaça o smoke test após mudar `torch.compile`, porque caches podem aumentar VRAM.

## Rodar

```bash
simpletuner train example=minimaxh3-fl2va-convrot-int8.peft-lora
```

Faça um smoke test curto antes de uma execução longa e confirme que `h3_drift_loss`, loss normal e samples de validação evoluem de forma coerente.
