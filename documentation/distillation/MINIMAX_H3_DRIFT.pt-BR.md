# Destilação de Drift do MiniMax H3

MiniMax H3 é um modelo de video/audio flow-matching já destilado. Em training LoRA ou LyCORIS normal, o adapter aprende o target do dataset, mas pode deslocar demais o comportamento destilado do checkpoint base: guidance, balanço de modalidades e o layout da sequência video/audio empacotada.

`h3_drift` compara a prediction do adapter com a prediction do mesmo modelo quando o adapter está desativado. Ele não carrega outro teacher e não usa cache de distillation. A cada batch o SimpleTuner:

1. calcula a loss normal do MiniMax H3 com o adapter ativo;
2. desativa temporariamente o adapter;
3. roda a base congelada com `torch.no_grad()` no mesmo prepared batch;
4. calcula MSE entre predictions de video/audio;
5. reativa o adapter e faz backprop da loss combinada.

```text
total = sft_loss_weight * normal_h3_loss + loss_weight * frozen_base_prediction_mse
```

## Quando usar

Use em LoRA ou LyCORIS de MiniMax H3, a menos que você queira remover a destilação original. É útil para LoRAs de estilo/conceito, FL2VA/Ref2VA, training conjunto audio/video e flavours quantizados como `convrot-int8` e `convrot-int4`.

Full-rank não é suportado. Quando o transformer inteiro é atualizado, não há uma rota base congelada confiável para comparar.

## Config rápido

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

Os exemplos H3 incluídos ativam isso por padrão. `loss_weight: 0.5` mantém o target do dataset como principal, mas dá peso suficiente à referência base para conter drift.

## Chaves

- `loss_weight`: multiplicador da loss contra a base congelada. Comece com `0.25` a `0.5`; use `1.0` se a validação perder comportamento base.
- `sft_loss_weight`: multiplicador da loss normal. Normalmente fica em `1.0`.
- `balance`: `token` faz média por elementos válidos; `modality` faz média por modalidade depois dos pesos.
- `video_weight`: peso do termo de drift de video.
- `audio_weight`: peso do termo de drift de audio.

## Video e Audio

`minimax_h3_target_mode: "auto"` vira video-only. Use `"video"` para não treinar audio target rows, ou `"av"` para joint audio/video. Também pode ser definido por data backend com `h3_target_mode` ou `minimax_h3_target_mode`.

O distiller segue o prepared batch: compara só video em video-only, compara video e `audio_prediction` em `av`, e respeita `audio_latent_mask`, `sample_weight` e visual masks.

## Manter a destilação CFG

MiniMax H3 é CFG-distilled. O checkpoint base normalmente valida com `validation_guidance: 1.0`, `validation_guidance_real: 1.0` e `validation_disable_unconditional: true`. Negative prompting não faz parte do contrato base.

SimpleTuner suporta real CFG e negative prompt encoding porque a comunidade pode de-destilar H3. `h3_drift` faz a pressão oposta: mantém o adapter perto da conditional prediction da base. Para ensinar negative prompts ou de-destilar, reduza `loss_weight` ou desative o distiller.

## Logs e custo

Logs principais: `h3_drift_loss`, `h3_drift_video_loss`, `h3_drift_audio_loss`, element counts, `h3_drift_weighted_loss`, `h3_drift_sft_loss` e `total`.

Cada step ganha um forward pass extra, mas não mantém um segundo transformer na memória. Funciona com ConvRot, RamTorch, musubi block swap, gradient checkpointing e attention offload; ainda assim, benchmark cada preset porque o forward extra pode mudar o backend mais rápido.

## Troubleshooting

- Erro de low-rank: use `model_type: "lora"`.
- Audio loss zero: batch video-only, target mode não é `av`, ou `audio_latent_mask` exclui tudo.
- Adapter aprende pouco: reduza `loss_weight`, aumente rank ou treine por mais tempo.
- Audio deriva: use `balance: "modality"` ou aumente `audio_weight`.
