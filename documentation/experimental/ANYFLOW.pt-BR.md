# AnyFlow

AnyFlow e um modo experimental de destilacao para modelos flow-matching. Ele condiciona o modelo em dois tempos de fluxo, o timestep normal `t` e um timestep de referencia menor `r`, para aprender um mapa de fluxo ao longo de um intervalo em vez de apenas uma velocidade rectified-flow pontual.

No SimpleTuner:

- `--distillation_method=anyflow` ativa o `AnyFlowDistiller`.
- O distiller chama `enable_flowmap_time_conditioning()` no componente treinado durante a inicializacao.
- Cada batch preparado recebe `flowmap_r_timesteps`.
- O target normal e substituido por um target AnyFlow antes do calculo da loss.

AnyFlow e online no SimpleTuner. Ele nao requer cache ODE precomputado.

Para um exemplo de continuacao Wan usando os checkpoints AnyFlow publicados pela NVIDIA, veja [Quickstart de continuacao AnyFlow](/documentation/quickstart/ANYFLOW.pt-BR.md).

## Configuracao rapida

```json
{
  "model_type": "lora",
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

Treinamento de text encoder e bloqueado para todos os metodos de destilacao do SimpleTuner, incluindo AnyFlow.

## Como funciona

Para cada batch flow-matching, o SimpleTuner:

1. Usa o `prepare_batch()` normal para amostrar `sigmas`, `timesteps`, `noisy_latents` e o target base.
2. Amostra `r < t` no intervalo atual.
3. Escreve `flowmap_r_timesteps` no batch para que o wrapper passe isso como `r_timestep`.
4. Constroi o target de treinamento.
5. Usa a loss normal para comparar a predicao com esse target.

Com `target_mode=online_teacher`, o target e uma velocidade media do noisy latent em `t` ate `r`. Em LoRA e LyCORIS, o distiller desativa temporariamente o adapter para o rollout teacher e reativa depois.

Com `target_mode=linear`, nenhum rollout teacher e usado. O target e `noise - latents`. Isso e util para smoke tests e ablations, mas nao e o objetivo completo de mapa teacher do AnyFlow.

## Opcoes

- `target_mode`: `online_teacher` ou `linear`. Padrao: `online_teacher`.
- `teacher_rollout_steps`: passos Euler online entre `t` e `r`. Padrao: `1`.
- `r_timestep_sampler`: `uniform` ou `zero`. Padrao: `uniform`.
- `min_interval_ratio`: intervalo normalizado minimo entre `t` e `r`. Padrao: `0.02`.
- `gate_value`: peso de mistura do embedding delta FlowMap. Padrao: `0.25`.
- `deltatime_type`: `r` ou `t-r`. Padrao: `r`.
- `loss_weight`: multiplicador da loss ja calculada. Padrao: `1.0`.
- `timestep_scale`: override para modelos com escala de timestep customizada. Normalmente deixe sem definir.

## Limites

- Requer modelo flow-matching.
- Requer timesteps escalares por amostra. Intervalos AnyFlow tokenwise ainda nao estao conectados.
- Requer `r_timestep < timestep`; timestep zero e rejeitado.
- O modo online teacher atual e pensado para LoRA/LyCORIS. Full-rank online teacher precisa de wiring separado de student/teacher.
- Validacao padrao ainda pode rodar sem `r_timestep`, mas sampling AnyFlow de poucos passos precisa de suporte no sampler ou pipeline para passar o endpoint do intervalo como `r_timestep`. Essa integracao de geracao ainda e um follow-up.
