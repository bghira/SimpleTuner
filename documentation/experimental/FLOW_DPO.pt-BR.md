# Flow-DPO e Masked Flow-DPO

Flow-DPO é um método experimental de destilação para modelos flow-matching que treina um adapter de baixo rank com pares preferred/rejected. No SimpleTuner ele é apenas para LoRA/LyCORIS. Full-model Flow-DPO não é suportado, e treino de text encoder é bloqueado para métodos de destilação.

O SimpleTuner reutiliza o sistema de reference dataset. O dataset normal `image` ou `video` fornece a amostra preferred, e um dataset `conditioning` pareado com `conditioning_type=reference_strict` fornece a amostra rejected. Veja [`conditioning_type`](../DATALOADER.pt-BR.md#conditioning_type) e [`conditioning_data`](../DATALOADER.pt-BR.md#conditioning_data).

## Como Funciona

Em cada batch, o SimpleTuner:

1. Roda o modelo com adapter ligado nos latentes preferred.
2. Roda o modelo com adapter ligado nos latentes rejected usando o mesmo prompt, noise e timestep.
3. Desliga o adapter LoRA/LyCORIS e roda as duas predições como reference congelada.
4. Aplica a loss de margem Flow-DPO.

```text
win_adv  = L(reference_win, target_win) - L(policy_win, target_win)
lose_adv = L(policy_lose, target_lose) - L(reference_lose, target_lose)
loss     = -logsigmoid(beta / 2 * (win_adv + lose_adv))
```

Para modelos flow-matching, o target é `noise - latents`.

## Masked Flow-DPO

Se o batch também inclui um dataset `conditioning_type=mask` ou `conditioning_type=segmentation`, o SimpleTuner aplica a mask aos erros DPO antes da redução. Isso concentra o sinal de preferência na região que difere entre preferred e rejected.

`anchor_alpha` adiciona um regularizador MSE global no lado preferred, entre a predição com adapter ligado e desligado. Esse anchor é apenas win-side e não usa mask.

## Configuração

Configuração mínima:

```bash
--model_type=lora
--distillation_method=flow_dpo
--flow_custom_timesteps=801,694,548,338
--flow_timesteps_mode=round-robin
```

Chaves comuns de `distillation_config`:

```json
{
  "flow_dpo": {
    "beta": 1.0,
    "auto_beta": true,
    "auto_beta_target_gf": 0.2,
    "auto_beta_decay": 0.99,
    "norm_type": "sum",
    "mask_dilate": 1,
    "anchor_alpha": 0.0,
    "sft_loss_weight": 0.0
  }
}
```

- `norm_type=sum` corresponde à formulação comum do Flow-DPO.
- `auto_beta=true` ajusta beta pela magnitude média da margem, útil em datasets pareados pequenos.
- `flow_timesteps_mode=fixed-list` amostra aleatoriamente de `flow_custom_timesteps`.
- `flow_timesteps_mode=round-robin` percorre `flow_custom_timesteps` em ciclo.
- `sft_loss_weight` padrão é `0.0`, então a loss diffusion normal não é misturada.

## Formato do Dataset

O dataset rejected deve ser pareado ao preferred com `reference_strict`:

```json
[
  {
    "id": "preferred",
    "dataset_type": "image",
    "type": "local",
    "instance_data_dir": "/data/win",
    "conditioning_data": ["rejected"]
  },
  {
    "id": "rejected",
    "dataset_type": "conditioning",
    "conditioning_type": "reference_strict",
    "type": "local",
    "instance_data_dir": "/data/lose",
    "source_dataset_id": "preferred"
  }
]
```

Para Masked Flow-DPO, adicione também o dataset de mask na mesma lista `conditioning_data`.

## Limites

Flow-DPO atualmente exige:

- Modelo flow-matching.
- `model_type=lora`.
- Dataset conditioning pareado com `reference_strict`.
- Sem treino de text encoder.

Ele não carrega uma segunda cópia completa dos pesos. O reference pass desliga o adapter treinável, incluindo multipliers LyCORIS, e depois religa para o caminho policy.
