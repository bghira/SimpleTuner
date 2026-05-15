# Flow-DPO y Masked Flow-DPO

Flow-DPO es un método experimental de destilación para modelos flow-matching que entrena un adapter de bajo rango con pares preferred/rejected. En SimpleTuner solo se admite LoRA/LyCORIS. No se admite full-model Flow-DPO, y el entrenamiento del text encoder queda bloqueado para los métodos de destilación.

SimpleTuner reutiliza el sistema de reference datasets. El dataset normal `image` o `video` aporta la muestra preferred, y un dataset `conditioning` emparejado con `conditioning_type=reference_strict` aporta la muestra rejected. Consulta [`conditioning_type`](../DATALOADER.es.md#conditioning_type) y [`conditioning_data`](../DATALOADER.es.md#conditioning_data).

## Qué Hace

En cada batch, SimpleTuner:

1. Ejecuta el modelo con adapter activado sobre los latentes preferred.
2. Ejecuta el modelo con adapter activado sobre los latentes rejected con el mismo prompt, noise y timestep.
3. Desactiva el adapter LoRA/LyCORIS y repite ambas predicciones como reference congelada.
4. Aplica la pérdida de margen Flow-DPO.

```text
win_adv  = L(reference_win, target_win) - L(policy_win, target_win)
lose_adv = L(policy_lose, target_lose) - L(reference_lose, target_lose)
loss     = -logsigmoid(beta / 2 * (win_adv + lose_adv))
```

Para modelos flow-matching, el target es `noise - latents`.

## Masked Flow-DPO

Si el batch también incluye un dataset `conditioning_type=mask` o `conditioning_type=segmentation`, SimpleTuner aplica esa mask a los errores DPO antes de reducirlos. Esto concentra la señal de preferencia en la región que cambia entre preferred y rejected.

`anchor_alpha` añade un regularizador MSE global entre las predicciones con adapter activado y desactivado tanto en las muestras preferred como rejected. Este anchor no usa mask, así que limita el drift de todo el frame.

## Configuración

Configuración mínima:

```bash
--model_type=lora
--distillation_method=flow_dpo
--flow_custom_timesteps=801,694,548,338
--flow_timesteps_mode=round-robin
```

Claves comunes de `distillation_config`:

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

- `norm_type=sum` coincide con la formulación habitual de Flow-DPO. `mean` promedia todos los elementos latentes, y `masked_mean` promedia los elementos activos de la mask cuando hay una mask.
- `auto_beta=true` adapta beta con la magnitud media de la margen, útil para datasets pareados pequeños.
- `flow_timesteps_mode=fixed-list` muestrea aleatoriamente de `flow_custom_timesteps`.
- `flow_timesteps_mode=round-robin` recorre `flow_custom_timesteps` en ciclo. Los ranks distribuidos usan offsets distintos, y los runs reanudados inicializan el cursor desde `global_step`.
- `sft_loss_weight` por defecto es `0.0`, así que no se mezcla la diffusion loss normal.

SimpleTuner registra los valores principales de salud Flow-DPO: beta, margin, ventajas win/lose, errores policy/reference, porcentaje de margins negativos y gradient factor. Las métricas extendidas de reward-hacking del model card original pertenecen al tooling de análisis de esa release y todavía no se emiten todas desde SimpleTuner.

## Forma del Dataset

El dataset rejected debe emparejarse con el preferred usando `reference_strict`:

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

Para Masked Flow-DPO, añade también el dataset de mask a la misma lista `conditioning_data`.

## Límites

Flow-DPO actualmente requiere:

- Un modelo flow-matching.
- `model_type=lora`.
- Un dataset conditioning emparejado con `reference_strict`.
- Sin entrenamiento de text encoder.

No carga una segunda copia completa de los pesos. El reference pass desactiva el adapter entrenable, incluidos los multipliers LyCORIS, y luego lo reactiva para la ruta policy.
