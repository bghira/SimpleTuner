# NextLat

NextLat es un objetivo auxiliar que enseña a un transformer a hacer que sus hidden states predigan el siguiente hidden state.

El paper de Next-Latent Prediction estudia transformers de estilo lenguaje y argumenta que next-token prediction estándar no obliga al modelo a comprimir la historia en estados internos estables. NextLat añade una transición autosupervisada en latent space: desde el hidden state actual, predecir el siguiente. En SimpleTuner, esta idea se adapta como regularizador experimental para familias transformer compatibles.

La inferencia no cambia. NextLat añade una loss de entrenamiento y un pequeño predictor, no un sampler nuevo.

## ELI5

El entrenamiento estándar dice: "con lo que has visto, predice el siguiente output".

NextLat añade: "también haz que tus notas internas puedan predecir tus siguientes notas internas".

En modelos de imagen, video y audio, esas notas internas son tokens ocultos dentro del transformer. Si el modelo aprende transiciones internas suaves, puede formar un plan más coherente entre tokens, frames, patches o posiciones RVQ.

## Qué Cambia

Durante entrenamiento:

1. SimpleTuner captura hidden states de un bloque transformer.
2. El predictor recibe cada hidden token excepto el último.
3. Predice el hidden token siguiente.
4. El hidden token real siguiente se usa como target sin gradiente.
5. La loss auxiliar se suma a la loss normal.

El modelo base sigue entrenando con su objetivo principal. NextLat solo añade una presión para que sus estados internos tengan dinámica predictiva.

## Pseudocódigo

```text
para cada batch:
    prediccion = modelo(batch)
    main_loss = loss_normal(prediccion, target)

    hidden = hidden_states_capturados
    actual = hidden tokens 0..N-2
    siguiente = hidden tokens 1..N-1

    pred_siguiente = nextlat_predictor(actual)
    nextlat_loss = distancia(pred_siguiente, stop_gradient(siguiente))

    total_loss = main_loss + nextlat_weight * nextlat_loss
    entrenar_con(total_loss)
```

Con KL opcional, si la familia expone una cabeza compatible:

```text
pred_logits = logits_head(pred_siguiente)
target_logits = logits_head(stop_gradient(siguiente))
total_loss += nextlat_kl_weight * agreement_loss(pred_logits, target_logits)
```

La mayoría de usuarios debe dejar `nextlat_kl_weight=0`.

## Comportamiento en SimpleTuner

- Funciona en familias transformer que exponen hidden states.
- Captura un bloque elegido por `nextlat_block_index`.
- `-1` significa el último bloque soportado.
- Aplana hidden states de imagen, video, audio o tokens en una secuencia.
- Predice un paso adelante en el orden de tokens ocultos.
- El target se detach para que el predictor aprenda sin mover el target hacia atrás.
- El predictor se guarda como módulo entrenable extra cuando el modo de entrenamiento puede guardarlo.

Usa PEFT LoRA estándar o full-model training salvo que la guía del modelo indique otro modo compatible.

## Configuración Rápida

### WebUI

1. Abre **Training → Loss functions**.
2. Activa **NextLat**.
3. Mantén **NextLat Block Index** en `-1` para la primera prueba.
4. Usa un **NextLat Weight** pequeño y positivo.
5. Deja **NextLat State Loss** en `smooth_l1`.
6. Deja **NextLat KL Weight** en `0` salvo recomendación.

### Config JSON / CLI

```json
{
  "nextlat_enabled": true,
  "nextlat_block_index": -1,
  "nextlat_weight": 0.05,
  "nextlat_state_loss": "smooth_l1",
  "nextlat_kl_weight": 0.0
}
```

## Opciones

- `nextlat_enabled`: activa NextLat.
- `nextlat_block_index`: bloque transformer zero-based; `-1` usa el último soportado.
- `nextlat_weight`: multiplicador de la loss auxiliar; debe ser mayor que cero.
- `nextlat_state_loss`: `smooth_l1` por defecto o `mse`.
- `nextlat_kl_weight`: KL opcional si la familia ofrece una cabeza compatible.

## Cómo Elegir Valores

| Situación | Inicio sugerido |
| --- | --- |
| Primera LoRA transformer | `nextlat_block_index=-1`, `nextlat_weight=0.02` a `0.05` |
| Planner AR/RVQ | bloque tardío, `smooth_l1`, peso pequeño |
| Transformer de video | bloque medio-tardío si el final restringe demasiado |
| Loss auxiliar inestable | bajar `nextlat_weight` antes de cambiar el bloque |
| Guía recomienda KL | usar solo el valor documentado |

## Logs

- `nextlat_loss`: loss auxiliar ponderada.
- `nextlat_state_loss`: loss cruda de predicción de hidden states.
- `nextlat_kl_loss`: término KL opcional.

La loss cruda sirve para seguir tendencias; no tiene que estar en la misma escala que la loss principal.

## Compatibilidad

Consulta la tabla de funciones en [Quick Start](../QUICKSTART.es.md).

Requisitos:

- El modelo debe exponer hidden states del transformer.
- El bloque elegido debe existir y poder capturarse.
- La secuencia capturada necesita al menos dos hidden tokens.
- El modo de entrenamiento debe guardar el predictor de NextLat.

NextLat puede combinarse naturalmente con LayerSync, Internal Guidance y CREPA, pero aumenta memoria porque los hidden states deben mantenerse hasta calcular la loss auxiliar.

## Qué Esperar

NextLat tiende a ayudar cuando la coherencia interna importa: planners de códigos RVQ/audio, video transformers, image transformers con estructura espacial y modelos multimodales que necesitan un plan interno estable.

Puede no ayudar si el experimento es muy pequeño, si el peso domina la loss principal o si la familia no expone hidden states útiles.

## Consejos Prácticos

- Empieza con una ablación corta.
- Mantén `nextlat_weight` bajo y súbelo solo si mejora validación.
- Prefiere `smooth_l1`.
- Prueba `-1` primero; luego bloque medio-tardío si hace falta.
- Deja KL apagado salvo guía específica.
- Si sube mucho VRAM, baja batch size o desactiva otros regularizadores de hidden states.

## Referencias

- [Paper Next-Latent Prediction](https://arxiv.org/abs/2511.05963)
- [Código de referencia NextLat](https://github.com/JaydenTeoh/NextLat)
