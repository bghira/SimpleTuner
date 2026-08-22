# Explorative Modeling (XM)

Explorative Modeling, abreviado como XM en SimpleTuner, es una técnica de entrenamiento que permite al modelo probar más de una elección oculta para el mismo ejemplo supervisado y aprender solo de la elección que mejor encaja con el objetivo.

El trabajo original de Explorative Modeling lo presenta como un tercer eje de escalado para modelos generativos: además de más datos y más parámetros, el modelo puede gastar más cómputo de entrenamiento explorando más candidatos. En SimpleTuner, XM es un objetivo experimental para familias de imagen, video, audio y modelos autorregresivos compatibles.

La inferencia no cambia. XM solo cambia cómo se construye, evalúa y reduce el batch durante entrenamiento.

## ELI5

Imagina pedirle a una persona que dibuje una imagen objetivo, pero dejarle hacer cuatro bocetos antes de calificar. En vez de promediar los cuatro, calificas el boceto que más se parece al objetivo y enseñas a partir de ese.

La idea central es:

1. Crear varios candidatos para el mismo sample.
2. Ejecutar el modelo sobre todos ellos.
3. Puntuar cada candidato contra el target real.
4. Conservar el mejor candidato por sample o bloque de tokens.
5. Retropropagar solo la loss seleccionada.

Esto ayuda cuando el objetivo puede explicarse de varias formas válidas. Un único camino forzado puede enseñar al modelo a promediar posibilidades; varios caminos explorados le permiten comprometerse con un modo plausible.

## Qué Cambia

XM no añade un sampler de inferencia nuevo, un formato de checkpoint nuevo ni un segundo teacher. Cambia la selección durante entrenamiento:

- El entrenamiento estándar samplea un candidato y aprende de él.
- XM samplea `K` candidatos y aprende del candidato con menor loss.
- Un `K` más alto da más exploración, pero cuesta más cómputo.

En modelos de difusión y flow, el candidato suele ser el ruido usado para construir el latent ruidoso en el timestep seleccionado.

En modelos autorregresivos de tokens, como planners RVQ/audio, el candidato es un route embedding aprendido que da al modelo varios caminos internos para la misma secuencia supervisada.

## Comportamiento en SimpleTuner

### Modelos de Difusión y Flow

Para familias de difusión o flow matching compatibles, usa `xm_training_target=noise`.

SimpleTuner:

1. Samplea el timestep o sigma normal de entrenamiento.
2. Repite el batch `xm_candidate_count` veces.
3. Genera un ruido distinto para cada candidato.
4. Construye latents ruidosos con cada ruido candidato.
5. Ejecuta el modelo sobre el batch expandido.
6. Calcula la loss normal de entrenamiento para cada candidato.
7. Elige el candidato de menor loss por sample original.
8. Retropropaga la loss seleccionada.

El modelo sigue aprendiendo su prediction type normal: flow velocity, epsilon, v-prediction o sample prediction según la familia.

### Modelos Autorregresivos y RVQ

Para planners autorregresivos compatibles, usa `xm_training_target=route`.

SimpleTuner:

1. Añade una pequeña tabla de route embeddings aprendidos.
2. Repite cada secuencia supervisada sobre los candidatos de ruta.
3. Inserta la señal de ruta en la entrada del modelo.
4. Calcula token losses para cada ruta.
5. Selecciona la mejor ruta para todo el sample o para bloques configurados.
6. Retropropaga solo la loss de la ruta seleccionada.

Esto es útil para planners tipo global LM que predicen códigos RVQ de audio u otros streams discretos. La ruta da varias explicaciones internas del mismo target sin cambiar el decode de inferencia.

## Pseudocódigo

```text
para cada batch:
    candidatos = []

    para candidate_id en 1..K:
        entrada = crear_candidato(batch, candidate_id)
        prediccion = modelo(entrada)
        loss = comparar(prediccion, target)
        candidatos.append(loss)

    loss_seleccionada = menor_loss_por_sample_o_bloque(candidatos)
    entrenar_con(loss_seleccionada)
```

Para difusión:

```text
entrada = agregar_ruido(clean_latent, ruido_candidato, timestep)
loss = diffusion_or_flow_loss(modelo(entrada), target_de_entrenamiento)
```

Para selección autorregresiva de rutas:

```text
entrada = agregar_route_embedding(secuencia_tokens, ruta_candidata)
loss = token_loss(modelo(entrada), tokens_objetivo)
```

## Configuración Rápida

### WebUI

1. Abre **Training → Loss functions**.
2. Activa **XM**.
3. Pon **XM Candidates** en `2` o `4`.
4. Elige **XM Training Target**:
   - `noise` para modelos de difusión o flow.
   - `route` para planners autorregresivos/RVQ.
5. Mantén **XM Selection Scope** en `sample` salvo que la guía del modelo recomiende block selection.
6. Deja **XM Block Size** en `0` salvo que uses selección por bloques basada en rutas.

### Config JSON / CLI

```json
{
  "xm_enabled": true,
  "xm_candidate_count": 4,
  "xm_training_target": "noise",
  "xm_selection_scope": "sample",
  "xm_block_size": 0
}
```

Para entrenamiento AR/RVQ con rutas:

```json
{
  "xm_enabled": true,
  "xm_candidate_count": 4,
  "xm_training_target": "route",
  "xm_selection_scope": "block",
  "xm_block_size": 16
}
```

## Opciones

- `xm_enabled`: activa XM.
- `xm_candidate_count`: candidatos por sample. Debe ser al menos `2` cuando XM está activo.
- `xm_training_target`: tipo de candidato: `noise` para difusión/flow, `route` para planners de tokens.
- `xm_selection_scope`: granularidad de selección. `sample` elige un ganador por sample; `block` elige por bloques cuando la familia lo soporta.
- `xm_block_size`: tamaño del bloque de tokens o frames. `0` significa la secuencia supervisada completa.

## Cómo Elegir Valores

| Situación | Inicio sugerido |
| --- | --- |
| LoRA de imagen o video | `xm_candidate_count=2`, `xm_training_target=noise`, `xm_selection_scope=sample` |
| Dataset grande o ambiguo | Probar `xm_candidate_count=4` |
| Planner RVQ/audio | `xm_training_target=route`, `xm_selection_scope=block`, block size de la guía del modelo |
| Primera prueba en una familia | Mantener block size `0` y comparar validación contra una baseline sin XM |

El coste suele crecer aproximadamente con el número de candidatos.

## Logs

XM puede registrar:

- `xm_loss`: loss seleccionada.
- `xm_candidate_loss_mean`: loss media antes de seleccionar.
- `xm_candidate_0_wins`, `xm_candidate_1_wins`, etc.: frecuencia de victoria por candidato.
- `xm_route_usage` o entradas por ruta en modelos AR/RVQ.

Buenas señales: varios candidatos ganan a veces, validación mejora y las rutas no colapsan durante mucho tiempo.

Señales preocupantes: un candidato gana siempre desde el inicio, baja la loss pero empeora validación, o el coste de memoria/tiempo obliga a reducir demasiado el batch.

## Compatibilidad

Consulta la tabla de funciones en [Quick Start](../QUICKSTART.es.md).

Reglas generales:

- XM de difusión/flow usa candidatos de ruido y selección por sample.
- XM AR/RVQ usa candidatos de ruta y puede soportar selección por bloques.
- Las familias no soportadas fallan explícitamente.

Para XM de ruido en difusión, SimpleTuner actualmente trata estas funciones como incompatibles salvo que una familia indique lo contrario: TwinFlow, Scheduled Sampling, `input_perturbation`, CREPA self-flow y loss de segmentación estocástica.

## Relación con Otras Funciones

- **MixFlow** cambia la trayectoria de entrenamiento; XM cambia la selección de candidatos.
- **Diff2Flow** cambia el target de modelos legacy; XM puede seleccionar candidatos antes de reducir la loss donde esté soportado.
- **NextLat** regulariza dinámicas de hidden states; XM elige rutas o ruidos candidatos.
- **LayerSync y CREPA** alinean representaciones; XM selecciona el candidato más explicativo.

## Consejos Prácticos

- Usa seeds de validación fijos al comparar.
- Baja el batch size si `xm_candidate_count` presiona VRAM.
- No juzgues XM solo por training loss; mira validación y diversidad.
- En AR/RVQ, evita block size `1` salvo recomendación específica.
- Haz primero una ablación corta: mismo modelo, dataset y seed, solo XM on/off.

## Referencias

- [Página del proyecto Explorative Modeling](https://explorative-modeling.github.io/)
- [Paper Explorative Modeling](https://arxiv.org/abs/2607.27372)
