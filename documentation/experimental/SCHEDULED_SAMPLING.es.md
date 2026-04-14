# Scheduled Sampling (Rollout)

## Antecedentes

El entrenamiento estándar de difusión se basa en "Teacher Forcing". Tomamos una imagen limpia, le añadimos una cantidad precisa de ruido y pedimos al modelo que prediga ese ruido (o la velocidad/imagen original). La entrada al modelo siempre está "perfectamente" ruidosa: se encuentra exactamente en el schedule teórico de ruido.

Sin embargo, durante la inferencia (generación), el modelo se alimenta de sus propias salidas. Si comete un pequeño error en el paso $t$, ese error alimenta el paso $t-1$. Estos errores se acumulan, causando que la generación se desvíe del manifold de imágenes válidas. Esta discrepancia entre entrenamiento (entradas perfectas) e inferencia (entradas imperfectas) se llama **sesgo de exposición**.

**Scheduled Sampling** (a menudo llamado "Rollout" en este contexto) aborda esto entrenando al modelo con sus propias salidas generadas.

## Cómo funciona

En lugar de simplemente añadir ruido a una imagen limpia, el bucle de entrenamiento realiza ocasionalmente una mini sesión de inferencia:

1.  Elegir un **timestep objetivo** $t$ (el paso en el que queremos entrenar).
2.  Elegir un **timestep fuente** $t+k$ (un paso más ruidoso más atrás en el schedule).
3.  Usar los pesos *actuales* del modelo para generar (denoise) de $t+k$ hasta $t$.
4.  Usar este latente autogenerado y ligeramente imperfecto en el paso $t$ como entrada para la pasada de entrenamiento.

Al hacer esto, el modelo ve entradas que contienen exactamente los artefactos y errores que produce en ese momento. Aprende a decir: "Ah, cometí este error, así lo corrijo", llevando efectivamente la generación de vuelta al camino válido.

## Configuración

Esta función es experimental y añade overhead computacional, pero puede mejorar significativamente la adherencia al prompt y la estabilidad estructural, especialmente en datasets pequeños (Dreambooth).

Para habilitarla, debes configurar un `max_step_offset` distinto de cero.

### Configuración básica

Agrega lo siguiente a tu `config.json`:

```json
{
  "scheduled_sampling_max_step_offset": 10,
  "scheduled_sampling_probability": 1.0,
  "scheduled_sampling_sampler": "unipc"
}
```

### Referencia de opciones

#### `scheduled_sampling_max_step_offset` (Integer)
**Default:** `0` (Desactivado)
El número máximo de pasos a desplegar. Si se establece en `10`, el trainer elegirá una longitud de rollout aleatoria entre 0 y 10 para cada muestra.
> 🟢 **Recomendación:** Empieza pequeño (p. ej., `5` a `10`). Incluso rollouts cortos ayudan al modelo a aprender corrección de errores sin ralentizar drásticamente el entrenamiento.

#### `scheduled_sampling_probability` (Float)
**Default:** `0.0`
La probabilidad (0.0 a 1.0) de que cualquier elemento del batch haga rollout.
*   `1.0`: Cada muestra hace rollout (más costoso).
*   `0.5`: 50% de muestras con entrenamiento estándar, 50% con rollout.

#### `scheduled_sampling_ramp_steps` (Integer)
**Default:** `0`
Si se configura, la probabilidad subirá linealmente desde `scheduled_sampling_prob_start` (por defecto 0.0) hasta `scheduled_sampling_prob_end` (por defecto 0.5) a lo largo de este número de pasos globales.
> 🟢 **Tip:** Esto actúa como un "warmup". Permite que el modelo aprenda a denoising básico antes de introducir la tarea más difícil de corregir sus propios errores.

#### `scheduled_sampling_sampler` (String)
**Default:** `unipc`
El solver usado para los pasos de generación del rollout.
*   **Opciones:** `unipc` (recomendado, rápido y preciso), `euler`, `dpm`.
*   `unipc` suele ser el mejor equilibrio entre velocidad y precisión para estas ráfagas cortas de sampling.

### Flow Matching + ReflexFlow

Para modelos de flow-matching (`--prediction_type flow_matching`), scheduled sampling ahora soporta mitigación de sesgo de exposición estilo ReflexFlow:

*   `scheduled_sampling_reflexflow`: Habilita mejoras de ReflexFlow durante el rollout (se habilita automáticamente para modelos flow-matching cuando scheduled sampling está activo; pasa `--scheduled_sampling_reflexflow=false` para desactivarlo).
*   `scheduled_sampling_reflexflow_alpha`: Escala el peso de pérdida basado en sesgo de exposición (compensación de frecuencia).
*   `scheduled_sampling_reflexflow_beta1`: Escala el regularizador direccional anti-deriva (por defecto 10.0 para igualar el paper).
*   `scheduled_sampling_reflexflow_beta2`: Escala la pérdida compensada por frecuencia (por defecto 1.0).

Estos reutilizan las predicciones/latentes de rollout que ya calculas, evitando una pasada adicional de gradiente, y ayudan a mantener rollouts sesgados alineados con la trayectoria limpia mientras enfatizan componentes faltantes de baja frecuencia al inicio del denoising.

### Impacto en el rendimiento

> ⚠️ **Advertencia:** Habilitar rollout requiere ejecutar el modelo en modo de inferencia *dentro* del bucle de entrenamiento.
>
> Si configuras `max_step_offset=10`, el modelo puede ejecutar hasta 10 pasadas forward extra por paso de entrenamiento. Esto reducirá tu `it/s` (iteraciones por segundo). Ajusta `scheduled_sampling_probability` para equilibrar la velocidad de entrenamiento vs. ganancias de calidad.
