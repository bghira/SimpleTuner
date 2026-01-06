# Diff2Flow (Puente de difusión a flujo)

## Antecedentes

Históricamente, los modelos de difusión se han categorizado por sus objetivos de predicción:
*   **Epsilon ($\epsilon$):** Predice el ruido añadido a la imagen (SD 1.5, SDXL).
*   **V-Prediction ($v$):** Predice una velocidad que combina ruido y datos (SD 2.0, SDXL Refiner).

Modelos de última generación como **Flux**, **Stable Diffusion 3** y **AuraFlow** usan **Flow Matching** (específicamente Rectified Flow). Flow Matching trata el proceso de generación como una ecuación diferencial ordinaria (ODE) que mueve partículas desde una distribución de ruido a una distribución de datos a lo largo de trayectorias rectas.

Esta trayectoria en línea recta suele ser más fácil para los solvers, permitiendo menos pasos y una generación más estable.

## El puente

**Diff2Flow** es un adaptador liviano que permite que modelos "legacy" (Epsilon o V-pred) se entrenen con un objetivo de Flow Matching sin cambiar su arquitectura subyacente.

Funciona convirtiendo matemáticamente la salida nativa del modelo (p. ej., una predicción epsilon) en un campo vectorial de flujo $u_t(x|1)$ y luego calculando la pérdida contra el objetivo de flujo ($x_1 - x_0$, o `noise - latents`).

> 🟡 **Estado experimental:** Esta función cambia efectivamente el paisaje de pérdida que ve el modelo. Aunque es teóricamente sólida, altera significativamente la dinámica de entrenamiento. Está pensada principalmente para investigación y experimentación.

## Configuración

Para usar Diff2Flow, necesitas habilitar el puente y, opcionalmente, cambiar la función de pérdida.

### Configuración básica

Agrega estas claves a tu `config.json`:

```json
{
  "diff2flow_enabled": true,
  "diff2flow_loss": true
}
```

### Referencia de opciones

#### `--diff2flow_enabled` (Boolean)
**Default:** `false`
Inicializa el puente matemático. Esto asigna un pequeño buffer para cálculos de timestep, pero no cambia el comportamiento de entrenamiento por sí solo a menos que `diff2flow_loss` también esté configurado.
*   **Requerido para:** `diff2flow_loss`.
*   **Modelos compatibles:** Cualquier modelo que use `epsilon` o `v_prediction` (SD1.5, SD2.x, SDXL, DeepFloyd IF, PixArt Alpha).

#### `--diff2flow_loss` (Boolean)
**Default:** `false`
Cambia el objetivo de entrenamiento.
*   **False:** El modelo minimiza el error entre su predicción y el objetivo estándar (p. ej., `MSE(pred_noise, real_noise)`).
*   **True:** El modelo minimiza el error entre la predicción *convertida a flujo* y el objetivo de flujo (`noise - latents`).

### Sinergias

Diff2Flow se combina extremadamente bien con **Scheduled Sampling**.

Cuando combinas:
1.  **Diff2Flow** (enderezando las trayectorias)
2.  **Scheduled Sampling** (entrenamiento con rollouts autogenerados)

En efecto aproximas la receta de entrenamiento usada por modelos **Reflow** o **Rectified Flow**, lo que puede aportar estabilidad y calidad modernas a arquitecturas antiguas como SDXL.
