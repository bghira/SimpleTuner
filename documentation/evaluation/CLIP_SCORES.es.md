# Seguimiento de puntuaciones CLIP

Las puntuaciones CLIP están vagamente relacionadas con la capacidad del modelo para seguir prompts; no están relacionadas con la calidad/fidelidad de la imagen.

La puntuación `clip/mean` de tu modelo indica qué tan cerca se alinean las características extraídas de la imagen con las características extraídas del prompt. Actualmente es una métrica popular para determinar la adherencia general al prompt, aunque normalmente se evalúa con una cantidad muy grande (~5.000) de prompts de prueba (p. ej., Parti Prompts).

La generación de puntuaciones CLIP durante el preentrenamiento puede ayudar a demostrar que el modelo se acerca a su objetivo, pero una vez que se alcanza un valor `clip/mean` alrededor de `.30` a `.39`, la comparación parece volverse menos significativa. Modelos que muestran una puntuación CLIP promedio alrededor de `.33` pueden superar a un modelo con una puntuación CLIP promedio de `.36` en análisis humano. Sin embargo, un modelo con una puntuación CLIP promedio muy baja alrededor de `0.18` a `0.22` probablemente tendrá un rendimiento bastante pobre.

Dentro de una sola ejecución de prueba, algunos prompts darán una puntuación CLIP muy baja alrededor de `0.14` (valor `clip/min` en los gráficos del tracker) aunque sus imágenes se alineen bastante bien con el prompt del usuario y tengan alta calidad; por el contrario, puntuaciones CLIP tan altas como `0.39` (valor `clip/max` en los gráficos del tracker) pueden aparecer en imágenes de calidad cuestionable, ya que esta prueba no pretende capturar esa información. Por eso normalmente se usa una cantidad tan grande de prompts para medir el rendimiento del modelo, _y aun así_...

Por sí solas, las puntuaciones CLIP no tardan mucho en calcularse; sin embargo, la cantidad de prompts requerida para una evaluación significativa puede hacer que tome muchísimo tiempo.

Como no cuesta mucho ejecutarlo, no hace daño incluir la evaluación CLIP en entrenamientos pequeños. Quizás descubras un patrón en las salidas donde tenga sentido abandonar una ejecución de entrenamiento o ajustar otros hiperparámetros como la tasa de aprendizaje.

Para incluir una biblioteca de prompts estándar para evaluación, se puede proporcionar `--validation_prompt_library` y luego generaremos un benchmark relativo entre ejecuciones de entrenamiento.

En `config.json`:

```json
{
  ...
  "evaluation_type": "clip",
  "pretrained_evaluation_model_name_or_path": "openai/clip-vit-large-patch14-336",
  "report_to": "tensorboard", # or wandb
  ...
}
```

## Compatibilidad

SageAttention no es compatible actualmente con el seguimiento de puntuaciones CLIP. Debe desactivarse una u otra.
