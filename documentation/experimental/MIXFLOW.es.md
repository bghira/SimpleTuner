# Entrenamiento MixFlow

MixFlow es un método de post-entrenamiento para modelos flow-matching. Entrena el modelo en el timestep $t$ con una interpolación real más ruidosa. Esto reduce la diferencia entre las interpolaciones exactas del entrenamiento y los latents imperfectos encontrados durante el muestreo.

## Configuración

```json
{
  "mixflow_enabled": true,
  "mixflow_gamma": 0.8
}
```

`mixflow_gamma` controla el rango de interpolación ralentizada. `0.8` es el valor predeterminado del artículo. `0.0` conserva la interpolación estándar, pero mantiene el muestreo de timesteps de MixFlow.

MixFlow muestrea el timestep orientado a datos desde $Beta(2,1)$. SimpleTuner almacena flow sigma en la dirección opuesta, orientada al ruido. La implementación usa $sigma = 1 - sqrt(U)$ y después aplica el flow schedule shift configurado. El modelo recibe el timestep original. El latent de entrada usa:

$$
sigma_{input} = sigma + U' gamma (1 - sigma)
$$

El objetivo de velocidad no cambia para un flow lineal. La inferencia no cambia.

## Compatibilidad

Todas las familias SimpleTuner con prediction type `flow_matching` usan la ruta MixFlow compartida. Los wrappers gestionan convenciones data-ward, transformaciones sigma no lineales y entradas conjuntas de audio/video.

MixFlow no se puede combinar con custom/uniform/Beta/fast flow schedules, Self-Flow, TwinFlow, scheduled sampling o distillation. Schedule shift sigue siendo compatible.

Usa MixFlow para post-entrenar un modelo flow existente. Empieza con el learning rate y optimizer de una continuación convencional corta y compara muestras de validación con seed fijo contra el checkpoint inicial.

## Referencias

- [Artículo MixFlow](https://arxiv.org/abs/2512.19311)
- [Implementación de referencia](https://github.com/fudan-generative-vision/MixFlow)
