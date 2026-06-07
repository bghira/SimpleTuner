# AnyFlow

AnyFlow es un modo experimental de destilación para modelos de flow matching. Entrena el modelo con dos tiempos de flujo, el timestep normal `t` y un timestep de referencia menor `r`, para que aprenda un mapa de flujo sobre un intervalo en vez de solo una velocidad rectified-flow puntual.

En SimpleTuner:

- `--distillation_method=anyflow` activa `AnyFlowDistiller`.
- El distiller llama `enable_flowmap_time_conditioning()` en el componente entrenado durante el arranque.
- Cada batch preparado recibe `flowmap_r_timesteps`.
- El target normal se reemplaza por un target AnyFlow antes de calcular la pérdida.

AnyFlow es online en SimpleTuner. No requiere una caché ODE precalculada.

Para un ejemplo de continuación Wan usando los checkpoints AnyFlow publicados por NVIDIA, consulta [Guía rápida de continuación AnyFlow](/documentation/quickstart/ANYFLOW.es.md).

## Configuración rápida

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

El entrenamiento del text encoder está bloqueado para todos los métodos de destilación de SimpleTuner, incluido AnyFlow.

## Cómo funciona

Para cada batch de flow matching, SimpleTuner:

1. Usa el `prepare_batch()` normal para muestrear `sigmas`, `timesteps`, `noisy_latents` y el target base.
2. Muestrea `r < t` dentro del intervalo actual.
3. Escribe `flowmap_r_timesteps` en el batch para que el wrapper lo pase como `r_timestep`.
4. Construye el target de entrenamiento.
5. Deja que la pérdida normal compare la predicción contra ese target.

Con `target_mode=online_teacher`, el target es una velocidad media desde el latent ruidoso en `t` hacia `r`. En LoRA y LyCORIS, el distiller desactiva temporalmente el adapter para el rollout del teacher y lo reactiva después.

Con `target_mode=linear`, no se usa rollout del teacher. El target es `noise - latents`. Sirve para smoke tests y ablaciones, pero no es el objetivo completo de mapa AnyFlow.

## Opciones

- `target_mode`: `online_teacher` o `linear`. Predeterminado: `online_teacher`.
- `teacher_rollout_steps`: pasos Euler online entre `t` y `r`. Predeterminado: `1`.
- `r_timestep_sampler`: `uniform` o `zero`. Predeterminado: `uniform`.
- `min_interval_ratio`: intervalo normalizado mínimo entre `t` y `r`. Predeterminado: `0.02`.
- `gate_value`: peso de mezcla para el embedding delta FlowMap. Predeterminado: `0.25`.
- `deltatime_type`: `r` o `t-r`. Predeterminado: `r`.
- `loss_weight`: multiplicador de la pérdida ya calculada. Predeterminado: `1.0`.
- `timestep_scale`: override para modelos con escala de timestep personalizada. Déjalo sin definir normalmente.

## Límites

- Requiere un modelo flow-matching.
- Requiere timesteps escalares por muestra. Los intervalos tokenwise aún no están cableados.
- Requiere `r_timestep < timestep`; timestep cero se rechaza.
- El modo online teacher actual está pensado para LoRA/LyCORIS. Full-rank online teacher necesita cableado separado de student/teacher.
- La validación está conectada mediante el hook de scheduler del distiller AnyFlow. El scheduler activo del pipeline se proxifica, y el transformer/UNet de validación recibe el siguiente extremo del intervalo como `r_timestep` o `timestep_r`. Esto cubre los pipelines de validación registrados con soporte FlowMap; las rutas de validación custom o externas aún deben pasar el kwarg de timestep FlowMap por su cuenta.
