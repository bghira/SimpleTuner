# AnyFlow

SimpleTuner implementa NVIDIA AnyFlow como dos etapas de entrenamiento explícitas para modelos de flow matching. Ambas
etapas entrenan un modelo que recibe el tiempo de flujo actual `t` y un extremo de intervalo `r`.

- `stage=forward` implementa el objetivo forward MeanFlow de NVIDIA.
- `stage=onpolicy` implementa Flow Map Backward Simulation y DMD on-policy mientras coentrena el objetivo forward.

Los modos eliminados `online_teacher` y `linear` eran objetivos específicos de SimpleTuner y ya no se aceptan.

Para un ejemplo de continuación Wan usando los checkpoints publicados por NVIDIA, consulta
[Guía rápida de continuación AnyFlow](/documentation/quickstart/ANYFLOW.es.md).


Qwen Image 2.1 admite condicionamiento de intervalos FlowMap. El [piloto Qwen de tres etapas](../quickstart/QWEN_IMAGE.es.md) mezcla captions largas de CC12M, e621 y pocos ejemplos de Domokun, seguidos de un ajuste breve con regularización. El anclaje a la predicción de la base prueba la conservación; no demuestra que Qwen recibiera destilación de guidance.

El entrenamiento AnyFlow compilado permite temporalmente al menos 32 variantes Dynamo por frame para profesor, generador, discriminador, modos de gradiente, padding y distintos batches. Se conservan los límites mayores del usuario y se restaura el límite anterior al terminar. En Torch 2.11, agotar las ocho variantes predeterminadas puede cambiar la recomputación del checkpoint a ejecución eager y fallar la comprobación de metadatos de los tensores guardados. Usa `dynamo_dynamic: true` para captions de longitud variable; el piloto Qwen ya lo habilita.

Los checkpoints de AnyFlow incluyen `anyflow_rng_state_<rank>.pt` para cada proceso. Este archivo conserva los generadores aleatorios privados usados para muestrear intervalos y realizar rollouts on-policy. La reanudación requiere la misma etapa, semilla, configuración de datos y topología distribuida. Los checkpoints antiguos sin este estado no permiten una continuación exacta y se rechazan; usa `init_lora` con un optimizador nuevo para iniciar otra ejecución desde su adapter.

`init_lora` usa por defecto el paso guardado en el adapter. Al iniciar una etapa nueva, establece `init_lora_step: 0` para que el límite de pasos y el planificador empiecen desde cero.

La exportación de adaptadores elimina los wrappers regionales del compilador de los nombres de tensores y conserva los tensores del embedding de intervalo. Los modelos normales y compilados pueden recargarlos. Con `diffusion_ratio: 1.0`, todos los intervalos tienen `r=t`, por lo que se omiten las dos predicciones de diferencias finitas cuya contribución es cero.

## Etapa forward

```json
{
  "model_type": "lora",
  "distillation_method": "anyflow",
  "distillation_config": {
    "anyflow": {
      "stage": "forward",
      "diffusion_ratio": 0.5,
      "consistency_ratio": 0.25,
      "central_difference_epsilon": 0.005,
      "meanflow_weight_type": "beta08",
      "meanflow_adaptive_weighting": true,
      "gate_value": 0.25,
      "deltatime_type": "r",
      "loss_weight": 1.0
    }
  }
}
```

Para cada batch global, la etapa forward:

1. Muestrea dos tiempos de flujo uniformes y los ordena como `t >= r`.
2. Asigna 50% de las muestras a intervalos de difusión (`r=t`), 25% a intervalos de endpoint (`r=0`) y el resto a intervalos arbitrarios.
3. Aplica el flow shift del scheduler del modelo a ambos extremos.
4. Evalúa una diferencia central a lo largo de la ruta latente recta.
5. Construye el target tangente MeanFlow y aplica el weighting normalizado `beta08` de NVIDIA.
6. Balancea cada muestra no-diffusion contra la media global de pérdida de la rama diffusion.

## Etapa on-policy

Inicia esta etapa desde un adapter AnyFlow forward mediante `init_lora` y un estado nuevo del optimizador. Reanuda checkpoints de estado de entrenamiento solo dentro de la misma etapa, configuración de datos y topología distribuida:

```json
{
  "model_type": "lora",
  "lora_type": "standard",
  "init_lora": "path-or-repo-to-forward-anyflow-adapter",
  "init_lora_step": 0,
  "learning_rate": 0.000002,
  "optimizer_beta1": 0.0,
  "optimizer_beta2": 0.999,
  "optimizer_config": "weight_decay=0.0",
  "distillation_method": "anyflow",
  "distillation_config": {
    "anyflow": {
      "stage": "onpolicy",
      "cotrain_forward": true,
      "rollout_step_counts": [2, 4, 8, 16, 50],
      "dmd_weight": 1.0,
      "dmd_batch_size": 1,
      "real_score_guidance_scale": 0.0,
      "discriminator_lr": 0.000002,
      "discriminator_betas": [0.0, 0.999],
      "discriminator_weight_decay": 0.0,
      "discriminator_grad_clip": 1.0
    }
  }
}
```

La etapa on-policy usa tres roles de score. El entrenamiento LoRA estándar comparte un transformer base congelado entre ellos:

- El adapter AnyFlow cargado es el generador.
- El modelo base con adapters desactivados es el score real congelado.
- Un adapter `anyflow_discriminator` optimizado por separado es el score fake.

Cada actualización del generador selecciona un presupuesto de rollout de `rollout_step_counts`, ejecuta un rollout
FlowMap diferenciable, añade ruido al latent generado en un tiempo uniforme desplazado y aplica el gradiente DMD
normalizado de NVIDIA. Cada actualización del discriminador ejecuta un rollout del student sin gradientes, muestrea un
tiempo desplazado logit-normal y entrena el score fake sobre el target flow normal. El adapter y optimizador del
discriminador se guardan junto a cada checkpoint de SimpleTuner como `anyflow_discriminator.safetensors` y
`anyflow_discriminator_optim.pt`.

MiniMax-H3 ya contiene destilación CFG, por lo que sus ejecuciones on-policy normalmente deben mantener
`real_score_guidance_scale=0`. Los modelos que requieren una pasada CFG externa para el score real deben cachear
embeddings de texto negativos y pueden configurar la escala explícitamente.

Cuando `--seed` está definido, AnyFlow samplea intervalos MeanFlow, schedules de rollout, latentes de rollout, ruido DMD
y sigmas DMD desde un generador Torch aislado por dispositivo. Esto mantiene estables las muestras AnyFlow cuando código
de entrenamiento no relacionado consume el RNG global de Torch. No vuelve bit-estable el backward de attention en CUDA.

## Configuración compartida

- `stage`: `forward` u `onpolicy`. Predeterminado: `forward`.
- `diffusion_ratio`: fracción del batch global que usa `r=t`. Predeterminado: `0.5`.
- `consistency_ratio`: fracción del batch global que usa `r=0`. Predeterminado: `0.25`.
- `central_difference_epsilon`: offset normalizado en tiempo desplazado. Predeterminado: `0.005`, igual a `5/1000` de NVIDIA.
- `meanflow_weight_type`: `beta08` o `uniform`. Predeterminado: `beta08`.
- `meanflow_adaptive_weighting`: balancea muestras no-diffusion contra la rama diffusion. Predeterminado: `true`.
- `gate_value`: mezcla del embedding delta-timestep FlowMap. Predeterminado: `0.25`.
- `deltatime_type`: `r` o `t-r`. Predeterminado: `r`.
- `loss_weight`: multiplicador de la pérdida forward MeanFlow. Predeterminado: `1.0`.

## Límites

- AnyFlow requiere un modelo flow-matching con conditioning de intervalo FlowMap específico del modelo.
- El entrenamiento on-policy actualmente requiere LoRA PEFT estándar. Compartir la base evita asignar copias del generador, score real y discriminador de un transformer grande en cada rank DDP.
- El entrenamiento MiniMax-H3 audio-video conjunto se rechaza. Video usa schedule shift 12 y audio usa shift 3; se necesitan targets MeanFlow y rollouts nativos de doble schedule antes de que el entrenamiento AV sea válido.
- El entrenamiento del text encoder está desactivado para todos los métodos de destilación de SimpleTuner.
- La validación usa `AnyFlowValidationScheduler`, que suministra el siguiente endpoint de intervalo a los componentes FlowMap registrados.

## Logs

El entrenamiento forward añade `anyflow_forward_loss`, valores de timestep e intervalo, y fracciones globales de rama.
El entrenamiento on-policy también añade `anyflow_dmd_loss`, `anyflow_dmd_gradient_norm`, `anyflow_dmd_sigma` y
`anyflow_rollout_steps`.
