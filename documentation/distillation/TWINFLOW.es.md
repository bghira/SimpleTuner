# Entrenamiento TwinFlow (RCGM) de pocos pasos

TwinFlow es una receta ligera y autónoma de pocos pasos basada en **recursive consistency gradient matching (RCGM)**. **No forma parte de las opciones principales de `distillation_method`**—se activa directamente con flags `twinflow_*`. El loader deja `twinflow_enabled` en `false` por defecto en configs obtenidas del hub para que las configs transformer vanilla queden intactas.

TwinFlow en SimpleTuner:
* Solo flow-matching a menos que enlaces explícitamente modelos de difusión con `diff2flow_enabled` + `twinflow_allow_diff2flow`.
* Teacher EMA por defecto; captura/restauración de RNG **siempre activada** alrededor de pases teacher/CFG para reflejar la ejecución TwinFlow de referencia.
* Embeddings de signo opcionales para semántica de tiempo negativo están cableados en transformers pero solo se usan cuando `twinflow_enabled` es true; las configs HF sin flag evitan cambios de comportamiento.
* Las pérdidas por defecto usan RCGM + real-velocity; opcionalmente habilita entrenamiento auto-adversarial completo con `twinflow_adversarial_enabled: true` para las pérdidas L_adv y L_rectify. Espera generación de 1–4 pasos con guidance `0.0`.
* El logging en W&B puede emitir un scatter experimental de trayectoria TwinFlow (teoría no verificada) para depuración.

---

## Configuración rápida (modelo flow-matching)

Añade los bits de TwinFlow a tu configuración habitual (deja `distillation_method` sin definir/nulo):

```json
{
  "model_family": "sd3",
  "model_type": "lora",
  "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3.5-large",
  "output_dir": "output/sd3-twinflow",

  "distillation_method": null,
  "use_ema": true,

  "twinflow_enabled": true,
  "twinflow_target_step_count": 2,
  "twinflow_estimate_order": 2,
  "twinflow_enhanced_ratio": 0.5,
  "twinflow_delta_t": 0.01,
  "twinflow_target_clamp": 1.0,

  "learning_rate": 1e-4,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "mixed_precision": "bf16",
  "validation_guidance": 0.0,
  "validation_num_inference_steps": 2
}
```

Para modelos de difusión (epsilon/v prediction) activa explícitamente:

```json
{
  "prediction_type": "epsilon",
  "diff2flow_enabled": true,
  "twinflow_allow_diff2flow": true
}
```

> Por defecto, TwinFlow usa las pérdidas RCGM + real-velocity. Habilita `twinflow_adversarial_enabled: true` para entrenamiento auto-adversarial completo con pérdidas L_adv y L_rectify (sin necesidad de discriminador externo).

---

## Qué esperar (datos del paper)

De arXiv:2512.05150 (texto del PDF):
* Los benchmarks de inferencia se midieron en una **sola A100 (BF16)** con throughput (batch=10) y latencia (batch=1) a 1024×1024. No se incluyen números exactos en el texto, solo el hardware.
* Una **comparación de memoria GPU** (1024×1024) para Qwen-Image-20B (LoRA) y SANA-1.6B muestra que TwinFlow entra donde DMD2 / SANA-Sprint pueden hacer OOM.
* Las configs de entrenamiento (Tabla 6) listan **batch sizes 128/64/32/24** y **pasos de entrenamiento 30k–60k (o 7k–10k en ejecuciones más cortas)**; LR constante, decay de EMA a menudo 0.99.
* El PDF **no** reporta totales de GPU, layouts de nodo o tiempo de pared.

Trata esto como expectativas orientativas, no garantías. Para hardware/runtime exacto necesitarías confirmación del autor.

---

## Opciones clave

* `twinflow_enabled`: Activa la pérdida auxiliar RCGM; mantén `distillation_method` vacío y scheduled sampling deshabilitado. Por defecto `false` si falta en la config.
* `twinflow_target_step_count` (1–4 recomendado): Guía el entrenamiento y se reutiliza para validación/inferencia. Guidance se fuerza a `0.0` porque CFG está integrado.
* `twinflow_estimate_order`: Orden de integración para el rollout RCGM (predeterminado 2). Valores más altos añaden pases del teacher.
* `twinflow_enhanced_ratio`: Refinamiento opcional estilo CFG de objetivos a partir de predicciones cond/uncond del teacher (0.5 por defecto; 0.0 para desactivar). Usa RNG capturado para mantener cond/uncond alineados.
* `twinflow_delta_t` / `twinflow_target_clamp`: Dan forma al objetivo recursivo; los valores por defecto reflejan los ajustes estables del paper.
* `use_ema` + `twinflow_require_ema` (por defecto true): Los pesos EMA se usan como teacher. Configura `twinflow_allow_no_ema_teacher: true` solo si aceptas calidad de estudiante como teacher.
* `twinflow_allow_diff2flow`: Habilita el puente para modelos epsilon/v-prediction cuando `diff2flow_enabled` también es true.
* Captura/restauración de RNG: Siempre habilitado para reflejar la implementación TwinFlow de referencia en pases teacher/CFG consistentes. No hay opción para desactivarlo.
* Embeddings de signo: Cuando `twinflow_enabled` es true, los modelos pasan `twinflow_time_sign` a transformers que soportan `timestep_sign`; de lo contrario no se usa embedding extra.

### Rama adversarial (TwinFlow completo)

Habilita el entrenamiento auto-adversarial del paper original para mejor calidad:

* `twinflow_adversarial_enabled` (por defecto false): Habilita las pérdidas L_adv y L_rectify. Usan tiempo negativo para entrenar una trayectoria "fake", permitiendo coincidencia de distribuciones sin discriminadores externos.
* `twinflow_adversarial_weight` (por defecto 1.0): Multiplicador de peso para la pérdida adversarial (L_adv).
* `twinflow_rectify_weight` (por defecto 1.0): Multiplicador de peso para la pérdida de rectificación (L_rectify).

Cuando se habilita, el entrenamiento genera muestras fake mediante generación de un paso, luego entrena ambas:
- **L_adv**: Pérdida de velocidad fake con tiempo negativo—enseña al modelo a mapear muestras fake de vuelta a ruido.
- **L_rectify**: Pérdida de coincidencia de distribución—alinea predicciones de trayectorias real y fake para caminos más rectos.

---

## Flujo de entrenamiento y validación

1. Entrena como en una ejecución flow-matching normal (no se necesita distiller). EMA debe existir salvo que optes explícitamente por desactivarla; la alineación de RNG es automática.
2. La validación sustituye automáticamente el **scheduler TwinFlow/UCGM** y usa `twinflow_target_step_count` pasos con `guidance_scale=0.0`.
3. Para pipelines exportados, adjunta el scheduler manualmente:

```python
from simpletuner.helpers.training.custom_schedule import TwinFlowScheduler

pipe = ...  # your loaded diffusers pipeline
pipe.scheduler = TwinFlowScheduler(num_train_timesteps=1000, prediction_type="flow_matching", shift=1.0)
pipe.scheduler.set_timesteps(num_inference_steps=2, device=pipe.device)
result = pipe(prompt="A cinematic portrait, 35mm", guidance_scale=0.0, num_inference_steps=2).images
```

---

## Logging

* Cuando `report_to=wandb` y `twinflow_enabled=true`, el trainer puede registrar un scatter experimental de trayectoria TwinFlow (σ vs tt vs sign). El gráfico es solo para depuración y se etiqueta en la UI como “experimental/theory unverified”.

---

## Solución de problemas

* **Error sobre flow-matching**: TwinFlow requiere `prediction_type=flow_matching` a menos que habilites `diff2flow_enabled` + `twinflow_allow_diff2flow`.
* **EMA requerida**: Habilita `use_ema` o configura `twinflow_allow_no_ema_teacher: true` / `twinflow_require_ema: false` si aceptas el fallback de estudiante como teacher.
* **Calidad plana a 1 paso**: Prueba `twinflow_target_step_count: 2`–`4`, mantén guidance en `0.0` y reduce `twinflow_enhanced_ratio` si hay sobreajuste.
* **Deriva teacher/estudiante**: La alineación RNG siempre está habilitada; la deriva debería venir de desajuste del modelo, no de diferencias estocásticas. Si tu transformer no tiene `timestep_sign`, deja `twinflow_enabled` apagado o actualiza el modelo para consumirlo antes de habilitar TwinFlow.
