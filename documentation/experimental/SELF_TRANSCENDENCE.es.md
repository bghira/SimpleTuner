# Self-Transcendence

Self-Transcendence entrena bloques superficiales de un Transformer de difusión con objetivos internos, sin encoder visual externo. La implementación sigue el método de dos etapas de [Sun et al.](https://arxiv.org/abs/2601.07773).

Se aplica a familias de difusión de imagen, vídeo y audio que exponen estados de tokens latentes. No admite UNets, modelos autorregresivos ni LyCORIS. Admite entrenamiento completo y PEFT LoRA estándar.

## Etapa 1: guía estructural VAE

La etapa 1 proyecta un bloque superficial al objetivo de difusión de la familia del modelo en el espacio latente VAE: velocidad de flujo, epsilon, predicción v o muestra limpia. El objetivo se divide en parches sobre la cuadrícula de tokens del modelo sin descartar valores.

```json
{
  "distillation_method": "self_transcendence",
  "distillation_config": {"self_transcendence": {
    "stage": "vae", "student_block": 8, "weight": 0.5,
    "timestep_min": 0.4, "timestep_max": 0.7,
    "projector_hidden_dim": 2048
  }}
}
```

Guarda el adaptador o checkpoint para usarlo como profesor fijo en la etapa 2.

## Etapa 2: representación autoguiada

El profesor fijo procesa la misma entrada ruidosa con el texto y con el prompt vacío almacenado. El CFG en el espacio de características combina los estados profundos y supervisa el bloque superficial de un alumno nuevo.

En PEFT LoRA, crea un adaptador alumno nuevo y configura `teacher_adapter_path` con el safetensors de la etapa 1:

```json
{
  "distillation_method": "self_transcendence",
  "distillation_config": {"self_transcendence": {
    "stage": "self", "student_block": 8, "teacher_block": 16,
    "teacher_adapter_path": "output/stage1/pytorch_lora_weights.safetensors",
    "cfg_scale": 30.0, "weight": 0.5,
    "timestep_min": 0.4, "timestep_max": 0.7,
    "stop_step": 5000, "projector_hidden_dim": 2048
  }}
}
```

Profesor y alumno deben usar el mismo modelo base, rank PEFT y módulos objetivo. Sin `teacher_adapter_path`, la etapa 2 captura los parámetros entrenables presentes tras reanudar. Esto permite entrenamiento completo y pruebas de una etapa, pero no reproduce el alumno nuevo del artículo.

Los índices de bloque empiezan en cero. Empieza con el alumno cerca de 1/3 de la profundidad y el profesor cerca de 2/3. Después de `stop_step` se omiten los forwards del profesor y se mantiene una ruta de proyección con peso cero para DDP. Los embeddings del prompt vacío se almacenan automáticamente.

Se registran `self_transcendence/loss`, `self_transcendence/weight` y `self_transcendence/teacher_cfg_scale` en la etapa 2. No puede combinarse con otro destilador ni con entrenamiento del encoder de texto.
