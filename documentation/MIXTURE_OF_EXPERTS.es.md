# Mixture-of-Experts

SimpleTuner permite dividir la tarea de entrenamiento en dos, de modo que las etapas de self-attention y cross-attention de la inferencia puedan separarse efectivamente entre dos conjuntos de pesos completamente distintos.

En este ejemplo, usaremos el esfuerzo colaborativo de SegMind con Hugging Face, [SSD-1B](https://huggingface.co/segmind/ssd-1b) para crear dos modelos nuevos que entrenan de forma más confiable y tienen mejores detalles finos resultantes que un solo modelo.

Gracias al tamaño reducido del modelo SSD-1B, es posible entrenar incluso en hardware más liviano. Como comenzamos el modelo desde sus pesos preentrenados, debemos respetar su licencia Apache 2.0, pero es relativamente sencillo. ¡Incluso puedes usar los pesos resultantes en un entorno comercial!

Cuando se introdujeron SDXL 0.9 y 1.0, ambos contenían un modelo base completo con un refiner de split-schedule.

- El modelo base fue entrenado en pasos 999 a 0
  - El modelo base tiene más de 3B parámetros y funciona de manera totalmente independiente.
- El modelo refiner fue entrenado en pasos 199 a 0
  - El modelo refiner también tiene más de 3B parámetros, un desperdicio aparentemente innecesario de recursos. No funciona por sí solo sin deformaciones sustanciales y un sesgo hacia salidas caricaturescas.

Veamos cómo podemos mejorar esta situación.


## El modelo base, "Stage One"

La primera parte de un mixture-of-experts se conoce como el modelo base. Como se mencionó en el caso de SDXL, se entrena en los 1000 timesteps, pero no es necesario. La siguiente configuración entrenará el modelo base solo en 650 pasos de los 1000 totales, ahorrando tiempo y entrenando de forma más confiable.

### Configuración de entorno

Configurar los siguientes valores en tu `config/config.env` nos permitirá comenzar:

```bash
# Ensure these aren't incorrectly set.
export USE_BITFIT=false
export USE_DORA=false
# lora could be used here instead, but the concept hasn't been explored.
export MODEL_TYPE="full"
export MODEL_FAMILY="sdxl"
export MODEL_NAME="segmind/SSD-1B"
# The original Segmind model used a learning rate of 1e-5, which is
# probably too high for whatever batch size most users can pull off.
export LEARNING_RATE=4e-7

# We really want this as high as you can tolerate.
# - If training is very slow, ensure your CHECKPOINT_STEPS and VALIDATION_STEPS
#   are set low enough that you'll get a checkpoint every couple hours.
# - The original Segmind models used a batch size of 32 with 4 accumulations.
export TRAIN_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=1

# If you are running on a beefy machine that doesn't fully utilise its VRAM during training, set this to "false" and your training will go faster.
export USE_GRADIENT_CHECKPOINTING=true

# Enable first stage model training
export TRAINER_EXTRA_ARGS="--refiner_training --refiner_training_strength=0.35 --refiner_training_invert_schedule"

# Optionally reparameterise it to v-prediction/zero-terminal SNR. 'sample' prediction_type can be used instead for x-prediction.
# This will start out looking pretty terrible until about 1500-2500 steps have passed, but it could be very worthwhile.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --prediction_type=v_prediction --rescale_betas_zero_snr --training_scheduler_timestep_spacing=trailing"
```

### Configuración del dataloader

No se requieren consideraciones especiales para la configuración del dataloader. Consulta la [guía de configuración del dataloader](DATALOADER.md) para más información.

### Validación

Actualmente, SimpleTuner no activa el modelo de segunda etapa durante las evaluaciones de la etapa uno.

Trabajo futuro dará soporte a esto como opción, en caso de que un modelo de etapa dos ya exista o se esté entrenando en paralelo.

---

## El modelo refiner, "Stage Two"

### Comparación con entrenar el refiner de SDXL

- A diferencia del refiner de SDXL, al usar Segmind SSD-1B para ambas etapas los text embeds **sí** pueden compartirse entre los dos trabajos de entrenamiento
  - El refiner de SDXL usa un layout de text embeds distinto al del modelo base SDXL.
- Los VAE embeds **sí** pueden compartirse entre trabajos de entrenamiento, igual que el refiner de SDXL. Ambos modelos usan el mismo layout de entrada.
- No se usa puntaje estético para los modelos Segmind; en su lugar usan los mismos inputs de microcondicionamiento que SDXL, p. ej., coordenadas de crop
- El entrenamiento es mucho más rápido, ya que el modelo es mucho más pequeño y los text embeds pueden reutilizarse del entrenamiento de la etapa uno

### Configuración de entorno

Actualiza los siguientes valores en tu `config/config.env` para cambiar el entrenamiento al modelo de etapa dos. Puede ser conveniente conservar una copia de la configuración del modelo base.

```bash
# Update your OUTPUT_DIR value, so that we don't overwrite the stage one model checkpoints.
export OUTPUT_DIR="/some/new/path"

# We'll swap --refiner_training_invert_schedule for --validation_using_datasets
# - Train the end of the model instead of the beginning
# - Validate using images as input for partial denoising to evaluate fine detail improvements
export TRAINER_EXTRA_ARGS="--refiner_training --refiner_training_strength=0.35 --validation_using_datasets"

# Don't update these values if you've set them on the stage one. Be sure to use the same parameterisation for both models!
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --prediction_type=v_prediction --rescale_betas_zero_snr --training_scheduler_timestep_spacing=trailing"
```

### Formato del dataset

Las imágenes deben ser de alta calidad exclusivamente: elimina cualquier dataset que encuentres cuestionable en términos de artefactos de compresión u otros errores.

Fuera de eso, se puede usar exactamente la misma configuración de dataloader entre los dos trabajos de entrenamiento.

Si quieres un dataset de demostración, [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) es una buena opción con licencia permisiva.

### Validación

El entrenamiento del refiner de etapa dos seleccionará automáticamente imágenes de cada uno de tus conjuntos de entrenamiento y las usará como inputs para denoising parcial en el tiempo de validación.

## Seguimiento de puntuaciones CLIP

Si deseas habilitar evaluaciones para puntuar el rendimiento del modelo, consulta [este documento](evaluation/CLIP_SCORES.md) para información sobre configuración e interpretación de puntuaciones CLIP.

# Pérdida de evaluación estable

Si deseas usar pérdida MSE estable para puntuar el rendimiento del modelo, consulta [este documento](evaluation/EVAL_LOSS.md) para información sobre configuración e interpretación de la pérdida de evaluación.

## Juntarlo todo en inferencia

Si quieres conectar ambos modelos para experimentar en un script simple, esto te ayudará a empezar:

```py
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, UniPCMultistepScheduler
from torch import float16, cuda
from torch.backends import mps

# For a training_refiner_strength of .35, you'll set the base model strength to 0.65.
# Formula: 1 - training_refiner_strength
training_refiner_strength = 0.35
base_model_power = 1 - training_refiner_strength
# Reduce this for lower quality but speed-up.
num_inference_steps = 40
# Update these to your local or hugging face hub paths.
stage_1_model_id = 'bghira/terminus-xl-velocity-v2'
stage_2_model_id = 'bghira/terminus-xl-refiner'
torch_device = 'cuda' if cuda.is_available() else 'mps' if mps.is_available() else 'cpu'

pipe = StableDiffusionXLPipeline.from_pretrained(stage_1_model_id, add_watermarker=False, torch_dtype=float16).to(torch_device)
pipe.scheduler = UniPCMultistepScheduler.from_pretrained(stage_1_model_id, subfolder="scheduler")
img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(stage_2_model_id).to(device=torch_device, dtype=float16)
img2img_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(stage_1_model_id, subfolder="scheduler")

prompt = "An astronaut riding a green horse"

# Important: update this to True if you reparameterised the models.
use_zsnr = True

image = pipe(
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    denoising_end=base_model_power,
    guidance_scale=9.2,
    guidance_rescale=0.7 if use_zsnr else 0.0,
    output_type="latent",
).images
image = img2img_pipe(
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    denoising_start=base_model_power,
    guidance_scale=9.2,
    guidance_rescale=0.7 if use_zsnr else 0.0,
    image=image,
).images[0]
image.save('demo.png', format="PNG")
```

Algunas experimentaciones que puedes hacer:
- Juega con valores como `base_model_power` o `num_inference_steps`, que deben ser los mismos para ambos pipelines.
- También puedes jugar con `guidance_scale`, `guidance_rescale`, que pueden configurarse de forma distinta en cada etapa. Estos afectan contraste y realismo.
- Usar prompts separados entre el modelo base y el refiner para cambiar el foco de guidance en los detalles finos.
