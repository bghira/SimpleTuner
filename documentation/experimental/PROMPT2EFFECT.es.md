# Prompt2Effect

Prompt2Effect es un flujo experimental solo por CLI para entrenar una hiperrred que genera pesos PEFT LoRA a partir de un prompt de efecto. Esta separado del entrenador normal de denoising de imagen/video de SimpleTuner.

La distincion importante es que Prompt2Effect no hace que entrenar la hiperrred tarde 3.3 segundos. Mueve el trabajo costoso a una etapa unica de entrenamiento sobre una biblioteca de LoRAs de efectos ya existentes. Despues de tener esa hiperrred, generar un nuevo LoRA desde un prompt es una sola pasada forward.

## Que Entrena

Las muestras de entrenamiento son checkpoints LoRA existentes, no archivos multimedia:

- un prompt de efecto
- un checkpoint PEFT LoRA para ese efecto
- un modelo base fijo y un esquema fijo de capas objetivo

El paso de preparacion convierte cada actualizacion LoRA en factores canonicos por SVD. La perdida de entrenamiento es MSE normalizado sobre esos factores LoRA canonicos, no una perdida de difusion sobre latentes.

## Familias Soportadas

Los scripts soportan actualmente:

- `ltxvideo2`
- sabores I2V de `wan`
- `hunyuanvideo`

El artefacto generado es un archivo normal `pytorch_lora_weights.safetensors` con claves PEFT `lora_A`, `lora_B` y `alpha`.

## Archivos

Prompt2Effect vive en `scripts/prompt2effect/`:

- `prepare.py`: valida un manifiesto de LoRAs y escribe objetivos canonicos por SVD.
- `train.py`: entrena la hiperrred Prompt2Effect.
- `generate.py`: emite un PEFT LoRA desde una hiperrred entrenada y un prompt de efecto.

Esto no esta expuesto en la WebUI.

## Manifiesto

Crea un archivo JSONL con un LoRA de efecto por linea:

```json
{"id":"blue_mood","effect_prompt":"blue mood cinematic atmosphere","lora_path":"/path/to/pytorch_lora_weights.safetensors"}
```

Todos los LoRAs de una ejecucion de Prompt2Effect deben usar el mismo esquema de modulos objetivo y las mismas dimensiones de entrada/salida. Usa `--rank` durante la preparacion para elegir el rango LoRA canonico/generado; si se omite, se usa el rango del primer LoRA.

## Preparar Objetivos

```bash
.venv/bin/python scripts/prompt2effect/prepare.py \
  --manifest /path/to/effects.jsonl \
  --output_dir cache/prompt2effect/wan-i2v-targets \
  --model_family wan \
  --model_flavour i2v-14b-2.1
```

Opciones utiles:

- `--model_family`: `ltxvideo2`, `wan` o `hunyuanvideo`.
- `--base_model`: sobrescribe el repo o ruta local del modelo base.
- `--model_flavour`: usa un valor conocido de la familia cuando no se da `--base_model`.
- `--target_modules`: sufijos PEFT separados por comas, `default` o `all-linear`.
- `--rank`: rango del LoRA generado. Por defecto usa el rango del primer LoRA fuente.
- `--component_subfolder`: subcarpeta del componente del modelo base. Por defecto usa la subcarpeta transformer de la familia.

`prepare.py` escribe:

- `schema.json`
- `targets.safetensors`

Falla si un LoRA no tiene modulos requeridos, contiene modulos inesperados o no coincide con las formas de tensores del modelo base.

## Entrenar

```bash
.venv/bin/python scripts/prompt2effect/train.py \
  --prepared_dir cache/prompt2effect/wan-i2v-targets \
  --output_dir output/prompt2effect/wan-i2v \
  --text_encoder_model google/t5-v1_1-base \
  --max_train_steps 10000
```

El codificador de texto esta congelado y solo codifica prompts de efecto. Los pesos del modelo base tambien estan congelados y se usan como condicionamiento estructural para la hiperrred.

Por defecto, los pesos base permanecen en CPU. Usa `--base_weights_device training` solo cuando las capas objetivo seleccionadas caben en el acelerador.

## Generar Un LoRA

```bash
.venv/bin/python scripts/prompt2effect/generate.py \
  --checkpoint output/prompt2effect/wan-i2v/prompt2effect_hypernetwork.pt \
  --prompt "blue mood cinematic atmosphere" \
  --output output/blue_mood_prompt2effect
```

El directorio de salida contendra `pytorch_lora_weights.safetensors`. Cargalo como cualquier otro PEFT LoRA de SimpleTuner/Diffusers.

## Limites

- Solo PEFT LoRA lineal. LyCORIS, LoRA convolucional, vectores de magnitud DoRA y tensores sidecar arbitrarios aun no estan soportados en este flujo.
- Una hiperrred esta ligada a una familia de modelo, forma del modelo base, esquema de modulos objetivo y rango.
- Los scripts no estan integrados con Accelerate, la WebUI ni el gestor principal de checkpoints de SimpleTuner.
- La calidad de entrenamiento depende de la cantidad y diversidad de LoRAs de efecto fuente. Unos pocos LoRAs bastan para probar la ruta, no para esperar generalizacion.
- Los LoRAs generados deben validarse normalmente antes de publicarlos o usarlos en flujos de produccion.
