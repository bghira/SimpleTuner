# Guía rápida de zlab i1

Esta guía cubre el entrenamiento de LoRA para [zlab-princeton i1](https://huggingface.co/zlab-princeton/i1-3B). i1 es un transformer flow-matching de 3B publicado con receta JAX/TPU y pesos de inferencia PyTorch. SimpleTuner lo entrena con una integración PyTorch nativa y usa una conversión Diffusers safetensors en [`bghira/zlab-i1-diffusers`](https://huggingface.co/bghira/zlab-i1-diffusers).

i1 usa el VAE de FLUX.2, un text encoder T5Gemma, latentes de 32 canales y una caption nula aprendida para CFG.

## Requisitos de hardware

Para entrenamiento LoRA a 1024px, empieza con:

- una GPU moderna de 24G con cuantización int8 para LoRAs pequeñas
- 40G+ para trabajar con más margen
- multi-GPU si subes rank, dataset o reduces cuantización

Los ejemplos usan `int8-quanto`, `bf16`, `gradient_checkpointing=true` y `train_batch_size=1`. CUDA es la ruta esperada; Apple GPU no es recomendable.

## Ejemplos incluidos

```bash
simpletuner train example=zlab-i1.peft-lora
simpletuner train example=zlab-i1.lycoris-lokr
```

Empieza con PEFT LoRA. Usa LyCORIS LoKr cuando quieras esa factorización en lugar de un LoRA estándar.

## Ajustes clave

```json
{
  "model_type": "lora",
  "model_family": "zlab_i1",
  "model_flavour": "3b",
  "pretrained_vae_model_name_or_path": "black-forest-labs/FLUX.2-dev",
  "validation_resolution": "1024x1024",
  "validation_guidance": 12.0,
  "validation_guidance_rescale": 0.7,
  "validation_num_inference_steps": 250
}
```

El flavour `3b` resuelve a `bghira/zlab-i1-diffusers`, con el transformer en el subfolder Diffusers estándar `transformer/` y pesos safetensors. Solo define `pretrained_transformer_model_name_or_path` si pruebas una conversión propia.

## Validación

La validación funciona con el pipeline nativo de i1. Para una prueba rápida:

```bash
simpletuner train example=zlab-i1.peft-lora validation_num_inference_steps=4 num_eval_images=1
```

Cuatro pasos solo comprueban que el pipeline genera y guarda imágenes. Usa 250 pasos antes de evaluar calidad.

## Funciones avanzadas

i1 usa las rutas comunes de transformers de SimpleTuner:

- TwinFlow funciona en modo flow-matching nativo. El timestep de i1 se ignora como en upstream, así que TwinFlow modifica la trayectoria latente ruidosa y el objetivo, no añade un embedding temporal nuevo.
- CREPA Self-Flow y LayerSync usan el buffer de hidden states de tokens de imagen. Configura los índices de bloque contra las 29 capas transformer de i1.
- TREAD enruta solo tokens de imagen. Los tokens de texto permanecen intactos para conservar la semántica de la máscara de T5Gemma.
- La validación acepta CFG Zero*, salto de CFG con `validation_no_cfg_until_timestep` y skip-layer guidance con `validation_guidance_skip_layers`.
- RamTorch, Musubi block swap y VAE tiling están soportados. Mantén RamTorch y Musubi como opciones mutuamente excluyentes.

## Dataset

i1 necesita cachés VAE propias porque espera latentes de 32 canales del VAE de FLUX.2. No reutilices cachés de SDXL, Flux.1, PixArt u otras familias.

```json
[
  {
    "id": "my-i1-dataset",
    "type": "local",
    "instance_data_dir": "/datasets/my-subject",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/zlab_i1/my-i1-dataset"
  }
]
```

Empieza sin cambiar el ejemplo PEFT, verifica benchmark base, pérdidas finitas, imagen de validación y `pytorch_lora_weights.safetensors`; luego cambia dataset y prompts.
