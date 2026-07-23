# Guía rápida de Mage-Flow

Esta guía cubre el entrenamiento LoRA de Mage-Flow en SimpleTuner. Mage-Flow es la familia rectified-flow de Microsoft para generación y edición de imágenes: un transformer MMDiT de 4B, acondicionamiento Qwen3-VL y Mage-VAE con latentes de 128 canales y reducción 16x.

## Hardware

Mage-Flow es más pequeño que Flux.1 y Qwen-Image, pero sigue cargando un transformer grande y un text encoder Qwen3-VL congelado.

Puntos de partida:

- `bf16`, 512px, batch 1 para pruebas rápidas
- `bf16`, 1024px, batch 1 para LoRA normal en GPUs grandes
- `int8-torchao` o NF4 si falta VRAM
- flavours Turbo con 4 pasos de validación

Recomendado: 24GB como mínimo práctico para pruebas reducidas o cuantizadas, 48GB para trabajo cómodo a 1024px y 80GB para edit training o batches mayores.

## Configuración

Instala SimpleTuner:

```bash
pip install 'simpletuner[cuda]'
```

Mage-Flow usa atencion empaquetada de longitud variable. Para usar FlashAttention 2 sin compilar el paquete `flash-attn` localmente, define `"attention_mechanism": "flash-attn-varlen-hub"` para que SimpleTuner cargue el kernel desde Hugging Face Hub. Deja el valor predeterminado `diffusers` para PyTorch SDPA.

Config inicial para texto a imagen:

```json
{
  "model_family": "mageflow",
  "model_flavour": "base",
  "model_type": "lora",
  "pretrained_model_name_or_path": "microsoft/Mage-Flow-Base",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "optimizer": "optimi-lion",
  "learning_rate": 1e-4,
  "lora_rank": 32,
  "train_batch_size": 1,
  "resolution": 1024,
  "validation_resolution": "1024x1024",
  "validation_num_inference_steps": 30,
  "validation_guidance": 5.0
}
```

Flavours disponibles:

- `base` - `microsoft/Mage-Flow-Base`
- `default` - `microsoft/Mage-Flow`
- `turbo` - `microsoft/Mage-Flow-Turbo`
- `edit-base` - `microsoft/Mage-Flow-Edit-Base`
- `edit` - `microsoft/Mage-Flow-Edit`
- `edit-turbo` - `microsoft/Mage-Flow-Edit-Turbo`

Para edición:

```json
{
  "model_family": "mageflow",
  "model_flavour": "edit-turbo",
  "pretrained_model_name_or_path": "microsoft/Mage-Flow-Edit-Turbo",
  "validation_num_inference_steps": 4
}
```

## Mage Flow (Edit) Considerations

Los checkpoints Mage-Flow edit no requieren un dataset de condicionamiento o referencia. Microsoft entreno los modelos edit conjuntamente en tareas de generacion y edicion, asi que el prior generativo se conserva. En SimpleTuner puedes seguir usando un dataset normal de imagenes para LoRA de sujeto, estilo o concepto aunque `model_flavour` sea `edit-base`, `edit` o `edit-turbo`.

Usa pares source/target solo cuando quieras entrenar comportamiento de edicion. SimpleTuner usa automaticamente la pipeline compatible con edicion; cuando no se proporciona imagen de condicionamiento, la validacion y el prompt encoding usan el flujo text-to-image.

## Dataloader

Para subject/style LoRA usa el dataloader de imágenes normal con cache separado:

```json
[
  {
    "id": "dreambooth-1024",
    "type": "local",
    "instance_data_dir": "/path/to/images",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square",
    "resolution": 1024,
    "resolution_type": "pixel",
    "metadata_backend": "discovery",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "cache_dir_vae": "cache/vae/mageflow/dreambooth-1024"
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/mageflow"
  }
]
```

Para entrenar comportamiento de edicion opcional, usa pares source/target. El caption debe ser la instruccion de edicion, no solo una descripcion de la imagen final.

## Presets de memoria

Mage-Flow incluye presets de RAMTorch y Musubi block swap en el menu de optimizacion de memoria. Usa RAMTorch para mantener pesos del transformer en CPU RAM; usa Musubi block swap para transmitir solo los ultimos bloques durante forward y backward. Son opciones mutuamente excluyentes en el configurador.

## Validación y cuantización

Usa 20 pasos para `default`, 30 para `base` y 4 para `turbo` / `edit-turbo`.

```json
{
  "base_model_precision": "int8-torchao",
  "quantize_via": "cpu"
}
```

SimpleTuner incluye una copia vendorizada del código MIT de Mage-Flow y lo envuelve en pipelines nativas de Diffusers para validación.
