# Inicio rápido de Boogu-Image 0.1

Esta guía cubre el entrenamiento LoRA y LyCORIS LoKr para Boogu-Image 0.1 en SimpleTuner. Boogu-Image es un modelo de imágenes con flow matching que incluye variantes de texto a imagen, turbo y edición. La integración de SimpleTuner usa código local de pipeline y transformer, y los pipelines exportados están alojados en Hugging Face bajo el namespace `SimpleTuner`.

Configs iniciales incluidas:

```bash
simpletuner/examples/boogu-image-v0.1.peft-lora/config.json
simpletuner/examples/boogu-image-v0.1.lycoris-lokr/config.json
```

## Requisitos de hardware

Trata Boogu-Image como un modelo transformer de imagen grande. Para las primeras pruebas, usa entrenamiento a 1024px, batch size 1, precisión mixta bf16 y gradient checkpointing.

Puntos de partida recomendados:

- **Valor predeterminado:** `v0.1-base`, pesos LoRA bf16, rank 16.
- **Menos VRAM:** usa una variante FP8 como `v0.1-base-fp8`, `v0.1-turbo-fp8` o `v0.1-edit-fp8`.
- **Validación/inferencia rápida:** usa la variante turbo, teniendo en cuenta el estado del assistant LoRA.
- **Edición:** usa `v0.1-edit` o `v0.1-edit-fp8` con datos condicionados por pares.

El uso de memoria depende del rank, optimizador, resolución de validación, offload, compile y si se usan pesos FP8. Un solo H100 puede entrenar el ejemplo PEFT LoRA incluido durante 1000 pasos a 1024px con muestras de benchmark y validación activadas.

En GPUs más pequeñas, empieza con pesos FP8, rank 8-16, `train_batch_size=1`, gradient checkpointing y model/group offload.

### Offload de memoria

El offload agrupado de módulos puede reducir la presión de VRAM cuando los pesos del transformer son el cuello de botella:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream
```

Offload opcional a disco:

```bash
--group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Los streams solo son efectivos en CUDA; SimpleTuner los desactiva en ROCm, MPS y CPU.
- No combines group offload con otras estrategias de CPU offload.
- Prefiere NVMe local rápido al usar offload a disco.

### Torch compile y atención

En GPUs NVIDIA, usa los alias de atención de kernels de Hugging Face Hub cuando estén disponibles:

```json
{
  "attention_mechanism": "flash-attn-3-hub",
  "dynamo_backend": "inductor",
  "dynamo_use_regional_compilation": true
}
```

Si la validación compilada produce imágenes negras en una combinación concreta de GPU o driver, desactiva primero torch compile y vuelve a probar antes de cambiar la receta de entrenamiento.

## Instalación

Instala SimpleTuner con pip:

```bash
pip install 'simpletuner[cuda]'

# Usuarios de CUDA 13 / Blackwell
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

Para instalación manual o setup de desarrollo, consulta la [documentación de instalación](../INSTALL.md).

## Configurar el entorno

### Método de interfaz web

La WebUI de SimpleTuner puede crear una configuración de entrenamiento para Boogu-Image:

```bash
simpletuner server
```

Abre http://localhost:8001 y elige `boogu_image` como familia de modelo.

### Método manual / línea de comandos

Copia `config/config.json.example` a `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Revisa estos valores:

- `model_type` - `lora`.
- `lora_type` - `standard` para PEFT LoRA o `lycoris` para LyCORIS LoKr.
- `model_family` - `boogu_image`.
- `model_flavour` - `v0.1-base`, `v0.1-base-fp8`, `v0.1-turbo`, `v0.1-turbo-fp8`, `v0.1-edit` o `v0.1-edit-fp8`.
- `pretrained_model_name_or_path` - normalmente déjalo sin configurar para que el flavour seleccione el pipeline `SimpleTuner/Boogu-Image-0.1-*`.
- `output_dir` - directorio para checkpoints e imágenes de validación.
- `train_batch_size` - empieza con `1`.
- `resolution` - empieza con `1024`.
- `resolution_type` - usa `pixel_area` para buckets multiaspecto.
- `validation_resolution` - usa `1024x1024`; puedes separar varios tamaños con comas.
- `validation_guidance` - empieza alrededor de `4.0` para base/edit.
- `validation_num_inference_steps` - empieza alrededor de `30`; turbo puede usar menos pasos.
- `mixed_precision` - usa `bf16` en GPUs NVIDIA modernas.
- `gradient_checkpointing` - mantenlo activado.
- `flow_schedule_shift` - los ejemplos usan `3`.

Config PEFT LoRA mínima:

```json
{
  "model_type": "lora",
  "model_family": "boogu_image",
  "model_flavour": "v0.1-base",
  "lora_type": "standard",
  "lora_rank": 16,
  "lora_alpha": 16,
  "output_dir": "output/models-boogu-image-v0.1",
  "train_batch_size": 1,
  "validation_resolution": "1024x1024",
  "validation_guidance": 4.0,
  "validation_num_inference_steps": 30,
  "validation_prompt": "a polished product photo of a ceramic mug on a walnut desk",
  "validation_steps": 50,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "flow_schedule_shift": 3,
  "optimizer": "adamw_bf16",
  "learning_rate": 1e-4,
  "lr_scheduler": "constant_with_warmup",
  "lr_warmup_steps": 10,
  "max_train_steps": 1000,
  "resolution": 1024,
  "resolution_type": "pixel_area",
  "data_backend_config": "config/examples/multidatabackend-small-dreambooth-1024px.json"
}
```

## Ejecutar los ejemplos

```bash
simpletuner train example=boogu-image-v0.1.peft-lora
simpletuner train example=boogu-image-v0.1.lycoris-lokr
```

Forma de checkout de desarrollo:

```bash
simpletuner train env=examples/boogu-image-v0.1.peft-lora
simpletuner train env=examples/boogu-image-v0.1.lycoris-lokr
```

## Variantes FP8

Usa los flavours `-fp8` para cargar los pesos FP8 exportados:

```json
{
  "model_family": "boogu_image",
  "model_flavour": "v0.1-base-fp8"
}
```

Lo mismo aplica a `v0.1-turbo-fp8` y `v0.1-edit-fp8`. No necesitas apuntar SimpleTuner a archivos `.bin` de Boogu.

## Assistant LoRA de Turbo

SimpleTuner activa la ruta de assistant LoRA para `v0.1-turbo` y `v0.1-turbo-fp8`. El path del adaptador actualmente es un placeholder `None` porque todavía no hay un adaptador separado publicado para esta integración.

Hasta que exista ese adaptador, usa turbo como pipeline exportado y valida la calidad directamente. Para el baseline más predecible, empieza con `v0.1-base`.

## Entrenamiento de edición

Los flavours de edición requieren datos condicionados por pares. Usa la misma estructura de dataset de referencia por pares descrita en el [quickstart de Qwen Image Edit](./QWEN_EDIT.md).

Para LoRA texto-a-imagen, usa los flavours base o turbo.

## Prompts de validación

`validation_prompt` es el prompt principal de validación. Para más cobertura, añade una librería:

```json
{
  "product": "a polished product photo of <token> on a walnut desk",
  "studio": "a clean studio portrait of <token> with softbox lighting",
  "cinematic": "a cinematic scene featuring <token>, detailed lighting, shallow depth of field"
}
```

Y apúntala desde la configuración:

```json
{
  "validation_prompt_library": "config/user_prompt_library.json"
}
```

Usa prompts suficientemente distintos para detectar sobreajuste, colapso de prompt y deriva de estilo.

## Inferencia

Después de entrenar, carga el adaptador guardado con el mismo flavour de pipeline. El archivo principal suele ser:

```bash
output/models-boogu-image-v0.1/pytorch_lora_weights.safetensors
```
