# ERNIE-Image [base / turbo] Guía Rápida

Esta guía cubre el entrenamiento de una LoRA para ERNIE-Image. ERNIE-Image es la familia single-stream flow-matching de Baidu, y SimpleTuner soporta los sabores `base` y `turbo`.

## Requisitos de hardware

ERNIE no es un modelo pequeño. Planea algo parecido a otros transformers single-stream grandes:

- el objetivo más realista es una GPU de 24 GB o más usando cuantización int8 + LoRA en bf16
- 16 GB puede funcionar con offload agresivo y RamTorch, pero será más lento
- multi-GPU, FSDP2 y offload adicional hacia CPU/RAM también ayudan

No se recomienda entrenar en Apple GPU.

## Instalación

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

Consulta también la [documentación de instalación](../INSTALL.md).

## Configuración del entorno

### WebUI

```bash
simpletuner server
```

Después selecciona la familia ERNIE en el asistente de entrenamiento.

### Línea de comandos

La forma más simple es usar el ejemplo incluido:

- config de ejemplo: `simpletuner/examples/ernie.peft-lora/config.json`
- entorno local ejecutable: `config/ernie-example/config.json`

Ejecución:

```bash
simpletuner train --env ernie-example
```

Si lo configuras manualmente:

- `model_type`: `lora`
- `model_family`: `ernie`
- `model_flavour`: `base` o `turbo`
- `pretrained_model_name_or_path`:
  - `base`: `baidu/ERNIE-Image`
  - `turbo`: `baidu/ERNIE-Image-Turbo`
- `resolution`: empieza con `512`
- `train_batch_size`: `1`
- `ramtorch`: `true`
- `ramtorch_text_encoder`: `true`
- `gradient_checkpointing`: `true`

El ejemplo usa:

- `max_train_steps: 100`
- `optimizer: optimi-lion`
- `learning_rate: 1e-4`
- `validation_guidance: 4.0`
- `validation_num_inference_steps: 20`

### Assistant LoRA para Turbo

ERNIE Turbo ya expone soporte para assistant LoRA, pero todavía no tiene una ruta predeterminada de adaptador.

- flavour soportado: `turbo`
- nombre de peso por defecto: `pytorch_lora_weights.safetensors`
- valor que debes aportar: `assistant_lora_path`

Si tienes un assistant adapter propio:

```json
{
  "assistant_lora_path": "your-org/your-ernie-turbo-assistant-lora",
  "assistant_lora_weight_name": "pytorch_lora_weights.safetensors"
}
```

Si no quieres usarlo:

```json
{
  "disable_assistant_lora": true
}
```

### Dataset y captions

El ejemplo usa:

- `dataset_name`: `RareConcepts/Domokun`
- `caption_strategy`: `instanceprompt`
- `instance_prompt`: `🟫`

Eso sirve como smoke test, pero ERNIE suele responder mejor a texto real que a un trigger de un solo token. Para entrenamiento real, usa captions más descriptivos.

### Funciones avanzadas

ERNIE también soporta:

- TREAD
- LayerSync
- captura de hidden states estilo REPA / CREPA
- carga de assistant LoRA para turbo

Primero asegúrate de que el entrenamiento base funciona, y luego añade extras.
