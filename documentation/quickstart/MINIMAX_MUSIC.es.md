# Guía rápida de MiniMax Music 3

Esta guía configura SimpleTuner para entrenar LoRA de MiniMax Music 3.

## Resumen

MiniMax Music 3 es un generador musical condicionado por descripción y letras. El layout Diffusers usa un modelo Qwen3 autoregresivo para el condicionamiento de texto/audio, un transformer flow-matching sobre latents DAV de 128 canales y un decoder/vocoder para audio de validación.

SimpleTuner admite:

- entrenamiento LoRA, LyCORIS y full-rank del transformer
- VAECache desde audio bruto mediante el autoencoder original `dav.pth`
- caption, lyrics y duration desde metadatos de datasets de audio
- validación con `validation_prompt`, `validation_lyrics`, `validation_audio_duration` y bibliotecas de prompts
- importación/exportación de LoRA ComfyUI MiniMax Music con `lora_format: "comfyui"`
- AnyFlow, TwinFlow, CREPA self-flow y LayerSync

## Requisitos de hardware

MiniMax Music 3 tiene un flow transformer de 2.4B y un modelo Qwen3 AR de 8B para condicionamiento.

- **Mínimo:** GPU NVIDIA con 24GB+ de VRAM para LoRA conservador.
- **Recomendado:** 48GB+ de VRAM, u offload a CPU/RAM para ranks mayores, clips más largos y validación frecuente.
- **Mac:** MPS puede funcionar para algunas partes, pero CUDA es el objetivo práctico para entrenamiento y validación.

Empieza con `base_model_precision: "int8-quanto"`, `text_encoder_1_precision: "int8-quanto"` y `gradient_checkpointing: true`. Si el text encoder sigue siendo el cuello de botella, usa offload del text encoder antes de subir el LoRA rank.

## Requisitos previos

Instala SimpleTuner y FFmpeg para cargar audio:

```bash
pip install simpletuner
```

Para instalación manual o entorno de desarrollo, consulta la [documentación de instalación](../INSTALL.md).

## Configuración

Crea una carpeta dedicada:

```bash
mkdir -p config/minimaxmusic-training-demo
```

Crea `config/minimaxmusic-training-demo/config.json`:

<details>
<summary>Ver configuración de ejemplo</summary>

```json
{
  "model_family": "minimaxmusic",
  "model_type": "lora",
  "model_flavour": "music3",
  "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-Music3",
  "pretrained_vae_model_name_or_path": "SimpleTuner/MiniMax-Music-3-Encoder",
  "resolution": 512,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "int8-quanto",
  "gradient_checkpointing": true,
  "lora_rank": 64,
  "lora_format": "comfyui",
  "optimizer": "adamw_bf16",
  "learning_rate": 0.00005,
  "train_batch_size": 1,
  "vae_batch_size": 1,
  "data_backend_config": "config/minimaxmusic-training-demo/multidatabackend.json",
  "validation_prompt": "bright synth pop with clean vocal melody and crisp percussion",
  "validation_lyrics": "[verse]\nturning sparks into a skyline\n[chorus]\nwe keep singing through the night",
  "validation_audio_duration": 30,
  "validation_guidance": 1.7,
  "validation_num_inference_steps": 30,
  "validation_steps": 50,
  "validation_disable_unconditional": true
}
```
</details>

Plantillas listas:

- `simpletuner/examples/minimaxmusic-music3.peft-lora`
- `simpletuner/examples/minimaxmusic-audio.json`
- `simpletuner/examples/minimaxmusic-prompts.json`

Ejecuta el ejemplo:

```bash
simpletuner train example=minimaxmusic-music3.peft-lora
```

## VAECache

El cache de audio bruto de MiniMax Music 3 usa el autoencoder DAV. El repositorio VAE recomendado de SimpleTuner es `SimpleTuner/MiniMax-Music-3-Encoder`, con el componente convertido en `audio_vae/` para carga estilo Diffusers.

El repositorio upstream `MiniMaxAI/MiniMax-Music3` también incluye el `dav.pth` original, y SimpleTuner puede cargarlo directamente. Si usas un directorio Diffusers local convertido, conserva `dav.pth` en la raíz del checkpoint o apunta `pretrained_vae_model_name_or_path` a una ruta o repositorio Hub que contenga `dav.pth` o `audio_vae/`. Un subdirectorio `vocoder/` basta para decode de validación, pero no para VAE caching de audio bruto.

## Dataset

MiniMax Music 3 requiere un dataset **audio** y un backend de cache **text embeds**.

```json
[
  {
    "id": "minimaxmusic-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "Yi3852/ACEStep-Songs",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "audio": {
      "bucket_strategy": "duration",
      "duration_interval": 3.0,
      "max_duration_seconds": 30
    },
    "cache_dir_vae": "cache/vae/{model_family}/minimaxmusic-demo-data"
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/{model_family}"
  }
]
```

Para archivos locales, usa `.txt` para la descripción y `.lyrics` para las letras:

```text
datasets/minimaxmusic-audio/
├── track_01.wav
├── track_01.txt
└── track_01.lyrics
```

## Validación

- **`validation_prompt`**: descripción musical o tags.
- **`validation_lyrics`**: letras cantadas. Usa string vacío para instrumental.
- **`validation_audio_duration`**: duración del clip en segundos.
- **`validation_guidance`**: escala CFG. Empieza cerca de `1.5` a `2.0`.
- **`validation_num_inference_steps`**: pasos de sampling. Empieza cerca de `30`.
- **`validation_steps`**: frecuencia de renderizado de validación.
- **`validation_prompt_library`**: usa `"audio"` para la biblioteca integrada de caption + lyrics musical.
- **`user_prompt_library`**: ruta a una biblioteca JSON. Las entradas pueden usar `prompt` o `caption`, y `lyrics` multilinea opcional.

## Entrenamiento

```bash
simpletuner train env=minimaxmusic-training-demo
```

Para comenzar desde un LoRA MiniMax Music 3 existente:

```bash
simpletuner train env=minimaxmusic-training-demo --init_lora=/path/to/adapter.safetensors --init_lora_step=0
```

Si el adapter está en formato ComfyUI nativo, conserva `lora_format: "comfyui"` en la configuración. SimpleTuner lo convertirá durante el entrenamiento y exportará en el mismo formato.

## Funciones avanzadas

MiniMax Music 3 usa el flujo de entrenamiento flow-matching de SimpleTuner, así que AnyFlow, TwinFlow, CREPA self-flow y LayerSync están disponibles. Empieza con LoRA estándar y activa una función avanzada por vez.

## Solución de problemas

- **`VAE caching requires the original dav.pth checkpoint`**: usa `SimpleTuner/MiniMax-Music-3-Encoder` o `MiniMaxAI/MiniMax-Music3`, conserva `dav.pth` en la raíz del checkpoint local, o apunta `pretrained_vae_model_name_or_path` a una ubicación que lo contenga.
- **Lyrics ausentes**: confirma que los metadatos incluyen `lyrics`, o coloca archivos `.lyrics` junto al audio al usar `caption_strategy: "textfile"`.
- **OOM en text embedding o validación**: reduce `validation_audio_duration`, usa int8 para el text encoder o habilita offload del text encoder.
