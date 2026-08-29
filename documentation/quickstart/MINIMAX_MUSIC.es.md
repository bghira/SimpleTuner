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

Para expandir una identidad vocal objetivo entre estilos o géneros, configura el workflow RVC `data_transforms` descrito en [Voice Cloning Data Transforms](../experimental/VOICE_CLONING.es.md).

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

## Entrenamiento del modelo de lenguaje (etapa AR)

El modelo de lenguaje Qwen3 que planifica los códigos semánticos de MiniMax Music 3 puede entrenarse en lugar del DiT musical — útil para palabras disparadoras estilo dreambooth que vinculan un estilo musical a una palabra clave.

Consulta [fiona crapple](https://huggingface.co/terminusresearch/minimax-music3-lm-lora-fiona-crapple) para ver un ejemplo completo de entrenamiento de LM LoRA producido con este modo, con su configuración, checkpoints y comparaciones de audio.

```json
{
  "minimax_music_train_component": "language_model",
  "minimax_music_lm_max_frames": 0,
  "minimax_music_lm_window_mode": "prefix"
}
```

Requisitos y diferencias respecto al entrenamiento del DiT:

- Cada muestra del dataset debe proporcionar `prompt` (o `tags`), `lyrics` y un metadato `audio_tokens_path` que apunte a un archivo `.pt` con códigos RVQ crudos por codebook con forma `[frames, codebooks]` (códigos semánticos `< 16384`, residuales `< audio_vocab_size`, sin offsets de vocabulario). Expórtalos con `precompute_rvq_codes.py --raw-codes` desde el repositorio dedicado `minimax-music3-latent-replanner`.
- La pérdida es entropía cruzada de siguiente token sobre el codebook semántico, enmascarada a las posiciones de audio; el depth decoder RVQ permanece congelado y aporta los embeddings de entrada de los códigos residuales.
- Solo se admite LoRA PEFT estándar y `lora_format: "comfyui"` se rechaza. Los checkpoints guardan `pytorch_lora_weights.safetensors` con claves de adaptador con prefijo `language_model.`.
- El audio de validación dentro del entrenador está deshabilitado en este modo; renderiza desde los checkpoints guardados con la pila de generación estándar.
- En este modo no hay caché de VAE ni de embeddings de texto — el entrenamiento lee los tokens directamente, así que `cache_dir_vae` y los backends de text embeds no se usan.
- Coloca tu palabra clave (p. ej. `"fiona crapple"`) en el campo caption/`prompt` de cada muestra; mantén las letras sin modificar.
- Para ejecuciones cortas con límite de frames, usa `minimax_music_lm_window_mode: "random"` para muestrear ventanas RVQ posicionadas en vez de entrenar siempre intros. Las ventanas aleatorias agregan inicio/fin/duración al prompt y omiten la letra completa salvo que la muestra proporcione `lyrics_window`.
- Para entrenar la estructura de canciones, usa `minimax_music_lm_window_mode: "continuation"`. Los últimos `minimax_music_lm_target_frames` reciben pérdida y los frames visibles anteriores quedan como contexto causal enmascarado. Los recortes `full` empiezan en el inicio de la canción; los `random` pueden desplazarse por la pista conservando al menos un segmento nativo de 128 frames. Las duraciones se ajustan al intervalo nativo de 128 frames/5,12 segundos; un máximo de `0` usa la pista disponible.

Ejemplo de continuación con prefijo completo y límite de memoria:

```json
{
  "minimax_music_lm_window_mode": "continuation",
  "minimax_music_lm_target_frames": 128,
  "minimax_music_lm_continuation_crop_mode": "full",
  "minimax_music_lm_min_duration_seconds": 5.12,
  "minimax_music_lm_max_duration_seconds": 30.72
}
```

Cambia el modo a `random` para entrenar continuaciones posicionadas con el mismo límite. Esos recortes añaden su rango temporal al prompt y omiten las letras completas salvo que exista `lyrics_window`. Cuando son posibles tramos terminales y no terminales, un 25% fijo alcanza el final real para que la supervisión EOS no dependa de la longitud de la pista. El muestreo ocurre durante el collate LM sobre la secuencia RVQ completa almacenada; no modifica el audio ni la caché del dataset.
- **Preservación de prior**: añade un segundo backend de audio con `is_regularisation_data: true` que contenga canciones no relacionadas (se permiten letras vacías). En esos lotes la pérdida apunta a la distribución de siguiente token del modelo base congelado en lugar de los códigos reales, de modo que el LoRA se mantiene quirúrgico: los captions no relacionados siguen prediciendo exactamente como lo haría el modelo base, lo que reduce notablemente el sangrado de estilo.

## Solución de problemas

- **`VAE caching requires the original dav.pth checkpoint`**: usa `SimpleTuner/MiniMax-Music-3-Encoder` o `MiniMaxAI/MiniMax-Music3`, conserva `dav.pth` en la raíz del checkpoint local, o apunta `pretrained_vae_model_name_or_path` a una ubicación que lo contenga.
- **Lyrics ausentes**: confirma que los metadatos incluyen `lyrics`, o coloca archivos `.lyrics` junto al audio al usar `caption_strategy: "textfile"`.
- **OOM en text embedding o validación**: reduce `validation_audio_duration`, usa int8 para el text encoder o habilita offload del text encoder.

## Experimentos relacionados con MiniMax Music 3

- [Encoders RVQ abiertos](https://huggingface.co/SimpleTuner/open-rvq-encoder-minimax-music3)
- [Integración de audio de referencia RVQ](https://github.com/bghira/minimax-music3-rvq-reference-audio)
- [LoRA del LM Fiona Crapple](https://huggingface.co/terminusresearch/minimax-music3-lm-lora-fiona-crapple)
- [Refinador latente](https://github.com/bghira/minimax-music3-latent-refiner) y [pesos v0.10](https://huggingface.co/terminusresearch/minimax-music3-latent-refiner-v0.10)
- [Replanificador latente](https://github.com/bghira/minimax-music3-latent-replanner) y [registro experimental](https://huggingface.co/terminusresearch/minimax-music3-replanner-experiment)
