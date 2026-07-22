# Inicio rapido de Cosmos3

Entrena un LyCORIS LoKr para NVIDIA Cosmos3.

## Notas del modelo

- `model_family`: `cosmos3`
- `model_flavour` predeterminado: `nano`
- Flavours admitidos:

| Flavour | Modelo Hub | Notas |
| --- | --- | --- |
| `edge` | `nvidia/Cosmos3-Edge` | modelo omni edge de 4B |
| `nano` | `nvidia/Cosmos3-Nano` | modelo omni 16B |
| `super` | `nvidia/Cosmos3-Super` | modelo omni 65B |
| `super-t2i` | `nvidia/Cosmos3-Super-Text2Image` | modelo texto-a-imagen 65B |
| `super-i2v` | `nvidia/Cosmos3-Super-Image2Video` | modelo imagen-a-video 65B, video sin audio |

- Cosmos3 usa IDs del tokenizer directamente.
- Los prompts positivos se convierten a captions JSON de Cosmos3 durante la tokenizacion.
- Los prompts negativos no se convierten a JSON.
- No agregues un backend `text_embeds`.
- Imagen y video usan la cache VAE normal.
- Video + audio usa la cache VAE normal y una cache VAE de audio.
- `super-i2v` requiere latentes de condicionamiento.
- Datasets de accion y politica no se cubren aqui.

## Componentes

SimpleTuner usa componentes Cosmos3 separados por defecto:

| Flavour | Reasoner | Generator |
| --- | --- | --- |
| `edge` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-edge` | `SimpleTuner/cosmos3-component-generation-layers-bf16-edge` |
| `nano` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-nano` | `SimpleTuner/cosmos3-component-generation-layers-bf16-nano` |
| `super` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-super` | `SimpleTuner/cosmos3-component-generation-layers-bf16-super` |
| `super-t2i` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-super-t2i` | `SimpleTuner/cosmos3-component-generation-layers-bf16-super-t2i` |
| `super-i2v` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-super-i2v` | `SimpleTuner/cosmos3-component-generation-layers-bf16-super-i2v` |

- Mantén `cosmos3_reasoner_component: auto`.
- Mantén `cosmos3_generator_component: auto`.
- Las salidas del reasoner se cachean por la ruta de text embeds.
- `text_cache_disable: true` vuelve a ejecutar el reasoner congelado durante el entrenamiento.

## Hardware

- Empieza con `model_flavour: nano`.
- Usa `mixed_precision: bf16`.
- Empieza con `base_model_precision: no_change`.
- Mantén `train_batch_size: 1`.
- Activa `gradient_checkpointing`.
- Usa los componentes generator separados para reducir memoria al cargar el transformer.

Group offload opcional:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream
```

- Streams solo son CUDA.
- No lo combines con `--enable_model_cpu_offload`.
- Agrega `--group_offload_to_disk_path /fast-ssd/simpletuner-offload` cuando falte RAM del sistema.

## Instalacion

```bash
pip install 'simpletuner[cuda]'

# Usuarios CUDA 13 / Blackwell
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

Instalacion de desarrollo:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Configs de ejemplo

| Ejemplo | Flavour | Dataset | Medio | Backend |
| --- | --- | --- | --- | --- |
| `cosmos3-image.lycoris-lokr` | `nano` | `RareConcepts/Domokun` | imagen | `multidatabackend-cosmos3-domokun-512px.json` |
| `cosmos3-image-48g.lycoris-lokr` | `nano` | `RareConcepts/Domokun` | imagen, ajustado para 48 GB | `multidatabackend-cosmos3-domokun-1024-arb.json` |
| `cosmos3-image-80g.lycoris-lokr` | `nano` | `RareConcepts/Domokun` | imagen, ajustado para 80 GB | `multidatabackend-cosmos3-domokun-1024-arb.json` |
| `cosmos3-video.lycoris-lokr` | `nano` | `sayakpaul/video-dataset-disney-organized` | video | `multidatabackend-cosmos3-disney-video-480p+49f.json` |
| `cosmos3-video-audio.lycoris-lokr` | `nano` | `bghira/Synchronised-Drumming-Gemini3Captions` | video + audio | `multidatabackend-cosmos3-drumming-video-audio-480p+49f.json` |
| `cosmos3-super-i2v.lycoris-lokr` | `super-i2v` | `sayakpaul/video-dataset-disney-organized` | imagen-a-video | `multidatabackend-cosmos3-disney-i2v-480p+49f.json` |

Los ejemplos de imagen `48g` y `80g` son variantes de la receta nano image LoKr ajustadas por tamaño de memoria. Ambos usan el backend 1024px aspect-ratio. La config `48g` mantiene gradient checkpointing con `gradient_checkpointing_interval: 2`; la config `80g` desactiva gradient checkpointing y habilita `flash-attn-3-hub`.

## Campos requeridos

- `model_family`: `cosmos3`
- `model_type`: `lora`
- `lora_type`: `lycoris`
- `base_model_precision`: `no_change`
- `mixed_precision`: `bf16`
- `train_batch_size`: `1`
- `gradient_checkpointing`: `true`

## Notas de datasets

### Imagen

- Dataset: [`RareConcepts/Domokun`](https://huggingface.co/datasets/RareConcepts/Domokun)
- Backend: `config/examples/multidatabackend-cosmos3-domokun-512px.json`
- Tipo de backend: `huggingface`
- Estrategia de captions: `instanceprompt`
- Cache de text embeds: solo cache del reasoner
- Cache de audio: no se usa

### Video

- Dataset: [`sayakpaul/video-dataset-disney-organized`](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized)
- Backend: `config/examples/multidatabackend-cosmos3-disney-video-480p+49f.json`
- Tipo de backend: `huggingface`
- Columnas: `video`, `prompt`
- Cache de text embeds: solo cache del reasoner
- Cache de audio: no se usa

### Video con audio

- Dataset: [`bghira/Synchronised-Drumming-Gemini3Captions`](https://huggingface.co/datasets/bghira/Synchronised-Drumming-Gemini3Captions)
- Backend: `config/examples/multidatabackend-cosmos3-drumming-video-audio-480p+49f.json`
- Tipo de backend: `local`
- Archivos: videos `.mpeg` con captions `.txt` adyacentes
- Cache de text embeds: solo cache del reasoner
- Cache de audio: generada desde el backend de video

Descarga el dataset antes de entrenar:

```bash
huggingface-cli download \
  --repo-type=dataset \
  bghira/Synchronised-Drumming-Gemini3Captions \
  --local-dir datasets/Synchronised-Drumming-Gemini3Captions
```

Bloque de audio:

```json
"audio": {
  "auto_split": true,
  "sample_rate": 16000,
  "channels": 1,
  "duration_interval": 3.0,
  "allow_zero_audio": false
}
```

### Super-I2V

- Dataset: [`sayakpaul/video-dataset-disney-organized`](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized)
- Backend: `config/examples/multidatabackend-cosmos3-disney-i2v-480p+49f.json`
- Tipo de backend: `huggingface`
- Columnas: `video`, `prompt`
- `video.is_i2v`: `true`
- Tipo de condicionamiento: `reference_strict`
- Latentes de condicionamiento: requeridos
- Cache de audio: no se usa

El backend I2V marca el dataset de video con:

```json
"video": {
  "num_frames": 49,
  "min_frames": 49,
  "is_i2v": true
}
```

SimpleTuner crea el backend de referencia estricta desde esta marca.

## Ejecutar

```bash
simpletuner train example=cosmos3-image.lycoris-lokr
simpletuner train example=cosmos3-image-48g.lycoris-lokr
simpletuner train example=cosmos3-image-80g.lycoris-lokr
simpletuner train example=cosmos3-video.lycoris-lokr
simpletuner train example=cosmos3-video-audio.lycoris-lokr
simpletuner train example=cosmos3-super-i2v.lycoris-lokr
```

## Validacion

- Ejemplo de imagen: `validation_resolution: 512x512`
- Ejemplos de video: `validation_resolution: 768x432`
- Ejemplos de video: `validation_num_video_frames: 49`
- Ejemplo Super-I2V: usa entradas de validacion de condicionamiento.

## Referencias

- [Pipeline Cosmos3 en Diffusers](https://huggingface.co/docs/diffusers/en/api/pipelines/cosmos3)
- [Coleccion NVIDIA Cosmos3](https://huggingface.co/collections/nvidia/cosmos3)
