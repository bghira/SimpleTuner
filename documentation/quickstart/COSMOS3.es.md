## Inicio rapido de Cosmos3

Entrena un LyCORIS LoKr para NVIDIA Cosmos3.

## Notas del modelo

- `model_family`: `cosmos3`
- Flavour predeterminado: `nano`
- Flavours:
  - `nano`: `nvidia/Cosmos3-Nano`, 16B
  - `super`: `nvidia/Cosmos3-Super`, 65B
  - `super-t2i`: `nvidia/Cosmos3-Super-Text2Image`, 65B
- Cosmos3 consume IDs del tokenizer directamente en SimpleTuner.
- Los prompts positivos se convierten en captions JSON estructurados de Cosmos3 durante la tokenizacion.
- Los prompts negativos no se convierten a JSON.
- No agregues un backend `text_embeds`.
- Estos ejemplos no agregan backends `image_embeds`.
- Las muestras de imagen y video usan la cache VAE normal.
- El video con audio usa la cache VAE normal y una cache VAE de audio.
- Los datasets de accion y politica no se cubren aqui.

## Hardware

- Empieza con `model_flavour: nano`.
- Usa `mixed_precision: bf16`.
- Empieza con `base_model_precision: no_change`.
- Mantén `train_batch_size: 1`.
- Activa `gradient_checkpointing`.

Group offload opcional:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream
```

- Los streams solo son para CUDA.
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

| Ejemplo | Dataset | Medio | Backend |
| --- | --- | --- | --- |
| `cosmos3-image.lycoris-lokr` | `RareConcepts/Domokun` | imagen | `multidatabackend-cosmos3-domokun-512px.json` |
| `cosmos3-video.lycoris-lokr` | `sayakpaul/video-dataset-disney-organized` | video | `multidatabackend-cosmos3-disney-video-480p+49f.json` |
| `cosmos3-video-audio.lycoris-lokr` | `bghira/Synchronised-Drumming-Gemini3Captions` | video + audio | `multidatabackend-cosmos3-drumming-video-audio-480p+49f.json` |

## Campos requeridos

- `model_family`: `cosmos3`
- `model_flavour`: `nano`
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
- Cache de text embeds: no se usa

### Video

- Dataset: [`sayakpaul/video-dataset-disney-organized`](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized)
- Backend: `config/examples/multidatabackend-cosmos3-disney-video-480p+49f.json`
- Tipo de backend: `huggingface`
- Columnas: `video`, `prompt`
- Cache de text embeds: no se usa
- Cache de audio: no se usa

### Video con audio

- Dataset: [`bghira/Synchronised-Drumming-Gemini3Captions`](https://huggingface.co/datasets/bghira/Synchronised-Drumming-Gemini3Captions)
- Backend: `config/examples/multidatabackend-cosmos3-drumming-video-audio-480p+49f.json`
- Tipo de backend: `local`
- Archivos: videos `.mpeg` con captions `.txt` adyacentes
- Cache de text embeds: no se usa
- Cache de audio: autogenerada desde el backend de video

Descarga el dataset antes de entrenar:

```bash
huggingface-cli download \
  --repo-type=dataset \
  bghira/Synchronised-Drumming-Gemini3Captions \
  --local-dir datasets/Synchronised-Drumming-Gemini3Captions
```

El backend incluye:

```json
"audio": {
  "auto_split": true,
  "sample_rate": 16000,
  "channels": 1,
  "duration_interval": 3.0,
  "allow_zero_audio": false
}
```

SimpleTuner inyecta un dataset de audio desde ese bloque y guarda latentes de audio en una cache VAE separada.

## Ejecutar

```bash
simpletuner train example=cosmos3-image.lycoris-lokr
simpletuner train example=cosmos3-video.lycoris-lokr
simpletuner train example=cosmos3-video-audio.lycoris-lokr
```

## Validacion

- Ejemplo de imagen: `validation_resolution: 512x512`
- Ejemplos de video: `validation_resolution: 768x432`
- Ejemplos de video: `validation_num_video_frames: 49`
- La validacion de generacion de audio puede necesitar ajustes de validacion basados en dataset.

## Referencias

- [Pipeline Cosmos3 en Diffusers](https://huggingface.co/docs/diffusers/en/api/pipelines/cosmos3)
- [Coleccion NVIDIA Cosmos3](https://huggingface.co/collections/nvidia/cosmos3)
