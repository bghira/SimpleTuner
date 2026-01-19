# Inicio rapido de HeartMuLa

En este ejemplo, entrenaremos el modelo HeartMuLa oss 3B de generacion de audio.

## Vision general

HeartMuLa es un transformador autoregresivo de 3B parametros que predice tokens de audio discretos a partir de etiquetas y letras. Los tokens se decodifican con HeartCodec para producir formas de onda.

## Requisitos de hardware

HeartMuLa es un modelo de 3B parametros, lo que lo hace relativamente liviano comparado con modelos grandes de generacion de imagenes como Flux.

- **Minimo:** GPU NVIDIA con 12GB+ de VRAM (p. ej., 3060, 4070).
- **Recomendado:** GPU NVIDIA con 24GB+ de VRAM (p. ej., 3090, 4090, A10G) para batch sizes mayores.
- **Mac:** Compatible via MPS en Apple Silicon (requiere ~36GB+ de memoria unificada).

### Requisitos de almacenamiento

> ⚠️ **Advertencia sobre dataset de tokens:** HeartMuLa se entrena con tokens de audio precomputados. SimpleTuner no genera tokens durante el entrenamiento, asi que tu dataset debe proporcionar metadatos `audio_tokens` o `audio_tokens_path`. Los archivos de tokens pueden ser grandes, asi que planifica el espacio en disco.

> 💡 **Consejo:** Usar cuantizacion `int8-quanto` permite entrenar en GPUs con menos VRAM (p. ej., 12GB-16GB) con minima perdida de calidad.

## Requisitos previos

Asegurate de tener un entorno Python 3.10+ funcional.

```bash
pip install simpletuner
```

## Configuracion

Se recomienda mantener tus configuraciones organizadas. Crearemos una carpeta dedicada para esta demo.

```bash
mkdir -p config/heartmula-training-demo
```

### Ajustes criticos

Crea `config/heartmula-training-demo/config.json` con estos valores:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "model_family": "heartmula",
  "model_type": "lora",
  "model_flavour": "3b",
  "pretrained_model_name_or_path": "HeartMuLa/HeartMuLa-oss-3B",
  "resolution": 0,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "data_backend_config": "config/heartmula-training-demo/multidatabackend.json"
}
```
</details>

### Ajustes de validacion

Anade estos valores a tu `config.json` para monitorear el progreso:

- **`validation_prompt`**: Etiquetas o una descripcion del audio (p. ej., "Pop upbeat con sintetizadores brillantes").
- **`validation_lyrics`**: (Opcional) Letras para que el modelo cante. Usa una cadena vacia para instrumentales.
- **`validation_audio_duration`**: Duracion en segundos para clips de validacion (predeterminado: 30.0).
- **`validation_guidance`**: Escala de guidance (empieza alrededor de 1.5 - 3.0).
- **`validation_step_interval`**: Con que frecuencia generar muestras (p. ej., cada 100 pasos).

### Funciones experimentales avanzadas

<details>
<summary>Mostrar detalles experimentales avanzados</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposicion y mejora la calidad de salida dejando que el modelo genere sus propias entradas durante el entrenamiento.

> ⚠️ Estas funciones aumentan la sobrecarga computacional del entrenamiento.

</details>

## Configuracion del dataset

HeartMuLa requiere un dataset **especifico para audio** con tokens precomputados.

Cada muestra debe proporcionar:

- `tags` (cadena)
- `lyrics` (cadena; puede estar vacia)
- `audio_tokens` o `audio_tokens_path`

Los arrays de tokens deben ser 2D con forma `[frames, num_codebooks]` o `[num_codebooks, frames]`.

> 💡 **Nota:** HeartMuLa no usa un codificador de texto separado, asi que no se requiere un backend de text-embeds.

### Opcion 1: Dataset de Hugging Face (tokens en columnas)

Crea `config/heartmula-training-demo/multidatabackend.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
[
  {
    "id": "heartmula-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "your-org/heartmula-audio-tokens",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "config": {
      "audio_caption_fields": ["tags"],
      "lyrics_column": "lyrics"
    }
  }
]
```
</details>

Asegurate de que tu dataset incluya columnas `audio_tokens` o `audio_tokens_path` junto con los campos de texto.

### Opcion 2: Archivos de audio locales + metadatos de tokens

Crea `config/heartmula-training-demo/multidatabackend.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
[
  {
    "id": "my-audio-dataset",
    "type": "local",
    "dataset_type": "audio",
    "instance_data_dir": "datasets/my_audio_files",
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "disabled": false
  }
]
```
</details>

Asegurate de que tu backend de metadatos suministre `audio_tokens` o `audio_tokens_path` para cada muestra.

### Estructura de datos

Coloca tus archivos de audio en `datasets/my_audio_files`. SimpleTuner soporta una amplia gama de formatos incluyendo:

- **Sin perdida:** `.wav`, `.flac`, `.aiff`, `.alac`
- **Con perdida:** `.mp3`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.opus`

> ℹ️ **Nota:** Para soportar formatos como MP3, AAC y WMA, debes tener **FFmpeg** instalado en tu sistema.

Para etiquetas y letras, coloca archivos de texto correspondientes junto a tus archivos de audio si usas `caption_strategy: textfile`:

- **Audio:** `track_01.wav`
- **Etiquetas (Prompt):** `track_01.txt` (Contiene la descripcion de texto, p. ej., "Una balada de jazz lenta")
- **Letras (Opcional):** `track_01.lyrics` (Contiene el texto de las letras)

Proporciona los arrays de tokens mediante metadatos (por ejemplo, entradas `audio_tokens_path` que apunten a archivos `.npy` o `.npz`).

<details>
<summary>Ejemplo de estructura del dataset</summary>

```text
datasets/my_audio_files/
├── track_01.wav
├── track_01.txt
├── track_01.lyrics
└── track_01.tokens.npy
```
</details>

> ⚠️ **Nota sobre letras:** HeartMuLa espera una cadena de letras para cada muestra. Para datos instrumentales, proporciona una cadena vacia en lugar de omitir el campo.

## Entrenamiento

Inicia el entrenamiento especificando tu entorno:

```bash
simpletuner train env=heartmula-training-demo
```

Este comando le dice a SimpleTuner que busque `config.json` dentro de `config/heartmula-training-demo/`.

> 💡 **Consejo (Continuar entrenamiento):** Para continuar un fine-tuning desde una LoRA existente, usa la opcion `--init_lora`:
> ```bash
> simpletuner train env=heartmula-training-demo --init_lora=/path/to/existing_lora.safetensors
> ```

## Solucion de problemas

- **Errores de validacion:** Asegurate de no intentar usar funciones de validacion centradas en imagenes como `num_validation_images` > 1 (mapeado conceptualmente al batch size para audio) o metricas basadas en imagen (puntaje CLIP).
- **Problemas de memoria:** Si te quedas sin memoria, intenta reducir `train_batch_size` o habilitar `gradient_checkpointing`.
