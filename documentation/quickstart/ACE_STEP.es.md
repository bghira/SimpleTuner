# Inicio rápido de ACE-Step

En este ejemplo, entrenaremos el modelo ACE-Step v1 3.5B de generación de audio.

## Visión general

ACE-Step es un modelo flow-matching basado en transformer de 3.5B parámetros diseñado para síntesis de audio de alta calidad. Soporta generación texto-a-audio y puede condicionarse con letras.

## Requisitos de hardware

ACE-Step es un modelo de 3.5B parámetros, lo que lo hace relativamente liviano comparado con modelos grandes de generación de imágenes como Flux.

- **Mínimo:** GPU NVIDIA con 12GB+ de VRAM (p. ej., 3060, 4070).
- **Recomendado:** GPU NVIDIA con 24GB+ de VRAM (p. ej., 3090, 4090, A10G) para batch sizes mayores.
- **Mac:** Compatible vía MPS en Apple Silicon (requiere ~36GB+ de memoria unificada).

### Requisitos de almacenamiento

> ⚠️ **Advertencia de uso de disco:** El caché VAE para modelos de audio puede ser sustancial. Por ejemplo, un clip de audio de 60 segundos puede producir un archivo latente en caché de ~89MB. Esta estrategia de caché se usa para reducir drásticamente los requisitos de VRAM durante el entrenamiento. Asegúrate de tener suficiente espacio en disco para el caché de tu dataset.

> 💡 **Consejo:** Para datasets grandes, puedes usar la opción `--vae_cache_disable` para deshabilitar la escritura de embeddings a disco. Esto habilita implícitamente el caché bajo demanda, lo que ahorra espacio en disco pero aumentará el tiempo de entrenamiento y el uso de memoria porque las codificaciones se realizan durante el bucle de entrenamiento.

> 💡 **Consejo:** Usar cuantización `int8-quanto` permite entrenar en GPUs con menos VRAM (p. ej., 12GB-16GB) con mínima pérdida de calidad.

## Requisitos previos

Asegúrate de tener un entorno Python 3.10+ funcional.

```bash
pip install simpletuner
```

## Configuración

Se recomienda mantener tus configuraciones organizadas. Crearemos una carpeta dedicada para esta demo.

```bash
mkdir -p config/acestep-training-demo
```

### Ajustes críticos

Crea `config/acestep-training-demo/config.json` con estos valores:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "model_family": "ace_step",
  "model_type": "lora",
  "model_flavour": "base",
  "pretrained_model_name_or_path": "ACE-Step/ACE-Step-v1-3.5B",
  "resolution": 0,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "data_backend_config": "config/acestep-training-demo/multidatabackend.json"
}
```
</details>

### Ajustes de validación

Añade estos valores a tu `config.json` para monitorear el progreso:

- **`validation_prompt`**: Una descripción de texto del audio que quieres generar (p. ej., "A catchy pop song with upbeat drums").
- **`validation_lyrics`**: (Opcional) Letras para que el modelo cante.
- **`validation_audio_duration`**: Duración en segundos para clips de validación (predeterminado: 30.0).
- **`validation_guidance`**: Escala de guidance (predeterminado: ~3.0 - 5.0).
- **`validation_step_interval`**: Con qué frecuencia generar muestras (p. ej., cada 100 pasos).

### Funciones experimentales avanzadas

<details>
<summary>Mostrar detalles experimentales avanzados</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposición y mejora la calidad de salida dejando que el modelo genere sus propias entradas durante el entrenamiento.

> ⚠️ Estas funciones aumentan la sobrecarga computacional del entrenamiento.

</details>

## Configuración del dataset

ACE-Step requiere una configuración de dataset **específica para audio**.

### Opción 1: Dataset demo (Hugging Face)

Para un inicio rápido, puedes usar el preset preparado [ACEStep-Songs](../data_presets/preset_audio_dataset_with_lyrics.md).

Crea `config/acestep-training-demo/multidatabackend.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
[
  {
    "id": "acestep-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "Yi3852/ACEStep-Songs",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "cache_dir_vae": "cache/vae/{model_family}/acestep-demo-data"
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
</details>

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

### Opción 2: Archivos de audio locales

Crea `config/acestep-training-demo/multidatabackend.json`:

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
</details>

### Estructura de datos

Coloca tus archivos de audio en `datasets/my_audio_files`. SimpleTuner soporta una amplia gama de formatos, incluyendo:

- **Sin pérdida:** `.wav`, `.flac`, `.aiff`, `.alac`
- **Con pérdida:** `.mp3`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.opus`

> ℹ️ **Nota:** Para soportar formatos como MP3, AAC y WMA, debes tener **FFmpeg** instalado en tu sistema.

Para captions y letras, coloca los archivos de texto correspondientes junto a tus archivos de audio:

- **Audio:** `track_01.wav`
- **Caption (Prompt):** `track_01.txt` (contiene la descripción de texto, p. ej., "A slow jazz ballad")
- **Lyrics (Opcional):** `track_01.lyrics` (contiene el texto de las letras)

<details>
<summary>Ejemplo de layout del dataset</summary>

```text
datasets/my_audio_files/
├── track_01.wav
├── track_01.txt
└── track_01.lyrics
```
</details>

> 💡 **Avanzado:** Si tu dataset usa una convención de nombres diferente (p. ej., `_lyrics.txt`), puedes personalizar esto en tu configuración del dataset.

<details>
<summary>Ver ejemplo de nombre personalizado para lyrics</summary>

```json
"audio": {
  "lyrics_filename_format": "{filename}_lyrics.txt"
}
```
</details>

> ⚠️ **Nota sobre lyrics:** Si no se encuentra un archivo `.lyrics` para una muestra, los embeddings de letras se rellenan con ceros. ACE-Step espera condicionamiento con letras; entrenar fuertemente con datos sin letras (instrumentales) puede requerir más pasos para que el modelo aprenda a generar audio instrumental de alta calidad con entradas de letras en cero.

## Entrenamiento

Inicia el entrenamiento especificando tu entorno:

```bash
simpletuner train env=acestep-training-demo
```

Este comando le dice a SimpleTuner que busque `config.json` dentro de `config/acestep-training-demo/`.

> 💡 **Consejo (Continuar entrenamiento):** Para continuar el fine-tuning desde una LoRA existente (p. ej., los checkpoints oficiales de ACE-Step o adaptadores de la comunidad), usa la opción `--init_lora`:
> ```bash
> simpletuner train env=acestep-training-demo --init_lora=/path/to/existing_lora.safetensors
> ```

### Entrenar el embedder de letras (estilo upstream)

El trainer upstream de ACE-Step ajusta el embedder de letras junto con el denoiser. Para reflejar ese comportamiento en SimpleTuner (solo full o LoRA estándar):

- Habilítalo: `lyrics_embedder_train: true`
- Overrides opcionales (de lo contrario se reutilizan el optimizador/scheduler principal):
  - `lyrics_embedder_lr`
  - `lyrics_embedder_optimizer`
  - `lyrics_embedder_lr_scheduler`

Fragmento de ejemplo:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "lyrics_embedder_train": true,
  "lyrics_embedder_lr": 5e-5,
  "lyrics_embedder_optimizer": "torch-adamw",
  "lyrics_embedder_lr_scheduler": "cosine_with_restarts"
}
```
</details>
Los pesos del embedder se guardan con los saves de LoRA y se restauran al reanudar.

## Solución de problemas

- **Errores de validación:** Asegúrate de no usar funciones de validación centradas en imagen como `num_validation_images` > 1 (mapeado conceptualmente a batch size para audio) o métricas basadas en imágenes (puntuación CLIP).
- **Problemas de memoria:** Si tienes OOM, intenta reducir `train_batch_size` o habilitar `gradient_checkpointing`.

## Migrar desde el trainer upstream

Si vienes de los scripts originales de entrenamiento de ACE-Step, aquí tienes cómo se mapean los parámetros a `config.json` de SimpleTuner:

| Parámetro upstream | SimpleTuner `config.json` | Predeterminado / Notas |
| :--- | :--- | :--- |
| `--learning_rate` | `learning_rate` | `1e-4` |
| `--num_workers` | `dataloader_num_workers` | `8` |
| `--max_steps` | `max_train_steps` | `2000000` |
| `--every_n_train_steps` | `checkpointing_steps` | `2000` |
| `--precision` | `mixed_precision` | `"fp16"` o `"bf16"` (usa `"no"` para fp32) |
| `--accumulate_grad_batches` | `gradient_accumulation_steps` | `1` |
| `--gradient_clip_val` | `max_grad_norm` | `0.5` |
| `--shift` | `flow_schedule_shift` | `3.0` (Específico de ACE-Step) |

### Convertir datos sin procesar

Si tienes archivos de audio/texto/letras sin procesar y quieres usar el formato de dataset de Hugging Face (como usa la herramienta upstream `convert2hf_dataset.py`), puedes usar el dataset resultante directamente en SimpleTuner.

El convertidor upstream produce un dataset con columnas `tags` y `norm_lyrics`. Para usarlas, configura tu backend así:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "path/to/converted/dataset",
    "config": {
        "audio_caption_fields": ["tags"],
        "lyrics_column": "norm_lyrics"
    }
}
```
</details>
