# Guía rápida de Kontext [dev] Mini

> 📝  Kontext comparte el 90 % del flujo de entrenamiento con Flux, así que este archivo solo enumera lo que *difiere*. Cuando un paso **no** se menciona aquí, sigue las [instrucciones](../quickstart/FLUX.md) originales.


---

## 1. Resumen del modelo

|                                                  | Flux‑dev               | Kontext‑dev                                 |
| ------------------------------------------------ | -------------------    | ------------------------------------------- |
| Licencia                                         | No comercial           | No comercial                                |
| Guidance                                         | Destilado (CFG ≈ 1)    | Destilado (CFG ≈ 1)                         |
| Variantes disponibles                            | *dev*, schnell, [pro]   | *dev*, [pro, max]                          |
| Longitud de secuencia T5                          | 512 dev, 256 schnell   | 512 dev                                     |
| Tiempo típico de inferencia 1024 px<br>(4090 @ CFG 1)  | ≈ 20 s                  | **≈ 80 s**                                  |
| VRAM para LoRA 1024 px @ int8‑quanto               | 18 G                   | **24 G**                                    |

Kontext mantiene el backbone del transformer de Flux pero introduce **condicionamiento con referencia emparejada**.

Hay dos modos `conditioning_type` disponibles para Kontext:

* `conditioning_type=reference_loose` (✅ estable) – la referencia puede diferir en relación de aspecto/tamaño de la edición.
  - Ambos datasets se escanean para metadatos, se agrupan por aspecto y se recortan de forma independiente, lo que puede aumentar sustancialmente el tiempo de arranque.
  - Esto puede ser un problema si deseas garantizar la alineación de las imágenes de edición y referencia, como en un dataloader que usa una sola imagen por nombre de archivo.
* `conditioning_type=reference_strict` (✅ estable) – la referencia se pretransforma exactamente como el recorte de edición.
  - Así debes configurar tus datasets si necesitas una alineación perfecta entre recortes / aspect bucketing entre tus imágenes de edición y referencia.
  - Antes requería `--vae_cache_ondemand` y algo más de uso de VRAM, pero ya no.
  - Duplica los metadatos de recorte / aspect bucket del dataset fuente al inicio, para que no tengas que hacerlo.

Para definiciones de campos, consulta [`conditioning_type`](../DATALOADER.md#conditioning_type) y [`conditioning_data`](../DATALOADER.md#conditioning_data). Para controlar cómo se muestrean múltiples conjuntos de condicionamiento, usa `conditioning_multidataset_sampling` como se describe en [OPTIONS](../OPTIONS.md#--conditioning_multidataset_sampling).


---

## 2. Requisitos de hardware

* **RAM del sistema**: la cuantización todavía requiere 50 GB.
* **GPU**: una 3090 (24 G) es el mínimo realista para entrenamiento a 1024 px **con int8‑quanto**.
  * Los sistemas Hopper H100/H200 con Flash Attention 3 pueden habilitar `--fuse_qkv_projections` para acelerar mucho el entrenamiento.
  * Si entrenas a 512 px puedes encajar en una tarjeta de 12 G, pero espera batches lentos (la longitud de secuencia sigue siendo grande).


---

## 3. Diferencia rápida de configuración

A continuación está el conjunto *mínimo* de cambios que necesitas en `config/config.json` en comparación con tu configuración típica de entrenamiento de Flux.

<details>
<summary>Ver ejemplo de config</summary>

```jsonc
{
  "model_family":   "flux",
  "model_flavour": "kontext",                       // <‑‑ cambia esto de "dev" a "kontext"
  "base_model_precision": "int8-quanto",            // cabe en 24 G a 1024 px
  "gradient_checkpointing": true,
  "fuse_qkv_projections": false,                    // <‑‑ úsalo para acelerar el entrenamiento en sistemas Hopper H100/H200. ADVERTENCIA: requiere flash-attn instalado manualmente.
  "lora_rank": 16,
  "learning_rate": 1e-5,
  "optimizer": "optimi-lion",                       // <‑‑ usa Lion para resultados más rápidos, y adamw_bf16 para resultados más lentos pero posiblemente más estables.
  "max_train_steps": 10000,
  "validation_guidance": 2.5,                       // <‑‑ kontext funciona mejor con guidance 2.5
  "validation_resolution": "1024x1024",
  "conditioning_multidataset_sampling": "random"    // <-- configurar esto en "combined" cuando tienes dos datasets de condicionamiento definidos hará que se muestren simultáneamente en lugar de alternar.
}
```
</details>

### Funciones experimentales avanzadas

<details>
<summary>Mostrar detalles experimentales avanzados</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposición y mejora la calidad de la salida al permitir que el modelo genere sus propias entradas durante el entrenamiento.

> ⚠️ Estas funciones aumentan la sobrecarga computacional del entrenamiento.

</details>

### Fragmento de dataloader (multi‑data‑backend)

Si has curado manualmente un dataset de pares de imágenes, puedes configurarlo usando dos directorios separados: uno para las imágenes de edición y otro para las de referencia.

El campo `conditioning_data` en el dataset de edición debe apuntar al `id` del dataset de referencia.

<details>
<summary>Ver ejemplo de config</summary>

```jsonc
[
  {
    "id": "my-edited-images",
    "type": "local",
    "cache_dir_vae": "/cache/vae/flux/kontext/edited-images",   // <-- donde se almacenan las salidas del VAE
    "instance_data_dir": "/datasets/edited-images",             // <-- usa rutas absolutas
    "conditioning_data": [
      "my-reference-images"                                     // <‑‑ este debe ser tu "id" del conjunto de referencia
                                                                // puedes especificar un segundo conjunto para alternar o combinar, p. ej. ["reference-images", "reference-images2"]
    ],
    "resolution": 1024,
    "caption_strategy": "textfile"                              // <-- estos captions deben contener las instrucciones de edición
  },
  {
    "id": "my-reference-images",
    "type": "local",
    "cache_dir_vae": "/cache/vae/flux/kontext/ref-images",      // <-- donde se almacenan las salidas del VAE. debe ser distinto de otras rutas VAE del dataset.
    "instance_data_dir": "/datasets/reference-images",          // <-- usa rutas absolutas
    "conditioning_type": "reference_strict",                    // <‑‑ si esto se configura en reference_loose, las imágenes se recortan de forma independiente de las imágenes de edición
    "resolution": 1024,
    "caption_strategy": null,                                   // <‑‑ no se necesitan captions para referencias, pero si están disponibles, se usarán EN LUGAR de los captions de edición
                                                                // NOTA: no puedes definir captions de condicionamiento separadas cuando usas conditioning_multidataset_sampling=combined.
                                                                // Solo se usarán los captions de los datasets de edición.
  }
]
```
</details>

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

*Cada imagen de edición **debe** tener nombres de archivo y extensiones con correspondencia 1‑a‑1 en ambas carpetas de dataset. SimpleTuner unirá automáticamente el embedding de referencia al condicionamiento de la edición.

Existe un ejemplo preparado de dataset demo [Kontext Max derived](https://huggingface.co/datasets/terminusresearch/KontextMax-Edit-smol) que contiene imágenes de referencia y edición junto con sus archivos de captions para explorar y tener una mejor idea de cómo configurarlo.

### Configurar una división de validación dedicada

Aquí hay un ejemplo de configuración que usa un conjunto de entrenamiento con 200,000 muestras y un conjunto de validación con solo unas pocas.
Deja `validation_using_datasets` sin configurar o en `false` para Kontext. La validación ya usa automáticamente el dataset de referencia emparejado; `eval_dataset_id` solo elige qué dataset de edición aporta las muestras de validación.

En tu `config.json` deberías agregar:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "eval_dataset_id": "edited-images",
}
```
</details>

Para tu `multidatabackend.json`, `edited-images` y `reference-images` deberían contener datos de validación con el mismo diseño que una división de entrenamiento habitual.

<details>
<summary>Ver ejemplo de config</summary>

```json
[
    {
        "id": "edited-images",
        "disabled": false,
        "type": "local",
        "instance_data_dir": "/datasets/edit/edited-images",
        "minimum_image_size": 1024,
        "maximum_image_size": 1536,
        "target_downsample_size": 1024,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "caption_strategy": "textfile",
        "cache_dir_vae": "cache/vae/flux-edit",
        "vae_cache_clear_each_epoch": false,
        "conditioning_data": ["reference-images"]
    },
    {
        "id": "reference-images",
        "disabled": false,
        "type": "local",
        "instance_data_dir": "/datasets/edit/reference-images",
        "minimum_image_size": 1024,
        "maximum_image_size": 1536,
        "target_downsample_size": 1024,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "caption_strategy": null,
        "cache_dir_vae": "cache/vae/flux-ref",
        "vae_cache_clear_each_epoch": false,
        "conditioning_type": "reference_strict"
    },
    {
        "id": "subjects200k-left",
        "disabled": false,
        "type": "huggingface",
        "dataset_name": "Yuanshi/Subjects200K",
        "caption_strategy": "huggingface",
        "metadata_backend": "huggingface",
        "resolution": 512,
        "resolution_type": "pixel_area",
        "conditioning_data": ["subjects200k-right"],
        "huggingface": {
            "caption_column": "description.description_0",
            "image_column": "image",
            "composite_image_config": {
                "enabled": true,
                "image_count": 2,
                "select_index": 0
            }
        }
    },
    {
        "id": "subjects200k-right",
        "disabled": false,
        "type": "huggingface",
        "dataset_type": "conditioning",
        "conditioning_type": "reference_strict",
        "source_dataset_id": "subjects200k-left",
        "dataset_name": "Yuanshi/Subjects200K",
        "caption_strategy": "huggingface",
        "metadata_backend": "huggingface",
        "resolution": 512,
        "resolution_type": "pixel_area",
        "huggingface": {
            "caption_column": "description.description_1",
            "image_column": "image",
            "composite_image_config": {
                "enabled": true,
                "image_count": 2,
                "select_index": 1
            }
        }
    },

    {
        "id": "text-embed-cache",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/flux"
    }
]
```
</details>

### Generación automática de pares Referencia‑Edición

Si no tienes pares referencia-edición preexistentes, SimpleTuner puede generarlos automáticamente a partir de un solo dataset. Esto es especialmente útil para entrenar modelos para:
- Mejora de imagen / super-resolución
- Eliminación de artefactos JPEG
- Deblurring
- Otras tareas de restauración

#### Ejemplo: dataset de entrenamiento de deblurring

<details>
<summary>Ver ejemplo de config</summary>

```jsonc
[
  {
    "id": "high-quality-images",
    "type": "local",
    "instance_data_dir": "/path/to/sharp-images",
    "resolution": 1024,
    "caption_strategy": "textfile",
    "conditioning": [
      {
        "type": "superresolution",
        "blur_radius": 3.0,
        "blur_type": "gaussian",
        "add_noise": true,
        "noise_level": 0.02,
        "captions": ["enhance sharpness", "deblur", "increase clarity", "sharpen image"]
      }
    ]
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/kontext"
  }
]
```
</details>

Esta configuración:
1. Crea versiones borrosas (se convierten en las imágenes de "referencia") a partir de tus imágenes nítidas de alta calidad
2. Usa las imágenes originales de alta calidad como objetivo de pérdida de entrenamiento
3. Entrena Kontext para mejorar/deblur la imagen de referencia de baja calidad

> **NOTA**: No puedes definir `captions` en un dataset de condicionamiento cuando usas `conditioning_multidataset_sampling=combined`. Se usarán los captions del dataset de edición.

#### Ejemplo: Eliminación de artefactos JPEG

<details>
<summary>Ver ejemplo de config</summary>

```jsonc
[
  {
    "id": "pristine-images",
    "type": "local",
    "instance_data_dir": "/path/to/pristine-images",
    "resolution": 1024,
    "caption_strategy": "textfile",
    "conditioning": [
      {
        "type": "jpeg_artifacts",
        "quality_mode": "range",
        "quality_range": [10, 30],
        "compression_rounds": 2,
        "captions": ["remove compression artifacts", "restore quality", "fix jpeg artifacts"]
      }
    ]
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/kontext"
  }
]
```
</details>

#### Notas importantes

1. **La generación ocurre al inicio**: Las versiones degradadas se crean automáticamente cuando comienza el entrenamiento
2. **Caché**: Las imágenes generadas se guardan, por lo que ejecuciones posteriores no las regenerarán
3. **Estrategia de captions**: El campo `captions` en la configuración de condicionamiento proporciona prompts específicos de tarea que funcionan mejor que descripciones genéricas de imagen
4. **Rendimiento**: Estos generadores basados en CPU (blur, JPEG) son rápidos y usan múltiples procesos
5. **Espacio en disco**: Asegúrate de tener suficiente espacio en disco para las imágenes generadas, ya que pueden ser grandes. Por desgracia, aún no hay capacidad para crearlas bajo demanda.

Para más tipos de condicionamiento y configuraciones avanzadas, consulta la [documentación de ControlNet](../CONTROLNET.md).

---

## 4. Consejos de entrenamiento específicos para Kontext

1. **Secuencias más largas → pasos más lentos.** Espera
~0.4 it/s en una sola 4090 a 1024 px, LoRA de rango 1, bf16 + int8.
2. **Explora para encontrar los ajustes correctos.** No se sabe mucho sobre el ajuste fino de Kontext; por seguridad, mantente en `1e‑5` (Lion) o `5e‑4` (AdamW).
3. **Vigila picos de VRAM durante el caché del VAE.** Si haces OOM, agrega `--offload_during_startup=true`, baja tu `resolution`, o posiblemente habilita el tiling del VAE vía tu `config.json`.
4. **Puedes entrenarlo sin imágenes de referencia, pero no actualmente vía SimpleTuner.** Actualmente, las cosas están algo hardcodeadas para requerir imágenes condicionales, pero puedes proporcionar datasets normales junto a tus pares de edición para que aprenda sujetos y semejanzas.
5. **Re-destilación de guidance.** Al igual que Flux‑dev, Kontext‑dev está destilado con CFG; si necesitas diversidad, reentrena con `validation_guidance_real > 1` y usa un nodo Adaptive‑Guidance en inferencia, aunque esto tardará MUCHO más en converger, y requerirá un LoRA de alto rango o un Lycoris LoKr para tener éxito.
6. **El entrenamiento de rango completo probablemente sea una pérdida de tiempo.** Kontext está diseñado para entrenarse con rango bajo, y el entrenamiento de rango completo probablemente no dará mejores resultados que un Lycoris LoKr, que normalmente superará a un LoRA estándar con menos trabajo buscando los mejores parámetros. Si quieres probarlo de todas formas, tendrás que usar DeepSpeed.
7. **Puedes usar dos o más imágenes de referencia para entrenar.** Como ejemplo, si tienes imágenes sujeto-sujeto-escena para insertar los dos sujetos en una sola escena, puedes proporcionar todas las imágenes relevantes como entradas de referencia. Solo asegúrate de que los nombres de archivo coincidan en todas las carpetas.

---

## 5. Trampas de inferencia

- Haz coincidir la precisión de entrenamiento e inferencia; el entrenamiento int8 funciona mejor con inferencia int8 y así sucesivamente.
- Va a ser muy lento debido a que dos imágenes pasan por el sistema al mismo tiempo. Espera 80 s por edición 1024 px en una 4090.

---

## 6. Hoja de solución rápida de problemas

| Síntoma                                 | Causa probable              | Solución rápida                                         |
| --------------------------------------- | ---------------------------- | ------------------------------------------------------ |
| OOM durante la cuantización             | No hay suficiente RAM del sistema | Usa `quantize_via=cpu`                                 |
| Imagen de referencia ignorada / sin edición aplicada | Emparejamiento incorrecto del dataloader | Asegura nombres de archivo idénticos y el campo `conditioning_data` |
| Artefactos de cuadrícula cuadrada       | Ediciones de baja calidad dominan | Crea dataset de mayor calidad, baja el LR, evita Lion |

---

## 7. Lecturas adicionales

Para opciones avanzadas de ajuste (LoKr, cuantización NF4, DeepSpeed, etc.) consulta [la guía rápida original de Flux](../quickstart/FLUX.md) – todos los flags funcionan igual a menos que se indique lo contrario arriba.
