# Integracion con CaptionFlow

SimpleTuner puede usar [CaptionFlow](https://github.com/bghira/CaptionFlow) para generar captions de datasets de imagen desde la Web UI. CaptionFlow es un sistema de captioning escalable basado en vLLM, con orquestador, workers GPU, almacenamiento con checkpoints y configuracion YAML. En SimpleTuner aparece como la subpestana **Captioning** en la pagina Datasets, por lo que los trabajos de captioning usan la misma cola local de GPU que los trabajos de entrenamiento y cache.

Usa esta integracion cuando quieras generar o actualizar captions antes de entrenar sin salir del flujo de SimpleTuner.

## Instalacion

CaptionFlow es opcional. Instala el target de captioning en el mismo entorno virtual que usa SimpleTuner:

```bash
pip install "simpletuner[captioning]"
```

En entornos CUDA 13, usa el target CUDA 13 mostrado por la Web UI. Incluye la rueda de vLLM esperada por ese runtime.

## Que gestiona SimpleTuner

Al iniciar un trabajo de Captioning, SimpleTuner:

- asigna el dataset seleccionado a un procesador de CaptionFlow;
- inicia un orquestador CaptionFlow local en `127.0.0.1`;
- inicia uno o mas workers GPU locales mediante la cola de jobs;
- captura logs del orquestador y los workers en el workspace del job;
- hace checkpoint ordenado del almacenamiento de CaptionFlow antes de exportar;
- escribe captions `.txt` en el directorio del dataset para datasets locales;
- escribe exportaciones JSONL en el workspace de CaptionFlow para datasets de Hugging Face.

Las dependencias de CaptionFlow no son necesarias para que la pestana aparezca. Si faltan, la pestana muestra el comando de instalacion en lugar del builder.

## Modo Builder

La vista **Builder** cubre el flujo comun de captioning de una sola etapa:

- seleccion de dataset desde la configuracion activa del dataloader;
- modelo, prompt, sampling, tamano de batch, tamano de chunk y memoria GPU;
- cantidad de workers y comportamiento de cola;
- exportacion de archivos de texto para datasets locales.

El modelo por defecto es `Qwen/Qwen2.5-VL-3B-Instruct`. Los datasets locales exportan archivos de texto junto a las imagenes usando el campo de salida seleccionado. Los datasets de Hugging Face no se escriben de vuelta al dataset remoto; se exportan como JSONL en el workspace de CaptionFlow.

## Modo Raw Config

Usa **Raw Config** cuando necesites funciones de CaptionFlow que el builder no modela, como captioning multi-etapa, modelos por etapa, sampling por etapa o cadenas de prompts donde una etapa consume la salida de otra.

Raw config acepta YAML o JSON. Puedes pegar una configuracion completa con raiz `orchestrator:` o solo el objeto del orquestador.

SimpleTuner sobrescribe intencionalmente estos campos en runtime:

- `host`, `port` y `ssl`;
- `dataset`, basado en el dataset de SimpleTuner seleccionado;
- `storage.data_dir` y `storage.checkpoint_dir`, dentro del workspace del job;
- `auth.worker_tokens` y `auth.admin_tokens`.

Otros ajustes del orquestador, incluidos `chunk_size`, `chunks_per_request`, `storage.caption_buffer_size`, `vllm.sampling`, `vllm.inference_prompts` y `vllm.stages`, se conservan salvo que SimpleTuner necesite aplicar un valor por defecto.

## Ejemplo multi-etapa

Este raw config ejecuta una etapa de caption detallado y luego pasa `{caption}` a una etapa de resumen. SimpleTuner completa el dataset seleccionado, rutas de storage, puertos y tokens al lanzar el job.

```yaml
orchestrator:
  chunk_size: 1000
  chunks_per_request: 1
  chunk_buffer_multiplier: 2
  min_chunk_buffer: 10
  vllm:
    model: "Qwen/Qwen2.5-VL-3B-Instruct"
    tensor_parallel_size: 1
    max_model_len: 16384
    dtype: "float16"
    gpu_memory_utilization: 0.92
    enforce_eager: true
    disable_mm_preprocessor_cache: true
    limit_mm_per_prompt:
      image: 1
    batch_size: 8
    sampling:
      temperature: 0.7
      top_p: 0.95
      max_tokens: 256
    stages:
      - name: "base_caption"
        prompts:
          - "describe this image in detail"
        output_field: "caption"
      - name: "caption_shortening"
        model: "Qwen/Qwen2.5-VL-7B-Instruct"
        prompts:
          - "Please condense this elaborate caption to only the important details: {caption}"
        output_field: "captions"
        requires: ["base_caption"]
        gpu_memory_utilization: 0.35
```

## Documentacion externa

- [Repositorio de CaptionFlow](https://github.com/bghira/CaptionFlow)
- [README de CaptionFlow](https://github.com/bghira/CaptionFlow#readme)
- [Ejemplos de orquestador de CaptionFlow](https://github.com/bghira/CaptionFlow/tree/main/examples/orchestrator)

Usa los ejemplos upstream como referencia para campos avanzados. Cuando ejecutes CaptionFlow mediante SimpleTuner, recuerda que SimpleTuner controla el dataset, los puertos locales, las rutas del workspace de storage y los tokens de autenticacion.
