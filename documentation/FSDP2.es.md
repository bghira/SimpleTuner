# Entrenamiento fragmentado / multi-GPU con FSDP2

SimpleTuner ahora incluye soporte de primera clase para PyTorch Fully Sharded Data Parallel v2 (FSDP respaldado por DTensor). El WebUI usa por defecto la implementación v2 para ejecuciones de modelo completo y expone los flags más importantes de accelerate para que puedas escalar a hardware multi-GPU sin escribir scripts de lanzamiento personalizados.

> ⚠️ FSDP2 apunta a releases recientes de PyTorch 2.x con el stack distribuido DTensor habilitado en builds CUDA. El WebUI solo muestra controles de paralelismo de contexto en hosts CUDA; otros backends se consideran experimentales.

## ¿Qué es FSDP2?

FSDP2 es la siguiente iteración del motor de datos paralelos con sharding de PyTorch. En lugar de la lógica legacy de parámetros planos de FSDP v1, el plugin v2 se apoya en DTensor. Fragmenta parámetros del modelo, gradientes y optimizadores entre ranks mientras mantiene un conjunto de trabajo pequeño por rank. En comparación con enfoques estilo ZeRO clásicos, mantiene el flujo de lanzamiento de Hugging Face accelerate, por lo que checkpoints, optimizadores y rutas de inferencia siguen siendo compatibles con el resto de SimpleTuner.

## Resumen de funciones

- Toggle en WebUI (Hardware → Accelerate) que genera un FullyShardedDataParallelPlugin con defaults razonables
- Normalización automática de CLI (`--fsdp_version`, `--fsdp_state_dict_type`, `--fsdp_auto_wrap_policy`) para que la escritura manual de flags sea tolerante
- Sharding opcional de paralelismo de contexto (`--context_parallel_size`, `--context_parallel_comm_strategy`) sobre FSDP2 para modelos de secuencias largas
- Modal de detección de bloques transformer integrado que inspecciona el modelo base y sugiere nombres de clase para auto-wrapping
- Metadatos de detección cacheados en `~/.simpletuner/fsdp_block_cache.json`, con acciones de mantenimiento de un clic en la configuración del WebUI
- Selector de formato de checkpoint (sharded vs full) y un modo de reanudación eficiente en RAM de CPU para límites de memoria del host ajustados

## Limitaciones conocidas

- FSDP2 solo se puede habilitar cuando `model_type` es `full`. Ejecuciones tipo PEFT/LoRA siguen usando rutas estándar de un solo dispositivo.
- DeepSpeed y FSDP son mutuamente excluyentes. Proporcionar `--fsdp_enable` y una configuración DeepSpeed genera un error explícito en flujos CLI y WebUI.
- El paralelismo de contexto se limita a sistemas CUDA y requiere `--context_parallel_size > 1` con `--fsdp_version=2`.
- Las pasadas de validación ahora funcionan con `--fsdp_reshard_after_forward=true` - los modelos envueltos con FSDP se pasan directamente a pipelines, que manejan all-gather/reshard de forma transparente.
- La detección de bloques instancia el modelo base localmente. Espera una pausa corta y mayor uso de memoria del host al escanear checkpoints grandes.
- FSDP v1 se mantiene por compatibilidad hacia atrás pero está marcado como deprecado en la UI y en los logs de CLI.

## Habilitar FSDP2

### Método 1: WebUI (recomendado)

1. Abre el WebUI de SimpleTuner y carga la configuración de entrenamiento que planeas ejecutar.
2. Cambia a **Hardware → Accelerate**.
3. Activa **Enable FSDP v2**. El selector de versión se pondrá por defecto en `2`; déjalo así salvo que necesites intencionalmente v1.
4. (Opcional) Ajusta:
   - **Reshard After Forward** para cambiar VRAM por comunicación
   - **Checkpoint Format** entre `Sharded` y `Full`
   - **CPU RAM Efficient Loading** si reanudas con límites de memoria del host ajustados
   - **Auto Wrap Policy** y **Transformer Classes to Wrap** (ver flujo de detección abajo)
   - **Context Parallel Size / Rotation** cuando necesites sharding de secuencia
5. Guarda la configuración. La superficie de lanzamiento del trainer ahora pasará el plugin de accelerate correcto.

### Método 2: CLI

Usa `simpletuner-train` con los mismos flags que se muestran en el WebUI. Ejemplo para una ejecución de modelo completo SDXL en dos GPUs:

```bash
simpletuner-train \
  --model_type=full \
  --model_family=sdxl \
  --output_dir=/data/experiments/sdxl-fsdp2 \
  --fsdp_enable \
  --fsdp_version=2 \
  --fsdp_state_dict_type=SHARDED_STATE_DICT \
  --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
  --num_processes=2
```

Si ya mantienes un archivo de configuración de accelerate, puedes seguir usándolo; SimpleTuner fusiona el plugin FSDP en los parámetros de lanzamiento en lugar de sobrescribir toda tu configuración.

## Paralelismo de contexto

El paralelismo de contexto está disponible como una capa opcional sobre FSDP2 para hosts CUDA. Configura `--context_parallel_size` (o el campo correspondiente en WebUI) con el número de GPUs que deben dividir la dimensión de secuencia. La comunicación ocurre vía:

- `allgather` (default) – prioriza el solapamiento y es el mejor punto de partida
- `alltoall` – cargas de trabajo de nicho con ventanas de atención muy grandes pueden beneficiarse, a costa de mayor orquestación

El trainer fuerza `fsdp_enable` y `fsdp_version=2` cuando se solicita paralelismo de contexto. Configurar el tamaño de vuelta a `1` deshabilita limpiamente la función y normaliza la cadena de rotación para que las configuraciones guardadas se mantengan consistentes.

## Flujo de detección de bloques FSDP

SimpleTuner incluye un detector que inspecciona el modelo base seleccionado y expone las clases de módulo más adecuadas para auto-wrapping de FSDP:

1. Selecciona una **Model Family** (y opcionalmente un **Model Flavour**) en el formulario del trainer.
2. Introduce la ruta del checkpoint si entrenas desde un directorio de pesos personalizado.
3. Haz clic en **Detect Blocks** junto a **Transformer Classes to Wrap**. SimpleTuner instanciará el modelo, recorrerá sus módulos y registrará totales de parámetros por clase.
4. Revisa el análisis en el modal:
   - **Selecciona** las clases que deben envolverse (check boxes en la primera columna)
   - **Total Params** destaca qué módulos dominan tu presupuesto de parámetros
   - `_no_split_modules` (si existen) se muestran como badges y deberían añadirse a tus listas de exclusión
5. Presiona **Apply Selection** para poblar `--fsdp_transformer_layer_cls_to_wrap`.
6. Aperturas posteriores reutilizan el resultado cacheado salvo que pulses **Refresh Detection**.

Los resultados de detección viven en `~/.simpletuner/fsdp_block_cache.json` y están indexados por familia de modelo, ruta de checkpoint y flavour. Usa **Settings → WebUI Preferences → Cache Maintenance → Clear FSDP Detection Cache** al cambiar entre checkpoints divergentes o después de actualizar pesos del modelo.

## Manejo de checkpoints

- **Sharded state dict** (`SHARDED_STATE_DICT`) guarda shards locales por rank y escala bien para modelos grandes.
- **Full state dict** (`FULL_STATE_DICT`) reúne parámetros en el rank 0 para compatibilidad con tooling externo; espera mayor presión de memoria.
- **CPU RAM Efficient Loading** retrasa la materialización en todos los ranks al reanudar para aplanar picos de memoria del host.
- **Reshard After Forward** mantiene shards de parámetros ligeros entre forward passes. La validación ahora funciona correctamente pasando modelos envueltos por FSDP directamente a pipelines de diffusers.

Elige la combinación que se alinee con tu cadencia de reanudación y tooling downstream. Checkpoints sharded + carga eficiente en RAM es la combinación más segura para modelos muy grandes.

## Herramientas de mantenimiento

El WebUI expone helpers de mantenimiento bajo **WebUI Preferences → Cache Maintenance**:

- **Clear FSDP Detection Cache** elimina todos los escaneos de bloques cacheados (wrapper sobre `FSDP_SERVICE.clear_cache()`).
- **Clear DeepSpeed Offload Cache** sigue disponible para usuarios de ZeRO; opera independientemente de FSDP.

Ambas acciones muestran notificaciones tipo toast y actualizan el área de estado de mantenimiento para que puedas confirmar el resultado sin revisar logs.

## Troubleshooting

| Síntoma | Causa probable | Arreglo |
|---------|----------------|---------|
| `"FSDP and DeepSpeed cannot be enabled simultaneously."` | Se especificaron ambos plugins (p. ej., JSON de DeepSpeed + `--fsdp_enable`). | Elimina la config de DeepSpeed o deshabilita FSDP. |
| `"Context parallelism requires FSDP2."` | `context_parallel_size > 1` mientras FSDP está apagado o en v1. | Habilita FSDP, mantén `--fsdp_version=2`, o baja el tamaño a `1`. |
| La detección de bloques falla con `Unknown model_family` | El formulario no tiene una familia o flavour soportados. | Elige un modelo del desplegable; familias personalizadas deben registrarse en `model_families`. |
| La detección muestra clases obsoletas | Se reutilizó el resultado cacheado. | Haz clic en **Refresh Detection** o limpia el caché desde WebUI Preferences. |
| Reanudar agota la RAM del host | Recolección de full state dict durante la carga. | Cambia a `SHARDED_STATE_DICT` y/o habilita carga eficiente en RAM de CPU. |

## Referencia de flags CLI

- `--fsdp_enable` – activa FullyShardedDataParallelPlugin
- `--fsdp_version` – elige entre `1` y `2` (default `2`, v1 está deprecado)
- `--fsdp_reshard_after_forward` – libera shards de parámetros post-forward (default `true`)
- `--fsdp_state_dict_type` – `SHARDED_STATE_DICT` (default) o `FULL_STATE_DICT`
- `--fsdp_cpu_ram_efficient_loading` – reduce picos de memoria del host al reanudar
- `--fsdp_auto_wrap_policy` – `TRANSFORMER_BASED_WRAP`, `SIZE_BASED_WRAP`, `NO_WRAP` o una ruta callable con puntos
- `--fsdp_transformer_layer_cls_to_wrap` – lista de clases separadas por comas, poblada por el detector
- `--context_parallel_size` – fragmenta atención entre esta cantidad de ranks (solo CUDA + FSDP2)
- `--context_parallel_comm_strategy` – estrategia de rotación `allgather` (default) o `alltoall`
- `--num_processes` – total de ranks pasados a accelerate cuando no se proporciona un archivo de configuración

Estos se mapean 1:1 con los controles del WebUI bajo Hardware → Accelerate, así que una configuración exportada desde la interfaz puede reproducirse en CLI sin ajustes adicionales.
