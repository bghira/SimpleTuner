# Opciones del script de entrenamiento de SimpleTuner

## Visión general

Esta guía ofrece un desglose amigable de las opciones de línea de comandos disponibles en el script `train.py` de SimpleTuner. Estas opciones ofrecen un alto grado de personalización, permitiéndote entrenar tu modelo para ajustarlo a tus requisitos específicos.

### Formato del archivo de configuración JSON

El nombre de archivo JSON esperado es `config.json` y los nombres de clave son los mismos que los `--argumentos` de abajo. El prefijo `--` no es obligatorio en el archivo JSON, pero también puede dejarse.

¿Buscas ejemplos listos para ejecutar? Consulta los presets curados en [simpletuner/examples/README.md](/simpletuner/examples/README.md).

### Script de configuración fácil (***RECOMENDADO***)

El comando `simpletuner configure` puede usarse para configurar un archivo `config.json` con valores predeterminados casi ideales.

#### Modificar configuraciones existentes

El comando `configure` es capaz de aceptar un único argumento, un `config.json` compatible, lo que permite la modificación interactiva de tu configuración de entrenamiento:

```bash
simpletuner configure config/foo/config.json
```

Donde `foo` es tu entorno de configuración; o simplemente usa `config/config.json` si no estás usando entornos de configuración.

<img width="1484" height="560" alt="image" src="https://github.com/user-attachments/assets/67dec8d8-3e41-42df-96e6-f95892d2814c" />

> ⚠️ Para usuarios ubicados en países donde Hugging Face Hub no es fácilmente accesible, deberías añadir `HF_ENDPOINT=https://hf-mirror.com` a tu `~/.bashrc` o `~/.zshrc` dependiendo de qué `$SHELL` use tu sistema.

---

## 🌟 Configuración central del modelo

### `--model_type`

- **Qué**: Selecciona si se crea una LoRA o un fine-tune completo.
- **Opciones**: lora, full.
- **Predeterminado**: lora
  - Si se usa lora, `--lora_type` dicta si se usa PEFT o LyCORIS. Algunos modelos (PixArt) funcionan solo con adaptadores LyCORIS.

### `--model_family`

- **Qué**: Determina qué arquitectura de modelo se está entrenando.
- **Opciones**: pixart_sigma, flux, sd3, sdxl, kolors, legacy

### `--lora_format`

- **Qué**: Selecciona el formato de claves del checkpoint LoRA para cargar/guardar.
- **Opciones**: `diffusers` (predeterminado), `comfyui`
- **Notas**:
  - `diffusers` es el esquema estándar de PEFT/Diffusers.
  - `comfyui` convierte hacia/desde claves estilo ComfyUI (`diffusion_model.*` con tensores `lora_A/lora_B` y `.alpha`). Flux, Flux2, Lumina2 y Z-Image detectarán automáticamente entradas ComfyUI incluso si esto se deja en `diffusers`, pero cámbialo a `comfyui` para forzar salida ComfyUI al guardar.

### `--fuse_qkv_projections`

- **Qué**: Fusiona las proyecciones QKV en los bloques de atención del modelo para un uso más eficiente del hardware.
- **Nota**: Solo disponible con NVIDIA H100 o H200 con Flash Attention 3 instalado manualmente.

### `--offload_during_startup`

- **Qué**: Descarga los pesos del codificador de texto a CPU mientras se hace el caché del VAE.
- **Por qué**: Esto es útil para modelos grandes como HiDream y Wan 2.1, que pueden quedarse sin memoria al cargar el caché del VAE. Esta opción no afecta la calidad del entrenamiento, pero para codificadores de texto muy grandes o CPUs lentas, puede extender sustancialmente el tiempo de arranque con muchos datasets. Está desactivada por defecto por esta razón.
- **Consejo**: Complementa la función de offload en grupo de abajo para sistemas con memoria especialmente limitada.

### `--offload_during_save`

- **Qué**: Mueve temporalmente toda la canalización a CPU mientras `save_hooks.py` prepara los checkpoints para que todos los pesos FP8/cuantizados se escriban fuera del dispositivo.
- **Por qué**: Guardar pesos fp8-quanto puede disparar el uso de VRAM (por ejemplo, durante la serialización de `state_dict()`). Esta opción mantiene el modelo en el acelerador para entrenar, pero lo descarga brevemente cuando se dispara un guardado para evitar OOM de CUDA.
- **Consejo**: Actívalo solo cuando el guardado falle con errores de OOM; el cargador devuelve el modelo después y el entrenamiento se reanuda sin problemas.

### `--delete_model_after_load`

- **Qué**: Elimina los archivos de modelo del caché de HuggingFace después de cargarlos en memoria.
- **Por qué**: Reduce el uso de disco para configuraciones con presupuestos ajustados que cobran por gigabyte usado. Después de cargar los modelos en VRAM/RAM, el caché en disco ya no es necesario hasta la siguiente ejecución. Esto traslada la carga de almacenamiento al ancho de banda de red en ejecuciones posteriores.
- **Notas**:
  - El VAE **no** se elimina si la validación está habilitada, ya que se necesita para generar imágenes de validación.
  - Los codificadores de texto se eliminan después de que la fábrica de backend de datos completa el arranque (después del caché de embeddings).
  - Los modelos Transformer/UNet se eliminan inmediatamente después de la carga.
  - En configuraciones multinodo, solo el rank local 0 en cada nodo realiza la eliminación. Los fallos de eliminación se ignoran silenciosamente para manejar condiciones de carrera en almacenamiento de red compartido.
  - Esto **no** afecta a los checkpoints de entrenamiento guardados; solo al caché del modelo base preentrenado.

### `--trust_remote_code`

- **Qué**: Permite que Transformers y los tokenizers ejecuten código Python personalizado del repositorio del modelo cuando el checkpoint depende de clases personalizadas upstream.
- **Predeterminado**: `False`
- **Por qué**: Es necesario para los checkpoints ACE-Step v1.5, que incluyen código personalizado de `AutoModel` y tokenizer en el repositorio upstream.
- **Advertencia**: Actívalo solo para repositorios de modelos en los que confíes.

### `--enable_group_offload`

- **Qué**: Habilita el offload de módulos agrupados de diffusers para que los bloques del modelo se puedan preparar en CPU (o disco) entre pasadas hacia delante.
- **Por qué**: Reduce drásticamente el uso máximo de VRAM en transformadores grandes (Flux, Wan, Auraflow, LTXVideo, Cosmos2Image) con un impacto mínimo en rendimiento cuando se usa con streams CUDA.
- **Notas**:
  - Es mutuamente excluyente con `--enable_model_cpu_offload`; elige una estrategia por ejecución.
  - Requiere diffusers **v0.33.0** o más reciente.

### `--group_offload_type`

- **Opciones**: `block_level` (predeterminado), `leaf_level`
- **Qué**: Controla cómo se agrupan las capas. `block_level` equilibra el ahorro de VRAM con el rendimiento, mientras que `leaf_level` maximiza el ahorro a costa de más transferencias de CPU.

### `--group_offload_blocks_per_group`

- **Qué**: Al usar `block_level`, el número de bloques de transformer que se agrupan en un solo grupo de offload.
- **Predeterminado**: `1`
- **Por qué**: Aumentar este número reduce la frecuencia de transferencias (más rápido) pero mantiene más parámetros en el acelerador (usa más VRAM).

### `--group_offload_use_stream`

- **Qué**: Usa un stream CUDA dedicado para solapar transferencias host/dispositivo con cómputo.
- **Predeterminado**: `False`
- **Notas**:
  - Recurre automáticamente a transferencias estilo CPU en backends no CUDA (Apple MPS, ROCm, CPU).
  - Recomendado al entrenar en GPUs NVIDIA con capacidad de motor de copia disponible.

### `--group_offload_to_disk_path`

- **Qué**: Ruta de directorio usada para volcar parámetros agrupados a disco en lugar de RAM.
- **Por qué**: Útil para presupuestos de RAM de CPU extremadamente ajustados (p. ej., estación de trabajo con gran NVMe).
- **Consejo**: Usa un SSD local rápido; los sistemas de archivos de red ralentizarán significativamente el entrenamiento.

### `--musubi_blocks_to_swap`

- **Qué**: Intercambio de bloques Musubi para LongCat-Video, Wan, LTXVideo, Kandinsky5-Video, Qwen-Image, Flux, Flux.2, Cosmos2Image y HunyuanVideo: mantiene los últimos N bloques del transformer en CPU y transmite pesos por bloque durante el forward.
- **Predeterminado**: `0` (desactivado)
- **Notas**: Offload de pesos estilo Musubi; reduce VRAM con coste en rendimiento y se omite cuando los gradientes están habilitados.

### `--musubi_block_swap_device`

- **Qué**: Cadena de dispositivo para almacenar bloques de transformer intercambiados (p. ej., `cpu`, `cuda:0`).
- **Predeterminado**: `cpu`
- **Notas**: Solo se usa cuando `--musubi_blocks_to_swap` > 0.

### `--ramtorch`

- **Qué**: Reemplaza capas `nn.Linear` con implementaciones RamTorch transmitidas desde CPU.
- **Por qué**: Comparte los pesos Linear en memoria de CPU y los transmite al acelerador para reducir la presión de VRAM.
- **Notas**:
  - Requiere CUDA o ROCm (no compatible con Apple/MPS).
  - Es mutuamente excluyente con `--enable_group_offload`.
  - Habilita automáticamente `--set_grads_to_none`.

### `--ramtorch_target_modules`

- **Qué**: Patrones glob separados por comas que limitan qué módulos Linear se convierten a RamTorch.
- **Predeterminado**: Todas las capas Linear se convierten cuando no se proporciona un patrón.
- **Notas**:
  - Coincide con nombres de módulos completamente calificados o nombres de clase usando la sintaxis glob de `fnmatch`.
  - Los patrones deben incluir un comodín `.*` al final para coincidir con las capas dentro de un bloque. Por ejemplo, `transformer_blocks.0.*` coincide con todas las capas dentro del bloque 0, mientras que `transformer_blocks.*` coincide con todos los bloques transformer. Un nombre simple como `transformer_blocks.0` sin `.*` también funciona (se expande automáticamente), pero se recomienda la forma explícita con comodín para mayor claridad.
  - Ejemplo: `"transformer_blocks.*,single_transformer_blocks.0.*,single_transformer_blocks.1.*"`

### `--ramtorch_text_encoder`

- **Qué**: Aplica reemplazos RamTorch a todas las capas Linear del codificador de texto.
- **Predeterminado**: `False`

### `--ramtorch_vae`

- **Qué**: Conversión RamTorch experimental solo para las capas Linear del bloque medio del VAE.
- **Predeterminado**: `False`
- **Notas**: Los bloques de convolución up/down del VAE se dejan sin cambios.

### `--ramtorch_controlnet`

- **Qué**: Aplica reemplazos RamTorch a las capas Linear de ControlNet cuando se entrena un ControlNet.
- **Predeterminado**: `False`

### `--ramtorch_transformer_percent`

- **Qué**: Porcentaje (0-100) de capas Linear del transformer a descargar con RamTorch.
- **Predeterminado**: `100` (todas las capas elegibles)
- **Por qué**: Permite una descarga parcial para equilibrar el ahorro de VRAM con el rendimiento. Valores más bajos mantienen más capas en la GPU para un entrenamiento más rápido, mientras se reduce el uso de memoria.
- **Notas**: Las capas se seleccionan desde el inicio del orden de recorrido del módulo. Se puede combinar con `--ramtorch_target_modules`.

### `--ramtorch_text_encoder_percent`

- **Qué**: Porcentaje (0-100) de capas Linear del codificador de texto a descargar con RamTorch.
- **Predeterminado**: `100` (todas las capas elegibles)
- **Por qué**: Permite la descarga parcial de codificadores de texto cuando `--ramtorch_text_encoder` está habilitado.
- **Notas**: Solo aplica cuando `--ramtorch_text_encoder` está habilitado.

### `--ramtorch_disable_sync_hooks`

- **Qué**: Desactiva los hooks de sincronización CUDA que se añaden después de las capas RamTorch.
- **Predeterminado**: `False` (hooks de sincronización habilitados)
- **Por qué**: Los hooks de sincronización corrigen condiciones de carrera en el sistema de buffering ping-pong de RamTorch que pueden causar salidas no deterministas. Desactivarlos puede mejorar el rendimiento pero arriesga resultados incorrectos.
- **Notas**: Solo desactivar si experimentas problemas con los hooks de sincronización o quieres probar sin ellos.

### `--ramtorch_disable_extensions`

- **Qué**: Solo aplica RamTorch a capas Linear, omite Embedding/RMSNorm/LayerNorm/Conv.
- **Predeterminado**: `True` (extensiones deshabilitadas)
- **Por qué**: SimpleTuner extiende RamTorch más allá de las capas Linear para incluir capas Embedding, RMSNorm, LayerNorm y Conv. Usa esto para desactivar esas extensiones y solo descargar capas Linear.
- **Notas**: Puede reducir el ahorro de VRAM pero puede ayudar a depurar problemas con los tipos de capas extendidas.

### `--pretrained_model_name_or_path`

- **Qué**: Ruta al modelo preentrenado o su identificador en <https://huggingface.co/models>.
- **Por qué**: Para especificar el modelo base desde el que comenzarás a entrenar. Usa `--revision` y `--variant` para especificar versiones concretas desde un repositorio. También admite rutas de un solo archivo `.safetensors` para SDXL, Flux y SD3.x.

### `--pretrained_t5_model_name_or_path`

- **Qué**: Ruta al modelo T5 preentrenado o su identificador en <https://huggingface.co/models>.
- **Por qué**: Al entrenar PixArt, quizá quieras usar una fuente específica para tus pesos T5 para evitar descargarlos varias veces al cambiar el modelo base desde el que entrenas.

### `--pretrained_gemma_model_name_or_path`

- **Qué**: Ruta al modelo Gemma preentrenado o su identificador en <https://huggingface.co/models>.
- **Por qué**: Al entrenar modelos basados en Gemma (por ejemplo LTX-2, Sana o Lumina2), puedes apuntar a un checkpoint Gemma compartido sin cambiar la ruta del modelo base de difusión.

### `--max_grounding_entities`
- Numero maximo de entidades de grounding por imagen para anotaciones espaciales estilo GLIGEN. Por defecto: 0 (deshabilitado). Valores tipicos: 4-16.

### `--pretrained_grounding_model_name_or_path`
- Modelo preentrenado opcional para la extraccion de features de imagen por entidad. Por defecto: None.

### `--custom_text_encoder_intermediary_layers`

- **Qué**: Sobrescribe qué capas de estado oculto extraer del encoder de texto para modelos FLUX.2.
- **Formato**: Array JSON de índices de capas, ej: `[10, 20, 30]`
- **Por defecto**: Se usan los valores por defecto específicos del modelo cuando no se establece:
  - FLUX.2-dev (Mistral-3): `[10, 20, 30]`
  - FLUX.2-klein (Qwen3): `[9, 18, 27]`
- **Por qué**: Permite experimentar con diferentes combinaciones de estados ocultos del encoder de texto para alineación personalizada o propósitos de investigación.
- **Nota**: Esta opción es experimental y solo aplica a modelos FLUX.2. Cambiar los índices de capas invalidará los embeddings de texto en caché y requerirá regenerarlos. El número de capas debe coincidir con la entrada esperada por el modelo (3 capas).

### `--gradient_checkpointing`

- **Qué**: Durante el entrenamiento, los gradientes se calcularán por capa y se acumularán para ahorrar en los requisitos máximos de VRAM a costa de un entrenamiento más lento.

### `--gradient_checkpointing_interval`

- **Qué**: Hace checkpoint cada *n* bloques, donde *n* es un valor mayor que cero. Un valor de 1 es equivalente a dejar `--gradient_checkpointing` habilitado, y un valor de 2 hará checkpoint en bloques alternos.
- **Nota**: SDXL y Flux son actualmente los únicos modelos que soportan esta opción. SDXL usa una implementación algo improvisada.

### `--gradient_checkpointing_backend`

- **Opciones**: `torch`, `unsloth`
- **Qué**: Selecciona la implementación para gradient checkpointing.
  - `torch` (por defecto): Checkpointing estándar de PyTorch que recalcula activaciones durante el backward pass. ~20% de overhead de tiempo.
  - `unsloth`: Descarga activaciones a CPU de forma asíncrona en lugar de recalcular. ~30% más ahorro de memoria con solo ~2% de overhead. Requiere ancho de banda PCIe rápido.
- **Nota**: Solo efectivo cuando `--gradient_checkpointing` está habilitado. El backend `unsloth` requiere CUDA.

### `--refiner_training`

- **Qué**: Habilita el entrenamiento de una serie de modelos de mezcla de expertos personalizada. Consulta [Mixture-of-Experts](MIXTURE_OF_EXPERTS.md) para más información sobre estas opciones.

## Precisión

### `--quantize_via`

- **Opciones**: `cpu`, `accelerator`, `pipeline`
  - En `accelerator`, puede funcionar moderadamente más rápido con el riesgo de posibles OOM en tarjetas de 24G para un modelo tan grande como Flux.
  - En `cpu`, la cuantización tarda unos 30 segundos. (**Predeterminado**)
  - `pipeline` delega la cuantización a Diffusers usando `--quantization_config` o presets compatibles con pipeline (p. ej., `nf4-bnb`, `int8-torchao`, `fp8-torchao`, `int8-quanto`, o checkpoints `.gguf`).

### `--base_model_precision`

- **Qué**: Reduce la precisión del modelo y entrena usando menos memoria. Hay tres backends de cuantización compatibles: BitsAndBytes (pipeline), TorchAO (pipeline o manual) y Optimum Quanto (pipeline o manual).

#### Presets de pipeline de Diffusers

- `nf4-bnb` se carga a través de Diffusers con una configuración BitsAndBytes NF4 de 4 bits (solo CUDA). Requiere `bitsandbytes` y una build de diffusers con soporte BnB.
- `int4-torchao`, `int8-torchao` y `fp8-torchao` usan TorchAoConfig vía Diffusers (CUDA). Requiere `torchao` y diffusers/transformers recientes.
- `int8-quanto`, `int4-quanto`, `int2-quanto`, `fp8-quanto` y `fp8uz-quanto` usan QuantoConfig vía Diffusers. Diffusers asigna FP8-NUZ a pesos float8; usa cuantización manual de quanto si necesitas la variante NUZ.
- `.gguf` checkpoints se detectan automáticamente y se cargan con `GGUFQuantizationConfig` cuando está disponible. Instala versiones recientes de diffusers/transformers para soporte GGUF.

#### Optimum Quanto

Proporcionada por Hugging Face, la librería optimum-quanto tiene un soporte robusto en todas las plataformas compatibles.

- `int8-quanto` es la más compatible y probablemente produce los mejores resultados
  - entrenamiento más rápido para RTX4090 y probablemente otras GPUs
  - usa matmul acelerado por hardware en dispositivos CUDA para int8, int4
    - int4 sigue siendo terriblemente lento
  - funciona con `TRAINING_DYNAMO_BACKEND=inductor` (`torch.compile()`)
- `fp8uz-quanto` es una variante fp8 experimental para dispositivos CUDA y ROCm.
  - mejor soportado en silicio AMD como Instinct o arquitecturas más nuevas
  - puede ser ligeramente más rápido que `int8-quanto` en una 4090 para entrenamiento, pero no para inferencia (1 segundo más lento)
  - funciona con `TRAINING_DYNAMO_BACKEND=inductor` (`torch.compile()`)
- `fp8-quanto` no usará (por ahora) matmul fp8, no funciona en sistemas Apple.
  - aún no hay matmul fp8 por hardware en dispositivos CUDA o ROCm, por lo que posiblemente sea notablemente más lento que int8
    - usa el kernel MARLIN para GEMM fp8
  - incompatible con dynamo, deshabilitará automáticamente dynamo si se intenta esa combinación.

#### TorchAO

Una biblioteca más nueva de PyTorch; AO nos permite reemplazar los linears y las convoluciones 2D (p. ej., modelos tipo unet) con contrapartes cuantizadas.
<!-- Additionally, it provides an experimental CPU offload optimiser that essentially provides a simpler reimplementation of DeepSpeed. -->

- `int8-torchao` reducirá el consumo de memoria al mismo nivel que cualquiera de los niveles de precisión de Quanto
  - al momento de escribir, corre ligeramente más lento (11s/iter) que Quanto (9s/iter) en Apple MPS
  - cuando no se usa `torch.compile`, misma velocidad y uso de memoria que `int8-quanto` en dispositivos CUDA, perfil de velocidad desconocido en ROCm
  - al usar `torch.compile`, más lento que `int8-quanto`
- `fp8-torchao` solo está disponible para aceleradores Hopper (H100, H200) o más nuevos (Blackwell B200)

##### Optimizadores

TorchAO incluye optimizadores 4bit y 8bit de uso general: `ao-adamw8bit`, `ao-adamw4bit`

También proporciona dos optimizadores dirigidos a usuarios de Hopper (H100 o mejor): `ao-adamfp8` y `ao-adamwfp8`

#### SDNQ (SD.Next Quantization Engine)

[SDNQ](https://github.com/disty0/sdnq) es una librería de cuantización optimizada para entrenamiento que funciona en todas las plataformas: AMD (ROCm), Apple (MPS) y NVIDIA (CUDA). Proporciona entrenamiento cuantizado con redondeo estocástico y estados de optimizador cuantizados para eficiencia de memoria.

##### Niveles de precisión recomendados

**Para fine-tuning completo** (se actualizan los pesos del modelo):
- `uint8-sdnq` - Mejor equilibrio entre ahorro de memoria y calidad de entrenamiento
- `uint16-sdnq` - Mayor precisión para máxima calidad (p. ej., Stable Cascade)
- `int16-sdnq` - Alternativa signed de 16 bits
- `fp16-sdnq` - FP16 cuantizado, máxima precisión con los beneficios de SDNQ

**Para entrenamiento LoRA** (pesos base congelados):
- `int8-sdnq` - 8 bits con signo, buena opción general
- `int6-sdnq`, `int5-sdnq` - Menor precisión, menos memoria
- `uint5-sdnq`, `uint4-sdnq`, `uint3-sdnq`, `uint2-sdnq` - Compresión agresiva

**Nota:** `int7-sdnq` está disponible pero no se recomienda (lento y no mucho más pequeño que int8).

**Importante:** Por debajo de 5 bits de precisión, SDNQ habilita automáticamente SVD (Singular Value Decomposition) con 8 pasos para mantener la calidad. SVD tarda más en cuantizar y es no determinista, por lo que Disty0 proporciona modelos SVD pre-cuantizados en HuggingFace. SVD añade overhead de cómputo durante el entrenamiento, así que evítalo para fine-tuning completo donde los pesos se actualizan activamente.

**Características clave:**
- Multiplataforma: Funciona de forma idéntica en hardware AMD, Apple y NVIDIA
- Optimizado para entrenamiento: Usa redondeo estocástico para reducir acumulación de errores de cuantización
- Eficiente en memoria: Soporta buffers de estado del optimizador cuantizados
- Matmul desacoplado: La precisión de los pesos y la precisión de matmul son independientes (matmul INT8/FP8/FP16 disponible)

##### Optimizadores SDNQ

SDNQ incluye optimizadores con buffers de estado cuantizados opcionales para ahorro adicional de memoria:

- `sdnq-adamw` - AdamW con buffers de estado cuantizados (uint8, group_size=32)
- `sdnq-adamw+no_quant` - AdamW sin estados cuantizados (para comparación)
- `sdnq-adafactor` - Adafactor con buffers de estado cuantizados
- `sdnq-came` - Optimizador CAME con buffers de estado cuantizados
- `sdnq-lion` - Optimizador Lion con buffers de estado cuantizados
- `sdnq-muon` - Optimizador Muon con buffers de estado cuantizados
- `sdnq-muon+quantized_matmul` - Muon con matmul INT8 en el cálculo de zeropower

Todos los optimizadores SDNQ usan redondeo estocástico por defecto y pueden configurarse con `--optimizer_config` para ajustes personalizados como `use_quantized_buffers=false` para desactivar la cuantización de estado.

**Opciones específicas de Muon:**
- `use_quantized_matmul` - Habilita matmul INT8/FP8/FP16 en zeropower_via_newtonschulz5
- `quantized_matmul_dtype` - Precisión de matmul: `int8` (GPUs de consumo), `fp8` (datacenter), `fp16`
- `zeropower_dtype` - Precisión para el cálculo de zeropower (se ignora cuando `use_quantized_matmul=True`)
- Prefija argumentos con `muon_` o `adamw_` para establecer valores diferentes para Muon vs fallback AdamW

**Modelos pre-cuantizados:** Disty0 proporciona modelos SVD uint4 pre-cuantizados en [huggingface.co/collections/Disty0/sdnq](https://huggingface.co/collections/Disty0/sdnq). Cárgalos normalmente y luego convierte con `convert_sdnq_model_to_training()` después de importar SDNQ (SDNQ debe importarse antes de cargar para registrarse con Diffusers).

**Nota sobre checkpointing:** Los modelos de entrenamiento SDNQ se guardan tanto en formato nativo PyTorch (`.pt`) para reanudar entrenamiento como en formato safetensors para inferencia. El formato nativo es obligatorio para una reanudación correcta, ya que la clase `SDNQTensor` de SDNQ usa serialización personalizada.

**Consejo de espacio en disco:** Para ahorrar espacio en disco, puedes mantener solo los pesos cuantizados y usar el script `dequantize_sdnq_training.py` de SDNQ ([enlace](https://github.com/Disty0/sdnq/blob/main/scripts/dequantize_sdnq_training.py)) para de-cuantizar cuando sea necesario para inferencia.

### `--quantization_config`

- **Qué**: Objeto JSON o ruta de archivo que describe overrides de `quantization_config` de Diffusers cuando se usa `--quantize_via=pipeline`.
- **Cómo**: Acepta JSON inline (o un archivo) con entradas por componente. Las claves pueden incluir `unet`, `transformer`, `text_encoder` o `default`.
- **Ejemplos**:

```json
{
  "unet": {"load_in_4bit": true, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16"},
  "text_encoder": {"quant_type": {"group_size": 128}}
}
```

Este ejemplo habilita BnB NF4 de 4 bits en el UNet y TorchAO int4 en el codificador de texto.

#### Torch Dynamo

Habilita `torch.compile()` desde la WebUI visitando **Hardware → Accelerate (advanced)** y configurando **Torch Dynamo Backend** con tu compilador preferido (por ejemplo, *inductor*). Los toggles adicionales te permiten elegir un **modo** de optimización, habilitar guardas de **forma dinámica** u optar por la **compilación regional** para acelerar arranques en frío en modelos transformer muy profundos.

La misma configuración puede expresarse en `config/config.env`:

```bash
TRAINING_DYNAMO_BACKEND=inductor
```

Opcionalmente puedes combinar esto con `--dynamo_mode=max-autotune` u otras flags de Dynamo expuestas en la UI para un control más fino.

Ten en cuenta que los primeros pasos del entrenamiento serán más lentos de lo normal debido a la compilación en segundo plano.

Para persistir los ajustes en `config.json`, añade las claves equivalentes:

```json
{
  "dynamo_backend": "inductor",
  "dynamo_mode": "max-autotune",
  "dynamo_fullgraph": false,
  "dynamo_dynamic": false,
  "dynamo_use_regional_compilation": true
}
```

Omite cualquier entrada que quieras heredar de los valores predeterminados de Accelerate (por ejemplo, deja fuera `dynamo_mode` para usar la selección automática).

### `--attention_mechanism`

Se soportan mecanismos de atención alternativos, con distintos niveles de compatibilidad u otros compromisos:

- `diffusers` usa los kernels SDPA nativos de PyTorch y es el predeterminado.
- `xformers` habilita el kernel de atención [xformers](https://github.com/facebookresearch/xformers) de Meta (entrenamiento + inferencia) cuando el modelo subyacente expone `enable_xformers_memory_efficient_attention`.
- `flash-attn`, `flash-attn-2`, `flash-attn-3` y `flash-attn-3-varlen` se enganchan al helper `attention_backend` de Diffusers para enrutar la atención a través de los kernels FlashAttention v1/2/3. Instala las wheels correspondientes de `flash-attn` / `flash-attn-interface` y ten en cuenta que FA3 actualmente requiere GPUs Hopper.
- `flex` selecciona el backend FlexAttention de PyTorch 2.5 (FP16/BF16 en CUDA). Debes compilar/instalar los kernels Flex por separado; consulta [documentation/attention/FLEX.md](attention/FLEX.md).
- `cudnn`, `native-efficient`, `native-flash`, `native-math`, `native-npu` y `native-xla` seleccionan el backend SDPA correspondiente expuesto por `torch.nn.attention.sdpa_kernel`. Son útiles cuando quieres determinismo (`native-math`), el kernel SDPA de CuDNN o los aceleradores nativos del proveedor (NPU/XLA).
- `sla` habilita [Sparse–Linear Attention (SLA)](https://github.com/thu-ml/SLA), proporcionando un kernel híbrido disperso/lineal afinable que puede usarse tanto para entrenamiento como para validación sin gating adicional.
  - Instala el paquete SLA (por ejemplo con `pip install -e ~/src/SLA`) antes de seleccionar este backend.
  - SimpleTuner guarda los pesos de proyección aprendidos de SLA en `sla_attention.pt` dentro de cada checkpoint; conserva este archivo con el resto del checkpoint para que las reanudaciones e inferencia mantengan el estado SLA entrenado.
  - Como el backbone se ajusta alrededor del comportamiento mixto disperso/lineal de SLA, SLA será necesario también en inferencia. Consulta `documentation/attention/SLA.md` para una guía enfocada.
  - Usa `--sla_config '{"topk":0.15,"blkq":32,"tie_feature_map_qk":false}'` (JSON o sintaxis de dict de Python) para sobrescribir los valores predeterminados de runtime de SLA si necesitas experimentar.
- `sageattention`, `sageattention-int8-fp16-triton`, `sageattention-int8-fp16-cuda` y `sageattention-int8-fp8-cuda` envuelven los kernels correspondientes de [SageAttention](https://github.com/thu-ml/SageAttention). Están orientados a inferencia y deben usarse con `--sageattention_usage` para protegerse de entrenamiento accidental.
  - En términos simples, SageAttention reduce el requisito de cómputo para inferencia

> ℹ️ Los selectores de backend Flash/Flex/PyTorch dependen del despachador `attention_backend` de Diffusers, por lo que actualmente benefician a modelos estilo transformer que ya optan por esa ruta de código (Flux, Wan 2.x, LTXVideo, QwenImage, etc.). Los UNet clásicos de SD/SDXL siguen usando SDPA de PyTorch directamente.

Usar `--sageattention_usage` para habilitar entrenamiento con SageAttention debe hacerse con cuidado, ya que no rastrea ni propaga gradientes desde sus implementaciones CUDA personalizadas para los linears QKV.

- Esto hace que estas capas queden completamente sin entrenar, lo que podría causar colapso del modelo o mejoras leves en entrenamientos cortos.

---

## 📰 Publicación

### `--push_to_hub`

- **Qué**: Si se proporciona, tu modelo se subirá a [Huggingface Hub](https://huggingface.co) cuando finalice el entrenamiento. Usar `--push_checkpoints_to_hub` además subirá cada checkpoint intermedio.

### `--push_to_hub_background`

- **Qué**: Sube a Hugging Face Hub desde un worker en segundo plano para que los pushes de checkpoints no pausen el bucle de entrenamiento.
- **Por qué**: Mantiene el entrenamiento y la validación en marcha mientras las subidas al Hub proceden de forma asíncrona. Las subidas finales aún se esperan antes de que la ejecución termine para que los fallos se expongan.

### `--webhook_config`

- **Qué**: Configuración para destinos de webhook (p. ej., Discord, endpoints personalizados) para recibir eventos de entrenamiento en tiempo real.
- **Por qué**: Te permite monitorear ejecuciones de entrenamiento con herramientas y paneles externos, recibiendo notificaciones en etapas clave del entrenamiento.
- **Notas**: El campo `job_id` en las cargas útiles del webhook puede poblarse configurando la variable de entorno `SIMPLETUNER_JOB_ID` antes del entrenamiento:
  ```bash
  export SIMPLETUNER_JOB_ID="my-training-run-name"
  python train.py
  ```
Esto es útil para herramientas de monitoreo que reciben webhooks de múltiples ejecuciones de entrenamiento para identificar qué configuración envió cada evento. Si SIMPLETUNER_JOB_ID no está configurado, job_id será null en las cargas útiles del webhook.

### `--publishing_config`

- **Qué**: JSON/dict/ruta de archivo opcional que describe destinos de publicación que no son de Hugging Face (almacenamiento compatible con S3, Backblaze B2, Azure Blob Storage, Dropbox).
- **Por qué**: Refleja el parseo de `--webhook_config` para que puedas distribuir artefactos más allá del Hub. La publicación se ejecuta en el proceso principal después de la validación usando el `output_dir` actual.
- **Notas**: Los proveedores se suman a `--push_to_hub`. Instala los SDK de cada proveedor (p. ej., `boto3`, `azure-storage-blob`, `dropbox`) dentro de tu `.venv` cuando los habilites. Consulta `documentation/publishing/README.md` para ejemplos completos.

### `--hub_model_id`

- **Qué**: El nombre del modelo en Huggingface Hub y el directorio de resultados locales.
- **Por qué**: Este valor se usa como nombre de directorio bajo la ubicación especificada como `--output_dir`. Si se proporciona `--push_to_hub`, este se convertirá en el nombre del modelo en Huggingface Hub.

### `--modelspec_comment`

- **Qué**: Texto incorporado en los metadatos del archivo safetensors como `modelspec.comment`
- **Por defecto**: None (deshabilitado)
- **Notas**:
  - Visible en visualizadores de modelos externos (ComfyUI, herramientas de info de modelos)
  - Acepta una cadena o un array de cadenas (unidas con saltos de línea)
  - Soporta marcadores `{env:VAR_NAME}` para sustitución de variables de entorno
  - Soporta `{current_step}`, `{current_epoch}` y `{timestamp}` cuando se escribe el metadata
  - `{timestamp}` usa un valor UTC en formato ISO 8601
  - Cada checkpoint usa el valor de configuración actual en el momento del guardado

**Ejemplo (cadena)**:
```json
"modelspec_comment": "Entrenado en mi dataset personalizado v2.1"
```

**Ejemplo (array para múltiples líneas)**:
```json
"modelspec_comment": [
  "Ejecución de entrenamiento: experiment-42",
  "Dataset: custom-portraits-v2",
  "Notas: {env:TRAINING_NOTES}"
]
```

### `--disable_benchmark`

- **Qué**: Desactiva la validación/benchmark de arranque que ocurre en el paso 0 sobre el modelo base. Estas salidas se concatenan al lado izquierdo de tus imágenes de validación del modelo entrenado.

## 📂 Almacenamiento y gestión de datos

### `--data_backend_config`

- **Qué**: Ruta a tu configuración de dataset de SimpleTuner.
- **Por qué**: Se pueden combinar múltiples datasets en distintos medios de almacenamiento en una sola sesión de entrenamiento.
- **Notas**:
  - Los valores de texto cargados desde `config.json` y `config.toml` soportan `{env:VAR_NAME}`
  - Los valores de texto dentro del `multidatabackend.json` referenciado también soportan `{env:VAR_NAME}`
- **Ejemplo**: Consulta [multidatabackend.json.example](/multidatabackend.json.example) para un ejemplo de configuración y [este documento](DATALOADER.md) para más información sobre la configuración del data loader.

### `--override_dataset_config`

- **Qué**: Cuando se proporciona, permite que SimpleTuner ignore las diferencias entre la configuración en caché dentro del dataset y los valores actuales.
- **Por qué**: Cuando SimpleTuner se ejecuta por primera vez sobre un dataset, creará un documento de caché que contiene información sobre todo en ese dataset. Esto incluye la configuración del dataset, incluyendo sus valores de configuración de "crop" y "resolution". Cambiarlos arbitrariamente o por accidente podría hacer que tus trabajos de entrenamiento fallen aleatoriamente, por lo que se recomienda encarecidamente no usar este parámetro y, en su lugar, resolver las diferencias que quieras aplicar en tu dataset por otro medio.

### `--data_backend_sampling`

- **Qué**: Al usar múltiples backends de datos, el muestreo puede hacerse con distintas estrategias.
- **Opciones**:
  - `uniform` - el comportamiento anterior de v0.9.8.1 y versiones previas donde la longitud del dataset no se consideraba, solo los pesos de probabilidad manuales.
  - `auto-weighting` - el comportamiento predeterminado donde se usa la longitud del dataset para muestrear equitativamente todos los datasets, manteniendo un muestreo uniforme de toda la distribución de datos.
    - Esto es necesario si tienes datasets de distintos tamaños y quieres que el modelo aprenda por igual.
    - Pero ajustar `repeats` manualmente es **necesario** para muestrear correctamente imágenes Dreambooth frente a tu conjunto de regularización

### `--vae_cache_scan_behaviour`

- **Qué**: Configura el comportamiento de la comprobación de escaneo de integridad.
- **Por qué**: Un dataset podría tener ajustes incorrectos aplicados en múltiples puntos del entrenamiento, p. ej., si eliminas accidentalmente los archivos de caché `.json` de tu dataset y cambias la configuración del backend de datos para usar imágenes cuadradas en lugar de crops por aspecto. Esto dará como resultado un caché de datos inconsistente, que puede corregirse configurando `scan_for_errors` en `true` en tu archivo de configuración `multidatabackend.json`. Cuando se ejecuta este escaneo, se basa en el ajuste de `--vae_cache_scan_behaviour` para determinar cómo resolver la inconsistencia: `recreate` (predeterminado) eliminará la entrada de caché problemática para que pueda recrearse, y `sync` actualizará los metadatos del bucket para reflejar la realidad de la muestra real de entrenamiento. Valor recomendado: `recreate`.

### `--dataloader_prefetch`

- **Qué**: Recupera lotes por adelantado.
- **Por qué**: Especialmente al usar lotes grandes, el entrenamiento se "pausará" mientras las muestras se recuperan del disco (incluso NVMe), lo que afecta las métricas de utilización de la GPU. Al habilitar el prefetch del dataloader se mantendrá un buffer lleno de lotes completos, de modo que puedan cargarse instantáneamente.

> ⚠️ Esto solo es realmente relevante para H100 o mejor a baja resolución donde la E/S se convierte en el cuello de botella. Para la mayoría de los otros casos, es una complejidad innecesaria.

### `--dataloader_prefetch_qlen`

- **Qué**: Aumenta o reduce el número de lotes mantenidos en memoria.
- **Por qué**: Al usar prefetch del dataloader, se mantienen 10 entradas en memoria por GPU/proceso. Esto puede ser demasiado o muy poco. Este valor puede ajustarse para incrementar el número de lotes preparados por adelantado.

### `--compress_disk_cache`

- **Qué**: Comprime los cachés en disco del VAE y de embeddings de texto.
- **Por qué**: El codificador T5 usado por DeepFloyd, SD3 y PixArt produce embeddings de texto muy grandes que terminan siendo en su mayoría espacio vacío para captions cortas o redundantes. Habilitar `--compress_disk_cache` puede reducir el espacio consumido hasta en un 75%, con ahorros promedio del 40%.

> ⚠️ Deberás eliminar manualmente los directorios de caché existentes para que puedan recrearse con compresión por el trainer.

---

## 🌈 Procesamiento de imagen y texto

Muchas configuraciones se establecen a través del [dataloader config](DATALOADER.md), pero estas se aplicarán de forma global.

### `--resolution_type`

- **Qué**: Esto le dice a SimpleTuner si debe usar cálculos de tamaño `area` o cálculos de borde `pixel`. También se admite un enfoque híbrido de `pixel_area`, que permite usar píxeles en lugar de megapíxeles para medidas de `area`.
- **Opciones**:
  - `resolution_type=pixel_area`
    - Un valor de `resolution` de 1024 se mapeará internamente a una medición de área precisa para un bucketizado eficiente por aspecto.
    - Tamaños resultantes de ejemplo para `1024`: 1024x1024, 1216x832, 832x1216
  - `resolution_type=pixel`
    - Todas las imágenes del dataset tendrán su lado menor redimensionado a esta resolución para el entrenamiento, lo que podría resultar en un gran uso de VRAM por el tamaño de las imágenes resultantes.
    - Tamaños resultantes de ejemplo para `1024`: 1024x1024, 1766x1024, 1024x1766
  - `resolution_type=area`
    - **Obsoleto**. Usa `pixel_area` en su lugar.

### `--resolution`

- **Qué**: Resolución de imagen de entrada expresada en longitud de borde en píxeles
- **Predeterminado**: 1024
- **Nota**: Este es el valor global por defecto si un dataset no tiene una resolución definida.

### `--validation_resolution`

- **Qué**: Resolución de imagen de salida, medida en píxeles o en formato `widthxheight`, como `1024x1024`. Se pueden definir múltiples resoluciones separadas por comas.
- **Por qué**: Todas las imágenes generadas durante la validación tendrán esta resolución. Útil si el modelo se entrena con una resolución distinta.

### `--validation_method`

- **Qué**: Elige cómo se ejecutan las validaciones.
- **Opciones**: `simpletuner-local` (predeterminado) ejecuta la canalización incorporada; `external-script` ejecuta un ejecutable proporcionado por el usuario.
- **Por qué**: Te permite delegar la validación a un sistema externo sin pausar el entrenamiento por el trabajo de la canalización local.

### `--validation_external_script`

- **Qué**: Ejecutable que se ejecuta cuando `--validation_method=external-script`. Usa separación estilo shell, por lo que debes entrecomillar la cadena de comando adecuadamente.
- **Marcadores**: Puedes incrustar estos tokens (formateados con `.format`) para pasar contexto de entrenamiento. Los valores faltantes se reemplazan por una cadena vacía salvo que se indique:
  - `{local_checkpoint_path}` → directorio del último checkpoint bajo `output_dir` (requiere al menos un checkpoint).
  - `{global_step}` → paso global actual.
  - `{tracker_run_name}` → valor de `--tracker_run_name`.
  - `{tracker_project_name}` → valor de `--tracker_project_name`.
  - `{model_family}` → valor de `--model_family`.
  - `{model_type}` / `{lora_type}` → tipo de modelo y variante LoRA.
  - `{huggingface_path}` → valor de `--hub_model_id` (si está configurado).
  - `{remote_checkpoint_path}` → URL remota de tu última subida (vacío para el hook de validación).
  - Cualquier valor de configuración `validation_*` (p. ej., `validation_num_inference_steps`, `validation_guidance`, `validation_noise_scheduler`).
- **Ejemplo**: `--validation_external_script="/opt/tools/validate.sh {local_checkpoint_path} {global_step}"`

### `--validation_external_background`

- **Qué**: Cuando está configurado, `--validation_external_script` se lanza en segundo plano (fire-and-forget).
- **Por qué**: Mantén el entrenamiento en marcha sin esperar al script externo; en este modo no se comprueban códigos de salida.

### `--post_upload_script`

- **Qué**: Ejecutable opcional que se ejecuta después de que cada proveedor de publicación y la subida a Hugging Face Hub termina (subidas finales del modelo y de checkpoints). Se ejecuta de forma asíncrona para que el entrenamiento no se bloquee.
- **Marcadores**: Mismas sustituciones que `--validation_external_script`, además de `{remote_checkpoint_path}` (URI devuelta por el proveedor) para que puedas reenviar la URL publicada a sistemas downstream.
- **Notas**:
  - Los scripts se ejecutan por proveedor/subida; los errores se registran pero no detienen el entrenamiento.
  - Los scripts también se invocan cuando no ocurre ninguna subida remota, por lo que puedes usarlos para automatización local (p. ej., ejecutar inferencia en otra GPU).
  - SimpleTuner no ingiere resultados de tu script; registra directamente en tu tracker si quieres métricas o imágenes.
- **Ejemplo**:
  ```bash
  --post_upload_script='/opt/hooks/notify.sh {remote_checkpoint_path} {tracker_project_name} {tracker_run_name}'
  ```
  Donde `/opt/hooks/notify.sh` podría publicar en tu sistema de tracking:
  ```bash
  #!/usr/bin/env bash
  REMOTE="$1"
  PROJECT="$2"
  RUN="$3"
  curl -X POST "https://tracker.internal/api/runs/${PROJECT}/${RUN}/artifacts" \
       -H "Content-Type: application/json" \
       -d "{\"remote_uri\":\"${REMOTE}\"}"
  ```
- **Ejemplos funcionales**:
  - `simpletuner/examples/external-validation/replicate_post_upload.py` muestra un hook de Replicate que consume `{remote_checkpoint_path}`, `{model_family}`, `{model_type}`, `{lora_type}` y `{huggingface_path}` para disparar inferencia después de las subidas.
  - `simpletuner/examples/external-validation/wavespeed_post_upload.py` muestra un hook de WaveSpeed usando los mismos marcadores más el polling asíncrono de WaveSpeed.
  - `simpletuner/examples/external-validation/fal_post_upload.py` muestra un hook de Flux LoRA en fal.ai (requiere `FAL_KEY`).
  - `simpletuner/examples/external-validation/use_second_gpu.py` ejecuta inferencia Flux LoRA en una GPU secundaria y funciona incluso sin subidas remotas.

### `--post_checkpoint_script`

- **Qué**: Ejecutable que se ejecuta inmediatamente después de que cada directorio de checkpoint se escribe en disco (antes de que comiencen las subidas). Se ejecuta de forma asíncrona en el proceso principal.
- **Marcadores**: Mismas sustituciones que `--validation_external_script`, incluyendo `{local_checkpoint_path}`, `{global_step}`, `{tracker_run_name}`, `{tracker_project_name}`, `{model_family}`, `{model_type}`, `{lora_type}`, `{huggingface_path}` y cualquier valor de configuración `validation_*`. `{remote_checkpoint_path}` se resuelve a vacío para este hook.
- **Notas**:
  - Se dispara para checkpoints programados, manuales y de rolling tan pronto como terminan de guardarse localmente.
  - Útil para lanzar automatizaciones locales (copiar a otro volumen, ejecutar jobs de evaluación) sin esperar a que las subidas terminen.
- **Ejemplo**:
  ```bash
  --post_checkpoint_script='/opt/hooks/run_eval.sh {local_checkpoint_path} {global_step}'
  ```


### `--validation_adapter_path`

- **Qué**: Carga temporalmente un único adaptador LoRA al ejecutar validaciones programadas.
- **Formatos**:
  - Repo de Hugging Face: `org/repo` o `org/repo:weight_name.safetensors` (por defecto `pytorch_lora_weights.safetensors`).
  - Ruta de archivo local o directorio que apunte a un adaptador safetensors.
- **Notas**:
  - Mutuamente excluyente con `--validation_adapter_config`; proporcionar ambos lanza un error.
  - El adaptador solo se adjunta para validaciones (los pesos base del entrenamiento permanecen intactos).

### `--validation_adapter_name`

- **Qué**: Identificador opcional que se aplica al adaptador temporal cargado vía `--validation_adapter_path`.
- **Por qué**: Controla cómo se etiqueta la ejecución del adaptador en logs/Web UI y garantiza nombres de adaptador predecibles cuando se prueban varios adaptadores en secuencia.

### `--validation_adapter_strength`

- **Qué**: Multiplicador de fuerza aplicado al habilitar el adaptador temporal (predeterminado `1.0`).
- **Por qué**: Te permite barrer escalas LoRA más ligeras/pesadas durante la validación sin alterar el estado de entrenamiento; acepta cualquier valor mayor que cero.

### `--validation_adapter_mode`

- **Opciones**: `adapter_only`, `comparison`, `none`
- **Qué**:
  - `adapter_only`: ejecuta validaciones solo con el adaptador temporal adjunto.
  - `comparison`: genera muestras tanto del modelo base como con el adaptador habilitado para revisión lado a lado.
  - `none`: omite adjuntar el adaptador (útil para desactivar la función sin borrar flags de CLI).

### `--validation_adapter_config`

- **Qué**: Archivo JSON o JSON inline que describe múltiples combinaciones de adaptadores de validación para iterar.
- **Formato**: Ya sea un array de entradas o un objeto con un array `runs`. Cada entrada puede incluir:
  - `label`: Nombre amigable mostrado en logs/UI.
  - `path`: ID de repo de Hugging Face o ruta local (mismos formatos que `--validation_adapter_path`).
  - `adapter_name`: Identificador opcional por adaptador.
  - `strength`: Override escalar opcional.
  - `adapters`/`paths`: Array de objetos/cadenas para cargar múltiples adaptadores en una sola ejecución.
- **Notas**:
  - Cuando se proporciona, las opciones de adaptador único (`--validation_adapter_path`, `--validation_adapter_name`, `--validation_adapter_strength`, `--validation_adapter_mode`) se ignoran/deshabilitan en la UI.
  - Cada ejecución se carga una a la vez y se desmonta completamente antes de comenzar la siguiente.

### `--validation_preview`

- **Qué**: Transmite vistas previas intermedias de validación durante el muestreo de difusión usando Tiny AutoEncoders
- **Predeterminado**: False
- **Por qué**: Habilita la vista previa en tiempo real de imágenes de validación a medida que se generan, decodificadas mediante modelos Tiny AutoEncoder livianos y enviadas a través de callbacks de webhook. Esto permite monitorear la progresión de las muestras de validación paso a paso en lugar de esperar a la generación completa.
- **Notas**:
  - Solo disponible en familias de modelos con soporte de Tiny AutoEncoder (p. ej., Flux, SDXL, SD3)
  - Requiere configuración de webhook para recibir imágenes de vista previa
  - Usa `--validation_preview_steps` para controlar con qué frecuencia se decodifican las vistas previas

### `--validation_preview_steps`

- **Qué**: Intervalo para decodificar y transmitir vistas previas de validación
- **Predeterminado**: 1
- **Por qué**: Controla con qué frecuencia se decodifican los latentes intermedios durante el muestreo de validación. Establecerlo en un valor más alto (p. ej., 3) reduce el overhead de ejecutar el Tiny AutoEncoder al decodificar solo cada N pasos de muestreo.
- **Ejemplo**: Con `--validation_num_inference_steps=20` y `--validation_preview_steps=5`, recibirás 4 imágenes de vista previa durante el proceso de generación (en los pasos 5, 10, 15, 20).

### `--evaluation_type`

- **Qué**: Habilita la evaluación CLIP de imágenes generadas durante las validaciones.
- **Por qué**: Los puntajes CLIP calculan la distancia de las características de la imagen generada al prompt de validación proporcionado. Esto puede dar una idea de si la adherencia al prompt mejora, aunque requiere un gran número de prompts de validación para tener valor significativo.
- **Opciones**: "none" o "clip"
- **Programación**: Usa `--eval_steps_interval` para programación por pasos o `--eval_epoch_interval` para programación por épocas (fracciones como `0.5` se ejecutan varias veces por época). Si ambos están configurados, el trainer registra una advertencia y ejecuta ambos calendarios.

- **Programación**: Usa `--eval_steps_interval` para programación por pasos o `--eval_epoch_interval` para programación por épocas (fracciones como `0.5` se ejecutan varias veces por época). Si ambos están configurados, el trainer registra una advertencia y ejecuta ambos calendarios.

### `--eval_loss_disable`

- **Qué**: Desactiva el cálculo de pérdida de evaluación durante la validación.
- **Por qué**: Cuando se configura un dataset de eval, la pérdida se calcula automáticamente. Si la evaluación CLIP también está habilitada, ambas se ejecutarán. Este flag te permite desactivar selectivamente la pérdida de eval manteniendo la evaluación CLIP habilitada.

### `--validation_using_datasets`

- **Qué**: Usa imágenes de datasets de entrenamiento para validación en lugar de generación pura de texto a imagen.
- **Por qué**: Habilita el modo de validación imagen-a-imagen (img2img) o imagen-a-video (i2v) donde el modelo usa imágenes de entrenamiento como entradas de conditioning. Útil para:
  - Probar modelos de edición/inpainting que requieren imágenes de entrada
  - Evaluar qué tan bien el modelo preserva la estructura de imagen
  - Modelos que soportan flujos duales texto-a-imagen E imagen-a-imagen (ej., Flux2, LTXVideo2)
  - **Modelos de video I2V** (HunyuanVideo, WAN, Kandinsky5Video): Usa imágenes de un dataset de imágenes como entrada de conditioning del primer frame para validación de generación de video
- **Notas**:
  - Requiere que el modelo tenga un pipeline `IMG2IMG` o `IMG2VIDEO` registrado
  - Puede combinarse con `--eval_dataset_id` para obtener imágenes de un dataset específico
  - Para modelos i2v, permite usar un dataset de imágenes simple para validación sin la configuración compleja de emparejamiento de datasets de conditioning usada durante el entrenamiento
  - La fuerza de des-ruido se controla con los ajustes normales de timestep de validación

### `--eval_dataset_id`

- **Qué**: ID de dataset específico a usar para obtener imágenes de evaluación/validación.
- **Por qué**: Al usar `--validation_using_datasets` o validación basada en conditioning, controla qué dataset provee las imágenes de entrada:
  - Sin esta opción, las imágenes se seleccionan aleatoriamente de todos los datasets de entrenamiento
  - Con esta opción, solo se usa el dataset especificado para entradas de validación
- **Notas**:
  - El ID de dataset debe coincidir con un dataset configurado en tu config de dataloader
  - Útil para mantener evaluación consistente usando un dataset de eval dedicado
  - Para modelos de conditioning, los datos de conditioning del dataset (si existen) también se usarán

---

## Entendiendo Modos de Conditioning y Validación

SimpleTuner soporta tres paradigmas principales para modelos que usan entradas de conditioning (imágenes de referencia, señales de control, etc.):

### 1. Modelos que REQUIEREN Conditioning

Algunos modelos no pueden funcionar sin entradas de conditioning:

- **Flux Kontext**: Siempre necesita imágenes de referencia para entrenamiento estilo edición
- **Entrenamiento ControlNet**: Requiere imágenes de señal de control

Para estos modelos, un dataset de conditioning es obligatorio. La WebUI mostrará opciones de conditioning como requeridas, y el entrenamiento fallará sin ellas.

### 2. Modelos que SOPORTAN Conditioning Opcional

Algunos modelos pueden operar en modos texto-a-imagen E imagen-a-imagen:

- **Flux2**: Soporta entrenamiento dual T2I/I2I con imágenes de referencia opcionales
- **LTXVideo2**: Soporta T2V e I2V (imagen-a-video) con conditioning de primer frame opcional
- **LongCat-Video**: Soporta conditioning de frames opcional
- **HunyuanVideo i2v**: Soporta I2V con conditioning de primer frame (flavours: `i2v-480p`, `i2v-720p`, etc.)
- **WAN i2v**: Soporta I2V con conditioning de primer frame
- **Kandinsky5Video i2v**: Soporta I2V con conditioning de primer frame

Para estos modelos, PUEDES agregar datasets de conditioning pero no es obligatorio. La WebUI mostrará opciones de conditioning como opcionales.

**Atajo de Validación I2V**: Para modelos de video i2v, puedes usar `--validation_using_datasets` con un dataset de imágenes (especificado via `--eval_dataset_id`) para obtener imágenes de conditioning de validación directamente, sin necesidad de configurar el emparejamiento completo de datasets de conditioning usado durante el entrenamiento.

### 3. Modos de Validación

| Modo | Flag | Comportamiento |
|------|------|----------------|
| **Texto-a-Imagen/Video** | (por defecto) | Genera solo desde prompts de texto |
| **Basado en Dataset (img2img)** | `--validation_using_datasets` | Des-ruido parcial de imágenes de datasets |
| **Basado en Dataset (i2v)** | `--validation_using_datasets` | Para modelos de video i2v, usa imágenes como conditioning de primer frame |
| **Basado en Conditioning** | (auto cuando se configura conditioning) | Usa entradas de conditioning durante validación |

**Combinando modos**: Cuando un modelo soporta conditioning Y `--validation_using_datasets` está habilitado:
- El sistema de validación obtiene imágenes de datasets
- Si esos datasets tienen datos de conditioning, se usan automáticamente
- Usa `--eval_dataset_id` para controlar qué dataset provee entradas

**Modelos I2V con `--validation_using_datasets`**: Para modelos de video i2v (HunyuanVideo, WAN, Kandinsky5Video), habilitar este flag permite usar un dataset de imágenes simple para validación. Las imágenes se usan como entradas de conditioning de primer frame para generar videos de validación, sin necesidad de la configuración compleja de emparejamiento de datasets de conditioning.

### Tipos de Datos de Conditioning

Diferentes modelos esperan diferentes datos de conditioning:

| Tipo | Modelos | Configuración de Dataset |
|------|---------|-------------------------|
| `conditioning` | ControlNet, Control | `type: conditioning` en config de dataset |
| `image` | Flux Kontext | `type: image` (dataset de imagen estándar) |
| `latents` | Flux, Flux2 | Conditioning se codifica con VAE automáticamente |

---

### `--caption_strategy`

- **Qué**: Estrategia para derivar captions de imagen. **Opciones**: `textfile`, `filename`, `parquet`, `instanceprompt`
- **Por qué**: Determina cómo se generan los captions para imágenes de entrenamiento.
  - `textfile` usará el contenido de un archivo `.txt` con el mismo nombre que la imagen
  - `filename` aplicará una limpieza al nombre del archivo antes de usarlo como caption.
  - `parquet` requiere un archivo parquet presente en el dataset y usará la columna `caption` como caption salvo que se proporcione `parquet_caption_column`. Todas las captions deben estar presentes a menos que se proporcione `parquet_fallback_caption_column`.
  - `instanceprompt` usará el valor de `instance_prompt` en la configuración del dataset como prompt para cada imagen del dataset.

### `--conditioning_multidataset_sampling` {#--conditioning_multidataset_sampling}

- **Qué**: Cómo muestrear desde múltiples datasets de condicionamiento. **Opciones**: `combined`, `random`
- **Por qué**: Al entrenar con múltiples datasets de condicionamiento (p. ej., múltiples imágenes de referencia o señales de control), esto determina cómo se usan:
  - `combined` une las entradas de condicionamiento, mostrándolas simultáneamente durante el entrenamiento. Útil para tareas de composición multi-imagen.
  - `random` selecciona aleatoriamente un dataset de condicionamiento por muestra, alternando condiciones durante el entrenamiento.
- **Nota**: Al usar `combined`, no puedes definir `captions` separados en datasets de condicionamiento; se usan las captions del dataset fuente.
- **Ver también**: [DATALOADER.md](DATALOADER.md#conditioning_data) para configurar múltiples datasets de condicionamiento.

---

## 🎛 Parámetros de entrenamiento

### `--num_train_epochs`

- **Qué**: Número de épocas de entrenamiento (el número de veces que se ven todas las imágenes). Configurar esto en 0 permitirá que `--max_train_steps` tenga prioridad.
- **Por qué**: Determina el número de repeticiones de imagen, lo que impacta la duración del proceso de entrenamiento. Más épocas tienden a resultar en sobreajuste, pero podrían ser necesarias para aprender los conceptos que deseas entrenar. Un valor razonable podría estar entre 5 y 50.

### `--max_train_steps`

- **Qué**: Número de pasos de entrenamiento tras los cuales salir del entrenamiento. Si se establece en 0, permitirá que `--num_train_epochs` tenga prioridad.
- **Por qué**: Útil para acortar la duración del entrenamiento.

### `--ignore_final_epochs`

- **Qué**: Ignora las últimas épocas contadas en favor de `--max_train_steps`.
- **Por qué**: Al cambiar la longitud del dataloader, el entrenamiento puede terminar antes de lo que quieres porque el cálculo de épocas cambia. Esta opción ignorará las épocas finales y en su lugar continuará entrenando hasta que se alcance `--max_train_steps`.

### `--learning_rate`

- **Qué**: Tasa de aprendizaje inicial tras el posible warmup.
- **Por qué**: La tasa de aprendizaje se comporta como un "tamaño de paso" para las actualizaciones de gradiente: demasiado alta y nos pasamos de la solución; demasiado baja y nunca llegamos a la solución ideal. Un valor mínimo para un ajuste `full` puede ser tan bajo como `1e-7` hasta un máximo de `1e-6`, mientras que para ajuste `lora` un valor mínimo podría ser `1e-5` con un máximo tan alto como `1e-3`. Cuando se usa una tasa de aprendizaje más alta, es ventajoso usar una red EMA con warmup de tasa de aprendizaje; consulta `--use_ema`, `--lr_warmup_steps` y `--lr_scheduler`.

### `--lr_scheduler`

- **Qué**: Cómo escalar la tasa de aprendizaje en el tiempo.
- **Opciones**: constant, constant_with_warmup, cosine, cosine_with_restarts, **polynomial** (recomendado), linear
- **Por qué**: Los modelos se benefician de ajustes continuos de la tasa de aprendizaje para explorar mejor el paisaje de pérdida. Un calendario cosine se usa como predeterminado; permite que el entrenamiento transicione suavemente entre dos extremos. Si se usa una tasa constante, es común seleccionar un valor demasiado alto o demasiado bajo, causando divergencia (demasiado alto) o quedándose atrapado en un mínimo local (demasiado bajo). Un calendario polynomial se combina mejor con un warmup, donde se acercará gradualmente al valor `learning_rate` antes de ralentizarse y aproximarse a `--lr_end` al final.

### `--optimizer`

- **Qué**: El optimizador a usar para el entrenamiento.
- **Opciones**: adamw_bf16, ao-adamw8bit, ao-adamw4bit, ao-adamfp8, ao-adamwfp8, adamw_schedulefree, adamw_schedulefree+aggressive, adamw_schedulefree+no_kahan, optimi-stableadamw, optimi-adamw, optimi-lion, optimi-radam, optimi-ranger, optimi-adan, optimi-adam, optimi-sgd, soap, bnb-adagrad, bnb-adagrad8bit, bnb-adam, bnb-adam8bit, bnb-adamw, bnb-adamw8bit, bnb-adamw-paged, bnb-adamw8bit-paged, bnb-lion, bnb-lion8bit, bnb-lion-paged, bnb-lion8bit-paged, bnb-ademamix, bnb-ademamix8bit, bnb-ademamix-paged, bnb-ademamix8bit-paged, prodigy

> Nota: Algunos optimizadores pueden no estar disponibles en hardware no NVIDIA.

### `--optimizer_config`

- **Qué**: Ajusta la configuración del optimizador.
- **Por qué**: Como los optimizadores tienen tantas configuraciones diferentes, no es viable proporcionar un argumento de línea de comandos para cada una. En su lugar, puedes proporcionar una lista separada por comas de valores para sobrescribir cualquiera de los valores predeterminados.
- **Ejemplo**: Tal vez quieras establecer `d_coef` para el optimizador **prodigy**: `--optimizer_config=d_coef=0.1`

> Nota: Las betas del optimizador se sobrescriben usando parámetros dedicados, `--optimizer_beta1`, `--optimizer_beta2`.

### `--train_batch_size`

- **Qué**: Tamaño de batch para el data loader de entrenamiento.
- **Por qué**: Afecta el consumo de memoria del modelo, la calidad de convergencia y la velocidad de entrenamiento. Cuanto mayor sea el batch size, mejores serán los resultados, pero un batch muy alto puede resultar en sobreajuste o entrenamiento desestabilizado, además de aumentar innecesariamente la duración de la sesión. La experimentación está justificada, pero en general, quieres maximizar la memoria de video sin reducir la velocidad de entrenamiento.

### `--gradient_accumulation_steps`

- **Qué**: Número de pasos de actualización a acumular antes de realizar un pase hacia atrás/actualización, esencialmente dividiendo el trabajo en múltiples lotes para ahorrar memoria a costa de mayor tiempo de entrenamiento.
- **Por qué**: Útil para manejar modelos o datasets más grandes.

> Nota: No habilites el pase backward fusionado para ningún optimizador al usar pasos de acumulación de gradiente.

### `--allow_dataset_oversubscription` {#--allow_dataset_oversubscription}

- **Qué**: Ajusta automáticamente los `repeats` del dataset cuando el dataset es más pequeño que el tamaño de batch efectivo.
- **Por qué**: Evita fallos de entrenamiento cuando el tamaño de tu dataset no cumple los requisitos mínimos de tu configuración multi-GPU.
- **Cómo funciona**:
  - Calcula el **tamaño de batch efectivo**: `train_batch_size × num_gpus × gradient_accumulation_steps`
  - Si cualquier bucket de aspecto tiene menos muestras que el tamaño de batch efectivo, incrementa automáticamente `repeats`
  - Solo aplica cuando `repeats` no está configurado explícitamente en tu configuración de dataset
  - Registra una advertencia mostrando el ajuste y el razonamiento
- **Casos de uso**:
  - Datasets pequeños (< 100 imágenes) con múltiples GPUs
  - Experimentar con distintos tamaños de batch sin reconfigurar datasets
  - Prototipado antes de recopilar un dataset completo
- **Ejemplo**: Con 25 imágenes, 8 GPUs y `train_batch_size=4`, el tamaño de batch efectivo es 32. Este flag establecería automáticamente `repeats=1` para proporcionar 50 muestras (25 × 2).
- **Nota**: Esto **no** sobrescribirá valores `repeats` configurados manualmente en tu configuración de dataloader. Similar a `--disable_bucket_pruning`, este flag ofrece conveniencia sin comportamiento sorprendente.

Consulta la guía [DATALOADER.md](DATALOADER.md#automatic-dataset-oversubscription) para más detalles sobre el tamaño de dataset para entrenamiento multi-GPU.

---

## 🛠 Optimizaciones avanzadas

### `--use_ema`

- **Qué**: Mantener una media móvil exponencial de tus pesos durante la vida de entrenamiento del modelo es como fusionar periódicamente el modelo en sí mismo.
- **Por qué**: Puede mejorar la estabilidad del entrenamiento a costa de más recursos del sistema y un ligero incremento en el tiempo de entrenamiento.

### `--ema_device`

- **Opciones**: `cpu`, `accelerator`; predeterminado: `cpu`
- **Qué**: Elige dónde viven los pesos EMA entre actualizaciones.
- **Por qué**: Mantener la EMA en el acelerador da las actualizaciones más rápidas pero consume VRAM. Mantenerla en CPU reduce la presión de memoria pero requiere mover pesos a menos que se configure `--ema_cpu_only`.

### `--ema_cpu_only`

- **Qué**: Evita que los pesos EMA se muevan de vuelta al acelerador para actualizaciones cuando `--ema_device=cpu`.
- **Por qué**: Ahorra el tiempo de transferencia host-dispositivo y el uso de VRAM para EMAs grandes. No tiene efecto si `--ema_device=accelerator` porque los pesos ya residen en el acelerador.

### `--ema_foreach_disable`

- **Qué**: Desactiva el uso de kernels `torch._foreach_*` para actualizaciones EMA.
- **Por qué**: Algunos backends o combinaciones de hardware tienen problemas con ops foreach. Deshabilitarlos vuelve a la implementación escalar a costa de actualizaciones ligeramente más lentas.

### `--ema_update_interval`

- **Qué**: Reduce la frecuencia con la que se actualizan los parámetros sombra EMA.
- **Por qué**: Actualizar en cada paso es innecesario para muchos flujos de trabajo. Por ejemplo, `--ema_update_interval=100` solo realiza una actualización EMA cada 100 pasos de optimizador, reduciendo overhead cuando `--ema_device=cpu` o `--ema_cpu_only` está habilitado.

### `--ema_decay`

- **Qué**: Controla el factor de suavizado usado al aplicar actualizaciones EMA.
- **Por qué**: Valores más altos (p. ej., `0.999`) hacen que la EMA responda lentamente pero produce pesos muy estables. Valores más bajos (p. ej., `0.99`) se adaptan más rápido a nuevas señales de entrenamiento.

### `--snr_gamma`

- **Qué**: Utiliza un factor de pérdida ponderado por min-SNR.
- **Por qué**: La gamma de SNR mínima pondera el factor de pérdida de un timestep por su posición en el calendario. Los timesteps muy ruidosos reducen su contribución y los menos ruidosos la aumentan. El valor recomendado por el paper original es **5**, pero puedes usar valores tan bajos como **1** o tan altos como **20**, típicamente el máximo; más allá de 20, las matemáticas no cambian mucho. Un valor de **1** es el más fuerte.

### `--use_soft_min_snr`

- **Qué**: Entrena un modelo usando un ponderado más gradual en el paisaje de pérdida.
- **Por qué**: Al entrenar modelos de difusión en píxeles, simplemente se degradarán sin usar una programación específica de ponderación de pérdida. Este es el caso con DeepFloyd, donde se encontró que soft-min-snr-gamma era esencialmente obligatorio para buenos resultados. Puedes tener éxito con entrenamiento de modelos de difusión latente, pero en experimentos pequeños se encontró que potencialmente produce resultados borrosos.

### `--diff2flow_enabled`

- **Qué**: Habilita el puente Diffusion-to-Flow para modelos epsilon o v-prediction.
- **Por qué**: Permite que los modelos entrenados con objetivos de difusión estándar usen objetivos de flow-matching (ruido - latentes) sin cambiar la arquitectura del modelo.
- **Nota**: Función experimental.

### `--diff2flow_loss`

- **Qué**: Entrena con pérdida de Flow Matching en lugar de la pérdida nativa de predicción.
- **Por qué**: Cuando se habilita junto con `--diff2flow_enabled`, calcula la pérdida contra el objetivo de flujo (ruido - latentes) en lugar del objetivo nativo del modelo (epsilon o velocidad).
- **Nota**: Requiere `--diff2flow_enabled`.

### `--scheduled_sampling_max_step_offset`

- **Qué**: Número máximo de pasos para hacer "rollout" durante el entrenamiento.
- **Por qué**: Habilita Scheduled Sampling (Rollout), donde el modelo genera sus propias entradas durante algunos pasos en el entrenamiento. Esto ayuda al modelo a aprender a corregir sus propios errores y reduce el sesgo de exposición.
- **Predeterminado**: 0 (desactivado). Configura un entero positivo (p. ej., 5 o 10) para habilitarlo.

### `--scheduled_sampling_strategy`

- **Qué**: Estrategia para elegir el offset de rollout.
- **Opciones**: `uniform`, `biased_early`, `biased_late`.
- **Predeterminado**: `uniform`.
- **Por qué**: Controla la distribución de longitudes de rollout. `uniform` muestrea uniformemente; `biased_early` favorece rollouts más cortos; `biased_late` favorece rollouts más largos.

### `--scheduled_sampling_probability`

- **Qué**: Probabilidad de aplicar un offset de rollout no cero para una muestra dada.
- **Predeterminado**: 0.0.
- **Por qué**: Controla con qué frecuencia se aplica scheduled sampling. Un valor de 0.0 lo desactiva incluso si `max_step_offset` > 0. Un valor de 1.0 lo aplica a cada muestra.

### `--scheduled_sampling_prob_start`

- **Qué**: Probabilidad inicial para scheduled sampling al inicio del ramp.
- **Predeterminado**: 0.0.

### `--scheduled_sampling_prob_end`

- **Qué**: Probabilidad final para scheduled sampling al final del ramp.
- **Predeterminado**: 0.5.

### `--scheduled_sampling_ramp_steps`

- **Qué**: Número de pasos para aumentar la probabilidad de `prob_start` a `prob_end`.
- **Predeterminado**: 0 (sin ramp).

### `--scheduled_sampling_start_step`

- **Qué**: Paso global para iniciar el ramp de scheduled sampling.
- **Predeterminado**: 0.0.

### `--scheduled_sampling_ramp_shape`

- **Qué**: Forma del ramp de probabilidad.
- **Opciones**: `linear`, `cosine`.
- **Predeterminado**: `linear`.

### `--scheduled_sampling_sampler`

- **Qué**: El solver usado para los pasos de generación de rollout.
- **Opciones**: `unipc`, `euler`, `dpm`, `rk4`.
- **Predeterminado**: `unipc`.

### `--scheduled_sampling_order`

- **Qué**: El orden del solver usado para el rollout.
- **Predeterminado**: 2.

### `--scheduled_sampling_reflexflow`

- **Qué**: Habilita mejoras estilo ReflexFlow (anti-drift + ponderación compensada por frecuencia) durante el scheduled sampling para modelos de flow-matching.
- **Por qué**: Reduce el sesgo de exposición al hacer rollout en modelos de flow-matching añadiendo regularización direccional y ponderación de pérdida consciente del sesgo.
- **Predeterminado**: Se habilita automáticamente para modelos de flow-matching cuando `--scheduled_sampling_max_step_offset` > 0; sobrescribe con `--scheduled_sampling_reflexflow=false`.

### `--scheduled_sampling_reflexflow_alpha`

- **Qué**: Factor de escala para el peso de compensación de frecuencia derivado del sesgo de exposición.
- **Predeterminado**: 1.0.
- **Por qué**: Valores más altos aumentan el peso de las regiones con mayor sesgo de exposición durante el rollout en modelos de flow-matching.

### `--scheduled_sampling_reflexflow_beta1`

- **Qué**: Peso para el regularizador anti-drift (direccional) de ReflexFlow.
- **Predeterminado**: 10.0.
- **Por qué**: Controla cuán fuertemente se incentiva al modelo a alinear su dirección predicha con la muestra limpia objetivo al usar scheduled sampling en modelos de flow-matching.

### `--scheduled_sampling_reflexflow_beta2`

- **Qué**: Peso para el término de compensación de frecuencia (reponderación de pérdida) de ReflexFlow.
- **Predeterminado**: 1.0.
- **Por qué**: Escala la pérdida de flow-matching reponderada, coincidiendo con el control β₂ descrito en el paper de ReflexFlow.

---

## 🎯 CREPA (Cross-frame Representation Alignment)

CREPA es una técnica de regularización para fine-tuning de modelos de difusión de video que mejora la consistencia temporal al alinear estados ocultos con características visuales preentrenadas de fotogramas adyacentes. Basado en el paper ["Cross-Frame Representation Alignment for Fine-Tuning Video Diffusion Models"](https://arxiv.org/abs/2506.09229).

### `--crepa_enabled`

- **Qué**: Habilita la regularización CREPA durante el entrenamiento.
- **Por qué**: Mejora la consistencia semántica entre fotogramas de video al alinear estados ocultos DiT con características DINOv2 de fotogramas vecinos.
- **Predeterminado**: `false`
- **Nota**: Solo aplica a modelos de difusión basados en Transformer (estilo DiT). Para modelos UNet (SDXL, SD1.5, Kolors), usa U-REPA.

### `--crepa_block_index`

- **Qué**: Qué bloque del transformer usar para estados ocultos para el alineamiento.
- **Por qué**: El paper recomienda el bloque 8 para CogVideoX y el bloque 10 para Hunyuan Video. Los bloques más tempranos tienden a funcionar mejor ya que actúan como la porción "encoder" del DiT.
- **Requerido**: Sí, cuando CREPA está habilitado.

### `--crepa_lambda`

- **Qué**: Peso de la pérdida de alineamiento CREPA relativo a la pérdida principal de entrenamiento.
- **Por qué**: Controla cuán fuertemente la regularización de alineamiento influye en el entrenamiento. El paper usa 0.5 para CogVideoX y 1.0 para Hunyuan Video.
- **Predeterminado**: `0.5`

### `--crepa_adjacent_distance`

- **Qué**: Distancia `d` para el alineamiento con fotogramas vecinos.
- **Por qué**: Según la ecuación 6 del paper, $K = \{f-d, f+d\}$ define qué fotogramas vecinos alinear. Con `d=1`, cada fotograma se alinea con sus vecinos inmediatos.
- **Predeterminado**: `1`

### `--crepa_adjacent_tau`

- **Qué**: Coeficiente de temperatura para la ponderación exponencial por distancia.
- **Por qué**: Controla qué tan rápido decae el peso de alineamiento con la distancia entre fotogramas vía $e^{-|k-f|/\tau}$. Valores más bajos se enfocan más en vecinos inmediatos.
- **Predeterminado**: `1.0`

### `--crepa_cumulative_neighbors`

- **Qué**: Usa modo acumulativo en lugar de modo adyacente.
- **Por qué**:
  - **Modo adyacente (predeterminado)**: Solo alinea con fotogramas a distancia exacta `d` (coincide con $K = \{f-d, f+d\}$ del paper)
  - **Modo acumulativo**: Alinea con todos los fotogramas desde distancia 1 hasta `d`, proporcionando gradientes más suaves
- **Predeterminado**: `false`

### `--crepa_normalize_neighbour_sum`

- **Qué**: Normaliza la suma de vecinos por la suma de pesos por fotograma.
- **Por qué**: Mantiene `crepa_alignment_score` en [-1, 1] y hace la escala de la pérdida más literal. Desviación experimental de la ecuación (6) del paper.
- **Predeterminado**: `false`

### `--crepa_normalize_by_frames`

- **Qué**: Normaliza la pérdida de alineamiento por el número de fotogramas.
- **Por qué**: Garantiza una escala de pérdida consistente sin importar la longitud del video. Deshabilítalo para dar a videos más largos una señal de alineamiento más fuerte.
- **Predeterminado**: `true`

### `--crepa_spatial_align`

- **Qué**: Usa interpolación espacial cuando los recuentos de tokens difieren entre DiT y el encoder.
- **Por qué**: Los estados ocultos DiT y las características DINOv2 pueden tener diferentes resoluciones espaciales. Cuando está habilitado, la interpolación bilineal los alinea espacialmente. Cuando está deshabilitado, recurre a un pooling global.
- **Predeterminado**: `true`

### `--crepa_model`

- **Qué**: Qué encoder preentrenado usar para extracción de características.
- **Por qué**: El paper usa DINOv2-g (ViT-Giant). Variantes más pequeñas como `dinov2_vitb14` usan menos memoria.
- **Predeterminado**: `dinov2_vitg14`
- **Opciones**: `dinov2_vitg14`, `dinov2_vitb14`, `dinov2_vits14`

### `--crepa_encoder_frames_batch_size`

- **Qué**: Cuántos fotogramas procesa en paralelo el encoder de características externo. Cero o negativo para todos los fotogramas del batch completo simultáneamente. Si el número no es divisor, el resto se manejará como un batch más pequeño.
- **Por qué**: Dado que los encoders tipo DINO son modelos de imagen, pueden procesar fotogramas en batches troceados para menor uso de VRAM a costa de velocidad.
- **Predeterminado**: `-1`

### `--crepa_use_backbone_features`

- **Qué**: Omite el encoder externo y alinea un bloque estudiante con un bloque maestro dentro del modelo de difusión.
- **Por qué**: Evita cargar DINOv2 cuando el backbone ya tiene una capa semántica más fuerte para supervisar.
- **Predeterminado**: `false`

### `--crepa_teacher_block_index`

- **Qué**: Índice del bloque maestro al usar características del backbone.
- **Por qué**: Te permite alinear un bloque estudiante temprano con un bloque maestro más profundo sin un encoder externo. Si no se establece, cae en el bloque estudiante.
- **Predeterminado**: Usa `crepa_block_index` si no se proporciona.

### `--crepa_encoder_image_size`

- **Qué**: Resolución de entrada para el encoder.
- **Por qué**: Los modelos DINOv2 funcionan mejor a su resolución de entrenamiento. El modelo gigante usa 518x518.
- **Predeterminado**: `518`

### `--crepa_scheduler`

- **Qué**: Programa para el decaimiento del coeficiente CREPA durante el entrenamiento.
- **Por qué**: Permite reducir la fuerza de regularización CREPA a medida que avanza el entrenamiento, previniendo el sobreajuste en características profundas del encoder.
- **Opciones**: `constant`, `linear`, `cosine`, `polynomial`
- **Predeterminado**: `constant`

### `--crepa_warmup_steps`

- **Qué**: Número de pasos para incrementar linealmente el peso CREPA desde 0 hasta `crepa_lambda`.
- **Por qué**: Un calentamiento gradual puede ayudar a estabilizar el entrenamiento temprano antes de que la regularización CREPA entre en efecto.
- **Predeterminado**: `0`

### `--crepa_decay_steps`

- **Qué**: Pasos totales para el decaimiento (después del calentamiento). Establece a 0 para decaer durante todo el entrenamiento.
- **Por qué**: Controla la duración de la fase de decaimiento. El decaimiento comienza después de que se completa el calentamiento.
- **Predeterminado**: `0` (usa `max_train_steps`)

### `--crepa_lambda_end`

- **Qué**: Peso CREPA final después de que se completa el decaimiento.
- **Por qué**: Establecerlo a 0 desactiva efectivamente CREPA al final del entrenamiento, útil para text2video donde CREPA puede causar artefactos.
- **Predeterminado**: `0.0`

### `--crepa_power`

- **Qué**: Factor de potencia para el decaimiento polinomial. 1.0 = lineal, 2.0 = cuadrático, etc.
- **Por qué**: Valores más altos causan un decaimiento inicial más rápido que se ralentiza hacia el final.
- **Predeterminado**: `1.0`

### `--crepa_cutoff_step`

- **Qué**: Paso de corte duro después del cual CREPA se desactiva.
- **Por qué**: Útil para desactivar CREPA después de que el modelo ha convergido en el alineamiento temporal.
- **Predeterminado**: `0` (sin corte basado en pasos)

### `--crepa_similarity_threshold`

- **Qué**: Umbral de EMA de similitud en el cual se activa el corte de CREPA.
- **Por qué**: Cuando el promedio móvil exponencial del puntaje de alineamiento (`crepa_alignment_score`) alcanza este valor, CREPA se desactiva para prevenir el sobreajuste en características profundas del encoder. Esto es particularmente útil para entrenamiento text2video. El puntaje puede superar 1.0 si no se habilita `crepa_normalize_neighbour_sum`.
- **Predeterminado**: None (desactivado)

### `--crepa_similarity_ema_decay`

- **Qué**: Factor de decaimiento del promedio móvil exponencial para el seguimiento de similitud.
- **Por qué**: Valores más altos proporcionan un seguimiento más suave (0.99 ≈ ventana de 100 pasos), valores más bajos reaccionan más rápido a los cambios.
- **Predeterminado**: `0.99`

### `--crepa_threshold_mode`

- **Qué**: Comportamiento cuando se alcanza el umbral de similitud.
- **Opciones**: `permanent` (CREPA permanece desactivado una vez que se alcanza el umbral), `recoverable` (CREPA se reactiva si la similitud cae)
- **Predeterminado**: `permanent`

### Ejemplo de configuración

```toml
# Habilitar CREPA para fine-tuning de video
crepa_enabled = true
crepa_block_index = 8          # Ajusta según tu modelo
crepa_lambda = 0.5
crepa_adjacent_distance = 1
crepa_adjacent_tau = 1.0
crepa_cumulative_neighbors = false
crepa_normalize_neighbour_sum = false
crepa_normalize_by_frames = true
crepa_spatial_align = true
crepa_model = "dinov2_vitg14"
crepa_encoder_frames_batch_size = -1
crepa_use_backbone_features = false
# crepa_teacher_block_index = 16
crepa_encoder_image_size = 518

# Programación CREPA (opcional)
# crepa_scheduler = "cosine"           # Tipo de decaimiento: constant, linear, cosine, polynomial
# crepa_warmup_steps = 100             # Calentamiento antes de que CREPA entre en efecto
# crepa_decay_steps = 1000             # Pasos para el decaimiento (0 = todo el entrenamiento)
# crepa_lambda_end = 0.0               # Peso final después del decaimiento
# crepa_cutoff_step = 5000             # Paso de corte duro (0 = desactivado)
# crepa_similarity_threshold = 0.9    # Corte basado en similitud
# crepa_threshold_mode = "permanent"   # permanent o recoverable
```

---

## 🎯 U-REPA (Alineamiento de Representaciones para UNet)

U-REPA es una técnica de regularización para modelos de difusión basados en UNet (SDXL, SD1.5, Kolors). Alinea las características del bloque medio de la UNet con características visuales preentrenadas y añade una pérdida de manifold para preservar la estructura de similitud relativa.

### `--urepa_enabled`

- **Qué**: Habilita la regularización U-REPA durante el entrenamiento.
- **Por qué**: Añade alineamiento de representaciones para el bloque medio de la UNet usando un encoder visual congelado.
- **Predeterminado**: `false`
- **Nota**: Solo aplica a modelos UNet (SDXL, SD1.5, Kolors).

### `--urepa_lambda`

- **Qué**: Peso de la pérdida de alineamiento U-REPA relativo a la pérdida principal.
- **Por qué**: Controla la fuerza de la regularización.
- **Predeterminado**: `0.5`

### `--urepa_manifold_weight`

- **Qué**: Peso de la pérdida de manifold relativo a la pérdida de alineamiento.
- **Por qué**: Prioriza la estructura de similitud relativa (valor por defecto del paper: 3.0).
- **Predeterminado**: `3.0`

### `--urepa_model`

- **Qué**: Identificador de torch hub para el encoder visual congelado.
- **Por qué**: Por defecto DINOv2 ViT-G/14; modelos más pequeños (p. ej., `dinov2_vits14`) son más rápidos.
- **Predeterminado**: `dinov2_vitg14`

### `--urepa_encoder_image_size`

- **Qué**: Resolución de entrada para el preprocesamiento del encoder.
- **Por qué**: Usa la resolución nativa del encoder (518 para DINOv2 ViT-G/14; 224 para ViT-S/14).
- **Predeterminado**: `518`

### `--urepa_use_tae`

- **Qué**: Usa Tiny AutoEncoder en lugar del VAE completo para decodificar.
- **Por qué**: Más rápido y usa menos VRAM, pero con menor calidad de decodificación.
- **Predeterminado**: `false`

### `--urepa_scheduler`

- **Qué**: Programa de decaimiento del coeficiente U-REPA durante el entrenamiento.
- **Por qué**: Permite reducir la fuerza de la regularización a medida que avanza el entrenamiento.
- **Opciones**: `constant`, `linear`, `cosine`, `polynomial`
- **Predeterminado**: `constant`

### `--urepa_warmup_steps`

- **Qué**: Número de pasos para aumentar linealmente el peso de 0 a `urepa_lambda`.
- **Por qué**: El warmup ayuda a estabilizar el inicio del entrenamiento.
- **Predeterminado**: `0`

### `--urepa_decay_steps`

- **Qué**: Pasos totales de decaimiento (después del warmup). 0 significa decaer durante todo el entrenamiento.
- **Por qué**: Controla la duración de la fase de decaimiento.
- **Predeterminado**: `0` (usa `max_train_steps`)

### `--urepa_lambda_end`

- **Qué**: Peso final de U-REPA después del decaimiento.
- **Por qué**: 0 desactiva efectivamente U-REPA al final del entrenamiento.
- **Predeterminado**: `0.0`

### `--urepa_power`

- **Qué**: Exponente del decaimiento polinomial. 1.0 = lineal, 2.0 = cuadrático, etc.
- **Por qué**: Valores mayores decaen más rápido al inicio y más lento al final.
- **Predeterminado**: `1.0`

### `--urepa_cutoff_step`

- **Qué**: Paso de corte después del cual se desactiva U-REPA.
- **Por qué**: Útil para apagar U-REPA después de que el modelo converge en el alineamiento.
- **Predeterminado**: `0` (sin corte)

### `--urepa_similarity_threshold`

- **Qué**: Umbral de similitud (EMA) para desactivar U-REPA.
- **Por qué**: Cuando el promedio móvil exponencial de `urepa_similarity` alcanza este valor, U-REPA se desactiva para evitar sobreajuste.
- **Predeterminado**: None (deshabilitado)

### `--urepa_similarity_ema_decay`

- **Qué**: Factor de decaimiento del promedio móvil exponencial de la similitud.
- **Por qué**: Valores altos suavizan (0.99 ≈ ventana de 100 pasos); valores bajos reaccionan más rápido.
- **Predeterminado**: `0.99`

### `--urepa_threshold_mode`

- **Qué**: Comportamiento al alcanzar el umbral.
- **Opciones**: `permanent` (se apaga para siempre), `recoverable` (se reactiva si la similitud cae)
- **Predeterminado**: `permanent`

### Ejemplo de configuración

```toml
# Habilita U-REPA para fine-tuning de UNet (SDXL, SD1.5, Kolors)
urepa_enabled = true
urepa_lambda = 0.5
urepa_manifold_weight = 3.0
urepa_model = "dinov2_vitg14"
urepa_encoder_image_size = 518
urepa_use_tae = false

# U-REPA Scheduling (opcional)
# urepa_scheduler = "cosine"           # Tipo de decaimiento: constant, linear, cosine, polynomial
# urepa_warmup_steps = 100             # Warmup antes de U-REPA
# urepa_decay_steps = 1000             # Pasos de decaimiento (0 = entrenamiento completo)
# urepa_lambda_end = 0.0               # Peso final después del decaimiento
# urepa_cutoff_step = 5000             # Corte duro (0 = deshabilitado)
# urepa_similarity_threshold = 0.9     # Corte basado en similitud
# urepa_threshold_mode = "permanent"   # permanent o recoverable
```

---

## 🔄 Checkpointing y reanudación

### `--checkpoint_step_interval` (alias: `--checkpointing_steps`)

- **Qué**: Intervalo en el que se guardan los checkpoints del estado de entrenamiento (en pasos).
- **Por qué**: Útil para reanudar entrenamiento y para inferencia. Cada *n* iteraciones, se guardará un checkpoint parcial en formato `.safetensors`, vía el layout de sistema de archivos de Diffusers.

---

## 🔁 LayerSync (Autoalineación de estados ocultos)

LayerSync anima a una capa "estudiante" a igualar una capa "maestra" más fuerte dentro del mismo transformer, usando similitud coseno sobre tokens ocultos.

### `--layersync_enabled`

- **Qué**: Habilita el alineamiento de estados ocultos LayerSync entre dos bloques transformer dentro del mismo modelo.
- **Notas**: Asigna un buffer de estado oculto; falla al iniciar si faltan flags requeridos.
- **Predeterminado**: `false`

### `--layersync_student_block`

- **Qué**: Índice del bloque transformer a tratar como ancla estudiante.
- **Indexación**: Acepta profundidades estilo paper de LayerSync basadas en 1 o IDs de capa basados en 0; la implementación prueba `idx-1` primero y luego `idx`.
- **Requerido**: Sí cuando LayerSync está habilitado.

### `--layersync_teacher_block`

- **Qué**: Índice del bloque transformer a tratar como objetivo maestro (puede ser más profundo que el estudiante).
- **Indexación**: Misma preferencia 1-based, luego fallback 0-based que el bloque estudiante.
- **Predeterminado**: Usa el bloque estudiante cuando se omite para que la pérdida sea auto-similitud.

### `--layersync_lambda`

- **Qué**: Peso de la pérdida de alineamiento coseno de LayerSync entre los estados ocultos del estudiante y del maestro (similitud coseno negativa).
- **Efecto**: Escala el regularizador auxiliar añadido sobre la pérdida base; valores más altos empujan a los tokens del estudiante a alinearse más fuertemente con los tokens del maestro.
- **Nombre upstream**: `--reg-weight` en el codebase original de LayerSync.
- **Requerido**: Debe ser > 0 cuando LayerSync está habilitado (de lo contrario el entrenamiento aborta).
- **Predeterminado**: `0.2` cuando LayerSync está habilitado (coincide con el repo de referencia), `0.0` en otro caso.

Mapeo de opciones upstream (LayerSync → SimpleTuner):
- `--encoder-depth` → `--layersync_student_block` (acepta profundidad 1-based como en upstream, o índice de capa 0-based)
- `--gt-encoder-depth` → `--layersync_teacher_block` (se prefiere 1-based; por defecto usa el estudiante cuando se omite)
- `--reg-weight` → `--layersync_lambda`

> Notas: LayerSync siempre separa el estado oculto del maestro antes de la similitud, igual que la implementación de referencia. Depende de modelos que expongan estados ocultos del transformer (la mayoría de backbones transformer en SimpleTuner) y añade memoria por paso para el buffer de estados ocultos; desactívalo si la VRAM es limitada.

### `--checkpoint_epoch_interval`

- **Qué**: Ejecuta checkpointing cada N épocas completadas.
- **Por qué**: Complementa los checkpoints por paso asegurando que siempre captures el estado en los límites de época, incluso cuando los conteos de pasos varían con el muestreo multi-dataset.

### `--resume_from_checkpoint`

- **Qué**: Especifica si y desde dónde reanudar el entrenamiento. Acepta `latest`, un nombre/ruta local de checkpoint o un URI S3/R2.
- **Por qué**: Permite continuar entrenando desde un estado guardado, ya sea especificado manualmente o el más reciente disponible.
- **Reanudación remota**: Proporciona un URI completo (`s3://bucket/jobs/.../checkpoint-100`) o una clave relativa al bucket (`jobs/.../checkpoint-100`). `latest` solo funciona con `output_dir` local.
- **Requisitos**: La reanudación remota necesita una entrada S3 en publishing_config (bucket + credenciales) que pueda leer el checkpoint.
- **Notas**: Los checkpoints remotos deben incluir `checkpoint_manifest.json` (generado por ejecuciones recientes de SimpleTuner). Un checkpoint se compone de un subdirectorio `unet` y opcionalmente `unet_ema`. El `unet` puede colocarse en cualquier layout de Diffusers para SDXL, permitiendo usarlo como lo harías con un modelo normal.

> ℹ️ Los modelos transformer como PixArt, SD3 o Hunyuan usan los nombres de subcarpeta `transformer` y `transformer_ema`.

### `--disk_low_threshold`

- **Qué**: Espacio mínimo libre en disco requerido antes de guardar checkpoints.
- **Por qué**: Previene que el entrenamiento falle por errores de disco lleno al detectar espacio bajo tempranamente y tomar una acción configurada.
- **Formato**: Cadena de tamaño como `100G`, `50M`, `1T`, `500K`, o bytes simples.
- **Por defecto**: Ninguno (función desactivada)

### `--disk_low_action`

- **Qué**: Acción a tomar cuando el espacio en disco está por debajo del umbral.
- **Opciones**: `stop`, `wait`, `script`
- **Por defecto**: `stop`
- **Comportamiento**:
  - `stop`: Termina el entrenamiento inmediatamente con un mensaje de error.
  - `wait`: Hace bucle cada 30 segundos hasta que el espacio esté disponible. Puede esperar indefinidamente.
  - `script`: Ejecuta el script especificado por `--disk_low_script` para liberar espacio.

### `--disk_low_script`

- **Qué**: Ruta a un script de limpieza para ejecutar cuando el espacio en disco es bajo.
- **Por qué**: Permite limpieza automatizada (ej: eliminar checkpoints antiguos, limpiar caché) cuando el espacio en disco es bajo.
- **Notas**: Solo se usa cuando `--disk_low_action=script`. El script debe ser ejecutable. Si el script falla o no libera suficiente espacio, el entrenamiento se detendrá con un error.
- **Por defecto**: Ninguno

---

## 📊 Registro y monitoreo

### `--logging_dir`

- **Qué**: Directorio para logs de TensorBoard.
- **Por qué**: Te permite monitorear el progreso de entrenamiento y métricas de rendimiento.

### `--report_to`

- **Qué**: Especifica la plataforma para reportar resultados y logs.
- **Por qué**: Habilita integración con plataformas como TensorBoard, wandb o comet_ml para monitoreo. Usa múltiples valores separados por comas para reportar a múltiples trackers;
- **Opciones**: wandb, tensorboard, comet_ml

## Variables de configuración del entorno

Las opciones anteriores aplican en su mayor parte a `config.json`, pero algunas entradas deben configurarse en `config.env`.

- `TRAINING_NUM_PROCESSES` debe configurarse al número de GPUs del sistema. Para la mayoría de los casos de uso, esto es suficiente para habilitar entrenamiento DistributedDataParallel (DDP). Usa `num_processes` dentro de `config.json` si prefieres no usar `config.env`.
- `TRAINING_DYNAMO_BACKEND` por defecto es `no` pero puede configurarse a cualquier backend de torch.compile soportado (p. ej., `inductor`, `aot_eager`, `cudagraphs`) y combinarse con `--dynamo_mode`, `--dynamo_fullgraph` o `--dynamo_use_regional_compilation` para un ajuste más fino
- `SIMPLETUNER_LOG_LEVEL` por defecto es `INFO` pero puede configurarse a `DEBUG` para añadir más información para reportes de problemas en `debug.log`
- `VENV_PATH` puede configurarse a la ubicación de tu entorno virtual de python si no está en la ubicación típica `.venv`
- `ACCELERATE_EXTRA_ARGS` puede dejarse sin configurar o contener argumentos extra como `--multi_gpu` o flags específicos de FSDP

---

Este es un resumen básico pensado para ayudarte a empezar. Para una lista completa de opciones y explicaciones más detalladas, consulta la especificación completa:

```
usage: train.py [-h] --model_family
                {kolors,auraflow,omnigen,flux,deepfloyd,cosmos2image,sana,qwen_image,pixart_sigma,sdxl,sd1x,sd2x,wan,hidream,sd3,lumina2,ltxvideo}
                [--model_flavour MODEL_FLAVOUR] [--controlnet [CONTROLNET]]
                [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH]
                --output_dir OUTPUT_DIR [--logging_dir LOGGING_DIR]
                --model_type {full,lora} [--seed SEED]
                [--resolution RESOLUTION]
                [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                [--prediction_type {epsilon,v_prediction,sample,flow_matching}]
                [--pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH]
                [--vae_dtype {default,fp32,fp16,bf16}]
                [--vae_cache_ondemand [VAE_CACHE_ONDEMAND]]
                [--accelerator_cache_clear_interval ACCELERATOR_CACHE_CLEAR_INTERVAL]
                [--aspect_bucket_rounding {1,2,3,4,5,6,7,8,9}]
                [--base_model_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_1_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_2_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_3_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_4_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--gradient_checkpointing_interval GRADIENT_CHECKPOINTING_INTERVAL]
                [--offload_during_startup [OFFLOAD_DURING_STARTUP]]
                [--quantize_via {cpu,accelerator,pipeline}]
                [--quantization_config QUANTIZATION_CONFIG]
                [--fuse_qkv_projections [FUSE_QKV_PROJECTIONS]]
                [--control [CONTROL]]
                [--controlnet_custom_config CONTROLNET_CUSTOM_CONFIG]
                [--controlnet_model_name_or_path CONTROLNET_MODEL_NAME_OR_PATH]
                [--tread_config TREAD_CONFIG]
                [--pretrained_transformer_model_name_or_path PRETRAINED_TRANSFORMER_MODEL_NAME_OR_PATH]
                [--pretrained_transformer_subfolder PRETRAINED_TRANSFORMER_SUBFOLDER]
                [--pretrained_unet_model_name_or_path PRETRAINED_UNET_MODEL_NAME_OR_PATH]
                [--pretrained_unet_subfolder PRETRAINED_UNET_SUBFOLDER]
                [--pretrained_t5_model_name_or_path PRETRAINED_T5_MODEL_NAME_OR_PATH]
                [--pretrained_gemma_model_name_or_path PRETRAINED_GEMMA_MODEL_NAME_OR_PATH]
                [--revision REVISION] [--variant VARIANT]
                [--base_model_default_dtype {bf16,fp32}]
                [--unet_attention_slice [UNET_ATTENTION_SLICE]]
                [--num_train_epochs NUM_TRAIN_EPOCHS]
                [--max_train_steps MAX_TRAIN_STEPS]
                [--train_batch_size TRAIN_BATCH_SIZE]
                [--learning_rate LEARNING_RATE] --optimizer
                {adamw_bf16,ao-adamw8bit,ao-adamw4bit,ao-adamfp8,ao-adamwfp8,adamw_schedulefree,adamw_schedulefree+aggressive,adamw_schedulefree+no_kahan,optimi-stableadamw,optimi-adamw,optimi-lion,optimi-radam,optimi-ranger,optimi-adan,optimi-adam,optimi-sgd,soap,prodigy}
                [--optimizer_config OPTIMIZER_CONFIG]
                [--lr_scheduler {linear,sine,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                [--lr_warmup_steps LR_WARMUP_STEPS]
                [--checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT]
                [--gradient_checkpointing [GRADIENT_CHECKPOINTING]]
                [--train_text_encoder [TRAIN_TEXT_ENCODER]]
                [--text_encoder_lr TEXT_ENCODER_LR]
                [--lr_num_cycles LR_NUM_CYCLES] [--lr_power LR_POWER]
                [--use_soft_min_snr [USE_SOFT_MIN_SNR]] [--use_ema [USE_EMA]]
                [--ema_device {accelerator,cpu}]
                [--ema_cpu_only [EMA_CPU_ONLY]]
                [--ema_update_interval EMA_UPDATE_INTERVAL]
                [--ema_foreach_disable [EMA_FOREACH_DISABLE]]
                [--ema_decay EMA_DECAY] [--lora_rank LORA_RANK]
                [--lora_alpha LORA_ALPHA] [--lora_type {standard,lycoris}]
                [--lora_dropout LORA_DROPOUT]
                [--lora_init_type {default,gaussian,loftq,olora,pissa}]
                [--peft_lora_mode {standard,singlora}]
                [--peft_lora_target_modules PEFT_LORA_TARGET_MODULES]
                [--singlora_ramp_up_steps SINGLORA_RAMP_UP_STEPS]
                [--init_lora INIT_LORA] [--lycoris_config LYCORIS_CONFIG]
                [--init_lokr_norm INIT_LOKR_NORM]
                [--flux_lora_target {mmdit,context,context+ffs,all,all+ffs,ai-toolkit,tiny,nano,controlnet,all+ffs+embedder,all+ffs+embedder+controlnet}]
                [--use_dora [USE_DORA]]
                [--resolution_type {pixel,area,pixel_area}]
                --data_backend_config DATA_BACKEND_CONFIG
                [--caption_strategy {filename,textfile,instance_prompt,parquet}]
                [--conditioning_multidataset_sampling {combined,random}]
                [--instance_prompt INSTANCE_PROMPT]
                [--parquet_caption_column PARQUET_CAPTION_COLUMN]
                [--parquet_filename_column PARQUET_FILENAME_COLUMN]
                [--ignore_missing_files [IGNORE_MISSING_FILES]]
                [--vae_cache_scan_behaviour {recreate,sync}]
                [--vae_enable_slicing [VAE_ENABLE_SLICING]]
                [--vae_enable_tiling [VAE_ENABLE_TILING]]
                [--vae_enable_patch_conv [VAE_ENABLE_PATCH_CONV]]
                [--vae_batch_size VAE_BATCH_SIZE]
                [--caption_dropout_probability CAPTION_DROPOUT_PROBABILITY]
                [--tokenizer_max_length TOKENIZER_MAX_LENGTH]
                [--validation_step_interval VALIDATION_STEP_INTERVAL]
                [--validation_epoch_interval VALIDATION_EPOCH_INTERVAL]
                [--disable_benchmark [DISABLE_BENCHMARK]]
                [--validation_prompt VALIDATION_PROMPT]
                [--num_validation_images NUM_VALIDATION_IMAGES]
                [--num_eval_images NUM_EVAL_IMAGES]
                [--eval_steps_interval EVAL_STEPS_INTERVAL]
                [--eval_epoch_interval EVAL_EPOCH_INTERVAL]
                [--eval_timesteps EVAL_TIMESTEPS]
                [--eval_dataset_pooling [EVAL_DATASET_POOLING]]
                [--evaluation_type {none,clip}]
                [--pretrained_evaluation_model_name_or_path PRETRAINED_EVALUATION_MODEL_NAME_OR_PATH]
                [--validation_guidance VALIDATION_GUIDANCE]
                [--validation_num_inference_steps VALIDATION_NUM_INFERENCE_STEPS]
                [--validation_on_startup [VALIDATION_ON_STARTUP]]
                [--validation_using_datasets [VALIDATION_USING_DATASETS]]
                [--validation_torch_compile [VALIDATION_TORCH_COMPILE]]
                [--validation_guidance_real VALIDATION_GUIDANCE_REAL]
                [--validation_no_cfg_until_timestep VALIDATION_NO_CFG_UNTIL_TIMESTEP]
                [--validation_negative_prompt VALIDATION_NEGATIVE_PROMPT]
                [--validation_randomize [VALIDATION_RANDOMIZE]]
                [--validation_seed VALIDATION_SEED]
                [--validation_disable [VALIDATION_DISABLE]]
                [--validation_prompt_library [VALIDATION_PROMPT_LIBRARY]]
                [--user_prompt_library USER_PROMPT_LIBRARY]
                [--eval_dataset_id EVAL_DATASET_ID]
                [--validation_stitch_input_location {left,right}]
                [--validation_guidance_rescale VALIDATION_GUIDANCE_RESCALE]
                [--validation_disable_unconditional [VALIDATION_DISABLE_UNCONDITIONAL]]
                [--validation_guidance_skip_layers VALIDATION_GUIDANCE_SKIP_LAYERS]
                [--validation_guidance_skip_layers_start VALIDATION_GUIDANCE_SKIP_LAYERS_START]
                [--validation_guidance_skip_layers_stop VALIDATION_GUIDANCE_SKIP_LAYERS_STOP]
                [--validation_guidance_skip_scale VALIDATION_GUIDANCE_SKIP_SCALE]
                [--validation_lycoris_strength VALIDATION_LYCORIS_STRENGTH]
                [--validation_noise_scheduler {ddim,ddpm,euler,euler-a,unipc,dpm++,perflow}]
                [--validation_num_video_frames VALIDATION_NUM_VIDEO_FRAMES]
                [--validation_audio_only [VALIDATION_AUDIO_ONLY]]
                [--validation_resolution VALIDATION_RESOLUTION]
                [--validation_seed_source {cpu,gpu}]
                [--i_know_what_i_am_doing [I_KNOW_WHAT_I_AM_DOING]]
                [--flow_sigmoid_scale FLOW_SIGMOID_SCALE]
                [--flux_fast_schedule [FLUX_FAST_SCHEDULE]]
                [--flow_use_uniform_schedule [FLOW_USE_UNIFORM_SCHEDULE]]
                [--flow_use_beta_schedule [FLOW_USE_BETA_SCHEDULE]]
                [--flow_beta_schedule_alpha FLOW_BETA_SCHEDULE_ALPHA]
                [--flow_beta_schedule_beta FLOW_BETA_SCHEDULE_BETA]
                [--flow_schedule_shift FLOW_SCHEDULE_SHIFT]
                [--flow_schedule_auto_shift [FLOW_SCHEDULE_AUTO_SHIFT]]
                [--flux_guidance_mode {constant,random-range}]
                [--flux_attention_masked_training [FLUX_ATTENTION_MASKED_TRAINING]]
                [--flux_guidance_value FLUX_GUIDANCE_VALUE]
                [--flux_guidance_min FLUX_GUIDANCE_MIN]
                [--flux_guidance_max FLUX_GUIDANCE_MAX]
                [--t5_padding {zero,unmodified}]
                [--sd3_clip_uncond_behaviour {empty_string,zero}]
                [--sd3_t5_uncond_behaviour {empty_string,zero}]
                [--soft_min_snr_sigma_data SOFT_MIN_SNR_SIGMA_DATA]
                [--mixed_precision {no,fp16,bf16,fp8}]
                [--attention_mechanism {diffusers,xformers,flash-attn,flash-attn-2,flash-attn-3,flash-attn-3-varlen,flex,cudnn,native-efficient,native-flash,native-math,native-npu,native-xla,sla,sageattention,sageattention-int8-fp16-triton,sageattention-int8-fp16-cuda,sageattention-int8-fp8-cuda}]
                [--sageattention_usage {training,inference,training+inference}]
                [--disable_tf32 [DISABLE_TF32]]
                [--set_grads_to_none [SET_GRADS_TO_NONE]]
                [--noise_offset NOISE_OFFSET]
                [--noise_offset_probability NOISE_OFFSET_PROBABILITY]
                [--input_perturbation INPUT_PERTURBATION]
                [--input_perturbation_steps INPUT_PERTURBATION_STEPS]
                [--lr_end LR_END] [--lr_scale [LR_SCALE]]
                [--lr_scale_sqrt [LR_SCALE_SQRT]]
                [--ignore_final_epochs [IGNORE_FINAL_EPOCHS]]
                [--freeze_encoder_before FREEZE_ENCODER_BEFORE]
                [--freeze_encoder_after FREEZE_ENCODER_AFTER]
                [--freeze_encoder_strategy {before,between,after}]
                [--layer_freeze_strategy {none,bitfit}]
                [--fully_unload_text_encoder [FULLY_UNLOAD_TEXT_ENCODER]]
                [--save_text_encoder [SAVE_TEXT_ENCODER]]
                [--text_encoder_limit TEXT_ENCODER_LIMIT]
                [--prepend_instance_prompt [PREPEND_INSTANCE_PROMPT]]
                [--only_instance_prompt [ONLY_INSTANCE_PROMPT]]
                [--data_aesthetic_score DATA_AESTHETIC_SCORE]
                [--delete_unwanted_images [DELETE_UNWANTED_IMAGES]]
                [--delete_problematic_images [DELETE_PROBLEMATIC_IMAGES]]
                [--disable_bucket_pruning [DISABLE_BUCKET_PRUNING]]
                [--disable_segmented_timestep_sampling [DISABLE_SEGMENTED_TIMESTEP_SAMPLING]]
                [--preserve_data_backend_cache [PRESERVE_DATA_BACKEND_CACHE]]
                [--override_dataset_config [OVERRIDE_DATASET_CONFIG]]
                [--cache_dir CACHE_DIR] [--cache_dir_text CACHE_DIR_TEXT]
                [--cache_dir_vae CACHE_DIR_VAE]
                [--compress_disk_cache [COMPRESS_DISK_CACHE]]
                [--aspect_bucket_disable_rebuild [ASPECT_BUCKET_DISABLE_REBUILD]]
                [--keep_vae_loaded [KEEP_VAE_LOADED]]
                [--skip_file_discovery SKIP_FILE_DISCOVERY]
                [--data_backend_sampling {uniform,auto-weighting}]
                [--image_processing_batch_size IMAGE_PROCESSING_BATCH_SIZE]
                [--write_batch_size WRITE_BATCH_SIZE]
                [--read_batch_size READ_BATCH_SIZE]
                [--enable_multiprocessing [ENABLE_MULTIPROCESSING]]
                [--max_workers MAX_WORKERS]
                [--aws_max_pool_connections AWS_MAX_POOL_CONNECTIONS]
                [--torch_num_threads TORCH_NUM_THREADS]
                [--dataloader_prefetch [DATALOADER_PREFETCH]]
                [--dataloader_prefetch_qlen DATALOADER_PREFETCH_QLEN]
                [--aspect_bucket_worker_count ASPECT_BUCKET_WORKER_COUNT]
                [--aspect_bucket_alignment {8,16,24,32,64}]
                [--minimum_image_size MINIMUM_IMAGE_SIZE]
                [--maximum_image_size MAXIMUM_IMAGE_SIZE]
                [--target_downsample_size TARGET_DOWNSAMPLE_SIZE]
                [--max_upscale_threshold MAX_UPSCALE_THRESHOLD]
                [--metadata_update_interval METADATA_UPDATE_INTERVAL]
                [--debug_aspect_buckets [DEBUG_ASPECT_BUCKETS]]
                [--debug_dataset_loader [DEBUG_DATASET_LOADER]]
                [--print_filenames [PRINT_FILENAMES]]
                [--print_sampler_statistics [PRINT_SAMPLER_STATISTICS]]
                [--timestep_bias_strategy {earlier,later,range,none}]
                [--timestep_bias_begin TIMESTEP_BIAS_BEGIN]
                [--timestep_bias_end TIMESTEP_BIAS_END]
                [--timestep_bias_multiplier TIMESTEP_BIAS_MULTIPLIER]
                [--timestep_bias_portion TIMESTEP_BIAS_PORTION]
                [--training_scheduler_timestep_spacing {leading,linspace,trailing}]
                [--inference_scheduler_timestep_spacing {leading,linspace,trailing}]
                [--loss_type {l2,huber,smooth_l1}]
                [--huber_schedule {snr,exponential,constant}]
                [--huber_c HUBER_C] [--snr_gamma SNR_GAMMA]
                [--masked_loss_probability MASKED_LOSS_PROBABILITY]
                [--hidream_use_load_balancing_loss [HIDREAM_USE_LOAD_BALANCING_LOSS]]
                [--hidream_load_balancing_loss_weight HIDREAM_LOAD_BALANCING_LOSS_WEIGHT]
                [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2]
                [--optimizer_beta1 OPTIMIZER_BETA1]
                [--optimizer_beta2 OPTIMIZER_BETA2]
                [--optimizer_cpu_offload_method {none}]
                [--gradient_precision {unmodified,fp32}]
                [--adam_weight_decay ADAM_WEIGHT_DECAY]
                [--adam_epsilon ADAM_EPSILON] [--prodigy_steps PRODIGY_STEPS]
                [--max_grad_norm MAX_GRAD_NORM]
                [--grad_clip_method {value,norm}]
                [--optimizer_offload_gradients [OPTIMIZER_OFFLOAD_GRADIENTS]]
                [--fuse_optimizer [FUSE_OPTIMIZER]]
                [--optimizer_release_gradients [OPTIMIZER_RELEASE_GRADIENTS]]
                [--push_to_hub [PUSH_TO_HUB]]
                [--push_to_hub_background [PUSH_TO_HUB_BACKGROUND]]
                [--push_checkpoints_to_hub [PUSH_CHECKPOINTS_TO_HUB]]
                [--publishing_config PUBLISHING_CONFIG]
                [--hub_model_id HUB_MODEL_ID]
                [--model_card_private [MODEL_CARD_PRIVATE]]
                [--model_card_safe_for_work [MODEL_CARD_SAFE_FOR_WORK]]
                [--model_card_note MODEL_CARD_NOTE]
                [--modelspec_comment MODELSPEC_COMMENT]
                [--report_to {tensorboard,wandb,comet_ml,all,none}]
                [--checkpoint_step_interval CHECKPOINT_STEP_INTERVAL]
                [--checkpoint_epoch_interval CHECKPOINT_EPOCH_INTERVAL]
                [--checkpointing_rolling_steps CHECKPOINTING_ROLLING_STEPS]
                [--checkpointing_use_tempdir [CHECKPOINTING_USE_TEMPDIR]]
                [--checkpoints_rolling_total_limit CHECKPOINTS_ROLLING_TOTAL_LIMIT]
                [--tracker_run_name TRACKER_RUN_NAME]
                [--tracker_project_name TRACKER_PROJECT_NAME]
                [--tracker_image_layout {gallery,table}]
                [--enable_watermark [ENABLE_WATERMARK]]
                [--framerate FRAMERATE]
                [--seed_for_each_device [SEED_FOR_EACH_DEVICE]]
                [--snr_weight SNR_WEIGHT]
                [--rescale_betas_zero_snr [RESCALE_BETAS_ZERO_SNR]]
                [--webhook_config WEBHOOK_CONFIG]
                [--webhook_reporting_interval WEBHOOK_REPORTING_INTERVAL]
                [--distillation_method {lcm,dcm,dmd,perflow}]
                [--distillation_config DISTILLATION_CONFIG]
                [--ema_validation {none,ema_only,comparison}]
                [--local_rank LOCAL_RANK] [--ltx_train_mode {t2v,i2v}]
                [--ltx_i2v_prob LTX_I2V_PROB]
                [--ltx_partial_noise_fraction LTX_PARTIAL_NOISE_FRACTION]
                [--ltx_protect_first_frame [LTX_PROTECT_FIRST_FRAME]]
                [--offload_param_path OFFLOAD_PARAM_PATH]
                [--offset_noise [OFFSET_NOISE]]
                [--quantize_activations [QUANTIZE_ACTIVATIONS]]
                [--refiner_training [REFINER_TRAINING]]
                [--refiner_training_invert_schedule [REFINER_TRAINING_INVERT_SCHEDULE]]
                [--refiner_training_strength REFINER_TRAINING_STRENGTH]
                [--sdxl_refiner_uses_full_range [SDXL_REFINER_USES_FULL_RANGE]]
                [--sana_complex_human_instruction SANA_COMPLEX_HUMAN_INSTRUCTION]

The following SimpleTuner command-line options are available:

options:
  -h, --help            show this help message and exit
  --model_family {kolors,auraflow,omnigen,flux,deepfloyd,cosmos2image,sana,qwen_image,pixart_sigma,sdxl,sd1x,sd2x,wan,hidream,sd3,lumina2,ltxvideo,ace_step,heartmula}
                        The base model architecture family to train
  --model_flavour MODEL_FLAVOUR
                        Specific variant of the selected model family.
                        Los flavours de ACE-Step son `base`, `v15-turbo`,
                        `v15-base` y `v15-sft`. Los flavours v1.5 soportan
                        entrenamiento y validación de audio integrada, y
                        requieren `--trust_remote_code` para el repositorio
                        upstream.
  --controlnet [CONTROLNET]
                        Train ControlNet (full or LoRA) branches alongside the
                        primary network.
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                        Optional override of the model checkpoint. Leave blank
                        to use the default path for the selected model
                        flavour.
  --output_dir OUTPUT_DIR
                        Directory where model checkpoints and logs will be
                        saved
  --logging_dir LOGGING_DIR
                        Directory for TensorBoard logs
  --model_type {full,lora}
                        Choose between full model training or LoRA adapter
                        training
  --seed SEED           Seed used for deterministic training behaviour
  --resolution RESOLUTION
                        Resolution for training images
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        Select checkpoint to resume training from
  --prediction_type {epsilon,v_prediction,sample,flow_matching}
                        The parameterization type for the diffusion model
  --pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH
                        Path to pretrained VAE model
  --vae_dtype {default,fp32,fp16,bf16}
                        Precision for VAE encoding/decoding. Lower precision
                        saves memory.
  --vae_cache_ondemand [VAE_CACHE_ONDEMAND]
                        Process VAE latents during training instead of
                        precomputing them
  --vae_cache_disable [VAE_CACHE_DISABLE]
                        Implicitly enables on-demand caching and disables
                        writing embeddings to disk.
  --accelerator_cache_clear_interval ACCELERATOR_CACHE_CLEAR_INTERVAL
                        Clear the cache from VRAM every X steps to prevent
                        memory leaks
  --aspect_bucket_rounding {1,2,3,4,5,6,7,8,9}
                        Number of decimal places to round aspect ratios to for
                        bucket creation
  --base_model_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for loading the base model. Lower precision
                        saves memory.
  --text_encoder_1_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_2_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_3_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_4_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --gradient_checkpointing_interval GRADIENT_CHECKPOINTING_INTERVAL
                        Checkpoint every N transformer blocks
  --offload_during_startup [OFFLOAD_DURING_STARTUP]
                        Offload text encoders to CPU during VAE caching
  --quantize_via {cpu,accelerator,pipeline}
                        Where to perform model quantization
  --quantization_config QUANTIZATION_CONFIG
                        JSON or file path describing Diffusers quantization
                        config for pipeline quantization
  --fuse_qkv_projections [FUSE_QKV_PROJECTIONS]
                        Enables Flash Attention 3 when supported; otherwise
                        falls back to PyTorch SDPA.
  --control [CONTROL]   Enable channel-wise control style training
  --controlnet_custom_config CONTROLNET_CUSTOM_CONFIG
                        Custom configuration for ControlNet models
  --controlnet_model_name_or_path CONTROLNET_MODEL_NAME_OR_PATH
                        Path to ControlNet model weights to preload
  --tread_config TREAD_CONFIG
                        Configuration for TREAD training method
  --pretrained_transformer_model_name_or_path PRETRAINED_TRANSFORMER_MODEL_NAME_OR_PATH
                        Path to pretrained transformer model
  --pretrained_transformer_subfolder PRETRAINED_TRANSFORMER_SUBFOLDER
                        Subfolder containing transformer model weights
  --pretrained_unet_model_name_or_path PRETRAINED_UNET_MODEL_NAME_OR_PATH
                        Path to pretrained UNet model
  --pretrained_unet_subfolder PRETRAINED_UNET_SUBFOLDER
                        Subfolder containing UNet model weights
  --pretrained_t5_model_name_or_path PRETRAINED_T5_MODEL_NAME_OR_PATH
                        Path to pretrained T5 model
  --pretrained_gemma_model_name_or_path PRETRAINED_GEMMA_MODEL_NAME_OR_PATH
                        Path to pretrained Gemma model
  --revision REVISION   Git branch/tag/commit for model version
  --variant VARIANT     Model variant (e.g., fp16, bf16)
  --base_model_default_dtype {bf16,fp32}
                        Default precision for quantized base model weights
  --unet_attention_slice [UNET_ATTENTION_SLICE]
                        Enable attention slicing for SDXL UNet
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Number of times to iterate through the entire dataset
  --max_train_steps MAX_TRAIN_STEPS
                        Maximum number of training steps (0 = use epochs
                        instead)
  --train_batch_size TRAIN_BATCH_SIZE
                        Number of samples processed per forward/backward pass
                        (per device).
  --learning_rate LEARNING_RATE
                        Base learning rate for training
  --optimizer {adamw_bf16,ao-adamw8bit,ao-adamw4bit,ao-adamfp8,ao-adamwfp8,adamw_schedulefree,adamw_schedulefree+aggressive,adamw_schedulefree+no_kahan,optimi-stableadamw,optimi-adamw,optimi-lion,optimi-radam,optimi-ranger,optimi-adan,optimi-adam,optimi-sgd,soap,prodigy}
                        Optimization algorithm for training
  --optimizer_config OPTIMIZER_CONFIG
                        Comma-separated key=value pairs forwarded to the
                        selected optimizer
  --lr_scheduler {linear,sine,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
                        How learning rate changes during training
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of steps to accumulate gradients
  --lr_warmup_steps LR_WARMUP_STEPS
                        Number of steps to gradually increase LR from 0
  --checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT
                        Maximum number of checkpoints to keep on disk
  --gradient_checkpointing [GRADIENT_CHECKPOINTING]
                        Trade compute for memory during training
  --train_text_encoder [TRAIN_TEXT_ENCODER]
                        Also train the text encoder (CLIP) model
  --text_encoder_lr TEXT_ENCODER_LR
                        Separate learning rate for text encoder
  --lr_num_cycles LR_NUM_CYCLES
                        Number of cosine annealing cycles
  --lr_power LR_POWER   Power for polynomial decay scheduler
  --use_soft_min_snr [USE_SOFT_MIN_SNR]
                        Use soft clamping instead of hard clamping for Min-SNR
  --use_ema [USE_EMA]   Maintain an exponential moving average copy of the
                        model during training.
  --ema_device {accelerator,cpu}
                        Where to keep the EMA weights in-between updates.
  --ema_cpu_only [EMA_CPU_ONLY]
                        Keep EMA weights exclusively on CPU even when
                        ema_device would normally move them.
  --ema_update_interval EMA_UPDATE_INTERVAL
                        Update EMA weights every N optimizer steps
  --ema_foreach_disable [EMA_FOREACH_DISABLE]
                        Fallback to standard tensor ops instead of
                        torch.foreach updates.
  --ema_decay EMA_DECAY
                        Smoothing factor for EMA updates (closer to 1.0 =
                        slower drift).
  --lora_rank LORA_RANK
                        Dimension of LoRA update matrices
  --lora_alpha LORA_ALPHA
                        Scaling factor for LoRA updates
  --lora_type {standard,lycoris}
                        LoRA implementation type
  --lora_dropout LORA_DROPOUT
                        LoRA dropout randomly ignores neurons during training.
                        This can help prevent overfitting.
  --lora_init_type {default,gaussian,loftq,olora,pissa}
                        The initialization type for the LoRA model
  --peft_lora_mode {standard,singlora}
                        PEFT LoRA training mode
  --peft_lora_target_modules PEFT_LORA_TARGET_MODULES
                        JSON array (or path to a JSON file) listing PEFT
                        LoRA target module names. Overrides preset targets.
  --singlora_ramp_up_steps SINGLORA_RAMP_UP_STEPS
                        Number of ramp-up steps for SingLoRA
  --slider_lora_target [SLIDER_LORA_TARGET]
                        Route LoRA training to slider-friendly targets
                        (self-attn + conv/time embeddings). Only affects
                        standard PEFT LoRA.
  --init_lora INIT_LORA
                        Specify an existing LoRA or LyCORIS safetensors file
                        to initialize the adapter
  --lycoris_config LYCORIS_CONFIG
                        Path to LyCORIS configuration JSON file
  --init_lokr_norm INIT_LOKR_NORM
                        Perturbed normal initialization for LyCORIS LoKr
                        layers
  --flux_lora_target {mmdit,context,context+ffs,all,all+ffs,ai-toolkit,tiny,nano,controlnet,all+ffs+embedder,all+ffs+embedder+controlnet}
                        Which layers to train in Flux models
  --use_dora [USE_DORA]
                        Enable DoRA (Weight-Decomposed LoRA)
  --resolution_type {pixel,area,pixel_area}
                        How to interpret the resolution value
  --data_backend_config DATA_BACKEND_CONFIG
                        Select a saved dataset configuration (managed in
                        Datasets & Environments tabs)
  --caption_strategy {filename,textfile,instance_prompt,parquet}
                        How to load captions for images
  --conditioning_multidataset_sampling {combined,random}
                        How to sample from multiple conditioning datasets
  --instance_prompt INSTANCE_PROMPT
                        Instance prompt for training
  --parquet_caption_column PARQUET_CAPTION_COLUMN
                        Column name containing captions in parquet files
  --parquet_filename_column PARQUET_FILENAME_COLUMN
                        Column name containing image paths in parquet files
  --ignore_missing_files [IGNORE_MISSING_FILES]
                        Continue training even if some files are missing
  --vae_cache_scan_behaviour {recreate,sync}
                        How to scan VAE cache for missing files
  --vae_enable_slicing [VAE_ENABLE_SLICING]
                        Enable VAE attention slicing for memory efficiency
  --vae_enable_tiling [VAE_ENABLE_TILING]
                        Enable VAE tiling for large images
  --vae_enable_patch_conv [VAE_ENABLE_PATCH_CONV]
                        Enable patch-based 3D conv for HunyuanVideo VAE to
                        reduce peak VRAM (slight slowdown)
  --vae_batch_size VAE_BATCH_SIZE
                        Batch size for VAE encoding during caching
  --caption_dropout_probability CAPTION_DROPOUT_PROBABILITY
                        Caption dropout will randomly drop captions and, for
                        SDXL, size conditioning inputs based on this
                        probability
  --tokenizer_max_length TOKENIZER_MAX_LENGTH
                        Override the tokenizer sequence length (advanced).
  --validation_step_interval VALIDATION_STEP_INTERVAL
                        Run validation every N training steps (deprecated alias: --validation_steps)
  --validation_epoch_interval VALIDATION_EPOCH_INTERVAL
                        Run validation every N training epochs
  --disable_benchmark [DISABLE_BENCHMARK]
                        Skip generating baseline comparison images before
                        training starts
  --validation_prompt VALIDATION_PROMPT
                        Prompt to use for validation images
  --num_validation_images NUM_VALIDATION_IMAGES
                        Number of images to generate per validation
  --num_eval_images NUM_EVAL_IMAGES
                        Number of images to generate for evaluation metrics
  --eval_steps_interval EVAL_STEPS_INTERVAL
                        Run evaluation every N training steps
  --eval_epoch_interval EVAL_EPOCH_INTERVAL
                        Run evaluation every N training epochs (decimals run
                        multiple times per epoch)
  --eval_timesteps EVAL_TIMESTEPS
                        Number of timesteps for evaluation
  --eval_dataset_pooling [EVAL_DATASET_POOLING]
                        Combine evaluation metrics from all datasets into a
                        single chart
  --evaluation_type {none,clip}
                        Type of evaluation metrics to compute
  --pretrained_evaluation_model_name_or_path PRETRAINED_EVALUATION_MODEL_NAME_OR_PATH
                        Path to pretrained model for evaluation metrics
  --validation_guidance VALIDATION_GUIDANCE
                        CFG guidance scale for validation images
  --validation_num_inference_steps VALIDATION_NUM_INFERENCE_STEPS
                        Number of diffusion steps for validation renders
  --validation_on_startup [VALIDATION_ON_STARTUP]
                        Run validation on the base model before training
                        starts
  --validation_using_datasets [VALIDATION_USING_DATASETS]
                        Use random images from training datasets for
                        validation
  --validation_torch_compile [VALIDATION_TORCH_COMPILE]
                        Use torch.compile() on validation pipeline for speed
  --validation_guidance_real VALIDATION_GUIDANCE_REAL
                        CFG value for distilled models (e.g., FLUX schnell)
  --validation_no_cfg_until_timestep VALIDATION_NO_CFG_UNTIL_TIMESTEP
                        Skip CFG for initial timesteps (Flux only)
  --validation_negative_prompt VALIDATION_NEGATIVE_PROMPT
                        Negative prompt for validation images
  --validation_randomize [VALIDATION_RANDOMIZE]
                        Use random seeds for each validation
  --validation_seed VALIDATION_SEED
                        Fixed seed for reproducible validation images
  --validation_disable [VALIDATION_DISABLE]
                        Completely disable validation image generation
  --validation_prompt_library [VALIDATION_PROMPT_LIBRARY]
                        Use SimpleTuner's built-in prompt library
  --user_prompt_library USER_PROMPT_LIBRARY
                        Path to custom JSON prompt library
  --eval_dataset_id EVAL_DATASET_ID
                        Specific dataset to use for evaluation metrics
  --validation_stitch_input_location {left,right}
                        Where to place input image in img2img validations
  --validation_guidance_rescale VALIDATION_GUIDANCE_RESCALE
                        CFG rescale value for validation
  --validation_disable_unconditional [VALIDATION_DISABLE_UNCONDITIONAL]
                        Disable unconditional image generation during
                        validation
  --validation_guidance_skip_layers VALIDATION_GUIDANCE_SKIP_LAYERS
                        JSON list of transformer layers to skip during
                        classifier-free guidance
  --validation_guidance_skip_layers_start VALIDATION_GUIDANCE_SKIP_LAYERS_START
                        Starting layer index to skip guidance
  --validation_guidance_skip_layers_stop VALIDATION_GUIDANCE_SKIP_LAYERS_STOP
                        Ending layer index to skip guidance
  --validation_guidance_skip_scale VALIDATION_GUIDANCE_SKIP_SCALE
                        Scale guidance strength when applying layer skipping
  --validation_lycoris_strength VALIDATION_LYCORIS_STRENGTH
                        Strength multiplier for LyCORIS validation
  --validation_noise_scheduler {ddim,ddpm,euler,euler-a,unipc,dpm++,perflow}
                        Noise scheduler for validation
  --validation_num_video_frames VALIDATION_NUM_VIDEO_FRAMES
                        Number of frames for video validation
  --validation_audio_only [VALIDATION_AUDIO_ONLY]
                        Disable video generation during validation and emit
                        audio only
  --validation_resolution VALIDATION_RESOLUTION
                        Override resolution for validation images (pixels or
                        megapixels)
  --validation_seed_source {cpu,gpu}
                        Source device used to generate validation seeds
  --i_know_what_i_am_doing [I_KNOW_WHAT_I_AM_DOING]
                        Unlock experimental overrides and bypass built-in
                        safety limits.
  --flow_sigmoid_scale FLOW_SIGMOID_SCALE
                        Scale factor for sigmoid timestep sampling for flow-
                        matching models.
  --flux_fast_schedule [FLUX_FAST_SCHEDULE]
                        Use experimental fast schedule for Flux training
  --flow_use_uniform_schedule [FLOW_USE_UNIFORM_SCHEDULE]
                        Use uniform schedule instead of sigmoid for flow-
                        matching
  --flow_use_beta_schedule [FLOW_USE_BETA_SCHEDULE]
                        Use beta schedule instead of sigmoid for flow-matching
  --flow_beta_schedule_alpha FLOW_BETA_SCHEDULE_ALPHA
                        Alpha value for beta schedule (default: 2.0)
  --flow_beta_schedule_beta FLOW_BETA_SCHEDULE_BETA
                        Beta value for beta schedule (default: 2.0)
  --flow_schedule_shift FLOW_SCHEDULE_SHIFT
                        Shift the noise schedule for flow-matching models
  --flow_schedule_auto_shift [FLOW_SCHEDULE_AUTO_SHIFT]
                        Auto-adjust schedule shift based on image resolution
  --flux_guidance_mode {constant,random-range}
                        Guidance mode for Flux training
  --flux_attention_masked_training [FLUX_ATTENTION_MASKED_TRAINING]
                        Enable attention masked training for Flux models
  --flux_guidance_value FLUX_GUIDANCE_VALUE
                        Guidance value for constant mode
  --flux_guidance_min FLUX_GUIDANCE_MIN
                        Minimum guidance value for random-range mode
  --flux_guidance_max FLUX_GUIDANCE_MAX
                        Maximum guidance value for random-range mode
  --t5_padding {zero,unmodified}
                        Padding behavior for T5 text encoder
  --sd3_clip_uncond_behaviour {empty_string,zero}
                        How SD3 handles unconditional prompts
  --sd3_t5_uncond_behaviour {empty_string,zero}
                        How SD3 T5 handles unconditional prompts
  --soft_min_snr_sigma_data SOFT_MIN_SNR_SIGMA_DATA
                        Sigma data for soft min SNR weighting
  --mixed_precision {no,fp16,bf16,fp8}
                        Precision for training computations
  --attention_mechanism {diffusers,xformers,flash-attn,flash-attn-2,flash-attn-3,flash-attn-3-varlen,flex,cudnn,native-efficient,native-flash,native-math,native-npu,native-xla,sla,sageattention,sageattention-int8-fp16-triton,sageattention-int8-fp16-cuda,sageattention-int8-fp8-cuda}
                        Attention computation backend
  --sageattention_usage {training,inference,training+inference}
                        When to use SageAttention
  --disable_tf32 [DISABLE_TF32]
                        Force IEEE FP32 precision (disables TF32) using
                        PyTorch's fp32_precision controls when available
  --set_grads_to_none [SET_GRADS_TO_NONE]
                        Set gradients to None instead of zero
  --noise_offset NOISE_OFFSET
                        Add noise offset to training
  --noise_offset_probability NOISE_OFFSET_PROBABILITY
                        Probability of applying noise offset
  --input_perturbation INPUT_PERTURBATION
                        Add additional noise only to the inputs fed to the
                        model during training
  --input_perturbation_steps INPUT_PERTURBATION_STEPS
                        Only apply input perturbation over the first N steps
                        with linear decay
  --lr_end LR_END       A polynomial learning rate will end up at this value
                        after the specified number of warmup steps
  --lr_scale [LR_SCALE]
                        Scale the learning rate by the number of GPUs,
                        gradient accumulation steps, and batch size
  --lr_scale_sqrt [LR_SCALE_SQRT]
                        If using --lr_scale, use the square root of (number of
                        GPUs * gradient accumulation steps * batch size)
  --ignore_final_epochs [IGNORE_FINAL_EPOCHS]
                        When provided, the max epoch counter will not
                        determine the end of the training run
  --freeze_encoder_before FREEZE_ENCODER_BEFORE
                        When using 'before' strategy, we will freeze layers
                        earlier than this
  --freeze_encoder_after FREEZE_ENCODER_AFTER
                        When using 'after' strategy, we will freeze layers
                        later than this
  --freeze_encoder_strategy {before,between,after}
                        When freezing the text encoder, we can use the
                        'before', 'between', or 'after' strategy
  --layer_freeze_strategy {none,bitfit}
                        When freezing parameters, we can use the 'none' or
                        'bitfit' strategy
  --fully_unload_text_encoder [FULLY_UNLOAD_TEXT_ENCODER]
                        If set, will fully unload the text_encoder from memory
                        when not in use
  --save_text_encoder [SAVE_TEXT_ENCODER]
                        If set, will save the text encoder after training
  --text_encoder_limit TEXT_ENCODER_LIMIT
                        When training the text encoder, we want to limit how
                        long it trains for to avoid catastrophic loss
  --prepend_instance_prompt [PREPEND_INSTANCE_PROMPT]
                        When determining the captions from the filename,
                        prepend the instance prompt as an enforced keyword
  --only_instance_prompt [ONLY_INSTANCE_PROMPT]
                        Use the instance prompt instead of the caption from
                        filename
  --data_aesthetic_score DATA_AESTHETIC_SCORE
                        Since currently we do not calculate aesthetic scores
                        for data, we will statically set it to one value. This
                        is only used by the SDXL Refiner
  --delete_unwanted_images [DELETE_UNWANTED_IMAGES]
                        If set, will delete images that are not of a minimum
                        size to save on disk space for large training runs
  --delete_problematic_images [DELETE_PROBLEMATIC_IMAGES]
                        If set, any images that error out during load will be
                        removed from the underlying storage medium
  --disable_bucket_pruning [DISABLE_BUCKET_PRUNING]
                        When training on very small datasets, you might not
                        care that the batch sizes will outpace your image
                        count. Setting this option will prevent SimpleTuner
                        from deleting your bucket lists that do not meet the
                        minimum image count requirements. Use at your own
                        risk, it may end up throwing off your statistics or
                        epoch tracking
  --disable_segmented_timestep_sampling [DISABLE_SEGMENTED_TIMESTEP_SAMPLING]
                        By default, the timestep schedule is divided into
                        roughly `train_batch_size` number of segments, and
                        then each of those are sampled from separately. This
                        improves the selection distribution, but may not be
                        desired in certain training scenarios, eg. when
                        limiting the timestep selection range
  --preserve_data_backend_cache [PRESERVE_DATA_BACKEND_CACHE]
                        For very large cloud storage buckets that will never
                        change, enabling this option will prevent the trainer
                        from scanning it at startup, by preserving the cache
                        files that we generate. Be careful when using this,
                        as, switching datasets can result in the preserved
                        cache being used, which would be problematic.
                        Currently, cache is not stored in the dataset itself
                        but rather, locally. This may change in a future
                        release
  --override_dataset_config [OVERRIDE_DATASET_CONFIG]
                        When provided, the dataset's config will not be
                        checked against the live backend config
  --cache_dir CACHE_DIR
                        The directory where the downloaded models and datasets
                        will be stored
  --cache_dir_text CACHE_DIR_TEXT
                        This is the path to a local directory that will
                        contain your text embed cache
  --cache_dir_vae CACHE_DIR_VAE
                        This is the path to a local directory that will
                        contain your VAE outputs
  --compress_disk_cache [COMPRESS_DISK_CACHE]
                        If set, will gzip-compress the disk cache for Pytorch
                        files. This will save substantial disk space, but may
                        slow down the training process
  --aspect_bucket_disable_rebuild [ASPECT_BUCKET_DISABLE_REBUILD]
                        When using a randomised aspect bucket list, the VAE
                        and aspect cache are rebuilt on each epoch. With a
                        large and diverse enough dataset, rebuilding the
                        aspect list may take a long time, and this may be
                        undesirable. This option will not override
                        vae_cache_clear_each_epoch. If both options are
                        provided, only the VAE cache will be rebuilt
  --keep_vae_loaded [KEEP_VAE_LOADED]
                        If set, will keep the VAE loaded in memory. This can
                        reduce disk churn, but consumes VRAM during the
                        forward pass
  --skip_file_discovery SKIP_FILE_DISCOVERY
                        Comma-separated values of which stages to skip
                        discovery for. Skipping any stage will speed up
                        resumption, but will increase the risk of errors, as
                        missing images or incorrectly bucketed images may not
                        be caught. Valid options: aspect, vae, text, metadata
  --data_backend_sampling {uniform,auto-weighting}
                        When using multiple data backends, the sampling
                        weighting can be set to 'uniform' or 'auto-weighting'
  --image_processing_batch_size IMAGE_PROCESSING_BATCH_SIZE
                        When resizing and cropping images, we do it in
                        parallel using processes or threads. This defines how
                        many images will be read into the queue before they
                        are processed
  --write_batch_size WRITE_BATCH_SIZE
                        When using certain storage backends, it is better to
                        batch smaller writes rather than continuous
                        dispatching. In SimpleTuner, write batching is
                        currently applied during VAE caching, when many small
                        objects are written. This mostly applies to S3, but
                        some shared server filesystems may benefit as well.
                        Default: 64
  --read_batch_size READ_BATCH_SIZE
                        Used by the VAE cache to prefetch image data. This is
                        the number of images to read ahead
  --enable_multiprocessing [ENABLE_MULTIPROCESSING]
                        If set, will use processes instead of threads during
                        metadata caching operations
  --max_workers MAX_WORKERS
                        How many active threads or processes to run during VAE
                        caching
  --aws_max_pool_connections AWS_MAX_POOL_CONNECTIONS
                        When using AWS backends, the maximum number of
                        connections to keep open to the S3 bucket at a single
                        time
  --torch_num_threads TORCH_NUM_THREADS
                        The number of threads to use for PyTorch operations.
                        This is not the same as the number of workers
  --dataloader_prefetch [DATALOADER_PREFETCH]
                        When provided, the dataloader will read-ahead and
                        attempt to retrieve latents, text embeds, and other
                        metadata ahead of the time when the batch is required,
                        so that it can be immediately available
  --dataloader_prefetch_qlen DATALOADER_PREFETCH_QLEN
                        Set the number of prefetched batches
  --aspect_bucket_worker_count ASPECT_BUCKET_WORKER_COUNT
                        The number of workers to use for aspect bucketing.
                        This is a CPU-bound task, so the number of workers
                        should be set to the number of CPU threads available.
                        If you use an I/O bound backend, an even higher value
                        may make sense. Default: 12
  --aspect_bucket_alignment {8,16,24,32,64}
                        When training diffusion models, the image sizes
                        generally must align to a 64 pixel interval
  --minimum_image_size MINIMUM_IMAGE_SIZE
                        The minimum resolution for both sides of input images
  --maximum_image_size MAXIMUM_IMAGE_SIZE
                        When cropping images that are excessively large, the
                        entire scene context may be lost, eg. the crop might
                        just end up being a portion of the background. To
                        avoid this, a maximum image size may be provided,
                        which will result in very-large images being
                        downsampled before cropping them. This value uses
                        --resolution_type to determine whether it is a pixel
                        edge or megapixel value
  --target_downsample_size TARGET_DOWNSAMPLE_SIZE
                        When using --maximum_image_size, very-large images
                        exceeding that value will be downsampled to this
                        target size before cropping
  --max_upscale_threshold MAX_UPSCALE_THRESHOLD
                        Limit upscaling of small images to prevent quality
                        degradation (opt-in). When set, filters out aspect
                        buckets requiring upscaling beyond this threshold.
                        For example, 0.2 allows up to 20% upscaling. Default
                        (None) allows unlimited upscaling. Must be between 0
                        and 1.
  --metadata_update_interval METADATA_UPDATE_INTERVAL
                        When generating the aspect bucket indicies, we want to
                        save it every X seconds
  --debug_aspect_buckets [DEBUG_ASPECT_BUCKETS]
                        If set, will print excessive debugging for aspect
                        bucket operations
  --debug_dataset_loader [DEBUG_DATASET_LOADER]
                        If set, will print excessive debugging for data loader
                        operations
  --print_filenames [PRINT_FILENAMES]
                        If any image files are stopping the process eg. due to
                        corruption or truncation, this will help identify
                        which is at fault
  --print_sampler_statistics [PRINT_SAMPLER_STATISTICS]
                        If provided, will print statistics about the dataset
                        sampler. This is useful for debugging
  --timestep_bias_strategy {earlier,later,range,none}
                        Strategy for biasing timestep sampling
  --timestep_bias_begin TIMESTEP_BIAS_BEGIN
                        Beginning of timestep bias range
  --timestep_bias_end TIMESTEP_BIAS_END
                        End of timestep bias range
  --timestep_bias_multiplier TIMESTEP_BIAS_MULTIPLIER
                        Multiplier for timestep bias probability
  --timestep_bias_portion TIMESTEP_BIAS_PORTION
                        Portion of training steps to apply timestep bias
  --training_scheduler_timestep_spacing {leading,linspace,trailing}
                        Timestep spacing for training scheduler
  --inference_scheduler_timestep_spacing {leading,linspace,trailing}
                        Timestep spacing for inference scheduler
  --loss_type {l2,huber,smooth_l1}
                        Loss function for training
  --huber_schedule {snr,exponential,constant}
                        Schedule for Huber loss transition threshold
  --huber_c HUBER_C     Transition point between L2 and L1 regions for Huber
                        loss
  --snr_gamma SNR_GAMMA
                        SNR weighting gamma value (0 = disabled)
  --masked_loss_probability MASKED_LOSS_PROBABILITY
                        Probability of applying masked loss weighting per
                        batch
  --hidream_use_load_balancing_loss [HIDREAM_USE_LOAD_BALANCING_LOSS]
                        Apply experimental load balancing loss when training
                        HiDream models.
  --hidream_load_balancing_loss_weight HIDREAM_LOAD_BALANCING_LOSS_WEIGHT
                        Strength multiplier for HiDream load balancing loss.
  --adam_beta1 ADAM_BETA1
                        First moment decay rate for Adam optimizers
  --adam_beta2 ADAM_BETA2
                        Second moment decay rate for Adam optimizers
  --optimizer_beta1 OPTIMIZER_BETA1
                        First moment decay rate for optimizers
  --optimizer_beta2 OPTIMIZER_BETA2
                        Second moment decay rate for optimizers
  --optimizer_cpu_offload_method {none}
                        Method for CPU offloading optimizer states
  --gradient_precision {unmodified,fp32}
                        Precision for gradient computation
  --adam_weight_decay ADAM_WEIGHT_DECAY
                        L2 regularisation strength for Adam-family optimizers.
  --adam_epsilon ADAM_EPSILON
                        Small constant added for numerical stability.
  --prodigy_steps PRODIGY_STEPS
                        Number of steps Prodigy should spend adapting its
                        learning rate.
  --max_grad_norm MAX_GRAD_NORM
                        Gradient clipping threshold to prevent exploding
                        gradients.
  --grad_clip_method {value,norm}
                        Strategy for applying max_grad_norm during clipping.
  --optimizer_offload_gradients [OPTIMIZER_OFFLOAD_GRADIENTS]
                        Move optimizer gradients to CPU to save GPU memory.
  --fuse_optimizer [FUSE_OPTIMIZER]
                        Enable fused kernels when offloading to reduce memory
                        overhead.
  --optimizer_release_gradients [OPTIMIZER_RELEASE_GRADIENTS]
                        Free gradient tensors immediately after optimizer step
                        when using Optimi optimizers.
  --push_to_hub [PUSH_TO_HUB]
                        Automatically upload the trained model to your Hugging
                        Face Hub repository.
  --push_to_hub_background [PUSH_TO_HUB_BACKGROUND]
                        Run Hub uploads in a background worker so training is
                        not blocked while pushing.
  --push_checkpoints_to_hub [PUSH_CHECKPOINTS_TO_HUB]
                        Upload intermediate checkpoints to the same Hugging
                        Face repository during training.
  --publishing_config PUBLISHING_CONFIG
                        Optional JSON/file path describing additional
                        publishing targets (S3/Backblaze B2/Azure Blob/Dropbox).
  --hub_model_id HUB_MODEL_ID
                        If left blank, SimpleTuner derives a name from the
                        project settings when pushing to Hub.
  --model_card_private [MODEL_CARD_PRIVATE]
                        Create the Hugging Face repository as private instead
                        of public.
  --model_card_safe_for_work [MODEL_CARD_SAFE_FOR_WORK]
                        Remove the default NSFW warning from the generated
                        model card on Hugging Face Hub.
  --model_card_note MODEL_CARD_NOTE
                        Optional note that appears at the top of the generated
                        model card.
  --modelspec_comment MODELSPEC_COMMENT
                        Text embedded in safetensors file metadata as
                        modelspec.comment, visible in external model viewers.
  --report_to {tensorboard,wandb,comet_ml,all,none}
                        Where to log training metrics
  --checkpoint_step_interval CHECKPOINT_STEP_INTERVAL
                        Save model checkpoint every N steps (deprecated alias: --checkpointing_steps)
  --checkpoint_epoch_interval CHECKPOINT_EPOCH_INTERVAL
                        Save model checkpoint every N epochs
  --checkpointing_rolling_steps CHECKPOINTING_ROLLING_STEPS
                        Rolling checkpoint window size for continuous
                        checkpointing
  --checkpointing_use_tempdir [CHECKPOINTING_USE_TEMPDIR]
                        Use temporary directory for checkpoint files before
                        final save
  --checkpoints_rolling_total_limit CHECKPOINTS_ROLLING_TOTAL_LIMIT
                        Maximum number of rolling checkpoints to keep
  --tracker_run_name TRACKER_RUN_NAME
                        Name for this training run in tracking platforms
  --tracker_project_name TRACKER_PROJECT_NAME
                        Project name in tracking platforms
  --tracker_image_layout {gallery,table}
                        How validation images are displayed in trackers
  --enable_watermark [ENABLE_WATERMARK]
                        Add invisible watermark to generated images
  --framerate FRAMERATE
                        Framerate for video model training
  --seed_for_each_device [SEED_FOR_EACH_DEVICE]
                        Use a unique deterministic seed per GPU instead of
                        sharing one seed across devices.
  --snr_weight SNR_WEIGHT
                        Weight factor for SNR-based loss scaling
  --rescale_betas_zero_snr [RESCALE_BETAS_ZERO_SNR]
                        Rescale betas for zero terminal SNR
  --webhook_config WEBHOOK_CONFIG
                        Path to webhook configuration file
  --webhook_reporting_interval WEBHOOK_REPORTING_INTERVAL
                        Interval for webhook reports (seconds)
  --distillation_method {lcm,dcm,dmd,perflow}
                        Method for model distillation
  --distillation_config DISTILLATION_CONFIG
                        Path to distillation configuration file
  --ema_validation {none,ema_only,comparison}
                        Control how EMA weights are used during validation
                        runs.
  --local_rank LOCAL_RANK
                        Local rank for distributed training
  --ltx_train_mode {t2v,i2v}
                        Training mode for LTX models
  --ltx_i2v_prob LTX_I2V_PROB
                        Probability of using image-to-video training for LTX
  --ltx_partial_noise_fraction LTX_PARTIAL_NOISE_FRACTION
                        Fraction of noise to add for LTX partial training
  --ltx_protect_first_frame [LTX_PROTECT_FIRST_FRAME]
                        Protect the first frame from noise in LTX training
  --offload_param_path OFFLOAD_PARAM_PATH
                        Path to offloaded parameter files
  --offset_noise [OFFSET_NOISE]
                        Enable offset-noise training
  --quantize_activations [QUANTIZE_ACTIVATIONS]
                        Quantize model activations during training
  --refiner_training [REFINER_TRAINING]
                        Enable refiner model training mode
  --refiner_training_invert_schedule [REFINER_TRAINING_INVERT_SCHEDULE]
                        Invert the noise schedule for refiner training
  --refiner_training_strength REFINER_TRAINING_STRENGTH
                        Strength of refiner training
  --sdxl_refiner_uses_full_range [SDXL_REFINER_USES_FULL_RANGE]
                        Use full timestep range for SDXL refiner
  --sana_complex_human_instruction SANA_COMPLEX_HUMAN_INSTRUCTION
                        Complex human instruction for Sana model training
```
