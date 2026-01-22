# Configuración

Para usuarios que deseen usar Docker u otra plataforma de orquestación de contenedores, consulta primero [este documento](DOCKER.md).

## Instalación

Para usuarios en Windows 10 o superior, hay una guía de instalación basada en Docker y WSL disponible aquí: [este documento](DOCKER.md).

### Método de instalación con pip

Puedes instalar SimpleTuner con pip, recomendado para la mayoría de usuarios:

```bash
# para CUDA
pip install 'simpletuner[cuda]'
# para CUDA 13 / Blackwell (GPUs NVIDIA serie B)
pip install 'simpletuner[cuda13]'
# para ROCm
pip install 'simpletuner[rocm]'
# para Apple Silicon
pip install 'simpletuner[apple]'
# solo CPU (no recomendado)
pip install 'simpletuner[cpu]'
# soporte para JPEG XL (opcional)
pip install 'simpletuner[jxl]'

# requisitos de desarrollo (opcional, solo para enviar PRs o ejecutar tests)
pip install 'simpletuner[dev]'
```

### Método del repositorio Git

Para desarrollo local o pruebas, puedes clonar el repositorio de SimpleTuner y configurar el venv de Python:

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git

cd SimpleTuner

# si python --version muestra 3.11 o 3.12, puedes actualizar a 3.13.
python3.13 -m venv .venv

source .venv/bin/activate
```

> ℹ️ Puedes usar tu propia ruta de venv configurando `export VENV_PATH=/path/to/.venv` en tu archivo `config/config.env`.

**Nota:** Actualmente instalamos la rama `release`; la rama `main` puede contener funciones experimentales que podrían dar mejores resultados o usar menos memoria.

Instala SimpleTuner con detección automática de plataforma:

```bash
# Instalación básica (auto-detecta CUDA/ROCm/Apple)
pip install -e .

# Con soporte JPEG XL
pip install -e .[jxl]
```

**Nota:** El setup.py detecta automáticamente tu plataforma (CUDA/ROCm/Apple) e instala las dependencias adecuadas.

#### Pasos posteriores para NVIDIA Hopper / Blackwell

Opcionalmente, hardware Hopper (o más nuevo) puede usar FlashAttention3 para mejorar el rendimiento de inferencia y entrenamiento al usar `torch.compile`.

Necesitarás ejecutar la siguiente secuencia de comandos desde tu directorio de SimpleTuner, con tu venv activo:

```bash
git clone https://github.com/Dao-AILab/flash-attention
pushd flash-attention
  pushd hopper
    python setup.py install
  popd
popd
```

> ⚠️ La gestión del build de flash_attn tiene poco soporte en SimpleTuner por ahora. Esto puede romperse con actualizaciones, requiriendo que vuelvas a ejecutar este procedimiento de build manualmente de vez en cuando.

#### Pasos posteriores para AMD ROCm

Lo siguiente debe ejecutarse para que un AMD MI300X sea utilizable:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
  python3 -m pip install --upgrade pip
  python3 -m pip install .
popd
```

> ℹ️ **Defaults de aceleración ROCm**: Cuando SimpleTuner detecta un build PyTorch con HIP habilitado exporta automáticamente `PYTORCH_TUNABLEOP_ENABLED=1` (a menos que ya lo hayas configurado) para que los kernels TunableOp estén disponibles. En dispositivos MI300/gfx94x también configuramos `HIPBLASLT_ALLOW_TF32=1` por defecto, habilitando rutas TF32 de hipBLASLt sin ajustes manuales de entorno.

### Todas las plataformas

- 2a. **Opción Uno (Recomendada)**: Ejecuta `simpletuner configure`
- 2b. **Opción Dos**: Copia `config/config.json.example` a `config/config.json` y luego completa los detalles.

> ⚠️ Para usuarios ubicados en países donde Hugging Face Hub no es fácilmente accesible, agrega `HF_ENDPOINT=https://hf-mirror.com` a tu `~/.bashrc` o `~/.zshrc` dependiendo de qué `$SHELL` usa tu sistema.

#### Entrenamiento con múltiples GPUs

SimpleTuner ahora incluye **detección y configuración automática de GPU** a través del WebUI. En la primera carga, se te guiará por un paso de onboarding que detecta tus GPUs y configura Accelerate automáticamente.

##### Auto-detección WebUI (Recomendado)

Cuando lanzas el WebUI por primera vez o usas `simpletuner configure`, encontrarás un paso de onboarding "Accelerate GPU Defaults" que:

1. **Detecta automáticamente** todas las GPUs disponibles en tu sistema
2. **Muestra detalles de GPU** incluyendo nombre, memoria e IDs de dispositivo
3. **Recomienda ajustes óptimos** para entrenamiento multi-GPU
4. **Ofrece tres modos de configuración:**

   - **Auto Mode** (Recomendado): Usa todas las GPUs detectadas con un conteo de procesos óptimo
   - **Manual Mode**: Selecciona GPUs específicas o configura un conteo de procesos personalizado
   - **Disabled Mode**: Solo entrenamiento en una GPU

**Cómo funciona:**
- El sistema detecta tu hardware GPU vía CUDA/ROCm
- Calcula el `--num_processes` óptimo basado en dispositivos disponibles
- Configura `CUDA_VISIBLE_DEVICES` automáticamente cuando se seleccionan GPUs específicas
- Guarda tus preferencias para futuras ejecuciones de entrenamiento

##### Configuración manual

Si no usas el WebUI, puedes controlar la visibilidad de GPUs directamente en tu `config.json`:

```json
{
  "accelerate_visible_devices": [0, 1, 2],
  "num_processes": 3
}
```

Esto restringirá el entrenamiento a las GPUs 0, 1 y 2, lanzando 3 procesos.

3. Si estás usando `--report_to='wandb'` (el default), lo siguiente te ayudará a reportar estadísticas:

```bash
wandb login
```

Sigue las instrucciones que se imprimen para localizar tu API key y configurarla.

Una vez hecho esto, tus sesiones de entrenamiento y datos de validación estarán disponibles en Weights & Biases.

> ℹ️ Si deseas deshabilitar por completo el reporte a Weights & Biases o Tensorboard, usa `--report-to=none`


4. Lanza el entrenamiento con simpletuner; los logs se escribirán en `debug.log`

```bash
simpletuner train
```

> ⚠️ En este punto, si usaste `simpletuner configure`, ¡ya terminaste! Si no, estos comandos funcionarán, pero se requiere configuración adicional. Consulta [el tutorial](TUTORIAL.md) para más información.

### Ejecutar pruebas unitarias

Para ejecutar pruebas unitarias y asegurar que la instalación terminó correctamente:

```bash
python -m unittest discover tests/
```

## Avanzado: Múltiples entornos de configuración

Para usuarios que entrenan múltiples modelos o necesitan cambiar rápidamente entre distintos datasets o ajustes, se inspeccionan dos variables de entorno al inicio.

Para usarlas:

```bash
simpletuner train env=default config_backend=env
```

- `env` será `default` por defecto, que apunta al directorio típico `SimpleTuner/config/` que esta guía te ayudó a configurar
  - Usar `simpletuner train env=pixart` usaría el directorio `SimpleTuner/config/pixart` para encontrar `config.env`
- `config_backend` será `env` por defecto, que usa el archivo típico `config.env` que esta guía te ayudó a configurar
  - Opciones soportadas: `env`, `json`, `toml`, o `cmd` si dependes de ejecutar `train.py` manualmente
  - Usar `simpletuner train config_backend=json` buscaría `SimpleTuner/config/config.json` en lugar de `config.env`
  - De forma similar, `config_backend=toml` usará `config.env`

Puedes crear `config/config.env` que contenga uno o ambos valores:

```bash
ENV=default
CONFIG_BACKEND=json
```

Se recordarán en ejecuciones posteriores. Ten en cuenta que se pueden añadir además de las opciones multiGPU descritas [arriba](#entrenamiento-con-multiples-gpus).

## Datos de entrenamiento

Hay un dataset público disponible [en Hugging Face Hub](https://huggingface.co/datasets/bghira/pseudo-camera-10k) con aproximadamente 10k imágenes con captions como nombres de archivo, listo para usar con SimpleTuner.

Puedes organizar las imágenes en una sola carpeta o en subdirectorios.

### Guías de selección de imágenes

**Requisitos de calidad:**
- Sin artefactos JPEG ni imágenes borrosas: los modelos modernos los detectarán
- Evita ruido CMOS granulado (aparecerá en todas las imágenes generadas)
- Sin marcas de agua, badges o firmas (se aprenderán)
- Los frames de película generalmente no funcionan por compresión (usa stills de producción)

**Especificaciones técnicas:**
- Imágenes óptimamente divisibles por 64 (permite reutilizar sin redimensionar)
- Mezcla imágenes cuadradas y no cuadradas para capacidades equilibradas
- Usa datasets variados y de alta calidad para mejores resultados

### Captioning

SimpleTuner proporciona [scripts de captioning](/scripts/toolkit/README.md) para renombrado masivo de archivos. Formatos de caption soportados:
- Nombre de archivo como caption (default)
- Archivos de texto con `--caption_strategy=textfile`
- JSONL, CSV o archivos avanzados de metadatos

**Herramientas de captioning recomendadas:**
- **InternVL2**: Mejor calidad pero lento (datasets pequeños)
- **BLIP3**: Mejor opción liviana con buen seguimiento de instrucciones
- **Florence2**: La más rápida pero con salidas que a algunos no les gustan

### Tamaño de batch de entrenamiento

Tu tamaño máximo de batch depende de VRAM y resolución:
```
uso_vram = batch size * resolución + requisitos_base
```

**Principios clave:**
- Usa el batch size más alto posible sin problemas de VRAM
- Mayor resolución = más VRAM = batch size más bajo
- Si batch size 1 a 128x128 no funciona, el hardware es insuficiente

#### Requisitos de dataset multi-GPU

Cuando entrenas con múltiples GPUs, tu dataset debe ser lo suficientemente grande para el **tamaño efectivo de batch**:
```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

**Ejemplo:** Con 4 GPUs y `train_batch_size=4`, necesitas al menos 16 muestras por bucket de aspecto.

**Soluciones para datasets pequeños:**
- Usa `--allow_dataset_oversubscription` para ajustar repeats automáticamente
- Configura `repeats` manualmente en tu dataloader config
- Reduce batch size o número de GPUs

Consulta [DATALOADER.md](DATALOADER.md#multi-gpu-training-and-dataset-sizing) para detalles completos.

## Publicar en Hugging Face Hub

Para hacer push automático de modelos al Hub al finalizar, añade a `config/config.json`:

```json
{
  "push_to_hub": true,
  "hub_model_name": "your-model-name"
}
```

Inicia sesión antes de entrenar:
```bash
huggingface-cli login
```

## Debugging

Habilita logging detallado añadiendo a `config/config.env`:

```bash
export SIMPLETUNER_LOG_LEVEL=DEBUG
export SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG
```

Se creará un archivo `debug.log` en la raíz del proyecto con todas las entradas de log.
