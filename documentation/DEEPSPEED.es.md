# Offload de DeepSpeed / entrenamiento multi-GPU

SimpleTuner v0.7 introdujo soporte preliminar para entrenar SDXL usando DeepSpeed ZeRO etapas 1 a 3.
En v3.0, este soporte ha mejorado mucho, con un builder de configuración en WebUI, mejor soporte de optimizadores y mejor gestión de offload.

> ⚠️ DeepSpeed no está disponible en macOS (MPS) ni en sistemas ROCm.

**Entrenando SDXL 1.0 con 9237MiB de VRAM**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:08:00.0 Off |                  Off |
|  0%   43C    P2   100W / 450W |   9237MiB / 24564MiB |    100%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     11500      C   ...uner/.venv/bin/python3.13     9232MiB |
+-----------------------------------------------------------------------------+
```

Estos ahorros de memoria se han logrado mediante el uso de DeepSpeed ZeRO Stage 2 offload. Sin eso, el U-net de SDXL consumirá más de 24G de VRAM, causando la temida excepción CUDA Out of Memory.

## ¿Qué es DeepSpeed?

ZeRO significa **Zero Redundancy Optimizer**. Esta técnica reduce el consumo de memoria de cada GPU al particionar los distintos estados de entrenamiento del modelo (pesos, gradientes y estados del optimizador) a través de los dispositivos disponibles (GPUs y CPUs).

ZeRO se implementa como etapas incrementales de optimizaciones, donde las optimizaciones de etapas anteriores están disponibles en etapas posteriores. Para profundizar en ZeRO, consulta el [paper](https://arxiv.org/abs/1910.02054v3) original (1910.02054v3).

## Problemas conocidos

### Soporte LoRA

Debido a cómo DeepSpeed cambia las rutinas de guardado de modelos, actualmente no se admite entrenar LoRA con DeepSpeed.

Esto puede cambiar en una versión futura.

### Habilitar / deshabilitar DeepSpeed en checkpoints existentes

Actualmente en SimpleTuner, DeepSpeed no puede **habilitarse** al reanudar desde un checkpoint que **no** usaba DeepSpeed.

A la inversa, DeepSpeed no puede **deshabilitarse** cuando se intenta reanudar el entrenamiento desde un checkpoint entrenado usando DeepSpeed.

Para evitar este problema, exporta el pipeline de entrenamiento a un conjunto completo de pesos de modelo antes de intentar habilitar/deshabilitar DeepSpeed en una sesión de entrenamiento en curso.

Es poco probable que este soporte llegue a materializarse, ya que el optimizador de DeepSpeed es muy distinto de las opciones habituales.

## Etapas de DeepSpeed

DeepSpeed ofrece tres niveles de optimización para entrenar un modelo, y cada incremento tiene más overhead.

Especialmente para entrenamiento multi-GPU, las transferencias a CPU actualmente no están altamente optimizadas dentro de DeepSpeed.

Por este overhead, se recomienda seleccionar el nivel **más bajo** de DeepSpeed que funcione.

### Etapa 1

Los estados del optimizador (p. ej., para Adam, pesos de 32 bits y las estimaciones de primer y segundo momento) se particionan entre los procesos, de modo que cada proceso actualiza solo su partición.

### Etapa 2

Los gradientes reducidos de 32 bits para actualizar los pesos del modelo también se particionan de modo que cada proceso retiene solo los gradientes correspondientes a su porción de estados del optimizador.

### Etapa 3

Los parámetros del modelo de 16 bits se particionan entre los procesos. ZeRO-3 los recolecta y particiona automáticamente durante las pasadas forward y backward.

## Habilitar DeepSpeed

El [tutorial oficial](https://www.deepspeed.ai/tutorials/zero/) está muy bien estructurado e incluye muchos escenarios que no se describen aquí.

### Método 1: WebUI Configuration Builder (Recomendado)

SimpleTuner ahora ofrece un WebUI amigable para configurar DeepSpeed:

1. Navega al WebUI de SimpleTuner
2. Cambia a la pestaña **Hardware** y abre la sección **Accelerate & Distributed**
3. Haz clic en el botón **DeepSpeed Builder** junto al campo `DeepSpeed Config (JSON)`
4. Usa la interfaz interactiva para:
   - Seleccionar etapa de optimización ZeRO (1, 2 o 3)
   - Configurar opciones de offload (CPU, NVMe)
   - Elegir optimizadores y schedulers
   - Establecer parámetros de acumulación y clipping de gradientes
5. Previsualiza la configuración JSON generada
6. Guarda y aplica la configuración

El builder mantiene la estructura JSON consistente y cambia optimizadores no soportados a valores seguros cuando es necesario, ayudándote a evitar errores comunes de configuración.

### Método 2: Configuración JSON manual

Para usuarios que prefieren editar directamente, puedes añadir la configuración de DeepSpeed directamente en tu archivo `config.json`:

```json
{
  "deepspeed_config": {
    "zero_optimization": {
      "stage": 2,
      "offload_param": {
        "device": "cpu",
        "pin_memory": true
      },
      "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true
      }
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "optimizer": {
      "type": "AdamW",
      "params": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01
      }
    },
    "scheduler": {
      "type": "WarmupLR",
      "params": {
        "warmup_min_lr": 0,
        "warmup_max_lr": 1e-4,
        "warmup_num_steps": 500
      }
    },
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 2
  }
}
```

**Opciones clave de configuración:**

- `zero_optimization.stage`: Configura 1, 2 o 3 para distintos niveles de optimización ZeRO
- `offload_param.device`: Usa "cpu" o "nvme" para offload de parámetros
- `offload_optimizer.device`: Usa "cpu" o "nvme" para offload del optimizador
- `optimizer.type`: Elige entre optimizadores compatibles (AdamW, Adam, Adagrad, Lamb, etc.)
- `gradient_accumulation_steps`: Número de pasos para acumular gradientes

**Ejemplo de offload NVMe:**
```json
{
  "deepspeed_config": {
    "zero_optimization": {
      "stage": 3,
      "offload_param": {
        "device": "nvme",
        "nvme_path": "/path/to/nvme/storage",
        "buffer_size": 100000000.0,
        "pin_memory": true
      }
    }
  }
}
```

### Método 3: Configuración manual vía accelerate config

Para usuarios avanzados, DeepSpeed aún puede habilitarse mediante `accelerate config`:

```
----------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
No distributed training
Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)? [yes/NO]:NO
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO
----------------------------------------------------------------------------------------------------------------------------
What should be your DeepSpeed's ZeRO optimization stage?
1
How many gradient accumulation steps you're passing in your script? [1]: 4
Do you want to use gradient clipping? [yes/NO]:
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]:
How many GPU(s) should be used for distributed training? [1]:
----------------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?bf16
accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml
```

Esto da como resultado el siguiente archivo yaml:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 4
  zero3_init_flag: false
  zero_stage: 1
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## Configurar SimpleTuner

SimpleTuner no requiere configuración especial para usar DeepSpeed.

Si usas ZeRO etapa 2 o 3 con offload NVMe, puedes proporcionar `--offload_param_path=/path/to/offload` para almacenar los archivos de offload de parámetros/optimizador en una partición dedicada. Idealmente este almacenamiento debe ser un dispositivo NVMe, pero cualquier almacenamiento sirve.

### Mejoras recientes (v0.7+)

#### WebUI Configuration Builder
SimpleTuner ahora incluye un builder integral de configuración DeepSpeed en la WebUI, permitiéndote:
- Crear configuraciones JSON personalizadas de DeepSpeed mediante una interfaz intuitiva
- Auto-descubrir parámetros disponibles
- Visualizar el impacto de la configuración antes de aplicarla
- Guardar y reutilizar plantillas de configuración

#### Soporte mejorado de optimizadores
El sistema ahora incluye mejor normalización y validación de nombres de optimizadores:
- **Optimizadores soportados**: AdamW, Adam, Adagrad, Lamb, OneBitAdam, OneBitLamb, ZeroOneAdam, MuAdam, MuAdamW, MuSGD, Lion, Muon
- **Optimizadores no soportados** (reemplazados automáticamente por AdamW): cpuadam, fusedadam
- Advertencias de fallback automáticas cuando se especifican optimizadores no soportados

#### Gestión de offload mejorada
- **Limpieza automática**: Los directorios de swap de offload de DeepSpeed obsoletos se eliminan automáticamente para evitar estados corruptos al reanudar
- **Soporte NVMe mejorado**: Mejor manejo de rutas de offload NVMe con asignación automática de tamaño de buffer
- **Detección de plataforma**: DeepSpeed se deshabilita automáticamente en plataformas incompatibles (macOS/ROCm)

#### Validación de configuración
- Normalización automática de nombres de optimizadores y estructura de configuración cuando aplicas cambios
- Guardas de seguridad para selecciones de optimizador no soportadas y JSON malformado
- Manejo de errores y logging mejorados para troubleshooting

### Optimizador DeepSpeed / scheduler de learning rate

DeepSpeed usa su propio scheduler de learning rate y, por defecto, una versión altamente optimizada de AdamW, aunque no 8bit. Esto parece menos importante para DeepSpeed, ya que las cosas tienden a mantenerse más cerca de la CPU.

Si `scheduler` u `optimizer` están configurados en tu `default_config.yaml`, se usarán. Si no se definen `scheduler` u `optimizer`, se usarán las opciones por defecto `AdamW` y `WarmUp` como optimizador y scheduler, respectivamente.

## Algunos resultados rápidos de pruebas

Usando una GPU 4090 24G:

* Ahora podemos entrenar el U-net completo a 1 megapíxel (área de píxeles 1024^2) usando solo **13102MiB de VRAM para batch size 8**
  * Esto funcionó a 8 segundos por iteración. Esto significa que 1000 pasos de entrenamiento pueden hacerse en poco menos de 2 horas y media.
  * Como se indica en el tutorial de DeepSpeed, puede ser ventajoso intentar ajustar el batch size a un valor más bajo, para que la VRAM disponible se use para parámetros y estados del optimizador.
    * Sin embargo, SDXL es un modelo relativamente pequeño y potencialmente podemos evitar algunas recomendaciones si el impacto de rendimiento es aceptable.
* A tamaño de imagen **128x128** con batch size de 8, el entrenamiento consume tan solo **9237MiB de VRAM**. Este es un caso de uso potencialmente de nicho para entrenamiento de pixel art, que requiere un mapeo 1:1 con el espacio latente.

Dentro de estos parámetros, encontrarás distintos grados de éxito y probablemente incluso puedas meter el entrenamiento completo del U-net en tan solo 8GiB de VRAM a 1024x1024 para batch size 1 (no probado).

Dado que SDXL fue entrenado por muchos pasos en una distribución amplia de resoluciones y relaciones de aspecto, incluso puedes reducir el área de píxeles a .75 megapíxeles, aproximadamente 768x768, y optimizar aún más el uso de memoria.

# Soporte para dispositivos AMD

No tengo GPUs AMD de consumidor ni de estación de trabajo; sin embargo, hay reportes de que la MI50 (que ahora sale de soporte) y otras tarjetas Instinct de gama alta **sí** funcionan con DeepSpeed. AMD mantiene un repositorio para su implementación.

## Troubleshooting

### Problemas comunes y soluciones

#### "DeepSpeed crash on resume"
**Problema**: El entrenamiento falla al reanudar desde un checkpoint con offload de DeepSpeed habilitado.

**Solución**: SimpleTuner ahora limpia automáticamente los directorios de swap de offload de DeepSpeed obsoletos para evitar estados corruptos al reanudar. Este problema se resolvió en las últimas actualizaciones.

#### "Unsupported optimizer error"
**Problema**: La configuración de DeepSpeed contiene nombres de optimizadores no soportados.

**Solución**: El sistema ahora normaliza automáticamente los nombres de optimizadores y reemplaza optimizadores no soportados (cpuadam, fusedadam) con AdamW. Se registran advertencias cuando ocurren fallbacks.

#### "DeepSpeed not available on this platform"
**Problema**: Las opciones de DeepSpeed están deshabilitadas o no disponibles.

**Solución**: DeepSpeed solo se soporta en sistemas CUDA. Se deshabilita automáticamente en macOS (MPS) y sistemas ROCm. Esto es por diseño para evitar problemas de compatibilidad.

#### "NVMe offload path issues"
**Problema**: Errores relacionados con la configuración de rutas de offload NVMe.

**Solución**: Asegúrate de que `--offload_param_path` apunte a un directorio válido con suficiente espacio. El sistema ahora maneja automáticamente la asignación de tamaño de buffer y validación de rutas.

#### "Configuration validation errors"
**Problema**: Parámetros inválidos en la configuración de DeepSpeed.

**Solución**: Usa el builder de configuración en WebUI para generar el JSON; normaliza selecciones de optimizador y estructura antes de aplicar la configuración.

### Información de depuración

Para solucionar problemas de DeepSpeed, revisa lo siguiente:
- Compatibilidad de hardware vía la pestaña Hardware del WebUI (Hardware → Accelerate) o `nvidia-smi`
- Configuración de DeepSpeed en los logs de entrenamiento
- Permisos de la ruta de offload y espacio disponible
- Logs de detección de plataforma

# Entrenamiento EMA (Exponential moving average)

Aunque EMA es una gran forma de suavizar gradientes y mejorar capacidades de generalización de los pesos resultantes, es un asunto muy pesado en memoria.

EMA mantiene una copia sombra de los parámetros del modelo en memoria, duplicando esencialmente la huella del modelo. Para SimpleTuner, EMA no pasa por el módulo Accelerator, lo que significa que no está afectado por DeepSpeed. Esto significa que los ahorros de memoria que vimos con el U-net base no se logran con el modelo EMA.

Sin embargo, por defecto, el modelo EMA se mantiene en CPU.
