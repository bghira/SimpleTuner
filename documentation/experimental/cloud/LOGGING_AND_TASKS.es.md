# Registro estructurado y tareas en segundo plano

Este documento cubre el sistema de registro estructurado y los workers de tareas en segundo plano en la función de entrenamiento en la nube de SimpleTuner.

## Tabla de contenidos

- [Registro estructurado](#registro-estructurado)
  - [Configuración](#configuración)
  - [Formato de registro JSON](#formato-de-registro-json)
  - [LogContext para inyección de campos](#logcontext-para-inyección-de-campos)
  - [IDs de correlación](#ids-de-correlación)
- [Tareas en segundo plano](#tareas-en-segundo-plano)
  - [Worker de sondeo del estado de trabajos](#worker-de-sondeo-del-estado-de-trabajos)
  - [Worker de procesamiento de cola](#worker-de-procesamiento-de-cola)
  - [Worker de expiración de aprobaciones](#worker-de-expiración-de-aprobaciones)
  - [Opciones de configuración](#opciones-de-configuración)
- [Depuración con logs](#depuración-con-logs)

---

## Registro estructurado

El entrenamiento en la nube de SimpleTuner usa un sistema de logging JSON estructurado que ofrece una salida de logs consistente y procesable, con seguimiento automático de IDs de correlación para trazado distribuido.

### Configuración

Configura el logging mediante variables de entorno:

```bash
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
export SIMPLETUNER_LOG_LEVEL="INFO"

# Format: "json" (structured) or "text" (traditional)
export SIMPLETUNER_LOG_FORMAT="json"

# Optional: Log to file in addition to stdout
export SIMPLETUNER_LOG_FILE="/var/log/simpletuner/cloud.log"
```

<details>
<summary>Configuración programática</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.structured_logging import (
    configure_structured_logging,
    init_from_env,
)

# Configure with explicit options
configure_structured_logging(
    level="INFO",
    json_output=True,
    log_file="/var/log/simpletuner/cloud.log",
    include_stack_info=False,  # Include stack traces for errors
)

# Or initialize from environment variables
init_from_env()
```

</details>

### Formato de registro JSON

Cuando la salida JSON está habilitada, cada entrada de log incluye:

<details>
<summary>Ejemplo de entrada de log JSON</summary>

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "INFO",
  "logger": "simpletuner.cloud.jobs",
  "message": "Job submitted successfully",
  "correlation_id": "abc123def456",
  "source": {
    "file": "jobs.py",
    "line": 350,
    "function": "submit_job"
  },
  "extra": {
    "job_id": "xyz789",
    "provider": "replicate",
    "cost_estimate": 2.50
  }
}
```

</details>

| Campo | Descripción |
|-------|-------------|
| `timestamp` | Marca de tiempo ISO 8601 en UTC |
| `level` | Nivel de log (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `logger` | Jerarquía del nombre del logger |
| `message` | Mensaje de log legible por humanos |
| `correlation_id` | ID de trazado de solicitud (autogenerado o propagado) |
| `source` | Archivo, número de línea y nombre de función |
| `extra` | Campos estructurados adicionales de LogContext |

### LogContext para inyección de campos

Usa `LogContext` para añadir campos estructurados automáticamente a todos los logs dentro de un alcance:

<details>
<summary>Ejemplo de uso de LogContext</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.structured_logging import (
    get_logger,
    LogContext,
)

logger = get_logger("simpletuner.cloud.jobs")

async def process_job(job_id: str, provider: str):
    # All logs within this block include job_id and provider
    with LogContext(job_id=job_id, provider=provider):
        logger.info("Starting job processing")

        # Nested context adds more fields
        with LogContext(step="validation"):
            logger.info("Validating configuration")

        with LogContext(step="submission"):
            logger.info("Submitting to provider")

        logger.info("Job processing complete")
```

Los logs de salida incluirán los campos de contexto:

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "INFO",
  "logger": "simpletuner.cloud.jobs",
  "message": "Starting job processing",
  "correlation_id": "abc123",
  "extra": {
    "job_id": "xyz789",
    "provider": "replicate"
  }
}
```

</details>

Campos comunes para inyectar:

| Campo | Propósito |
|-------|-----------|
| `job_id` | Identificador del trabajo de entrenamiento |
| `provider` | Proveedor en la nube (replicate, etc.) |
| `user_id` | Usuario autenticado |
| `step` | Fase de procesamiento (validation, upload, submission) |
| `attempt` | Número de intento de reintento |

### IDs de correlación

Los IDs de correlación permiten el trazado de solicitudes a través de límites de servicio. Estos:

1. **Se autogeneran** para cada nuevo hilo de solicitud si no existen
2. **Se propagan** mediante el encabezado HTTP `X-Correlation-ID`
3. **Se almacenan** en almacenamiento local de hilo para inyección automática en logs
4. **Se incluyen** en solicitudes HTTP salientes a proveedores de nube

<details>
<summary>Diagrama de flujo de ID de correlación</summary>

```
User Request
     |
     v
[X-Correlation-ID: abc123]  <-- Incoming header (or auto-generated)
     |
     v
[Thread-local storage]  <-- set_correlation_id("abc123")
     |
     +---> Log entry: {"correlation_id": "abc123", ...}
     |
     +---> Outbound HTTP: X-Correlation-ID: abc123
           (to Replicate API)
```

</details>

<details>
<summary>Gestión manual de IDs de correlación</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.http_client import (
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
)

# Get current ID (auto-generates if none exists)
current_id = get_correlation_id()

# Set a specific ID (e.g., from incoming request header)
set_correlation_id("request-abc-123")

# Clear when request completes
clear_correlation_id()
```

</details>

<details>
<summary>ID de correlación en clientes HTTP</summary>

El factory de clientes HTTP incluye automáticamente el ID de correlación en las solicitudes salientes:

```python
from simpletuner.simpletuner_sdk.server.services.cloud.http_client import (
    get_async_client,
)

# Correlation ID is automatically added to X-Correlation-ID header
async with get_async_client() as client:
    response = await client.get("https://api.replicate.com/v1/predictions")
    # Request includes: X-Correlation-ID: <current-id>
```

</details>

---

## Tareas en segundo plano

El sistema de entrenamiento en la nube ejecuta varios workers en segundo plano para manejar operaciones asíncronas.

### Administrador de tareas en segundo plano

Todas las tareas en segundo plano se gestionan mediante el singleton `BackgroundTaskManager`:

<details>
<summary>Uso del administrador de tareas</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.background_tasks import (
    get_task_manager,
    start_background_tasks,
    stop_background_tasks,
)

# Start all configured tasks (typically in app lifespan)
await start_background_tasks()

# Stop gracefully on shutdown
await stop_background_tasks()
```

</details>

### Worker de sondeo del estado de trabajos

El worker de sondeo de trabajos sincroniza los estados de trabajos desde los proveedores en la nube. Esto es útil cuando los webhooks no están disponibles (p. ej., detrás de un firewall).

**Propósito:**
- Sondear trabajos activos (pending, uploading, queued, running) desde proveedores en la nube
- Actualizar el almacén local de trabajos con el estado actual
- Emitir eventos SSE cuando cambia el estado
- Actualizar entradas de cola para estados terminales

<details>
<summary>Diagrama de flujo de sondeo</summary>

```
[Every 30 seconds]
     |
     v
List active jobs from local store
     |
     v
Group by provider
     |
     +---> [replicate] --> Get status from API --> Update local job
     |
     v
Emit SSE events for status changes
     |
     v
Update queue on terminal statuses (completed, failed, cancelled)
```

</details>

<details>
<summary>Lógica de habilitación automática</summary>

El worker de sondeo se inicia automáticamente si no hay una URL de webhook configurada:

```python
# In background_tasks.py
async def _should_auto_enable_polling(self) -> bool:
    config = await store.get_config("replicate")
    return not config.get("webhook_url")  # Enable if no webhook
```

</details>

### Worker de procesamiento de cola

Maneja la programación y el despacho de trabajos según la prioridad de la cola y los límites de concurrencia.

**Propósito:**
- Procesar la cola de trabajos cada 5 segundos
- Despachar trabajos según prioridad
- Respetar límites de concurrencia por usuario/organización
- Manejar transiciones de estado de entradas de cola

**Intervalo de procesamiento de cola:** 5 segundos (fijo)

### Worker de expiración de aprobaciones

Expira y rechaza automáticamente solicitudes de aprobación pendientes que han superado su fecha límite.

**Propósito:**
- Revisar solicitudes de aprobación expiradas cada 5 minutos
- Rechazar automáticamente trabajos con aprobaciones expiradas
- Actualizar entradas de cola a estado fallido
- Emitir notificaciones SSE para aprobaciones expiradas

<details>
<summary>Diagrama de flujo de procesamiento</summary>

```
[Every 5 minutes]
     |
     v
List pending approval requests
     |
     v
Filter expired requests (past deadline)
     |
     v
Mark approval requests as expired
     |
     +---> Update queue entries to "failed"
     |
     +---> Update job status to "cancelled"
     |
     +---> Emit SSE "approval_expired" events
```

</details>

### Opciones de configuración

#### Variable de entorno

```bash
# Set custom polling interval (seconds)
export SIMPLETUNER_JOB_POLL_INTERVAL="60"
```

<details>
<summary>Archivo de configuración empresarial</summary>

Crea `simpletuner-enterprise.yaml`:

```yaml
background:
  job_status_polling:
    enabled: true
    interval_seconds: 30
  queue_processing:
    enabled: true
    interval_seconds: 5
```

</details>

#### Propiedades de configuración

| Propiedad | Predeterminado | Descripción |
|----------|----------------|-------------|
| `job_polling_enabled` | false (auto si no hay webhook) | Habilitar sondeo explícito |
| `job_polling_interval` | 30 segundos | Intervalo de sondeo |
| Procesamiento de cola | Siempre habilitado | No se puede deshabilitar |
| Expiración de aprobaciones | Siempre habilitado | Verifica cada 5 minutos |

<details>
<summary>Acceso programático a la configuración</summary>

```python
from simpletuner.simpletuner_sdk.server.config.enterprise import get_enterprise_config

config = get_enterprise_config()

if config.job_polling_enabled:
    interval = config.job_polling_interval
    print(f"Polling enabled with {interval}s interval")
```

</details>
