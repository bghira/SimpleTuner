# Cola de trabajos

El sistema de cola gestiona la programación de trabajos, límites de concurrencia y asignación de GPU para trabajos de entrenamiento locales y en la nube. Habilita funciones como programación nocturna, gestión de recursos GPU y uso controlado de recursos.

## Resumen

Cuando envías un trabajo de entrenamiento en la nube, se agrega a la cola y se procesa según:

- **Prioridad** - Los trabajos con mayor prioridad se ejecutan primero
- **Límites de concurrencia** - Límites globales y por usuario evitan agotamiento de recursos
- **FIFO dentro de la prioridad** - Trabajos al mismo nivel de prioridad se ejecutan en orden de envío

## Estado de la cola

Accede al panel de cola haciendo clic en el **icono de cola** en la barra de acciones de la pestaña Cloud. El panel muestra:

| Métrica | Descripción |
|--------|-------------|
| **Queued** | Trabajos en espera de ejecución |
| **Running** | Trabajos en ejecución actualmente |
| **Max Concurrent** | Límite global de trabajos simultáneos |
| **Avg Wait** | Tiempo promedio en cola |

## Niveles de prioridad

Los trabajos se asignan prioridad según el nivel de usuario:

| Nivel de usuario | Prioridad | Valor |
|------------|----------|-------|
| Admin | Urgent | 30 |
| Lead | High | 20 |
| Researcher | Normal | 10 |
| Viewer | Low | 0 |

Valores más altos = mayor prioridad = se procesa primero.

### Override de prioridad

Leads y admins pueden sobrescribir la prioridad de un trabajo para situaciones específicas (p. ej., experimentos urgentes).

## Límites de concurrencia

Dos límites controlan cuántos trabajos pueden ejecutarse simultáneamente:

### Límite global (`max_concurrent`)

Máximo de trabajos en ejecución entre todos los usuarios. Default: **5 trabajos**.

### Límite por usuario (`user_max_concurrent`)

Máximo de trabajos que un usuario puede ejecutar a la vez. Default: **2 trabajos**.

Esto evita que un usuario consuma todos los slots disponibles.

### Actualizar límites

Los admins pueden actualizar límites vía el panel de cola o la API.

<details>
<summary>Ejemplo</summary>

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{"max_concurrent": 10, "user_max_concurrent": 3}'
```

</details>

## Ciclo de vida de trabajos en la cola

1. **Submitted** - Trabajo creado, agregado a la cola con estado `pending`
2. **Pending** - Esperando un slot (límite de concurrencia)
3. **Running** - Entrenando activamente en GPU en la nube
4. **Completed/Failed** - Estado terminal, removido de la cola activa

## Endpoints de API

### Listar entradas de cola

```
GET /api/queue
```

Parámetros:
- `status` - Filtrar por estado (pending, running, blocked)
- `limit` - Máximo de entradas a devolver (default: 50)
- `include_completed` - Incluir trabajos finalizados

### Estadísticas de cola

```
GET /api/queue/stats
```

<details>
<summary>Ejemplo de respuesta</summary>

```json
{
  "queue_depth": 3,
  "running": 2,
  "max_concurrent": 5,
  "user_max_concurrent": 2,
  "avg_wait_seconds": 45.2,
  "by_status": {"pending": 3, "running": 2},
  "by_user": {"1": 2, "2": 3}
}
```

</details>

### Mi estado de cola

```
GET /api/queue/me
```

Devuelve la posición de cola del usuario actual, trabajos pendientes y trabajos en ejecución.

<details>
<summary>Ejemplo de respuesta</summary>

```json
{
  "pending_count": 2,
  "running_count": 1,
  "blocked_count": 0,
  "best_position": 3,
  "pending_jobs": [...],
  "running_jobs": [...]
}
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `pending_count` | int | Número de trabajos esperando en cola |
| `running_count` | int | Número de trabajos en ejecución |
| `blocked_count` | int | Número de trabajos en espera de aprobación |
| `best_position` | int o null | Posición del trabajo pendiente de mayor prioridad del usuario |
| `pending_jobs` | array | Lista de detalles de trabajos pendientes |
| `running_jobs` | array | Lista de detalles de trabajos en ejecución |

El campo `best_position` indica la posición en cola del mejor trabajo pendiente del usuario (mayor prioridad o más antiguo). Esto ayuda a entender cuándo comenzará el próximo trabajo. Un valor `null` significa que el usuario no tiene trabajos pendientes.

</details>

### Posición de trabajo

```
GET /api/queue/position/{job_id}
```

Devuelve la posición en cola de un trabajo específico.

### Cancelar trabajo en cola

```
POST /api/queue/{job_id}/cancel
```

Cancela un trabajo que aún no ha comenzado.

### Aprobar trabajo bloqueado

```
POST /api/queue/{job_id}/approve
```

Solo admin. Aprueba un trabajo que requiere aprobación (p. ej., excede un umbral de costo).

### Rechazar trabajo bloqueado

```
POST /api/queue/{job_id}/reject?reason=<reason>
```

Solo admin. Rechaza un trabajo bloqueado con un motivo.

### Actualizar concurrencia

```
POST /api/queue/concurrency
```

<details>
<summary>Request body</summary>

```json
{
  "max_concurrent": 10,
  "user_max_concurrent": 3
}
```

</details>

### Disparar procesamiento

```
POST /api/queue/process
```

Solo admin. Dispara manualmente el procesamiento de cola (normalmente automático).

### Limpieza de entradas antiguas

```
POST /api/queue/cleanup?days=30
```

Solo admin. Elimina entradas completadas más antiguas que los días especificados.

**Parámetros:**

| Parámetro | Tipo | Default | Rango | Descripción |
|-----------|------|---------|-------|-------------|
| `days` | int | 30 | 1-365 | Período de retención en días |

**Comportamiento:**

Elimina entradas de cola que:
- Tienen estado terminal (`completed`, `failed` o `cancelled`)
- Tienen `completed_at` más antiguo que los días especificados

Los trabajos activos (pending, running, blocked) nunca se eliminan por limpieza.

<details>
<summary>Respuesta y ejemplos</summary>

**Respuesta:**

```json
{
  "success": true,
  "deleted": 42,
  "days": 30
}
```

**Uso de ejemplo:**

```bash
# Limpiar entradas más antiguas de 7 días
curl -X POST "http://localhost:8000/api/queue/cleanup?days=7" \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Limpiar entradas más antiguas de 90 días (limpieza trimestral)
curl -X POST "http://localhost:8000/api/queue/cleanup?days=90" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

</details>

## Arquitectura

<details>
<summary>Diagrama del sistema</summary>

```
┌─────────────────────────────────────────────────────────────┐
│                     Job Submission                          │
│              (routes/cloud/jobs.py:submit_job)              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  JobSubmissionService                       │
│              Uploads data, submits to provider              │
│                    Enqueues job                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    QueueScheduler                           │
│   ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│   │ Queue Store │  │   Policy     │  │ Background Task │    │
│   │  (SQLite)   │  │  (Priority)  │  │    (5s loop)    │    │
│   └─────────────┘  └──────────────┘  └─────────────────┘    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   QueueDispatcher                           │
│            Updates job status, syncs with provider          │
└─────────────────────────────────────────────────────────────┘
```

</details>

### Componentes

| Componente | Ubicación | Descripción |
|-----------|----------|-------------|
| `JobRepository` | `storage/job_repository.py` | Persistencia SQLite unificada para trabajos y cola |
| `JobRepoQueueAdapter` | `queue/job_repo_adapter.py` | Adaptador para compatibilidad del scheduler |
| `QueueScheduler` | `queue/scheduler.py` | Lógica de scheduling |
| `SchedulingPolicy` | `queue/scheduler.py` | Algoritmo de prioridad/equidad |
| `QueueDispatcher` | `queue/dispatcher.py` | Maneja el despacho de trabajos |
| `QueueEntry` | `queue/models.py` | Modelo de datos de entrada de cola |
| `LocalGPUAllocator` | `services/local_gpu_allocator.py` | Asignación de GPU para trabajos locales |

### Esquema de base de datos

Las entradas de cola y trabajos se almacenan en la base SQLite unificada (`~/.simpletuner/cloud/jobs.db`).

<details>
<summary>Definición de esquema</summary>

```sql
CREATE TABLE queue (
    id INTEGER PRIMARY KEY,
    job_id TEXT UNIQUE NOT NULL,
    user_id INTEGER,
    team_id TEXT,
    provider TEXT DEFAULT 'replicate',
    config_name TEXT,
    priority INTEGER DEFAULT 10,
    priority_override INTEGER,
    status TEXT DEFAULT 'pending',
    position INTEGER DEFAULT 0,
    queued_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    estimated_cost REAL DEFAULT 0.0,
    requires_approval INTEGER DEFAULT 0,
    approval_id INTEGER,
    attempt INTEGER DEFAULT 1,
    max_attempts INTEGER DEFAULT 3,
    error_message TEXT,
    metadata TEXT,
    -- GPU allocation (schema v3)
    allocated_gpus TEXT,          -- JSON array of device indices, e.g., "[0,1]"
    job_type TEXT DEFAULT 'cloud', -- "local" or "cloud"
    num_processes INTEGER DEFAULT 1 -- Number of GPUs required
);
```

Las migraciones se ejecutan automáticamente al inicio.

</details>

## Concurrencia de GPU local

Al enviar trabajos de entrenamiento local, el sistema de cola rastrea la asignación de GPU para evitar conflictos de recursos. Los trabajos se encolan si las GPUs requeridas no están disponibles.

### Seguimiento de asignación de GPU

Cada trabajo local especifica:

- **num_processes** - Número de GPUs requeridas (de `--num_processes`)
- **device_ids** - Índices de GPU preferidos (de `--accelerate_visible_devices`)

El asignador rastrea qué GPUs están asignadas a trabajos en ejecución y solo inicia nuevos trabajos cuando hay recursos disponibles.

### Opciones CLI

#### Enviar trabajos

<details>
<summary>Ejemplos</summary>

```bash
# Enviar un trabajo, encolar si GPUs no están disponibles (default)
simpletuner jobs submit my-config

# Rechazar inmediatamente si GPUs no están disponibles
simpletuner jobs submit my-config --no-wait

# Usar cualquier GPU disponible en lugar de IDs configurados
simpletuner jobs submit my-config --any-gpu

# Dry-run para comprobar disponibilidad de GPU
simpletuner jobs submit my-config --dry-run
```

</details>

#### Listar trabajos

<details>
<summary>Ejemplos</summary>

```bash
# Listar trabajos recientes
simpletuner jobs list

# Listar con campos específicos
simpletuner jobs list -o job_id,status,config_name

# Salida JSON con campos personalizados
simpletuner jobs list --format json -o job_id,status

# Acceder a campos anidados con notación dot
simpletuner jobs list --format json -o job_id,metadata.run_name

# Filtrar por estado
simpletuner jobs list --status running
simpletuner jobs list --status queued

# Limitar resultados
simpletuner jobs list -l 10
```

La opción `-o` (output) soporta notación dot para acceder a campos anidados en los metadatos del trabajo. Por ejemplo, `metadata.run_name` extrae el campo `run_name` del objeto de metadatos del trabajo.

</details>

### API de estado de GPU

El estado de asignación de GPU está disponible vía el endpoint de estado del sistema:

```
GET /api/system/status?include_allocation=true
```

<details>
<summary>Ejemplo de respuesta</summary>

```json
{
  "timestamp": 1704067200.0,
  "load_avg_5min": 2.5,
  "memory_percent": 45.2,
  "gpus": [...],
  "gpu_inventory": {
    "backend": "cuda",
    "count": 4,
    "capabilities": {...}
  },
  "gpu_allocation": {
    "allocated_gpus": [0, 1],
    "available_gpus": [2, 3],
    "running_local_jobs": 1,
    "devices": [
      {"index": 0, "name": "A100", "memory_gb": 40, "allocated": true, "job_id": "abc123"},
      {"index": 1, "name": "A100", "memory_gb": 40, "allocated": true, "job_id": "abc123"},
      {"index": 2, "name": "A100", "memory_gb": 40, "allocated": false, "job_id": null},
      {"index": 3, "name": "A100", "memory_gb": 40, "allocated": false, "job_id": null}
    ]
  }
}
```

</details>

Las estadísticas de cola también incluyen info de GPU local:

```
GET /api/queue/stats
```

<details>
<summary>Ejemplo de respuesta</summary>

```json
{
  "queue_depth": 3,
  "running": 2,
  "max_concurrent": 5,
  "local_gpu_max_concurrent": 6,
  "local_job_max_concurrent": 2,
  "local": {
    "running_jobs": 1,
    "pending_jobs": 0,
    "allocated_gpus": [0, 1],
    "available_gpus": [2, 3],
    "total_gpus": 4,
    "max_concurrent_gpus": 6,
    "max_concurrent_jobs": 2
  }
}
```

</details>

### Límites de concurrencia local

Controla cuántos trabajos locales y GPUs pueden usarse simultáneamente mediante el endpoint de concurrencia existente:

```
GET /api/queue/concurrency
POST /api/queue/concurrency
```

El endpoint de concurrencia ahora acepta límites locales de GPU junto con límites de nube:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `max_concurrent` | int | Máximo de trabajos en la nube en ejecución (default: 5) |
| `user_max_concurrent` | int | Máximo de trabajos en la nube por usuario (default: 2) |
| `local_gpu_max_concurrent` | int o null | Máximo de GPUs para trabajos locales (null = ilimitado) |
| `local_job_max_concurrent` | int | Máximo de trabajos locales simultáneos (default: 1) |

<details>
<summary>Ejemplo</summary>

```bash
# Permitir hasta 2 trabajos locales usando hasta 6 GPUs en total
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{"local_gpu_max_concurrent": 6, "local_job_max_concurrent": 2}'
```

</details>

### API de envío de trabajos locales

```
POST /api/queue/submit
```

<details>
<summary>Request y respuesta</summary>

**Request body:**

```json
{
  "config_name": "my-training-config",
  "no_wait": false,
  "any_gpu": false
}
```

**Respuesta:**

```json
{
  "success": true,
  "job_id": "abc123",
  "status": "running",
  "allocated_gpus": [0, 1],
  "queue_position": null
}
```

</details>

Valores de estado:

| Estado | Descripción |
|--------|-------------|
| `running` | Trabajo iniciado de inmediato con GPUs asignadas |
| `queued` | Trabajo en cola, iniciará cuando haya GPUs disponibles |
| `rejected` | GPUs no disponibles y `no_wait` era true |

### Procesamiento automático de trabajos

Cuando un trabajo se completa o falla, sus GPUs se liberan y la cola se procesa para iniciar trabajos pendientes. Esto ocurre automáticamente vía los hooks de lifecycle del process keeper.

**Comportamiento de cancelación**: Cuando se cancela un trabajo, las GPUs se liberan pero los trabajos pendientes NO se inician automáticamente. Esto evita condiciones de carrera durante cancelaciones masivas (`simpletuner jobs cancel --all`) donde trabajos pendientes iniciarían antes de poder ser cancelados. Usa `POST /api/queue/process` o reinicia el servidor para disparar el procesamiento de cola manualmente tras la cancelación.

## Despacho a workers

Los trabajos pueden enviarse a workers remotos en lugar de ejecutarse en las GPUs locales del orquestador. Consulta [Worker Orchestration](experimental/server/WORKERS.md) para la configuración completa.

### Destinos de trabajo

Al enviar un trabajo, especifica dónde debe ejecutarse:

| Destino | Comportamiento |
|--------|----------|
| `auto` (default) | Intenta workers remotos primero, hace fallback a GPUs locales |
| `worker` | Envía solo a workers remotos; encola si no hay disponibles |
| `local` | Ejecuta solo en GPUs locales del orquestador |

<details>
<summary>Ejemplos</summary>

```bash
# CLI
simpletuner jobs submit my-config --target=worker

# API
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{"config_name": "my-config", "target": "worker"}'
```

</details>

### Selección de workers

Los trabajos pueden especificar requisitos de etiquetas para el matching de workers:

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-config",
    "target": "worker",
    "worker_labels": {"gpu_type": "a100*", "location": "us-*"}
  }'
```

Las etiquetas soportan patrones glob. El scheduler asigna trabajos a workers según:

1. Requisitos de etiquetas (todos deben coincidir)
2. Requisitos de cantidad de GPU
3. Disponibilidad de worker (estado IDLE)
4. Orden FIFO dentro de workers coincidentes

### Comportamiento al inicio

Al iniciar el servidor, el sistema de cola procesa automáticamente cualquier trabajo local pendiente. Si hay GPUs disponibles, los trabajos encolados se iniciarán inmediatamente sin intervención manual. Esto asegura que los trabajos enviados antes de un reinicio continúen procesándose cuando el servidor vuelva en línea.

La secuencia de inicio:
1. El servidor inicializa el asignador de GPU
2. Se recuperan trabajos locales pendientes de la cola
3. Para cada trabajo pendiente con GPUs disponibles, el trabajo se inicia
4. Trabajos que no pueden iniciar (GPUs insuficientes) permanecen en cola

Nota: Los trabajos en la nube son gestionados por el scheduler de la cola de nube separado, que también se reanuda al inicio.

## Configuración

Los límites de concurrencia de la cola se configuran vía la API y se persisten en la base de datos de la cola.

**Vía Web UI:** pestaña Cloud → Queue Panel → Settings

<details>
<summary>Ejemplo de configuración API</summary>

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{
    "max_concurrent": 5,
    "user_max_concurrent": 2,
    "team_max_concurrent": 10,
    "enable_fair_share": false
  }'
```

</details>

| Ajuste | Default | Descripción |
|---------|---------|-------------|
| `max_concurrent` | 5 | Máximo global de trabajos en ejecución |
| `user_max_concurrent` | 2 | Máximo de trabajos en ejecución por usuario |
| `team_max_concurrent` | 10 | Máximo de trabajos en ejecución por equipo |
| `enable_fair_share` | false | Habilitar límites de concurrencia por equipo |

### Scheduling de fair-share

Cuando `enable_fair_share: true`, el scheduler considera la afiliación a equipos para evitar que un solo equipo monopolice recursos.

#### Cómo funciona

Fair-share agrega una tercera capa de control de concurrencia:

| Capa | Límite | Propósito |
|-------|-------|---------|
| Global | `max_concurrent` | Trabajos totales entre todos los usuarios/equipos |
| Por usuario | `user_max_concurrent` | Evita que un usuario consuma todos los slots |
| Por equipo | `team_max_concurrent` | Evita que un equipo consuma todos los slots |

Cuando un trabajo se considera para despacho:

1. Verificar límite global → omitir si está al máximo
2. Verificar límite por usuario → omitir si el usuario está al máximo
3. Si fair-share está habilitado Y el trabajo tiene `team_id`:
   - Verificar límite por equipo → omitir si el equipo está al máximo

Los trabajos sin `team_id` no están sujetos a límites por equipo.

#### Habilitar fair-share

**Vía UI:** pestaña Cloud → Queue Panel → Toggle "Fair-Share Scheduling"

<details>
<summary>Ejemplo de API</summary>

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{
    "max_concurrent": 10,
    "user_max_concurrent": 3,
    "team_max_concurrent": 5,
    "enable_fair_share": true
  }'
```

</details>

#### Asignación de equipos

Los equipos se asignan a usuarios en el panel admin. Cuando un usuario envía un trabajo, su ID de equipo se adjunta a la entrada de cola. El scheduler rastrea trabajos en ejecución por equipo y hace cumplir el límite.

<details>
<summary>Escenario de ejemplo</summary>

Con `max_concurrent=6`, `user_max_concurrent=2`, `team_max_concurrent=3`:

| Equipo | Usuarios | Enviados | En ejecución | Bloqueados |
|------|-------|-----------|---------|---------|
| Alpha | Alice, Bob | 4 | 3 (en límite de equipo) | 1 |
| Beta | Carol | 3 | 2 | 1 (esperando slot global) |

- El equipo Alpha tiene 3 en ejecución (en `team_max_concurrent`)
- Total en ejecución es 5 (bajo `max_concurrent=6`)
- El trabajo de Carol está bloqueado porque: 5+1=6, en límite global
- El 4º trabajo de Alice está bloqueado porque: equipo en 3/3

Esto asegura que ningún equipo monopolice la cola aunque envíen muchos trabajos.

</details>

### Prevención de hambre (starvation)

Los trabajos que esperan más de `starvation_threshold_minutes` reciben un boost de prioridad para evitar esperas indefinidas.

## Flujo de aprobación

Los trabajos pueden marcarse como requieren aprobación (p. ej., cuando el costo estimado excede un umbral):

1. Trabajo enviado con `requires_approval: true`
2. Trabajo entra en estado `blocked`
3. Admin revisa en el panel de cola o vía API
4. Admin aprueba o rechaza
5. Si se aprueba, el trabajo pasa a `pending` y se programa normalmente

Consulta [Enterprise Guide](experimental/server/ENTERPRISE.md) para configuración de reglas de aprobación.

## Troubleshooting

### Trabajos atascados en la cola

<details>
<summary>Pasos de depuración</summary>

Revisa límites de concurrencia:
```bash
curl http://localhost:8000/api/queue/stats
```

Si `running` es igual a `max_concurrent`, los trabajos esperan slots.

</details>

### La cola no se procesa

<details>
<summary>Pasos de depuración</summary>

El procesador en background corre cada 5 segundos. Revisa logs del servidor por errores:
```
Queue scheduler started with 5s processing interval
```

Si no aparece, el scheduler pudo no iniciarse.

</details>

### Trabajo desapareció de la cola

<details>
<summary>Pasos de depuración</summary>

Verifica si se completó o falló:
```bash
curl "http://localhost:8000/api/queue?include_completed=true"
```

</details>

### Trabajos locales muestran running pero no entrenan

<details>
<summary>Pasos de depuración</summary>

Si `jobs list` muestra trabajos locales como "running" pero no hay entrenamiento:

1. Revisa estado de asignación de GPU:
   ```bash
   simpletuner jobs status --format json
   ```
   Mira el campo `local.allocated_gpus` - debería mostrar qué GPUs están en uso.

2. Si `allocated_gpus` está vacío pero el conteo running no es cero, el estado de la cola puede estar inconsistente. Reinicia el servidor para disparar la reconciliación automática de la cola.

3. Revisa logs del servidor por errores de asignación de GPU:
   ```
   Failed to allocate GPUs [0] to job <job_id>
   ```

</details>

### La profundidad de cola muestra un conteo incorrecto

<details>
<summary>Explicación</summary>

La profundidad de cola y los conteos de trabajos en ejecución se calculan por separado para trabajos locales y en la nube:

- **Trabajos locales**: Rastreado vía `LocalGPUAllocator` basado en estado de asignación de GPU
- **Trabajos en la nube**: Rastreado vía `QueueScheduler` basado en estado del proveedor

Usa `simpletuner jobs status --format json` para ver el desglose:
- `local.running_jobs` - Trabajos locales en ejecución
- `local.pending_jobs` - Trabajos locales en cola esperando GPUs
- `running` - Total de trabajos en ejecución (cola cloud)
- `queue_depth` - Trabajos pendientes en cola cloud

</details>

## Ver también

- [Worker Orchestration](experimental/server/WORKERS.md) - Registro de workers distribuidos y despacho de trabajos
- [Cloud Training Tutorial](experimental/cloud/TUTORIAL.md) - Primeros pasos con entrenamiento en la nube
- [Enterprise Guide](experimental/server/ENTERPRISE.md) - Configuración multiusuario, aprobaciones, gobernanza
- [Operations Guide](experimental/cloud/OPERATIONS_TUTORIAL.md) - Despliegue en producción
