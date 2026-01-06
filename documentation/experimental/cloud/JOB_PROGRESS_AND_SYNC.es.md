# Progreso de trabajos y API de sincronización

Este documento cubre los mecanismos de SimpleTuner para monitorear el progreso de trabajos de entrenamiento en la nube y mantener sincronizado el estado local de trabajos con los proveedores.

## Resumen

SimpleTuner ofrece múltiples enfoques para rastrear el estado de los trabajos:

| Método | Caso de uso | Latencia | Uso de recursos |
|--------|----------|---------|----------------|
| API de progreso inline | Sondeo de UI para trabajos en ejecución | Baja (5 s por defecto) | Llamadas por trabajo |
| Sincronización de trabajos (pull) | Descubrir trabajos del proveedor | Media (bajo demanda) | Llamada batch a la API |
| Parámetro `sync_active` | Actualizar estados activos | Media (bajo demanda) | Llamadas por trabajo activo |
| Poller en background | Actualizaciones automáticas de estado | Configurable (30 s por defecto) | Sondeo continuo |
| Webhooks | Notificaciones push en tiempo real | Instantánea | Sin sondeo |

## API de progreso inline

### Endpoint

```
GET /api/cloud/jobs/{job_id}/inline-progress
```

### Propósito

Devuelve información compacta de progreso para un solo trabajo en ejecución, útil para mostrar actualizaciones inline en una lista de trabajos sin obtener logs completos.

### Respuesta

```json
{
  "job_id": "abc123",
  "stage": "Training",
  "last_log": "Step 1500/5000 - loss: 0.0234",
  "progress": 30.0
}
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `job_id` | string | Identificador del trabajo |
| `stage` | string o null | Etapa actual: `Preprocessing`, `Warmup`, `Training`, `Validation`, `Saving checkpoint` |
| `last_log` | string o null | Última línea de log (truncada a 80 chars) |
| `progress` | float o null | Porcentaje de progreso (0-100) basado en parseo de pasos/épocas |

### Detección de etapa

La API analiza líneas recientes de logs para determinar la etapa:

- **Preprocessing**: detectado cuando los logs contienen "preprocessing" o "loading"
- **Warmup**: detectado cuando los logs contienen "warming up" o "warmup"
- **Training**: detectado cuando los logs contienen patrones de "step" o "epoch"
- **Validation**: detectado cuando los logs contienen "validat"
- **Saving checkpoint**: detectado cuando los logs contienen "checkpoint"

### Cálculo de progreso

El porcentaje de progreso se extrae de patrones de log como:
- `step 1500/5000` -> 30%
- `epoch 3/10` -> 30%

### Cuándo usarlo

Usa la API de progreso inline cuando:
- Muestras estado compacto en tarjetas de lista de trabajos
- Sondeas frecuentemente (cada 5 s) solo trabajos en ejecución
- Necesitas mínima transferencia de datos por solicitud

<details>
<summary>Ejemplo de cliente (JavaScript)</summary>

```javascript
async function fetchInlineProgress() {
    const runningJobs = jobs.filter(j => j.status === 'running');

    for (const job of runningJobs) {
        try {
            const response = await fetch(
                `/api/cloud/jobs/${job.job_id}/inline-progress`
            );
            if (response.ok) {
                const data = await response.json();
                // Update job card with progress info
                job.inline_stage = data.stage;
                job.inline_log = data.last_log;
                job.inline_progress = data.progress;
            }
        } catch (error) {
            // Silently ignore - job may have completed
        }
    }
}

// Poll every 5 seconds
setInterval(fetchInlineProgress, 5000);
```

</details>

## Mecanismos de sincronización de trabajos

SimpleTuner ofrece dos enfoques de sincronización para mantener actualizado el estado local de trabajos con los proveedores.

### 1. Sincronización completa del proveedor

#### Endpoint

```
POST /api/cloud/jobs/sync
```

#### Propósito

Descubre trabajos del proveedor en la nube que pueden no existir en el almacenamiento local. Esto es útil cuando:
- Los trabajos se enviaron fuera de SimpleTuner (directamente vía API de Replicate)
- El almacenamiento local de trabajos se restableció o se corrompió
- Quieres importar trabajos históricos

#### Respuesta

```json
{
  "synced": 3,
  "message": "Discovered 3 new jobs from Replicate"
}
```

#### Comportamiento

1. Obtiene hasta 100 trabajos recientes de Replicate
2. Para cada trabajo:
   - Si no está en el store local: crea un nuevo registro `UnifiedJob`
   - Si ya está en el store: actualiza estado, costo y timestamps
3. Devuelve el conteo de trabajos recién descubiertos

<details>
<summary>Ejemplo de cliente</summary>

```bash
# Sync jobs from Replicate
curl -X POST http://localhost:8001/api/cloud/jobs/sync

# Response
{"synced": 2, "message": "Discovered 2 new jobs from Replicate"}
```

</details>

#### Botón de sincronización en la Web UI

El dashboard de Cloud incluye un botón de sincronización para descubrir trabajos huérfanos:

1. Haz clic en el botón **Sync** en la barra de herramientas de la lista de trabajos
2. El botón muestra un spinner de carga durante la sincronización
3. En éxito, un toast muestra: *"Discovered N jobs from Replicate"*
4. La lista de trabajos y las métricas se refrescan automáticamente

**Casos de uso:**
- Descubrir trabajos enviados directamente vía API o consola web de Replicate
- Recuperación tras un reset de base de datos
- Importar trabajos desde una cuenta de equipo compartida en Replicate

El botón Sync llama internamente a `POST /api/cloud/jobs/sync` y luego recarga la lista de trabajos y las métricas del dashboard.

### 2. Sincronización de estado de trabajos activos (`sync_active`)

#### Endpoint

```
GET /api/cloud/jobs?sync_active=true
```

#### Propósito

Refresca el estado de todos los trabajos activos (no terminales) antes de devolver la lista de trabajos. Esto aporta estado actualizado sin esperar el polling en background.

#### Estados activos

Los trabajos en estos estados se consideran "activos" y se sincronizan:
- `pending` - Trabajo enviado pero aún no iniciado
- `uploading` - Subida de datos en progreso
- `queued` - En espera en la cola del proveedor
- `running` - Entrenamiento en progreso

#### Comportamiento

1. Antes de listar trabajos, obtiene el estado actual de cada trabajo activo en la nube
2. Actualiza el store local con:
   - Estado actual
   - Timestamps `started_at` / `completed_at`
   - `cost_usd` (costo acumulado)
   - `error_message` (si falló)
3. Devuelve la lista de trabajos actualizada

<details>
<summary>Ejemplo de cliente (JavaScript)</summary>

```javascript
// Load jobs with active status sync
async function loadJobs(syncActive = false) {
    const params = new URLSearchParams({
        limit: '50',
        provider: 'replicate',
    });

    if (syncActive) {
        params.set('sync_active', 'true');
    }

    const response = await fetch(`/api/cloud/jobs?${params}`);
    const data = await response.json();
    return data.jobs;
}

// Use sync_active during periodic refresh
setInterval(() => loadJobs(true), 30000);
```

</details>

### Comparación: sync vs sync_active

| Función | `POST /jobs/sync` | `GET /jobs?sync_active=true` |
|---------|-------------------|------------------------------|
| Descubre trabajos nuevos | Sí | No |
| Actualiza trabajos existentes | Sí | Sí (solo activos) |
| Alcance | Todos los trabajos del proveedor | Solo trabajos activos locales |
| Caso de uso | Importación inicial, recuperación | Refresco regular de estado |
| Rendimiento | Más pesado (consulta batch) | Más liviano (selectivo) |

## Configuración del poller en background

El poller en background sincroniza automáticamente estados de trabajos activos sin intervención del cliente.

### Comportamiento por defecto

- **Auto‑habilitado**: Si no hay URL de webhook configurada, el polling inicia automáticamente
- **Intervalo por defecto**: 30 segundos
- **Alcance**: Todos los trabajos activos en la nube

<details>
<summary>Configuración enterprise</summary>

Para despliegues en producción, configura el polling vía `simpletuner-enterprise.yaml`:

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

<details>
<summary>Variables de entorno</summary>

```bash
# Set custom polling interval (in seconds)
export SIMPLETUNER_JOB_POLL_INTERVAL=60
```

</details>

### Cómo funciona

1. Al iniciar el servidor, `BackgroundTaskManager` verifica:
   - Si la configuración enterprise habilita polling explícitamente, usa ese intervalo
   - De lo contrario, si no hay webhook configurado, auto‑habilita con intervalo de 30 s
2. En cada intervalo, el poller:
   - Lista todos los trabajos con estado activo
   - Agrupa por proveedor
   - Obtiene el estado actual de cada proveedor
   - Actualiza el store local
   - Emite eventos SSE para cambios de estado
   - Actualiza entradas de cola para estados terminales

<details>
<summary>Eventos SSE</summary>

Cuando el poller detecta cambios de estado, transmite eventos SSE:

```javascript
// Subscribe to SSE events
const eventSource = new EventSource('/api/events');

eventSource.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'job_status_changed') {
        console.log(`Job ${data.job_id} is now ${data.status}`);
        // Refresh job list
        loadJobs();
    }
});
```

</details>

<details>
<summary>Acceso programático</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.background_tasks import (
    get_task_manager,
    start_background_tasks,
    stop_background_tasks,
)

# Get the manager
manager = get_task_manager()

# Check if running
if manager._running:
    print("Background tasks are active")

# Manual start/stop (usually handled by app lifespan)
await start_background_tasks()
await stop_background_tasks()
```

</details>

## Buenas prácticas

### 1. Elige la estrategia de sincronización adecuada

| Escenario | Enfoque recomendado |
|----------|---------------------|
| Carga inicial de página | `GET /jobs` sin sync (rápido) |
| Refresco periódico (30 s) | `GET /jobs?sync_active=true` |
| Usuario hace clic en "Refresh" | `POST /jobs/sync` para descubrir |
| Detalles de trabajo en ejecución | API de progreso inline (5 s) |
| Despliegue en producción | Poller en background + webhooks |

### 2. Evita el sobre‑sondeo

<details>
<summary>Ejemplo</summary>

```javascript
// Good: Poll inline progress only for running jobs
const runningJobs = jobs.filter(j => j.status === 'running');

// Bad: Poll all jobs regardless of status
for (const job of jobs) { /* ... */ }
```

</details>

### 3. Usa SSE para actualizaciones en tiempo real

<details>
<summary>Ejemplo</summary>

En lugar de sondeo agresivo, suscríbete a eventos SSE:

```javascript
// Combine SSE with conservative polling
const eventSource = new EventSource('/api/events');

eventSource.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'job_status_changed') {
        loadJobs();  // Refresh on status change
    }
});

// Fallback: poll every 30 seconds
setInterval(() => loadJobs(true), 30000);
```

</details>

### 4. Maneja estados terminales

<details>
<summary>Ejemplo</summary>

Deja de sondear trabajos que alcanzaron estados terminales:

```javascript
const terminalStates = ['completed', 'failed', 'cancelled'];

function shouldPollJob(job) {
    return !terminalStates.includes(job.status);
}
```

</details>

### 5. Configura webhooks para producción

<details>
<summary>Ejemplo</summary>

Los webhooks eliminan la necesidad de sondeo por completo:

```yaml
# In provider config
webhook_url: "https://your-server.com/api/webhooks/replicate"
```

Cuando los webhooks están configurados:
- El polling en background se deshabilita (a menos que se habilite explícitamente)
- Las actualizaciones de estado llegan en tiempo real vía callbacks del proveedor
- Se reducen las llamadas a la API del proveedor

</details>

## Solución de problemas

### Los trabajos no se actualizan

<details>
<summary>Pasos de depuración</summary>

1. Verifica si el poller en background está corriendo:
   ```bash
   # Look for log line on startup
   grep "job status polling" server.log
   ```

2. Verifica conectividad del proveedor:
   ```bash
   curl http://localhost:8001/api/cloud/providers/replicate/validate
   ```

3. Fuerza una sync:
   ```bash
   curl -X POST http://localhost:8001/api/cloud/jobs/sync
   ```

</details>

### No se reciben eventos SSE

<details>
<summary>Pasos de depuración</summary>

1. Verifica el límite de conexiones SSE (5 por IP por defecto)
2. Verifica que EventSource esté conectado:
   ```javascript
   eventSource.addEventListener('open', () => {
       console.log('SSE connected');
   });
   ```

</details>

### Uso alto de la API del proveedor

<details>
<summary>Soluciones</summary>

Si estás golpeando límites de rate:
1. Aumenta `job_polling_interval` en la config enterprise
2. Reduce la frecuencia de polling de progreso inline
3. Configura webhooks para eliminar el polling

</details>
