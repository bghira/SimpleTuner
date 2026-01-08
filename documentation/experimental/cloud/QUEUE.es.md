# Sistema de colas

> **Estado:** Experimental
> **Disponible en:** Web UI (pestaña Cloud)

El sistema de colas gestiona la planificación de trabajos, los límites de concurrencia y la asignación de recursos equitativa para trabajos de entrenamiento en la nube. Siempre está activo, incluso en modo de usuario único, habilitando funciones como programación nocturna de trabajos y uso controlado de recursos.

## Resumen

Cuando envías un trabajo de entrenamiento en la nube, se agrega a la cola y se procesa en función de:

- **Prioridad** - Los trabajos con mayor prioridad se ejecutan primero
- **Límites de concurrencia** - Los límites globales y por usuario evitan el agotamiento de recursos
- **FIFO dentro de la prioridad** - Los trabajos con la misma prioridad se ejecutan en orden de envío

## Estado de la cola

Accede al panel de cola haciendo clic en el **icono de cola** en la barra de acciones de la pestaña Cloud. El panel muestra:

| Métrica | Descripción |
|---------|-------------|
| **En cola** | Trabajos esperando para ejecutarse |
| **En ejecución** | Trabajos que se están ejecutando |
| **Máximo concurrente** | Límite global de trabajos simultáneos |
| **Espera promedio** | Tiempo promedio que los trabajos pasan en cola |

## Niveles de prioridad

Los trabajos reciben prioridad según el nivel de usuario:

| Nivel de usuario | Prioridad | Valor |
|-----------------|-----------|-------|
| Admin | Urgent | 30 |
| Lead | High | 20 |
| Researcher | Normal | 10 |
| Viewer | Low | 0 |

Valores más altos = mayor prioridad = se procesa primero.

### Anulación de prioridad

Leads y admins pueden anular la prioridad de un trabajo en situaciones específicas (p. ej., experimentos urgentes).

## Límites de concurrencia

Dos límites controlan cuántos trabajos pueden ejecutarse simultáneamente:

### Límite global (`max_concurrent`)

Máximo de trabajos ejecutándose entre todos los usuarios. Predeterminado: **5 trabajos**.

### Límite por usuario (`user_max_concurrent`)

Máximo de trabajos que cualquier usuario puede ejecutar a la vez. Predeterminado: **2 trabajos**.

Esto evita que un usuario consuma todos los slots disponibles.

### Actualizar límites

Los admins pueden actualizar límites vía el panel de cola o la API:

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{"max_concurrent": 10, "user_max_concurrent": 3}'
```

## Ciclo de vida del trabajo en la cola

1. **Submitted** - Trabajo creado, agregado a la cola con estado `pending`
2. **Pending** - Esperando un slot (límite de concurrencia)
3. **Running** - Entrenando activamente en GPU en la nube
4. **Completed/Failed** - Estado terminal, eliminado de la cola activa

## Endpoints de API

### Listar entradas de cola

```http
GET /api/queue
```

Parámetros:
- `status` - Filtrar por estado (pending, running, blocked)
- `limit` - Máximo de entradas a devolver (predeterminado: 50)
- `include_completed` - Incluir trabajos finalizados

### Estadísticas de cola

```http
GET /api/queue/stats
```

Devuelve:
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

### Estado de mi cola

```http
GET /api/queue/me
```

Devuelve la posición en cola del usuario actual, trabajos pendientes y trabajos en ejecución.

**Respuesta:**

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
| `pending_count` | int | Número de trabajos esperando en la cola |
| `running_count` | int | Número de trabajos en ejecución |
| `blocked_count` | int | Número de trabajos en espera de aprobación |
| `best_position` | int or null | Posición del trabajo pendiente de mayor prioridad del usuario |
| `pending_jobs` | array | Lista de detalles de trabajos pendientes |
| `running_jobs` | array | Lista de detalles de trabajos en ejecución |

El campo `best_position` indica la posición en cola del mejor trabajo pendiente del usuario (mayor prioridad o enviado antes). Esto ayuda a los usuarios a entender cuándo iniciará su siguiente trabajo. Un valor de `null` significa que el usuario no tiene trabajos pendientes.

### Posición de trabajo

```http
GET /api/queue/position/{job_id}
```

Devuelve la posición en cola de un trabajo específico.

### Cancelar trabajo en cola

```http
POST /api/queue/{job_id}/cancel
```

Cancela un trabajo que aún no ha iniciado.

### Aprobar trabajo bloqueado

```http
POST /api/queue/{job_id}/approve
```

Solo admin. Aprueba un trabajo que requiere aprobación (p. ej., supera el umbral de costo).

### Rechazar trabajo bloqueado

```http
POST /api/queue/{job_id}/reject?reason=<reason>
```

Solo admin. Rechaza un trabajo bloqueado con un motivo.

### Actualizar concurrencia

```http
POST /api/queue/concurrency
```

Cuerpo:
```json
{
  "max_concurrent": 10,
  "user_max_concurrent": 3
}
```

### Disparar procesamiento

```http
POST /api/queue/process
```

Solo admin. Dispara manualmente el procesamiento de la cola (normalmente automático).

### Limpiar entradas antiguas

```http
POST /api/queue/cleanup?days=30
```

Solo admin. Elimina entradas completadas más antiguas que los días especificados.
