# Orquestación de workers

La orquestación de workers de SimpleTuner permite distribuir trabajos de entrenamiento entre múltiples máquinas con GPU. Los workers se registran con un orquestador central, reciben eventos de despacho en tiempo real y reportan estado de vuelta.

## Resumen

La arquitectura orquestador/worker permite:

- **Entrenamiento distribuido** - Ejecuta trabajos en cualquier máquina con GPUs, en cualquier lugar
- **Auto-descubrimiento** - Los workers se auto-registran con capacidades de GPU
- **Despacho en tiempo real** - Trabajos despachados vía SSE (Server-Sent Events)
- **Flota mixta** - Combina workers efímeros en la nube con máquinas on-prem persistentes
- **Tolerancia a fallos** - Trabajos huérfanos se reencolan automáticamente

## Tipos de worker

| Tipo | Ciclo de vida | Caso de uso |
|------|-----------|----------|
| **Ephemeral** | Se apaga tras completar el trabajo | Instancias spot en la nube (RunPod, Vast.ai) |
| **Persistent** | Permanece en línea entre trabajos | GPUs on-prem, instancias reservadas |

## Inicio rápido

### 1. Iniciar el orquestador

Ejecuta el servidor de SimpleTuner en tu máquina central:

```bash
simpletuner server --host 0.0.0.0 --port 8001
```

Para producción, habilita SSL:

```bash
simpletuner server --host 0.0.0.0 --port 8001 --ssl
```

### 2. Crear un token de worker

**Vía Web UI:** Administration → Workers → Create Worker

**Vía API:**

```bash
curl -s -X POST http://localhost:8001/api/admin/workers \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpu-worker-1",
    "worker_type": "persistent",
    "labels": {"location": "datacenter-a", "gpu_type": "a100"}
  }'
```

La respuesta incluye el token (solo se muestra una vez):

```json
{
  "worker_id": "w_abc123",
  "token": "wt_xxxxxxxxxxxx",
  "name": "gpu-worker-1"
}
```

### 3. Iniciar el worker

En la máquina con GPU:

```bash
simpletuner worker \
  --orchestrator-url https://orchestrator.example.com:8001 \
  --worker-token wt_xxxxxxxxxxxx \
  --name gpu-worker-1 \
  --persistent
```

O vía variables de entorno:

```bash
export SIMPLETUNER_ORCHESTRATOR_URL=https://orchestrator.example.com:8001
export SIMPLETUNER_WORKER_TOKEN=wt_xxxxxxxxxxxx
export SIMPLETUNER_WORKER_NAME=gpu-worker-1
export SIMPLETUNER_WORKER_PERSISTENT=true

simpletuner worker
```

El worker:

1. Se conecta al orquestador
2. Reporta capacidades de GPU (auto-detectadas)
3. Entra al loop de despacho de trabajos
4. Envía heartbeats cada 30 segundos

### 4. Enviar trabajos a workers

**Vía Web UI:** Configura tu entrenamiento, luego haz clic en **Train in Cloud** → selecciona **Worker** como destino.

**Vía API:**

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H "Content-Type: application/json" \
  -d '{
    "config_name": "my-training-config",
    "target": "worker"
  }'
```

Opciones de destino:

| Destino | Comportamiento |
|--------|----------|
| `worker` | Despacha solo a workers remotos |
| `local` | Ejecuta en GPUs del orquestador |
| `auto` | Prefiere worker si hay disponible, hace fallback a local |

## Referencia CLI

```
simpletuner worker [OPTIONS]

OPTIONS:
  --orchestrator-url URL   Orchestrator panel URL (or SIMPLETUNER_ORCHESTRATOR_URL)
  --worker-token TOKEN     Authentication token (or SIMPLETUNER_WORKER_TOKEN)
  --name NAME              Worker name (default: hostname)
  --persistent             Stay online between jobs (default: ephemeral)
  -v, --verbose            Enable debug logging
```

### Modo efímero vs persistente

**Ephemeral (default):**
- El worker se apaga después de completar un trabajo
- Ideal para instancias spot en la nube que facturan por minuto
- El orquestador limpia workers efímeros offline tras 1 hora

**Persistent (`--persistent`):**
- El worker permanece en línea esperando nuevos trabajos
- Reconexion automática si cae la conexión
- Úsalo para GPUs on-prem o instancias reservadas

## Ciclo de vida del worker

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  CONNECTING │ ──▶ │    IDLE     │ ──▶ │    BUSY     │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  DRAINING   │     │   OFFLINE   │
                    └─────────────┘     └─────────────┘
```

| Estado | Descripción |
|--------|-------------|
| `CONNECTING` | El worker establece conexión |
| `IDLE` | Listo para recibir trabajos |
| `BUSY` | Ejecutando un trabajo actualmente |
| `DRAINING` | Termina el trabajo actual y luego se apaga |
| `OFFLINE` | Desconectado (timeout de heartbeat) |

## Monitoreo de salud

El orquestador monitorea la salud de los workers:

- **Intervalo de heartbeat:** 30 segundos (worker → orquestador)
- **Umbral de timeout:** 120 segundos sin heartbeat → marcar offline
- **Loop de health check:** Corre cada 60 segundos en el orquestador

### Manejo de fallos

**Worker se va offline durante un trabajo:**

1. El trabajo se marca como fallido tras el timeout
2. Si quedan reintentos (default: 3), el trabajo se reencola
3. El siguiente worker disponible toma el trabajo

**Reinicio del orquestador:**

1. Los workers se reconectan automáticamente
2. Los workers reportan trabajos en progreso
3. El orquestador reconcilia estado y reanuda

## Matching de GPU

Los workers reportan capacidades de GPU al registrarse:

```json
{
  "gpu_count": 2,
  "gpu_name": "NVIDIA A100-SXM4-80GB",
  "gpu_vram_gb": 80,
  "accelerator_type": "cuda"
}
```

Los trabajos pueden especificar requisitos de GPU:

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H "Content-Type: application/json" \
  -d '{
    "config_name": "my-config",
    "target": "worker",
    "worker_labels": {"gpu_type": "a100*"}
  }'
```

El scheduler empareja trabajos con workers basado en:

1. Requisitos de cantidad de GPU
2. Match de etiquetas (patrones glob soportados)
3. Disponibilidad del worker (estado IDLE)

## Etiquetas

Las etiquetas proporcionan selección flexible de workers:

**Asignar etiquetas al crear el worker:**

```bash
curl -s -X POST http://localhost:8001/api/admin/workers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "worker-1",
    "labels": {
      "location": "us-west",
      "gpu_type": "a100",
      "team": "nlp"
    }
  }'
```

**Seleccionar workers por etiqueta:**

```bash
# Match workers with team=nlp
curl -s -X POST http://localhost:8001/api/queue/submit \
  -d '{"config_name": "my-config", "worker_labels": {"team": "nlp"}}'

# Match workers with gpu_type starting with "a100"
curl -s -X POST http://localhost:8001/api/queue/submit \
  -d '{"config_name": "my-config", "worker_labels": {"gpu_type": "a100*"}}'
```

## Operaciones de admin

### Listar workers

```bash
curl -s http://localhost:8001/api/admin/workers | jq
```

Respuesta:

```json
{
  "workers": [
    {
      "id": "w_abc123",
      "name": "gpu-worker-1",
      "status": "idle",
      "worker_type": "persistent",
      "gpu_count": 2,
      "gpu_name": "A100",
      "labels": {"location": "datacenter-a"},
      "last_heartbeat": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### Drenar un worker

Finaliza el trabajo actual de forma elegante y evita nuevos despachos:

```bash
curl -s -X POST http://localhost:8001/api/admin/workers/w_abc123/drain
```

El worker:

1. Completa cualquier trabajo en ejecución
2. Entra en estado DRAINING
3. Rechaza nuevos trabajos
4. Se desconecta al completar el trabajo (ephemeral) o permanece en estado draining (persistent)

### Rotar token

Regenera el token de autenticación de un worker:

```bash
curl -s -X POST http://localhost:8001/api/admin/workers/w_abc123/token
```

El token anterior se invalida de inmediato. Actualiza el worker con el nuevo token.

### Eliminar un worker

```bash
curl -s -X DELETE http://localhost:8001/api/admin/workers/w_abc123
```

Solo funciona si el worker está offline.

## Seguridad

### Autenticación por token

- Los workers se autentican vía el header `X-Worker-Token`
- Los tokens se hashean con SHA-256 antes de almacenarse
- Los tokens nunca salen del orquestador después de su creación
- Rota tokens periódicamente por seguridad

### Seguridad de red

Para producción:

1. Usa el flag `--ssl` o termina TLS en un reverse proxy
2. Restringe registro de workers a redes confiables
3. Usa reglas de firewall para limitar acceso a endpoints `/api/workers/*`

### Registro de auditoría

Todas las acciones de workers se registran:

- Intentos de registro (éxito/fallo)
- Eventos de despacho de trabajos
- Transiciones de estado
- Rotaciones de tokens
- Operaciones de admin

Consulta [Audit Guide](AUDIT.md) para acceso a logs.

## Troubleshooting

### Worker no puede conectar

**"Connection refused"**
- Verifica URL y puerto del orquestador
- Revisa reglas de firewall para permitir conexiones entrantes
- Asegúrate de que el orquestador corre con `--host 0.0.0.0`

**"Invalid token"**
- El token pudo haberse rotado—solicita uno nuevo
- Revisa espacios en blanco en la cadena del token

**"SSL certificate verify failed"**
- Usa `--ssl-no-verify` para certificados autofirmados (solo desarrollo)
- O agrega el certificado CA al trust store del sistema

### Worker se va offline inesperadamente

**Timeout de heartbeat (120s)**
- Revisa estabilidad de red entre worker y orquestador
- Busca agotamiento de recursos (CPU/memoria) en el worker
- Incrementa `SIMPLETUNER_HEARTBEAT_TIMEOUT` si estás en una red poco confiable

**Crash de proceso**
- Revisa logs del worker por excepciones de Python
- Verifica que los drivers de GPU funcionen (`nvidia-smi`)
- Asegura espacio en disco suficiente para entrenamiento

### Trabajos no se despachan a workers

**No hay workers idle**
- Revisa estado de workers en el panel admin
- Verifica que los workers estén conectados e IDLE
- Revisa mismatch de etiquetas entre trabajo y workers

**Requisitos de GPU no cumplidos**
- El trabajo requiere más GPUs que cualquier worker disponible
- Ajusta `--num_processes` en la configuración de entrenamiento

## Referencia de API

### Endpoints de worker (Worker → Orchestrator)

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/workers/register` | POST | Registrar y reportar capacidades |
| `/api/workers/stream` | GET | Stream SSE para despacho de trabajos |
| `/api/workers/heartbeat` | POST | Keepalive periódico |
| `/api/workers/job/{id}/status` | POST | Reportar progreso de trabajo |
| `/api/workers/disconnect` | POST | Notificación de shutdown elegante |

### Endpoints de admin (requiere permiso `admin.workers`)

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/admin/workers` | GET | Listar todos los workers |
| `/api/admin/workers` | POST | Crear token de worker |
| `/api/admin/workers/{id}` | DELETE | Eliminar worker |
| `/api/admin/workers/{id}/drain` | POST | Drenar worker |
| `/api/admin/workers/{id}/token` | POST | Rotar token |

## Ver también

- [Enterprise Guide](ENTERPRISE.md) - SSO, cuotas, flujos de aprobación
- [Job Queue](../../JOB_QUEUE.md) - Scheduling de cola y prioridades
- [Cloud Training](../cloud/README.md) - Integración con proveedores cloud
- [API Tutorial](../../api/TUTORIAL.md) - Entrenamiento local vía REST API
