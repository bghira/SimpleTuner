# Tutorial de entrenamiento en la nube

Esta guía recorre la ejecución de trabajos de entrenamiento de SimpleTuner en infraestructura GPU en la nube. Cubre los flujos de trabajo de la Web UI y la API REST.

## Requisitos previos

- SimpleTuner instalado y servidor en ejecución (ver el [tutorial de API local](../../api/TUTORIAL.md#start-the-server))
- Datasets preparados localmente con captions (los mismos [requisitos de dataset](../../api/TUTORIAL.md#optional-upload-datasets-over-the-api-local-backends) que el entrenamiento local)
- Una cuenta de proveedor en la nube (ver [Proveedores compatibles](#provider-setup))
- Para uso de API: un shell con `curl` y `jq`

## Configuración del proveedor {#provider-setup}

El entrenamiento en la nube requiere credenciales del proveedor elegido. Sigue la guía de configuración de tu proveedor:

| Proveedor | Guía de configuración |
|-----------|------------------------|
| Replicate | [REPLICATE.md](REPLICATE.md#quick-start) |

Tras completar la configuración del proveedor, vuelve aquí para enviar trabajos.

## Inicio rápido

Con tu proveedor configurado:

1. Abre `http://localhost:8001` y ve a la pestaña **Cloud**
2. Verifica tus credenciales en **Settings** (icono de engranaje) → **Validate**
3. Configura tu entrenamiento en las pestañas Model/Training/Dataloader
4. Haz clic en **Train in Cloud**
5. Revisa el resumen de carga y haz clic en **Submit**

## Recepción de modelos entrenados

Cuando termina el entrenamiento, tu modelo necesita un destino. Configura uno antes de tu primer trabajo.

### Opción 1: HuggingFace Hub (Recomendado)

Publica directamente en tu cuenta de HuggingFace:

1. Obtén un [token de HuggingFace](https://huggingface.co/settings/tokens) con acceso de escritura
2. Configura la variable de entorno:
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```
3. En la pestaña **Publishing**, habilita "Push to Hub" y configura el nombre del repo

### Opción 2: Descarga local vía webhook

Haz que los modelos se suban de vuelta a tu máquina. Requiere exponer tu servidor a internet.

1. Inicia un túnel:
   ```bash
   ngrok http 8001   # or: cloudflared tunnel --url http://localhost:8001
   ```
2. Copia la URL pública (p. ej., `https://abc123.ngrok.io`)
3. En la pestaña Cloud → Settings → Webhook URL, pega la URL
4. Los modelos se guardan en `~/.simpletuner/cloud_outputs/`

### Opción 3: S3 externo

Sube a cualquier endpoint compatible con S3 (AWS S3, MinIO, Backblaze B2, Cloudflare R2):

1. En la pestaña **Publishing**, configura los ajustes de S3
2. Proporciona endpoint, bucket, access key, secret key

## Flujo de la Web UI

### Enviar trabajos

1. **Configura tu entrenamiento** en las pestañas Model/Training/Dataloader
2. **Ve a la pestaña Cloud** y selecciona tu proveedor
3. **Haz clic en Train in Cloud** para abrir el diálogo previo al envío
4. **Revisa el resumen de carga**—los datasets locales se empaquetarán y cargarán
5. **Opcionalmente define un nombre de ejecución** para seguimiento
6. **Haz clic en Submit**

### Monitorear trabajos

La lista de trabajos muestra todos los trabajos de nube y locales con:

- **Indicador de estado**: Queued → Running → Completed/Failed
- **Progreso en vivo**: Paso de entrenamiento, valores de loss (cuando están disponibles)
- **Seguimiento de costos**: Costo estimado basado en tiempo de GPU

Haz clic en un trabajo para ver detalles:
- Snapshot de configuración del trabajo
- Logs en tiempo real (haz clic en **View Logs**)
- Acciones: Cancel, Delete (tras completarse)

### Panel de ajustes

Haz clic en el icono de engranaje para acceder a:

- **Validación de API key** y estado de cuenta
- **Webhook URL** para entrega local de modelos
- **Límites de costos** para evitar gastos excesivos
- **Información de hardware** (tipo de GPU, costo por hora)

## Flujo de trabajo de API

### Enviar un trabajo

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name_to_load": "my-training-config",
    "tracker_run_name": "api-test-run"
  }' | jq
```

Reemplaza `PROVIDER` con el nombre de tu proveedor (p. ej., `replicate`).

O envía con configuración inline:

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -d '{
    "config": {
      "--model_family": "flux",
      "--model_type": "lora",
      "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
      "--output_dir": "/outputs/flux-lora",
      "--max_train_steps": 1000,
      "--lora_rank": 16
    },
    "dataloader_config": [
      {
        "id": "training-images",
        "type": "local",
        "dataset_type": "image",
        "instance_data_dir": "/data/datasets/my-dataset",
        "caption_strategy": "textfile",
        "resolution": 1024
      }
    ]
  }' | jq
```

### Monitorear estado del trabajo

```bash
# Get job details
curl -s http://localhost:8001/api/cloud/jobs/JOB_ID | jq

# List all jobs
curl -s 'http://localhost:8001/api/cloud/jobs?limit=10' | jq

# Sync status of active jobs from provider
curl -s 'http://localhost:8001/api/cloud/jobs?sync_active=true' | jq
```

### Obtener logs del trabajo

```bash
curl -s http://localhost:8001/api/cloud/jobs/JOB_ID/logs | jq '.logs'
```

### Cancelar un trabajo en ejecución

```bash
curl -s -X POST http://localhost:8001/api/cloud/jobs/JOB_ID/cancel | jq
```

### Eliminar un trabajo completado

```bash
curl -s -X DELETE http://localhost:8001/api/cloud/jobs/JOB_ID | jq
```

## Integración CI/CD

### Envío idempotente de trabajos

Evita trabajos duplicados con claves de idempotencia:

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name_to_load": "my-config",
    "idempotency_key": "ci-build-12345"
  }' | jq
```

Si la misma clave se envía nuevamente dentro de 24 horas, recibirás el trabajo existente en lugar de crear uno duplicado.

### Ejemplo de GitHub Actions

```yaml
name: Cloud Training

on:
  push:
    branches: [main]
    paths:
      - 'training-configs/**'

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Submit Training Job
        env:
          SIMPLETUNER_URL: ${{ secrets.SIMPLETUNER_URL }}
        run: |
          RESPONSE=$(curl -s -X POST "$SIMPLETUNER_URL/api/cloud/jobs/submit?provider=replicate" \
            -H 'Content-Type: application/json' \
            -d '{
              "config_name_to_load": "production-lora",
              "idempotency_key": "gh-${{ github.sha }}",
              "tracker_run_name": "gh-run-${{ github.run_number }}"
            }')

          JOB_ID=$(echo $RESPONSE | jq -r '.job_id')
          echo "Submitted job: $JOB_ID"
          echo "JOB_ID=$JOB_ID" >> $GITHUB_ENV

      - name: Wait for Completion
        run: |
          while true; do
            STATUS=$(curl -s "$SIMPLETUNER_URL/api/cloud/jobs/$JOB_ID" | jq -r '.job.status')
            echo "Job status: $STATUS"

            case $STATUS in
              completed) exit 0 ;;
              failed|cancelled) exit 1 ;;
              *) sleep 60 ;;
            esac
          done
```

### Autenticación con API key

Para pipelines automatizados, crea API keys en lugar de autenticación por sesión.

**Vía UI:** pestaña Cloud → Settings → API Keys → Create New Key

**Vía API:**

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/auth/api-keys' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_SESSION_TOKEN' \
  -d '{
    "name": "ci-pipeline",
    "expires_days": 90,
    "scoped_permissions": ["job.submit", "job.view.own"]
  }'
```

La clave completa solo se devuelve una vez. Guárdala de forma segura.

**Uso de una API key:**

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer stk_abc123...' \
  -d '{...}'
```

**Permisos con alcance:**

| Permiso | Descripción |
|---------|-------------|
| `job.submit` | Enviar nuevos trabajos |
| `job.view.own` | Ver trabajos propios |
| `job.cancel.own` | Cancelar trabajos propios |
| `job.view.all` | Ver todos los trabajos (admin) |

## Solución de problemas

Para problemas específicos del proveedor (credenciales, cola, hardware), consulta la documentación de tu proveedor:

- [Solución de problemas de Replicate](REPLICATE.md#troubleshooting)

### Problemas generales

**Falla la carga de datos**
- Verifica que las rutas de dataset existan y sean legibles
- Revisa el espacio en disco disponible para el empaquetado ZIP
- Busca errores en la consola del navegador o en la respuesta de la API

**El webhook no recibe eventos**
- Asegura que tu instancia local sea accesible públicamente (túnel en ejecución)
- Verifica que la URL del webhook sea correcta (incluye https://)
- Revisa la salida de terminal de SimpleTuner para errores de manejo de webhooks

## Referencia de API

### Endpoints agnósticos al proveedor

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/cloud/jobs` | GET | Listar trabajos con filtros opcionales |
| `/api/cloud/jobs/submit` | POST | Enviar un nuevo trabajo de entrenamiento |
| `/api/cloud/jobs/sync` | POST | Sincronizar trabajos desde el proveedor |
| `/api/cloud/jobs/{id}` | GET | Obtener detalles del trabajo |
| `/api/cloud/jobs/{id}/logs` | GET | Obtener logs del trabajo |
| `/api/cloud/jobs/{id}/cancel` | POST | Cancelar un trabajo en ejecución |
| `/api/cloud/jobs/{id}` | DELETE | Eliminar un trabajo completado |
| `/api/metrics` | GET | Obtener métricas de trabajo y costo |
| `/api/cloud/metrics/cost-limit` | GET | Obtener estado actual del límite de costo |
| `/api/cloud/providers/{provider}` | PUT | Actualizar ajustes del proveedor |
| `/api/cloud/storage/{bucket}/{key}` | PUT | Endpoint de subida compatible con S3 |

Para endpoints específicos del proveedor, ver:
- [Referencia de API de Replicate](REPLICATE.md#api-reference)

Para detalles de esquema completo, consulta los docs OpenAPI en `http://localhost:8001/docs`.

## Ver también

- [README.md](README.md) – Descripción general de arquitectura y estado de proveedores
- [REPLICATE.md](REPLICATE.md) – Configuración y detalles del proveedor Replicate
- [ENTERPRISE.md](../server/ENTERPRISE.md) – SSO, aprobaciones y gobernanza
- [Tutorial completo de operaciones en la nube](OPERATIONS_TUTORIAL.md) – Despliegue en producción y monitoreo
- [Tutorial completo de API local](../../api/TUTORIAL.md) – Entrenamiento local completo vía API
- [Configuración del Dataloader](../../DATALOADER.md) – Referencia de configuración de datasets
