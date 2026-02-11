# Tutorial de entrenamiento por API

## Introducción

Esta guía recorre cómo ejecutar trabajos de entrenamiento de SimpleTuner **completamente a través de la API HTTP** mientras mantienes la configuración y la gestión de datasets en la línea de comandos. Replica la estructura de otros tutoriales pero omite la incorporación en la WebUI. Vas a:

- instalar e iniciar el servidor unificado
- descubrir y descargar el esquema OpenAPI
- crear y actualizar entornos con llamadas REST
- validar, lanzar y monitorear trabajos de entrenamiento vía `/api/training`
- bifurcar en dos configuraciones probadas: un fine-tune completo de PixArt Sigma 900M y una ejecución LoRA LyCORIS de Flux Kontext

## Requisitos previos

- Python 3.10–3.13, Git y `pip`
- SimpleTuner instalado en un entorno virtual (`pip install 'simpletuner[cuda]'` o la variante que coincida con tu plataforma)
  - CUDA 13 / Blackwell users (NVIDIA B-series GPUs): `pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130`
- Acceso a repos requeridos de Hugging Face (`huggingface-cli login` antes de descargar modelos con restricciones)
- Datasets preparados localmente con captions (archivos de texto de caption para PixArt, carpetas de edición/referencia emparejadas para Kontext)
- Un shell con `curl` y `jq`

## Inicia el servidor

Desde tu checkout de SimpleTuner (o el entorno donde el paquete está instalado):

```bash
simpletuner server --port 8001
```

La API vive en `http://localhost:8001`. Deja el servidor en ejecución mientras emites los siguientes comandos en otra terminal.

> **Consejo:** Si ya tienes un entorno de configuración listo para entrenar, puedes iniciar el servidor con `--env` para comenzar el entrenamiento automáticamente una vez que el servidor esté completamente cargado:
>
> ```bash
> simpletuner server --port 8001 --env my-training-config
> ```
>
> Esto valida tu configuración al inicio y lanza el entrenamiento inmediatamente después de que el servidor esté listo—útil para despliegues desatendidos o automatizados. La opción `--env` funciona de manera idéntica a `simpletuner train --env`.

### Configuración y despliegue

Para uso en producción, puedes configurar la dirección de enlace y el puerto:

| Opción | Variable de entorno | Predeterminado | Descripción |
|--------|---------------------|---------------|-------------|
| `--host` | `SIMPLETUNER_HOST` | `0.0.0.0` | Dirección a la que se enlaza el servidor (usa `127.0.0.1` detrás de un proxy inverso) |
| `--port` | `SIMPLETUNER_PORT` | `8001` | Puerto al que se enlaza el servidor |

<details>
<summary><b>Opciones de despliegue en producción (TLS, Proxy inverso, Systemd, Docker)</b></summary>

Para despliegues en producción, se recomienda usar un proxy inverso para la terminación TLS.

#### Configuración de Nginx

```nginx
server {
    listen 443 ssl http2;
    server_name training.example.com;

    # TLS configuration (example using Let's Encrypt paths)
    ssl_certificate /etc/letsencrypt/live/training.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/training.example.com/privkey.pem;

    # WebSocket support for SSE streaming (Critical for real-time logs)
    location /api/training/stream {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        # SSE-specific settings
        proxy_buffering off;
        proxy_read_timeout 86400s;
    }

    # Main application
    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        # Large file uploads for datasets
        client_max_body_size 10G;
        proxy_request_buffering off;
    }
}
```

#### Configuración de Caddy

```caddyfile
training.example.com {
    reverse_proxy 127.0.0.1:8001 {
        # SSE streaming support
        flush_interval -1
    }
    # Large file uploads
    request_body {
        max_size 10GB
    }
}
```

#### Servicio systemd

```ini
[Unit]
Description=SimpleTuner Training Server
After=network.target

[Service]
Type=simple
User=trainer
WorkingDirectory=/home/trainer/simpletuner-workspace
Environment="SIMPLETUNER_HOST=127.0.0.1"
Environment="SIMPLETUNER_PORT=8001"
ExecStart=/home/trainer/simpletuner-workspace/.venv/bin/simpletuner server
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

#### Docker Compose con Traefik

```yaml
version: '3.8'
services:
  simpletuner:
    image: ghcr.io/bghira/simpletuner:latest
    command: simpletuner server --host 0.0.0.0 --port 8001
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.simpletuner.rule=Host(`training.example.com`)"
      - "traefik.http.services.simpletuner.loadbalancer.server.port=8001"
```
</details>

## Autenticación

SimpleTuner soporta autenticación multiusuario. En el primer arranque, necesitarás crear una cuenta de admin.

### Configuración inicial

Comprueba si se necesita configuración:

```bash
curl -s http://localhost:8001/api/cloud/auth/setup/status | jq
```

Si `needs_setup` es `true`, crea el primer admin:

```bash
curl -s -X POST http://localhost:8001/api/cloud/auth/setup/first-admin \
  -H 'Content-Type: application/json' \
  -d '{
    "email": "admin@example.com",
    "username": "admin",
    "password": "your-secure-password"
  }'
```

### Claves de API

Para acceso automatizado, genera una clave de API después de iniciar sesión:

```bash
# Login first (stores session cookie)
curl -s -X POST http://localhost:8001/api/cloud/auth/login \
  -H 'Content-Type: application/json' \
  -c cookies.txt \
  -d '{"username": "admin", "password": "your-secure-password"}'

# Create an API key
curl -s -X POST http://localhost:8001/api/cloud/auth/api-keys \
  -H 'Content-Type: application/json' \
  -b cookies.txt \
  -d '{"name": "automation-key"}' | jq
```

Usa la clave devuelta (prefijada con `st_`) en solicitudes posteriores:

```bash
curl -s http://localhost:8001/api/training/status \
  -H 'X-API-Key: st_your_key_here'
```

### Gestión de usuarios

Los admins pueden crear usuarios adicionales mediante la API o la página **Manage Users** de la WebUI:

```bash
# Create a new user (requires admin session)
curl -s -X POST http://localhost:8001/api/users \
  -H 'Content-Type: application/json' \
  -b cookies.txt \
  -d '{
    "email": "researcher@example.com",
    "username": "researcher",
    "password": "their-password",
    "level_names": ["researcher"]
  }'
```

> **Nota:** El registro público está deshabilitado por defecto. Los admins pueden habilitarlo en la pestaña **Manage Users → Registration**, pero se recomienda mantenerlo deshabilitado para despliegues privados.

## Descubre la API

FastAPI sirve documentación interactiva y el esquema OpenAPI:

```bash
# FastAPI Swagger UI
python -m webbrowser http://localhost:8001/docs

# ReDoc view
python -m webbrowser http://localhost:8001/redoc

# Download the schema for local inspection
curl -o openapi.json http://localhost:8001/openapi.json
jq '.info' openapi.json
```

Cada endpoint usado en este tutorial está documentado ahí bajo las etiquetas `configurations` y `training`.

## Ruta rápida: ejecutar sin entornos

Si prefieres **omitir por completo la gestión de config/entornos**, puedes lanzar una ejecución de entrenamiento puntual enviando el payload completo estilo CLI directamente a los endpoints de entrenamiento:

1. Crea o reutiliza un JSON de dataloader que describa tu dataset. El trainer solo necesita la ruta referenciada por `--data_backend_config`.

   ```bash
   cat <<'JSON' > config/multidatabackend-once.json
   [
     {
       "id": "demo-images",
       "type": "local",
       "dataset_type": "image",
       "instance_data_dir": "/data/datasets/demo",
       "caption_strategy": "textfile",
       "resolution": 1024,
       "resolution_type": "pixel_area"
     },
     {
       "id": "demo-text-embeds",
       "type": "local",
       "dataset_type": "text_embeds",
       "default": true,
       "cache_dir": "/data/cache/text/demo"
     }
   ]
   JSON
   ```

2. Valida la configuración inline. Proporciona cada argumento CLI requerido (`--model_family`, `--model_type`, `--pretrained_model_name_or_path`, `--output_dir`, `--data_backend_config` y `--num_train_epochs` o `--max_train_steps`):

   ```bash
   curl -s -X POST http://localhost:8001/api/training/validate \
     -F __active_tab__=model \
     -F --model_family=pixart_sigma \
     -F --model_type=full \
     -F --model_flavour=900M-1024-v0.6 \
     -F --pretrained_model_name_or_path=terminusresearch/pixart-900m-1024-ft-v0.6 \
     -F --output_dir=/workspace/output/inline-demo \
     -F --data_backend_config=config/multidatabackend-once.json \
     -F --train_batch_size=1 \
     -F --learning_rate=0.0001 \
     -F --max_train_steps=200 \
     -F --num_train_epochs=0
   ```

   Un fragmento verde de “Configuration Valid” confirma que el trainer aceptará el payload.

3. Lanza el entrenamiento con los **mismos** campos de formulario (puedes añadir overrides como `--seed` o `--validation_prompt`):

   ```bash
   curl -s -X POST http://localhost:8001/api/training/start \
     -F __active_tab__=model \
     -F --model_family=pixart_sigma \
     -F --model_type=full \
     -F --model_flavour=900M-1024-v0.6 \
     -F --pretrained_model_name_or_path=terminusresearch/pixart-900m-1024-ft-v0.6 \
     -F --output_dir=/workspace/output/inline-demo \
     -F --data_backend_config=config/multidatabackend-once.json \
     -F --train_batch_size=1 \
     -F --learning_rate=0.0001 \
     -F --max_train_steps=200 \
     -F --num_train_epochs=0 \
     -F --validation_prompt='test shot of <token>'
   ```

El servidor fusiona automáticamente los ajustes enviados con sus valores predeterminados, escribe la configuración resuelta en el archivo activo e inicia el entrenamiento. Puedes reutilizar el mismo enfoque para cualquier familia de modelos—las secciones restantes cubren un flujo completo cuando quieres entornos reutilizables.

### Monitoreo de ejecuciones ad-hoc

Puedes seguir el progreso a través de los mismos endpoints de estado usados más adelante en la guía:

- Consulta `GET /api/training/status` para estado general, ID de trabajo activo e información de etapa de arranque.
- Obtén logs incrementales con `GET /api/training/events?since_index=N` o transmítelos via WebSocket en `/api/training/events/stream`.

Para actualizaciones tipo push, suministra ajustes de webhook junto con tus campos de formulario:

```bash
curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=model \
  -F --model_family=pixart_sigma \
  ... \
  -F --webhook_config='[{"webhook_type":"raw","callback_url":"https://example.com/simpletuner","log_level":"info","ssl_no_verify":false}]' \
  -F --webhook_reporting_interval=10
```

El payload debe serializarse como JSON en una cadena; el servidor publica actualizaciones del ciclo de vida del trabajo en `callback_url`. Consulta la descripción de `--webhook_config` en `documentation/OPTIONS.md` o la plantilla de ejemplo `config/webhooks.json` para los campos compatibles.

<details>
<summary><b>Configuración de webhook para proxies inversos</b></summary>

Cuando uses un proxy inverso con HTTPS, tu URL de webhook debe ser el endpoint público:

1.  **Servidor público:** Usa `https://training.example.com/webhook/callback`
2.  **Túnel:** Usa ngrok o cloudflared para desarrollo local.

**Solución de problemas de logs en tiempo real (SSE):**
Si `GET /api/training/events` funciona pero el stream se queda colgado:
*   **Nginx:** Asegúrate de `proxy_buffering off;` y que `proxy_read_timeout` sea alto (p. ej., 86400s).
*   **CloudFlare:** Termina conexiones de larga duración; usa CloudFlare Tunnel o evita el proxy para el endpoint de streaming.
</details>

### Disparar validación manual

Si quieres forzar una pasada de evaluación **entre** intervalos de validación programados, llama al nuevo endpoint:

```bash
curl -s -X POST http://localhost:8001/api/training/validation/run
```

- El servidor responde con el `job_id` activo.
- El trainer encola una ejecución de validación que se dispara inmediatamente después de la siguiente sincronización de gradientes (no interrumpe el micro-batch actual).
- La ejecución reutiliza tus prompts/ajustes de validación configurados, por lo que las imágenes resultantes aparecen en los streams de eventos/logs habituales.
- Para descargar la validación a un ejecutable externo en lugar de la canalización integrada, establece `--validation_method=external-script` en tu configuración (o payload) y apunta `--validation_external_script` a tu script. Puedes pasar contexto de entrenamiento al script con marcadores: `{local_checkpoint_path}`, `{global_step}`, `{tracker_run_name}`, `{tracker_project_name}`, `{model_family}`, `{huggingface_path}`, `{remote_checkpoint_path}` (vacío para validación), además de cualquier valor de configuración `validation_*` (p. ej., `validation_num_inference_steps`, `validation_guidance`, `validation_noise_scheduler`). Habilita `--validation_external_background` si quieres que el script sea fire-and-forget sin bloquear el entrenamiento.
- ¿Quieres disparar automatización inmediatamente después de que cada checkpoint se escriba localmente (incluso mientras las subidas se ejecutan en segundo plano)? Configura `--post_checkpoint_script='/opt/hooks/run_eval.sh {local_checkpoint_path} {global_step}'`. Usa los mismos marcadores que los hooks de validación; `{remote_checkpoint_path}` se resuelve a vacío para este hook.
- ¿Prefieres mantener las subidas integradas de SimpleTuner y pasar la URL remota resultante a tu propia herramienta? Configura `--post_upload_script` en su lugar; se dispara una vez por proveedor de publicación/subida a Hugging Face Hub con `{remote_checkpoint_path}` (si lo proporciona el backend) y los mismos marcadores de contexto. SimpleTuner no ingiere resultados de tu script, así que registra artefactos/métricas en tu tracker por tu cuenta.
  - Ejemplo: `--post_upload_script='/opt/hooks/notify.sh {remote_checkpoint_path} {tracker_project_name} {tracker_run_name}'` donde `notify.sh` llama a tu API de tracker.
  - Ejemplos funcionales:
    - `simpletuner/examples/external-validation/replicate_post_upload.py` dispara una inferencia de Replicate usando `{remote_checkpoint_path}`, `{model_family}`, `{model_type}`, `{lora_type}` y `{huggingface_path}`.
    - `simpletuner/examples/external-validation/wavespeed_post_upload.py` dispara una inferencia de WaveSpeed y hace polling de finalización usando los mismos marcadores.
    - `simpletuner/examples/external-validation/fal_post_upload.py` dispara una inferencia Flux LoRA en fal.ai (requiere `FAL_KEY` y `model_family` que contenga `flux`).
    - `simpletuner/examples/external-validation/use_second_gpu.py` ejecuta inferencia Flux LoRA en otra GPU sin requerir subidas.

Si no hay un trabajo activo, el endpoint devuelve HTTP 400, así que consulta `/api/training/status` primero al automatizar reintentos.

### Disparar checkpoint manual

Para persistir el estado actual del modelo inmediatamente (sin esperar al siguiente checkpoint programado), llama:

```bash
curl -s -X POST http://localhost:8001/api/training/checkpoint/run
```

- El servidor responde con el `job_id` activo.
- El trainer guarda un checkpoint después de la siguiente sincronización de gradientes usando los mismos ajustes que los checkpoints programados (reglas de subida, retención rolling, etc.).
- La limpieza rolling y las notificaciones de webhook se comportan exactamente igual que en un checkpoint programado.

Como con la validación, el endpoint devuelve HTTP 400 si no hay un trabajo de entrenamiento en ejecución.

### Transmitir vistas previas de validación

Los modelos que exponen hooks de Tiny AutoEncoder (o equivalentes) pueden emitir **vistas previas de validación por paso** mientras una imagen/video aún se está muestreando. Habilita la función añadiendo los flags CLI a tu payload:

```bash
curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=validation \
  -F --validation_preview=true \
  -F --validation_preview_steps=4 \
  -F --validation_num_inference_steps=20 \
  …other fields…
```

- `--validation_preview` (por defecto `false`) desbloquea el decodificador de vistas previas.
- `--validation_preview_steps` determina con qué frecuencia emitir fotogramas intermedios. Con el ejemplo anterior, recibes eventos en los pasos 1,5,9,13,17,20 (siempre se emite el primer paso, luego cada 4 pasos).

Cada vista previa se publica como un evento `validation.image` (ver `simpletuner/helpers/training/validation.py:899-929`). Puedes consumirlos vía webhooks raw, `GET /api/training/events` o el stream SSE en `/api/training/events/stream`. Un payload típico se ve así:

```json
{
  "type": "validation.image",
  "title": "Validation (step 5/20): night bench",
  "body": "night bench shot of <token>",
  "data": {
    "step": 5,
    "timestep": 563.0,
    "resolution": [1024, 1024],
    "validation_type": "intermediary",
    "prompt": "night bench shot of <token>",
    "step_label": "5/20"
  },
  "images": [
    {"src": "data:image/png;base64,...", "mime_type": "image/png"}
  ]
}
```

Los modelos con capacidad de video adjuntan un array `videos` en su lugar (URIs de datos GIF con `mime_type: image/gif`). Como estos eventos se transmiten casi en tiempo real, puedes mostrarlos directamente en dashboards o enviarlos a Slack/Discord mediante un backend de webhook raw.

## Flujo de trabajo común de la API

1. **Crear un entorno** – `POST /api/configs/environments`
2. **Rellenar el archivo del dataloader** – editar el `config/<env>/multidatabackend.json` generado
3. **Actualizar hiperparámetros de entrenamiento** – `PUT /api/configs/<env>`
4. **Activar el entorno** – `POST /api/configs/<env>/activate`
5. **Validar parámetros de entrenamiento** – `POST /api/training/validate`
6. **Lanzar entrenamiento** – `POST /api/training/start`
7. **Monitorear o detener el trabajo** – `/api/training/status`, `/api/training/events`, `/api/training/stop`, `/api/training/cancel`

Cada ejemplo a continuación sigue este flujo.

## Opcional: subir datasets vía la API (backends locales)

Si el dataset aún no está en la máquina donde corre SimpleTuner, puedes enviarlo por HTTP antes de configurar el dataloader. Los endpoints de subida respetan el `datasets_dir` configurado (durante el onboarding de la WebUI) y están pensados para sistemas de archivos locales:

1. **Crear una carpeta destino** bajo tu raíz de datasets:

   ```bash
   DATASETS_DIR=${DATASETS_DIR:-/workspace/simpletuner/datasets}
   curl -s -X POST http://localhost:8001/api/datasets/folders \
     -F parent_path="$DATASETS_DIR" \
     -F folder_name="pixart-upload"
   ```

2. **Subir archivos o un ZIP** (se aceptan imágenes y metadatos opcionales `.txt/.jsonl/.csv`):

   ```bash
   # Upload a zip (automatically extracted on the server)
   curl -s -X POST http://localhost:8001/api/datasets/upload/zip \
     -F target_path="$DATASETS_DIR/pixart-upload" \
     -F file=@/path/to/dataset.zip

   # Or upload individual files
   curl -s -X POST http://localhost:8001/api/datasets/upload \
     -F target_path="$DATASETS_DIR/pixart-upload" \
     -F files[]=@image001.png \
     -F files[]=@image001.txt
   ```

> **Solución de problemas de subidas:** Si las subidas grandes fallan con un error "Entity Too Large" al usar un proxy inverso, asegúrate de haber incrementado el límite de tamaño del body (p. ej., `client_max_body_size 10G;` en Nginx o `request_body { max_size 10GB }` en Caddy).

Después de que termine la subida, apunta tu entrada de `multidatabackend.json` a la nueva carpeta (por ejemplo, `"/data/datasets/pixart-upload"`).

## Ejemplo: PixArt Sigma 900M fine-tune completo

### 1. Crea el entorno vía REST

```bash
curl -s -X POST http://localhost:8001/api/configs/environments \
  -H 'Content-Type: application/json' \
  -d
```json
{
        "name": "pixart-api-demo",
        "model_family": "pixart_sigma",
        "model_flavour": "900M-1024-v0.6",
        "model_type": "full",
        "description": "PixArt 900M API-driven training"
      }
```

Esto crea `config/pixart-api-demo/` y un `multidatabackend.json` inicial.

### 2. Conecta el dataset

Edita el archivo de dataloader (reemplaza rutas con las ubicaciones reales de tu dataset/caché):

```bash
cat <<'JSON' > config/pixart-api-demo/multidatabackend.json
[
  {
    "id": "pixart-camera",
    "id": "pixart-camera",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "/data/datasets/pseudo-camera-10k",
    "caption_strategy": "filename",
    "resolution": 1.0,
    "resolution_type": "area",
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "cache_dir_vae": "/data/cache/vae/pixart/pseudo-camera-10k",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square",
    "metadata_backend": "discovery"
  },
  {
    "id": "pixart-text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "/data/cache/text/pixart/pseudo-camera-10k",
    "write_batch_size": 128
  }
]
JSON
```

### 3. Actualiza hiperparámetros a través de la API

Obtén la configuración actual, fusiona overrides y vuelve a subir el resultado:

```bash
curl -s http://localhost:8001/api/configs/pixart-api-demo \
  | jq '.config + {
      "--output_dir": "/workspace/output/pixart900m",
      "--train_batch_size": 2,
      "--gradient_accumulation_steps": 2,
      "--learning_rate": 0.0001,
      "--optimizer": "adamw_bf16",
      "--lr_scheduler": "cosine",
      "--lr_warmup_steps": 500,
      "--max_train_steps": 1800,
      "--num_train_epochs": 0,
      "--validation_prompt": "a studio portrait of <token> wearing a leather jacket",
      "--validation_guidance": 3.8,
      "--validation_resolution": "1024x1024",
      "--validation_num_inference_steps": 28,
      "--cache_dir_vae": "/data/cache/vae/pixart",
      "--seed": 1337,
      "--resume_from_checkpoint": "latest",
      "--base_model_precision": "bf16",
      "--dataloader_prefetch": true,
      "--report_to": "none",
      "--checkpoints_total_limit": 4,
      "--validation_seed": 12345,
      "--data_backend_config": "pixart-api-demo/multidatabackend.json"
    }' > /tmp/pixart-config.json

jq '{
      "name": "pixart-api-demo",
      "description": "PixArt 900M full tune (API)",
      "tags": ["pixart", "api"],
      "config": .
    }' /tmp/pixart-config.json > /tmp/pixart-update.json

curl -s -X PUT http://localhost:8001/api/configs/pixart-api-demo \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/pixart-update.json
```

### 4. Activa el entorno

```bash
curl -s -X POST http://localhost:8001/api/configs/pixart-api-demo/activate
```

### 5. Valida antes de lanzar

`validate` consume datos codificados como formulario. Como mínimo, asegura que uno de `num_train_epochs` o `max_train_steps` sea 0:

```bash
curl -s -X POST http://localhost:8001/api/training/validate \
  -F __active_tab__=model \
  -F --num_train_epochs=0
```

Un bloque de éxito (`Configuration Valid`) significa que el trainer acepta la configuración fusionada.

### 6. Inicia el entrenamiento

```bash
curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=model \
  -F --num_train_epochs=0
```

La respuesta incluye el ID del trabajo. El entrenamiento se ejecuta con los parámetros guardados en el paso 3.

### 7. Monitorea y detén

```bash
# Query coarse status
curl -s http://localhost:8001/api/training/status | jq

# Stream incremental log events
curl -s 'http://localhost:8001/api/training/events?since_index=0' | jq

# Cancel or stop
curl -s -X POST http://localhost:8001/api/training/stop
curl -s -X POST http://localhost:8001/api/training/cancel -F job_id=<JOB_ID>
```

Notas de PixArt:

- Mantén el dataset lo bastante grande para el `train_batch_size * gradient_accumulation_steps` elegido
- Configura `HF_ENDPOINT` si necesitas un mirror y autentícate antes de descargar `terminusresearch/pixart-900m-1024-ft-v0.6`
- Ajusta `--validation_guidance` entre 3.6 y 4.4 dependiendo de tus prompts

## Ejemplo: Flux Kontext LyCORIS LoRA

Kontext comparte la mayor parte de su pipeline con Flux Dev, pero necesita imágenes editadas/referencia emparejadas.

### 1. Aprovisiona el entorno

```bash
curl -s -X POST http://localhost:8001/api/configs/environments \
  -H 'Content-Type: application/json' \
  -d
```json
{
        "name": "kontext-api-demo",
        "model_family": "flux",
        "model_flavour": "kontext",
        "model_type": "lora",
        "lora_type": "lycoris",
        "description": "Flux Kontext LoRA via API"
      }
```

### 2. Describe el dataloader emparejado

Kontext necesita pares de edición/referencia más un caché de embeddings de texto:

```bash
cat <<'JSON' > config/kontext-api-demo/multidatabackend.json
[
  {
    "id": "kontext-edit",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "/data/datasets/kontext/edit",
    "conditioning_data": ["kontext-reference"],
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "caption_strategy": "textfile",
    "minimum_image_size": 768,
    "maximum_image_size": 1536,
    "target_downsample_size": 1024,
    "cache_dir_vae": "/data/cache/vae/kontext/edit",
    "crop": true,
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square"
  },
  {
    "id": "kontext-reference",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/data/datasets/kontext/reference",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "/data/cache/vae/kontext/reference"
  },
  {
    "id": "kontext-text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "/data/cache/text/kontext"
  }
]
JSON
```

Asegúrate de que los nombres de archivo coincidan entre las carpetas de edición y referencia; SimpleTuner empareja los embeddings según el nombre.

### 3. Aplica hiperparámetros específicos de Kontext

```bash
curl -s http://localhost:8001/api/configs/kontext-api-demo \
  | jq '.config + {
      "--output_dir": "/workspace/output/kontext",
      "--train_batch_size": 1,
      "--gradient_accumulation_steps": 4,
      "--learning_rate": 0.00001,
      "--optimizer": "optimi-lion",
      "--lr_scheduler": "cosine",
      "--lr_warmup_steps": 200,
      "--max_train_steps": 12000,
      "--num_train_epochs": 0,
      "--validation_prompt": "a cinematic 1024px product photo of <token>",
      "--validation_guidance": 2.5,
      "--validation_resolution": "1024x1024",
      "--validation_num_inference_steps": 30,
      "--cache_dir_vae": "/data/cache/vae/kontext",
      "--seed": 777,
      "--resume_from_checkpoint": "latest",
      "--base_model_precision": "int8-quanto",
      "--dataloader_prefetch": true,
      "--report_to": "wandb",
      "--lora_rank": 16,
      "--lora_dropout": 0.05,
      "--conditioning_multidataset_sampling": "combined",
      "--clip_skip": 2,
      "--data_backend_config": "kontext-api-demo/multidatabackend.json"
    }' > /tmp/kontext-config.json

jq '{
      "name": "kontext-api-demo",
      "description": "Flux Kontext LyCORIS (API)",
      "tags": ["flux", "kontext", "api"],
      "config": .
    }' /tmp/kontext-config.json > /tmp/kontext-update.json

curl -s -X PUT http://localhost:8001/api/configs/kontext-api-demo \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/kontext-update.json
```

### 4. Activa, valida y lanza

```bash
curl -s -X POST http://localhost:8001/api/configs/kontext-api-demo/activate

curl -s -X POST http://localhost:8001/api/training/validate \
  -F __active_tab__=model \
  -F --num_train_epochs=0

curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=model \
  -F --num_train_epochs=0
```

Consejos de Kontext:

- `conditioning_type=reference_strict` mantiene los crops alineados; cambia a `reference_loose` si tus datasets difieren en relación de aspecto
- Cuantiza a `int8-quanto` para mantenerte dentro de 24 GB de VRAM a 1024 px; la precisión completa requiere GPUs de clase Hopper/Blackwell
- Para ejecuciones multinodo, configura `--accelerate_config` o `CUDA_VISIBLE_DEVICES` antes de iniciar el servidor

## Enviar trabajos locales con cola consciente de GPU

Al ejecutar en una máquina multi-GPU, puedes enviar trabajos de entrenamiento locales a través de la API de cola con asignación de GPU. Los trabajos se encolan si las GPUs requeridas no están disponibles.

### Comprobar disponibilidad de GPU

```bash
curl -s "http://localhost:8001/api/system/status?include_allocation=true" | jq '.gpu_allocation'
```

La respuesta muestra qué GPUs están disponibles:

```json
{
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
```

También puedes obtener estadísticas de cola incluyendo info local de GPU:

```bash
curl -s http://localhost:8001/api/queue/stats | jq '.local'
```

### Enviar un trabajo local

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "no_wait": false,
    "any_gpu": false
  }'
```

Opciones:

| Opción | Predeterminado | Descripción |
|--------|---------------|-------------|
| `config_name` | required | Nombre del entorno de entrenamiento a ejecutar |
| `no_wait` | false | Si es true, rechaza inmediatamente cuando no haya GPUs disponibles |
| `any_gpu` | false | Si es true, usa cualquier GPU disponible en lugar de los IDs de dispositivo configurados |

Respuesta:

```json
{
  "success": true,
  "job_id": "abc123",
  "status": "running",
  "allocated_gpus": [0, 1],
  "queue_position": null
}
```

El campo `status` indica el resultado:

- `running` - El trabajo se inició de inmediato con las GPUs asignadas
- `queued` - El trabajo quedó en cola y comenzará cuando haya GPUs disponibles
- `rejected` - GPUs no disponibles y `no_wait` era true

### Configurar límites de concurrencia locales

### Configurar límites de concurrencia locales

Los admins pueden limitar cuántos trabajos locales y GPUs pueden usarse mediante el endpoint de concurrencia de la cola:

```bash
# Get current limits
curl -s http://localhost:8001/api/queue/stats | jq '{local_gpu_max_concurrent, local_job_max_concurrent}'

# Update limits (alongside cloud limits)
curl -s -X POST http://localhost:8001/api/queue/concurrency \
  -H 'Content-Type: application/json' \
  -d '{
    "local_gpu_max_concurrent": 6,
    "local_job_max_concurrent": 2
  }'
```

Configura `local_gpu_max_concurrent` en `null` para uso ilimitado de GPU.

### Alternativa CLI

La misma funcionalidad está disponible vía CLI:

```bash
# Submit with default queuing behavior
simpletuner jobs submit my-config

# Reject if GPUs unavailable
simpletuner jobs submit my-config --no-wait

# Use any available GPUs
simpletuner jobs submit my-config --any-gpu

# Preview what would happen (dry-run)
simpletuner jobs submit my-config --dry-run
```

## Despachar trabajos a workers remotos

Si tienes máquinas GPU remotas registradas como workers (ver [Worker Orchestration](../experimental/server/WORKERS.md)), puedes despachar trabajos a ellas mediante la API de cola.

### Comprobar workers disponibles

```bash
curl -s http://localhost:8001/api/admin/workers | jq '.workers[] | {name, status, gpu_name, gpu_count}'
```

### Enviar a un destino específico

```bash
# Prefer remote workers, fall back to local GPUs (default)
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "auto"
  }'

# Force dispatch to remote workers only
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "worker"
  }'

# Run only on orchestrator's local GPUs
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "local"
  }'
```

### Seleccionar workers por etiqueta

Los workers pueden tener etiquetas para filtrar (p. ej., tipo de GPU, ubicación, equipo):

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "worker",
    "worker_labels": {"gpu_type": "a100*", "team": "nlp"}
  }'
```

Las etiquetas soportan patrones glob (`*` coincide con cualquier caracter).

## Endpoints útiles de un vistazo

- `GET /api/configs/` – listar entornos (pasa `?config_type=model` para configs de entrenamiento)
- `GET /api/configs/examples` – enumerar plantillas incluidas
- `POST /api/configs/{name}/dataloader` – regenerar un archivo de dataloader si quieres defaults
- `GET /api/training/status` – estado de alto nivel, `job_id` activo e información de etapa de arranque
- `GET /api/training/events?since_index=N` – stream incremental de logs del trainer
- `POST /api/training/checkpoints` – listar checkpoints del directorio de salida del trabajo activo
- `GET /api/system/status?include_allocation=true` – métricas del sistema con info de asignación de GPU
- `GET /api/queue/stats` – estadísticas de cola incluyendo asignación local de GPU
- `POST /api/queue/submit` – enviar un trabajo local o de worker con cola consciente de GPU
- `POST /api/queue/concurrency` – actualizar límites de concurrencia en nube y locales
- `GET /api/admin/workers` – listar workers registrados y su estado

## Dónde continuar

- Explora definiciones de opciones específicas en `documentation/OPTIONS.md`
- Combina estas llamadas REST con `jq`/`yq` o un cliente Python para automatización
- Conecta WebSockets en `/api/training/events/stream` para dashboards en tiempo real
- Reutiliza las configs exportadas (`GET /api/configs/<env>/export`) para versionar configuraciones funcionales
- **Ejecuta entrenamiento en GPUs en la nube** vía Replicate—ver el [Tutorial de entrenamiento en la nube](../experimental/cloud/TUTORIAL.md)

Con estos patrones puedes automatizar por completo el entrenamiento de SimpleTuner sin tocar la WebUI, mientras sigues apoyándote en el proceso de configuración CLI probado en batalla.
