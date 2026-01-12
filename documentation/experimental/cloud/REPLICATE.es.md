# Integración con Replicate

Replicate es una plataforma en la nube para ejecutar modelos de ML. SimpleTuner usa el sistema de contenedores Cog de Replicate para ejecutar trabajos de entrenamiento en GPUs en la nube.

- **Modelo:** `simpletuner/advanced-trainer`
- **GPU predeterminada:** L40S (48GB VRAM)

## Inicio rápido

1. Crea una [cuenta de Replicate](https://replicate.com/signin) y obtén un [token de API](https://replicate.com/account/api-tokens)
2. Configura la variable de entorno:
   ```bash
   export REPLICATE_API_TOKEN="r8_your_token_here"
   simpletuner server
   ```
3. Abre la Web UI → pestaña Cloud → haz clic en **Validate** para verificar

## Flujo de datos

| Tipo de datos | Destino | Retención |
|--------------|---------|-----------|
| Imágenes de entrenamiento | Servidores de carga de Replicate (GCP) | Eliminadas después del trabajo |
| Configuración de entrenamiento | API de Replicate | Almacenada con metadatos del trabajo |
| Token de API | Solo tu entorno | Nunca almacenado por SimpleTuner |
| Modelo entrenado | HuggingFace Hub, S3 o local | Bajo tu control |
| Logs del trabajo | Servidores de Replicate | 30 días |

**Límite de carga:** La API de carga de archivos de Replicate acepta archivos de hasta 100 MiB. SimpleTuner bloquea los envíos cuando el archivo empaquetado supera este límite.

<details>
<summary>Detalles de la ruta de datos</summary>

1. **Carga:** Imágenes locales → HTTPS POST → `api.replicate.com`
2. **Entrenamiento:** Replicate descarga los datos a una instancia GPU efímera
3. **Salida:** Modelo entrenado → Tu destino configurado
4. **Limpieza:** Replicate elimina los datos de entrenamiento tras completar el trabajo

Consulta la [documentación de seguridad de Replicate](https://replicate.com/docs/reference/security) para más detalles.

</details>

## Hardware y costos {#costs}

| Hardware | VRAM | Costo | Mejor para |
|----------|------|-------|------------|
| L40S | 48GB | ~$3.50/hr | La mayoría del entrenamiento LoRA |
| A100 (80GB) | 80GB | ~$5.00/hr | Modelos grandes, fine-tuning completo |

### Costos típicos de entrenamiento

| Tipo de entrenamiento | Pasos | Tiempo | Costo |
|-----------------------|-------|--------|-------|
| LoRA (Flux) | 1000 | 30-60 min | $2-4 |
| LoRA (Flux) | 2000 | 1-2 horas | $4-8 |
| LoRA (SDXL) | 2000 | 45-90 min | $3-6 |
| Fine-tune completo | 5000+ | 4-12 horas | $15-50 |

### Protección de costos

Configura límites de gasto en la pestaña Cloud → Settings:
- Habilita "Cost Limit" con monto/período (diario/semanal/mensual)
- Elige acción: **Warn** o **Block**

## Entrega de resultados

### Opción 1: HuggingFace Hub (Recomendado)

1. Configura la variable de entorno `HF_TOKEN`
2. Pestaña Publishing → habilita "Push to Hub"
3. Configura `hub_model_id` (p. ej., `username/my-lora`)

### Opción 2: Descarga local vía webhook

1. Inicia un túnel: `ngrok http 8080` o `cloudflared tunnel --url http://localhost:8080`
2. Pestaña Cloud → establece **Webhook URL** con la URL del túnel
3. Los modelos se descargan en `~/.simpletuner/cloud_outputs/`

### Opción 3: S3 externo

Configura la publicación en S3 en la pestaña Publishing (AWS S3, MinIO, Backblaze B2, etc.).

## Configuración de red {#network}

### Endpoints de API {#api-endpoints}

SimpleTuner se conecta a estos endpoints de Replicate:

| Destino | Propósito | Requerido |
|---------|-----------|-----------|
| `api.replicate.com` | Llamadas de API (envío de trabajos, estado) | Sí |
| `*.replicate.delivery` | Subidas/descargas de archivos | Sí |
| `www.replicatestatus.com` | API de página de estado | No (degrada de forma adecuada) |
| `api.replicate.com/v1/webhooks/default/secret` | Secreto de firma de webhook | Solo si la validación de firma está habilitada |

### IPs de origen de webhooks {#webhook-ips}

Los webhooks de Replicate provienen de la región `us-west1` de Google Cloud:

| Rango de IP | Notas |
|-------------|-------|
| `34.82.0.0/16` | Fuente primaria de webhook |
| `35.185.0.0/16` | Rango secundario |

Para los rangos de IP más actuales:
- Revisa la [documentación de webhooks de Replicate](https://replicate.com/docs/webhooks)
- O usa los [rangos de IP publicados por Google](https://www.gstatic.com/ipranges/cloud.json) filtrados por `us-west1`

<details>
<summary>Ejemplo de configuración de allowlist de IP</summary>

```bash
curl -X PUT http://localhost:8080/api/cloud/providers/replicate \
  -H "Content-Type: application/json" \
  -d '{
    "webhook_allowed_ips": ["34.82.0.0/16", "35.185.0.0/16"]
  }'
```

</details>

### Reglas de firewall {#firewall}

**Saliente (SimpleTuner → Replicate):**

| Destino | Puerto | Propósito |
|---------|--------|-----------|
| `api.replicate.com` | 443 | Llamadas de API |
| `*.replicate.delivery` | 443 | Subidas/descargas de archivos |
| `replicate.com` | 443 | Metadatos de modelos |

<details>
<summary>Rangos de IP para reglas de egreso estrictas</summary>

Replicate funciona en Google Cloud. Para reglas de firewall estrictas:

```
34.82.0.0/16
34.83.0.0/16
35.185.0.0/16 - 35.247.0.0/16  (all /16 blocks in this range)
```

**Alternativa más simple:** Permitir egreso basado en DNS a `*.replicate.com` y `*.replicate.delivery`.

</details>

**Entrante (Replicate → Tu servidor):**

```
Allow TCP from 34.82.0.0/16, 35.185.0.0/16 to your webhook port
```

## Despliegue en producción

Endpoint de webhook: **`POST /api/webhooks/replicate`**

Configura tu URL pública (sin ruta) en la pestaña Cloud. SimpleTuner agrega la ruta del webhook automáticamente.

<details>
<summary>Configuración de nginx</summary>

```nginx
upstream simpletuner {
    server 127.0.0.1:8080;
}

server {
    listen 443 ssl http2;
    server_name training.yourcompany.com;

    ssl_certificate     /etc/ssl/certs/training.crt;
    ssl_certificate_key /etc/ssl/private/training.key;

    location /api/webhooks/ {
        allow 34.82.0.0/16;
        allow 35.185.0.0/16;
        deny all;

        proxy_pass http://simpletuner;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;

        proxy_pass http://simpletuner;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

</details>

<details>
<summary>Configuración de Caddy</summary>

```caddyfile
training.yourcompany.com {
    @webhook path /api/webhooks/*
    handle @webhook {
        reverse_proxy localhost:8080
    }

    @internal remote_ip 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16
    handle @internal {
        reverse_proxy localhost:8080
    }

    respond "Forbidden" 403
}
```

</details>

<details>
<summary>Configuración de Traefik (Docker)</summary>

```yaml
services:
  simpletuner:
    image: simpletuner:latest
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.simpletuner.rule=Host(`training.yourcompany.com`)"
      - "traefik.http.routers.simpletuner.tls=true"
      - "traefik.http.services.simpletuner.loadbalancer.server.port=8080"
      - "traefik.http.middlewares.replicate-ips.ipwhitelist.sourcerange=34.82.0.0/16,35.185.0.0/16"
      - "traefik.http.routers.webhook.rule=Host(`training.yourcompany.com`) && PathPrefix(`/api/webhooks`)"
      - "traefik.http.routers.webhook.middlewares=replicate-ips"
      - "traefik.http.routers.webhook.tls=true"
```

</details>

## Eventos de webhook {#webhook-events}

| Evento | Descripción |
|--------|-------------|
| `start` | El trabajo comenzó a ejecutarse |
| `logs` | Salida de logs de entrenamiento |
| `output` | El trabajo produjo salida |
| `completed` | El trabajo finalizó exitosamente |
| `failed` | El trabajo falló con error |

## Solución de problemas {#troubleshooting}

**"REPLICATE_API_TOKEN not set"**
- Exporta la variable: `export REPLICATE_API_TOKEN="r8_..."`
- Reinicia SimpleTuner después de configurarla

**"Invalid token" o falla la validación**
- El token debe comenzar con `r8_`
- Genera un nuevo token desde el [panel de Replicate](https://replicate.com/account/api-tokens)
- Verifica que no haya espacios o saltos de línea extra

**Trabajo atascado en "queued"**
- Replicate pone en cola los trabajos cuando las GPUs están ocupadas
- Revisa la [página de estado de Replicate](https://replicate.statuspage.io/)

**El entrenamiento falla con OOM**
- Reduce el tamaño de batch
- Habilita gradient checkpointing
- Usa LoRA en lugar de fine-tuning completo

**Webhook no recibe eventos**
- Verifica que el túnel esté ejecutándose y accesible
- Comprueba que la URL del webhook incluya `https://`
- Prueba manualmente: `curl -X POST https://your-url/api/webhooks/replicate -d '{}'`

**Problemas de conexión a través del proxy**
```bash
# Test proxy connectivity to Replicate
curl -x http://proxy:8080 https://api.replicate.com/v1/account

# Check environment
env | grep -i proxy
```

## Referencia de API {#api-reference}

| Endpoint | Descripción |
|----------|-------------|
| `GET /api/cloud/providers/replicate/versions` | Listar versiones de modelo |
| `GET /api/cloud/providers/replicate/validate` | Validar credenciales |
| `GET /api/cloud/providers/replicate/billing` | Obtener saldo de crédito |
| `PUT /api/cloud/providers/replicate/token` | Guardar token de API |
| `DELETE /api/cloud/providers/replicate/token` | Eliminar token de API |
| `POST /api/cloud/jobs/submit` | Enviar trabajo de entrenamiento |
| `POST /api/webhooks/replicate` | Receptor de webhook |

## Enlaces

- [Documentación de Replicate](https://replicate.com/docs)
- [SimpleTuner en Replicate](https://replicate.com/simpletuner/advanced-trainer)
- [Tokens de API de Replicate](https://replicate.com/account/api-tokens)
- [Página de estado de Replicate](https://replicate.statuspage.io/)
