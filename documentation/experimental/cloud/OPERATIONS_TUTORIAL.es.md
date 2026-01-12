# Guía de operaciones de entrenamiento en la nube

Este documento cubre el despliegue en producción y las operaciones de la función de entrenamiento en la nube de SimpleTuner, con foco en la integración completa con la infraestructura DevOps existente.

## Arquitectura de red

### Conexiones salientes

El servidor realiza conexiones HTTPS salientes a los proveedores en la nube configurados. Cada proveedor tiene sus propios endpoints y requisitos.

**Detalles de red por proveedor:**
- [Endpoints de API de Replicate](REPLICATE.md#api-endpoints)

### Conexiones entrantes

| Origen | Endpoint | Propósito |
|--------|----------|-----------|
| Infraestructura del proveedor de nube | `/api/webhooks/{provider}` | Actualizaciones del estado de trabajos |
| Trabajo de entrenamiento en la nube | `/api/cloud/storage/{bucket}/{key}` | Subir salidas de entrenamiento |
| Sistemas de monitoreo | `/api/cloud/health`, `/api/cloud/metrics/prometheus` | Salud y métricas |

### Reglas de firewall

Los requisitos de firewall dependen del/de los proveedor(es) configurado(s).

**Reglas de firewall por proveedor:**
- [Configuración de firewall de Replicate](REPLICATE.md#firewall)

### Allowlisting de IPs para webhooks

Para mayor seguridad, puedes restringir la entrega de webhooks a rangos de IP específicos. Cuando está configurado, los webhooks de IPs fuera de la allowlist se rechazan con una respuesta 403 Forbidden.

**Configuración vía API:**

<details>
<summary>Ejemplo de configuración por API</summary>

```bash
# Set allowed IPs for a provider's webhooks
curl -X PUT http://localhost:8080/api/cloud/providers/{provider} \
  -H "Content-Type: application/json" \
  -d '{
    "webhook_allowed_ips": ["10.0.0.0/8", "192.168.0.0/16"]
  }'
```
</details>

**Configuración vía Web UI:**

1. Navega a la pestaña Cloud → Advanced Configuration
2. En la sección "Webhook Security", agrega rangos de IP
3. Usa notación CIDR (p. ej., `10.0.0.0/8`) o IPs individuales (`1.2.3.4/32`)

**Formato de IP:**

| Formato | Ejemplo | Descripción |
|---------|---------|-------------|
| IP única | `1.2.3.4/32` | Coincidencia exacta de IP |
| Subred | `10.0.0.0/8` | Red clase A |
| Rango estrecho | `192.168.1.0/24` | 256 direcciones |

**IPs de webhook por proveedor:**
- [IPs de webhook de Replicate](REPLICATE.md#webhook-ips)

**Comportamiento:**

| Escenario | Resultado |
|-----------|-----------|
| Sin allowlist configurada | Se aceptan todas las IPs |
| Arreglo vacío `[]` | Se aceptan todas las IPs |
| IP en allowlist | Webhook procesado |
| IP fuera de la allowlist | 403 Forbidden |

**Registro de auditoría:**

Los webhooks rechazados se registran en el rastro de auditoría:

```bash
curl "http://localhost:8080/api/audit?event_type=webhook_rejected&limit=100"
```

## Configuración de proxy

### Variables de entorno

<details>
<summary>Variables de entorno de proxy</summary>

```bash
# HTTP/HTTPS proxy
export HTTPS_PROXY="http://proxy.corp.example.com:8080"
export HTTP_PROXY="http://proxy.corp.example.com:8080"

# Custom CA bundle for corporate CAs
export SIMPLETUNER_CA_BUNDLE="/etc/pki/tls/certs/ca-bundle.crt"

# Disable SSL verification (NOT recommended for production)
export SIMPLETUNER_SSL_VERIFY="false"

# HTTP timeout (seconds)
export SIMPLETUNER_HTTP_TIMEOUT="60"
```
</details>

### Vía configuración del proveedor

<details>
<summary>Configuración vía API</summary>

```python
# Via API
PUT /api/cloud/providers/{provider}
{
    "ssl_verify": true,
    "ssl_ca_bundle": "/etc/pki/tls/certs/corporate-ca.crt",
    "proxy_url": "http://proxy:8080",
    "http_timeout": 60.0
}
```
</details>

### Vía Web UI (Advanced Configuration)

La pestaña Cloud incluye un panel de Advanced Configuration para ajustes de red:

| Ajuste | Descripción |
|--------|-------------|
| **SSL Verification** | Alterna habilitar/deshabilitar la verificación de certificados |
| **CA Bundle Path** | Paquete de autoridad certificadora personalizado para CAs corporativas |
| **Proxy URL** | Proxy HTTP para conexiones salientes |
| **HTTP Timeout** | Tiempo de espera de solicitudes en segundos (predeterminado: 30) |

#### Omisión de verificación SSL

Deshabilitar la verificación SSL requiere confirmación explícita por implicaciones de seguridad:

1. Haz clic en el interruptor de SSL Verification para deshabilitar
2. Aparece un diálogo de confirmación: *"Disabling SSL verification is a security risk. Only do this if you have a self-signed certificate or are behind a corporate proxy. Continue?"*
3. Haz clic en "OK" para confirmar y guardar la configuración

El reconocimiento es por sesión. Cambios posteriores dentro de la misma sesión no requerirán nueva confirmación.

#### Configuración de proxy corporativo

Para entornos que usan proxies HTTP:

1. Navega a la pestaña Cloud → Advanced Configuration
2. Ingresa la URL del proxy (p. ej., `http://proxy.corp.example.com:8080`)
3. Opcionalmente establece un CA bundle personalizado si tu proxy hace inspección TLS
4. Ajusta el HTTP timeout si tu proxy agrega latencia

Los ajustes se guardan de inmediato y aplican a todas las llamadas posteriores a la API del proveedor.

## Monitoreo de salud

### Endpoints

| Endpoint | Propósito | Respuesta |
|----------|-----------|-----------|
| `/api/cloud/health` | Chequeo completo de salud | JSON con estado de componentes |
| `/api/cloud/health/live` | Liveness de Kubernetes | `{"status": "ok"}` |
| `/api/cloud/health/ready` | Readiness de Kubernetes | `{"status": "ready"}` o 503 |

### Respuesta de chequeo de salud

<details>
<summary>Respuesta de ejemplo</summary>

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "timestamp": "2024-01-15T10:30:00Z",
  "components": [
    {
      "name": "database",
      "status": "healthy",
      "latency_ms": 1.2,
      "message": "SQLite database accessible"
    },
    {
      "name": "secrets",
      "status": "healthy",
      "message": "API token configured"
    }
  ]
}
```
</details>

Incluir verificaciones de API del proveedor (agrega latencia):
```
GET /api/cloud/health?include_providers=true
```

## Métricas de Prometheus

Endpoint de scraping: `/api/cloud/metrics/prometheus`

### Habilitar exportación de Prometheus

La exportación de Prometheus está deshabilitada por defecto. Habilítala desde la pestaña Metrics en el panel de administración o vía API:

<details>
<summary>Habilitar vía API</summary>

```bash
curl -X PUT http://localhost:8080/api/cloud/metrics/config \
  -H "Content-Type: application/json" \
  -d '{"prometheus_enabled": true, "prometheus_categories": ["jobs", "http", "health"]}'
```
</details>

### Categorías de métricas

Las métricas están organizadas en categorías que se pueden habilitar individualmente:

| Categoría | Descripción | Métricas clave |
|-----------|-------------|----------------|
| `jobs` | Conteo de trabajos, estado, profundidad de cola, costos | `simpletuner_jobs_total`, `simpletuner_cost_usd_total` |
| `http` | Conteo de solicitudes, errores, latencia | `simpletuner_http_requests_total`, `simpletuner_http_errors_total` |
| `rate_limits` | Violaciones de límite de tasa | `simpletuner_rate_limit_violations_total` |
| `approvals` | Métricas del flujo de aprobaciones | `simpletuner_approval_requests_pending` |
| `audit` | Actividad del log de auditoría | `simpletuner_audit_log_entries_total` |
| `health` | Uptime del servidor, salud de componentes | `simpletuner_uptime_seconds`, `simpletuner_health_database_latency_ms` |
| `circuit_breakers` | Estado del circuit breaker del proveedor | `simpletuner_circuit_breaker_state` |
| `provider` | Límites de costo, saldo de crédito | `simpletuner_cost_limit_percent_used` |

### Plantillas de configuración

Plantillas de arranque rápido para casos comunes:

| Plantilla | Categorías | Caso de uso |
|-----------|------------|-------------|
| `minimal` | jobs | Monitoreo liviano de trabajos |
| `standard` | jobs, http, health | Predeterminado recomendado |
| `security` | jobs, http, rate_limits, audit, approvals | Monitoreo de seguridad |
| `full` | Todas las categorías | Observabilidad completa |

<details>
<summary>Aplicar una plantilla</summary>

```bash
curl -X POST http://localhost:8080/api/cloud/metrics/config/templates/standard
```
</details>

### Métricas disponibles

<details>
<summary>Referencia de métricas</summary>

```
# Server uptime
simpletuner_uptime_seconds 3600.5

# Job metrics
simpletuner_jobs_total 150
simpletuner_jobs_by_status{status="completed"} 120
simpletuner_jobs_by_status{status="failed"} 10
simpletuner_jobs_by_status{status="running"} 5
simpletuner_jobs_active 8
simpletuner_cost_usd_total 450.25
simpletuner_job_duration_seconds_avg 1800.5

# HTTP metrics
simpletuner_http_requests_total{endpoint="POST_/api/cloud/jobs/submit"} 50
simpletuner_http_errors_total{endpoint_status="POST_/api/cloud/jobs/submit_500"} 2
simpletuner_http_request_latency_ms_avg{endpoint="POST_/api/cloud/jobs/submit"} 250.5

# Rate limiting
simpletuner_rate_limit_violations_total 15
simpletuner_rate_limit_tracked_clients 42

# Approvals
simpletuner_approval_requests_pending 3
simpletuner_approval_requests_by_status{status="approved"} 25

# Audit
simpletuner_audit_log_entries_total 1500
simpletuner_audit_log_entries_24h 120

# Circuit breakers (per provider)
simpletuner_circuit_breaker_state{provider="..."} 0
simpletuner_circuit_breaker_failures_total{provider="..."} 5

# Provider status (per provider)
simpletuner_cost_limit_percent_used{provider="..."} 45.5
simpletuner_credit_balance_usd{provider="..."} 150.00
```
</details>

### Configuración de Prometheus

<details>
<summary>Configuración de scraping en prometheus.yml</summary>

```yaml
scrape_configs:
  - job_name: 'simpletuner'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/api/cloud/metrics/prometheus'
    scrape_interval: 30s
```
</details>

### Previsualizar salida de métricas

Previsualiza lo que se exportará sin afectar la configuración:

```bash
curl "http://localhost:8080/api/cloud/metrics/config/preview?categories=jobs&categories=health"
```

## Limitación de tasa

### Resumen

SimpleTuner incluye limitación de tasa integrada para proteger contra abuso y asegurar uso justo de recursos. Los límites se aplican por IP con reglas configurables para distintos endpoints.

### Configuración

La limitación de tasa puede configurarse vía variables de entorno:

<details>
<summary>Variables de entorno</summary>

```bash
# Default rate limit for unmatched endpoints
export RATE_LIMIT_CALLS=100      # Requests per period
export RATE_LIMIT_PERIOD=60      # Period in seconds

# Set to 0 to disable rate limiting entirely
export RATE_LIMIT_CALLS=0
```
</details>

### Reglas de límite de tasa predeterminadas

Los distintos endpoints tienen límites según su sensibilidad:

| Patrón de endpoint | Límite | Periodo | Métodos | Razón |
|--------------------|--------|---------|---------|-------|
| `/api/auth/login` | 5 | 60s | POST | Protección contra fuerza bruta |
| `/api/auth/register` | 3 | 60s | POST | Abuso de registro de usuarios |
| `/api/auth/api-keys` | 10 | 60s | POST | Creación de API keys |
| `/api/cloud/jobs` | 20 | 60s | POST | Envío de trabajos |
| `/api/cloud/jobs/.+/cancel` | 30 | 60s | POST | Cancelación de trabajos |
| `/api/webhooks/` | 100 | 60s | All | Entrega de webhooks |
| `/api/cloud/storage/` | 50 | 60s | All | Subidas de almacenamiento |
| `/api/quotas/` | 30 | 60s | All | Operaciones de cuota |
| Todos los demás endpoints | 100 | 60s | All | Fallback predeterminado |

### Rutas excluidas

Las siguientes rutas están excluidas de la limitación de tasa:

- `/health` - Chequeos de salud
- `/api/events/stream` - Conexiones SSE
- `/static/` - Archivos estáticos
- `/api/cloud/hints` - Pistas de UI (no sensibles a seguridad)
- `/api/users/me` - Verificación del usuario actual
- `/api/cloud/providers` - Lista de proveedores

### Encabezados de respuesta

Todas las respuestas incluyen encabezados de límite de tasa:

```
X-RateLimit-Limit: 100        # Maximum requests allowed
X-RateLimit-Remaining: 95     # Requests remaining in period
X-RateLimit-Reset: 1705320000 # Unix timestamp when limit resets
```

<details>
<summary>Respuesta por exceder el límite</summary>

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 45
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1705320045

{"detail": "Rate limit exceeded. Please try again later."}
```
</details>

### Detección de IP del cliente

El middleware maneja correctamente encabezados de proxy:

1. `X-Forwarded-For` - Encabezado estándar de proxy (la primera IP es la del cliente)
2. `X-Real-IP` - Encabezado de proxy de Nginx
3. IP de conexión directa - Fallback

Los límites se omiten para localhost (`127.0.0.1`, `::1`) en desarrollo.
