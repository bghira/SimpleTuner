# Guía empresarial

Este documento cubre el despliegue de SimpleTuner en entornos multiusuario con autenticación, flujos de aprobación y gestión de cuotas.

## 1. Despliegue e infraestructura

### Métodos de configuración

La mayoría de funciones empresariales pueden configurarse vía **Web UI** (panel de Administración) o **REST API**. Algunos ajustes de infraestructura requieren un archivo de configuración o variables de entorno.

| Función | Web UI | API | Archivo de config |
|---------|--------|-----|------------------|
| Proveedores OIDC/LDAP | ✓ | ✓ | ✓ |
| Usuarios y roles | ✓ | ✓ | |
| Reglas de aprobación | ✓ | ✓ | |
| Cuotas | ✓ | ✓ | |
| Notificaciones | ✓ | ✓ | |
| Bypass de red (proxies confiables) | | | ✓ |
| Sondeo de trabajos en background | | | ✓ |
| Ajustes de TLS | | | ✓ |

El **archivo de config** (`simpletuner-enterprise.yaml` o `.json`) solo se necesita para ajustes de infraestructura que deben conocerse al inicio. SimpleTuner busca en estas ubicaciones:

1. `$SIMPLETUNER_ENTERPRISE_CONFIG` (variable de entorno)
2. `./simpletuner-enterprise.yaml` (directorio actual)
3. `~/.config/simpletuner/enterprise.yaml`
4. `/etc/simpletuner/enterprise.yaml`

El archivo soporta interpolación de variables de entorno con la sintaxis `${VAR}`.

### Checklist de inicio rápido

1.  **Inicia SimpleTuner**: `simpletuner server` (o `--webui` para uso local)
2.  **Configura vía UI**: Navega al panel de Administración para configurar usuarios, SSO, cuotas
3.  **Health checks** (para producción):
    *   Liveness: `GET /api/cloud/health/live` (200 OK)
    *   Readiness: `GET /api/cloud/health/ready` (200 OK)
    *   Deep Check: `GET /api/cloud/health` (incluye conectividad de proveedores)

### Seguridad de red y bypass de autenticación

<details>
<summary>Configurar proxies confiables y bypass de red interna (archivo de config requerido)</summary>

En entornos corporativos (VPNs, VPC privadas), quizá quieras confiar en tráfico interno o delegar autenticación a un gateway.

**simpletuner-enterprise.yaml:**

```yaml
network:
  # Trust headers from your load balancer (e.g., AWS ALB, Nginx)
  trust_proxy_headers: true
  trusted_proxies:
    - "10.0.0.0/8"
    - "192.168.0.0/16"

  # Optional: Trust specific internal subnets to bypass login
  bypass_auth_for_internal: true
  internal_networks:
    - "10.10.0.0/16"  # VPN Clients

auth:
  # Always allow health checks without auth
  bypass_paths:
    - "/health"
    - "/api/cloud/health"
    - "/api/cloud/metrics/prometheus"
```

</details>

### Balanceador de carga y configuración TLS

SimpleTuner espera un reverse proxy upstream para terminación TLS.

<details>
<summary>Ejemplo de reverse proxy nginx</summary>

```nginx
server {
    listen 443 ssl http2;
    server_name trainer.internal;

    ssl_certificate /etc/ssl/certs/simpletuner.crt;
    ssl_certificate_key /etc/ssl/private/simpletuner.key;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support for real-time logs/SSE
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

</details>

### Observabilidad (Prometheus y logging)

**Métricas:**
Scrapea `GET /api/cloud/metrics/prometheus` para insights operativos.
*   `simpletuner_jobs_active`: Profundidad de cola actual.
*   `simpletuner_cost_total_usd`: Seguimiento de gasto.
*   `simpletuner_uptime_seconds`: Disponibilidad.

**Logging:**
Configura `SIMPLETUNER_LOG_FORMAT=json` para ingesta en Splunk/Datadog/ELK.

<details>
<summary>Configuración de retención de datos</summary>

Configura períodos de retención para requisitos de cumplimiento vía variables de entorno:

| Variable | Default | Descripción |
|----------|---------|-------------|
| `SIMPLETUNER_JOB_RETENTION_DAYS` | 90 | Días para retener registros de trabajos completados |
| `SIMPLETUNER_AUDIT_RETENTION_DAYS` | 90 | Días para retener entradas de auditoría |

```bash
# SOC 2 / HIPAA: 1 year retention
export SIMPLETUNER_JOB_RETENTION_DAYS=365
export SIMPLETUNER_AUDIT_RETENTION_DAYS=365

# Disable automatic cleanup (manual management)
export SIMPLETUNER_JOB_RETENTION_DAYS=0
```

Establecer `0` deshabilita la limpieza automática. La limpieza corre diariamente.

</details>

---


## 2. Gestión de identidad y acceso (SSO)

SimpleTuner soporta OIDC (OpenID Connect) y LDAP para SSO con Okta, Azure AD, Keycloak y Active Directory.

### Configurar proveedores

**Vía Web UI:** Navega a **Administration → Auth** para añadir y configurar proveedores.

**Vía API:** Ver el [API Cookbook](#4-api-cookbook) para ejemplos con curl.

<details>
<summary>Vía archivo de config (para flujos IaC/GitOps)</summary>

Agrega a tu `simpletuner-enterprise.yaml`:

```yaml
oidc:
  enabled: true
  provider: "okta"  # or "azure_ad", "google"

  client_id: "0oa1234567890abcdef"
  client_secret: "${OIDC_CLIENT_SECRET}"
  issuer_url: "https://your-org.okta.com/oauth2/default"

  scopes: ["openid", "email", "profile", "groups"]

  # Map Identity Provider groups to SimpleTuner Roles
  role_mapping:
    claim: "groups"
    admin_groups: ["ML-Platform-Admins"]
    user_groups: ["ML-Researchers"]
```

</details>

<details>
<summary>Validación de estado OAuth entre workers</summary>

Al usar autenticación OIDC en despliegues multi-worker (p. ej., detrás de un balanceador con múltiples workers Gunicorn), la validación de estado OAuth debe funcionar entre todos los workers. SimpleTuner lo maneja automáticamente guardando el estado OAuth en la base de datos.

**Cómo funciona:**

1. **Generación de estado**: Cuando un usuario inicia login OIDC, se genera un token de estado aleatorio criptográfico y se guarda en la base de datos con el nombre del proveedor, URI de redirección y expiración de 10 minutos.

2. **Validación de estado**: Cuando llega el callback (potencialmente a otro worker), se busca el estado y se consume de forma atómica (de un solo uso).

3. **Limpieza**: Los estados expirados se purgan automáticamente durante operaciones normales.

No se necesita configuración adicional. El almacenamiento del estado OAuth usa la misma base de datos que trabajos y usuarios.

**Troubleshooting de errores "Invalid OAuth state":**
1. Verifica si el callback llegó dentro de los 10 minutos desde el inicio de login
2. Verifica que todos los workers compartan la misma ruta de base de datos
3. Comprueba permisos de escritura en la base de datos
4. Busca errores "Failed to store OAuth state" en logs

</details>

### Gestión de usuarios y roles

SimpleTuner usa un sistema de roles jerárquico. Los usuarios pueden gestionarse vía `GET/POST /api/users`.

| Rol | Prioridad | Descripción |
|------|----------|-------------|
| **Viewer** | 10 | Acceso de solo lectura al historial y logs de trabajos. |
| **Researcher** | 20 | Acceso estándar. Puede enviar trabajos y gestionar sus propias API keys. |
| **Lead** | 30 | Puede aprobar trabajos pendientes y ver uso de recursos del equipo. |
| **Admin** | 100 | Acceso total al sistema, incluyendo gestión de usuarios y configuración de reglas. |

---


## 3. Gobernanza y operaciones

### Flujos de aprobación

Controla costos y uso de recursos requiriendo aprobaciones para criterios específicos. Las reglas se evalúan al momento del envío.

**Workflow:**
1.  El usuario envía un trabajo -> el estado pasa a `pending_approval`.
2.  Los leads consultan `GET /api/approvals/requests`.
3.  El lead llama `POST /.../approve` o `reject`.
4.  El trabajo continúa automáticamente a la cola o se cancela.

<details>
<summary>Motor de reglas de aprobación</summary>

El motor evalúa los envíos contra reglas configuradas. Las reglas se procesan en orden de prioridad; la primera regla que coincide dispara el requisito de aprobación.

**Condiciones de regla disponibles:**

| Condición | Descripción |
|-----------|-------------|
| `cost_exceeds` | Dispara cuando el costo estimado supera el umbral (USD) |
| `hardware_type` | Coincide con tipo de hardware (patrón glob, p. ej., `a100*`) |
| `daily_jobs_exceed` | Dispara cuando el conteo diario de trabajos del usuario supera el umbral |
| `first_job` | Dispara para el primer trabajo del usuario |
| `config_pattern` | Coincide con patrones de nombre de config |
| `provider` | Coincide con nombre de proveedor específico |

**Ejemplo: requerir aprobación para trabajos sobre $50:**

```bash
curl -X POST http://localhost:8080/api/approvals/rules \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "name": "high_cost",
    "condition": "cost_exceeds",
    "threshold": "50",
    "required_approver_level": "lead",
    "exempt_levels": ["admin"]
  }'
```

Las reglas pueden especificar `exempt_levels` para permitir que ciertos usuarios eviten aprobación, y `applies_to_provider`/`applies_to_level` para acotar reglas.

</details>

<details>
<summary>Aprobación basada en email (flujo IMAP)</summary>

Para equipos que prefieren flujos basados en correo, SimpleTuner soporta aprobación por respuestas de email usando IMAP IDLE.

**Cómo funciona:**
1. El envío del trabajo dispara el requisito de aprobación
2. Se envía un email de notificación a aprobadores con un token de respuesta único
3. El manejador IMAP monitorea la bandeja usando IDLE (notificaciones push)
4. El aprobador responde con "approve" o "reject" (o aliases como `yes`, `lgtm`, `+1`)
5. El sistema parsea la respuesta y procesa la aprobación

Configura vía **Administration → Notifications** o API. Los tokens de respuesta expiran a las 24 horas y son de un solo uso.

</details>

### Cola de trabajos y concurrencia

El scheduler gestiona el uso justo de recursos. Consulta su [documentación dedicada](../../JOB_QUEUE.md) para detalles.

*   **Prioridad:** Admins > Leads > Researchers > Viewers.
*   **Concurrencia:** Se aplican límites globales y por usuario.
    *   Actualiza límites vía UI: **Cloud tab → Job Queue panel** (solo admin)
    *   Actualiza límites vía API: `POST /api/queue/concurrency` con `{"max_concurrent": 10, "user_max_concurrent": 3}`

### Sondeo de estado de trabajos (sin webhooks)

Para entornos seguros donde los webhooks públicos no son posibles, SimpleTuner incluye un poller en background.

Agrega a `simpletuner-enterprise.yaml`:

```yaml
background:
  job_status_polling:
    enabled: true
    interval_seconds: 30
```

Este servicio consulta la API del proveedor cada 30s y actualiza la base de datos local, emitiendo eventos en tiempo real a la UI vía SSE.

### Rotación de API keys

Gestiona credenciales de proveedores de nube de forma segura. Consulta **API Cookbook** para scripts de rotación y detalles específicos en la [documentación de Cloud Training](../cloud/README.md).

---


## 4. API Cookbook

<details>
<summary>Ejemplos de configuración OIDC/LDAP</summary>

**Keycloak (OIDC):**
```bash
curl -X POST http://localhost:8080/api/cloud/external-auth/providers \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "name": "keycloak",
    "provider_type": "oidc",
    "enabled": true,
    "config": {
      "issuer": "https://keycloak.example.com/realms/ml-training",
      "client_id": "simpletuner",
      "client_secret": "your-client-secret",
      "scopes": ["openid", "email", "profile", "roles"],
      "roles_claim": "realm_access.roles"
    }
  }'
```

**LDAP / Active Directory:**
```bash
curl -X POST http://localhost:8080/api/cloud/external-auth/providers \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "name": "corporate-ad",
    "provider_type": "ldap",
    "enabled": true,
    "level_mapping": {
      "CN=ML-Admins,OU=Groups,DC=corp,DC=com": ["admin"]
    },
    "config": {
      "server": "ldaps://ldap.corp.com:636",
      "base_dn": "DC=corp,DC=com",
      "bind_dn": "CN=svc-simpletuner,OU=Service Accounts,DC=corp,DC=com",
      "bind_password": "service-account-password",
      "user_search_filter": "(sAMAccountName={username})",
      "use_ssl": true
    }
  }'
```

</details>

<details>
<summary>Ejemplos de administración de usuarios</summary>

**Crear un Researcher:**
```bash
curl -X POST http://localhost:8080/api/users \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "email": "researcher@company.com",
    "username": "jsmith",
    "password": "secure_password_123",
    "level_names": ["researcher"]
  }'
```

**Conceder permiso personalizado:**
```bash
curl -X POST http://localhost:8080/api/users/123/permissions \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"permission_name": "admin.approve", "granted": true}'
```

</details>

<details>
<summary>Gestión de credenciales</summary>

SimpleTuner incluye gestión del ciclo de vida de credenciales para rastrear, rotar y auditar credenciales de API.

**Resolución de credenciales:** Al enviar trabajos, SimpleTuner revisa primero credenciales por usuario y luego hace fallback a credenciales globales (variables de entorno).

| Escenario | Por usuario | Global | Comportamiento |
|----------|------------|--------|----------------|
| **Clave org compartida** | ❌ | ✅ | Todos los usuarios comparten la API key de la org |
| **BYOK** | ✅ | ❌ | Cada usuario proporciona su propia key |
| **Híbrido** | Algunos | ✅ | Usuarios con key usan la suya, otros usan la global |

**Rotación:** Navega a **Admin > Auth** → usuario → **Manage Credentials** → **Rotate**. Credenciales obsoletas (>90 días) muestran un badge de advertencia.

</details>

#### Orquestación externa {#external-orchestration-airflow}

<details>
<summary>Ejemplo de Airflow</summary>

```python
def submit_and_wait(job_config, provider="replicate", **context):
    resp = requests.post(
        f"http://localhost:8080/api/cloud/{provider}/submit",
        json=job_config,
        headers={"Authorization": f"Bearer {TOKEN}"}
    )
    job_id = resp.json()["job_id"]

    while True:
        status = requests.get(f"http://localhost:8080/api/cloud/jobs/{job_id}")
        state = status.json()["status"]
        if state in ("completed", "failed", "cancelled"):
            return status.json()
        time.sleep(30)
```

</details>

---


## 5. Troubleshooting

**Fallos de health check**
*   `503 Service Unavailable`: Revisa conectividad de base de datos.
*   `Degraded`: Normalmente significa que un componente opcional (como la API de un proveedor cloud) es inalcanzable o no está configurado.

**Problemas de autenticación**
*   **OIDC Redirect Loop:** Verifica que `issuer_url` coincida exactamente con lo que espera el proveedor (¡los trailing slashes importan!).
*   **Bypass de auth interno:** Revisa logs del servidor por "Auth bypassed for IP..." para confirmar que el balanceador pasa el `X-Real-IP` correcto.

**Actualizaciones de trabajos estancadas**
*   Si los webhooks están bloqueados, asegúrate de habilitar **Job Status Polling** en `simpletuner-enterprise.yaml`.
*   Revisa `GET /api/cloud/metrics/prometheus` para `simpletuner_jobs_active` y ver si el estado interno cree que hay trabajos en ejecución.

**Métricas faltantes**
*   Asegúrate de que tu scraper de Prometheus esté configurado para `/api/cloud/metrics/prometheus` y no solo `/metrics`.

---


## 6. Organizaciones y cuotas de equipo

SimpleTuner soporta organizaciones jerárquicas y equipos con enforcement de cuotas por techo.

### Jerarquía

```
Organization (quota ceiling)
    └── Team (quota ceiling, bounded by org)
         └── User (limit, bounded by team and org)
```

### Modelo de techo

Las cuotas usan un modelo de techo donde los límites de la org son techos absolutos:
- **Cuota de org**: Techo absoluto para todos los miembros
- **Cuota de equipo**: Techo para miembros del equipo (no puede superar a la org)
- **Cuota de usuario/nivel**: Límites específicos (acotados por equipo y org)

**Ejemplo:**
- Techo de org: 100 trabajos concurrentes
- Techo de equipo: 20 trabajos concurrentes
- Límite de usuario: 50 trabajos concurrentes → **Efectivo: 20** (aplica techo de equipo)

**Reglas de enforcement:**
- Las cuotas de equipo se validan al configurarlas: intentar fijar una cuota de equipo mayor que el techo de la org devuelve HTTP 400
- Las cuotas de usuario se validan en runtime: el límite efectivo es el mínimo entre usuario, equipo y techos de org
- Reducir el techo de org no reduce automáticamente techos de equipos existentes (admin debe actualizar manualmente)

<details>
<summary>Ejemplos de API</summary>

**Crear organización:**
```bash
curl -X POST http://localhost:8080/api/orgs \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"name": "ML Research", "slug": "ml-research"}'
```

**Configurar techo de cuota de org:**
```bash
curl -X POST http://localhost:8080/api/orgs/1/quotas \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"quota_type": "concurrent_jobs", "limit_value": 100, "action": "block"}'
```

**Crear equipo:**
```bash
curl -X POST http://localhost:8080/api/orgs/1/teams \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"name": "NLP Team", "slug": "nlp"}'
```

**Agregar usuario al equipo:**
```bash
curl -X POST http://localhost:8080/api/orgs/1/teams/1/members \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"user_id": 123, "role": "member"}'
```

</details>

### Acciones de cuota y límite de costo

Cuando se alcanza una cuota o límite de costo, la `action` configurada determina el comportamiento:

| Acción | Comportamiento |
|--------|----------------|
| `warn` | El trabajo continúa con advertencia en logs/UI |
| `block` | Envío de trabajo rechazado |
| `notify` | El trabajo continúa, admins alertados |

<details>
<summary>Configuración de límite de costo</summary>

Los límites de costo pueden configurarse por proveedor vía **Cloud tab → Settings** o API:

```bash
curl -X PUT http://localhost:8080/api/cloud/providers/<provider>/config \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "cost_limit_enabled": true,
    "cost_limit_amount": 500.00,
    "cost_limit_period": "monthly",
    "cost_limit_action": "warn"
  }'
```

Revisa estado: `GET /api/cloud/metrics/cost-limit-status`

</details>

---


## 7. Limitaciones

### Trabajos de workflow/pipeline (DAGs)

SimpleTuner no soporta dependencias entre trabajos ni flujos multi-paso donde la salida de un trabajo alimenta a otro. Cada trabajo en la nube es independiente.

**Enfoque recomendado:** Usa herramientas de orquestación externa como Airflow, Prefect o Dagster para encadenar trabajos vía la REST API. Consulta el [ejemplo de Airflow](#external-orchestration-airflow) en el API Cookbook.

### Reanudar ejecuciones de entrenamiento

No hay soporte integrado para reanudar ejecuciones interrumpidas, fallidas o canceladas. Los trabajos en la nube no se recuperan automáticamente desde checkpoints.

**Workarounds:**
- Configura pushes frecuentes a HuggingFace Hub (`--push_checkpoints_to_hub`) para guardar estado intermedio
- Implementa gestión de checkpoints personalizada descargando outputs y usándolos como puntos de inicio para nuevos trabajos
- Para cargas críticas, considera dividir entrenamientos largos en segmentos más pequeños

<details>
<summary>Referencia de features UI</summary>

| Feature | Ubicación UI | API |
|---------|-------------|-----|
| Organizations & Teams | Administration → Orgs | `/api/orgs` |
| Quotas | Administration → Quotas | `/api/orgs/{id}/quotas` |
| OIDC/LDAP | Administration → Auth | `/api/cloud/external-auth/providers` |
| Users | Administration → Users | `/api/users` |
| Audit Logs | Sidebar → Audit Log | `/api/audit` |
| Queue | Cloud tab → Job Queue | `/api/queue/concurrency` |
| Approvals | Administration → Approvals | `/api/approvals/requests` |

La sección Administration es visible cuando no hay auth configurado (modo single-user) o el usuario tiene privilegios admin.

</details>

<details>
<summary>Flujo de onboarding empresarial</summary>

El panel Admin incluye un onboarding guiado que ayuda a configurar autenticación, organizaciones, equipos, cuotas y credenciales en orden.

| Paso | Feature |
|------|---------|
| 1 | Authentication (OIDC/LDAP) |
| 2 | Organization |
| 3 | Teams |
| 4 | Quotas |
| 5 | Credentials |

Cada paso puede completarse u omitirse. El estado persiste en localStorage del navegador.

</details>

---


## 8. Sistema de notificaciones

SimpleTuner incluye un sistema de notificaciones multicanal para estado de trabajos, aprobaciones, cuotas y eventos del sistema.

| Canal | Caso de uso |
|---------|----------|
| **Email** | Solicitudes de aprobación, finalización de trabajos (SMTP/IMAP) |
| **Webhook** | Integración CI/CD (JSON + firmas HMAC) |
| **Slack** | Notificaciones de equipo (webhooks entrantes) |

Configura vía **Administration → Notifications** o API.

<details>
<summary>Tipos de eventos</summary>

| Categoría | Eventos |
|----------|--------|
| Approval | `approval.required`, `approval.granted`, `approval.rejected`, `approval.expired` |
| Job | `job.submitted`, `job.started`, `job.completed`, `job.failed`, `job.cancelled` |
| Quota | `quota.warning`, `quota.exceeded`, `cost.warning`, `cost.exceeded` |
| System | `system.provider_error`, `system.provider_degraded`, `system.webhook_failure` |
| Auth | `auth.login_failure`, `auth.new_device` |

</details>

<details>
<summary>Ejemplos de configuración de canal</summary>

**Email:**
```bash
curl -X POST http://localhost:8080/api/cloud/notifications/channels \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "channel_type": "email",
    "name": "Primary Email",
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_use_tls": true
  }'
```

**Slack:**
```bash
curl -X POST http://localhost:8080/api/cloud/notifications/channels \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "channel_type": "slack",
    "name": "Training Alerts",
    "webhook_url": "https://hooks.slack.com/services/T00/B00/xxxx"
  }'
```

**Webhook:** payloads firmados con HMAC-SHA256 (header `X-SimpleTuner-Signature`).

</details>

---


## 9. Reglas de recursos

Las reglas de recursos proporcionan control de acceso granular para configs, tipos de hardware y rutas de salida usando patrones glob.

| Tipo | Patrón de ejemplo |
|------|-------------------|
| `config` | `team-x-*`, `production-*` |
| `hardware` | `gpu-a100*`, `*-80gb` |
| `provider` | `replicate`, `runpod` |

Las reglas usan acciones **allow/deny** con lógica de "más permisivo gana". Configura vía **Administration → Rules**.

<details>
<summary>Ejemplos de reglas</summary>

**Aislamiento de equipo:** Researchers solo pueden usar configs que empiezan con "team-x-"
```
Level: researcher
Rules:
  - config: "team-x-*" → allow
  - config: "*" → deny
```

**Restricciones de hardware:** Researchers limitados a T4/V100, leads pueden usar cualquier hardware
```
Level: researcher → hardware: "gpu-t4*" allow, "gpu-v100*" allow
Level: lead → hardware: "*" allow
```

</details>

---


## 10. Matriz de permisos

<details>
<summary>Matriz completa de permisos</summary>

### Permisos de trabajos

| Permiso | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `job.submit` | | ✓ | ✓ | ✓ |
| `job.view.own` | ✓ | ✓ | ✓ | ✓ |
| `job.view.all` | | | ✓ | ✓ |
| `job.cancel.own` | | ✓ | ✓ | ✓ |
| `job.cancel.all` | | | | ✓ |
| `job.priority.high` | | | ✓ | ✓ |
| `job.bypass.queue` | | | | ✓ |
| `job.bypass.approval` | | | | ✓ |

### Permisos de config

| Permiso | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `config.view` | ✓ | ✓ | ✓ | ✓ |
| `config.create` | | ✓ | ✓ | ✓ |
| `config.edit.own` | | ✓ | ✓ | ✓ |
| `config.edit.all` | | | | ✓ |

### Permisos de admin

| Permiso | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `admin.users` | | | | ✓ |
| `admin.approve` | | | ✓ | ✓ |
| `admin.audit` | | | ✓ | ✓ |
| `admin.config` | | | | ✓ |
| `queue.approve` | | | ✓ | ✓ |
| `queue.manage` | | | | ✓ |

### Permisos de org/equipo

| Permiso | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `org.view` | | | ✓ | ✓ |
| `org.create` | | | | ✓ |
| `team.view` | | | ✓ | ✓ |
| `team.create` | | | ✓ | ✓ |
| `team.manage.members` | | | ✓ | ✓ |

</details>

**Overrides de permisos:** Usuarios individuales pueden tener permisos otorgados o denegados vía **Administration → Users → Permission Overrides**.
