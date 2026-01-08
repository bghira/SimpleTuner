# Registro de auditoría

El sistema de auditoría de SimpleTuner proporciona un registro a prueba de manipulación de eventos relevantes para seguridad. Todas las acciones administrativas, eventos de autenticación y operaciones de trabajos se registran con verificación criptográfica de cadena.

## Resumen

El registro de auditoría captura:
- **Eventos de autenticación**: intentos de inicio de sesión (éxito/fallo), cierres de sesión, expiraciones de sesión
- **Gestión de usuarios**: creación, actualización, eliminación, cambios de permisos
- **Operaciones de API keys**: creación, revocación, uso
- **Gestión de credenciales**: cambios de credenciales de proveedores
- **Operaciones de trabajos**: envíos, cancelaciones, aprobaciones

## Acceder a los logs de auditoría

### Web UI

Navega a la pestaña **Audit** en el panel de admin para explorar entradas con opciones de filtro.

### CLI

```bash
# Listar entradas recientes de auditoría
simpletuner auth audit list

# Filtrar por tipo de evento
simpletuner auth audit list --event-type auth.login.failed

# Filtrar por usuario
simpletuner auth audit user 123

# Ver solo eventos de seguridad
simpletuner auth audit security

# Obtener estadísticas
simpletuner auth audit stats

# Verificar integridad de la cadena
simpletuner auth audit verify
```

### Endpoints de API

Todos los endpoints requieren el permiso `admin.audit`.

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/audit` | Listar entradas con filtros |
| GET | `/api/audit/stats` | Obtener estadísticas de auditoría |
| GET | `/api/audit/types` | Listar tipos de eventos disponibles |
| GET | `/api/audit/verify` | Verificar integridad de la cadena |
| GET | `/api/audit/user/{user_id}` | Obtener entradas de un usuario |
| GET | `/api/audit/security` | Obtener eventos relacionados con seguridad |

## Tipos de evento

### Eventos de autenticación

| Evento | Descripción |
|-------|-------------|
| `auth.login.success` | Inicio de sesión exitoso |
| `auth.login.failed` | Intento de inicio de sesión fallido |
| `auth.logout` | El usuario cerró sesión |
| `auth.session.expired` | Sesión expirada |
| `auth.api_key.used` | Se usó una API key |

### Eventos de gestión de usuarios

| Evento | Descripción |
|-------|-------------|
| `user.created` | Usuario nuevo creado |
| `user.updated` | Detalles de usuario actualizados |
| `user.deleted` | Usuario eliminado |
| `user.password.changed` | El usuario cambió su contraseña |
| `user.level.changed` | Nivel/rol de usuario cambiado |
| `user.permission.changed` | Permiso de usuario cambiado |

### Eventos de API key

| Evento | Descripción |
|-------|-------------|
| `api_key.created` | API key nueva creada |
| `api_key.revoked` | API key revocada |

### Eventos de credenciales

| Evento | Descripción |
|-------|-------------|
| `credential.created` | Credencial de proveedor añadida |
| `credential.deleted` | Credencial de proveedor eliminada |
| `credential.used` | Se usó la credencial |

### Eventos de trabajos

| Evento | Descripción |
|-------|-------------|
| `job.submitted` | Trabajo enviado a la cola |
| `job.cancelled` | Trabajo cancelado |
| `job.approved` | Aprobación de trabajo concedida |
| `job.rejected` | Aprobación de trabajo denegada |

## Parámetros de consulta

Al listar entradas de auditoría, puedes filtrar por:

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `event_type` | string | Filtrar por tipo de evento |
| `actor_id` | int | Filtrar por usuario que realizó la acción |
| `target_type` | string | Filtrar por tipo de recurso objetivo |
| `target_id` | string | Filtrar por ID del recurso objetivo |
| `since` | Fecha ISO | Timestamp de inicio |
| `until` | Fecha ISO | Timestamp de fin |
| `limit` | int | Máx entradas (1-500, default 50) |
| `offset` | int | Offset de paginación |

## Integridad de cadena

Cada entrada de auditoría incluye:
- Un hash criptográfico de su contenido
- Una referencia al hash de la entrada anterior
- Timestamp de un reloj monotónico

Esto crea una cadena hash que hace detectable la manipulación. Usa el endpoint de verificación o el comando CLI para comprobar integridad:

```bash
# Verify entire chain
simpletuner auth audit verify

# Verify specific range
simpletuner auth audit verify --start-id 100 --end-id 200
```

La verificación comprueba:
1. Que el hash de cada entrada coincide con su contenido
2. Que cada entrada referencia correctamente el hash de la anterior
3. Que no haya huecos en la secuencia

## Retención

Los logs de auditoría se almacenan en la base de datos de SimpleTuner. Configura la retención en tu despliegue:

```bash
# Environment variable for retention period (days)
SIMPLETUNER_AUDIT_RETENTION_DAYS=365
```

Las entradas antiguas pueden archivarse o purgarse según tus requisitos de cumplimiento.

## Consideraciones de seguridad

- Los logs de auditoría son append-only; las entradas no pueden modificarse ni eliminarse vía API
- Se requiere el permiso `admin.audit` para ver logs
- Los intentos de login fallidos se registran con direcciones IP para monitoreo de seguridad
- Considera reenviar logs de auditoría a un SIEM para despliegues en producción
