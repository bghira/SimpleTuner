# Autenticación externa (OIDC y LDAP)

SimpleTuner soporta proveedores de identidad externos para single sign-on (SSO). Los usuarios pueden autenticarse vía proveedores OIDC (Keycloak, Auth0, Okta, Azure AD, Google) o LDAP/Active Directory.

## Resumen

La autenticación externa proporciona:
- **Single Sign-On**: Los usuarios se autentican con credenciales corporativas existentes
- **Auto-provisioning**: Los nuevos usuarios se crean automáticamente en el primer login
- **Mapeo de niveles**: Mapea grupos/roles externos a niveles de acceso de SimpleTuner
- **Múltiples proveedores**: Configura varios proveedores simultáneamente

## Configurar OIDC

### Proveedores soportados

- Keycloak
- Auth0
- Okta
- Azure Active Directory
- Google Workspace
- Cualquier proveedor compatible con OpenID Connect

Se probó con OpenLDAP + Dex, Keycloak y Auth0.

### Configuración vía Web UI

1. Navega a **Administration > Manage Users > Auth Providers**
2. Haz clic en **Add Provider** y selecciona **OIDC Provider**
3. Ingresa los detalles del proveedor:
   - **Provider Name**: Identificador único (p. ej., `corporate-sso`)
   - **Type**: OIDC
   - **Issuer URL**: Endpoint de descubrimiento OIDC (p. ej., `https://auth.example.com/.well-known/openid-configuration`)
   - **Client ID**: De tu proveedor de identidad
   - **Client Secret**: De tu proveedor de identidad
   - **Scopes**: Usualmente `openid profile email`

### Configuración vía API

```bash
curl -X POST http://localhost:8001/api/cloud/external-auth/providers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "corporate-sso",
    "provider_type": "oidc",
    "enabled": true,
    "auto_create_users": true,
    "default_levels": ["researcher"],
    "config": {
      "client_id": "your-client-id",
      "client_secret": "your-client-secret",
      "discovery_url": "https://auth.example.com/.well-known/openid-configuration",
      "scopes": ["openid", "profile", "email"]
    }
  }'
```

### URL de callback OIDC

Configura tu proveedor de identidad con la URL de callback:

```
https://your-simpletuner-host/api/cloud/external-auth/oidc/{provider-name}/callback
```

Por ejemplo, si tu proveedor se llama `corporate-sso`:
```
https://simpletuner.example.com/api/cloud/external-auth/oidc/corporate-sso/callback
```

### Flujo de login OIDC

1. El usuario hace clic en "Login with SSO" en la página de login
2. El navegador redirige al proveedor de identidad
3. El usuario se autentica con sus credenciales corporativas
4. El proveedor de identidad redirige de vuelta al callback de SimpleTuner
5. SimpleTuner crea/actualiza el usuario local y establece la sesión

## Configurar LDAP

### Configuraciones soportadas

- Microsoft Active Directory
- OpenLDAP
- 389 Directory Server
- Cualquier servidor compatible con LDAPv3

Se probó con FreeIPA y OpenLDAP.

### Configuración vía Web UI

1. Navega a **Administration > Manage Users > Auth Providers**
2. Haz clic en **Add Provider** y selecciona **LDAP / Active Directory**
3. Ingresa los detalles LDAP:
   - **Provider Name**: Identificador único (p. ej., `company-ldap`)
   - **Type**: LDAP
   - **Server URL**: Dirección del servidor LDAP (p. ej., `ldap://ldap.example.com:389` o `ldaps://ldap.example.com:636`)
   - **Bind DN**: DN de la cuenta de servicio para consultas
   - **Bind Password**: Contraseña de la cuenta de servicio
   - **User Base DN**: Dónde buscar usuarios
   - **User Filter**: Filtro LDAP para encontrar usuarios
   - **Group Base DN**: Dónde buscar grupos (opcional)
   - **Group Filter**: Filtro LDAP para encontrar grupos (opcional)

### Configuración vía API

```bash
curl -X POST http://localhost:8001/api/cloud/external-auth/providers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "company-ldap",
    "provider_type": "ldap",
    "enabled": true,
    "auto_create_users": true,
    "default_levels": ["researcher"],
    "level_mapping": {
      "CN=SimpleTuner-Admins,OU=Groups,DC=example,DC=com": ["admin"],
      "CN=SimpleTuner-Users,OU=Groups,DC=example,DC=com": ["researcher"]
    },
    "config": {
      "server_url": "ldap://ldap.example.com:389",
      "use_tls": false,
      "bind_dn": "CN=simpletuner-svc,OU=Services,DC=example,DC=com",
      "bind_password": "service-account-password",
      "user_base_dn": "OU=Users,DC=example,DC=com",
      "user_filter": "(sAMAccountName={username})",
      "group_base_dn": "OU=Groups,DC=example,DC=com",
      "group_filter": "(member={user_dn})",
      "username_attribute": "sAMAccountName",
      "email_attribute": "mail",
      "display_name_attribute": "displayName"
    }
  }'
```

### Login LDAP

El login LDAP es por usuario/contraseña. Los usuarios ingresan sus credenciales LDAP en la página de login de SimpleTuner.

```bash
# API login
curl -X POST http://localhost:8001/api/cloud/external-auth/ldap/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "jsmith",
    "password": "ldap-password"
  }'
```

## Mapeo de niveles

Mapea grupos/roles externos a niveles de acceso de SimpleTuner:

```json
{
  "level_mapping": {
    "admin": ["admin"],
    "power-users": ["advanced"],
    "default": ["researcher"]
  }
}
```

Para OIDC, el mapeo usa roles/grupos del claim `groups` o `roles`.
Para LDAP, el mapeo usa DNs de grupos LDAP.

## Aprovisionamiento de usuarios

### Auto-provisioning

Cuando `auto_create_users` está habilitado (default), los usuarios se crean automáticamente en el primer login:

```json
{
  "auto_create_users": true,
  "default_levels": ["researcher"]
}
```

### Provisionamiento manual

Configura `auto_create_users: false` para requerir creación manual de usuarios. Los usuarios deben existir en SimpleTuner antes de poder iniciar sesión vía auth externa.

## Endpoints de API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/cloud/external-auth/providers` | Listar proveedores configurados |
| POST | `/api/cloud/external-auth/providers` | Crear proveedor |
| PATCH | `/api/cloud/external-auth/providers/{name}` | Actualizar proveedor |
| DELETE | `/api/cloud/external-auth/providers/{name}` | Eliminar proveedor |
| GET | `/api/cloud/external-auth/providers/{name}/test` | Probar conexión del proveedor |
| GET | `/api/cloud/external-auth/oidc/{provider}/start` | Iniciar flujo OIDC |
| GET | `/api/cloud/external-auth/oidc/callback` | Callback OIDC |
| POST | `/api/cloud/external-auth/ldap/login` | Login LDAP |
| GET | `/api/cloud/external-auth/available` | Listar proveedores disponibles (público) |

## Comandos CLI

```bash
# List providers (requires admin access)
simpletuner auth users auth-status

# External auth is managed via the web UI or API
# No dedicated CLI commands yet
```

## Notas de arquitectura

### Persistencia de estado OAuth

La autenticación OIDC requiere validar el parámetro `state` para prevenir ataques CSRF. SimpleTuner guarda el estado OAuth en la base de datos en lugar de en memoria para soportar:

- **Despliegues multi-worker**: Setups con balanceo donde el callback puede llegar a un worker distinto del que inició el flujo
- **Reinicios de servidor**: El estado sobrevive reinicios durante la ventana de autenticación
- **Escalado horizontal**: No se requieren sticky sessions

Los tokens de estado expiran a los 10 minutos y se eliminan tras validación exitosa.

## Consideraciones de seguridad

### OIDC

- Guarda client secrets de forma segura (variables de entorno o gestor de secretos)
- Usa HTTPS para todas las URLs de callback
- Valida el parámetro `state` para prevenir CSRF (manejado automáticamente vía estado respaldado en base de datos)
- Considera restringir URIs de redirección permitidas en el proveedor de identidad

### LDAP

- Usa LDAPS (puerto 636) o StartTLS en producción
- Usa una cuenta de servicio dedicada con permisos mínimos
- Rota credenciales de la cuenta de servicio regularmente
- Considera usar pooling de conexiones para despliegues de alto tráfico

### General

- La autenticación externa evita políticas locales de contraseñas
- Usuarios autenticados externamente no pueden restablecer su contraseña en SimpleTuner
- Los logs de auditoría registran todos los intentos de autenticación, incluyendo fallos
- Considera ajustes de timeout de sesión para cumplimiento

## Troubleshooting

### Problemas OIDC

**"Invalid or expired state"**
- Los tokens de estado expiran después de 10 minutos
- Verifica sincronización de reloj entre SimpleTuner y el proveedor de identidad

**"Discovery URL unreachable"**
- Verifica conectividad de red al proveedor de identidad
- Revisa reglas de firewall para permitir HTTPS saliente

**"Invalid client_id or client_secret"**
- Re-verifica credenciales en el proveedor de identidad
- Asegúrate de que no haya espacios al final en la configuración

### Problemas LDAP

**"Connection refused"**
- Revisa URL y puerto del servidor
- Verifica que el servicio LDAP esté en ejecución
- Revisa reglas de firewall

**"Invalid credentials"**
- Verifica el formato de bind DN (se requiere DN completo)
- Revisa la contraseña de bind
- Prueba con `ldapsearch` en CLI

**"User not found"**
- Revisa la ruta user_base_dn
- Verifica la sintaxis de user_filter
- Asegúrate de que el usuario exista y no esté deshabilitado

**"No groups returned"**
- Verifica la ruta group_base_dn
- Revisa la sintaxis de group_filter
- Asegúrate de que el atributo de membresía de grupo sea correcto

## Configuraciones de ejemplo

### Keycloak

```json
{
  "name": "keycloak",
  "provider_type": "oidc",
  "config": {
    "client_id": "simpletuner",
    "client_secret": "your-secret",
    "discovery_url": "https://keycloak.example.com/realms/myrealm/.well-known/openid-configuration",
    "scopes": ["openid", "profile", "email", "groups"]
  }
}
```

### Active Directory

```json
{
  "name": "active-directory",
  "provider_type": "ldap",
  "config": {
    "server_url": "ldaps://dc01.example.com:636",
    "use_tls": true,
    "bind_dn": "CN=simpletuner,OU=Service Accounts,DC=example,DC=com",
    "bind_password": "service-password",
    "user_base_dn": "OU=Users,DC=example,DC=com",
    "user_filter": "(sAMAccountName={username})",
    "group_base_dn": "OU=Groups,DC=example,DC=com",
    "group_filter": "(member={user_dn})",
    "username_attribute": "sAMAccountName",
    "email_attribute": "mail",
    "display_name_attribute": "displayName"
  },
  "level_mapping": {
    "CN=ST-Admins,OU=Groups,DC=example,DC=com": ["admin"],
    "CN=ST-Advanced,OU=Groups,DC=example,DC=com": ["advanced"],
    "CN=ST-Users,OU=Groups,DC=example,DC=com": ["researcher"]
  }
}
```

### Azure AD (OIDC)

```json
{
  "name": "azure-ad",
  "provider_type": "oidc",
  "config": {
    "client_id": "your-app-id",
    "client_secret": "your-client-secret",
    "discovery_url": "https://login.microsoftonline.com/your-tenant-id/v2.0/.well-known/openid-configuration",
    "scopes": ["openid", "profile", "email"]
  }
}
```
