# External Authentication (OIDC & LDAP)

SimpleTuner supports external identity providers for single sign-on (SSO). Users can authenticate via OIDC providers (Keycloak, Auth0, Okta, Azure AD, Google) or LDAP/Active Directory.

## Overview

External authentication provides:
- **Single Sign-On**: Users authenticate with existing corporate credentials
- **Auto-provisioning**: New users are created automatically on first login
- **Level mapping**: Map external groups/roles to SimpleTuner access levels
- **Multiple providers**: Configure multiple providers simultaneously

## Setting Up OIDC

### Supported Providers

- Keycloak
- Auth0
- Okta
- Azure Active Directory
- Google Workspace
- Any OpenID Connect compliant provider

Testing was done with OpenLDAP + Dex, Keycloak, and Auth0.

### Configuration via Web UI

1. Navigate to **Administration > Manage Users > Auth Providers**
2. Click **Add Provider** and select **OIDC Provider**
3. Enter provider details:
   - **Provider Name**: Unique identifier (e.g., `corporate-sso`)
   - **Type**: OIDC
   - **Issuer URL**: OIDC discovery endpoint (e.g., `https://auth.example.com/.well-known/openid-configuration`)
   - **Client ID**: From your identity provider
   - **Client Secret**: From your identity provider
   - **Scopes**: Usually `openid profile email`

### Configuration via API

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

### OIDC Callback URL

Configure your identity provider with the callback URL:

```
https://your-simpletuner-host/api/cloud/external-auth/oidc/{provider-name}/callback
```

For example, if your provider is named `corporate-sso`:
```
https://simpletuner.example.com/api/cloud/external-auth/oidc/corporate-sso/callback
```

### OIDC Login Flow

1. User clicks "Login with SSO" on the login page
2. Browser redirects to identity provider
3. User authenticates with their corporate credentials
4. Identity provider redirects back to SimpleTuner callback
5. SimpleTuner creates/updates local user and establishes session

## Setting Up LDAP

### Supported Configurations

- Microsoft Active Directory
- OpenLDAP
- 389 Directory Server
- Any LDAPv3 compliant server

Testing was done with FreeIPA and OpenLDAP.

### Configuration via Web UI

1. Navigate to **Administration > Manage Users > Auth Providers**
2. Click **Add Provider** and select **LDAP / Active Directory**
3. Enter LDAP details:
   - **Provider Name**: Unique identifier (e.g., `company-ldap`)
   - **Type**: LDAP
   - **Server URL**: LDAP server address (e.g., `ldap://ldap.example.com:389` or `ldaps://ldap.example.com:636`)
   - **Bind DN**: Service account DN for queries
   - **Bind Password**: Service account password
   - **User Base DN**: Where to search for users
   - **User Filter**: LDAP filter for finding users
   - **Group Base DN**: Where to search for groups (optional)
   - **Group Filter**: LDAP filter for finding groups (optional)

### Configuration via API

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

### LDAP Login

LDAP login is username/password based. Users enter their LDAP credentials on the SimpleTuner login page.

```bash
# API login
curl -X POST http://localhost:8001/api/cloud/external-auth/ldap/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "jsmith",
    "password": "ldap-password"
  }'
```

## Level Mapping

Map external groups/roles to SimpleTuner access levels:

```json
{
  "level_mapping": {
    "admin": ["admin"],
    "power-users": ["advanced"],
    "default": ["researcher"]
  }
}
```

For OIDC, the mapping uses roles/groups from the `groups` or `roles` claim.
For LDAP, the mapping uses LDAP group DNs.

## User Provisioning

### Auto-provisioning

When `auto_create_users` is enabled (default), users are automatically created on first login:

```json
{
  "auto_create_users": true,
  "default_levels": ["researcher"]
}
```

### Manual Provisioning

Set `auto_create_users: false` to require manual user creation. Users must exist in SimpleTuner before they can log in via external auth.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/cloud/external-auth/providers` | List configured providers |
| POST | `/api/cloud/external-auth/providers` | Create a provider |
| PATCH | `/api/cloud/external-auth/providers/{name}` | Update a provider |
| DELETE | `/api/cloud/external-auth/providers/{name}` | Delete a provider |
| GET | `/api/cloud/external-auth/providers/{name}/test` | Test provider connection |
| GET | `/api/cloud/external-auth/oidc/{provider}/start` | Start OIDC flow |
| GET | `/api/cloud/external-auth/oidc/callback` | OIDC callback |
| POST | `/api/cloud/external-auth/ldap/login` | LDAP login |
| GET | `/api/cloud/external-auth/available` | List available providers (public) |

## CLI Commands

```bash
# List providers (requires admin access)
simpletuner auth users auth-status

# External auth is managed via the web UI or API
# No dedicated CLI commands yet
```

## Architecture Notes

### OAuth State Persistence

OIDC authentication requires validating a `state` parameter to prevent CSRF attacks. SimpleTuner stores OAuth state in the database rather than in-memory to support:

- **Multi-worker deployments**: Load-balanced setups where the callback may hit a different worker than the one that initiated the flow
- **Server restarts**: State survives worker restarts during the authentication window
- **Horizontal scaling**: No sticky sessions required

State tokens expire after 10 minutes and are deleted after successful validation.

## Security Considerations

### OIDC

- Store client secrets securely (environment variables or secrets manager)
- Use HTTPS for all callback URLs
- Validate the `state` parameter to prevent CSRF attacks (handled automatically via database-backed state)
- Consider restricting allowed redirect URIs on the identity provider

### LDAP

- Use LDAPS (port 636) or StartTLS in production
- Use a dedicated service account with minimal permissions
- Rotate service account credentials regularly
- Consider using connection pooling for high-traffic deployments

### General

- External authentication bypasses local password policies
- Users authenticated externally cannot reset their password in SimpleTuner
- Audit logs record all authentication attempts including failures
- Consider session timeout settings for compliance

## Troubleshooting

### OIDC Issues

**"Invalid or expired state"**
- State tokens expire after 10 minutes
- Check clock synchronization between SimpleTuner and identity provider

**"Discovery URL unreachable"**
- Verify network connectivity to identity provider
- Check firewall rules allow outbound HTTPS

**"Invalid client_id or client_secret"**
- Re-verify credentials in identity provider
- Ensure no trailing whitespace in configuration

### LDAP Issues

**"Connection refused"**
- Check server URL and port
- Verify LDAP service is running
- Check firewall rules

**"Invalid credentials"**
- Verify bind DN format (full DN required)
- Check bind password
- Try testing with `ldapsearch` CLI

**"User not found"**
- Check user_base_dn path
- Verify user_filter syntax
- Ensure user exists and is not disabled

**"No groups returned"**
- Verify group_base_dn path
- Check group_filter syntax
- Ensure group membership attribute is correct

## Example Configurations

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
