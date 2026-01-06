# External Authentication (OIDC & LDAP)

SimpleTuner single sign-on (SSO) के लिए external identity providers सपोर्ट करता है। Users OIDC providers (Keycloak, Auth0, Okta, Azure AD, Google) या LDAP/Active Directory के माध्यम से authenticate कर सकते हैं।

## ओवरव्यू

External authentication प्रदान करता है:
- **Single Sign-On**: Users अपने मौजूदा corporate credentials से authenticate करते हैं
- **Auto-provisioning**: पहली login पर नए users अपने आप बन जाते हैं
- **Level mapping**: external groups/roles को SimpleTuner access levels से मैप करना
- **Multiple providers**: एक साथ कई providers कॉन्फ़िगर करना

## OIDC सेटअप

### Supported Providers

- Keycloak
- Auth0
- Okta
- Azure Active Directory
- Google Workspace
- कोई भी OpenID Connect compliant provider

Testing OpenLDAP + Dex, Keycloak, और Auth0 के साथ किया गया है।

### Web UI के जरिए कॉन्फ़िगरेशन

1. **Administration > Manage Users > Auth Providers** पर जाएँ
2. **Add Provider** पर क्लिक करें और **OIDC Provider** चुनें
3. provider विवरण भरें:
   - **Provider Name**: Unique identifier (जैसे `corporate-sso`)
   - **Type**: OIDC
   - **Issuer URL**: OIDC discovery endpoint (जैसे `https://auth.example.com/.well-known/openid-configuration`)
   - **Client ID**: आपके identity provider से
   - **Client Secret**: आपके identity provider से
   - **Scopes**: आमतौर पर `openid profile email`

### API के जरिए कॉन्फ़िगरेशन

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

अपने identity provider में callback URL कॉन्फ़िगर करें:

```
https://your-simpletuner-host/api/cloud/external-auth/oidc/{provider-name}/callback
```

उदाहरण के लिए, यदि आपका provider `corporate-sso` नाम का है:
```
https://simpletuner.example.com/api/cloud/external-auth/oidc/corporate-sso/callback
```

### OIDC Login Flow

1. User login page पर "Login with SSO" क्लिक करता है
2. Browser identity provider पर redirect होता है
3. User अपने corporate credentials से authenticate करता है
4. Identity provider SimpleTuner callback पर वापस redirect करता है
5. SimpleTuner local user को create/update करता है और session बनाता है

## LDAP सेटअप

### Supported Configurations

- Microsoft Active Directory
- OpenLDAP
- 389 Directory Server
- कोई भी LDAPv3 compliant server

Testing FreeIPA और OpenLDAP के साथ किया गया है।

### Web UI के जरिए कॉन्फ़िगरेशन

1. **Administration > Manage Users > Auth Providers** पर जाएँ
2. **Add Provider** पर क्लिक करें और **LDAP / Active Directory** चुनें
3. LDAP विवरण भरें:
   - **Provider Name**: Unique identifier (जैसे `company-ldap`)
   - **Type**: LDAP
   - **Server URL**: LDAP server address (जैसे `ldap://ldap.example.com:389` या `ldaps://ldap.example.com:636`)
   - **Bind DN**: queries के लिए service account DN
   - **Bind Password**: service account पासवर्ड
   - **User Base DN**: users ढूंढने का base DN
   - **User Filter**: users ढूंढने के लिए LDAP filter
   - **Group Base DN**: groups ढूंढने का base DN (optional)
   - **Group Filter**: groups ढूंढने के लिए LDAP filter (optional)

### API के जरिए कॉन्फ़िगरेशन

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

LDAP login username/password आधारित है। Users SimpleTuner login page पर अपने LDAP credentials दर्ज करते हैं।

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

External groups/roles को SimpleTuner access levels से मैप करें:

```json
{
  "level_mapping": {
    "admin": ["admin"],
    "power-users": ["advanced"],
    "default": ["researcher"]
  }
}
```

OIDC के लिए, mapping `groups` या `roles` claim से roles/groups उपयोग करती है।
LDAP के लिए, mapping LDAP group DNs उपयोग करती है।

## User Provisioning

### Auto-provisioning

जब `auto_create_users` सक्षम होता है (डिफ़ॉल्ट), तो users पहली login पर अपने आप बन जाते हैं:

```json
{
  "auto_create_users": true,
  "default_levels": ["researcher"]
}
```

### Manual Provisioning

`auto_create_users: false` सेट करें ताकि manual user creation की आवश्यकता हो। External auth से लॉगिन करने से पहले users SimpleTuner में मौजूद होने चाहिए।

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/cloud/external-auth/providers` | configured providers सूचीबद्ध करें |
| POST | `/api/cloud/external-auth/providers` | provider बनाएं |
| PATCH | `/api/cloud/external-auth/providers/{name}` | provider अपडेट करें |
| DELETE | `/api/cloud/external-auth/providers/{name}` | provider हटाएँ |
| GET | `/api/cloud/external-auth/providers/{name}/test` | provider connection टेस्ट करें |
| GET | `/api/cloud/external-auth/oidc/{provider}/start` | OIDC flow शुरू करें |
| GET | `/api/cloud/external-auth/oidc/callback` | OIDC callback |
| POST | `/api/cloud/external-auth/ldap/login` | LDAP login |
| GET | `/api/cloud/external-auth/available` | available providers सूचीबद्ध करें (public) |

## CLI Commands

```bash
# List providers (requires admin access)
simpletuner auth users auth-status

# External auth is managed via the web UI or API
# No dedicated CLI commands yet
```

## Architecture Notes

### OAuth State Persistence

OIDC authentication के लिए CSRF attacks से बचने हेतु `state` parameter validate करना आवश्यक है। SimpleTuner OAuth state को in-memory के बजाय database में स्टोर करता है, ताकि:

- **Multi-worker deployments**: load-balanced setups जहाँ callback किसी अलग worker पर आ सकता है
- **Server restarts**: authentication window के दौरान state worker restarts के बाद भी बनी रहे
- **Horizontal scaling**: sticky sessions की जरूरत नहीं

State tokens 10 मिनट में expire हो जाते हैं और सफल validation के बाद delete कर दिए जाते हैं।

## Security Considerations

### OIDC

- client secrets को सुरक्षित रखें (environment variables या secrets manager)
- सभी callback URLs के लिए HTTPS उपयोग करें
- CSRF attacks रोकने के लिए `state` parameter validate करें (database-backed state के जरिए स्वचालित)
- identity provider पर allowed redirect URIs सीमित करने पर विचार करें

### LDAP

- प्रोडक्शन में LDAPS (port 636) या StartTLS उपयोग करें
- न्यूनतम permissions के साथ dedicated service account उपयोग करें
- service account credentials नियमित रूप से rotate करें
- high-traffic deployments के लिए connection pooling का विचार करें

### General

- external authentication local password policies को bypass करता है
- externally authenticated users SimpleTuner में अपना पासवर्ड reset नहीं कर सकते
- audit logs सभी authentication attempts (failures सहित) रिकॉर्ड करते हैं
- compliance के लिए session timeout settings पर विचार करें

## Troubleshooting

### OIDC Issues

**"Invalid or expired state"**
- State tokens 10 मिनट में expire हो जाते हैं
- SimpleTuner और identity provider के बीच clock synchronization जांचें

**"Discovery URL unreachable"**
- identity provider तक network connectivity verify करें
- firewall rules में outbound HTTPS अनुमति दें

**"Invalid client_id or client_secret"**
- identity provider में credentials फिर से जांचें
- configuration में trailing whitespace न हो

### LDAP Issues

**"Connection refused"**
- server URL और port जांचें
- LDAP service चल रही है या नहीं जांचें
- firewall rules जांचें

**"Invalid credentials"**
- bind DN format verify करें (पूर्ण DN आवश्यक)
- bind password जांचें
- `ldapsearch` CLI से टेस्ट करें

**"User not found"**
- user_base_dn path जांचें
- user_filter syntax verify करें
- user मौजूद है और disabled नहीं है

**"No groups returned"**
- group_base_dn path जांचें
- group_filter syntax verify करें
- group membership attribute सही है या नहीं देखें

## उदाहरण कॉन्फ़िगरेशन

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
