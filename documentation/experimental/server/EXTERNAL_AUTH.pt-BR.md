# Autenticacao externa (OIDC e LDAP)

O SimpleTuner suporta provedores de identidade externos para single sign-on (SSO). Usuarios podem autenticar via provedores OIDC (Keycloak, Auth0, Okta, Azure AD, Google) ou LDAP/Active Directory.

## Visao geral

A autenticacao externa fornece:
- **Single Sign-On**: Usuarios autenticam com credenciais corporativas existentes
- **Auto-provisionamento**: Novos usuarios sao criados automaticamente no primeiro login
- **Mapeamento de nivel**: Mapear grupos/papeis externos para niveis de acesso do SimpleTuner
- **Multiplos provedores**: Configurar varios provedores simultaneamente

## Configurando OIDC

### Provedores suportados

- Keycloak
- Auth0
- Okta
- Azure Active Directory
- Google Workspace
- Qualquer provedor compativel com OpenID Connect

Testes foram feitos com OpenLDAP + Dex, Keycloak e Auth0.

### Configuracao via Web UI

1. Navegue para **Administration > Manage Users > Auth Providers**
2. Clique em **Add Provider** e selecione **OIDC Provider**
3. Insira detalhes do provedor:
   - **Provider Name**: Identificador unico (ex.: `corporate-sso`)
   - **Type**: OIDC
   - **Issuer URL**: Endpoint de discovery OIDC (ex.: `https://auth.example.com/.well-known/openid-configuration`)
   - **Client ID**: Do provedor de identidade
   - **Client Secret**: Do provedor de identidade
   - **Scopes**: Geralmente `openid profile email`

### Configuracao via API

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

Configure seu provedor de identidade com a URL de callback:

```
https://your-simpletuner-host/api/cloud/external-auth/oidc/{provider-name}/callback
```

Por exemplo, se seu provedor se chama `corporate-sso`:
```
https://simpletuner.example.com/api/cloud/external-auth/oidc/corporate-sso/callback
```

### Fluxo de login OIDC

1. O usuario clica em "Login with SSO" na pagina de login
2. O navegador redireciona para o provedor de identidade
3. O usuario autentica com suas credenciais corporativas
4. O provedor de identidade redireciona de volta para o callback do SimpleTuner
5. O SimpleTuner cria/atualiza o usuario local e estabelece a sessao

## Configurando LDAP

### Configuracoes suportadas

- Microsoft Active Directory
- OpenLDAP
- 389 Directory Server
- Qualquer servidor compativel com LDAPv3

Testes foram feitos com FreeIPA e OpenLDAP.

### Configuracao via Web UI

1. Navegue para **Administration > Manage Users > Auth Providers**
2. Clique em **Add Provider** e selecione **LDAP / Active Directory**
3. Insira detalhes do LDAP:
   - **Provider Name**: Identificador unico (ex.: `company-ldap`)
   - **Type**: LDAP
   - **Server URL**: Endereco do servidor LDAP (ex.: `ldap://ldap.example.com:389` ou `ldaps://ldap.example.com:636`)
   - **Bind DN**: DN da conta de servico para consultas
   - **Bind Password**: Senha da conta de servico
   - **User Base DN**: Onde buscar usuarios
   - **User Filter**: Filtro LDAP para encontrar usuarios
   - **Group Base DN**: Onde buscar grupos (opcional)
   - **Group Filter**: Filtro LDAP para encontrar grupos (opcional)

### Configuracao via API

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

Login LDAP e baseado em usuario/senha. Usuarios inserem suas credenciais LDAP na pagina de login do SimpleTuner.

```bash
# API login
curl -X POST http://localhost:8001/api/cloud/external-auth/ldap/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "jsmith",
    "password": "ldap-password"
  }'
```

## Mapeamento de niveis

Mapeie grupos/papeis externos para niveis de acesso do SimpleTuner:

```json
{
  "level_mapping": {
    "admin": ["admin"],
    "power-users": ["advanced"],
    "default": ["researcher"]
  }
}
```

Para OIDC, o mapeamento usa papeis/grupos do claim `groups` ou `roles`.
Para LDAP, o mapeamento usa DNs de grupos LDAP.

## Provisionamento de usuarios

### Auto-provisionamento

Quando `auto_create_users` esta habilitado (padrao), usuarios sao criados automaticamente no primeiro login:

```json
{
  "auto_create_users": true,
  "default_levels": ["researcher"]
}
```

### Provisionamento manual

Defina `auto_create_users: false` para exigir criacao manual de usuarios. Usuarios devem existir no SimpleTuner antes de poderem fazer login via auth externa.

## Endpoints de API

| Metodo | Endpoint | Descricao |
|--------|----------|-------------|
| GET | `/api/cloud/external-auth/providers` | Listar provedores configurados |
| POST | `/api/cloud/external-auth/providers` | Criar um provedor |
| PATCH | `/api/cloud/external-auth/providers/{name}` | Atualizar um provedor |
| DELETE | `/api/cloud/external-auth/providers/{name}` | Deletar um provedor |
| GET | `/api/cloud/external-auth/providers/{name}/test` | Testar conexao do provedor |
| GET | `/api/cloud/external-auth/oidc/{provider}/start` | Iniciar fluxo OIDC |
| GET | `/api/cloud/external-auth/oidc/callback` | Callback OIDC |
| POST | `/api/cloud/external-auth/ldap/login` | Login LDAP |
| GET | `/api/cloud/external-auth/available` | Listar provedores disponiveis (publico) |

## Comandos CLI

```bash
# List providers (requires admin access)
simpletuner auth users auth-status

# External auth is managed via the web UI or API
# No dedicated CLI commands yet
```

## Notas de arquitetura

### Persistencia de estado OAuth

A autenticacao OIDC exige validar o parametro `state` para prevenir ataques CSRF. O SimpleTuner armazena o estado OAuth no banco de dados em vez de memoria para suportar:

- **Deploys multi-worker**: Setups com load balancer onde o callback pode cair em um worker diferente
- **Reinicios do servidor**: O estado sobrevive a reinicios durante a janela de autenticacao
- **Escala horizontal**: Sem necessidade de sticky sessions

Tokens de estado expiram apos 10 minutos e sao deletados apos validacao bem-sucedida.

## Consideracoes de seguranca

### OIDC

- Armazene client secrets com seguranca (variaveis de ambiente ou secrets manager)
- Use HTTPS para todas as URLs de callback
- Valide o parametro `state` para evitar ataques CSRF (tratado automaticamente via estado no banco)
- Considere restringir URIs de redirect permitidas no provedor de identidade

### LDAP

- Use LDAPS (porta 636) ou StartTLS em producao
- Use uma conta de servico dedicada com permissoes minimas
- Rotacione credenciais da conta de servico regularmente
- Considere usar connection pooling para deploys de alto trafego

### Geral

- Autenticacao externa contorna politicas de senha locais
- Usuarios autenticados externamente nao podem redefinir senha no SimpleTuner
- Audit logs registram todas as tentativas de autenticacao, incluindo falhas
- Considere configuracoes de timeout de sessao para compliance

## Solucao de problemas

### Problemas OIDC

**"Invalid or expired state"**
- Tokens de estado expiram apos 10 minutos
- Verifique sincronizacao de relogio entre SimpleTuner e provedor de identidade

**"Discovery URL unreachable"**
- Verifique conectividade de rede com o provedor de identidade
- Verifique se o firewall permite HTTPS de saida

**"Invalid client_id or client_secret"**
- Revalide credenciais no provedor de identidade
- Garanta que nao ha espacos em branco na configuracao

### Problemas LDAP

**"Connection refused"**
- Verifique URL e porta do servidor
- Verifique se o servico LDAP esta rodando
- Verifique regras de firewall

**"Invalid credentials"**
- Verifique formato do bind DN (DN completo exigido)
- Verifique senha do bind
- Tente testar com `ldapsearch` CLI

**"User not found"**
- Verifique o caminho user_base_dn
- Verifique a sintaxe do user_filter
- Garanta que o usuario existe e nao esta desabilitado

**"No groups returned"**
- Verifique o caminho group_base_dn
- Verifique a sintaxe do group_filter
- Garanta que o atributo de membership do grupo esta correto

## Exemplos de configuracao

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
