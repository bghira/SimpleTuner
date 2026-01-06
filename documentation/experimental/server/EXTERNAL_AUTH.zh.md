# 外部认证（OIDC 与 LDAP）

SimpleTuner 支持外部身份提供方的单点登录（SSO）。用户可以通过 OIDC 提供方（Keycloak、Auth0、Okta、Azure AD、Google）或 LDAP/Active Directory 进行认证。

## 概览

外部认证提供：
- **单点登录**：用户使用现有企业凭据登录
- **自动开通**：首次登录自动创建用户
- **级别映射**：将外部组/角色映射到 SimpleTuner 访问级别
- **多提供方**：可同时配置多个提供方

## 配置 OIDC

### 支持的提供方

- Keycloak
- Auth0
- Okta
- Azure Active Directory
- Google Workspace
- 任意 OpenID Connect 兼容提供方

测试通过的组合包括 OpenLDAP + Dex、Keycloak 与 Auth0。

### 通过 Web UI 配置

1. 进入 **Administration > Manage Users > Auth Providers**
2. 点击 **Add Provider** 并选择 **OIDC Provider**
3. 填写提供方信息：
   - **Provider Name**：唯一标识（例如 `corporate-sso`）
   - **Type**：OIDC
   - **Issuer URL**：OIDC 发现端点（例如 `https://auth.example.com/.well-known/openid-configuration`）
   - **Client ID**：身份提供方分配
   - **Client Secret**：身份提供方分配
   - **Scopes**：通常为 `openid profile email`

### 通过 API 配置

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

### OIDC 回调 URL

在身份提供方配置以下回调地址：

```
https://your-simpletuner-host/api/cloud/external-auth/oidc/{provider-name}/callback
```

例如提供方名为 `corporate-sso` 时：
```
https://simpletuner.example.com/api/cloud/external-auth/oidc/corporate-sso/callback
```

### OIDC 登录流程

1. 用户在登录页点击 “Login with SSO”
2. 浏览器跳转至身份提供方
3. 用户使用企业凭据认证
4. 身份提供方回调到 SimpleTuner
5. SimpleTuner 创建/更新本地用户并建立会话

## 配置 LDAP

### 支持的配置

- Microsoft Active Directory
- OpenLDAP
- 389 Directory Server
- 任意 LDAPv3 兼容服务器

测试通过的组合包括 FreeIPA 和 OpenLDAP。

### 通过 Web UI 配置

1. 进入 **Administration > Manage Users > Auth Providers**
2. 点击 **Add Provider** 并选择 **LDAP / Active Directory**
3. 填写 LDAP 信息：
   - **Provider Name**：唯一标识（例如 `company-ldap`）
   - **Type**：LDAP
   - **Server URL**：LDAP 服务器地址（例如 `ldap://ldap.example.com:389` 或 `ldaps://ldap.example.com:636`）
   - **Bind DN**：查询用服务账号 DN
   - **Bind Password**：服务账号密码
   - **User Base DN**：用户搜索基准 DN
   - **User Filter**：用户搜索 LDAP 过滤器
   - **Group Base DN**：组搜索基准 DN（可选）
   - **Group Filter**：组搜索 LDAP 过滤器（可选）

### 通过 API 配置

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

### LDAP 登录

LDAP 登录为用户名/密码方式，用户在 SimpleTuner 登录页面输入 LDAP 凭据。

```bash
# API 登录
curl -X POST http://localhost:8001/api/cloud/external-auth/ldap/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "jsmith",
    "password": "ldap-password"
  }'
```

## 级别映射

将外部组/角色映射到 SimpleTuner 访问级别：

```json
{
  "level_mapping": {
    "admin": ["admin"],
    "power-users": ["advanced"],
    "default": ["researcher"]
  }
}
```

OIDC 使用 `groups` 或 `roles` claim 中的值。
LDAP 使用 LDAP 组 DN。

## 用户开通

### 自动开通

当 `auto_create_users` 启用（默认）时，用户首次登录会自动创建：

```json
{
  "auto_create_users": true,
  "default_levels": ["researcher"]
}
```

### 手动开通

设置 `auto_create_users: false` 后，用户必须先在 SimpleTuner 中手动创建，才能通过外部认证登录。

## API 端点

| 方法 | 端点 | 描述 |
|--------|----------|-------------|
| GET | `/api/cloud/external-auth/providers` | 列出已配置提供方 |
| POST | `/api/cloud/external-auth/providers` | 创建提供方 |
| PATCH | `/api/cloud/external-auth/providers/{name}` | 更新提供方 |
| DELETE | `/api/cloud/external-auth/providers/{name}` | 删除提供方 |
| GET | `/api/cloud/external-auth/providers/{name}/test` | 测试提供方连接 |
| GET | `/api/cloud/external-auth/oidc/{provider}/start` | 启动 OIDC 流程 |
| GET | `/api/cloud/external-auth/oidc/callback` | OIDC 回调 |
| POST | `/api/cloud/external-auth/ldap/login` | LDAP 登录 |
| GET | `/api/cloud/external-auth/available` | 列出可用提供方（公开） |

## CLI 命令

```bash
# 列出提供方（需要管理员权限）
simpletuner auth users auth-status

# 外部认证通过 Web UI 或 API 管理
# 暂无专用 CLI 命令
```

## 架构说明

### OAuth State 持久化

OIDC 认证需要验证 `state` 参数以防 CSRF。SimpleTuner 将 OAuth state 存储在数据库中，而非内存，以支持：

- **多 Worker 部署**：回调可能到达不同的 Worker
- **服务器重启**：认证窗口期间仍可验证
- **水平扩展**：无需黏性会话

State Token 10 分钟过期，并在验证成功后删除。

## 安全注意事项

### OIDC

- 安全存储 client secret（环境变量或密钥管理）
- 回调 URL 使用 HTTPS
- 验证 `state` 以防 CSRF（DB 中的 state 自动处理）
- 建议在身份提供方限制允许的重定向 URI

### LDAP

- 生产环境使用 LDAPS（636）或 StartTLS
- 使用最小权限服务账号
- 定期轮换服务账号凭据
- 高流量部署建议使用连接池

### 通用

- 外部认证会绕过本地密码策略
- 外部认证用户无法在 SimpleTuner 重置密码
- 所有认证尝试（含失败）都会写入审计日志
- 视合规要求配置会话超时

## 故障排查

### OIDC 问题

**“Invalid or expired state”**
- State Token 10 分钟过期
- 检查 SimpleTuner 与身份提供方的时钟同步

**“Discovery URL unreachable”**
- 验证到身份提供方的网络连通性
- 检查出站 HTTPS 防火墙规则

**“Invalid client_id or client_secret”**
- 重新核对身份提供方凭据
- 确保配置中无多余空格

### LDAP 问题

**“Connection refused”**
- 检查服务器 URL 和端口
- 确认 LDAP 服务运行中
- 检查防火墙规则

**“Invalid credentials”**
- 确认 bind DN 格式（必须是完整 DN）
- 检查 bind 密码
- 使用 `ldapsearch` 测试

**“User not found”**
- 检查 user_base_dn 路径
- 检查 user_filter 语法
- 确认用户存在且未禁用

**“No groups returned”**
- 检查 group_base_dn 路径
- 检查 group_filter 语法
- 确认组成员属性正确

## 示例配置

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

### Azure AD（OIDC）

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
