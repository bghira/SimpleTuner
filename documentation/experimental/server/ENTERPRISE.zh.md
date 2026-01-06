# 企业部署指南

本文介绍在多用户环境中部署 SimpleTuner，涵盖认证、审批流程与配额管理。

## 1. 部署与基础设施

### 配置方式

大多数企业功能可以通过 **Web UI**（Administration 面板）或 **REST API** 配置。少数基础设施级别的设置需要配置文件或环境变量。

| 功能 | Web UI | API | 配置文件 |
|---------|--------|-----|-------------|
| OIDC/LDAP 提供方 | ✓ | ✓ | ✓ |
| 用户与角色 | ✓ | ✓ | |
| 审批规则 | ✓ | ✓ | |
| 配额 | ✓ | ✓ | |
| 通知 | ✓ | ✓ | |
| 网络绕过（可信代理） | | | ✓ |
| 后台任务轮询 | | | ✓ |
| TLS 设置 | | | ✓ |

**配置文件**（`simpletuner-enterprise.yaml` 或 `.json`）仅用于必须在启动时确定的基础设施设置。SimpleTuner 按以下顺序查找：

1. `$SIMPLETUNER_ENTERPRISE_CONFIG`（环境变量）
2. `./simpletuner-enterprise.yaml`（当前目录）
3. `~/.config/simpletuner/enterprise.yaml`
4. `/etc/simpletuner/enterprise.yaml`

配置文件支持 `${VAR}` 语法的环境变量插值。

### 快速开始清单

1.  **启动 SimpleTuner**：`simpletuner server`（本地使用可加 `--webui`）
2.  **UI 配置**：进入 Administration 面板配置用户、SSO、配额
3.  **健康检查**（生产环境）：
    *   存活检查：`GET /api/cloud/health/live`（200 OK）
    *   就绪检查：`GET /api/cloud/health/ready`（200 OK）
    *   深度检查：`GET /api/cloud/health`（包含提供方连通性）

### 网络安全与认证绕过

<details>
<summary>配置可信代理与内部网络绕过（需配置文件）</summary>

在企业环境（VPN、私有 VPC）中，可能需要信任内部流量或将认证交给网关。

**simpletuner-enterprise.yaml:**

```yaml
network:
  # 信任负载均衡头部（如 AWS ALB、Nginx）
  trust_proxy_headers: true
  trusted_proxies:
    - "10.0.0.0/8"
    - "192.168.0.0/16"

  # 可选：信任特定内部网段绕过登录
  bypass_auth_for_internal: true
  internal_networks:
    - "10.10.0.0/16"  # VPN 客户端

auth:
  # 健康检查始终不需要认证
  bypass_paths:
    - "/health"
    - "/api/cloud/health"
    - "/api/cloud/metrics/prometheus"
```

</details>

### 负载均衡与 TLS 配置

SimpleTuner 预期由上游反向代理进行 TLS 终止。

<details>
<summary>nginx 反向代理示例</summary>

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

        # WebSocket 支持（实时日志/SSE）
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

</details>

### 可观测性（Prometheus 与日志）

**指标：**
抓取 `GET /api/cloud/metrics/prometheus` 获取运维指标。
*   `simpletuner_jobs_active`: 当前队列深度
*   `simpletuner_cost_total_usd`: 成本追踪
*   `simpletuner_uptime_seconds`: 运行时长

**日志：**
设置 `SIMPLETUNER_LOG_FORMAT=json` 便于接入 Splunk/Datadog/ELK。

<details>
<summary>数据保留配置</summary>

通过环境变量设置合规要求的保留期限：

| 变量 | 默认值 | 描述 |
|----------|---------|-------------|
| `SIMPLETUNER_JOB_RETENTION_DAYS` | 90 | 已完成任务保留天数 |
| `SIMPLETUNER_AUDIT_RETENTION_DAYS` | 90 | 审计日志保留天数 |

```bash
# SOC 2 / HIPAA：保留 1 年
export SIMPLETUNER_JOB_RETENTION_DAYS=365
export SIMPLETUNER_AUDIT_RETENTION_DAYS=365

# 禁用自动清理（手动管理）
export SIMPLETUNER_JOB_RETENTION_DAYS=0
```

设置为 `0` 将禁用自动清理。清理任务每日运行。

</details>

---


## 2. 身份与访问管理（SSO）

SimpleTuner 支持 OIDC（OpenID Connect）和 LDAP，可与 Okta、Azure AD、Keycloak、Active Directory 集成。

### 配置提供方

**Web UI：** 进入 **Administration → Auth** 添加并配置。

**API：** curl 示例见 [API Cookbook](#4-api-cookbook)。

<details>
<summary>通过配置文件（IaC/GitOps）</summary>

在 `simpletuner-enterprise.yaml` 中添加：

```yaml
oidc:
  enabled: true
  provider: "okta"  # 或 "azure_ad", "google"

  client_id: "0oa1234567890abcdef"
  client_secret: "${OIDC_CLIENT_SECRET}"
  issuer_url: "https://your-org.okta.com/oauth2/default"

  scopes: ["openid", "email", "profile", "groups"]

  # 将身份提供方组映射到 SimpleTuner 角色
  role_mapping:
    claim: "groups"
    admin_groups: ["ML-Platform-Admins"]
    user_groups: ["ML-Researchers"]
```

</details>

<details>
<summary>跨 Worker 的 OAuth State 校验</summary>

在多 Worker 部署（例如负载均衡后的多 Gunicorn Worker）中，OIDC 的 OAuth state 校验必须跨 Worker 生效。SimpleTuner 通过将 OAuth state 存入数据库自动支持。

**工作机制：**

1. **State 生成**：用户发起 OIDC 登录时生成加密随机的 state token，并将提供方名、重定向 URI 与 10 分钟过期信息写入数据库。

2. **State 校验**：回调可能落到不同 Worker，系统会从数据库读取 state 并原子性消费（一次性）。

3. **清理**：过期 state 在正常运行中自动清理。

无需额外配置。OAuth state 与任务和用户共用同一数据库。

**排查 “Invalid OAuth state” 错误：**
1. 确认回调在登录发起后 10 分钟内到达
2. 确认所有 Worker 使用同一个数据库路径
3. 检查数据库写权限
4. 查看日志中是否有 “Failed to store OAuth state”

</details>

### 用户管理与角色

SimpleTuner 采用层级角色系统。用户可通过 `GET/POST /api/users` 管理。

| 角色 | 优先级 | 描述 |
|------|----------|-------------|
| **Viewer** | 10 | 只读访问任务历史与日志。 |
| **Researcher** | 20 | 标准访问。可提交任务并管理自己的 API Key。 |
| **Lead** | 30 | 可审批待处理任务并查看团队资源使用。 |
| **Admin** | 100 | 全部权限，包括用户管理与规则配置。 |

---


## 3. 治理与运维

### 审批流程

通过审批规则控制成本与资源使用，规则在提交时评估。

**流程：**
1.  用户提交任务 → 状态变为 `pending_approval`
2.  Lead 查看 `GET /api/approvals/requests`
3.  Lead 调用 `POST /.../approve` 或 `reject`
4.  任务自动进入队列或被取消

<details>
<summary>审批规则引擎</summary>

规则引擎按优先级评估任务提交，首条匹配规则触发审批要求。

**可用条件：**

| 条件 | 描述 |
|-----------|-------------|
| `cost_exceeds` | 预计成本超过阈值（USD）触发 |
| `hardware_type` | 匹配硬件类型（glob 模式） |
| `daily_jobs_exceed` | 用户每日任务数超过阈值触发 |
| `first_job` | 用户首次任务触发 |
| `config_pattern` | 匹配配置名称模式 |
| `provider` | 匹配指定提供方 |

**示例：超过 $50 的任务需要审批**

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

规则可通过 `exempt_levels` 允许某些用户免审，也可用 `applies_to_provider`/`applies_to_level` 限定范围。

</details>

<details>
<summary>邮件审批（IMAP 工作流）</summary>

对于偏好邮件流程的团队，SimpleTuner 支持通过 IMAP IDLE 的邮件回复审批。

**流程说明：**
1. 任务提交触发审批
2. 向审批人发送包含响应 token 的通知邮件
3. IMAP 处理器使用 IDLE 监听收件箱
4. 审批人回复 “approve” 或 “reject”（或 `yes`、`lgtm`、`+1`）
5. 系统解析回复并处理审批

在 **Administration → Notifications** 或 API 中配置。响应 token 24 小时过期且仅可使用一次。

</details>

### 任务队列与并发

调度器管理资源的公平使用。详见 [专用文档](../../JOB_QUEUE.md)。

*   **优先级:** Admin > Lead > Researcher > Viewer
*   **并发限制:** 全局与用户级别
    *   UI 更新：**Cloud 选项卡 → Job Queue 面板**（仅管理员）
    *   API 更新：`POST /api/queue/concurrency`，例如 `{"max_concurrent": 10, "user_max_concurrent": 3}`

### 任务状态轮询（无需 Webhook）

对于无法公开 Webhook 的安全环境，SimpleTuner 提供后台轮询器。

在 `simpletuner-enterprise.yaml` 中添加：

```yaml
background:
  job_status_polling:
    enabled: true
    interval_seconds: 30
```

该服务每 30 秒调用提供方 API，更新本地数据库并通过 SSE 向 UI 发送实时事件。

### API Key 轮换

安全管理云提供方凭据。轮换脚本与提供方细节请参见 **API Cookbook** 与 [Cloud Training](../cloud/README.md)。

---


## 4. API Cookbook

<details>
<summary>OIDC/LDAP 配置示例</summary>

**Keycloak（OIDC）：**
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

**LDAP / Active Directory：**
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
<summary>用户管理示例</summary>

**创建 Researcher：**
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

**授予自定义权限：**
```bash
curl -X POST http://localhost:8080/api/users/123/permissions \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"permission_name": "admin.approve", "granted": true}'
```

</details>

<details>
<summary>凭据管理</summary>

SimpleTuner 提供 API 凭据的生命周期管理，用于跟踪、轮换与审计。

**凭据解析顺序：** 提交任务时，先检查用户级凭据，再回退到全局凭据（环境变量）。

| 场景 | 用户级 | 全局 | 行为 |
|----------|--------|--------|----------|
| **组织共享 Key** | ❌ | ✅ | 所有人共享组织 API Key |
| **BYOK** | ✅ | ❌ | 用户提供自己的 Key |
| **混合** | 部分 | ✅ | 有 Key 的用户用自己的，其他用全局 |

**轮换：** **Admin > Auth** → 用户 → **Manage Credentials** → **Rotate**。超过 90 天的凭据会显示警告标记。

</details>

#### 外部编排 {#external-orchestration-airflow}

<details>
<summary>Airflow 示例</summary>

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


## 5. 故障排查

**健康检查失败**
*   `503 Service Unavailable`：检查数据库连接。
*   `Degraded`：通常表示某个可选组件（如云提供方 API）不可达或未配置。

**认证问题**
*   **OIDC 重定向循环：** 确认 `issuer_url` 完全匹配（尾部斜杠很重要）。
*   **内部认证绕过：** 查看日志中是否有 “Auth bypassed for IP...”，并确认负载均衡传递正确的 `X-Real-IP`。

**任务更新停滞**
*   若 Webhook 被阻止，确保在 `simpletuner-enterprise.yaml` 中启用 **Job Status Polling**。
*   查看 `GET /api/cloud/metrics/prometheus` 中的 `simpletuner_jobs_active`。

**指标缺失**
*   确认 Prometheus 抓取 `/api/cloud/metrics/prometheus` 而不仅是 `/metrics`。

---


## 6. 组织与团队配额

SimpleTuner 支持层级组织与团队，以及上限式配额（ceiling）。

### 层级结构

```
Organization (quota ceiling)
    └── Team (quota ceiling, bounded by org)
         └── User (limit, bounded by team and org)
```

### 上限模型

配额使用上限模型：
- **组织配额**：对所有成员的绝对上限
- **团队配额**：团队成员上限（不能超过组织上限）
- **用户/级别配额**：具体限制（受团队与组织上限约束）

**示例：**
- 组织上限：100 并发任务
- 团队上限：20 并发任务
- 用户限制：50 并发任务 → **实际为 20**（受团队上限限制）

**执行规则：**
- 团队配额在设置时校验：超过组织上限返回 HTTP 400
- 用户配额在运行时校验：有效上限为用户/团队/组织的最小值
- 下调组织上限不会自动调整已有团队上限（需管理员手动更新）

<details>
<summary>API 示例</summary>

**创建组织：**
```bash
curl -X POST http://localhost:8080/api/orgs \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"name": "ML Research", "slug": "ml-research"}'
```

**设置组织配额上限：**
```bash
curl -X POST http://localhost:8080/api/orgs/1/quotas \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"quota_type": "concurrent_jobs", "limit_value": 100, "action": "block"}'
```

**创建团队：**
```bash
curl -X POST http://localhost:8080/api/orgs/1/teams \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"name": "NLP Team", "slug": "nlp"}'
```

**将用户加入团队：**
```bash
curl -X POST http://localhost:8080/api/orgs/1/teams/1/members \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"user_id": 123, "role": "member"}'
```

</details>

### 配额与成本上限动作

当达到配额或成本上限时，系统按照 `action` 决定行为：

| 动作 | 行为 |
|--------|----------|
| `warn` | 任务继续，但在日志/UI 中提示警告 |
| `block` | 拒绝任务提交 |
| `notify` | 任务继续，通知管理员 |

<details>
<summary>成本上限配置</summary>

成本上限可在 **Cloud 选项卡 → Settings** 或通过 API 配置：

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

查询状态：`GET /api/cloud/metrics/cost-limit-status`

</details>

---


## 7. 限制

### 工作流/流水线任务（DAG）

SimpleTuner 不支持任务依赖或多步骤工作流（一个任务输出作为另一个任务输入）。每个云任务都是独立的。

**建议：** 使用 Airflow、Prefect、Dagster 等外部编排工具通过 REST API 串联任务。见上方 [Airflow 示例](#external-orchestration-airflow)。

### 训练任务恢复

不支持恢复中断/失败/取消的训练任务。云任务不会自动从检查点恢复。

**替代方案：**
- 配置频繁的 HuggingFace Hub 推送（`--push_checkpoints_to_hub`）保存中间状态
- 下载输出并作为新任务的起点
- 关键任务可拆分为多个较短的训练段

<details>
<summary>UI 功能参考</summary>

| 功能 | UI 位置 | API |
|---------|-------------|-----|
| 组织与团队 | Administration → Orgs | `/api/orgs` |
| 配额 | Administration → Quotas | `/api/orgs/{id}/quotas` |
| OIDC/LDAP | Administration → Auth | `/api/cloud/external-auth/providers` |
| 用户 | Administration → Users | `/api/users` |
| 审计日志 | Sidebar → Audit Log | `/api/audit` |
| 队列 | Cloud 选项卡 → Job Queue | `/api/queue/concurrency` |
| 审批 | Administration → Approvals | `/api/approvals/requests` |

当未配置认证（单用户模式）或用户具备管理员权限时才会显示 Administration 区域。

</details>

<details>
<summary>企业版引导流程</summary>

Admin 面板包含引导式流程，按顺序协助配置认证、组织、团队、配额与凭据。

| 步骤 | 功能 |
|------|---------|
| 1 | 认证（OIDC/LDAP） |
| 2 | 组织 |
| 3 | 团队 |
| 4 | 配额 |
| 5 | 凭据 |

每一步可完成或跳过，状态保存在浏览器 localStorage。

</details>

---


## 8. 通知系统

SimpleTuner 提供多通道通知，用于任务状态、审批、配额与系统事件。

| 渠道 | 用途 |
|---------|----------|
| **Email** | 审批请求、任务完成（SMTP/IMAP） |
| **Webhook** | CI/CD 集成（JSON + HMAC 签名） |
| **Slack** | 团队通知（Incoming Webhook） |

通过 **Administration → Notifications** 或 API 配置。

<details>
<summary>事件类型</summary>

| 分类 | 事件 |
|----------|--------|
| Approval | `approval.required`, `approval.granted`, `approval.rejected`, `approval.expired` |
| Job | `job.submitted`, `job.started`, `job.completed`, `job.failed`, `job.cancelled` |
| Quota | `quota.warning`, `quota.exceeded`, `cost.warning`, `cost.exceeded` |
| System | `system.provider_error`, `system.provider_degraded`, `system.webhook_failure` |
| Auth | `auth.login_failure`, `auth.new_device` |

</details>

<details>
<summary>渠道配置示例</summary>

**Email：**
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

**Slack：**
```bash
curl -X POST http://localhost:8080/api/cloud/notifications/channels \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "channel_type": "slack",
    "name": "Training Alerts",
    "webhook_url": "https://hooks.slack.com/services/T00/B00/xxxx"
  }'
```

**Webhook：** 负载使用 HMAC-SHA256 签名（`X-SimpleTuner-Signature` 头）。

</details>

---


## 9. 资源规则

资源规则通过 glob 模式对配置、硬件类型与输出路径进行细粒度访问控制。

| 类型 | 示例模式 |
|------|----------------|
| `config` | `team-x-*`, `production-*` |
| `hardware` | `gpu-a100*`, `*-80gb` |
| `provider` | `replicate`, `runpod` |

规则使用 **allow/deny** 动作，采用“最宽松规则优先”的逻辑。通过 **Administration → Rules** 配置。

<details>
<summary>规则示例</summary>

**团队隔离：** Researcher 只能使用以 “team-x-” 开头的配置
```
Level: researcher
Rules:
  - config: "team-x-*" → allow
  - config: "*" → deny
```

**硬件限制：** Researcher 限制为 T4/V100，Lead 可使用任意硬件
```
Level: researcher → hardware: "gpu-t4*" allow, "gpu-v100*" allow
Level: lead → hardware: "*" allow
```

</details>

---


## 10. 权限矩阵

<details>
<summary>完整权限矩阵</summary>

### 任务权限

| 权限 | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `job.submit` | | ✓ | ✓ | ✓ |
| `job.view.own` | ✓ | ✓ | ✓ | ✓ |
| `job.view.all` | | | ✓ | ✓ |
| `job.cancel.own` | | ✓ | ✓ | ✓ |
| `job.cancel.all` | | | | ✓ |
| `job.priority.high` | | | ✓ | ✓ |
| `job.bypass.queue` | | | | ✓ |
| `job.bypass.approval` | | | | ✓ |

### 配置权限

| 权限 | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `config.view` | ✓ | ✓ | ✓ | ✓ |
| `config.create` | | ✓ | ✓ | ✓ |
| `config.edit.own` | | ✓ | ✓ | ✓ |
| `config.edit.all` | | | | ✓ |

### 管理权限

| 权限 | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `admin.users` | | | | ✓ |
| `admin.approve` | | | ✓ | ✓ |
| `admin.audit` | | | ✓ | ✓ |
| `admin.config` | | | | ✓ |
| `queue.approve` | | | ✓ | ✓ |
| `queue.manage` | | | | ✓ |

### 组织/团队权限

| 权限 | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `org.view` | | | ✓ | ✓ |
| `org.create` | | | | ✓ |
| `team.view` | | | ✓ | ✓ |
| `team.create` | | | ✓ | ✓ |
| `team.manage.members` | | | ✓ | ✓ |

</details>

**权限覆盖：** 可在 **Administration → Users → Permission Overrides** 为用户单独授予或拒绝权限。
