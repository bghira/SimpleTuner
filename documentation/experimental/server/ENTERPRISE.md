# Enterprise Guide

This document covers deploying SimpleTuner in multi-user environments with authentication, approval workflows, and quota management.

## 1. Deployment & Infrastructure

### Configuration Methods

Most enterprise features can be configured via the **Web UI** (Administration panel) or **REST API**. A few infrastructure-level settings require a config file or environment variables.

| Feature | Web UI | API | Config File |
|---------|--------|-----|-------------|
| OIDC/LDAP providers | ✓ | ✓ | ✓ |
| Users & roles | ✓ | ✓ | |
| Approval rules | ✓ | ✓ | |
| Quotas | ✓ | ✓ | |
| Notifications | ✓ | ✓ | |
| Network bypass (trusted proxies) | | | ✓ |
| Background job polling | | | ✓ |
| TLS settings | | | ✓ |

**Config file** (`simpletuner-enterprise.yaml` or `.json`) is only needed for infrastructure settings that must be known at startup. SimpleTuner searches these locations:

1. `$SIMPLETUNER_ENTERPRISE_CONFIG` (environment variable)
2. `./simpletuner-enterprise.yaml` (current directory)
3. `~/.config/simpletuner/enterprise.yaml`
4. `/etc/simpletuner/enterprise.yaml`

The file supports environment variable interpolation with `${VAR}` syntax.

### Quick Start Checklist

1.  **Start SimpleTuner**: `simpletuner server` (or `--webui` for local use)
2.  **Configure via UI**: Navigate to Administration panel to set up users, SSO, quotas
3.  **Health Checks** (for production):
    *   Liveness: `GET /api/cloud/health/live` (200 OK)
    *   Readiness: `GET /api/cloud/health/ready` (200 OK)
    *   Deep Check: `GET /api/cloud/health` (includes provider connectivity)

### Network Security & Authentication Bypass

<details>
<summary>Configuring trusted proxies and internal network bypass (config file required)</summary>

In corporate environments (VPNs, private VPCs), you may want to trust internal traffic or offload authentication to a gateway.

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

### Load Balancer & TLS Configuration

SimpleTuner expects an upstream reverse proxy for TLS termination.

<details>
<summary>nginx reverse proxy example</summary>

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

### Observability (Prometheus & Logging)

**Metrics:**
Scrape `GET /api/cloud/metrics/prometheus` for operational insights.
*   `simpletuner_jobs_active`: Current queue depth.
*   `simpletuner_cost_total_usd`: Spend tracking.
*   `simpletuner_uptime_seconds`: Availability.

**Logging:**
Set `SIMPLETUNER_LOG_FORMAT=json` for ingestion into Splunk/Datadog/ELK.

<details>
<summary>Data Retention Configuration</summary>

Configure retention periods for compliance requirements via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMPLETUNER_JOB_RETENTION_DAYS` | 90 | Days to retain completed job records |
| `SIMPLETUNER_AUDIT_RETENTION_DAYS` | 90 | Days to retain audit log entries |

```bash
# SOC 2 / HIPAA: 1 year retention
export SIMPLETUNER_JOB_RETENTION_DAYS=365
export SIMPLETUNER_AUDIT_RETENTION_DAYS=365

# Disable automatic cleanup (manual management)
export SIMPLETUNER_JOB_RETENTION_DAYS=0
```

Setting to `0` disables automatic cleanup. Cleanup runs daily.

</details>

---


## 2. Identity & Access Management (SSO)

SimpleTuner supports OIDC (OpenID Connect) and LDAP for SSO with Okta, Azure AD, Keycloak, and Active Directory.

### Configuring Providers

**Via Web UI:** Navigate to **Administration → Auth** to add and configure providers.

**Via API:** See the [API Cookbook](#4-api-cookbook) for curl examples.

<details>
<summary>Via config file (for IaC/GitOps workflows)</summary>

Add to your `simpletuner-enterprise.yaml`:

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
<summary>Cross-Worker OAuth State Validation</summary>

When using OIDC authentication in multi-worker deployments (e.g., behind a load balancer with multiple Gunicorn workers), OAuth state validation must work across all workers. SimpleTuner handles this automatically by storing OAuth state in the database.

**How it works:**

1. **State Generation**: When a user initiates OIDC login, a cryptographically random state token is generated and stored in the database with the provider name, redirect URI, and a 10-minute expiration.

2. **State Validation**: When the callback arrives (potentially to a different worker), the state is looked up and atomically consumed (single-use).

3. **Cleanup**: Expired states are automatically purged during normal operations.

No additional configuration is needed. OAuth state storage uses the same database as jobs and users.

**Troubleshooting "Invalid OAuth state" errors:**
1. Check if the callback arrived within 10 minutes of login initiation
2. Verify all workers share the same database path
3. Check database write permissions
4. Look for "Failed to store OAuth state" errors in logs

</details>

### User Management & Roles

SimpleTuner uses a hierarchical role system. Users can be managed via `GET/POST /api/users`.

| Role | Priority | Description |
|------|----------|-------------|
| **Viewer** | 10 | Read-only access to job history and logs. |
| **Researcher** | 20 | Standard access. Can submit jobs and manage their own API keys. |
| **Lead** | 30 | Can approve pending jobs and view team resource usage. |
| **Admin** | 100 | Full system access, including user management and rule configuration. |

---


## 3. Governance & Operations

### Approval Workflows

Control costs and resource usage by requiring approvals for specific criteria. Rules are evaluated at submission time.

**Workflow:**
1.  User submits job -> Status becomes `pending_approval`.
2.  Leads check `GET /api/approvals/requests`.
3.  Lead calls `POST /.../approve` or `reject`.
4.  Job automatically proceeds to queue or is cancelled.

<details>
<summary>Approval Rules Engine</summary>

The rules engine evaluates job submissions against configured rules. Rules are processed in priority order; the first matching rule triggers the approval requirement.

**Available Rule Conditions:**

| Condition | Description |
|-----------|-------------|
| `cost_exceeds` | Triggers when estimated cost exceeds threshold (USD) |
| `hardware_type` | Matches hardware type (glob pattern, e.g., `a100*`) |
| `daily_jobs_exceed` | Triggers when user's daily job count exceeds threshold |
| `first_job` | Triggers for a user's very first job |
| `config_pattern` | Matches config name patterns |
| `provider` | Matches specific provider name |

**Example: Require approval for jobs over $50:**

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

Rules can specify `exempt_levels` to allow certain users to bypass approval, and `applies_to_provider`/`applies_to_level` to scope rules.

</details>

<details>
<summary>Email-Based Approval (IMAP Workflow)</summary>

For teams that prefer email-based workflows, SimpleTuner supports approval via email replies using IMAP IDLE.

**How It Works:**
1. Job submission triggers approval requirement
2. Notification email sent to approvers with unique response token
3. IMAP handler monitors inbox using IDLE (push notifications)
4. Approver replies with "approve" or "reject" (or aliases like `yes`, `lgtm`, `+1`)
5. System parses response and processes approval

Configure via **Administration → Notifications** or API. Response tokens expire after 24 hours and are single-use.

</details>

### Job Queue & Concurrency

The scheduler manages fair usage of resources. See its [dedicated documentation](../../JOB_QUEUE.md) for details.

*   **Priority:** Admins > Leads > Researchers > Viewers.
*   **Concurrency:** Limits are enforced globally and per-user.
    *   Update limits via UI: **Cloud tab → Job Queue panel** (admin only)
    *   Update limits via API: `POST /api/queue/concurrency` with `{"max_concurrent": 10, "user_max_concurrent": 3}`

### Job Status Polling (No Webhooks Required)

For secure environments where public webhooks are impossible, SimpleTuner includes a background poller.

Add to `simpletuner-enterprise.yaml`:

```yaml
background:
  job_status_polling:
    enabled: true
    interval_seconds: 30
```

This service queries the provider API every 30s and updates the local database, emitting real-time events to the UI via SSE.

### API Key Rotation

Securely manage cloud provider credentials. See **API Cookbook** for rotation scripts and provider-specific details in the [Cloud Training documentation](../cloud/README.md).

---


## 4. API Cookbook

<details>
<summary>OIDC/LDAP Configuration Examples</summary>

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
<summary>User Administration Examples</summary>

**Create a Researcher:**
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

**Grant Custom Permission:**
```bash
curl -X POST http://localhost:8080/api/users/123/permissions \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"permission_name": "admin.approve", "granted": true}'
```

</details>

<details>
<summary>Credential Management</summary>

SimpleTuner includes credential lifecycle management for tracking, rotating, and auditing API credentials.

**Credential Resolution:** When submitting jobs, SimpleTuner checks per-user credentials first, then falls back to global credentials (environment variables).

| Scenario | Per-User | Global | Behavior |
|----------|----------|--------|----------|
| **Shared org key** | ❌ | ✅ | All users share the org's API key |
| **BYOK** | ✅ | ❌ | Each user provides their own key |
| **Hybrid** | Some | ✅ | Users with keys use theirs, others use global |

**Rotation:** Navigate to **Admin > Auth** → user → **Manage Credentials** → **Rotate**. Stale credentials (>90 days) display a warning badge.

</details>

#### External Orchestration {#external-orchestration-airflow}

<details>
<summary>Airflow example</summary>

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

**Health Check Failures**
*   `503 Service Unavailable`: Check database connectivity.
*   `Degraded`: Usually means an optional component (like a cloud provider API) is unreachable or unconfigured.

**Authentication Issues**
*   **OIDC Redirect Loop:** Verify `issuer_url` matches exactly what the provider expects (trailing slashes matter!).
*   **Internal Auth Bypass:** Check server logs for "Auth bypassed for IP..." to confirm your load balancer is passing the correct `X-Real-IP`.

**Job Updates Stalled**
*   If webhooks are blocked, ensure **Job Status Polling** is enabled in `simpletuner-enterprise.yaml`.
*   Check `GET /api/cloud/metrics/prometheus` for `simpletuner_jobs_active` to see if the internal state thinks jobs are running.

**Missing Metrics**
*   Ensure your Prometheus scraper is configured to hit `/api/cloud/metrics/prometheus` and not just `/metrics`.

---


## 6. Organizations & Team Quotas

SimpleTuner supports hierarchical organizations and teams with ceiling-based quota enforcement.

### Hierarchy

```
Organization (quota ceiling)
    └── Team (quota ceiling, bounded by org)
         └── User (limit, bounded by team and org)
```

### Ceiling Model

Quotas use a ceiling model where org limits are absolute ceilings:
- **Org quota**: Absolute ceiling for all members
- **Team quota**: Ceiling for team members (cannot exceed org)
- **User/Level quota**: Specific limits (bounded by team and org)

**Example:**
- Org ceiling: 100 concurrent jobs
- Team ceiling: 20 concurrent jobs
- User limit: 50 concurrent jobs → **Effective: 20** (team ceiling applies)

**Enforcement Rules:**
- Team quotas are validated at set-time: attempting to set a team quota higher than the org ceiling returns HTTP 400
- User quotas are validated at runtime: effective limit is the minimum of user, team, and org ceilings
- Reducing an org ceiling does not automatically reduce existing team ceilings (admin must update manually)

<details>
<summary>API Examples</summary>

**Create Organization:**
```bash
curl -X POST http://localhost:8080/api/orgs \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"name": "ML Research", "slug": "ml-research"}'
```

**Set Org Quota Ceiling:**
```bash
curl -X POST http://localhost:8080/api/orgs/1/quotas \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"quota_type": "concurrent_jobs", "limit_value": 100, "action": "block"}'
```

**Create Team:**
```bash
curl -X POST http://localhost:8080/api/orgs/1/teams \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"name": "NLP Team", "slug": "nlp"}'
```

**Add User to Team:**
```bash
curl -X POST http://localhost:8080/api/orgs/1/teams/1/members \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"user_id": 123, "role": "member"}'
```

</details>

### Quota and Cost Limit Actions

When a quota or cost limit is reached, the configured `action` determines behavior:

| Action | Behavior |
|--------|----------|
| `warn` | Job proceeds with warning in logs/UI |
| `block` | Job submission rejected |
| `notify` | Job proceeds, admins alerted |

<details>
<summary>Cost Limit Configuration</summary>

Cost limits can be configured per-provider via **Cloud tab → Settings** or API:

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

Check status: `GET /api/cloud/metrics/cost-limit-status`

</details>

---


## 7. Limitations

### Workflow / Pipeline Jobs (DAGs)

SimpleTuner does not support job dependencies or multi-step workflows where one job's output feeds into another. Each cloud job is independent.

**Recommended approach:** Use external orchestration tools like Airflow, Prefect, or Dagster to chain jobs via the REST API. See the [Airflow example](#external-orchestration-airflow) in the API Cookbook above.

### Resuming Training Runs

There is no built-in support for resuming interrupted, failed, or cancelled training runs. Cloud jobs do not automatically recover from checkpoints.

**Workarounds:**
- Configure frequent HuggingFace Hub pushes (`--push_checkpoints_to_hub`) to save intermediate state
- Implement custom checkpoint management by downloading outputs and using them as starting points for new jobs
- For mission-critical workloads, consider breaking long training runs into smaller segments

<details>
<summary>UI Feature Reference</summary>

| Feature | UI Location | API |
|---------|-------------|-----|
| Organizations & Teams | Administration → Orgs | `/api/orgs` |
| Quotas | Administration → Quotas | `/api/orgs/{id}/quotas` |
| OIDC/LDAP | Administration → Auth | `/api/cloud/external-auth/providers` |
| Users | Administration → Users | `/api/users` |
| Audit Logs | Sidebar → Audit Log | `/api/audit` |
| Queue | Cloud tab → Job Queue | `/api/queue/concurrency` |
| Approvals | Administration → Approvals | `/api/approvals/requests` |

The Administration section is visible when no auth is configured (single-user mode) or the user has admin privileges.

</details>

<details>
<summary>Enterprise Onboarding Flow</summary>

The Admin panel includes a guided onboarding that helps set up authentication, organizations, teams, quotas, and credentials in order.

| Step | Feature |
|------|---------|
| 1 | Authentication (OIDC/LDAP) |
| 2 | Organization |
| 3 | Teams |
| 4 | Quotas |
| 5 | Credentials |

Each step can be completed or skipped. State persists in browser localStorage.

</details>

---


## 8. Notification System

SimpleTuner includes a multi-channel notification system for job status, approvals, quotas, and system events.

| Channel | Use Case |
|---------|----------|
| **Email** | Approval requests, job completion (SMTP/IMAP) |
| **Webhook** | CI/CD integration (JSON + HMAC signatures) |
| **Slack** | Team notifications (incoming webhooks) |

Configure via **Administration → Notifications** or API.

<details>
<summary>Event Types</summary>

| Category | Events |
|----------|--------|
| Approval | `approval.required`, `approval.granted`, `approval.rejected`, `approval.expired` |
| Job | `job.submitted`, `job.started`, `job.completed`, `job.failed`, `job.cancelled` |
| Quota | `quota.warning`, `quota.exceeded`, `cost.warning`, `cost.exceeded` |
| System | `system.provider_error`, `system.provider_degraded`, `system.webhook_failure` |
| Auth | `auth.login_failure`, `auth.new_device` |

</details>

<details>
<summary>Channel Configuration Examples</summary>

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

**Webhook:** Payloads signed with HMAC-SHA256 (`X-SimpleTuner-Signature` header).

</details>

---


## 9. Resource Rules

Resource rules provide fine-grained access control for configs, hardware types, and output paths using glob patterns.

| Type | Example Pattern |
|------|-----------------|
| `config` | `team-x-*`, `production-*` |
| `hardware` | `gpu-a100*`, `*-80gb` |
| `provider` | `replicate`, `runpod` |

Rules use **allow/deny** actions with "most permissive wins" logic. Configure via **Administration → Rules**.

<details>
<summary>Rule Examples</summary>

**Team Isolation:** Researchers can only use configs starting with "team-x-"
```
Level: researcher
Rules:
  - config: "team-x-*" → allow
  - config: "*" → deny
```

**Hardware Restrictions:** Researchers limited to T4/V100, leads can use any hardware
```
Level: researcher → hardware: "gpu-t4*" allow, "gpu-v100*" allow
Level: lead → hardware: "*" allow
```

</details>

---


## 10. Permission Matrix

<details>
<summary>Full Permission Matrix</summary>

### Job Permissions

| Permission | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `job.submit` | | ✓ | ✓ | ✓ |
| `job.view.own` | ✓ | ✓ | ✓ | ✓ |
| `job.view.all` | | | ✓ | ✓ |
| `job.cancel.own` | | ✓ | ✓ | ✓ |
| `job.cancel.all` | | | | ✓ |
| `job.priority.high` | | | ✓ | ✓ |
| `job.bypass.queue` | | | | ✓ |
| `job.bypass.approval` | | | | ✓ |

### Config Permissions

| Permission | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `config.view` | ✓ | ✓ | ✓ | ✓ |
| `config.create` | | ✓ | ✓ | ✓ |
| `config.edit.own` | | ✓ | ✓ | ✓ |
| `config.edit.all` | | | | ✓ |

### Admin Permissions

| Permission | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `admin.users` | | | | ✓ |
| `admin.approve` | | | ✓ | ✓ |
| `admin.audit` | | | ✓ | ✓ |
| `admin.config` | | | | ✓ |
| `queue.approve` | | | ✓ | ✓ |
| `queue.manage` | | | | ✓ |

### Org/Team Permissions

| Permission | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `org.view` | | | ✓ | ✓ |
| `org.create` | | | | ✓ |
| `team.view` | | | ✓ | ✓ |
| `team.create` | | | ✓ | ✓ |
| `team.manage.members` | | | ✓ | ✓ |

</details>

**Permission Overrides:** Individual users can have permissions granted or denied via **Administration → Users → Permission Overrides**.
