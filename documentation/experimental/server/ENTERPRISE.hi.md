# Enterprise Guide

यह दस्तावेज़ SimpleTuner को multi-user environments में deploy करने के लिए authentication, approval workflows, और quota management के साथ मार्गदर्शन देता है।

## 1. Deployment & Infrastructure

### Configuration Methods

अधिकांश enterprise फीचर्स **Web UI** (Administration panel) या **REST API** से कॉन्फ़िगर किए जा सकते हैं। कुछ infrastructure-level settings को config file या environment variables चाहिए।

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

**Config file** (`simpletuner-enterprise.yaml` या `.json`) केवल उन infrastructure settings के लिए जरूरी है जिन्हें startup पर जानना होता है। SimpleTuner इन लोकेशन्स में खोजता है:

1. `$SIMPLETUNER_ENTERPRISE_CONFIG` (environment variable)
2. `./simpletuner-enterprise.yaml` (current directory)
3. `~/.config/simpletuner/enterprise.yaml`
4. `/etc/simpletuner/enterprise.yaml`

File `${VAR}` syntax के साथ environment variable interpolation सपोर्ट करती है।

### Quick Start Checklist

1.  **SimpleTuner शुरू करें**: `simpletuner server` (या local use के लिए `--webui`)
2.  **UI से कॉन्फ़िगर करें**: Administration panel में users, SSO, quotas सेट करें
3.  **Health Checks** (production के लिए):
    *   Liveness: `GET /api/cloud/health/live` (200 OK)
    *   Readiness: `GET /api/cloud/health/ready` (200 OK)
    *   Deep Check: `GET /api/cloud/health` (provider connectivity सहित)

### Network Security & Authentication Bypass

<details>
<summary>Trusted proxies और internal network bypass कॉन्फ़िगर करें (config file आवश्यक)</summary>

Corporate environments (VPNs, private VPCs) में आप internal traffic को trust करना या authentication को gateway पर offload करना चाह सकते हैं।

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

SimpleTuner TLS termination के लिए upstream reverse proxy की अपेक्षा करता है।

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
`GET /api/cloud/metrics/prometheus` स्क्रेप करें ताकि operational insights मिलें।
*   `simpletuner_jobs_active`: Current queue depth.
*   `simpletuner_cost_total_usd`: Spend tracking.
*   `simpletuner_uptime_seconds`: Availability.

**Logging:**
Splunk/Datadog/ELK ingestion के लिए `SIMPLETUNER_LOG_FORMAT=json` सेट करें।

<details>
<summary>Data Retention Configuration</summary>

Compliance requirements के लिए retention periods environment variables से कॉन्फ़िगर करें:

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMPLETUNER_JOB_RETENTION_DAYS` | 90 | Completed job records रखने के दिन |
| `SIMPLETUNER_AUDIT_RETENTION_DAYS` | 90 | Audit log entries रखने के दिन |

```bash
# SOC 2 / HIPAA: 1 year retention
export SIMPLETUNER_JOB_RETENTION_DAYS=365
export SIMPLETUNER_AUDIT_RETENTION_DAYS=365

# Disable automatic cleanup (manual management)
export SIMPLETUNER_JOB_RETENTION_DAYS=0
```

`0` सेट करने से automatic cleanup बंद हो जाता है। Cleanup रोज चलता है।

</details>

---
## 2. Identity & Access Management (SSO)

SimpleTuner Okta, Azure AD, Keycloak, और Active Directory के साथ SSO के लिए OIDC (OpenID Connect) और LDAP सपोर्ट करता है।

### Providers कॉन्फ़िगर करना

**Web UI से:** **Administration → Auth** पर जाएं और providers जोड़ें/कॉन्फ़िगर करें।

**API से:** curl examples के लिए [API Cookbook](#4-api-cookbook) देखें।

<details>
<summary>Config file के जरिए (IaC/GitOps workflows के लिए)</summary>

`simpletuner-enterprise.yaml` में जोड़ें:

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

Multi-worker deployments में OIDC authentication के लिए OAuth state validation को सभी workers में काम करना चाहिए (जैसे load balancer के पीछे multiple Gunicorn workers)। SimpleTuner OAuth state को database में स्टोर करके इसे अपने आप संभालता है।

**यह कैसे काम करता है:**

1. **State Generation**: जब user OIDC login शुरू करता है, एक cryptographically random state token जनरेट होता है और provider नाम, redirect URI, और 10-मिनट expiration के साथ database में स्टोर होता है।

2. **State Validation**: जब callback आता है (संभव है किसी दूसरे worker पर), state lookup होती है और atomically consume की जाती है (single-use)।

3. **Cleanup**: Expired states normal operations के दौरान अपने आप purge हो जाती हैं।

कोई अतिरिक्त config जरूरी नहीं। OAuth state storage jobs और users वाले same database का उपयोग करता है।

**"Invalid OAuth state" errors का ट्रबलशूटिंग:**
1. चेक करें कि callback login initiation के 10 मिनट के भीतर आया था
2. सत्यापित करें कि सभी workers एक ही database path शेयर करते हैं
3. Database write permissions चेक करें
4. Logs में "Failed to store OAuth state" errors देखें

</details>

### User Management & Roles

SimpleTuner hierarchical role system का उपयोग करता है। Users को `GET/POST /api/users` से manage किया जा सकता है।

| Role | Priority | Description |
|------|----------|-------------|
| **Viewer** | 10 | Job history और logs के लिए read-only access। |
| **Researcher** | 20 | Standard access। Jobs submit कर सकते हैं और अपनी API keys manage कर सकते हैं। |
| **Lead** | 30 | Pending jobs approve कर सकते हैं और team resource usage देख सकते हैं। |
| **Admin** | 100 | Full system access, user management और rule configuration सहित। |

---

## 3. Governance & Operations

### Approval Workflows

Specific criteria के लिए approvals अनिवार्य करके लागत और resource usage नियंत्रित करें। Rules submission time पर evaluate होते हैं।

**Workflow:**
1.  User job submit करता है -> Status `pending_approval` हो जाता है।
2.  Leads `GET /api/approvals/requests` देखते हैं।
3.  Lead `POST /.../approve` या `reject` कॉल करता है।
4.  Job अपने आप queue में आगे बढ़ती है या cancel हो जाती है।

<details>
<summary>Approval Rules Engine</summary>

Rules engine job submissions को configured rules के खिलाफ evaluate करता है। Rules priority order में प्रोसेस होते हैं; पहली matching rule approval requirement ट्रिगर करती है।

**Available Rule Conditions:**

| Condition | Description |
|-----------|-------------|
| `cost_exceeds` | अनुमानित लागत threshold (USD) से ऊपर होने पर ट्रिगर |
| `hardware_type` | hardware type match करता है (glob pattern, जैसे `a100*`) |
| `daily_jobs_exceed` | user की daily job count threshold से ऊपर होने पर ट्रिगर |
| `first_job` | user के पहले job पर ट्रिगर |
| `config_pattern` | config name patterns match करता है |
| `provider` | specific provider name match करता है |

**Example: $50 से ऊपर jobs के लिए approval जरूरी:**

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

Rules `exempt_levels` से कुछ users को bypass दे सकती हैं, और `applies_to_provider`/`applies_to_level` से scope तय हो सकता है।

</details>

<details>
<summary>Email-Based Approval (IMAP Workflow)</summary>

Email-based workflows पसंद करने वाली टीमों के लिए SimpleTuner IMAP IDLE के जरिए email reply से approval सपोर्ट करता है।

**यह कैसे काम करता है:**
1. Job submission approval requirement ट्रिगर करता है
2. Approvers को unique response token वाला notification email भेजा जाता है
3. IMAP handler inbox को IDLE (push notifications) के जरिए monitor करता है
4. Approver "approve" या "reject" (या `yes`, `lgtm`, `+1` जैसे aliases) से reply करता है
5. सिस्टम response parse करके approval प्रक्रिया पूरी करता है

**Administration → Notifications** या API से कॉन्फ़िगर करें। Response tokens 24 घंटे बाद expire होते हैं और single-use होते हैं।

</details>

### Job Queue & Concurrency

Scheduler resources का fair usage मैनेज करता है। विवरण के लिए इसकी [dedicated documentation](../../JOB_QUEUE.md) देखें।

*   **Priority:** Admins > Leads > Researchers > Viewers.
*   **Concurrency:** Limits globally और per-user enforce होते हैं।
    *   UI से limits अपडेट करें: **Cloud tab → Job Queue panel** (admin only)
    *   API से limits अपडेट करें: `POST /api/queue/concurrency` with `{"max_concurrent": 10, "user_max_concurrent": 3}`

### Job Status Polling (No Webhooks Required)

Secure environments में जहां public webhooks संभव नहीं, SimpleTuner background poller प्रदान करता है।

`simpletuner-enterprise.yaml` में जोड़ें:

```yaml
background:
  job_status_polling:
    enabled: true
    interval_seconds: 30
```

यह service हर 30s provider API query करती है और local database अपडेट करके UI को SSE के जरिए real-time events भेजती है।

### API Key Rotation

Cloud provider credentials को सुरक्षित तरीके से manage करें। Rotation scripts और provider-specific details के लिए **API Cookbook** और [Cloud Training documentation](../cloud/README.md) देखें।

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

**Researcher बनाएं:**
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

**Custom Permission दें:**
```bash
curl -X POST http://localhost:8080/api/users/123/permissions \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"permission_name": "admin.approve", "granted": true}'
```

</details>

<details>
<summary>Credential Management</summary>

SimpleTuner API credentials के tracking, rotating, और auditing के लिए credential lifecycle management प्रदान करता है।

**Credential Resolution:** jobs submit करते समय SimpleTuner पहले per-user credentials देखता है, फिर global credentials (environment variables) पर fallback करता है।

| Scenario | Per-User | Global | Behavior |
|----------|----------|--------|----------|
| **Shared org key** | ❌ | ✅ | सभी users org की API key शेयर करते हैं |
| **BYOK** | ✅ | ❌ | हर user अपनी key देता है |
| **Hybrid** | कुछ | ✅ | जिन users के पास keys हैं वे अपनी key इस्तेमाल करते हैं, बाकी global |

**Rotation:** **Admin > Auth** → user → **Manage Credentials** → **Rotate** पर जाएं। 90 दिन से पुराने credentials पर warning badge दिखता है।

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
*   `503 Service Unavailable`: database connectivity जांचें।
*   `Degraded`: आम तौर पर मतलब है कि कोई optional component (जैसे cloud provider API) unreachable या unconfigured है।

**Authentication Issues**
*   **OIDC Redirect Loop:** `issuer_url` बिल्कुल वैसा ही हो जैसा provider उम्मीद करता है (trailing slashes मायने रखते हैं!).
*   **Internal Auth Bypass:** server logs में "Auth bypassed for IP..." देखें ताकि पुष्टि हो कि आपका load balancer सही `X-Real-IP` पास कर रहा है।

**Job Updates Stalled**
*   अगर webhooks blocked हैं, तो `simpletuner-enterprise.yaml` में **Job Status Polling** enabled है या नहीं देखें।
*   `GET /api/cloud/metrics/prometheus` में `simpletuner_jobs_active` देखें कि internal state jobs running समझ रहा है या नहीं।

**Missing Metrics**
*   सुनिश्चित करें कि आपका Prometheus scraper `/metrics` के बजाय `/api/cloud/metrics/prometheus` हिट कर रहा है।

---

## 6. Organizations & Team Quotas

SimpleTuner hierarchical organizations और teams को ceiling-based quota enforcement के साथ सपोर्ट करता है।

### Hierarchy

```
Organization (quota ceiling)
    └── Team (quota ceiling, bounded by org)
         └── User (limit, bounded by team and org)
```

### Ceiling Model

Quotas ceiling model उपयोग करती हैं जहां org limits absolute ceilings होते हैं:
- **Org quota**: सभी सदस्यों के लिए absolute ceiling
- **Team quota**: टीम के सदस्यों के लिए ceiling (org से अधिक नहीं)
- **User/Level quota**: specific limits (team और org से bounded)

**Example:**
- Org ceiling: 100 concurrent jobs
- Team ceiling: 20 concurrent jobs
- User limit: 50 concurrent jobs → **Effective: 20** (team ceiling लागू होती है)

**Enforcement Rules:**
- Team quotas set-time पर validate होती हैं: org ceiling से बड़ा सेट करने पर HTTP 400
- User quotas runtime पर validate होती हैं: effective limit user/team/org ceilings का minimum है
- Org ceiling घटाने से existing team ceilings अपने आप नहीं बदलतीं (admin को manual अपडेट करना होगा)

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

### Quota और Cost Limit Actions

Quota या cost limit पर पहुंचने पर configured `action` behavior तय करता है:

| Action | Behavior |
|--------|----------|
| `warn` | Job warning के साथ आगे बढ़ती है |
| `block` | Job submission reject होता है |
| `notify` | Job आगे बढ़ती है, admins alert होते हैं |

<details>
<summary>Cost Limit Configuration</summary>

Cost limits प्रति-provider **Cloud tab → Settings** या API से कॉन्फ़िगर किए जा सकते हैं:

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

Status चेक करें: `GET /api/cloud/metrics/cost-limit-status`

</details>

---

## 7. Limitations

### Workflow / Pipeline Jobs (DAGs)

SimpleTuner job dependencies या multi-step workflows सपोर्ट नहीं करता जहां एक job का output दूसरी job में जाए। हर cloud job स्वतंत्र है।

**Recommended approach:** REST API के जरिए jobs chain करने के लिए Airflow, Prefect, या Dagster जैसे external orchestration tools उपयोग करें। ऊपर दिए API Cookbook में [Airflow example](#external-orchestration-airflow) देखें।

### Resuming Training Runs

Interrupted, failed, या cancelled training runs को resume करने के लिए built-in support नहीं है। Cloud jobs checkpoints से अपने आप recover नहीं करतीं।

**Workarounds:**
- Frequent HuggingFace Hub pushes (`--push_checkpoints_to_hub`) कॉन्फ़िगर करें
- Outputs download करके नए jobs के लिए starting points की तरह इस्तेमाल करके custom checkpoint management लागू करें
- Mission-critical workloads के लिए लंबी training को छोटे segments में विभाजित करें

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

No auth configured होने पर (single-user mode) या user के पास admin privileges होने पर Administration section दिखता है।

</details>

<details>
<summary>Enterprise Onboarding Flow</summary>

Admin panel में guided onboarding शामिल है जो authentication, organizations, teams, quotas, और credentials को क्रम में सेट करने में मदद करता है।

| Step | Feature |
|------|---------|
| 1 | Authentication (OIDC/LDAP) |
| 2 | Organization |
| 3 | Teams |
| 4 | Quotas |
| 5 | Credentials |

हर step पूरा या skip किया जा सकता है। State browser localStorage में persist होता है।

</details>

---

## 8. Notification System

SimpleTuner job status, approvals, quotas, और system events के लिए multi-channel notification system देता है।

| Channel | Use Case |
|---------|----------|
| **Email** | Approval requests, job completion (SMTP/IMAP) |
| **Webhook** | CI/CD integration (JSON + HMAC signatures) |
| **Slack** | Team notifications (incoming webhooks) |

**Administration → Notifications** या API से कॉन्फ़िगर करें।

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

**Webhook:** Payloads HMAC-SHA256 (`X-SimpleTuner-Signature` header) से signed होते हैं।

</details>

---

## 9. Resource Rules

Resource rules configs, hardware types, और output paths के लिए glob patterns के जरिए fine-grained access control देती हैं।

| Type | Example Pattern |
|------|-----------------|
| `config` | `team-x-*`, `production-*` |
| `hardware` | `gpu-a100*`, `*-80gb` |
| `provider` | `replicate`, `runpod` |

Rules **allow/deny** actions उपयोग करती हैं, जहां "most permissive wins" logic लागू होता है। **Administration → Rules** से कॉन्फ़िगर करें।

<details>
<summary>Rule Examples</summary>

**Team Isolation:** Researchers सिर्फ "team-x-" से शुरू होने वाले configs उपयोग कर सकते हैं
```
Level: researcher
Rules:
  - config: "team-x-*" → allow
  - config: "*" → deny
```

**Hardware Restrictions:** Researchers को T4/V100 तक सीमित करें, Leads किसी भी hardware का उपयोग कर सकते हैं
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

**Permission Overrides:** Individual users को **Administration → Users → Permission Overrides** से permissions grant/deny की जा सकती हैं।
