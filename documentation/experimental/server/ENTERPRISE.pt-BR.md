# Guia Enterprise

Este documento cobre o deploy do SimpleTuner em ambientes multi-usuario com autenticacao, workflows de aprovacao e gerenciamento de cotas.

## 1. Deploy e infraestrutura

### Metodos de configuracao

A maioria dos recursos enterprise pode ser configurada via **Web UI** (painel Administration) ou **REST API**. Algumas configuracoes de infraestrutura exigem um arquivo de config ou variaveis de ambiente.

| Recurso | Web UI | API | Arquivo de config |
|---------|--------|-----|-------------|
| Provedores OIDC/LDAP | ✓ | ✓ | ✓ |
| Usuarios e papeis | ✓ | ✓ | |
| Regras de aprovacao | ✓ | ✓ | |
| Cotas | ✓ | ✓ | |
| Notificacoes | ✓ | ✓ | |
| Bypass de rede (proxies confiaveis) | | | ✓ |
| Polling de jobs em background | | | ✓ |
| Configuracoes TLS | | | ✓ |

**Arquivo de config** (`simpletuner-enterprise.yaml` ou `.json`) e necessario apenas para configuracoes de infraestrutura que precisam ser conhecidas no startup. O SimpleTuner procura nesses locais:

1. `$SIMPLETUNER_ENTERPRISE_CONFIG` (variavel de ambiente)
2. `./simpletuner-enterprise.yaml` (diretorio atual)
3. `~/.config/simpletuner/enterprise.yaml`
4. `/etc/simpletuner/enterprise.yaml`

O arquivo suporta interpolacao de variaveis de ambiente com a sintaxe `${VAR}`.

### Checklist de inicio rapido

1.  **Inicie o SimpleTuner**: `simpletuner server` (ou `--webui` para uso local)
2.  **Configure via UI**: Navegue ao painel Administration para configurar usuarios, SSO, cotas
3.  **Health checks** (para producao):
    *   Liveness: `GET /api/cloud/health/live` (200 OK)
    *   Readiness: `GET /api/cloud/health/ready` (200 OK)
    *   Deep Check: `GET /api/cloud/health` (inclui conectividade com provedores)

### Seguranca de rede e bypass de autenticacao

<details>
<summary>Configurando proxies confiaveis e bypass de rede interna (arquivo de config necessario)</summary>

Em ambientes corporativos (VPNs, VPCs privadas), voce pode confiar no trafego interno ou descarregar autenticacao para um gateway.

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

### Load balancer e configuracao TLS

O SimpleTuner espera um reverse proxy upstream para terminacao TLS.

<details>
<summary>Exemplo de reverse proxy nginx</summary>

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

### Observabilidade (Prometheus e logging)

**Metricas:**
Scrape `GET /api/cloud/metrics/prometheus` para insights operacionais.
*   `simpletuner_jobs_active`: Profundidade atual da fila.
*   `simpletuner_cost_total_usd`: Rastreamento de gastos.
*   `simpletuner_uptime_seconds`: Disponibilidade.

**Logging:**
Defina `SIMPLETUNER_LOG_FORMAT=json` para ingestao no Splunk/Datadog/ELK.

<details>
<summary>Configuracao de retencao de dados</summary>

Configure periodos de retencao para requisitos de compliance via variaveis de ambiente:

| Variavel | Padrao | Descricao |
|----------|---------|-------------|
| `SIMPLETUNER_JOB_RETENTION_DAYS` | 90 | Dias para reter registros de jobs concluidos |
| `SIMPLETUNER_AUDIT_RETENTION_DAYS` | 90 | Dias para reter entradas de audit log |

```bash
# SOC 2 / HIPAA: 1 year retention
export SIMPLETUNER_JOB_RETENTION_DAYS=365
export SIMPLETUNER_AUDIT_RETENTION_DAYS=365

# Disable automatic cleanup (manual management)
export SIMPLETUNER_JOB_RETENTION_DAYS=0
```

Definir `0` desabilita a limpeza automatica. A limpeza roda diariamente.

</details>

---


## 2. Identidade e gerenciamento de acesso (SSO)

O SimpleTuner suporta OIDC (OpenID Connect) e LDAP para SSO com Okta, Azure AD, Keycloak e Active Directory.

### Configurando provedores

**Via Web UI:** Navegue para **Administration → Auth** para adicionar e configurar provedores.

**Via API:** Veja o [API Cookbook](#4-api-cookbook) para exemplos com curl.

<details>
<summary>Via arquivo de configuracao (para fluxos IaC/GitOps)</summary>

Adicione ao seu `simpletuner-enterprise.yaml`:

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
<summary>Validacao de estado OAuth entre workers</summary>

Ao usar autenticacao OIDC em deploys multi-worker (ex.: atras de um load balancer com varios workers Gunicorn), a validacao de estado OAuth precisa funcionar entre todos os workers. O SimpleTuner lida com isso automaticamente armazenando o estado OAuth no banco de dados.

**Como funciona:**

1. **Geracao de estado**: Quando um usuario inicia o login OIDC, um token de estado aleatorio e gerado e armazenado no banco com o nome do provedor, URI de redirect e expiracao de 10 minutos.

2. **Validacao de estado**: Quando o callback chega (potencialmente para outro worker), o estado e buscado e consumido atomicamente (uso unico).

3. **Limpeza**: Estados expirados sao removidos automaticamente durante operacoes normais.

Nenhuma configuracao adicional e necessaria. O armazenamento de estado OAuth usa o mesmo banco de jobs e usuarios.

**Solucao de problemas de "Invalid OAuth state":**
1. Verifique se o callback chegou dentro de 10 minutos do inicio do login
2. Verifique se todos os workers compartilham o mesmo caminho de banco
3. Verifique permissoes de escrita no banco
4. Procure erros "Failed to store OAuth state" nos logs

</details>

### Gerenciamento de usuarios e papeis

O SimpleTuner usa um sistema hierarquico de papeis. Usuarios podem ser gerenciados via `GET/POST /api/users`.

| Papel | Prioridade | Descricao |
|------|----------|-------------|
| **Viewer** | 10 | Acesso somente leitura ao historico de jobs e logs. |
| **Researcher** | 20 | Acesso padrao. Pode enviar jobs e gerenciar suas proprias API keys. |
| **Lead** | 30 | Pode aprovar jobs pendentes e ver uso de recursos da equipe. |
| **Admin** | 100 | Acesso total ao sistema, incluindo gerenciamento de usuarios e configuracao de regras. |

---

## 3. Governanca e operacoes

### Workflows de aprovacao

Controle custos e uso de recursos exigindo aprovacoes para criterios especificos. As regras sao avaliadas no momento do envio.

**Workflow:**
1.  Usuario envia job -> status vira `pending_approval`.
2.  Leads consultam `GET /api/approvals/requests`.
3.  Lead chama `POST /.../approve` ou `reject`.
4.  Job prossegue automaticamente para a fila ou e cancelado.

<details>
<summary>Motor de regras de aprovacao</summary>

O motor de regras avalia envios de job contra regras configuradas. As regras sao processadas por prioridade; a primeira regra que casa aciona a exigencia de aprovacao.

**Condicoes de regra disponiveis:**

| Condicao | Descricao |
|-----------|-------------|
| `cost_exceeds` | Aciona quando o custo estimado excede o limite (USD) |
| `hardware_type` | Casa tipo de hardware (glob, ex.: `a100*`) |
| `daily_jobs_exceed` | Aciona quando a contagem diaria de jobs do usuario excede o limite |
| `first_job` | Aciona para o primeiro job de um usuario |
| `config_pattern` | Casa padroes de nome de config |
| `provider` | Casa o nome do provedor |

**Exemplo: exigir aprovacao para jobs acima de $50:**

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

Regras podem especificar `exempt_levels` para permitir que certos usuarios ignorem aprovacao, e `applies_to_provider`/`applies_to_level` para limitar o escopo.

</details>

<details>
<summary>Aprovacao via email (workflow IMAP)</summary>

Para equipes que preferem fluxo via email, o SimpleTuner suporta aprovacao por resposta de email usando IMAP IDLE.

**Como funciona:**
1. O envio do job dispara a exigencia de aprovacao
2. Email de notificacao e enviado aos aprovadores com token de resposta unico
3. Handler IMAP monitora a caixa de entrada via IDLE (notificacoes push)
4. Aprovador responde com "approve" ou "reject" (ou aliases como `yes`, `lgtm`, `+1`)
5. O sistema analisa a resposta e processa a aprovacao

Configure via **Administration → Notifications** ou API. Tokens de resposta expiram apos 24 horas e sao de uso unico.

</details>

### Fila de jobs e concorrencia

O scheduler gerencia o uso justo de recursos. Veja a [documentacao dedicada](../../JOB_QUEUE.md) para detalhes.

*   **Prioridade:** Admins > Leads > Researchers > Viewers.
*   **Concorrencia:** Limites sao impostos globalmente e por usuario.
    *   Atualize limites via UI: **Cloud tab → Job Queue panel** (admin apenas)
    *   Atualize limites via API: `POST /api/queue/concurrency` com `{"max_concurrent": 10, "user_max_concurrent": 3}`

### Polling de status de jobs (sem webhooks)

Para ambientes seguros onde webhooks publicos sao impossiveis, o SimpleTuner inclui um poller em background.

Adicione em `simpletuner-enterprise.yaml`:

```yaml
background:
  job_status_polling:
    enabled: true
    interval_seconds: 30
```

Esse servico consulta a API do provedor a cada 30s e atualiza o banco local, emitindo eventos em tempo real para a UI via SSE.

### Rotacao de API keys

Gerencie credenciais de provedor com seguranca. Veja **API Cookbook** para scripts de rotacao e detalhes especificos de provedor em [Cloud Training documentation](../cloud/README.md).

---


## 4. API Cookbook

<details>
<summary>Exemplos de configuracao OIDC/LDAP</summary>

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
<summary>Exemplos de administracao de usuarios</summary>

**Criar um Researcher:**
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

**Conceder permissao customizada:**
```bash
curl -X POST http://localhost:8080/api/users/123/permissions \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"permission_name": "admin.approve", "granted": true}'
```

</details>

<details>
<summary>Gerenciamento de credenciais</summary>

O SimpleTuner inclui gerenciamento do ciclo de vida de credenciais para rastrear, rotacionar e auditar credenciais de API.

**Resolucao de credenciais:** Ao enviar jobs, o SimpleTuner verifica primeiro credenciais por usuario, depois cai para credenciais globais (variaveis de ambiente).

| Cenario | Por usuario | Global | Comportamento |
|----------|----------|--------|----------|
| **Chave org compartilhada** | ❌ | ✅ | Todos os usuarios compartilham a API key da org |
| **BYOK** | ✅ | ❌ | Cada usuario fornece sua propria chave |
| **Hibrido** | Alguns | ✅ | Usuarios com chave usam a deles, outros usam global |

**Rotacao:** Navegue para **Admin > Auth** → usuario → **Manage Credentials** → **Rotate**. Credenciais antigas (>90 dias) exibem badge de aviso.

</details>

#### Orquestracao externa {#external-orchestration-airflow}

<details>
<summary>Exemplo Airflow</summary>

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

## 5. Solucao de problemas

**Falhas de health check**
*   `503 Service Unavailable`: Verifique conectividade com o banco.
*   `Degraded`: Geralmente significa que um componente opcional (como API de provedor de nuvem) esta inacessivel ou nao configurado.

**Problemas de autenticacao**
*   **OIDC Redirect Loop:** Verifique se `issuer_url` corresponde exatamente ao esperado pelo provedor (barras finais importam!).
*   **Bypass de auth interno:** Verifique logs do servidor para "Auth bypassed for IP..." e confirme que o load balancer esta passando o `X-Real-IP` correto.

**Atualizacoes de job travadas**
*   Se webhooks estiverem bloqueados, garanta que **Job Status Polling** esteja habilitado em `simpletuner-enterprise.yaml`.
*   Verifique `GET /api/cloud/metrics/prometheus` para `simpletuner_jobs_active` e veja se o estado interno pensa que jobs estao rodando.

**Metricas ausentes**
*   Garanta que seu scraper Prometheus esteja configurado para `/api/cloud/metrics/prometheus` e nao apenas `/metrics`.

---


## 6. Organizacoes e cotas de equipe

O SimpleTuner suporta organizacoes e equipes hierarquicas com enforcement de cotas por teto.

### Hierarquia

```
Organization (quota ceiling)
    └── Team (quota ceiling, bounded by org)
         └── User (limit, bounded by team and org)
```

### Modelo de teto

Cotas usam um modelo de teto onde limites da org sao tetos absolutos:
- **Cota da org**: Teto absoluto para todos os membros
- **Cota da equipe**: Teto para membros da equipe (nao pode exceder a org)
- **Cota de usuario/nivel**: Limites especificos (limitados por equipe e org)

**Exemplo:**
- Teto da org: 100 jobs concorrentes
- Teto da equipe: 20 jobs concorrentes
- Limite do usuario: 50 jobs concorrentes → **Efetivo: 20** (teto da equipe se aplica)

**Regras de enforcement:**
- Cotas de equipe sao validadas no set: tentar definir uma cota de equipe acima do teto da org retorna HTTP 400
- Cotas de usuario sao validadas em runtime: limite efetivo e o minimo entre usuario, equipe e org
- Reduzir o teto da org nao reduz automaticamente tetos de equipe existentes (admin deve atualizar manualmente)

<details>
<summary>Exemplos de API</summary>

**Criar organizacao:**
```bash
curl -X POST http://localhost:8080/api/orgs \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"name": "ML Research", "slug": "ml-research"}'
```

**Definir teto de cota da org:**
```bash
curl -X POST http://localhost:8080/api/orgs/1/quotas \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"quota_type": "concurrent_jobs", "limit_value": 100, "action": "block"}'
```

**Criar equipe:**
```bash
curl -X POST http://localhost:8080/api/orgs/1/teams \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"name": "NLP Team", "slug": "nlp"}'
```

**Adicionar usuario a equipe:**
```bash
curl -X POST http://localhost:8080/api/orgs/1/teams/1/members \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"user_id": 123, "role": "member"}'
```

</details>

### Acoes de cota e limite de custo

Quando uma cota ou limite de custo e atingido, a `action` configurada determina o comportamento:

| Acao | Comportamento |
|--------|----------|
| `warn` | Job prossegue com aviso nos logs/UI |
| `block` | Envio de job rejeitado |
| `notify` | Job prossegue, admins sao alertados |

<details>
<summary>Configuracao de limite de custo</summary>

Limites de custo podem ser configurados por provedor via **Cloud tab → Settings** ou API:

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


## 7. Limitacoes

### Jobs de workflow/pipeline (DAGs)

O SimpleTuner nao suporta dependencias entre jobs ou workflows multi-etapas onde a saida de um job alimenta outro. Cada job em nuvem e independente.

**Abordagem recomendada:** Use ferramentas de orquestracao externa como Airflow, Prefect ou Dagster para encadear jobs via API REST. Veja o [exemplo Airflow](#external-orchestration-airflow) no API Cookbook acima.

### Retomar execucoes de treinamento

Nao ha suporte embutido para retomar execucoes interrompidas, falhas ou canceladas. Jobs em nuvem nao se recuperam automaticamente de checkpoints.

**Workarounds:**
- Configure pushes frequentes no HuggingFace Hub (`--push_checkpoints_to_hub`) para salvar estado intermediario
- Implemente gerenciamento de checkpoints customizado baixando outputs e usando-os como pontos de inicio para novos jobs
- Para workloads criticos, considere quebrar treinos longos em segmentos menores

<details>
<summary>Referencia de recursos da UI</summary>

| Recurso | Local na UI | API |
|---------|-------------|-----|
| Organizacoes e equipes | Administration → Orgs | `/api/orgs` |
| Cotas | Administration → Quotas | `/api/orgs/{id}/quotas` |
| OIDC/LDAP | Administration → Auth | `/api/cloud/external-auth/providers` |
| Usuarios | Administration → Users | `/api/users` |
| Audit Logs | Sidebar → Audit Log | `/api/audit` |
| Queue | Cloud tab → Job Queue | `/api/queue/concurrency` |
| Aprovacoes | Administration → Approvals | `/api/approvals/requests` |

A secao Administration fica visivel quando nao ha auth configurada (modo single-user) ou o usuario tem privilegios admin.

</details>

<details>
<summary>Fluxo de onboarding enterprise</summary>

O painel Admin inclui um onboarding guiado que ajuda a configurar autenticacao, organizacoes, equipes, cotas e credenciais em ordem.

| Passo | Recurso |
|------|---------|
| 1 | Autenticacao (OIDC/LDAP) |
| 2 | Organizacao |
| 3 | Equipes |
| 4 | Cotas |
| 5 | Credenciais |

Cada passo pode ser concluido ou pulado. O estado persiste no localStorage do navegador.

</details>

---


## 8. Sistema de notificacoes

O SimpleTuner inclui um sistema de notificacoes multi-canal para status de job, aprovacoes, cotas e eventos do sistema.

| Canal | Caso de uso |
|---------|----------|
| **Email** | Pedidos de aprovacao, conclusao de job (SMTP/IMAP) |
| **Webhook** | Integracao CI/CD (JSON + assinaturas HMAC) |
| **Slack** | Notificacoes de equipe (incoming webhooks) |

Configure via **Administration → Notifications** ou API.

<details>
<summary>Tipos de evento</summary>

| Categoria | Eventos |
|----------|--------|
| Approval | `approval.required`, `approval.granted`, `approval.rejected`, `approval.expired` |
| Job | `job.submitted`, `job.started`, `job.completed`, `job.failed`, `job.cancelled` |
| Quota | `quota.warning`, `quota.exceeded`, `cost.warning`, `cost.exceeded` |
| System | `system.provider_error`, `system.provider_degraded`, `system.webhook_failure` |
| Auth | `auth.login_failure`, `auth.new_device` |

</details>

<details>
<summary>Exemplos de configuracao de canal</summary>

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

**Webhook:** Payloads assinados com HMAC-SHA256 (header `X-SimpleTuner-Signature`).

</details>

---


## 9. Regras de recursos

Regras de recursos fornecem controle de acesso granular para configs, tipos de hardware e caminhos de output usando glob patterns.

| Tipo | Exemplo de padrao |
|------|-----------------|
| `config` | `team-x-*`, `production-*` |
| `hardware` | `gpu-a100*`, `*-80gb` |
| `provider` | `replicate`, `runpod` |

Regras usam acoes **allow/deny** com logica "most permissive wins". Configure via **Administration → Rules**.

<details>
<summary>Exemplos de regras</summary>

**Isolamento de equipe:** Researchers so podem usar configs comecando com "team-x-"
```
Level: researcher
Rules:
  - config: "team-x-*" → allow
  - config: "*" → deny
```

**Restricoes de hardware:** Researchers limitados a T4/V100, leads podem usar qualquer hardware
```
Level: researcher → hardware: "gpu-t4*" allow, "gpu-v100*" allow
Level: lead → hardware: "*" allow
```

</details>

---


## 10. Matriz de permissoes

<details>
<summary>Matriz completa de permissoes</summary>

### Permissoes de job

| Permissao | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `job.submit` | | ✓ | ✓ | ✓ |
| `job.view.own` | ✓ | ✓ | ✓ | ✓ |
| `job.view.all` | | | ✓ | ✓ |
| `job.cancel.own` | | ✓ | ✓ | ✓ |
| `job.cancel.all` | | | | ✓ |
| `job.priority.high` | | | ✓ | ✓ |
| `job.bypass.queue` | | | | ✓ |
| `job.bypass.approval` | | | | ✓ |

### Permissoes de config

| Permissao | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `config.view` | ✓ | ✓ | ✓ | ✓ |
| `config.create` | | ✓ | ✓ | ✓ |
| `config.edit.own` | | ✓ | ✓ | ✓ |
| `config.edit.all` | | | | ✓ |

### Permissoes admin

| Permissao | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `admin.users` | | | | ✓ |
| `admin.approve` | | | ✓ | ✓ |
| `admin.audit` | | | ✓ | ✓ |
| `admin.config` | | | | ✓ |
| `queue.approve` | | | ✓ | ✓ |
| `queue.manage` | | | | ✓ |

### Permissoes de org/equipe

| Permissao | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `org.view` | | | ✓ | ✓ |
| `org.create` | | | | ✓ |
| `team.view` | | | ✓ | ✓ |
| `team.create` | | | ✓ | ✓ |
| `team.manage.members` | | | ✓ | ✓ |

</details>

**Overrides de permissao:** Usuarios individuais podem ter permissoes concedidas ou negadas via **Administration → Users → Permission Overrides**.
