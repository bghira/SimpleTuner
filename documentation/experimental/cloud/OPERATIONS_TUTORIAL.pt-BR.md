# Guia de operacoes de treinamento em nuvem

Este documento cobre deploy em producao e operacoes do recurso de treinamento em nuvem do SimpleTuner, com foco em integracao completa com infraestrutura DevOps existente.

## Arquitetura de rede

### Conexoes de saida

O servidor faz conexoes HTTPS de saida para provedores de nuvem configurados. Cada provedor tem seus proprios endpoints e requisitos.

**Detalhes de rede por provedor:**
- [Endpoints da API Replicate](REPLICATE.md#api-endpoints)

### Conexoes de entrada

| Fonte | Endpoint | Proposito |
|--------|----------|---------|
| Infraestrutura do provedor de nuvem | `/api/webhooks/{provider}` | Atualizacoes de status de job |
| Job de treinamento em nuvem | `/api/cloud/storage/{bucket}/{key}` | Upload de outputs de treinamento |
| Sistemas de monitoramento | `/api/cloud/health`, `/api/cloud/metrics/prometheus` | Health e metricas |

### Regras de firewall

Os requisitos de firewall dependem do(s) provedor(es) configurado(s).

**Regras de firewall por provedor:**
- [Configuracao de firewall Replicate](REPLICATE.md#firewall)

### Allowlist de IP de webhooks

Para maior seguranca, voce pode restringir a entrega de webhooks a ranges de IP especificos. Quando configurado, webhooks de IPs fora da allowlist sao rejeitados com resposta 403 Forbidden.

**Configuracao via API:**

<details>
<summary>Exemplo de configuracao via API</summary>

```bash
# Set allowed IPs for a provider's webhooks
curl -X PUT http://localhost:8080/api/cloud/providers/{provider} \
  -H "Content-Type: application/json" \
  -d '{
    "webhook_allowed_ips": ["10.0.0.0/8", "192.168.0.0/16"]
  }'
```
</details>

**Configuracao via Web UI:**

1. Navegue para a aba Cloud → Advanced Configuration
2. Na secao "Webhook Security", adicione os ranges de IP
3. Use notacao CIDR (ex.: `10.0.0.0/8`) ou IPs individuais (`1.2.3.4/32`)

**Formato de IP:**

| Formato | Exemplo | Descricao |
|--------|---------|-------------|
| IP unico | `1.2.3.4/32` | Correspondencia exata de IP |
| Sub-rede | `10.0.0.0/8` | Rede Classe A |
| Range estreito | `192.168.1.0/24` | 256 enderecos |

**IPs de webhook por provedor:**
- [IPs de webhook do Replicate](REPLICATE.md#webhook-ips)

**Comportamento:**

| Cenario | Resultado |
|----------|--------|
| Nenhuma allowlist configurada | Todos os IPs aceitos |
| Array vazio `[]` | Todos os IPs aceitos |
| IP na allowlist | Webhook processado |
| IP fora da allowlist | 403 Forbidden |

**Audit logging:**

Webhooks rejeitados sao registrados no audit trail:

```bash
curl "http://localhost:8080/api/audit?event_type=webhook_rejected&limit=100"
```

## Configuracao de proxy

### Variaveis de ambiente

<details>
<summary>Variaveis de ambiente de proxy</summary>

```bash
# HTTP/HTTPS proxy
export HTTPS_PROXY="http://proxy.corp.example.com:8080"
export HTTP_PROXY="http://proxy.corp.example.com:8080"

# Custom CA bundle for corporate CAs
export SIMPLETUNER_CA_BUNDLE="/etc/pki/tls/certs/ca-bundle.crt"

# Disable SSL verification (NOT recommended for production)
export SIMPLETUNER_SSL_VERIFY="false"

# HTTP timeout (seconds)
export SIMPLETUNER_HTTP_TIMEOUT="60"
```
</details>

### Via configuracao do provedor

<details>
<summary>Configuracao via API</summary>

```python
# Via API
PUT /api/cloud/providers/{provider}
{
    "ssl_verify": true,
    "ssl_ca_bundle": "/etc/pki/tls/certs/corporate-ca.crt",
    "proxy_url": "http://proxy:8080",
    "http_timeout": 60.0
}
```
</details>

### Via Web UI (Advanced Configuration)

A aba Cloud inclui um painel de Advanced Configuration para configuracoes de rede:

| Configuracao | Descricao |
|---------|-------------|
| **SSL Verification** | Toggle para habilitar/desabilitar verificacao de certificado |
| **CA Bundle Path** | Bundle de CA customizado para CAs corporativas |
| **Proxy URL** | Proxy HTTP para conexoes de saida |
| **HTTP Timeout** | Timeout de requisicao em segundos (padrao: 30) |

#### Bypass de verificacao SSL

Desabilitar verificacao SSL exige confirmacao explicita devido a implicacoes de seguranca:

1. Clique no toggle de SSL Verification para desabilitar
2. Um dialogo de confirmacao aparece: *"Disabling SSL verification is a security risk. Only do this if you have a self-signed certificate or are behind a corporate proxy. Continue?"*
3. Clique em "OK" para confirmar e salvar a configuracao

A confirmacao e valida por sessao. Alternancias subsequentes na mesma sessao nao exigem nova confirmacao.

#### Configuracao de proxy corporativo

Para ambientes que usam proxies HTTP:

1. Navegue para a aba Cloud → Advanced Configuration
2. Informe a URL do proxy (ex.: `http://proxy.corp.example.com:8080`)
3. Opcionalmente defina um CA bundle customizado se seu proxy fizer inspecao TLS
4. Ajuste o timeout HTTP se seu proxy adicionar latencia

As configuracoes sao salvas imediatamente ao mudar e se aplicam a todas as chamadas de API do provedor.

## Monitoramento de health

### Endpoints

| Endpoint | Proposito | Resposta |
|----------|---------|----------|
| `/api/cloud/health` | Health check completo | JSON com status de componentes |
| `/api/cloud/health/live` | Liveness do Kubernetes | `{"status": "ok"}` |
| `/api/cloud/health/ready` | Readiness do Kubernetes | `{"status": "ready"}` ou 503 |

### Resposta de health check

<details>
<summary>Exemplo de resposta</summary>

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "timestamp": "2024-01-15T10:30:00Z",
  "components": [
    {
      "name": "database",
      "status": "healthy",
      "latency_ms": 1.2,
      "message": "SQLite database accessible"
    },
    {
      "name": "secrets",
      "status": "healthy",
      "message": "API token configured"
    }
  ]
}
```
</details>

Incluir checagens do provedor (adiciona latencia):
```
GET /api/cloud/health?include_providers=true
```

## Metricas Prometheus

Endpoint de scrape: `/api/cloud/metrics/prometheus`

### Habilitando exportacao Prometheus

A exportacao Prometheus vem desabilitada por padrao. Habilite via a aba Metrics no painel Admin ou via API:

<details>
<summary>Habilitar via API</summary>

```bash
curl -X PUT http://localhost:8080/api/cloud/metrics/config \
  -H "Content-Type: application/json" \
  -d '{"prometheus_enabled": true, "prometheus_categories": ["jobs", "http", "health"]}'
```
</details>

### Categorias de metricas

Metricas sao organizadas em categorias que podem ser habilitadas individualmente:

| Categoria | Descricao | Principais metricas |
|----------|-------------|-------------|
| `jobs` | Contagem de jobs, status, profundidade de fila, custos | `simpletuner_jobs_total`, `simpletuner_cost_usd_total` |
| `http` | Contagem de requisicoes, erros, latencia | `simpletuner_http_requests_total`, `simpletuner_http_errors_total` |
| `rate_limits` | Violacoes de rate limit | `simpletuner_rate_limit_violations_total` |
| `approvals` | Metricas de workflow de aprovacao | `simpletuner_approval_requests_pending` |
| `audit` | Atividade de audit log | `simpletuner_audit_log_entries_total` |
| `health` | Uptime do servidor, health de componentes | `simpletuner_uptime_seconds`, `simpletuner_health_database_latency_ms` |
| `circuit_breakers` | Estado do circuit breaker do provedor | `simpletuner_circuit_breaker_state` |
| `provider` | Limites de custo, saldo de creditos | `simpletuner_cost_limit_percent_used` |

### Templates de configuracao

Templates de inicio rapido para casos comuns:

| Template | Categorias | Caso de uso |
|----------|------------|----------|
| `minimal` | jobs | Monitoramento leve de jobs |
| `standard` | jobs, http, health | Padrao recomendado |
| `security` | jobs, http, rate_limits, audit, approvals | Monitoramento de seguranca |
| `full` | Todas as categorias | Observabilidade completa |

<details>
<summary>Aplicar um template</summary>

```bash
curl -X POST http://localhost:8080/api/cloud/metrics/config/templates/standard
```
</details>

### Metricas disponiveis

<details>
<summary>Referencia de metricas</summary>

```
# Server uptime
simpletuner_uptime_seconds 3600.5

# Job metrics
simpletuner_jobs_total 150
simpletuner_jobs_by_status{status="completed"} 120
simpletuner_jobs_by_status{status="failed"} 10
simpletuner_jobs_by_status{status="running"} 5
simpletuner_jobs_active 8
simpletuner_cost_usd_total 450.25
simpletuner_job_duration_seconds_avg 1800.5

# HTTP metrics
simpletuner_http_requests_total{endpoint="POST_/api/cloud/jobs/submit"} 50
simpletuner_http_errors_total{endpoint_status="POST_/api/cloud/jobs/submit_500"} 2
simpletuner_http_request_latency_ms_avg{endpoint="POST_/api/cloud/jobs/submit"} 250.5

# Rate limiting
simpletuner_rate_limit_violations_total 15
simpletuner_rate_limit_tracked_clients 42

# Approvals
simpletuner_approval_requests_pending 3
simpletuner_approval_requests_by_status{status="approved"} 25

# Audit
simpletuner_audit_log_entries_total 1500
simpletuner_audit_log_entries_24h 120

# Circuit breakers (per provider)
simpletuner_circuit_breaker_state{provider="..."} 0
simpletuner_circuit_breaker_failures_total{provider="..."} 5

# Provider status (per provider)
simpletuner_cost_limit_percent_used{provider="..."} 45.5
simpletuner_credit_balance_usd{provider="..."} 150.00
```
</details>

### Configuracao do Prometheus

<details>
<summary>Configuracao de scrape prometheus.yml</summary>

```yaml
scrape_configs:
  - job_name: 'simpletuner'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/api/cloud/metrics/prometheus'
    scrape_interval: 30s
```
</details>

### Preview do output de metricas

Veja o que sera exportado sem alterar a configuracao:

```bash
curl "http://localhost:8080/api/cloud/metrics/config/preview?categories=jobs&categories=health"
```

## Rate limiting

### Visao geral

O SimpleTuner inclui rate limiting embutido para proteger contra abuso e garantir uso justo de recursos. Limites sao aplicados por IP com regras configuraveis para endpoints diferentes.

### Configuracao

O rate limiting pode ser configurado via variaveis de ambiente:

<details>
<summary>Variaveis de ambiente</summary>

```bash
# Default rate limit for unmatched endpoints
export RATE_LIMIT_CALLS=100      # Requests per period
export RATE_LIMIT_PERIOD=60      # Period in seconds

# Set to 0 to disable rate limiting entirely
export RATE_LIMIT_CALLS=0
```
</details>

### Regras padrao de rate limit

Endpoints diferentes possuem limites diferentes com base em sensibilidade:

| Padrao de endpoint | Limite | Periodo | Metodos | Motivo |
|------------------|-------|--------|---------|--------|
| `/api/auth/login` | 5 | 60s | POST | Protecao contra brute force |
| `/api/auth/register` | 3 | 60s | POST | Abuso de cadastro |
| `/api/auth/api-keys` | 10 | 60s | POST | Criacao de API keys |
| `/api/cloud/jobs` | 20 | 60s | POST | Envio de jobs |
| `/api/cloud/jobs/.+/cancel` | 30 | 60s | POST | Cancelamento de jobs |
| `/api/webhooks/` | 100 | 60s | All | Entrega de webhooks |
| `/api/cloud/storage/` | 50 | 60s | All | Uploads de storage |
| `/api/quotas/` | 30 | 60s | All | Operacoes de quota |
| Todos os outros endpoints | 100 | 60s | All | Fallback padrao |

### Caminhos excluidos

Os caminhos a seguir sao excluidos do rate limiting:

- `/health` - Health checks
- `/api/events/stream` - Conexoes SSE
- `/static/` - Arquivos estaticos
- `/api/cloud/hints` - Dicas de UI (nao sensivel)
- `/api/users/me` - Checagem do usuario atual
- `/api/cloud/providers` - Lista de provedores

### Headers de resposta

Todas as respostas incluem headers de rate limit:

```
X-RateLimit-Limit: 100        # Maximum requests allowed
X-RateLimit-Remaining: 95     # Requests remaining in period
X-RateLimit-Reset: 1705320000 # Unix timestamp when limit resets
```

<details>
<summary>Resposta de rate limit excedido</summary>

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 45
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1705320045

{"detail": "Rate limit exceeded. Please try again later."}
```
</details>

### Deteccao de IP do cliente

O middleware lida corretamente com headers de proxy:

1. `X-Forwarded-For` - Header padrao de proxy (o primeiro IP e o cliente)
2. `X-Real-IP` - Header de proxy do Nginx
3. IP de conexao direta - Fallback

Rate limits sao ignorados para localhost (`127.0.0.1`, `::1`) em desenvolvimento.

### Audit logging

Violacoes de rate limit sao registradas no audit trail com:
- Endereco IP do cliente
- Endpoint requisitado
- Metodo HTTP
- Header User-Agent

Consultar audit logs para eventos de rate limit:

```bash
curl "http://localhost:8080/api/audit?event_type=rate_limited&limit=100"
```

### Regras customizadas de rate limit

<details>
<summary>Configuracao programatica</summary>

```python
from simpletuner.simpletuner_sdk.server.middleware.security_middleware import (
    RateLimitMiddleware,
)

# Custom rules: (pattern, calls, period, methods)
custom_rules = [
    (r"^/api/cloud/expensive$", 5, 300, ["POST"]),  # 5 per 5 minutes
    (r"^/api/cloud/public$", 1000, 60, None),       # 1000 per minute for all methods
]

app.add_middleware(
    RateLimitMiddleware,
    calls=100,           # Default limit
    period=60,           # Default period
    rules=custom_rules,  # Custom rules
    enable_audit=True,   # Log violations
)
```
</details>

### Rate limiting distribuido (Async Rate Limiter)

Para deploys multi-worker, o SimpleTuner fornece um rate limiter distribuido que usa o backend de estado configurado (SQLite, Redis, PostgreSQL ou MySQL) para compartilhar o estado de rate limit entre workers.

<details>
<summary>Obtendo um rate limiter</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.container import get_rate_limiter

# Create a rate limiter with sliding window
limiter = await get_rate_limiter(
    max_requests=100,    # Maximum requests in window
    window_seconds=60,   # Window duration
    key_prefix="api",    # Optional prefix for keys
)

# Check if a request should be allowed
allowed = await limiter.check("user:123")
if not allowed:
    raise RateLimitExceeded()

# Or use context manager for automatic tracking
async with limiter.track("user:123") as allowed:
    if not allowed:
        return Response(status_code=429)
    # Process request...
```
</details>

**Algoritmo de janela deslizante:**

O rate limiter usa um algoritmo de janela deslizante que fornece limitacao mais suave que janelas fixas:

```
Time:     |----60s window----|
Requests: [x x x x x][x x x]
          ↑ expired  ↑ counted
```

- Requisicoes sao timestampadas quando chegam
- Apenas requisicoes dentro da janela sao contadas
- Requisicoes antigas expiram e sao removidas automaticamente
- Sem o problema de "burst na borda da janela"

**Comportamento por backend:**

| Backend | Implementacao | Performance | Multi-Worker |
|---------|---------------|-------------|--------------|
| SQLite | Array JSON de timestamps | Boa | Lock de arquivo unico |
| Redis | Sorted set (ZSET) | Excelente | Suporte total |
| PostgreSQL | JSONB com indice | Muito boa | Suporte total |
| MySQL | Coluna JSON | Boa | Suporte total |

<details>
<summary>Rate limiters pre-configurados</summary>

```python
from simpletuner.simpletuner_sdk.server.routes.cloud._shared import (
    webhook_rate_limiter,  # 100 req/min for webhooks
    s3_rate_limiter,       # 50 req/min for S3 uploads
)

# Use in route handlers
@router.post("/webhooks/{provider}")
async def handle_webhook(request: Request):
    client_ip = request.client.host
    if not await webhook_rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    # Process webhook...
```
</details>

<details>
<summary>Monitorando uso de rate limit</summary>

```python
# Get current usage for a key
usage = await limiter.get_usage("user:123")
print(f"Requests in window: {usage['count']}/{usage['limit']}")
print(f"Window resets in: {usage['reset_in_seconds']}s")
```
</details>

## Endpoint de armazenamento (compativel com S3)

### Visao geral

O SimpleTuner fornece um endpoint compativel com S3 para enviar outputs de treinamento (checkpoints, amostras, logs) de jobs de treinamento em nuvem de volta para sua maquina local. Isso permite que jobs em nuvem "liguem para casa" com resultados.

### Arquitetura

```
┌─────────────────────┐          ┌─────────────────────┐
│   Cloud Training    │          │   Local SimpleTuner │
│   Job               │ ──────── │   Server            │
│                     │   HTTPS  │                     │
│   Uploads outputs   │          │ /api/cloud/storage/*│
│   via S3 protocol   │          │                     │
└─────────────────────┘          └─────────────────────┘
                                         │
                                         ▼
                                ┌─────────────────────┐
                                │   Local Filesystem  │
                                │   ~/.simpletuner/   │
                                │   outputs/{job_id}/ │
                                └─────────────────────┘
```

### Requisitos

Para jobs em nuvem fazerem upload para seu servidor local, voce precisa:

1. **Endpoint HTTPS publico** - Provedores de nuvem nao alcancam `localhost`
2. **Certificado SSL** - A maioria dos provedores exige HTTPS
3. **Acesso no firewall** - Permita conexoes de entrada na porta escolhida

### Opcao 1: Cloudflared Tunnel (recomendado)

[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/) fornece um tunel seguro sem abrir portas no firewall.

<details>
<summary>Instrucoes de setup</summary>

```bash
# Install cloudflared
# macOS
brew install cloudflared

# Linux
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared
sudo mv cloudflared /usr/local/bin/

# Create a tunnel (requires Cloudflare account)
cloudflared tunnel login
cloudflared tunnel create simpletuner

# Get your tunnel ID
cloudflared tunnel list
```

**Configuracao (`~/.cloudflared/config.yml`):**

```yaml
tunnel: YOUR_TUNNEL_ID
credentials-file: ~/.cloudflared/YOUR_TUNNEL_ID.json

ingress:
  - hostname: simpletuner.yourdomain.com
    service: http://localhost:8001
  - service: http_status:404
```

**Rodar o tunel:**

```bash
# Start the tunnel
cloudflared tunnel run simpletuner

# Or run as a service
sudo cloudflared service install
```

**Configure o SimpleTuner:**

```bash
# Set the public URL for S3 uploads
export SIMPLETUNER_PUBLIC_URL="https://simpletuner.yourdomain.com"
```

Ou via a aba Cloud → Advanced Configuration → Public URL.
</details>

### Opcao 2: ngrok

[ngrok](https://ngrok.com/) fornece tuneis rapidos para desenvolvimento.

<details>
<summary>Instrucoes de setup</summary>

```bash
# Install ngrok
# macOS
brew install ngrok

# Linux
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# Authenticate (requires ngrok account)
ngrok config add-authtoken YOUR_TOKEN
```

**Iniciar o tunel:**

```bash
# Start ngrok tunnel to SimpleTuner port
ngrok http 8001

# Note the HTTPS URL from the output:
# Forwarding: https://abc123.ngrok.io -> http://localhost:8001
```

**Configure o SimpleTuner:**

```bash
export SIMPLETUNER_PUBLIC_URL="https://abc123.ngrok.io"
```

**Nota:** URLs gratuitas do ngrok mudam a cada reinicio. Para producao, use um plano pago com dominios reservados ou Cloudflared.
</details>

### Opcao 3: IP publico direto

<details>
<summary>Instrucoes de setup</summary>

Se seu servidor tem um IP publico e voce pode abrir portas no firewall:

```bash
# Allow inbound HTTPS
sudo ufw allow 8001/tcp

# Or with iptables
sudo iptables -A INPUT -p tcp --dport 8001 -j ACCEPT
```

**Setup de certificado SSL:**

Para producao, use Let's Encrypt:

```bash
# Install certbot
sudo apt install certbot

# Get certificate (requires DNS pointing to your IP)
sudo certbot certonly --standalone -d simpletuner.yourdomain.com

# Configure SimpleTuner
export SIMPLETUNER_SSL_CERT="/etc/letsencrypt/live/simpletuner.yourdomain.com/fullchain.pem"
export SIMPLETUNER_SSL_KEY="/etc/letsencrypt/live/simpletuner.yourdomain.com/privkey.pem"
export SIMPLETUNER_PUBLIC_URL="https://simpletuner.yourdomain.com:8001"
```
</details>

### Configuracao do endpoint de armazenamento

Configure o comportamento do endpoint S3 via configuracoes do provedor:

<details>
<summary>Configuracao via API</summary>

```bash
curl -X PUT http://localhost:8001/api/cloud/providers/{provider} \
  -H "Content-Type: application/json" \
  -d '{
    "s3_endpoint_enabled": true,
    "s3_public_url": "https://simpletuner.yourdomain.com",
    "s3_output_path": "~/.simpletuner/outputs"
  }'
```
</details>

Ou via a aba Cloud → Advanced Configuration.

### Autenticacao de upload

Uploads S3 sao autenticados usando tokens de upload de curta duracao:

1. Quando um job e enviado, um token de upload unico e gerado
2. O token e passado ao job em nuvem como variavel de ambiente
3. O job usa o token como access key S3 ao enviar
4. Tokens expiram apos o job concluir ou ser cancelado

### Operacoes S3 suportadas

| Operacao | Endpoint | Descricao |
|-----------|----------|-------------|
| PUT Object | `PUT /api/cloud/storage/{bucket}/{key}` | Fazer upload de um arquivo |
| GET Object | `GET /api/cloud/storage/{bucket}/{key}` | Baixar um arquivo |
| List Objects | `GET /api/cloud/storage/{bucket}` | Listar objetos no bucket |
| List Buckets | `GET /api/cloud/storage` | Listar todos os buckets |

### Solucao de problemas de uploads de storage

**Uploads falham com "Unauthorized":**
- Verifique se o token de upload esta sendo passado corretamente
- Verifique se o ID do job corresponde ao token
- Garanta que o job ainda esteja em estado ativo (nao concluido/cancelado)

**Uploads com timeout:**
- Verifique se o tunel esta rodando (`cloudflared tunnel run` ou `ngrok http`)
- Verifique se a URL publica esta acessivel pela internet
- Teste com: `curl -I https://your-public-url/api/cloud/health`

**Erros de certificado SSL:**
- Para ngrok/cloudflared, o SSL e tratado automaticamente
- Para conexoes diretas, garanta que o certificado seja valido
- Verifique se certificados intermediarios estao incluidos na cadeia

<details>
<summary>Testes de firewall e conectividade</summary>

```bash
# Test local connectivity
curl http://localhost:8001/api/cloud/health

# Test from external (if direct IP)
curl https://your-public-ip:8001/api/cloud/health
```
</details>

**Ver progresso de upload:**

```bash
# Check current uploads
curl http://localhost:8001/api/cloud/jobs/{job_id}

# Response includes upload_progress
```

## Logging estruturado

### Configuracao

<details>
<summary>Variaveis de ambiente</summary>

```bash
# Log level: DEBUG, INFO, WARNING, ERROR
export SIMPLETUNER_LOG_LEVEL="INFO"

# Format: "json" or "text"
export SIMPLETUNER_LOG_FORMAT="json"

# Optional file output
export SIMPLETUNER_LOG_FILE="/var/log/simpletuner/cloud.log"
```
</details>

### Formato de log JSON

<details>
<summary>Exemplo de entrada de log</summary>

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "INFO",
  "logger": "simpletuner.cloud.jobs",
  "message": "Job submitted",
  "correlation_id": "abc123def456",
  "source": {
    "file": "jobs.py",
    "line": 350,
    "function": "submit_job"
  },
  "extra": {
    "job_id": "xyz789",
    "provider": "..."
  }
}
```
</details>

### Configuracao programatica

<details>
<summary>Configuracao em Python</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.structured_logging import (
    configure_structured_logging,
    get_logger,
    LogContext,
)

# Configure logging
configure_structured_logging(
    level="INFO",
    json_output=True,
    log_file="/var/log/simpletuner/cloud.log",
)

# Get a logger
logger = get_logger("mymodule")

# Log with context
with LogContext(job_id="abc123", provider="..."):
    logger.info("Processing job")  # Includes job_id and provider
```
</details>

## Backup e restore

### Localizacao do banco de dados

O banco SQLite fica em:
```
~/.simpletuner/config/cloud/jobs.db
```

Com arquivos WAL:
```
~/.simpletuner/config/cloud/jobs.db-wal
~/.simpletuner/config/cloud/jobs.db-shm
```

### Backup via linha de comando

<details>
<summary>Comandos de backup</summary>

```bash
# Simple copy (stop server first for consistency)
cp ~/.simpletuner/config/cloud/jobs.db /backup/jobs_$(date +%Y%m%d).db

# Online backup with sqlite3
sqlite3 ~/.simpletuner/config/cloud/jobs.db ".backup /backup/jobs.db"
```
</details>

### Backup programatico

<details>
<summary>API Python</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud import JobStore

store = JobStore()

# Create timestamped backup
backup_path = store.backup()
print(f"Backup created: {backup_path}")

# Custom backup path
backup_path = store.backup("/backup/custom_backup.db")

# List available backups
backups = store.list_backups()
for b in backups:
    print(f"  {b.name}: {b.stat().st_size / 1024:.1f} KB")

# Get database info
info = store.get_database_info()
print(f"Database: {info['size_mb']} MB, {info['job_count']} jobs")
```
</details>

### Restore

<details>
<summary>Restaurar de backup</summary>

```python
from pathlib import Path
from simpletuner.simpletuner_sdk.server.services.cloud import JobStore

store = JobStore()

# WARNING: This overwrites the current database!
success = store.restore(Path("/backup/jobs_backup_20240115_103000.db"))
```
</details>

### Script de backup automatizado

<details>
<summary>Script de backup via cron</summary>

```bash
#!/bin/bash
# /etc/cron.daily/simpletuner-backup

BACKUP_DIR="/backup/simpletuner"
RETENTION_DAYS=30
DB_PATH="$HOME/.simpletuner/config/cloud/jobs.db"

mkdir -p "$BACKUP_DIR"

# Create backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/jobs_backup_$TIMESTAMP.db"

sqlite3 "$DB_PATH" ".backup '$BACKUP_FILE'"

# Compress
gzip "$BACKUP_FILE"

# Remove old backups
find "$BACKUP_DIR" -name "jobs_backup_*.db.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup created: ${BACKUP_FILE}.gz"
```
</details>

## Gerenciamento de segredos

Veja [SECRETS_AND_CACHE.md](SECRETS_AND_CACHE.md) para documentacao detalhada sobre provedores de segredos.

### Provedores suportados

1. **Variaveis de ambiente** (padrao)
2. **Segredos em arquivo** (`~/.simpletuner/secrets.json` ou YAML)
3. **AWS Secrets Manager** (requer `pip install boto3`)
4. **HashiCorp Vault** (requer `pip install hvac`)

### Prioridade dos provedores

Segredos sao resolvidos em ordem:
1. Variaveis de ambiente (maior prioridade - permite overrides)
2. Segredos em arquivo
3. AWS Secrets Manager
4. HashiCorp Vault

## Solucao de problemas

### Problemas de conexao

**Proxy nao funciona:**

<details>
<summary>Depurar conectividade de proxy</summary>

```bash
# Test proxy connectivity
curl -x http://proxy:8080 https://your-provider-api-endpoint

# Check environment
env | grep -i proxy
```
</details>

**Erros de certificado SSL:**

<details>
<summary>Depurar problemas de SSL</summary>

```bash
# Test with custom CA
curl --cacert /path/to/ca.crt https://your-provider-api-endpoint

# Verify CA bundle
openssl verify -CAfile /path/to/ca.crt server.crt
```
</details>

**Troubleshooting especifico do provedor:**
- [Troubleshooting Replicate](REPLICATE.md#troubleshooting)

### Problemas no banco de dados

**Banco bloqueado:**

<details>
<summary>Resolucao de lock do banco</summary>

```bash
# Check for open connections
fuser ~/.simpletuner/config/cloud/jobs.db

# Force WAL checkpoint
sqlite3 ~/.simpletuner/config/cloud/jobs.db "PRAGMA wal_checkpoint(TRUNCATE)"
```
</details>

**Banco corrompido:**

<details>
<summary>Recuperacao do banco</summary>

```bash
# Check integrity
sqlite3 ~/.simpletuner/config/cloud/jobs.db "PRAGMA integrity_check"

# Recover (creates new database from good pages)
sqlite3 ~/.simpletuner/config/cloud/jobs.db ".recover" | sqlite3 jobs_recovered.db
```
</details>

### Falhas de health check

<details>
<summary>Debug de health check</summary>

```bash
# Test health endpoint
curl -s http://localhost:8080/api/cloud/health | jq .

# Check with provider checks included
curl -s 'http://localhost:8080/api/cloud/health?include_providers=true' | jq .
```
</details>
