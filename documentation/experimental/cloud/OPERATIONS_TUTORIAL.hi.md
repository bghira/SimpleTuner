# Cloud Training Operations Guide

यह दस्तावेज़ SimpleTuner के cloud training फीचर के लिए production deployment और operations को कवर करता है, खासकर मौजूदा DevOps infrastructure के साथ पूरी integration पर फोकस के साथ।

## Network Architecture

### Outbound Connections

सर्वर कॉन्फ़िगर किए गए cloud providers के लिए outbound HTTPS connections बनाता है। हर provider के अपने API endpoints और requirements होते हैं।

**Provider-specific network details:**
- [Replicate API Endpoints](REPLICATE.md#api-endpoints)

### Inbound Connections

| Source | Endpoint | Purpose |
|--------|----------|---------|
| Cloud provider infrastructure | `/api/webhooks/{provider}` | Job status updates |
| Cloud training job | `/api/cloud/storage/{bucket}/{key}` | Training outputs upload करना |
| Monitoring systems | `/api/cloud/health`, `/api/cloud/metrics/prometheus` | Health और metrics |

### Firewall Rules

Firewall requirements आपके configured provider(s) पर निर्भर करती हैं।

**Provider-specific firewall rules:**
- [Replicate Firewall Configuration](REPLICATE.md#firewall)

### Webhook IP Allowlisting

सुरक्षा बढ़ाने के लिए आप webhook delivery को specific IP ranges तक सीमित कर सकते हैं। कॉन्फ़िगर होने पर allowlist के बाहर की IPs से आए webhooks 403 Forbidden के साथ reject हो जाते हैं।

**API से कॉन्फ़िगरेशन:**

<details>
<summary>API configuration example</summary>

```bash
# Set allowed IPs for a provider's webhooks
curl -X PUT http://localhost:8080/api/cloud/providers/{provider} \
  -H "Content-Type: application/json" \
  -d '{
    "webhook_allowed_ips": ["10.0.0.0/8", "192.168.0.0/16"]
  }'
```
</details>

**Web UI से कॉन्फ़िगरेशन:**

1. Cloud tab → Advanced Configuration पर जाएं
2. "Webhook Security" सेक्शन में IP ranges जोड़ें
3. CIDR notation (जैसे `10.0.0.0/8`) या single IPs (`1.2.3.4/32`) उपयोग करें

**IP Format:**

| Format | Example | Description |
|--------|---------|-------------|
| Single IP | `1.2.3.4/32` | Exact IP match |
| Subnet | `10.0.0.0/8` | Class A network |
| Narrow range | `192.168.1.0/24` | 256 addresses |

**Provider-specific webhook IPs:**
- [Replicate Webhook IPs](REPLICATE.md#webhook-ips)

**Behavior:**

| Scenario | Result |
|----------|--------|
| No allowlist configured | सभी IPs accepted |
| Empty array `[]` | सभी IPs accepted |
| IP in allowlist | Webhook process होगा |
| IP not in allowlist | 403 Forbidden |

**Audit Logging:**

Rejected webhooks audit trail में log होते हैं:

```bash
curl "http://localhost:8080/api/audit?event_type=webhook_rejected&limit=100"
```

## Proxy Configuration

### Environment Variables

<details>
<summary>Proxy environment variables</summary>

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

### Provider Config के जरिए

<details>
<summary>API configuration</summary>

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

### Web UI के जरिए (Advanced Configuration)

Cloud tab में network settings के लिए Advanced Configuration panel है:

| Setting | Description |
|---------|-------------|
| **SSL Verification** | Certificate verification enable/disable toggle |
| **CA Bundle Path** | Corporate CAs के लिए custom certificate bundle |
| **Proxy URL** | Outbound connections के लिए HTTP proxy |
| **HTTP Timeout** | Request timeout seconds में (default: 30) |

#### SSL Verification Bypass

SSL verification बंद करना सुरक्षा जोखिम है इसलिए explicit acknowledgment चाहिए:

1. SSL Verification toggle बंद करें
2. एक confirmation dialog आएगा: *"Disabling SSL verification is a security risk. Only do this if you have a self-signed certificate or are behind a corporate proxy. Continue?"*
3. "OK" क्लिक करके पुष्टि करें और setting सेव करें

यह acknowledgment session-scoped है। उसी session में दोबारा toggle करने पर re-confirmation नहीं मांगा जाएगा।

#### Corporate Proxy Configuration

HTTP proxies वाले environments के लिए:

1. Cloud tab → Advanced Configuration पर जाएं
2. Proxy URL दर्ज करें (जैसे `http://proxy.corp.example.com:8080`)
3. अगर proxy TLS inspection करता है तो custom CA bundle सेट करें
4. अगर proxy latency जोड़ता है तो HTTP timeout समायोजित करें

Settings बदलते ही तुरंत सेव हो जाती हैं और आगे की सभी provider API calls पर लागू होती हैं।

## Health Monitoring

### Endpoints

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `/api/cloud/health` | Full health check | component status के साथ JSON |
| `/api/cloud/health/live` | Kubernetes liveness | `{"status": "ok"}` |
| `/api/cloud/health/ready` | Kubernetes readiness | `{"status": "ready"}` या 503 |

### Health Check Response

<details>
<summary>Example response</summary>

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

Provider API checks शामिल करने के लिए (latency बढ़ाता है):
```
GET /api/cloud/health?include_providers=true
```

## Prometheus Metrics

Scrape endpoint: `/api/cloud/metrics/prometheus`

### Prometheus Export Enable करना

Prometheus export डिफ़ॉल्ट रूप से disabled है। इसे Admin panel के Metrics tab या API से enable करें:

<details>
<summary>Enable via API</summary>

```bash
curl -X PUT http://localhost:8080/api/cloud/metrics/config \
  -H "Content-Type: application/json" \
  -d '{"prometheus_enabled": true, "prometheus_categories": ["jobs", "http", "health"]}'
```
</details>

### Metric Categories

Metrics categories में व्यवस्थित हैं जिन्हें अलग-अलग enable किया जा सकता है:

| Category | Description | Key Metrics |
|----------|-------------|-------------|
| `jobs` | Job counts, status, queue depth, costs | `simpletuner_jobs_total`, `simpletuner_cost_usd_total` |
| `http` | Request counts, errors, latency | `simpletuner_http_requests_total`, `simpletuner_http_errors_total` |
| `rate_limits` | Rate limit violations | `simpletuner_rate_limit_violations_total` |
| `approvals` | Approval workflow metrics | `simpletuner_approval_requests_pending` |
| `audit` | Audit log activity | `simpletuner_audit_log_entries_total` |
| `health` | Server uptime, component health | `simpletuner_uptime_seconds`, `simpletuner_health_database_latency_ms` |
| `circuit_breakers` | Provider circuit breaker state | `simpletuner_circuit_breaker_state` |
| `provider` | Cost limits, credit balance | `simpletuner_cost_limit_percent_used` |

### Configuration Templates

Common use cases के लिए quick-start templates:

| Template | Categories | Use Case |
|----------|------------|----------|
| `minimal` | jobs | Lightweight job monitoring |
| `standard` | jobs, http, health | Recommended default |
| `security` | jobs, http, rate_limits, audit, approvals | Security monitoring |
| `full` | All categories | Complete observability |

<details>
<summary>Apply a template</summary>

```bash
curl -X POST http://localhost:8080/api/cloud/metrics/config/templates/standard
```
</details>

### Available Metrics

<details>
<summary>Metrics reference</summary>

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

### Prometheus Configuration

<details>
<summary>prometheus.yml scrape config</summary>

```yaml
scrape_configs:
  - job_name: 'simpletuner'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/api/cloud/metrics/prometheus'
    scrape_interval: 30s
```
</details>

### Metrics Output Preview

Configuration बदले बिना क्या export होगा यह देखने के लिए:

```bash
curl "http://localhost:8080/api/cloud/metrics/config/preview?categories=jobs&categories=health"
```

## Rate Limiting

### Overview

SimpleTuner में built-in rate limiting है जो abuse से बचाव और fair resource usage के लिए है। Rate limits per-IP लागू होते हैं और अलग-अलग endpoints के लिए configurable rules हैं।

### Configuration

Rate limiting environment variables से कॉन्फ़िगर किया जा सकता है:

<details>
<summary>Environment variables</summary>

```bash
# Default rate limit for unmatched endpoints
export RATE_LIMIT_CALLS=100      # Requests per period
export RATE_LIMIT_PERIOD=60      # Period in seconds

# Set to 0 to disable rate limiting entirely
export RATE_LIMIT_CALLS=0
```
</details>

### Default Rate Limit Rules

अलग-अलग endpoints की संवेदनशीलता के आधार पर rate limits अलग हैं:

| Endpoint Pattern | Limit | Period | Methods | Reason |
|------------------|-------|--------|---------|--------|
| `/api/auth/login` | 5 | 60s | POST | Brute force protection |
| `/api/auth/register` | 3 | 60s | POST | User registration abuse |
| `/api/auth/api-keys` | 10 | 60s | POST | API key creation |
| `/api/cloud/jobs` | 20 | 60s | POST | Job submission |
| `/api/cloud/jobs/.+/cancel` | 30 | 60s | POST | Job cancellation |
| `/api/webhooks/` | 100 | 60s | All | Webhook delivery |
| `/api/cloud/storage/` | 50 | 60s | All | Storage uploads |
| `/api/quotas/` | 30 | 60s | All | Quota operations |
| All other endpoints | 100 | 60s | All | Default fallback |

### Excluded Paths

इन paths पर rate limiting लागू नहीं होती:

- `/health` - Health checks
- `/api/events/stream` - SSE connections
- `/static/` - Static files
- `/api/cloud/hints` - UI hints (security-sensitive नहीं)
- `/api/users/me` - Current user check
- `/api/cloud/providers` - Provider list

### Response Headers

हर response में rate limit headers शामिल होते हैं:

```
X-RateLimit-Limit: 100        # Maximum requests allowed
X-RateLimit-Remaining: 95     # Requests remaining in period
X-RateLimit-Reset: 1705320000 # Unix timestamp when limit resets
```

<details>
<summary>Rate limit exceeded response</summary>

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 45
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1705320045

{"detail": "Rate limit exceeded. Please try again later."}
```
</details>

### Client IP Detection

Middleware proxy headers सही से हैंडल करता है:

1. `X-Forwarded-For` - Standard proxy header (पहला IP client होता है)
2. `X-Real-IP` - Nginx proxy header
3. Direct connection IP - Fallback

Development में localhost (`127.0.0.1`, `::1`) के लिए rate limits bypass होती हैं।

### Audit Logging

Rate limit violations audit trail में log होते हैं, जिसमें शामिल है:
- Client IP address
- Requested endpoint
- HTTP method
- User-Agent header

Rate limit events के audit logs देखें:

```bash
curl "http://localhost:8080/api/audit?event_type=rate_limited&limit=100"
```

### Custom Rate Limit Rules

<details>
<summary>Programmatic configuration</summary>

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

### Distributed Rate Limiting (Async Rate Limiter)

Multi-worker deployments के लिए, SimpleTuner distributed rate limiter देता है जो configured state backend (SQLite, Redis, PostgreSQL, या MySQL) से workers के बीच rate limit state शेयर करता है।

<details>
<summary>Getting a rate limiter</summary>

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

**Sliding Window Algorithm:**

Rate limiter sliding window algorithm इस्तेमाल करता है जो fixed windows से smoother rate limiting देता है:

```
Time:     |----60s window----|
Requests: [x x x x x][x x x]
          ↑ expired  ↑ counted
```

- Requests timestamped होते हैं जब वे आते हैं
- केवल window के भीतर वाले requests गिने जाते हैं
- पुराने requests expire होकर अपने आप prune हो जाते हैं
- "burst at window boundary" समस्या नहीं होती

**Backend-Specific Behavior:**

| Backend | Implementation | Performance | Multi-Worker |
|---------|---------------|-------------|--------------|
| SQLite | JSON timestamp array | Good | Single file lock |
| Redis | Sorted set (ZSET) | Excellent | Full support |
| PostgreSQL | JSONB with index | Very Good | Full support |
| MySQL | JSON column | Good | Full support |

<details>
<summary>Pre-configured rate limiters</summary>

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
<summary>Monitoring rate limit usage</summary>

```python
# Get current usage for a key
usage = await limiter.get_usage("user:123")
print(f"Requests in window: {usage['count']}/{usage['limit']}")
print(f"Window resets in: {usage['reset_in_seconds']}s")
```
</details>

## Storage Endpoint (S3-Compatible)

### Overview

SimpleTuner cloud training jobs से आपके local machine पर training outputs (checkpoints, samples, logs) upload करने के लिए S3-compatible endpoint देता है। इससे cloud jobs results "call home" कर सकती हैं।

### Architecture

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

### Requirements

Cloud jobs को आपके local server पर upload करने के लिए:

1. **Public HTTPS endpoint** - Cloud providers `localhost` तक नहीं पहुंच सकते
2. **SSL certificate** - ज़्यादातर providers HTTPS मांगते हैं
3. **Firewall access** - चुने हुए port पर inbound connections allow करें

### Option 1: Cloudflared Tunnel (Recommended)

[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/) firewall ports खोले बिना सुरक्षित tunnel देता है।

<details>
<summary>Setup instructions</summary>

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

**Configuration (`~/.cloudflared/config.yml`):**

```yaml
tunnel: YOUR_TUNNEL_ID
credentials-file: ~/.cloudflared/YOUR_TUNNEL_ID.json

ingress:
  - hostname: simpletuner.yourdomain.com
    service: http://localhost:8001
  - service: http_status:404
```

**Run the tunnel:**

```bash
# Start the tunnel
cloudflared tunnel run simpletuner

# Or run as a service
sudo cloudflared service install
```

**Configure SimpleTuner:**

```bash
# Set the public URL for S3 uploads
export SIMPLETUNER_PUBLIC_URL="https://simpletuner.yourdomain.com"
```

या Cloud tab → Advanced Configuration → Public URL के जरिए।
</details>

### Option 2: ngrok

[ngrok](https://ngrok.com/) development के लिए quick tunnels देता है।

<details>
<summary>Setup instructions</summary>

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

**Start the tunnel:**

```bash
# Start ngrok tunnel to SimpleTuner port
ngrok http 8001

# Note the HTTPS URL from the output:
# Forwarding: https://abc123.ngrok.io -> http://localhost:8001
```

**Configure SimpleTuner:**

```bash
export SIMPLETUNER_PUBLIC_URL="https://abc123.ngrok.io"
```

**Note:** Free ngrok URLs हर restart पर बदल जाते हैं। Production के लिए paid plan (reserved domains) या Cloudflared इस्तेमाल करें।
</details>

### Option 3: Direct Public IP

<details>
<summary>Setup instructions</summary>

अगर आपके server के पास public IP है और आप firewall ports खोल सकते हैं:

```bash
# Allow inbound HTTPS
sudo ufw allow 8001/tcp

# Or with iptables
sudo iptables -A INPUT -p tcp --dport 8001 -j ACCEPT
```

**SSL Certificate Setup:**

Production के लिए Let's Encrypt उपयोग करें:

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

### Storage Endpoint Configuration

Provider settings के जरिए S3 endpoint behavior कॉन्फ़िगर करें:

<details>
<summary>API configuration</summary>

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

या Cloud tab → Advanced Configuration में।

### Upload Authentication

S3 uploads short-lived upload tokens से authenticated होती हैं:

1. job submit होने पर एक unique upload token generate होता है
2. token cloud job को environment variable के रूप में पास होता है
3. job upload करते समय token को S3 access key की तरह इस्तेमाल करता है
4. token job complete/cancel होने पर expire हो जाता है

### Supported S3 Operations

| Operation | Endpoint | Description |
|-----------|----------|-------------|
| PUT Object | `PUT /api/cloud/storage/{bucket}/{key}` | फ़ाइल upload करें |
| GET Object | `GET /api/cloud/storage/{bucket}/{key}` | फ़ाइल download करें |
| List Objects | `GET /api/cloud/storage/{bucket}` | bucket में objects सूचीबद्ध करें |
| List Buckets | `GET /api/cloud/storage` | सभी buckets सूचीबद्ध करें |

### Troubleshooting Storage Uploads

**"Unauthorized" के साथ uploads fail हो रहे हैं:**
- Upload token सही से पास हो रहा है या नहीं देखें
- Job ID token से match करता है या नहीं जांचें
- Job अभी active state में है या नहीं (completed/cancelled नहीं)

**Uploads timeout हो रहे हैं:**
- Tunnel चल रहा है या नहीं जांचें (`cloudflared tunnel run` या `ngrok http`)
- Public URL इंटरनेट से accessible है या नहीं सत्यापित करें
- टेस्ट करें: `curl -I https://your-public-url/api/cloud/health`

**SSL certificate errors:**
- ngrok/cloudflared के लिए SSL अपने आप संभाला जाता है
- direct connections के लिए certificate valid होना चाहिए
- intermediate certificates chain में शामिल हैं या नहीं देखें

<details>
<summary>Firewall and connectivity tests</summary>

```bash
# Test local connectivity
curl http://localhost:8001/api/cloud/health

# Test from external (if direct IP)
curl https://your-public-ip:8001/api/cloud/health
```
</details>

**Upload progress देखें:**

```bash
# Check current uploads
curl http://localhost:8001/api/cloud/jobs/{job_id}

# Response includes upload_progress
```

## Structured Logging

### Configuration

<details>
<summary>Environment variables</summary>

```bash
# Log level: DEBUG, INFO, WARNING, ERROR
export SIMPLETUNER_LOG_LEVEL="INFO"

# Format: "json" or "text"
export SIMPLETUNER_LOG_FORMAT="json"

# Optional file output
export SIMPLETUNER_LOG_FILE="/var/log/simpletuner/cloud.log"
```
</details>

### JSON Log Format

<details>
<summary>Example log entry</summary>

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

### Programmatic Configuration

<details>
<summary>Python configuration</summary>

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

## Backup and Restore

### Database Location

SQLite database यहां है:
```
~/.simpletuner/config/cloud/jobs.db
```

WAL files के साथ:
```
~/.simpletuner/config/cloud/jobs.db-wal
~/.simpletuner/config/cloud/jobs.db-shm
```

### Command-Line Backup

<details>
<summary>Backup commands</summary>

```bash
# Simple copy (stop server first for consistency)
cp ~/.simpletuner/config/cloud/jobs.db /backup/jobs_$(date +%Y%m%d).db

# Online backup with sqlite3
sqlite3 ~/.simpletuner/config/cloud/jobs.db ".backup /backup/jobs.db"
```
</details>

### Programmatic Backup

<details>
<summary>Python API</summary>

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
<summary>Restore from backup</summary>

```python
from pathlib import Path
from simpletuner.simpletuner_sdk.server.services.cloud import JobStore

store = JobStore()

# WARNING: This overwrites the current database!
success = store.restore(Path("/backup/jobs_backup_20240115_103000.db"))
```
</details>

### Automated Backup Script

<details>
<summary>Cron backup script</summary>

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

## Secrets Management

Secrets providers की विस्तृत डॉक्यूमेंटेशन के लिए [SECRETS_AND_CACHE.md](SECRETS_AND_CACHE.md) देखें।

### Supported Providers

1. **Environment Variables** (default)
2. **File-based Secrets** (`~/.simpletuner/secrets.json` या YAML)
3. **AWS Secrets Manager** (`pip install boto3` जरूरी)
4. **HashiCorp Vault** (`pip install hvac` जरूरी)

### Provider Priority

Secrets इस क्रम में resolve होते हैं:
1. Environment variables (सबसे उच्च priority - overrides के लिए)
2. File-based secrets
3. AWS Secrets Manager
4. HashiCorp Vault

## Troubleshooting

### Connection Issues

**Proxy काम नहीं कर रहा:**

<details>
<summary>Debug proxy connectivity</summary>

```bash
# Test proxy connectivity
curl -x http://proxy:8080 https://your-provider-api-endpoint

# Check environment
env | grep -i proxy
```
</details>

**SSL certificate errors:**

<details>
<summary>Debug SSL issues</summary>

```bash
# Test with custom CA
curl --cacert /path/to/ca.crt https://your-provider-api-endpoint

# Verify CA bundle
openssl verify -CAfile /path/to/ca.crt server.crt
```
</details>

**Provider-specific troubleshooting:**
- [Replicate Troubleshooting](REPLICATE.md#troubleshooting)

### Database Issues

**Locked database:**

<details>
<summary>Database lock resolution</summary>

```bash
# Check for open connections
fuser ~/.simpletuner/config/cloud/jobs.db

# Force WAL checkpoint
sqlite3 ~/.simpletuner/config/cloud/jobs.db "PRAGMA wal_checkpoint(TRUNCATE)"
```
</details>

**Corrupted database:**

<details>
<summary>Database recovery</summary>

```bash
# Check integrity
sqlite3 ~/.simpletuner/config/cloud/jobs.db "PRAGMA integrity_check"

# Recover (creates new database from good pages)
sqlite3 ~/.simpletuner/config/cloud/jobs.db ".recover" | sqlite3 jobs_recovered.db
```
</details>

### Health Check Failures

<details>
<summary>Health check debugging</summary>

```bash
# Test health endpoint
curl -s http://localhost:8080/api/cloud/health | jq .

# Check with provider checks included
curl -s 'http://localhost:8080/api/cloud/health?include_providers=true' | jq .
```
</details>
