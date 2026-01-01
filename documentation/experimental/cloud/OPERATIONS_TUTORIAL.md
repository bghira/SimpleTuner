# Cloud Training Operations Guide

This document covers production deployment and operations for SimpleTuner's cloud training feature, with a focus on complete integration with existing DevOps infrastructure.

## Network Architecture

### Outbound Connections

The server makes outbound HTTPS connections to configured cloud providers. Each provider has its own API endpoints and requirements.

**Provider-specific network details:**
- [Replicate API Endpoints](REPLICATE.md#api-endpoints)

### Inbound Connections

| Source | Endpoint | Purpose |
|--------|----------|---------|
| Cloud provider infrastructure | `/api/webhooks/{provider}` | Job status updates |
| Cloud training job | `/api/cloud/storage/{bucket}/{key}` | Upload training outputs |
| Monitoring systems | `/api/cloud/health`, `/api/cloud/metrics/prometheus` | Health and metrics |

### Firewall Rules

Firewall requirements depend on your configured provider(s).

**Provider-specific firewall rules:**
- [Replicate Firewall Configuration](REPLICATE.md#firewall)

### Webhook IP Allowlisting

For enhanced security, you can restrict webhook delivery to specific IP ranges. When configured, webhooks from IPs outside the allowlist are rejected with a 403 Forbidden response.

**Configuration via API:**

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

**Configuration via Web UI:**

1. Navigate to Cloud tab → Advanced Configuration
2. In the "Webhook Security" section, add IP ranges
3. Use CIDR notation (e.g., `10.0.0.0/8`) or single IPs (`1.2.3.4/32`)

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
| No allowlist configured | All IPs accepted |
| Empty array `[]` | All IPs accepted |
| IP in allowlist | Webhook processed |
| IP not in allowlist | 403 Forbidden |

**Audit Logging:**

Rejected webhooks are logged to the audit trail:

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

### Via Provider Config

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

### Via Web UI (Advanced Configuration)

The Cloud tab includes an Advanced Configuration panel for network settings:

| Setting | Description |
|---------|-------------|
| **SSL Verification** | Toggle to enable/disable certificate verification |
| **CA Bundle Path** | Custom certificate authority bundle for corporate CAs |
| **Proxy URL** | HTTP proxy for outbound connections |
| **HTTP Timeout** | Request timeout in seconds (default: 30) |

#### SSL Verification Bypass

Disabling SSL verification requires explicit acknowledgment due to security implications:

1. Click the SSL Verification toggle to disable
2. A confirmation dialog appears: *"Disabling SSL verification is a security risk. Only do this if you have a self-signed certificate or are behind a corporate proxy. Continue?"*
3. Click "OK" to confirm and save the setting

The acknowledgment is session-scoped. Subsequent toggles within the same session won't require re-confirmation.

#### Corporate Proxy Configuration

For environments using HTTP proxies:

1. Navigate to the Cloud tab → Advanced Configuration
2. Enter the proxy URL (e.g., `http://proxy.corp.example.com:8080`)
3. Optionally set a custom CA bundle if your proxy performs TLS inspection
4. Adjust the HTTP timeout if your proxy adds latency

Settings are saved immediately when changed and apply to all subsequent provider API calls.

## Health Monitoring

### Endpoints

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `/api/cloud/health` | Full health check | JSON with component status |
| `/api/cloud/health/live` | Kubernetes liveness | `{"status": "ok"}` |
| `/api/cloud/health/ready` | Kubernetes readiness | `{"status": "ready"}` or 503 |

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

Include provider API checks (adds latency):
```
GET /api/cloud/health?include_providers=true
```

## Prometheus Metrics

Scrape endpoint: `/api/cloud/metrics/prometheus`

### Enabling Prometheus Export

Prometheus export is disabled by default. Enable it via the Metrics tab in the Admin panel or via API:

<details>
<summary>Enable via API</summary>

```bash
curl -X PUT http://localhost:8080/api/cloud/metrics/config \
  -H "Content-Type: application/json" \
  -d '{"prometheus_enabled": true, "prometheus_categories": ["jobs", "http", "health"]}'
```
</details>

### Metric Categories

Metrics are organized into categories that can be individually enabled:

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

Quick-start templates for common use cases:

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

### Preview Metrics Output

Preview what will be exported without affecting configuration:

```bash
curl "http://localhost:8080/api/cloud/metrics/config/preview?categories=jobs&categories=health"
```

## Rate Limiting

### Overview

SimpleTuner includes built-in rate limiting to protect against abuse and ensure fair resource usage. Rate limits are applied per-IP with configurable rules for different endpoints.

### Configuration

Rate limiting can be configured via environment variables:

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

Different endpoints have different rate limits based on sensitivity:

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

The following paths are excluded from rate limiting:

- `/health` - Health checks
- `/api/events/stream` - SSE connections
- `/static/` - Static files
- `/api/cloud/hints` - UI hints (not security-sensitive)
- `/api/users/me` - Current user check
- `/api/cloud/providers` - Provider list

### Response Headers

All responses include rate limit headers:

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

The middleware properly handles proxy headers:

1. `X-Forwarded-For` - Standard proxy header (first IP is the client)
2. `X-Real-IP` - Nginx proxy header
3. Direct connection IP - Fallback

Rate limits are bypassed for localhost (`127.0.0.1`, `::1`) in development.

### Audit Logging

Rate limit violations are logged to the audit trail with:
- Client IP address
- Requested endpoint
- HTTP method
- User-Agent header

Query audit logs for rate limit events:

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

For multi-worker deployments, SimpleTuner provides a distributed rate limiter that uses the configured state backend (SQLite, Redis, PostgreSQL, or MySQL) to share rate limit state across workers.

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

The rate limiter uses a sliding window algorithm that provides smoother rate limiting than fixed windows:

```
Time:     |----60s window----|
Requests: [x x x x x][x x x]
          ↑ expired  ↑ counted
```

- Requests are timestamped when they arrive
- Only requests within the window are counted
- Older requests expire and are pruned automatically
- No "burst at window boundary" problem

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

SimpleTuner provides an S3-compatible endpoint for uploading training outputs (checkpoints, samples, logs) from cloud training jobs back to your local machine. This enables cloud training jobs to "call home" with results.

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

For cloud jobs to upload to your local server, you need:

1. **Public HTTPS endpoint** - Cloud providers can't reach `localhost`
2. **SSL certificate** - Most providers require HTTPS
3. **Firewall access** - Allow inbound connections on your chosen port

### Option 1: Cloudflared Tunnel (Recommended)

[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/) provides a secure tunnel without opening firewall ports.

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

Or via the Cloud tab → Advanced Configuration → Public URL.
</details>

### Option 2: ngrok

[ngrok](https://ngrok.com/) provides quick tunnels for development.

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

**Note:** Free ngrok URLs change on each restart. For production, use a paid plan with reserved domains or Cloudflared.
</details>

### Option 3: Direct Public IP

<details>
<summary>Setup instructions</summary>

If your server has a public IP and you can open firewall ports:

```bash
# Allow inbound HTTPS
sudo ufw allow 8001/tcp

# Or with iptables
sudo iptables -A INPUT -p tcp --dport 8001 -j ACCEPT
```

**SSL Certificate Setup:**

For production, use Let's Encrypt:

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

Configure the S3 endpoint behavior via provider settings:

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

Or via the Cloud tab → Advanced Configuration.

### Upload Authentication

S3 uploads are authenticated using short-lived upload tokens:

1. When a job is submitted, a unique upload token is generated
2. The token is passed to the cloud job as an environment variable
3. The job uses the token as the S3 access key when uploading
4. Tokens expire after the job completes or is cancelled

### Supported S3 Operations

| Operation | Endpoint | Description |
|-----------|----------|-------------|
| PUT Object | `PUT /api/cloud/storage/{bucket}/{key}` | Upload a file |
| GET Object | `GET /api/cloud/storage/{bucket}/{key}` | Download a file |
| List Objects | `GET /api/cloud/storage/{bucket}` | List objects in bucket |
| List Buckets | `GET /api/cloud/storage` | List all buckets |

### Troubleshooting Storage Uploads

**Uploads failing with "Unauthorized":**
- Verify the upload token is being passed correctly
- Check that the job ID matches the token
- Ensure the job is still in an active state (not completed/cancelled)

**Uploads timing out:**
- Check your tunnel is running (`cloudflared tunnel run` or `ngrok http`)
- Verify the public URL is accessible from the internet
- Test with: `curl -I https://your-public-url/api/cloud/health`

**SSL certificate errors:**
- For ngrok/cloudflared, SSL is handled automatically
- For direct connections, ensure your certificate is valid
- Check intermediate certificates are included in the chain

<details>
<summary>Firewall and connectivity tests</summary>

```bash
# Test local connectivity
curl http://localhost:8001/api/cloud/health

# Test from external (if direct IP)
curl https://your-public-ip:8001/api/cloud/health
```
</details>

**View upload progress:**

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

The SQLite database is stored at:
```
~/.simpletuner/config/cloud/jobs.db
```

With WAL files:
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

See [SECRETS_AND_CACHE.md](SECRETS_AND_CACHE.md) for detailed documentation on secrets providers.

### Supported Providers

1. **Environment Variables** (default)
2. **File-based Secrets** (`~/.simpletuner/secrets.json` or YAML)
3. **AWS Secrets Manager** (requires `pip install boto3`)
4. **HashiCorp Vault** (requires `pip install hvac`)

### Provider Priority

Secrets are resolved in order:
1. Environment variables (highest priority - allows overrides)
2. File-based secrets
3. AWS Secrets Manager
4. HashiCorp Vault

## Troubleshooting

### Connection Issues

**Proxy not working:**

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
