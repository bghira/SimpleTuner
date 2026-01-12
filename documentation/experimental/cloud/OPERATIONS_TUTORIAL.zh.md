# 云训练运维指南

本文涵盖 SimpleTuner 云训练功能的生产部署与运维，重点在于与现有 DevOps 基础设施的完整集成。

## 网络架构

### 出站连接

服务器会向配置的云提供商发起 HTTPS 出站连接。每个提供商都有各自的 API 端点与要求。

**提供商网络细节：**
- [Replicate API Endpoints](REPLICATE.md#api-endpoints)

### 入站连接

| 来源 | Endpoint | 目的 |
|--------|----------|---------|
| 云提供商基础设施 | `/api/webhooks/{provider}` | 任务状态更新 |
| 云训练任务 | `/api/cloud/storage/{bucket}/{key}` | 上传训练输出 |
| 监控系统 | `/api/cloud/health`, `/api/cloud/metrics/prometheus` | 健康与指标 |

### 防火墙规则

防火墙要求取决于你的提供商配置。

**提供商防火墙规则：**
- [Replicate Firewall Configuration](REPLICATE.md#firewall)

### Webhook IP 白名单

为增强安全性，你可以限制 Webhook 只允许来自特定 IP 范围。配置后，白名单外的 IP 将返回 403 Forbidden。

**API 配置：**

<details>
<summary>API 配置示例</summary>

```bash
# Set allowed IPs for a provider's webhooks
curl -X PUT http://localhost:8080/api/cloud/providers/{provider} \
  -H "Content-Type: application/json" \
  -d '{
    "webhook_allowed_ips": ["10.0.0.0/8", "192.168.0.0/16"]
  }'
```
</details>

**Web UI 配置：**

1. 进入 Cloud 选项卡 → Advanced Configuration
2. 在 “Webhook Security” 部分添加 IP 范围
3. 使用 CIDR（如 `10.0.0.0/8`）或单个 IP（`1.2.3.4/32`）

**IP 格式：**

| 格式 | 示例 | 说明 |
|--------|---------|-------------|
| 单一 IP | `1.2.3.4/32` | 精确 IP 匹配 |
| 子网 | `10.0.0.0/8` | A 类网络 |
| 窄范围 | `192.168.1.0/24` | 256 个地址 |

**提供商 Webhook IP：**
- [Replicate Webhook IPs](REPLICATE.md#webhook-ips)

**行为：**

| 场景 | 结果 |
|----------|--------|
| 未配置白名单 | 接受所有 IP |
| 空数组 `[]` | 接受所有 IP |
| IP 在白名单内 | 处理 Webhook |
| IP 不在白名单内 | 403 Forbidden |

**审计日志：**

被拒绝的 Webhook 会记录到审计日志：

```bash
curl "http://localhost:8080/api/audit?event_type=webhook_rejected&limit=100"
```

## 代理配置

### 环境变量

<details>
<summary>代理环境变量</summary>

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

### 通过提供商配置

<details>
<summary>API 配置</summary>

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

### Web UI（Advanced Configuration）

Cloud 选项卡包含 Advanced Configuration 面板用于网络设置：

| 设置 | 说明 |
|---------|-------------|
| **SSL Verification** | 启用/禁用证书校验 |
| **CA Bundle Path** | 企业 CA 的证书包路径 |
| **Proxy URL** | 出站连接的 HTTP 代理 |
| **HTTP Timeout** | 请求超时（秒，默认 30） |

#### SSL Verification 绕过

禁用 SSL 校验存在安全风险，因此需要显式确认：

1. 点击 SSL Verification 开关以禁用
2. 弹出确认对话框：*“Disabling SSL verification is a security risk. Only do this if you have a self-signed certificate or are behind a corporate proxy. Continue?”*
3. 点击 “OK” 确认并保存

该确认仅在当前会话内有效，后续切换无需再次确认。

#### 企业代理配置

对于使用 HTTP 代理的环境：

1. 进入 Cloud 选项卡 → Advanced Configuration
2. 输入代理 URL（例如 `http://proxy.corp.example.com:8080`）
3. 若代理执行 TLS 检查，可设置自定义 CA 包
4. 若代理带来延迟，可调整 HTTP 超时

设置会立即保存并应用于所有后续提供商 API 调用。

## 健康监控

### Endpoints

| Endpoint | 目的 | 响应 |
|----------|---------|----------|
| `/api/cloud/health` | 全量健康检查 | 组件状态 JSON |
| `/api/cloud/health/live` | Kubernetes liveness | `{"status": "ok"}` |
| `/api/cloud/health/ready` | Kubernetes readiness | `{"status": "ready"}` 或 503 |

### 健康检查响应

<details>
<summary>示例响应</summary>

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

包含提供商 API 检查（增加延迟）：
```
GET /api/cloud/health?include_providers=true
```

## Prometheus 指标

采集端点：`/api/cloud/metrics/prometheus`

### 启用 Prometheus 导出

默认禁用。可在 Admin 面板 Metrics 选项卡或 API 启用：

<details>
<summary>通过 API 启用</summary>

```bash
curl -X PUT http://localhost:8080/api/cloud/metrics/config \
  -H "Content-Type: application/json" \
  -d '{"prometheus_enabled": true, "prometheus_categories": ["jobs", "http", "health"]}'
```
</details>

### 指标分类

指标按类别组织，可单独启用：

| 类别 | 描述 | 关键指标 |
|----------|-------------|-------------|
| `jobs` | 任务数量、状态、队列深度、成本 | `simpletuner_jobs_total`, `simpletuner_cost_usd_total` |
| `http` | 请求数、错误、延迟 | `simpletuner_http_requests_total`, `simpletuner_http_errors_total` |
| `rate_limits` | 触发限流 | `simpletuner_rate_limit_violations_total` |
| `approvals` | 审批流程指标 | `simpletuner_approval_requests_pending` |
| `audit` | 审计日志活动 | `simpletuner_audit_log_entries_total` |
| `health` | 运行时间、组件健康 | `simpletuner_uptime_seconds`, `simpletuner_health_database_latency_ms` |
| `circuit_breakers` | 提供商断路器状态 | `simpletuner_circuit_breaker_state` |
| `provider` | 成本限额、余额 | `simpletuner_cost_limit_percent_used` |

### 配置模板

常见用途的快速模板：

| 模板 | 类别 | 用途 |
|----------|------------|----------|
| `minimal` | jobs | 轻量任务监控 |
| `standard` | jobs, http, health | 推荐默认 |
| `security` | jobs, http, rate_limits, audit, approvals | 安全监控 |
| `full` | 全部类别 | 全量可观测性 |

<details>
<summary>应用模板</summary>

```bash
curl -X POST http://localhost:8080/api/cloud/metrics/config/templates/standard
```
</details>

### 可用指标

<details>
<summary>指标参考</summary>

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

### Prometheus 配置

<details>
<summary>prometheus.yml 采集配置</summary>

```yaml
scrape_configs:
  - job_name: 'simpletuner'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/api/cloud/metrics/prometheus'
    scrape_interval: 30s
```
</details>

### 预览指标输出

不影响配置的情况下预览输出：

```bash
curl "http://localhost:8080/api/cloud/metrics/config/preview?categories=jobs&categories=health"
```

## 速率限制

### 概述

SimpleTuner 内置速率限制以防滥用并确保公平资源使用。按 IP 应用，可为不同端点配置规则。

### 配置

可通过环境变量配置：

<details>
<summary>环境变量</summary>

```bash
# Default rate limit for unmatched endpoints
export RATE_LIMIT_CALLS=100      # Requests per period
export RATE_LIMIT_PERIOD=60      # Period in seconds

# Set to 0 to disable rate limiting entirely
export RATE_LIMIT_CALLS=0
```
</details>

### 默认限流规则

不同端点按敏感度配置不同限制：

| Endpoint Pattern | Limit | Period | Methods | Reason |
|------------------|-------|--------|---------|--------|
| `/api/auth/login` | 5 | 60s | POST | 防暴力破解 |
| `/api/auth/register` | 3 | 60s | POST | 注册滥用 |
| `/api/auth/api-keys` | 10 | 60s | POST | API Key 创建 |
| `/api/cloud/jobs` | 20 | 60s | POST | 任务提交 |
| `/api/cloud/jobs/.+/cancel` | 30 | 60s | POST | 任务取消 |
| `/api/webhooks/` | 100 | 60s | All | Webhook 投递 |
| `/api/cloud/storage/` | 50 | 60s | All | 存储上传 |
| `/api/quotas/` | 30 | 60s | All | 配额操作 |
| All other endpoints | 100 | 60s | All | 默认兜底 |

### 排除路径

以下路径不受限流：

- `/health` - 健康检查
- `/api/events/stream` - SSE 连接
- `/static/` - 静态文件
- `/api/cloud/hints` - UI 提示（非敏感）
- `/api/users/me` - 当前用户检查
- `/api/cloud/providers` - 提供商列表

### 响应头

所有响应包含限流头：

```
X-RateLimit-Limit: 100        # Maximum requests allowed
X-RateLimit-Remaining: 95     # Requests remaining in period
X-RateLimit-Reset: 1705320000 # Unix timestamp when limit resets
```

<details>
<summary>超过限流的响应</summary>

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 45
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1705320045

{"detail": "Rate limit exceeded. Please try again later."}
```
</details>

### 客户端 IP 判定

中间件会处理代理头：

1. `X-Forwarded-For` - 标准代理头（第一个 IP 为客户端）
2. `X-Real-IP` - Nginx 代理头
3. 直连 IP - 兜底

开发环境下 localhost（`127.0.0.1`, `::1`）不受限流。

### 审计日志

限流违规会记录到审计日志，包括：
- 客户端 IP
- 请求端点
- HTTP 方法
- User-Agent 头

查询限流事件审计日志：

```bash
curl "http://localhost:8080/api/audit?event_type=rate_limited&limit=100"
```

### 自定义限流规则

<details>
<summary>程序化配置</summary>

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

### 分布式限流（Async Rate Limiter）

多 worker 部署中，SimpleTuner 提供分布式限流器，使用配置的状态后端（SQLite、Redis、PostgreSQL、MySQL）共享限流状态。

<details>
<summary>获取限流器</summary>

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

**滑动窗口算法：**

滑动窗口比固定窗口更平滑：

```
Time:     |----60s window----|
Requests: [x x x x x][x x x]
          ↑ expired  ↑ counted
```

- 请求到达时记录时间戳
- 仅统计窗口内请求
- 过期请求自动清理
- 避免窗口边界突发问题

**不同后端行为：**

| 后端 | 实现 | 性能 | 多 Worker |
|---------|---------------|-------------|--------------|
| SQLite | JSON 时间戳数组 | Good | 单文件锁 |
| Redis | 有序集合（ZSET） | Excellent | 完全支持 |
| PostgreSQL | JSONB + 索引 | Very Good | 完全支持 |
| MySQL | JSON 列 | Good | 完全支持 |

<details>
<summary>预配置限流器</summary>

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
<summary>监控限流使用情况</summary>

```python
# Get current usage for a key
usage = await limiter.get_usage("user:123")
print(f"Requests in window: {usage['count']}/{usage['limit']}")
print(f"Window resets in: {usage['reset_in_seconds']}s")
```
</details>

## 存储端点（S3 兼容）

### 概述

SimpleTuner 提供 S3 兼容端点，用于将云训练输出（检查点、样本、日志）上传回本地机器，使云训练可以“回传结果”。

### 架构

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

### 要求

云端任务上传到本地服务器需要：

1. **公网 HTTPS 端点** - 云提供商无法访问 `localhost`
2. **SSL 证书** - 多数提供商要求 HTTPS
3. **防火墙放行** - 允许选定端口入站

### 方案 1：Cloudflared Tunnel（推荐）

[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/) 可在不开放防火墙端口的情况下提供安全隧道。

<details>
<summary>设置说明</summary>

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

**配置（`~/.cloudflared/config.yml`）：**

```yaml
tunnel: YOUR_TUNNEL_ID
credentials-file: ~/.cloudflared/YOUR_TUNNEL_ID.json

ingress:
  - hostname: simpletuner.yourdomain.com
    service: http://localhost:8001
  - service: http_status:404
```

**运行隧道：**

```bash
# Start the tunnel
cloudflared tunnel run simpletuner

# Or run as a service
sudo cloudflared service install
```

**配置 SimpleTuner：**

```bash
# Set the public URL for S3 uploads
export SIMPLETUNER_PUBLIC_URL="https://simpletuner.yourdomain.com"
```

或通过 Cloud 选项卡 → Advanced Configuration → Public URL 设置。
</details>

### 方案 2：ngrok

[ngrok](https://ngrok.com/) 提供快速开发隧道。

<details>
<summary>设置说明</summary>

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

**启动隧道：**

```bash
# Start ngrok tunnel to SimpleTuner port
ngrok http 8001

# Note the HTTPS URL from the output:
# Forwarding: https://abc123.ngrok.io -> http://localhost:8001
```

**配置 SimpleTuner：**

```bash
export SIMPLETUNER_PUBLIC_URL="https://abc123.ngrok.io"
```

**注记：** 免费 ngrok URL 每次重启都会变化。生产环境建议使用付费固定域名或 Cloudflared。
</details>

### 方案 3：直接公网 IP

<details>
<summary>设置说明</summary>

若服务器有公网 IP 且可开防火墙端口：

```bash
# Allow inbound HTTPS
sudo ufw allow 8001/tcp

# Or with iptables
sudo iptables -A INPUT -p tcp --dport 8001 -j ACCEPT
```

**SSL 证书设置：**

生产建议使用 Let's Encrypt：

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

### 存储端点配置

通过提供商设置配置 S3 端点行为：

<details>
<summary>API 配置</summary>

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

或在 Cloud 选项卡 → Advanced Configuration 中设置。

### 上传认证

S3 上传使用短期上传令牌认证：

1. 提交任务时生成唯一上传令牌
2. 令牌作为环境变量传入云任务
3. 任务使用令牌作为 S3 访问密钥上传
4. 令牌在任务完成或取消后失效

### 支持的 S3 操作

| 操作 | Endpoint | 说明 |
|-----------|----------|-------------|
| PUT Object | `PUT /api/cloud/storage/{bucket}/{key}` | 上传文件 |
| GET Object | `GET /api/cloud/storage/{bucket}/{key}` | 下载文件 |
| List Objects | `GET /api/cloud/storage/{bucket}` | 列出桶内对象 |
| List Buckets | `GET /api/cloud/storage` | 列出所有桶 |

### 存储上传排障

**上传失败并提示 “Unauthorized”：**
- 确认上传令牌正确传入
- 检查任务 ID 与令牌是否匹配
- 确认任务仍为活跃状态（未完成/取消）

**上传超时：**
- 检查隧道是否运行（`cloudflared tunnel run` 或 `ngrok http`）
- 确认公网 URL 可访问
- 测试：`curl -I https://your-public-url/api/cloud/health`

**SSL 证书错误：**
- ngrok/cloudflared 自动处理 SSL
- 直连模式需确保证书有效
- 检查中间证书是否包含在链中

<details>
<summary>防火墙与连通性测试</summary>

```bash
# Test local connectivity
curl http://localhost:8001/api/cloud/health

# Test from external (if direct IP)
curl https://your-public-ip:8001/api/cloud/health
```
</details>

**查看上传进度：**

```bash
# Check current uploads
curl http://localhost:8001/api/cloud/jobs/{job_id}

# Response includes upload_progress
```

## 结构化日志

### 配置

<details>
<summary>环境变量</summary>

```bash
# Log level: DEBUG, INFO, WARNING, ERROR
export SIMPLETUNER_LOG_LEVEL="INFO"

# Format: "json" or "text"
export SIMPLETUNER_LOG_FORMAT="json"

# Optional file output
export SIMPLETUNER_LOG_FILE="/var/log/simpletuner/cloud.log"
```
</details>

### JSON 日志格式

<details>
<summary>示例日志</summary>

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

### 程序化配置

<details>
<summary>Python 配置</summary>

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

## 备份与恢复

### 数据库位置

SQLite 数据库位于：
```
~/.simpletuner/config/cloud/jobs.db
```

WAL 文件：
```
~/.simpletuner/config/cloud/jobs.db-wal
~/.simpletuner/config/cloud/jobs.db-shm
```

### 命令行备份

<details>
<summary>备份命令</summary>

```bash
# Simple copy (stop server first for consistency)
cp ~/.simpletuner/config/cloud/jobs.db /backup/jobs_$(date +%Y%m%d).db

# Online backup with sqlite3
sqlite3 ~/.simpletuner/config/cloud/jobs.db ".backup /backup/jobs.db"
```
</details>

### 程序化备份

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

### 恢复

<details>
<summary>从备份恢复</summary>

```python
from pathlib import Path
from simpletuner.simpletuner_sdk.server.services.cloud import JobStore

store = JobStore()

# WARNING: This overwrites the current database!
success = store.restore(Path("/backup/jobs_backup_20240115_103000.db"))
```
</details>

### 自动备份脚本

<details>
<summary>Cron 备份脚本</summary>

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

## 密钥管理

密钥提供商详见 [SECRETS_AND_CACHE.md](SECRETS_AND_CACHE.md)。

### 支持的提供商

1. **环境变量**（默认）
2. **基于文件的密钥**（`~/.simpletuner/secrets.json` 或 YAML）
3. **AWS Secrets Manager**（需 `pip install boto3`）
4. **HashiCorp Vault**（需 `pip install hvac`）

### 提供商优先级

密钥解析顺序：
1. 环境变量（最高优先级，可覆盖）
2. 文件密钥
3. AWS Secrets Manager
4. HashiCorp Vault

## 排障

### 连接问题

**代理不生效：**

<details>
<summary>调试代理连通性</summary>

```bash
# Test proxy connectivity
curl -x http://proxy:8080 https://your-provider-api-endpoint

# Check environment
env | grep -i proxy
```
</details>

**SSL 证书错误：**

<details>
<summary>调试 SSL 问题</summary>

```bash
# Test with custom CA
curl --cacert /path/to/ca.crt https://your-provider-api-endpoint

# Verify CA bundle
openssl verify -CAfile /path/to/ca.crt server.crt
```
</details>

**提供商排障：**
- [Replicate Troubleshooting](REPLICATE.md#troubleshooting)

### 数据库问题

**数据库锁定：**

<details>
<summary>解除数据库锁</summary>

```bash
# Check for open connections
fuser ~/.simpletuner/config/cloud/jobs.db

# Force WAL checkpoint
sqlite3 ~/.simpletuner/config/cloud/jobs.db "PRAGMA wal_checkpoint(TRUNCATE)"
```
</details>

**数据库损坏：**

<details>
<summary>数据库恢复</summary>

```bash
# Check integrity
sqlite3 ~/.simpletuner/config/cloud/jobs.db "PRAGMA integrity_check"

# Recover (creates new database from good pages)
sqlite3 ~/.simpletuner/config/cloud/jobs.db ".recover" | sqlite3 jobs_recovered.db
```
</details>

### 健康检查失败

<details>
<summary>健康检查调试</summary>

```bash
# Test health endpoint
curl -s http://localhost:8080/api/cloud/health | jq .

# Check with provider checks included
curl -s 'http://localhost:8080/api/cloud/health?include_providers=true' | jq .
```
</details>
