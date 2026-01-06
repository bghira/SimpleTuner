# クラウド学習オペレーションガイド

このドキュメントは、SimpleTuner のクラウド学習機能の本番デプロイと運用を扱い、既存の DevOps インフラとの完全な統合に焦点を当てています。

## ネットワークアーキテクチャ

### アウトバウンド接続

サーバは設定されたクラウドプロバイダへ HTTPS でアウトバウンド接続します。各プロバイダは独自の API エンドポイントと要件があります。

**プロバイダ別ネットワーク詳細:**
- [Replicate API Endpoints](REPLICATE.md#api-endpoints)

### インバウンド接続

| ソース | エンドポイント | 目的 |
|--------|----------|---------|
| クラウドプロバイダ基盤 | `/api/webhooks/{provider}` | ジョブ状態更新 |
| クラウド学習ジョブ | `/api/cloud/storage/{bucket}/{key}` | 学習成果のアップロード |
| 監視システム | `/api/cloud/health`, `/api/cloud/metrics/prometheus` | ヘルス/メトリクス |

### ファイアウォールルール

ファイアウォール要件はプロバイダ設定に依存します。

**プロバイダ別ファイアウォール:**
- [Replicate Firewall Configuration](REPLICATE.md#firewall)

### Webhook の IP 許可リスト

セキュリティ強化のため、Webhook を特定の IP 範囲に制限できます。許可リスト外の IP からの Webhook は 403 Forbidden で拒否されます。

**API による設定:**

<details>
<summary>API 設定例</summary>

```bash
# Set allowed IPs for a provider's webhooks
curl -X PUT http://localhost:8080/api/cloud/providers/{provider} \
  -H "Content-Type: application/json" \
  -d '{
    "webhook_allowed_ips": ["10.0.0.0/8", "192.168.0.0/16"]
  }'
```
</details>

**Web UI による設定:**

1. Cloud タブ → Advanced Configuration に移動
2. "Webhook Security" セクションで IP 範囲を追加
3. CIDR（例: `10.0.0.0/8`）または単一 IP（`1.2.3.4/32`）を使用

**IP 形式:**

| 形式 | 例 | 説明 |
|--------|---------|-------------|
| 単一 IP | `1.2.3.4/32` | 完全一致 |
| サブネット | `10.0.0.0/8` | Class A ネットワーク |
| 狭い範囲 | `192.168.1.0/24` | 256 アドレス |

**プロバイダ別 Webhook IP:**
- [Replicate Webhook IPs](REPLICATE.md#webhook-ips)

**挙動:**

| シナリオ | 結果 |
|----------|--------|
| 許可リスト未設定 | すべて許可 |
| 空配列 `[]` | すべて許可 |
| 許可リスト内 IP | Webhook 処理 |
| 許可リスト外 IP | 403 Forbidden |

**監査ログ:**

拒否された Webhook は監査ログに記録されます:

```bash
curl "http://localhost:8080/api/audit?event_type=webhook_rejected&limit=100"
```

## プロキシ設定

### 環境変数

<details>
<summary>プロキシ環境変数</summary>

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

### プロバイダ設定から

<details>
<summary>API 設定</summary>

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

Cloud タブの Advanced Configuration パネルでネットワーク設定を変更できます:

| 設定 | 説明 |
|---------|-------------|
| **SSL Verification** | 証明書検証の有効/無効を切替 |
| **CA Bundle Path** | 企業 CA 用の証明書バンドル |
| **Proxy URL** | アウトバウンド接続用 HTTP プロキシ |
| **HTTP Timeout** | リクエストタイムアウト（秒、既定 30） |

#### SSL 検証の無効化

セキュリティ上の影響があるため、無効化には明示的な確認が必要です:

1. SSL Verification トグルを無効化
2. 確認ダイアログ: *"Disabling SSL verification is a security risk. Only do this if you have a self-signed certificate or are behind a corporate proxy. Continue?"*
3. "OK" を押して保存

確認はセッション内でのみ有効です。同一セッション内での再切替には再確認は不要です。

#### 企業プロキシ設定

HTTP プロキシ環境向け:

1. Cloud タブ → Advanced Configuration へ移動
2. プロキシ URL を入力（例: `http://proxy.corp.example.com:8080`）
3. プロキシが TLS 検査を行う場合はカスタム CA バンドルを設定
4. 追加レイテンシがある場合は HTTP タイムアウトを調整

設定は変更と同時に保存され、以降のプロバイダ API 呼び出しに適用されます。

## ヘルス監視

### エンドポイント

| エンドポイント | 目的 | レスポンス |
|----------|---------|----------|
| `/api/cloud/health` | 全体ヘルスチェック | コンポーネント状態を含む JSON |
| `/api/cloud/health/live` | Kubernetes liveness | `{"status": "ok"}` |
| `/api/cloud/health/ready` | Kubernetes readiness | `{"status": "ready"}` または 503 |

### ヘルスチェックレスポンス

<details>
<summary>レスポンス例</summary>

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

プロバイダ API チェックも含める（レイテンシ増加）:
```
GET /api/cloud/health?include_providers=true
```

## Prometheus メトリクス

スクレイプエンドポイント: `/api/cloud/metrics/prometheus`

### Prometheus Export の有効化

既定では無効です。Admin パネルの Metrics タブまたは API で有効化します:

<details>
<summary>API で有効化</summary>

```bash
curl -X PUT http://localhost:8080/api/cloud/metrics/config \
  -H "Content-Type: application/json" \
  -d '{"prometheus_enabled": true, "prometheus_categories": ["jobs", "http", "health"]}'
```
</details>

### メトリクスカテゴリ

カテゴリごとに個別に有効化できます:

| カテゴリ | 説明 | 代表メトリクス |
|----------|-------------|-------------|
| `jobs` | ジョブ数、ステータス、キュー深度、コスト | `simpletuner_jobs_total`, `simpletuner_cost_usd_total` |
| `http` | リクエスト数、エラー、レイテンシ | `simpletuner_http_requests_total`, `simpletuner_http_errors_total` |
| `rate_limits` | レート制限違反 | `simpletuner_rate_limit_violations_total` |
| `approvals` | 承認ワークフロー | `simpletuner_approval_requests_pending` |
| `audit` | 監査ログ活動 | `simpletuner_audit_log_entries_total` |
| `health` | 稼働時間、コンポーネント健全性 | `simpletuner_uptime_seconds`, `simpletuner_health_database_latency_ms` |
| `circuit_breakers` | プロバイダのサーキットブレーカ | `simpletuner_circuit_breaker_state` |
| `provider` | コスト上限、クレジット残高 | `simpletuner_cost_limit_percent_used` |

### 設定テンプレート

一般的な用途向けのテンプレート:

| テンプレート | カテゴリ | 用途 |
|----------|------------|----------|
| `minimal` | jobs | 軽量なジョブ監視 |
| `standard` | jobs, http, health | 推奨デフォルト |
| `security` | jobs, http, rate_limits, audit, approvals | セキュリティ監視 |
| `full` | 全カテゴリ | 完全な可観測性 |

<details>
<summary>テンプレート適用</summary>

```bash
curl -X POST http://localhost:8080/api/cloud/metrics/config/templates/standard
```
</details>

### 利用可能メトリクス

<details>
<summary>メトリクス一覧</summary>

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

### Prometheus 設定

<details>
<summary>prometheus.yml の scrape 設定</summary>

```yaml
scrape_configs:
  - job_name: 'simpletuner'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/api/cloud/metrics/prometheus'
    scrape_interval: 30s
```
</details>

### メトリクス出力のプレビュー

設定に影響せずに出力を確認できます:

```bash
curl "http://localhost:8080/api/cloud/metrics/config/preview?categories=jobs&categories=health"
```

## レート制限

### 概要

SimpleTuner には不正利用防止と公平な資源利用のためのレート制限が組み込まれています。レート制限は IP 単位で適用され、エンドポイントごとにルールを設定できます。

### 設定

環境変数で設定可能です:

<details>
<summary>環境変数</summary>

```bash
# Default rate limit for unmatched endpoints
export RATE_LIMIT_CALLS=100      # Requests per period
export RATE_LIMIT_PERIOD=60      # Period in seconds

# Set to 0 to disable rate limiting entirely
export RATE_LIMIT_CALLS=0
```
</details>

### 既定のレート制限ルール

エンドポイントごとに感度に応じた制限が設定されています:

| エンドポイントパターン | 制限 | 期間 | メソッド | 理由 |
|------------------|-------|--------|---------|--------|
| `/api/auth/login` | 5 | 60s | POST | 総当たり対策 |
| `/api/auth/register` | 3 | 60s | POST | 登録の乱用対策 |
| `/api/auth/api-keys` | 10 | 60s | POST | API キー作成 |
| `/api/cloud/jobs` | 20 | 60s | POST | ジョブ送信 |
| `/api/cloud/jobs/.+/cancel` | 30 | 60s | POST | ジョブキャンセル |
| `/api/webhooks/` | 100 | 60s | All | Webhook 受信 |
| `/api/cloud/storage/` | 50 | 60s | All | ストレージアップロード |
| `/api/quotas/` | 30 | 60s | All | クォータ操作 |
| All other endpoints | 100 | 60s | All | 既定フォールバック |

### 除外パス

以下のパスはレート制限対象外です:

- `/health` - ヘルスチェック
- `/api/events/stream` - SSE 接続
- `/static/` - 静的ファイル
- `/api/cloud/hints` - UI ヒント（安全性低）
- `/api/users/me` - 現在のユーザー確認
- `/api/cloud/providers` - プロバイダ一覧

### レスポンスヘッダ

すべてのレスポンスにレート制限ヘッダが含まれます:

```
X-RateLimit-Limit: 100        # Maximum requests allowed
X-RateLimit-Remaining: 95     # Requests remaining in period
X-RateLimit-Reset: 1705320000 # Unix timestamp when limit resets
```

<details>
<summary>制限超過時のレスポンス</summary>

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 45
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1705320045

{"detail": "Rate limit exceeded. Please try again later."}
```
</details>

### クライアント IP 判定

ミドルウェアはプロキシヘッダを適切に扱います:

1. `X-Forwarded-For` - 標準プロキシヘッダ（先頭がクライアント IP）
2. `X-Real-IP` - Nginx プロキシヘッダ
3. 直結 IP - フォールバック

開発環境では localhost（`127.0.0.1`, `::1`）は制限対象外です。

### 監査ログ

レート制限違反は監査ログに記録されます:
- クライアント IP
- 要求エンドポイント
- HTTP メソッド
- User-Agent ヘッダ

レート制限イベントの監査ログを取得:

```bash
curl "http://localhost:8080/api/audit?event_type=rate_limited&limit=100"
```

### カスタムレート制限ルール

<details>
<summary>プログラムによる設定</summary>

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

### 分散レート制限（Async Rate Limiter）

マルチワーカー環境では、設定された状態バックエンド（SQLite、Redis、PostgreSQL、MySQL）を使ってレート制限状態を共有する分散レートリミッタを提供します。

<details>
<summary>レートリミッタ取得</summary>

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

**スライディングウィンドウ・アルゴリズム:**

固定ウィンドウより滑らかな制御が可能です:

```
Time:     |----60s window----|
Requests: [x x x x x][x x x]
          ↑ expired  ↑ counted
```

- リクエスト到着時にタイムスタンプを記録
- ウィンドウ内のリクエストのみカウント
- 古いリクエストは自動的に期限切れ・削除
- ウィンドウ境界でのバースト問題を回避

**バックエンド別挙動:**

| バックエンド | 実装 | パフォーマンス | マルチワーカー |
|---------|---------------|-------------|--------------|
| SQLite | JSON タイムスタンプ配列 | Good | 単一ファイルロック |
| Redis | Sorted set (ZSET) | Excellent | フルサポート |
| PostgreSQL | JSONB + index | Very Good | フルサポート |
| MySQL | JSON column | Good | フルサポート |

<details>
<summary>事前構成済みレートリミッタ</summary>

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
<summary>レート制限の利用状況監視</summary>

```python
# Get current usage for a key
usage = await limiter.get_usage("user:123")
print(f"Requests in window: {usage['count']}/{usage['limit']}")
print(f"Window resets in: {usage['reset_in_seconds']}s")
```
</details>

## ストレージエンドポイント（S3 互換）

### 概要

SimpleTuner は S3 互換エンドポイントを提供し、クラウド学習ジョブが学習成果（チェックポイント、サンプル、ログ）をローカルにアップロードできます。クラウドジョブが "call home" できるようになります。

### アーキテクチャ

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

### 要件

クラウドジョブからローカルサーバへアップロードするには:

1. **公開 HTTPS エンドポイント** - クラウドプロバイダは `localhost` に到達できません
2. **SSL 証明書** - 多くのプロバイダが HTTPS を要求
3. **ファイアウォール許可** - 選択したポートへのインバウンドを許可

### オプション 1: Cloudflared Tunnel（推奨）

[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/) はファイアウォールポートを開けずに安全なトンネルを提供します。

<details>
<summary>セットアップ手順</summary>

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

**設定（`~/.cloudflared/config.yml`）:**

```yaml
tunnel: YOUR_TUNNEL_ID
credentials-file: ~/.cloudflared/YOUR_TUNNEL_ID.json

ingress:
  - hostname: simpletuner.yourdomain.com
    service: http://localhost:8001
  - service: http_status:404
```

**トンネル起動:**

```bash
# Start the tunnel
cloudflared tunnel run simpletuner

# Or run as a service
sudo cloudflared service install
```

**SimpleTuner の設定:**

```bash
# Set the public URL for S3 uploads
export SIMPLETUNER_PUBLIC_URL="https://simpletuner.yourdomain.com"
```

または Cloud タブ → Advanced Configuration → Public URL で設定します。
</details>

### オプション 2: ngrok

[ngrok](https://ngrok.com/) は開発向けの簡易トンネルを提供します。

<details>
<summary>セットアップ手順</summary>

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

**トンネル起動:**

```bash
# Start ngrok tunnel to SimpleTuner port
ngrok http 8001

# Note the HTTPS URL from the output:
# Forwarding: https://abc123.ngrok.io -> http://localhost:8001
```

**SimpleTuner の設定:**

```bash
export SIMPLETUNER_PUBLIC_URL="https://abc123.ngrok.io"
```

**注記:** 無料 ngrok URL は再起動ごとに変わります。本番では予約ドメインのある有料プランまたは Cloudflared を推奨します。
</details>

### オプション 3: 直接パブリック IP

<details>
<summary>セットアップ手順</summary>

サーバがパブリック IP を持ち、ファイアウォールポートを開けられる場合:

```bash
# Allow inbound HTTPS
sudo ufw allow 8001/tcp

# Or with iptables
sudo iptables -A INPUT -p tcp --dport 8001 -j ACCEPT
```

**SSL 証明書の設定:**

本番では Let's Encrypt を使用:

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

### ストレージエンドポイント設定

プロバイダ設定で S3 エンドポイントの挙動を構成します:

<details>
<summary>API 設定</summary>

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

または Cloud タブ → Advanced Configuration から設定します。

### アップロード認証

S3 アップロードは短命のアップロードトークンで認証されます:

1. ジョブ送信時に一意のアップロードトークンを生成
2. トークンは環境変数としてクラウドジョブへ渡される
3. ジョブはトークンを S3 アクセスキーとして使用
4. トークンはジョブ完了またはキャンセルで失効

### 対応 S3 操作

| 操作 | エンドポイント | 説明 |
|-----------|----------|-------------|
| PUT Object | `PUT /api/cloud/storage/{bucket}/{key}` | ファイルのアップロード |
| GET Object | `GET /api/cloud/storage/{bucket}/{key}` | ファイルのダウンロード |
| List Objects | `GET /api/cloud/storage/{bucket}` | バケット内の一覧 |
| List Buckets | `GET /api/cloud/storage` | バケット一覧 |

### ストレージアップロードのトラブルシューティング

**"Unauthorized" で失敗:**
- アップロードトークンが正しく渡されているか確認
- ジョブ ID とトークンが一致しているか確認
- ジョブがアクティブ状態か確認（完了/キャンセルでない）

**アップロードがタイムアウト:**
- トンネルが起動しているか確認（`cloudflared tunnel run` / `ngrok http`）
- 公開 URL がインターネットから到達できるか確認
- `curl -I https://your-public-url/api/cloud/health` でテスト

**SSL 証明書エラー:**
- ngrok/cloudflared は SSL を自動処理
- 直接接続の場合は証明書の有効性を確認
- 中間証明書がチェーンに含まれているか確認

<details>
<summary>ファイアウォールと疎通テスト</summary>

```bash
# Test local connectivity
curl http://localhost:8001/api/cloud/health

# Test from external (if direct IP)
curl https://your-public-ip:8001/api/cloud/health
```
</details>

**アップロード進捗の確認:**

```bash
# Check current uploads
curl http://localhost:8001/api/cloud/jobs/{job_id}

# Response includes upload_progress
```

## 構造化ログ

### 設定

<details>
<summary>環境変数</summary>

```bash
# Log level: DEBUG, INFO, WARNING, ERROR
export SIMPLETUNER_LOG_LEVEL="INFO"

# Format: "json" or "text"
export SIMPLETUNER_LOG_FORMAT="json"

# Optional file output
export SIMPLETUNER_LOG_FILE="/var/log/simpletuner/cloud.log"
```
</details>

### JSON ログ形式

<details>
<summary>ログ例</summary>

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

### プログラムによる設定

<details>
<summary>Python 設定</summary>

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

## バックアップと復元

### データベースの場所

SQLite データベースは以下に保存されます:
```
~/.simpletuner/config/cloud/jobs.db
```

WAL ファイル:
```
~/.simpletuner/config/cloud/jobs.db-wal
~/.simpletuner/config/cloud/jobs.db-shm
```

### コマンドラインでのバックアップ

<details>
<summary>バックアップコマンド</summary>

```bash
# Simple copy (stop server first for consistency)
cp ~/.simpletuner/config/cloud/jobs.db /backup/jobs_$(date +%Y%m%d).db

# Online backup with sqlite3
sqlite3 ~/.simpletuner/config/cloud/jobs.db ".backup /backup/jobs.db"
```
</details>

### プログラムによるバックアップ

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

### 復元

<details>
<summary>バックアップから復元</summary>

```python
from pathlib import Path
from simpletuner.simpletuner_sdk.server.services.cloud import JobStore

store = JobStore()

# WARNING: This overwrites the current database!
success = store.restore(Path("/backup/jobs_backup_20240115_103000.db"))
```
</details>

### 自動バックアップスクリプト

<details>
<summary>Cron バックアップスクリプト</summary>

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

## シークレット管理

シークレットプロバイダの詳細は [SECRETS_AND_CACHE.md](SECRETS_AND_CACHE.md) を参照してください。

### 対応プロバイダ

1. **環境変数**（既定）
2. **ファイルベースのシークレット**（`~/.simpletuner/secrets.json` または YAML）
3. **AWS Secrets Manager**（`pip install boto3` が必要）
4. **HashiCorp Vault**（`pip install hvac` が必要）

### プロバイダ優先度

シークレットは以下の順で解決されます:
1. 環境変数（最優先、上書き可能）
2. ファイルベースのシークレット
3. AWS Secrets Manager
4. HashiCorp Vault

## トラブルシューティング

### 接続問題

**プロキシが動かない:**

<details>
<summary>プロキシ疎通のデバッグ</summary>

```bash
# Test proxy connectivity
curl -x http://proxy:8080 https://your-provider-api-endpoint

# Check environment
env | grep -i proxy
```
</details>

**SSL 証明書エラー:**

<details>
<summary>SSL 問題のデバッグ</summary>

```bash
# Test with custom CA
curl --cacert /path/to/ca.crt https://your-provider-api-endpoint

# Verify CA bundle
openssl verify -CAfile /path/to/ca.crt server.crt
```
</details>

**プロバイダ固有のトラブルシューティング:**
- [Replicate Troubleshooting](REPLICATE.md#troubleshooting)

### データベース問題

**データベースロック:**

<details>
<summary>ロック解除</summary>

```bash
# Check for open connections
fuser ~/.simpletuner/config/cloud/jobs.db

# Force WAL checkpoint
sqlite3 ~/.simpletuner/config/cloud/jobs.db "PRAGMA wal_checkpoint(TRUNCATE)"
```
</details>

**データベース破損:**

<details>
<summary>データベース復旧</summary>

```bash
# Check integrity
sqlite3 ~/.simpletuner/config/cloud/jobs.db "PRAGMA integrity_check"

# Recover (creates new database from good pages)
sqlite3 ~/.simpletuner/config/cloud/jobs.db ".recover" | sqlite3 jobs_recovered.db
```
</details>

### ヘルスチェック失敗

<details>
<summary>ヘルスチェックのデバッグ</summary>

```bash
# Test health endpoint
curl -s http://localhost:8080/api/cloud/health | jq .

# Check with provider checks included
curl -s 'http://localhost:8080/api/cloud/health?include_providers=true' | jq .
```
</details>
