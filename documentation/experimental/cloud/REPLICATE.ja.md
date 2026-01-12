# Replicate 連携

Replicate は ML モデルを実行するクラウドプラットフォームです。SimpleTuner は Replicate の Cog コンテナシステムを使ってクラウド GPU 上で学習ジョブを実行します。

- **モデル:** `simpletuner/advanced-trainer`
- **既定 GPU:** L40S（48GB VRAM）

## クイックスタート

1. [Replicate アカウント](https://replicate.com/signin)を作成し、[API トークン](https://replicate.com/account/api-tokens) を取得
2. 環境変数を設定:
   ```bash
   export REPLICATE_API_TOKEN="r8_your_token_here"
   simpletuner server
   ```
3. Web UI → Cloud タブ → **Validate** をクリックして確認

## データフロー

| データ種別 | 宛先 | 保持期間 |
|-----------|-------------|-----------|
| 学習画像 | Replicate アップロードサーバ（GCP） | ジョブ完了後削除 |
| 学習設定 | Replicate API | ジョブメタデータとして保存 |
| API トークン | ローカル環境のみ | SimpleTuner は保存しない |
| 学習済みモデル | HuggingFace Hub、S3、ローカル | ユーザー管理 |
| ジョブログ | Replicate サーバ | 30 日 |

**アップロード制限:** Replicate のファイルアップロード API は 100 MiB までのアーカイブに対応します。SimpleTuner はパッケージ済みアーカイブがこの上限を超える場合、送信をブロックします。

<details>
<summary>データ経路の詳細</summary>

1. **アップロード:** ローカル画像 → HTTPS POST → `api.replicate.com`
2. **学習:** Replicate が一時 GPU インスタンスへデータをダウンロード
3. **出力:** 学習済みモデル → 設定した宛先
4. **クリーンアップ:** ジョブ完了後に学習データを削除

詳細は [Replicate Security Docs](https://replicate.com/docs/reference/security) を参照してください。

</details>

## ハードウェアとコスト {#costs}

| ハードウェア | VRAM | コスト | 最適用途 |
|----------|------|------|----------|
| L40S | 48GB | ~$3.50/hr | ほとんどの LoRA 学習 |
| A100 (80GB) | 80GB | ~$5.00/hr | 大規模モデル、フル微調整 |

### 典型的な学習コスト

| 学習タイプ | ステップ | 時間 | コスト |
|---------------|-------|------|------|
| LoRA (Flux) | 1000 | 30-60 分 | $2-4 |
| LoRA (Flux) | 2000 | 1-2 時間 | $4-8 |
| LoRA (SDXL) | 2000 | 45-90 分 | $3-6 |
| Full fine-tune | 5000+ | 4-12 時間 | $15-50 |

### コスト保護

Cloud タブ → Settings で支出上限を設定:
- 「Cost Limit」を期間（日/週/月）と金額で有効化
- アクション: **Warn** または **Block** を選択

## 結果の受け取り

### オプション 1: HuggingFace Hub（推奨）

1. `HF_TOKEN` 環境変数を設定
2. Publishing タブ → "Push to Hub" を有効化
3. `hub_model_id` を設定（例: `username/my-lora`）

### オプション 2: Webhook によるローカルダウンロード

1. トンネル起動: `ngrok http 8080` または `cloudflared tunnel --url http://localhost:8080`
2. Cloud タブ → **Webhook URL** にトンネル URL を設定
3. モデルは `~/.simpletuner/cloud_outputs/` に保存

### オプション 3: 外部 S3

Publishing タブで S3 送信を設定（AWS S3、MinIO、Backblaze B2 など）。

## ネットワーク設定 {#network}

### API エンドポイント {#api-endpoints}

SimpleTuner が接続する Replicate のエンドポイント:

| 宛先 | 目的 | 必須 |
|-------------|---------|----------|
| `api.replicate.com` | API 呼び出し（送信、状態） | Yes |
| `*.replicate.delivery` | ファイルのアップロード/ダウンロード | Yes |
| `www.replicatestatus.com` | ステータスページ API | No（無くても動作） |
| `api.replicate.com/v1/webhooks/default/secret` | Webhook 署名シークレット | 署名検証時のみ |

### Webhook 送信元 IP {#webhook-ips}

Replicate Webhook は Google Cloud `us-west1` から送信されます:

| IP 範囲 | 備考 |
|----------|-------|
| `34.82.0.0/16` | 主な送信元 |
| `35.185.0.0/16` | セカンダリ範囲 |

最新の IP 範囲は以下を参照:
- [Replicate webhook documentation](https://replicate.com/docs/webhooks)
- [Google の公開 IP 範囲](https://www.gstatic.com/ipranges/cloud.json) から `us-west1` を抽出

<details>
<summary>IP 許可リスト設定例</summary>

```bash
curl -X PUT http://localhost:8080/api/cloud/providers/replicate \
  -H "Content-Type: application/json" \
  -d '{
    "webhook_allowed_ips": ["34.82.0.0/16", "35.185.0.0/16"]
  }'
```

</details>

### ファイアウォール {#firewall}

**アウトバウンド（SimpleTuner → Replicate）:**

| 宛先 | ポート | 目的 |
|-------------|------|---------|
| `api.replicate.com` | 443 | API 呼び出し |
| `*.replicate.delivery` | 443 | ファイル送受信 |
| `replicate.com` | 443 | モデルメタデータ |

<details>
<summary>厳格な egress ルール向け IP 範囲</summary>

Replicate は Google Cloud 上で動作します。厳格なファイアウォールでは以下:

```
34.82.0.0/16
34.83.0.0/16
35.185.0.0/16 - 35.247.0.0/16  (all /16 blocks in this range)
```

**簡易案:** `*.replicate.com` と `*.replicate.delivery` を DNS ベースで許可。

</details>

**インバウンド（Replicate → あなたのサーバ）:**

```
Allow TCP from 34.82.0.0/16, 35.185.0.0/16 to your webhook port
```

## 本番デプロイ

Webhook エンドポイント: **`POST /api/webhooks/replicate`**

Cloud タブで公開 URL（パスなし）を設定すると、SimpleTuner が Webhook パスを自動付与します。

<details>
<summary>nginx 設定</summary>

```nginx
upstream simpletuner {
    server 127.0.0.1:8080;
}

server {
    listen 443 ssl http2;
    server_name training.yourcompany.com;

    ssl_certificate     /etc/ssl/certs/training.crt;
    ssl_certificate_key /etc/ssl/private/training.key;

    location /api/webhooks/ {
        allow 34.82.0.0/16;
        allow 35.185.0.0/16;
        deny all;

        proxy_pass http://simpletuner;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;

        proxy_pass http://simpletuner;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

</details>

<details>
<summary>Caddy 設定</summary>

```caddyfile
training.yourcompany.com {
    @webhook path /api/webhooks/*
    handle @webhook {
        reverse_proxy localhost:8080
    }

    @internal remote_ip 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16
    handle @internal {
        reverse_proxy localhost:8080
    }

    respond "Forbidden" 403
}
```

</details>

<details>
<summary>Traefik 設定（Docker）</summary>

```yaml
services:
  simpletuner:
    image: simpletuner:latest
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.simpletuner.rule=Host(`training.yourcompany.com`)"
      - "traefik.http.routers.simpletuner.tls=true"
      - "traefik.http.services.simpletuner.loadbalancer.server.port=8080"
      - "traefik.http.middlewares.replicate-ips.ipwhitelist.sourcerange=34.82.0.0/16,35.185.0.0/16"
      - "traefik.http.routers.webhook.rule=Host(`training.yourcompany.com`) && PathPrefix(`/api/webhooks`)"
      - "traefik.http.routers.webhook.middlewares=replicate-ips"
      - "traefik.http.routers.webhook.tls=true"
```

</details>

## Webhook 事件 {#webhook-events}

| 事件 | 说明 |
|-------|-------------|
| `start` | 任务开始运行 |
| `logs` | 训练日志输出 |
| `output` | 任务产生输出 |
| `completed` | 任务成功完成 |
| `failed` | 任务失败 |

## 排障 {#troubleshooting}

**"REPLICATE_API_TOKEN not set"**
- 导出变量：`export REPLICATE_API_TOKEN="r8_..."`
- 设置后重启 SimpleTuner

**"Invalid token" 或验证失败**
- Token 应以 `r8_` 开头
- 从 [Replicate dashboard](https://replicate.com/account/api-tokens) 生成新 token
- 检查多余空格或换行

**任务卡在 "queued"**
- Replicate 在 GPU 忙时会排队
- 查看 [Replicate status page](https://replicate.statuspage.io/)

**训练 OOM**
- 降低 batch size
- 启用 gradient checkpointing
- 使用 LoRA 而非全量微调

**Webhook 未收到事件**
- 确认隧道运行且可访问
- 检查 webhook URL 包含 `https://`
- 手动测试：`curl -X POST https://your-url/api/webhooks/replicate -d '{}'`

**代理连接问题**
```bash
# Test proxy connectivity to Replicate
curl -x http://proxy:8080 https://api.replicate.com/v1/account

# Check environment
env | grep -i proxy
```

## API 参考 {#api-reference}

| Endpoint | 说明 |
|----------|-------------|
| `GET /api/cloud/providers/replicate/versions` | 列出模型版本 |
| `GET /api/cloud/providers/replicate/validate` | 验证凭据 |
| `GET /api/cloud/providers/replicate/billing` | 获取余额 |
| `PUT /api/cloud/providers/replicate/token` | 保存 API token |
| `DELETE /api/cloud/providers/replicate/token` | 删除 API token |
| `POST /api/cloud/jobs/submit` | 提交训练任务 |
| `POST /api/webhooks/replicate` | Webhook 接收端 |

## 链接

- [Replicate Documentation](https://replicate.com/docs)
- [SimpleTuner on Replicate](https://replicate.com/simpletuner/advanced-trainer)
- [Replicate API Tokens](https://replicate.com/account/api-tokens)
- [Replicate Status Page](https://replicate.statuspage.io/)
