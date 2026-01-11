# Replicate 集成

Replicate 是用于运行机器学习模型的云平台。SimpleTuner 使用 Replicate 的 Cog 容器系统在云端 GPU 上运行训练任务。

- **模型：** `simpletuner/advanced-trainer`
- **默认 GPU：** L40S（48GB 显存）

## 快速开始

1. 创建 [Replicate 账号](https://replicate.com/signin) 并获取 [API Token](https://replicate.com/account/api-tokens)
2. 设置环境变量：
   ```bash
   export REPLICATE_API_TOKEN="r8_your_token_here"
   simpletuner server
   ```
3. 打开 Web UI → Cloud 选项卡 → 点击 **Validate** 验证

## 数据流向

| 数据类型 | 目的地 | 保留时间 |
|-----------|-------------|-----------|
| 训练图像 | Replicate 上传服务器（GCP） | 任务完成后删除 |
| 训练配置 | Replicate API | 随任务元数据存储 |
| API Token | 仅保存在你的环境中 | SimpleTuner 从不存储 |
| 训练模型 | HuggingFace Hub、S3 或本地 | 由你控制 |
| 任务日志 | Replicate 服务器 | 30 天 |

**上传限制：** Replicate 的文件上传 API 仅支持最多 100 MiB 的归档文件。SimpleTuner 会在打包归档超过该限制时阻止提交。

<details>
<summary>数据路径细节</summary>

1. **上传：** 本地图像 → HTTPS POST → `api.replicate.com`
2. **训练：** Replicate 将数据下载到临时 GPU 实例
3. **输出：** 训练模型 → 你配置的目标
4. **清理：** 任务完成后 Replicate 删除训练数据

更多信息参见 [Replicate 安全文档](https://replicate.com/docs/reference/security)。

</details>

## 硬件与成本 {#costs}

| 硬件 | 显存 | 成本 | 适用场景 |
|----------|------|------|----------|
| L40S | 48GB | ~ $3.50/小时 | 大多数 LoRA 训练 |
| A100（80GB） | 80GB | ~ $5.00/小时 | 大模型、全量微调 |

### 典型训练成本

| 训练类型 | 步数 | 时间 | 成本 |
|---------------|-------|------|------|
| LoRA（Flux） | 1000 | 30-60 分钟 | $2-4 |
| LoRA（Flux） | 2000 | 1-2 小时 | $4-8 |
| LoRA（SDXL） | 2000 | 45-90 分钟 | $3-6 |
| 全量微调 | 5000+ | 4-12 小时 | $15-50 |

### 成本保护

在 Cloud 选项卡 → Settings 中设置支出上限：
- 启用“Cost Limit”并设置金额/周期（每日/每周/每月）
- 选择动作：**Warn** 或 **Block**

## 结果交付

### 选项 1：HuggingFace Hub（推荐）

1. 设置 `HF_TOKEN` 环境变量
2. Publishing 选项卡 → 启用 “Push to Hub”
3. 设置 `hub_model_id`（例如 `username/my-lora`）

### 选项 2：通过 Webhook 本地下载

1. 启动隧道：`ngrok http 8080` 或 `cloudflared tunnel --url http://localhost:8080`
2. Cloud 选项卡 → 设置 **Webhook URL** 为隧道 URL
3. 模型下载到 `~/.simpletuner/cloud_outputs/`

### 选项 3：外部 S3

在 Publishing 选项卡配置 S3 发布（AWS S3、MinIO、Backblaze B2 等）。

## 网络配置 {#network}

### API 端点 {#api-endpoints}

SimpleTuner 会连接这些 Replicate 端点：

| 目的地 | 用途 | 必需 |
|-------------|---------|----------|
| `api.replicate.com` | API 调用（任务提交、状态） | 是 |
| `*.replicate.delivery` | 文件上传/下载 | 是 |
| `www.replicatestatus.com` | 状态页 API | 否（会优雅降级） |
| `api.replicate.com/v1/webhooks/default/secret` | Webhook 签名密钥 | 仅在启用签名校验时需要 |

### Webhook 源 IP {#webhook-ips}

Replicate 的 Webhook 来自 Google Cloud 的 `us-west1` 区域：

| IP 段 | 说明 |
|----------|-------|
| `34.82.0.0/16` | 主要 Webhook 来源 |
| `35.185.0.0/16` | 次要范围 |

最新 IP 段请参考：
- [Replicate Webhook 文档](https://replicate.com/docs/webhooks)
- 或 [Google 公布的 IP 段](https://www.gstatic.com/ipranges/cloud.json)，筛选 `us-west1`

<details>
<summary>IP 允许列表配置示例</summary>

```bash
curl -X PUT http://localhost:8080/api/cloud/providers/replicate \
  -H "Content-Type: application/json" \
  -d '{
    "webhook_allowed_ips": ["34.82.0.0/16", "35.185.0.0/16"]
  }'
```

</details>

### 防火墙规则 {#firewall}

**出站（SimpleTuner → Replicate）：**

| 目的地 | 端口 | 用途 |
|-------------|------|---------|
| `api.replicate.com` | 443 | API 调用 |
| `*.replicate.delivery` | 443 | 文件上传/下载 |
| `replicate.com` | 443 | 模型元数据 |

<details>
<summary>严格出站规则的 IP 段</summary>

Replicate 运行在 Google Cloud 上。若需要严格的防火墙规则：

```
34.82.0.0/16
34.83.0.0/16
35.185.0.0/16 - 35.247.0.0/16  (该范围内所有 /16 段)
```

**更简单的替代方案：** 允许基于 DNS 的出站到 `*.replicate.com` 和 `*.replicate.delivery`。

</details>

**入站（Replicate → 你的服务器）：**

```
允许来自 34.82.0.0/16、35.185.0.0/16 到 webhook 端口的 TCP
```

## 生产部署

Webhook 端点：**`POST /api/webhooks/replicate`**

在 Cloud 选项卡中设置你的公网 URL（不含路径）。SimpleTuner 会自动追加 webhook 路径。

<details>
<summary>nginx 配置</summary>

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
<summary>Caddy 配置</summary>

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
<summary>Traefik 配置（Docker）</summary>

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

| 事件 | 描述 |
|-------|-------------|
| `start` | 任务开始运行 |
| `logs` | 训练日志输出 |
| `output` | 任务产生输出 |
| `completed` | 任务成功完成 |
| `failed` | 任务失败并返回错误 |

## 故障排查 {#troubleshooting}

**"REPLICATE_API_TOKEN not set"**
- 导出变量：`export REPLICATE_API_TOKEN="r8_..."`
- 设置后重启 SimpleTuner

**"Invalid token" 或验证失败**
- Token 应以 `r8_` 开头
- 在 [Replicate 控制台](https://replicate.com/account/api-tokens) 生成新 token
- 检查是否有多余空格或换行

**任务卡在 "queued"**
- Replicate 在 GPU 繁忙时会排队
- 查看 [Replicate 状态页](https://replicate.statuspage.io/)

**训练 OOM 失败**
- 减小 batch size
- 启用梯度检查点
- 使用 LoRA 而不是全量微调

**Webhook 未收到事件**
- 确认隧道运行且可访问
- 检查 webhook URL 是否包含 `https://`
- 手动测试：`curl -X POST https://your-url/api/webhooks/replicate -d '{}'`

**通过代理连接异常**
```bash
# 测试代理连通性
curl -x http://proxy:8080 https://api.replicate.com/v1/account

# 检查环境
env | grep -i proxy
```

## API 参考 {#api-reference}

| 端点 | 描述 |
|----------|-------------|
| `GET /api/cloud/providers/replicate/versions` | 列出模型版本 |
| `GET /api/cloud/providers/replicate/validate` | 验证凭据 |
| `GET /api/cloud/providers/replicate/billing` | 获取余额 |
| `PUT /api/cloud/providers/replicate/token` | 保存 API token |
| `DELETE /api/cloud/providers/replicate/token` | 删除 API token |
| `POST /api/cloud/jobs/submit` | 提交训练任务 |
| `POST /api/webhooks/replicate` | Webhook 接收端 |

## 链接

- [Replicate 文档](https://replicate.com/docs)
- [SimpleTuner on Replicate](https://replicate.com/simpletuner/advanced-trainer)
- [Replicate API Tokens](https://replicate.com/account/api-tokens)
- [Replicate 状态页](https://replicate.statuspage.io/)
