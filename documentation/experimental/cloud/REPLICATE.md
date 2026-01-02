# Replicate Integration

Replicate is a cloud platform for running ML models. SimpleTuner uses Replicate's Cog container system to run training jobs on cloud GPUs.

- **Model:** `simpletuner/advanced-trainer`
- **Default GPU:** L40S (48GB VRAM)

## Quick Start

1. Create a [Replicate account](https://replicate.com/signin) and get an [API token](https://replicate.com/account/api-tokens)
2. Set the environment variable:
   ```bash
   export REPLICATE_API_TOKEN="r8_your_token_here"
   simpletuner server
   ```
3. Open the web UI → Cloud tab → click **Validate** to verify

## Data Flow

| Data Type | Destination | Retention |
|-----------|-------------|-----------|
| Training images | Replicate upload servers (GCP) | Deleted after job |
| Training config | Replicate API | Stored with job metadata |
| API token | Your environment only | Never stored by SimpleTuner |
| Trained model | HuggingFace Hub, S3, or local | Your control |
| Job logs | Replicate servers | 30 days |

<details>
<summary>Data path details</summary>

1. **Upload:** Local images → HTTPS POST → `api.replicate.com`
2. **Training:** Replicate downloads data to ephemeral GPU instance
3. **Output:** Trained model → Your configured destination
4. **Cleanup:** Replicate deletes training data after job completion

See [Replicate Security Docs](https://replicate.com/docs/reference/security) for more.

</details>

## Hardware & Costs

| Hardware | VRAM | Cost | Best For |
|----------|------|------|----------|
| L40S | 48GB | ~$3.50/hr | Most LoRA training |
| A100 (80GB) | 80GB | ~$5.00/hr | Large models, full fine-tuning |

### Typical Training Costs

| Training Type | Steps | Time | Cost |
|---------------|-------|------|------|
| LoRA (Flux) | 1000 | 30-60 min | $2-4 |
| LoRA (Flux) | 2000 | 1-2 hours | $4-8 |
| LoRA (SDXL) | 2000 | 45-90 min | $3-6 |
| Full fine-tune | 5000+ | 4-12 hours | $15-50 |

### Cost Protection

Set spending limits in Cloud tab → Settings:
- Enable "Cost Limit" with amount/period (daily/weekly/monthly)
- Choose action: **Warn** or **Block**

## Results Delivery

### Option 1: HuggingFace Hub (Recommended)

1. Set `HF_TOKEN` environment variable
2. Publishing tab → enable "Push to Hub"
3. Set `hub_model_id` (e.g., `username/my-lora`)

### Option 2: Local Download via Webhook

1. Start a tunnel: `ngrok http 8080` or `cloudflared tunnel --url http://localhost:8080`
2. Cloud tab → set **Webhook URL** to tunnel URL
3. Models download to `~/.simpletuner/cloud_outputs/`

### Option 3: External S3

Configure S3 publishing in the Publishing tab (AWS S3, MinIO, Backblaze B2, etc.).

## Network Configuration {#network}

### API Endpoints {#api-endpoints}

SimpleTuner connects to these Replicate endpoints:

| Destination | Purpose | Required |
|-------------|---------|----------|
| `api.replicate.com` | API calls (job submission, status) | Yes |
| `*.replicate.delivery` | File uploads/downloads | Yes |
| `www.replicatestatus.com` | Status page API | No (degrades gracefully) |
| `api.replicate.com/v1/webhooks/default/secret` | Webhook signing secret | Only if signature validation enabled |

### Webhook Source IPs {#webhook-ips}

Replicate webhooks originate from Google Cloud's `us-west1` region:

| IP Range | Notes |
|----------|-------|
| `34.82.0.0/16` | Primary webhook source |
| `35.185.0.0/16` | Secondary range |

For the most current IP ranges:
- Check [Replicate webhook documentation](https://replicate.com/docs/webhooks)
- Or use [Google's published IP ranges](https://www.gstatic.com/ipranges/cloud.json) filtered for `us-west1`

<details>
<summary>IP allowlist configuration example</summary>

```bash
curl -X PUT http://localhost:8080/api/cloud/providers/replicate \
  -H "Content-Type: application/json" \
  -d '{
    "webhook_allowed_ips": ["34.82.0.0/16", "35.185.0.0/16"]
  }'
```

</details>

### Firewall Rules {#firewall}

**Outbound (SimpleTuner → Replicate):**

| Destination | Port | Purpose |
|-------------|------|---------|
| `api.replicate.com` | 443 | API calls |
| `*.replicate.delivery` | 443 | File uploads/downloads |
| `replicate.com` | 443 | Model metadata |

<details>
<summary>IP ranges for strict egress rules</summary>

Replicate runs on Google Cloud. For strict firewall rules:

```
34.82.0.0/16
34.83.0.0/16
35.185.0.0/16 - 35.247.0.0/16  (all /16 blocks in this range)
```

**Simpler alternative:** Allow DNS-based egress to `*.replicate.com` and `*.replicate.delivery`.

</details>

**Inbound (Replicate → Your Server):**

```
Allow TCP from 34.82.0.0/16, 35.185.0.0/16 to your webhook port
```

## Production Deployment

Webhook endpoint: **`POST /api/webhooks/replicate`**

Set your public URL (without path) in the Cloud tab. SimpleTuner appends the webhook path automatically.

<details>
<summary>nginx configuration</summary>

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
<summary>Caddy configuration</summary>

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
<summary>Traefik configuration (Docker)</summary>

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

## Webhook Events {#webhook-events}

| Event | Description |
|-------|-------------|
| `start` | Job started running |
| `logs` | Training log output |
| `output` | Job produced output |
| `completed` | Job finished successfully |
| `failed` | Job failed with error |

## Troubleshooting {#troubleshooting}

**"REPLICATE_API_TOKEN not set"**
- Export the variable: `export REPLICATE_API_TOKEN="r8_..."`
- Restart SimpleTuner after setting it

**"Invalid token" or validation fails**
- Token should start with `r8_`
- Generate a new token from [Replicate dashboard](https://replicate.com/account/api-tokens)
- Check for extra spaces or newlines

**Job stuck in "queued"**
- Replicate queues jobs when GPUs are busy
- Check [Replicate status page](https://replicate.statuspage.io/)

**Training fails with OOM**
- Reduce batch size
- Enable gradient checkpointing
- Use LoRA instead of full fine-tuning

**Webhook not receiving events**
- Verify tunnel is running and accessible
- Check webhook URL includes `https://`
- Test manually: `curl -X POST https://your-url/api/webhooks/replicate -d '{}'`

**Connection issues through proxy**
```bash
# Test proxy connectivity to Replicate
curl -x http://proxy:8080 https://api.replicate.com/v1/account

# Check environment
env | grep -i proxy
```

## API Reference {#api-reference}

| Endpoint | Description |
|----------|-------------|
| `GET /api/cloud/replicate/versions` | List model versions |
| `GET /api/cloud/replicate/validate` | Validate credentials |
| `GET /api/cloud/replicate/billing` | Get credit balance |
| `POST /api/cloud/replicate/submit` | Submit training job |
| `POST /api/webhooks/replicate` | Webhook receiver |

## Links

- [Replicate Documentation](https://replicate.com/docs)
- [SimpleTuner on Replicate](https://replicate.com/simpletuner/advanced-trainer)
- [Replicate API Tokens](https://replicate.com/account/api-tokens)
- [Replicate Status Page](https://replicate.statuspage.io/)
