# Replicate Integration

Replicate ML models चलाने के लिए एक cloud प्लेटफ़ॉर्म है। SimpleTuner Replicate के Cog container सिस्टम का उपयोग करके cloud GPUs पर training jobs चलाता है।

- **Model:** `simpletuner/advanced-trainer`
- **Default GPU:** L40S (48GB VRAM)

## Quick Start

1. [Replicate account](https://replicate.com/signin) बनाएं और [API token](https://replicate.com/account/api-tokens) प्राप्त करें
2. Environment variable सेट करें:
   ```bash
   export REPLICATE_API_TOKEN="r8_your_token_here"
   simpletuner server
   ```
3. Web UI → Cloud tab → **Validate** क्लिक करें

## Data Flow

| Data Type | Destination | Retention |
|-----------|-------------|-----------|
| Training images | Replicate upload servers (GCP) | Job के बाद delete |
| Training config | Replicate API | Job metadata के साथ stored |
| API token | केवल आपका environment | SimpleTuner कभी स्टोर नहीं करता |
| Trained model | HuggingFace Hub, S3, या local | आपके नियंत्रण में |
| Job logs | Replicate servers | 30 days |

**Upload limit:** Replicate की file upload API 100 MiB तक के archive स्वीकार करती है। SimpleTuner जब packaged archive इस सीमा से बड़ा हो, तो submission को block करता है।

<details>
<summary>Data path details</summary>

1. **Upload:** Local images → HTTPS POST → `api.replicate.com`
2. **Training:** Replicate डेटा को ephemeral GPU instance पर डाउनलोड करता है
3. **Output:** Trained model → आपका configured destination
4. **Cleanup:** Job पूरा होने के बाद Replicate training data हटाता है

अधिक जानकारी के लिए [Replicate Security Docs](https://replicate.com/docs/reference/security) देखें।

</details>

## Hardware & Costs {#costs}

| Hardware | VRAM | Cost | Best For |
|----------|------|------|----------|
| L40S | 48GB | ~$3.50/hr | अधिकांश LoRA training |
| A100 (80GB) | 80GB | ~$5.00/hr | बड़े मॉडल, full fine-tuning |

### Typical Training Costs

| Training Type | Steps | Time | Cost |
|---------------|-------|------|------|
| LoRA (Flux) | 1000 | 30-60 min | $2-4 |
| LoRA (Flux) | 2000 | 1-2 hours | $4-8 |
| LoRA (SDXL) | 2000 | 45-90 min | $3-6 |
| Full fine-tune | 5000+ | 4-12 hours | $15-50 |

### Cost Protection

Cloud tab → Settings में spending limits सेट करें:
- amount/period (daily/weekly/monthly) के साथ "Cost Limit" सक्षम करें
- action चुनें: **Warn** या **Block**

## Results Delivery

### Option 1: HuggingFace Hub (Recommended)

1. `HF_TOKEN` environment variable सेट करें
2. Publishing tab → "Push to Hub" सक्षम करें
3. `hub_model_id` सेट करें (जैसे `username/my-lora`)

### Option 2: Webhook के जरिए Local Download

1. Tunnel शुरू करें: `ngrok http 8080` या `cloudflared tunnel --url http://localhost:8080`
2. Cloud tab → **Webhook URL** में tunnel URL सेट करें
3. Models `~/.simpletuner/cloud_outputs/` में डाउनलोड होंगे

### Option 3: External S3

Publishing tab में S3 publishing कॉन्फ़िगर करें (AWS S3, MinIO, Backblaze B2, आदि)।

## Network Configuration {#network}

### API Endpoints {#api-endpoints}

SimpleTuner इन Replicate endpoints से कनेक्ट करता है:

| Destination | Purpose | Required |
|-------------|---------|----------|
| `api.replicate.com` | API calls (job submission, status) | Yes |
| `*.replicate.delivery` | File uploads/downloads | Yes |
| `www.replicatestatus.com` | Status page API | No (degrades gracefully) |
| `api.replicate.com/v1/webhooks/default/secret` | Webhook signing secret | केवल जब signature validation सक्षम हो |

### Webhook Source IPs {#webhook-ips}

Replicate webhooks Google Cloud के `us-west1` region से आते हैं:

| IP Range | Notes |
|----------|-------|
| `34.82.0.0/16` | Primary webhook source |
| `35.185.0.0/16` | Secondary range |

सबसे वर्तमान IP ranges के लिए:
- [Replicate webhook documentation](https://replicate.com/docs/webhooks) देखें
- या [Google's published IP ranges](https://www.gstatic.com/ipranges/cloud.json) में `us-west1` फ़िल्टर करें

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
<summary>Strict egress rules के लिए IP ranges</summary>

Replicate Google Cloud पर चलता है। strict firewall rules के लिए:

```
34.82.0.0/16
34.83.0.0/16
35.185.0.0/16 - 35.247.0.0/16  (इस रेंज के सभी /16 blocks)
```

**सरल विकल्प:** DNS-based egress को `*.replicate.com` और `*.replicate.delivery` तक allow करें।

</details>

**Inbound (Replicate → आपका सर्वर):**

```
Allow TCP from 34.82.0.0/16, 35.185.0.0/16 to your webhook port
```

## Production Deployment

Webhook endpoint: **`POST /api/webhooks/replicate`**

Cloud tab में अपना public URL (path के बिना) सेट करें। SimpleTuner webhook path अपने आप जोड़ देता है।

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
- variable export करें: `export REPLICATE_API_TOKEN="r8_..."`
- सेट करने के बाद SimpleTuner restart करें

**"Invalid token" या validation fails**
- Token `r8_` से शुरू होना चाहिए
- [Replicate dashboard](https://replicate.com/account/api-tokens) से नया token बनाएं
- extra spaces या newlines जांचें

**Job "queued" में अटका**
- GPUs busy होने पर Replicate jobs queue करता है
- [Replicate status page](https://replicate.statuspage.io/) देखें

**Training fails with OOM**
- batch size घटाएँ
- gradient checkpointing सक्षम करें
- full fine-tuning की जगह LoRA उपयोग करें

**Webhook events नहीं मिल रहे**
- सुनिश्चित करें कि tunnel चल रहा है और accessible है
- webhook URL में `https://` शामिल हो
- मैन्युअल टेस्ट: `curl -X POST https://your-url/api/webhooks/replicate -d '{}'`

**Proxy के जरिए कनेक्शन issues**
```bash
# Replicate तक proxy connectivity टेस्ट करें
curl -x http://proxy:8080 https://api.replicate.com/v1/account

# Environment जांचें
env | grep -i proxy
```

## API Reference {#api-reference}

| Endpoint | Description |
|----------|-------------|
| `GET /api/cloud/providers/replicate/versions` | Model versions सूचीबद्ध करें |
| `GET /api/cloud/providers/replicate/validate` | Credentials validate करें |
| `GET /api/cloud/providers/replicate/billing` | Credit balance प्राप्त करें |
| `PUT /api/cloud/providers/replicate/token` | API token save करें |
| `DELETE /api/cloud/providers/replicate/token` | API token हटाएँ |
| `POST /api/cloud/jobs/submit` | Training job submit करें |
| `POST /api/webhooks/replicate` | Webhook receiver |

## Links

- [Replicate Documentation](https://replicate.com/docs)
- [SimpleTuner on Replicate](https://replicate.com/simpletuner/advanced-trainer)
- [Replicate API Tokens](https://replicate.com/account/api-tokens)
- [Replicate Status Page](https://replicate.statuspage.io/)
