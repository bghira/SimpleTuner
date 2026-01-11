# Integracao com Replicate

Replicate e uma plataforma de nuvem para rodar modelos de ML. O SimpleTuner usa o sistema de containers Cog da Replicate para rodar jobs de treinamento em GPUs na nuvem.

- **Modelo:** `simpletuner/advanced-trainer`
- **GPU padrao:** L40S (48GB VRAM)

## Inicio rapido

1. Crie uma [conta Replicate](https://replicate.com/signin) e obtenha um [token de API](https://replicate.com/account/api-tokens)
2. Defina a variavel de ambiente:
   ```bash
   export REPLICATE_API_TOKEN="r8_your_token_here"
   simpletuner server
   ```
3. Abra a web UI → aba Cloud → clique em **Validate** para verificar

## Fluxo de dados

| Tipo de dado | Destino | Retencao |
|-----------|-------------|-----------|
| Imagens de treinamento | Servidores de upload Replicate (GCP) | Deletado apos o job |
| Config de treinamento | API da Replicate | Armazenado com metadados do job |
| Token de API | Apenas no seu ambiente | Nunca armazenado pelo SimpleTuner |
| Modelo treinado | HuggingFace Hub, S3 ou local | Sob seu controle |
| Logs do job | Servidores da Replicate | 30 dias |

**Limite de upload:** A API de upload de arquivos da Replicate aceita arquivos de ate 100 MiB. O SimpleTuner bloqueia envios quando o arquivo compactado excede esse limite.

<details>
<summary>Detalhes do caminho de dados</summary>

1. **Upload:** Imagens locais → HTTPS POST → `api.replicate.com`
2. **Treinamento:** Replicate baixa dados para uma instancia GPU efemera
3. **Saida:** Modelo treinado → destino configurado por voce
4. **Limpeza:** Replicate deleta dados de treinamento apos a conclusao do job

Veja [Replicate Security Docs](https://replicate.com/docs/reference/security) para mais.

</details>

## Hardware e custos {#costs}

| Hardware | VRAM | Custo | Melhor para |
|----------|------|------|----------|
| L40S | 48GB | ~$3.50/hr | Maioria dos treinamentos LoRA |
| A100 (80GB) | 80GB | ~$5.00/hr | Modelos grandes, fine-tuning completo |

### Custos tipicos de treinamento

| Tipo de treinamento | Steps | Tempo | Custo |
|---------------|-------|------|------|
| LoRA (Flux) | 1000 | 30-60 min | $2-4 |
| LoRA (Flux) | 2000 | 1-2 horas | $4-8 |
| LoRA (SDXL) | 2000 | 45-90 min | $3-6 |
| Fine-tuning completo | 5000+ | 4-12 horas | $15-50 |

### Protecao de custos

Defina limites de gasto na aba Cloud → Settings:
- Ative "Cost Limit" com valor/periodo (diario/semanal/mensal)
- Escolha a acao: **Warn** ou **Block**

## Entrega de resultados

### Opcao 1: HuggingFace Hub (recomendado)

1. Defina a variavel de ambiente `HF_TOKEN`
2. Aba Publishing → habilite "Push to Hub"
3. Defina `hub_model_id` (ex.: `username/my-lora`)

### Opcao 2: Download local via webhook

1. Inicie um tunel: `ngrok http 8080` ou `cloudflared tunnel --url http://localhost:8080`
2. Aba Cloud → defina **Webhook URL** para a URL do tunel
3. Modelos sao baixados para `~/.simpletuner/cloud_outputs/`

### Opcao 3: S3 externo

Configure a publicacao S3 na aba Publishing (AWS S3, MinIO, Backblaze B2, etc.).

## Configuracao de rede {#network}

### Endpoints de API {#api-endpoints}

O SimpleTuner se conecta a estes endpoints da Replicate:

| Destino | Proposito | Obrigatorio |
|-------------|---------|----------|
| `api.replicate.com` | Chamadas de API (envio de job, status) | Sim |
| `*.replicate.delivery` | Upload/download de arquivos | Sim |
| `www.replicatestatus.com` | API de pagina de status | Nao (degrada graciosamente) |
| `api.replicate.com/v1/webhooks/default/secret` | Segredo de assinatura de webhook | Apenas se validacao de assinatura estiver habilitada |

### IPs de origem de webhook {#webhook-ips}

Webhooks da Replicate se originam da regiao `us-west1` do Google Cloud:

| Range de IP | Notas |
|----------|-------|
| `34.82.0.0/16` | Origem primaria de webhook |
| `35.185.0.0/16` | Range secundario |

Para os ranges de IP mais atuais:
- Confira a [documentacao de webhooks da Replicate](https://replicate.com/docs/webhooks)
- Ou use os [ranges publicados do Google](https://www.gstatic.com/ipranges/cloud.json) filtrados por `us-west1`

<details>
<summary>Exemplo de configuracao de allowlist de IP</summary>

```bash
curl -X PUT http://localhost:8080/api/cloud/providers/replicate \
  -H "Content-Type: application/json" \
  -d '{
    "webhook_allowed_ips": ["34.82.0.0/16", "35.185.0.0/16"]
  }'
```

</details>

### Regras de firewall {#firewall}

**Outbound (SimpleTuner → Replicate):**

| Destino | Porta | Proposito |
|-------------|------|---------|
| `api.replicate.com` | 443 | Chamadas de API |
| `*.replicate.delivery` | 443 | Upload/download de arquivos |
| `replicate.com` | 443 | Metadados de modelo |

<details>
<summary>Ranges de IP para regras de egress estritas</summary>

A Replicate roda no Google Cloud. Para regras estritas de firewall:

```
34.82.0.0/16
34.83.0.0/16
35.185.0.0/16 - 35.247.0.0/16  (all /16 blocks in this range)
```

**Alternativa mais simples:** Permitir egress baseado em DNS para `*.replicate.com` e `*.replicate.delivery`.

</details>

**Inbound (Replicate → Seu servidor):**

```
Allow TCP from 34.82.0.0/16, 35.185.0.0/16 to your webhook port
```

## Deploy em producao

Endpoint de webhook: **`POST /api/webhooks/replicate`**

Defina sua URL publica (sem path) na aba Cloud. O SimpleTuner anexa o path do webhook automaticamente.

<details>
<summary>Configuracao nginx</summary>

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
<summary>Configuracao Caddy</summary>

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
<summary>Configuracao Traefik (Docker)</summary>

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

## Eventos de webhook {#webhook-events}

| Evento | Descricao |
|-------|-------------|
| `start` | Job comecou a rodar |
| `logs` | Saida de log do treinamento |
| `output` | Job produziu output |
| `completed` | Job finalizou com sucesso |
| `failed` | Job falhou com erro |

## Solucao de problemas {#troubleshooting}

**"REPLICATE_API_TOKEN not set"**
- Exporte a variavel: `export REPLICATE_API_TOKEN="r8_..."`
- Reinicie o SimpleTuner apos definir

**"Invalid token" ou validacao falha**
- Token deve comecar com `r8_`
- Gere um novo token no [dashboard da Replicate](https://replicate.com/account/api-tokens)
- Verifique espacos ou quebras de linha extras

**Job travado em "queued"**
- Replicate coloca jobs em fila quando GPUs estao ocupadas
- Verifique a [pagina de status da Replicate](https://replicate.statuspage.io/)

**Treinamento falha com OOM**
- Reduza batch size
- Ative gradient checkpointing
- Use LoRA em vez de fine-tuning completo

**Webhook nao recebe eventos**
- Verifique se o tunel esta rodando e acessivel
- Verifique se a URL do webhook inclui `https://`
- Teste manualmente: `curl -X POST https://your-url/api/webhooks/replicate -d '{}'`

**Problemas de conexao atraves de proxy**
```bash
# Test proxy connectivity to Replicate
curl -x http://proxy:8080 https://api.replicate.com/v1/account

# Check environment
env | grep -i proxy
```

## Referencia de API {#api-reference}

| Endpoint | Descricao |
|----------|-------------|
| `GET /api/cloud/providers/replicate/versions` | Listar versoes do modelo |
| `GET /api/cloud/providers/replicate/validate` | Validar credenciais |
| `GET /api/cloud/providers/replicate/billing` | Obter saldo de creditos |
| `PUT /api/cloud/providers/replicate/token` | Salvar token de API |
| `DELETE /api/cloud/providers/replicate/token` | Deletar token de API |
| `POST /api/cloud/jobs/submit` | Enviar job de treinamento |
| `POST /api/webhooks/replicate` | Receiver de webhook |

## Links

- [Documentacao da Replicate](https://replicate.com/docs)
- [SimpleTuner na Replicate](https://replicate.com/simpletuner/advanced-trainer)
- [Tokens de API da Replicate](https://replicate.com/account/api-tokens)
- [Pagina de status da Replicate](https://replicate.statuspage.io/)
