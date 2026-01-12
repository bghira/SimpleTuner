# Tutorial de treinamento em nuvem

Este guia mostra como rodar jobs de treinamento do SimpleTuner em infraestrutura de GPU na nuvem. Ele cobre tanto o fluxo na Web UI quanto na API REST.

## Pre-requisitos

- SimpleTuner instalado e servidor rodando (veja o [tutorial de API local](../../api/TUTORIAL.md#start-the-server))
- Datasets preparados localmente com captions (mesmos [requisitos de dataset](../../api/TUTORIAL.md#optional-upload-datasets-over-the-api-local-backends) do treinamento local)
- Conta em um provedor de nuvem (veja [Provedores suportados](#provider-setup))
- Para uso da API: um shell com `curl` e `jq`

## Setup do provedor {#provider-setup}

O treinamento em nuvem exige credenciais do provedor escolhido. Siga o guia de setup do seu provedor:

| Provedor | Guia de setup |
|----------|-------------|
| Replicate | [REPLICATE.md](REPLICATE.md#quick-start) |

Apos concluir o setup do provedor, volte aqui para enviar jobs.

## Inicio rapido

Com seu provedor configurado:

1. Abra `http://localhost:8001` e va para a aba **Cloud**
2. Verifique suas credenciais em **Settings** (icone de engrenagem) → **Validate**
3. Configure seu treinamento nas abas Model/Training/Dataloader
4. Clique em **Train in Cloud**
5. Revise o resumo de upload e clique em **Submit**

**Limite de upload (Replicate):** Arquivos empacotados devem ter 100 MiB ou menos. Uploads maiores sao bloqueados antes do envio.

## Recebendo modelos treinados

Depois que o treinamento termina, seu modelo precisa de um destino. Configure um antes do seu primeiro job.

### Opcao 1: HuggingFace Hub (recomendado)

Enviar direto para sua conta HuggingFace:

1. Obtenha um [token do HuggingFace](https://huggingface.co/settings/tokens) com permissao de escrita
2. Defina a variavel de ambiente:
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```
3. Na aba **Publishing**, habilite "Push to Hub" e defina o nome do repo

### Opcao 2: Download local via webhook

Faça os modelos voltarem para sua maquina. Requer expor seu servidor para a internet.

1. Inicie um tunel:
   ```bash
   ngrok http 8001   # ou: cloudflared tunnel --url http://localhost:8001
   ```
2. Copie a URL publica (ex.: `https://abc123.ngrok.io`)
3. Na aba Cloud → Settings → Webhook URL, cole a URL
4. Modelos chegam em `~/.simpletuner/cloud_outputs/`

### Opcao 3: S3 externo

Faça upload para qualquer endpoint compativel com S3 (AWS S3, MinIO, Backblaze B2, Cloudflare R2):

1. Na aba **Publishing**, configure as configuracoes S3
2. Informe endpoint, bucket, access key e secret key

## Fluxo da Web UI

### Enviando jobs

1. **Configure seu treinamento** nas abas Model/Training/Dataloader
2. **Navegue para a aba Cloud** e selecione seu provedor
3. **Clique em Train in Cloud** para abrir o dialogo pre-envio
4. **Revise o resumo de upload** — datasets locais serao empacotados e enviados
5. **Opcionalmente defina um run name** para tracking
6. **Clique em Submit**

### Monitorando jobs

A lista de jobs mostra todos os jobs em nuvem e locais com:

- **Indicador de status**: Queued → Running → Completed/Failed
- **Progresso ao vivo**: Step de treinamento, valores de loss (quando disponiveis)
- **Rastreamento de custo**: Custo estimado baseado no tempo de GPU

Clique em um job para ver detalhes:
- Snapshot de configuracao do job
- Logs em tempo real (clique em **View Logs**)
- Acoes: Cancel, Delete (apos conclusao)

### Painel de configuracoes

Clique no icone de engrenagem para acessar:

- **Validacao de API key** e status de conta
- **Webhook URL** para entrega local do modelo
- **Limites de custo** para evitar gastos descontrolados
- **Info de hardware** (tipo de GPU, custo por hora)

## Fluxo da API

### Enviar um job

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name_to_load": "my-training-config",
    "tracker_run_name": "api-test-run"
  }' | jq
```

Substitua `PROVIDER` pelo nome do seu provedor (ex.: `replicate`).

Ou envie com config inline:

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -d '{
    "config": {
      "--model_family": "flux",
      "--model_type": "lora",
      "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
      "--output_dir": "/outputs/flux-lora",
      "--max_train_steps": 1000,
      "--lora_rank": 16
    },
    "dataloader_config": [
      {
        "id": "training-images",
        "type": "local",
        "dataset_type": "image",
        "instance_data_dir": "/data/datasets/my-dataset",
        "caption_strategy": "textfile",
        "resolution": 1024
      }
    ]
  }' | jq
```

### Monitorar status do job

```bash
# Get job details
curl -s http://localhost:8001/api/cloud/jobs/JOB_ID | jq

# List all jobs
curl -s 'http://localhost:8001/api/cloud/jobs?limit=10' | jq

# Sync status of active jobs from provider
curl -s 'http://localhost:8001/api/cloud/jobs?sync_active=true' | jq
```

### Buscar logs do job

```bash
curl -s http://localhost:8001/api/cloud/jobs/JOB_ID/logs | jq '.logs'
```

### Cancelar um job em execucao

```bash
curl -s -X POST http://localhost:8001/api/cloud/jobs/JOB_ID/cancel | jq
```

### Deletar um job concluido

```bash
curl -s -X DELETE http://localhost:8001/api/cloud/jobs/JOB_ID | jq
```

## Integracao CI/CD

### Envio idempotente de jobs

Evite jobs duplicados com chaves de idempotencia:

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name_to_load": "my-config",
    "idempotency_key": "ci-build-12345"
  }' | jq
```

Se a mesma chave for enviada novamente dentro de 24 horas, voce recebe o job existente em vez de criar um duplicado.

### Exemplo de GitHub Actions

```yaml
name: Cloud Training

on:
  push:
    branches: [main]
    paths:
      - 'training-configs/**'

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Submit Training Job
        env:
          SIMPLETUNER_URL: ${{ secrets.SIMPLETUNER_URL }}
        run: |
          RESPONSE=$(curl -s -X POST "$SIMPLETUNER_URL/api/cloud/jobs/submit?provider=replicate" \
            -H 'Content-Type: application/json' \
            -d '{
              "config_name_to_load": "production-lora",
              "idempotency_key": "gh-${{ github.sha }}",
              "tracker_run_name": "gh-run-${{ github.run_number }}"
            }')

          JOB_ID=$(echo $RESPONSE | jq -r '.job_id')
          echo "Submitted job: $JOB_ID"
          echo "JOB_ID=$JOB_ID" >> $GITHUB_ENV

      - name: Wait for Completion
        run: |
          while true; do
            STATUS=$(curl -s "$SIMPLETUNER_URL/api/cloud/jobs/$JOB_ID" | jq -r '.job.status')
            echo "Job status: $STATUS"

            case $STATUS in
              completed) exit 0 ;;
              failed|cancelled) exit 1 ;;
              *) sleep 60 ;;
            esac
          done
```

### Autenticacao por API key

Para pipelines automatizados, crie API keys em vez de autenticacao por sessao.

**Via UI:** Aba Cloud → Settings → API Keys → Create New Key

**Via API:**

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/auth/api-keys' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_SESSION_TOKEN' \
  -d '{
    "name": "ci-pipeline",
    "expires_days": 90,
    "scoped_permissions": ["job.submit", "job.view.own"]
  }'
```

A chave completa e retornada apenas uma vez. Armazene com seguranca.

**Usando uma API key:**

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer stk_abc123...' \
  -d '{...}'
```

**Permissoes com escopo:**

| Permissao | Descricao |
|------------|-------------|
| `job.submit` | Enviar novos jobs |
| `job.view.own` | Ver seus proprios jobs |
| `job.cancel.own` | Cancelar seus proprios jobs |
| `job.view.all` | Ver todos os jobs (admin) |

## Solucao de problemas

Para problemas especificos do provedor (credenciais, fila, hardware), veja a documentacao do seu provedor:

- [Troubleshooting Replicate](REPLICATE.md#troubleshooting)

### Problemas gerais

**Upload de dados falha**
- Verifique se os caminhos dos datasets existem e sao legiveis
- Verifique espaco em disco disponivel para empacotamento zip
- Procure erros no console do navegador ou na resposta da API

**Webhook nao recebe eventos**
- Garanta que sua instancia local esta acessivel publicamente (tunel ativo)
- Verifique se a URL do webhook esta correta (incluindo https://)
- Verifique a saida do terminal do SimpleTuner para erros de webhook

## Referencia de API

### Endpoints agnosticos ao provedor

| Endpoint | Metodo | Descricao |
|----------|--------|-------------|
| `/api/cloud/jobs` | GET | Listar jobs com filtros opcionais |
| `/api/cloud/jobs/submit` | POST | Enviar um novo job de treinamento |
| `/api/cloud/jobs/sync` | POST | Sincronizar jobs do provedor |
| `/api/cloud/jobs/{id}` | GET | Obter detalhes do job |
| `/api/cloud/jobs/{id}/logs` | GET | Buscar logs do job |
| `/api/cloud/jobs/{id}/cancel` | POST | Cancelar um job em execucao |
| `/api/cloud/jobs/{id}` | DELETE | Deletar um job concluido |
| `/api/metrics` | GET | Obter metricas de job e custo |
| `/api/cloud/metrics/cost-limit` | GET | Obter status do limite de custo atual |
| `/api/cloud/providers/{provider}` | PUT | Atualizar configuracoes do provedor |
| `/api/cloud/storage/{bucket}/{key}` | PUT | Endpoint de upload compativel com S3 |

Para endpoints especificos do provedor, veja:
- [Replicate API Reference](REPLICATE.md#api-reference)

Para detalhes completos de schema, veja a documentacao OpenAPI em `http://localhost:8001/docs`.

## Veja tambem

- [README.md](README.md) – Visao geral de arquitetura e status do provedor
- [REPLICATE.md](REPLICATE.md) – Setup e detalhes do provedor Replicate
- [ENTERPRISE.md](../server/ENTERPRISE.md) – SSO, aprovacoes e governanca
- [Tutorial completo de operacoes em nuvem](OPERATIONS_TUTORIAL.md) – Deploy e monitoramento em producao
- [Tutorial completo de API local](../../api/TUTORIAL.md) – Treinamento local via API
- [Configuracao do Dataloader](../../DATALOADER.md) – Referencia de setup de datasets
