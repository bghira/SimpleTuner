# Logging estruturado e tarefas em background

Este documento cobre o sistema de logging estruturado e os workers de tarefas em background no recurso de treinamento em nuvem do SimpleTuner.

## Sumario

- [Logging estruturado](#logging-estruturado)
  - [Configuracao](#configuracao)
  - [Formato de log JSON](#formato-de-log-json)
  - [LogContext para injeção de campos](#logcontext-para-injeção-de-campos)
  - [IDs de correlacao](#ids-de-correlacao)
- [Tarefas em background](#tarefas-em-background)
  - [Worker de polling de status de jobs](#worker-de-polling-de-status-de-jobs)
  - [Worker de processamento de fila](#worker-de-processamento-de-fila)
  - [Worker de expiracao de aprovacoes](#worker-de-expiracao-de-aprovacoes)
  - [Opcoes de configuracao](#opcoes-de-configuracao)
- [Debugging com logs](#debugging-com-logs)

---

## Logging estruturado

O treinamento em nuvem do SimpleTuner usa um sistema de logging JSON estruturado que fornece saida consistente e parseavel com rastreamento automatico de IDs de correlacao para tracing distribuido.

### Configuracao

Configure o logging via variaveis de ambiente:

```bash
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
export SIMPLETUNER_LOG_LEVEL="INFO"

# Format: "json" (structured) or "text" (traditional)
export SIMPLETUNER_LOG_FORMAT="json"

# Optional: Log to file in addition to stdout
export SIMPLETUNER_LOG_FILE="/var/log/simpletuner/cloud.log"
```

<details>
<summary>Configuracao programatica</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.structured_logging import (
    configure_structured_logging,
    init_from_env,
)

# Configure with explicit options
configure_structured_logging(
    level="INFO",
    json_output=True,
    log_file="/var/log/simpletuner/cloud.log",
    include_stack_info=False,  # Include stack traces for errors
)

# Or initialize from environment variables
init_from_env()
```

</details>

### Formato de log JSON

Quando a saida JSON esta habilitada, cada entrada de log inclui:

<details>
<summary>Exemplo de entrada de log JSON</summary>

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "INFO",
  "logger": "simpletuner.cloud.jobs",
  "message": "Job submitted successfully",
  "correlation_id": "abc123def456",
  "source": {
    "file": "jobs.py",
    "line": 350,
    "function": "submit_job"
  },
  "extra": {
    "job_id": "xyz789",
    "provider": "replicate",
    "cost_estimate": 2.50
  }
}
```

</details>

| Campo | Descricao |
|-------|-------------|
| `timestamp` | Timestamp ISO 8601 em UTC |
| `level` | Nivel de log (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `logger` | Hierarquia do nome do logger |
| `message` | Mensagem de log legivel por humanos |
| `correlation_id` | ID de rastreamento da requisicao (gerado automaticamente ou propagado) |
| `source` | Arquivo, numero de linha e nome da funcao |
| `extra` | Campos estruturados adicionais do LogContext |

### LogContext para injecao de campos

Use `LogContext` para adicionar automaticamente campos estruturados a todos os logs dentro de um escopo:

<details>
<summary>Exemplo de uso do LogContext</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.structured_logging import (
    get_logger,
    LogContext,
)

logger = get_logger("simpletuner.cloud.jobs")

async def process_job(job_id: str, provider: str):
    # All logs within this block include job_id and provider
    with LogContext(job_id=job_id, provider=provider):
        logger.info("Starting job processing")

        # Nested context adds more fields
        with LogContext(step="validation"):
            logger.info("Validating configuration")

        with LogContext(step="submission"):
            logger.info("Submitting to provider")

        logger.info("Job processing complete")
```

Logs de saida incluirao os campos de contexto:

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "INFO",
  "logger": "simpletuner.cloud.jobs",
  "message": "Starting job processing",
  "correlation_id": "abc123",
  "extra": {
    "job_id": "xyz789",
    "provider": "replicate"
  }
}
```

</details>

Campos comuns para injetar:

| Campo | Proposito |
|-------|---------|
| `job_id` | Identificador do job de treinamento |
| `provider` | Provedor de nuvem (replicate, etc.) |
| `user_id` | Usuario autenticado |
| `step` | Fase de processamento (validation, upload, submission) |
| `attempt` | Numero da tentativa de retry |

### IDs de correlacao

IDs de correlacao habilitam o tracing de requisicoes entre servicos. Eles sao:

1. **Auto-gerados** para cada novo thread de requisicao se nao estiver presente
2. **Propagados** via header HTTP `X-Correlation-ID`
3. **Armazenados** em thread-local para injecao automatica nos logs
4. **Incluidos** em requisicoes HTTP de saida para provedores de nuvem

<details>
<summary>Diagrama de fluxo de ID de correlacao</summary>

```
User Request
     |
     v
[X-Correlation-ID: abc123]  <-- Incoming header (or auto-generated)
     |
     v
[Thread-local storage]  <-- set_correlation_id("abc123")
     |
     +---> Log entry: {"correlation_id": "abc123", ...}
     |
     +---> Outbound HTTP: X-Correlation-ID: abc123
           (to Replicate API)
```

</details>

<details>
<summary>Gerenciamento manual de ID de correlacao</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.http_client import (
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
)

# Get current ID (auto-generates if none exists)
current_id = get_correlation_id()

# Set a specific ID (e.g., from incoming request header)
set_correlation_id("request-abc-123")

# Clear when request completes
clear_correlation_id()
```

</details>

</details>

<details>
<summary>ID de correlacao em clientes HTTP</summary>

A factory de cliente HTTP inclui automaticamente o ID de correlacao em requisicoes de saida:

```python
from simpletuner.simpletuner_sdk.server.services.cloud.http_client import (
    get_async_client,
)

# Correlation ID is automatically added to X-Correlation-ID header
async with get_async_client() as client:
    response = await client.get("https://api.replicate.com/v1/predictions")
    # Request includes: X-Correlation-ID: <current-id>
```

</details>

---

## Tarefas em background

O sistema de treinamento em nuvem roda varios workers em background para lidar com operacoes assincronas.

### Gerenciador de tarefas em background

Todas as tarefas em background sao gerenciadas pelo singleton `BackgroundTaskManager`:

<details>
<summary>Uso do gerenciador de tarefas</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.background_tasks import (
    get_task_manager,
    start_background_tasks,
    stop_background_tasks,
)

# Start all configured tasks (typically in app lifespan)
await start_background_tasks()

# Stop gracefully on shutdown
await stop_background_tasks()
```

</details>

### Worker de polling de status de jobs

O worker de polling de jobs sincroniza status de jobs com provedores de nuvem. Isso e util quando webhooks nao estao disponiveis (ex.: atras de um firewall).

**Proposito:**
- Fazer polling de jobs ativos (pending, uploading, queued, running) nos provedores de nuvem
- Atualizar o armazenamento local de jobs com o status atual
- Emitir eventos SSE quando status mudam
- Atualizar entradas de fila para status terminais

<details>
<summary>Diagrama de fluxo de polling</summary>

```
[Every 30 seconds]
     |
     v
List active jobs from local store
     |
     v
Group by provider
     |
     +---> [replicate] --> Get status from API --> Update local job
     |
     v
Emit SSE events for status changes
     |
     v
Update queue on terminal statuses (completed, failed, cancelled)
```

</details>

<details>
<summary>Logica de auto-habilitacao</summary>

O worker de polling inicia automaticamente se nenhum webhook estiver configurado:

```python
# In background_tasks.py
async def _should_auto_enable_polling(self) -> bool:
    config = await store.get_config("replicate")
    return not config.get("webhook_url")  # Enable if no webhook
```

</details>

### Worker de processamento de fila

Lida com o agendamento e dispatch de jobs com base na prioridade da fila e nos limites de concorrencia.

**Proposito:**
- Processar a fila de jobs a cada 5 segundos
- Despachar jobs de acordo com a prioridade
- Respeitar limites de concorrencia por usuario/organizacao
- Lidar com transicoes de estado das entradas da fila

**Intervalo de processamento da fila:** 5 segundos (fixo)

### Worker de expiracao de aprovacoes

Expira e rejeita automaticamente requisicoes de aprovacao pendentes que passaram do prazo.

**Proposito:**
- Verificar requisicoes de aprovacao expiradas a cada 5 minutos
- Auto-rejeitar jobs com aprovacoes expiradas
- Atualizar entradas da fila para estado de falha
- Emitir notificacoes SSE para aprovacoes expiradas

<details>
<summary>Diagrama de fluxo de processamento</summary>

```
[Every 5 minutes]
     |
     v
List pending approval requests
     |
     v
Filter expired requests (past deadline)
     |
     v
Mark approval requests as expired
     |
     +---> Update queue entries to "failed"
     |
     +---> Update job status to "cancelled"
     |
     +---> Emit SSE "approval_expired" events
```

</details>

### Opcoes de configuracao

#### Variavel de ambiente

```bash
# Set custom polling interval (seconds)
export SIMPLETUNER_JOB_POLL_INTERVAL="60"
```

<details>
<summary>Arquivo de configuracao enterprise</summary>

Crie `simpletuner-enterprise.yaml`:

```yaml
background:
  job_status_polling:
    enabled: true
    interval_seconds: 30
  queue_processing:
    enabled: true
    interval_seconds: 5
```

</details>

#### Propriedades de configuracao

| Propriedade | Padrao | Descricao |
|----------|---------|-------------|
| `job_polling_enabled` | false (auto se nao houver webhook) | Habilitar polling explicito |
| `job_polling_interval` | 30 segundos | Intervalo de polling |
| Processamento de fila | Sempre habilitado | Nao pode ser desabilitado |
| Expiracao de aprovacao | Sempre habilitado | Checa a cada 5 minutos |

<details>
<summary>Acessando configuracao programaticamente</summary>

```python
from simpletuner.simpletuner_sdk.server.config.enterprise import get_enterprise_config

config = get_enterprise_config()

if config.job_polling_enabled:
    interval = config.job_polling_interval
    print(f"Polling enabled with {interval}s interval")
```

</details>

---

## Debugging com logs

### Encontrando entradas de log relacionadas

Use o ID de correlacao para rastrear uma requisicao entre todos os componentes:

<details>
<summary>Comandos de filtragem de log</summary>

```bash
# Find all logs for a specific request
grep '"correlation_id": "abc123"' /var/log/simpletuner/cloud.log

# Or with jq for JSON parsing
cat /var/log/simpletuner/cloud.log | jq 'select(.correlation_id == "abc123")'
```

</details>

<details>
<summary>Filtrando por job</summary>

```bash
# Find all logs for a specific job
cat /var/log/simpletuner/cloud.log | \
  jq 'select(.extra.job_id == "xyz789")'
```

</details>

<details>
<summary>Monitorando tarefas em background</summary>

```bash
# Watch polling activity
cat /var/log/simpletuner/cloud.log | \
  jq 'select(.message | contains("polling")) | {timestamp, message}'

# Monitor approval expirations
cat /var/log/simpletuner/cloud.log | \
  jq 'select(.message | contains("expired"))'
```

</details>

### Recomendacoes de nivel de log

| Ambiente | Nivel | Justificativa |
|-------------|-------|-----------|
| Desenvolvimento | DEBUG | Visibilidade total para troubleshooting |
| Staging | INFO | Operacao normal com eventos-chave |
| Producao | INFO ou WARNING | Equilibrio entre visibilidade e volume |

### Mensagens comuns de log

| Mensagem | Nivel | Significado |
|---------|-------|---------|
| "Starting job status polling" | INFO | Worker de polling iniciado |
| "Synced N active jobs" | DEBUG | Ciclo de polling concluido |
| "Queue scheduler started" | INFO | Processamento de fila ativo |
| "Expired N approval requests" | INFO | Aprovacoes auto-rejeitadas |
| "Failed to sync job X" | DEBUG | Sync de job individual falhou (transitorio) |
| "Error in job polling" | ERROR | Loop de polling encontrou erro |

### Integrando com agregadores de logs

O formato de log JSON e compativel com:

- **Elasticsearch/Kibana**: Ingestao direta de logs JSON
- **Splunk**: Parsing de JSON com extracao de campos
- **Datadog**: Pipeline de logs com parsing JSON
- **Loki/Grafana**: Use o parser `json`

<details>
<summary>Exemplo de configuracao Loki/Promtail</summary>

```yaml
scrape_configs:
  - job_name: simpletuner
    static_configs:
      - targets: [localhost]
        labels:
          job: simpletuner
          __path__: /var/log/simpletuner/cloud.log
    pipeline_stages:
      - json:
          expressions:
            level: level
            correlation_id: correlation_id
            job_id: extra.job_id
      - labels:
          level:
          correlation_id:
```

</details>

### Checklist de troubleshooting

1. **Requisicao nao esta sendo rastreada?**
   - Verifique se o header `X-Correlation-ID` esta sendo definido
   - Verifique se o `CorrelationIDFilter` esta anexado aos loggers

2. **Campos de contexto nao aparecem?**
   - Garanta que o codigo esteja dentro de um bloco `LogContext`
   - Verifique se a saida JSON esta habilitada

3. **Polling nao funciona?**
   - Verifique se o URL do webhook esta configurado (desabilita auto-polling)
   - Verifique a configuracao enterprise se estiver usando polling explicito
   - Procure por mensagens de log "Starting job status polling"

4. **Fila nao processa?**
   - Verifique a mensagem "Queue scheduler started"
   - Procure erros em "Failed to start queue processing"
