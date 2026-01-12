# Progresso de job e API de sync

Este documento cobre os mecanismos do SimpleTuner para monitorar o progresso de jobs de treinamento em nuvem e manter o estado local sincronizado com provedores de nuvem.

## Visao geral

O SimpleTuner fornece varias abordagens para rastrear o status de jobs:

| Metodo | Caso de uso | Latencia | Uso de recursos |
|--------|----------|---------|----------------|
| API de progresso inline | Polling da UI para jobs em execucao | Baixa (5s padrao) | Chamadas de API por job |
| Sync de jobs (pull) | Descobrir jobs do provedor | Media (sob demanda) | Chamada de API em lote |
| Parametro `sync_active` | Atualizar status de jobs ativos | Media (sob demanda) | Chamadas por job ativo |
| Poller em background | Atualizacoes automaticas de status | Configuravel (30s padrao) | Polling continuo |
| Webhooks | Notificacoes push em tempo real | Instantaneo | Sem polling |

## API de progresso inline

### Endpoint

```
GET /api/cloud/jobs/{job_id}/inline-progress
```

### Proposito

Retorna informacoes compactas de progresso para um unico job em execucao, adequado para exibir atualizacoes inline em uma lista de jobs sem buscar logs completos.

### Resposta

```json
{
  "job_id": "abc123",
  "stage": "Training",
  "last_log": "Step 1500/5000 - loss: 0.0234",
  "progress": 30.0
}
```

| Campo | Tipo | Descricao |
|-------|------|-------------|
| `job_id` | string | O identificador do job |
| `stage` | string ou null | Estagio atual de treinamento: `Preprocessing`, `Warmup`, `Training`, `Validation`, `Saving checkpoint` |
| `last_log` | string ou null | Ultima linha de log (truncada a 80 caracteres) |
| `progress` | float ou null | Percentual de progresso (0-100) com base em parsing de step/epoch |

### Deteccao de estagios

A API analisa linhas de log recentes para determinar o estagio de treinamento:

- **Preprocessing**: Detectado quando logs contem "preprocessing" ou "loading"
- **Warmup**: Detectado quando logs contem "warming up" ou "warmup"
- **Training**: Detectado quando logs contem padroes de "step" ou "epoch"
- **Validation**: Detectado quando logs contem "validat"
- **Saving checkpoint**: Detectado quando logs contem "checkpoint"

### Calculo de progresso

O percentual de progresso e extraido de padroes de log como:
- `step 1500/5000` -> 30%
- `epoch 3/10` -> 30%

### Quando usar

Use a API de progresso inline quando:
- Exibir status compacto em cards de lista de jobs
- Fazer polling frequente (a cada 5 segundos) apenas para jobs em execucao
- Voce precisa de transferencia minima de dados por requisicao

<details>
<summary>Exemplo de cliente (JavaScript)</summary>

```javascript
async function fetchInlineProgress() {
    const runningJobs = jobs.filter(j => j.status === 'running');

    for (const job of runningJobs) {
        try {
            const response = await fetch(
                `/api/cloud/jobs/${job.job_id}/inline-progress`
            );
            if (response.ok) {
                const data = await response.json();
                // Update job card with progress info
                job.inline_stage = data.stage;
                job.inline_log = data.last_log;
                job.inline_progress = data.progress;
            }
        } catch (error) {
            // Silently ignore - job may have completed
        }
    }
}

// Poll every 5 seconds
setInterval(fetchInlineProgress, 5000);
```

</details>

## Mecanismos de sync de jobs

O SimpleTuner fornece duas abordagens de sync para manter o estado local atualizado com provedores de nuvem.

### 1. Sync completo do provedor

#### Endpoint

```
POST /api/cloud/jobs/sync
```

#### Proposito

Descobre jobs do provedor de nuvem que podem nao existir no armazenamento local. Isso e util quando:
- Jobs foram enviados fora do SimpleTuner (diretamente pela API do Replicate)
- O armazenamento local de jobs foi resetado ou corrompido
- Voce quer importar jobs historicos

#### Resposta

```json
{
  "synced": 3,
  "message": "Discovered 3 new jobs from Replicate"
}
```

#### Comportamento

1. Busca ate 100 jobs recentes do Replicate
2. Para cada job:
   - Se nao estiver no armazenamento local: cria um novo registro `UnifiedJob`
   - Se ja estiver no armazenamento: atualiza status, custo e timestamps
3. Retorna a contagem de jobs recém-descobertos

<details>
<summary>Exemplo de cliente</summary>

```bash
# Sync jobs from Replicate
curl -X POST http://localhost:8001/api/cloud/jobs/sync

# Response
{"synced": 2, "message": "Discovered 2 new jobs from Replicate"}
```

</details>

#### Botao de sync na Web UI

O dashboard Cloud inclui um botao de sync para descobrir jobs orfaos:

1. Clique no botao **Sync** na barra de ferramentas da lista de jobs
2. O botao mostra um spinner de carregamento durante o sync
3. Em sucesso, um toast mostra: *"Discovered N jobs from Replicate"*
4. A lista de jobs e as metricas do dashboard sao atualizadas automaticamente

**Casos de uso:**
- Descobrir jobs enviados diretamente via API do Replicate ou console web
- Recuperar apos um reset de banco de dados
- Importar jobs de uma conta de equipe compartilhada no Replicate

O botao de sync chama `POST /api/cloud/jobs/sync` internamente e recarrega tanto a lista de jobs quanto as metricas do dashboard.

### 2. Sync de status de jobs ativos (`sync_active`)

#### Endpoint

```
GET /api/cloud/jobs?sync_active=true
```

#### Proposito

Atualiza o status de todos os jobs ativos (nao terminais) antes de retornar a lista. Isso fornece status atualizado sem esperar pelo polling em background.

#### Estados ativos

Jobs nestes estados sao considerados "ativos" e serao sincronizados:
- `pending` - Job enviado mas ainda nao iniciado
- `uploading` - Upload de dados em progresso
- `queued` - Aguardando na fila do provedor
- `running` - Treinamento em progresso

#### Comportamento

1. Antes de listar jobs, busca o status atual de cada job ativo em nuvem
2. Atualiza o armazenamento local com:
   - Status atual
   - Timestamps `started_at` / `completed_at`
   - `cost_usd` (custo acumulado)
   - `error_message` (se falhou)
3. Retorna a lista de jobs atualizada

<details>
<summary>Exemplo de cliente (JavaScript)</summary>

```javascript
// Load jobs with active status sync
async function loadJobs(syncActive = false) {
    const params = new URLSearchParams({
        limit: '50',
        provider: 'replicate',
    });

    if (syncActive) {
        params.set('sync_active', 'true');
    }

    const response = await fetch(`/api/cloud/jobs?${params}`);
    const data = await response.json();
    return data.jobs;
}

// Use sync_active during periodic refresh
setInterval(() => loadJobs(true), 30000);
```

</details>

### Comparacao: sync vs sync_active

| Recurso | `POST /jobs/sync` | `GET /jobs?sync_active=true` |
|---------|-------------------|------------------------------|
| Descobre novos jobs | Sim | Nao |
| Atualiza jobs existentes | Sim | Sim (apenas ativos) |
| Escopo | Todos os jobs do provedor | Apenas jobs locais ativos |
| Caso de uso | Importacao inicial, recuperacao | Atualizacao regular de status |
| Performance | Mais pesado (consulta em lote) | Mais leve (seletivo) |

## Configuracao do poller em background

O poller em background sincroniza automaticamente o status de jobs ativos sem intervencao do cliente.

### Comportamento padrao

- **Auto-habilitado**: Se nenhum webhook estiver configurado, o polling inicia automaticamente
- **Intervalo padrao**: 30 segundos
- **Escopo**: Todos os jobs ativos em nuvem

<details>
<summary>Configuracao enterprise</summary>

Para deploys em producao, configure o polling via `simpletuner-enterprise.yaml`:

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

<details>
<summary>Variaveis de ambiente</summary>

```bash
# Set custom polling interval (in seconds)
export SIMPLETUNER_JOB_POLL_INTERVAL=60
```

</details>

### Como funciona

1. No startup do servidor, o `BackgroundTaskManager` verifica:
   - Se a configuracao enterprise habilita polling explicitamente, use esse intervalo
   - Caso contrario, se nenhum webhook estiver configurado, habilita automaticamente com intervalo de 30s
2. A cada intervalo, o poller:
   - Lista todos os jobs com status ativo
   - Agrupa por provedor
   - Busca status atual de cada provedor
   - Atualiza o armazenamento local
   - Emite eventos SSE para mudancas de status
   - Atualiza entradas de fila para estados terminais

<details>
<summary>Eventos SSE</summary>

Quando o poller detecta mudancas de status, ele transmite eventos SSE:

```javascript
// Subscribe to SSE events
const eventSource = new EventSource('/api/events');

eventSource.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'job_status_changed') {
        console.log(`Job ${data.job_id} is now ${data.status}`);
        // Refresh job list
        loadJobs();
    }
});
```

</details>

<details>
<summary>Acesso programatico</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.background_tasks import (
    get_task_manager,
    start_background_tasks,
    stop_background_tasks,
)

# Get the manager
manager = get_task_manager()

# Check if running
if manager._running:
    print("Background tasks are active")

# Manual start/stop (usually handled by app lifespan)
await start_background_tasks()
await stop_background_tasks()
```

</details>

## Boas praticas

### 1. Escolha a estrategia de sync correta

| Cenario | Abordagem recomendada |
|----------|---------------------|
| Carregamento inicial da pagina | `GET /jobs` sem sync (rapido) |
| Atualizacao periodica (30s) | `GET /jobs?sync_active=true` |
| Usuario clica em "Refresh" | `POST /jobs/sync` para descoberta |
| Detalhes de job em execucao | API de progresso inline (5s) |
| Deploy em producao | Poller em background + webhooks |

### 2. Evite over-polling

<details>
<summary>Exemplo</summary>

```javascript
// Good: Poll inline progress only for running jobs
const runningJobs = jobs.filter(j => j.status === 'running');

// Bad: Poll all jobs regardless of status
for (const job of jobs) { /* ... */ }
```

</details>

### 3. Use SSE para atualizacoes em tempo real

<details>
<summary>Exemplo</summary>

Em vez de polling agressivo, assine eventos SSE:

```javascript
// Combine SSE with conservative polling
const eventSource = new EventSource('/api/events');

eventSource.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'job_status_changed') {
        loadJobs();  // Refresh on status change
    }
});

// Fallback: poll every 30 seconds
setInterval(() => loadJobs(true), 30000);
```

</details>

### 4. Lide com estados terminais

<details>
<summary>Exemplo</summary>

Pare de fazer polling em jobs que atingiram estados terminais:

```javascript
const terminalStates = ['completed', 'failed', 'cancelled'];

function shouldPollJob(job) {
    return !terminalStates.includes(job.status);
}
```

</details>

### 5. Configure webhooks para producao

<details>
<summary>Exemplo</summary>

Webhooks eliminam a necessidade de polling por completo:

```yaml
# In provider config
webhook_url: "https://your-server.com/api/webhooks/replicate"
```

Quando webhooks sao configurados:
- Polling em background e desabilitado (a menos que habilitado explicitamente)
- Atualizacoes de status chegam em tempo real via callbacks do provedor
- Reducao de chamadas de API para o provedor

</details>

## Solucao de problemas

### Jobs nao atualizam

<details>
<summary>Etapas de debugging</summary>

1. Verifique se o poller em background esta rodando:
   ```bash
   # Look for log line on startup
   grep "job status polling" server.log
   ```

2. Verifique conectividade com o provedor:
   ```bash
   curl http://localhost:8001/api/cloud/providers/replicate/validate
   ```

3. Force um sync:
   ```bash
   curl -X POST http://localhost:8001/api/cloud/jobs/sync
   ```

</details>

### Eventos SSE nao recebidos

<details>
<summary>Etapas de debugging</summary>

1. Verifique o limite de conexao SSE (5 por IP por padrao)
2. Verifique se o EventSource esta conectado:
   ```javascript
   eventSource.addEventListener('open', () => {
       console.log('SSE connected');
   });
   ```

</details>

### Alto uso da API do provedor

<details>
<summary>Solucoes</summary>

Se voce estiver batendo em rate limits:
1. Aumente `job_polling_interval` na configuracao enterprise
2. Reduza a frequencia do polling de progresso inline
3. Configure webhooks para eliminar polling

</details>
