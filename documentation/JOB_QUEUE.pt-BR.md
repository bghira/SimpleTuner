# Fila de Jobs

O sistema de fila gerencia o agendamento de jobs, limites de concorrencia e alocacao de GPU tanto para jobs locais quanto na nuvem. Ele habilita recursos como agendamento noturno, gerenciamento de recursos de GPU e uso controlado de recursos.

## Visao geral

Quando voce envia um job de treinamento na nuvem, ele e adicionado a fila e processado com base em:

- **Prioridade** - Jobs com prioridade mais alta rodam primeiro
- **Limites de concorrencia** - Limites globais e por usuario evitam exaustao de recursos
- **FIFO dentro da prioridade** - Jobs no mesmo nivel de prioridade rodam na ordem de envio

## Status da fila

Acesse o painel da fila clicando no **icone de fila** na barra de acoes da aba Cloud. O painel mostra:

| Metrica | Descricao |
|--------|-------------|
| **Queued** | Jobs aguardando para rodar |
| **Running** | Jobs em execucao |
| **Max Concurrent** | Limite global de jobs simultaneos |
| **Avg Wait** | Tempo medio que os jobs passam na fila |

## Niveis de prioridade

Os jobs recebem prioridade com base no nivel do usuario:

| Nivel de usuario | Prioridade | Valor |
|------------|----------|-------|
| Admin | Urgent | 30 |
| Lead | High | 20 |
| Researcher | Normal | 10 |
| Viewer | Low | 0 |

Valores mais altos = maior prioridade = processados primeiro.

### Sobrescrita de prioridade

Leads e admins podem sobrescrever a prioridade de um job em situacoes especificas (ex.: experimentos urgentes).

## Limites de concorrencia

Dois limites controlam quantos jobs podem rodar simultaneamente:

### Limite global (`max_concurrent`)

Numero maximo de jobs rodando entre todos os usuarios. Padrao: **5 jobs**.

### Limite por usuario (`user_max_concurrent`)

Numero maximo de jobs que um unico usuario pode rodar de uma vez. Padrao: **2 jobs**.

Isso impede que um usuario consuma todos os slots disponiveis.

### Atualizando limites

Admins podem atualizar limites via painel da fila ou API.

<details>
<summary>Exemplo</summary>

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{"max_concurrent": 10, "user_max_concurrent": 3}'
```

</details>

## Ciclo de vida do job na fila

1. **Submitted** - Job criado, adicionado a fila com status `pending`
2. **Pending** - Aguardando um slot (limite de concorrencia)
3. **Running** - Treinando ativamente em GPU na nuvem
4. **Completed/Failed** - Estado terminal, removido da fila ativa

## Endpoints da API

### Listar entradas da fila

```
GET /api/queue
```

Parametros:
- `status` - Filtrar por status (pending, running, blocked)
- `limit` - Maximo de entradas a retornar (padrao: 50)
- `include_completed` - Incluir jobs finalizados

### Estatisticas da fila

```
GET /api/queue/stats
```

<details>
<summary>Exemplo de resposta</summary>

```json
{
  "queue_depth": 3,
  "running": 2,
  "max_concurrent": 5,
  "user_max_concurrent": 2,
  "avg_wait_seconds": 45.2,
  "by_status": {"pending": 3, "running": 2},
  "by_user": {"1": 2, "2": 3}
}
```

</details>

### Meu status na fila

```
GET /api/queue/me
```

Retorna a posicao atual do usuario na fila, jobs pendentes e jobs em execucao.

<details>
<summary>Exemplo de resposta</summary>

```json
{
  "pending_count": 2,
  "running_count": 1,
  "blocked_count": 0,
  "best_position": 3,
  "pending_jobs": [...],
  "running_jobs": [...]
}
```

| Campo | Tipo | Descricao |
|-------|------|-------------|
| `pending_count` | int | Numero de jobs aguardando na fila |
| `running_count` | int | Numero de jobs em execucao |
| `blocked_count` | int | Numero de jobs aguardando aprovacao |
| `best_position` | int ou null | Posicao do job pendente de maior prioridade do usuario |
| `pending_jobs` | array | Lista de detalhes de jobs pendentes |
| `running_jobs` | array | Lista de detalhes de jobs em execucao |

O campo `best_position` indica a posicao na fila do melhor job pendente do usuario (maior prioridade ou mais antigo). Isso ajuda a entender quando o proximo job vai iniciar. Um valor `null` significa que o usuario nao tem jobs pendentes.

</details>

### Posicao do job

```
GET /api/queue/position/{job_id}
```

Retorna a posicao na fila para um job especifico.

### Cancelar job enfileirado

```
POST /api/queue/{job_id}/cancel
```

Cancela um job que ainda nao iniciou.

### Aprovar job bloqueado

```
POST /api/queue/{job_id}/approve
```

Somente admin. Aprova um job que requer aprovacao (ex.: excede limite de custo).

### Rejeitar job bloqueado

```
POST /api/queue/{job_id}/reject?reason=<reason>
```

Somente admin. Rejeita um job bloqueado com um motivo.

### Atualizar concorrencia

```
POST /api/queue/concurrency
```

<details>
<summary>Corpo da requisicao</summary>

```json
{
  "max_concurrent": 10,
  "user_max_concurrent": 3
}
```

</details>

### Disparar processamento

```
POST /api/queue/process
```

Somente admin. Dispara manualmente o processamento da fila (normalmente automatico).

### Limpar entradas antigas

```
POST /api/queue/cleanup?days=30
```

Somente admin. Remove entradas concluidas mais antigas que o numero de dias especificado.

**Parametros:**

| Parametro | Tipo | Padrao | Faixa | Descricao |
|-----------|------|---------|-------|-------------|
| `days` | int | 30 | 1-365 | Periodo de retencao em dias |

**Comportamento:**

Deleta entradas da fila que:
- Tenham um status terminal (`completed`, `failed` ou `cancelled`)
- Tenham um timestamp `completed_at` mais antigo que os dias especificados

Jobs ativos (pending, running, blocked) nunca sao removidos pela limpeza.

<details>
<summary>Resposta e exemplos</summary>

**Resposta:**

```json
{
  "success": true,
  "deleted": 42,
  "days": 30
}
```

**Exemplo de uso:**

```bash
# Limpar entradas mais antigas que 7 dias
curl -X POST "http://localhost:8000/api/queue/cleanup?days=7" \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Limpar entradas mais antigas que 90 dias (limpeza trimestral)
curl -X POST "http://localhost:8000/api/queue/cleanup?days=90" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

</details>

## Arquitetura

<details>
<summary>Diagrama do sistema</summary>

```
┌─────────────────────────────────────────────────────────────┐
│                     Job Submission                          │
│              (routes/cloud/jobs.py:submit_job)              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  JobSubmissionService                       │
│              Uploads data, submits to provider              │
│                    Enqueues job                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    QueueScheduler                           │
│   ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│   │ Queue Store │  │   Policy     │  │ Background Task │    │
│   │  (SQLite)   │  │  (Priority)  │  │    (5s loop)    │    │
│   └─────────────┘  └──────────────┘  └─────────────────┘    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   QueueDispatcher                           │
│            Updates job status, syncs with provider          │
└─────────────────────────────────────────────────────────────┘
```

</details>

### Componentes

| Componente | Localizacao | Descricao |
|-----------|----------|-------------|
| `JobRepository` | `storage/job_repository.py` | Persistencia SQLite unificada para jobs e fila |
| `JobRepoQueueAdapter` | `queue/job_repo_adapter.py` | Adapter para compatibilidade com o scheduler |
| `QueueScheduler` | `queue/scheduler.py` | Logica de agendamento |
| `SchedulingPolicy` | `queue/scheduler.py` | Algoritmo de prioridade/justica |
| `QueueDispatcher` | `queue/dispatcher.py` | Gerencia despacho de jobs |
| `QueueEntry` | `queue/models.py` | Modelo de dados de entrada na fila |
| `LocalGPUAllocator` | `services/local_gpu_allocator.py` | Alocacao de GPU para jobs locais |

### Esquema do banco de dados

Entradas de fila e jobs sao armazenadas no banco SQLite unificado (`~/.simpletuner/cloud/jobs.db`).

<details>
<summary>Definicao do esquema</summary>

```sql
CREATE TABLE queue (
    id INTEGER PRIMARY KEY,
    job_id TEXT UNIQUE NOT NULL,
    user_id INTEGER,
    team_id TEXT,
    provider TEXT DEFAULT 'replicate',
    config_name TEXT,
    priority INTEGER DEFAULT 10,
    priority_override INTEGER,
    status TEXT DEFAULT 'pending',
    position INTEGER DEFAULT 0,
    queued_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    estimated_cost REAL DEFAULT 0.0,
    requires_approval INTEGER DEFAULT 0,
    approval_id INTEGER,
    attempt INTEGER DEFAULT 1,
    max_attempts INTEGER DEFAULT 3,
    error_message TEXT,
    metadata TEXT,
    -- GPU allocation (schema v3)
    allocated_gpus TEXT,          -- JSON array of device indices, e.g., "[0,1]"
    job_type TEXT DEFAULT 'cloud', -- "local" or "cloud"
    num_processes INTEGER DEFAULT 1 -- Number of GPUs required
);
```

Migracoes rodam automaticamente na inicializacao.

</details>

## Concorrencia de GPU local

Ao enviar jobs de treinamento local, o sistema de fila rastreia a alocacao de GPU para evitar conflitos de recursos. Jobs sao enfileirados se as GPUs necessarias estiverem indisponiveis.

### Rastreamento de alocacao de GPU

Cada job local especifica:

- **num_processes** - Numero de GPUs necessarias (de `--num_processes`)
- **device_ids** - Indices de GPU preferidos (de `--accelerate_visible_devices`)

O allocator rastreia quais GPUs estao alocadas para jobs em execucao e so inicia novos jobs quando os recursos estao disponiveis.

### Opcoes de CLI

#### Enviando jobs

<details>
<summary>Exemplos</summary>

```bash
# Envie um job, enfileire se GPUs estiverem indisponiveis (padrao)
simpletuner jobs submit my-config

# Rejeite imediatamente se GPUs estiverem indisponiveis
simpletuner jobs submit my-config --no-wait

# Use quaisquer GPUs disponiveis em vez das IDs configuradas
simpletuner jobs submit my-config --any-gpu

# Dry-run para checar disponibilidade de GPU
simpletuner jobs submit my-config --dry-run
```

</details>

#### Listando jobs

<details>
<summary>Exemplos</summary>

```bash
# Listar jobs recentes
simpletuner jobs list

# Listar com campos especificos
simpletuner jobs list -o job_id,status,config_name

# Saida JSON com campos customizados
simpletuner jobs list --format json -o job_id,status

# Acessar campos aninhados usando notacao com ponto
simpletuner jobs list --format json -o job_id,metadata.run_name

# Filtrar por status
simpletuner jobs list --status running
simpletuner jobs list --status queued

# Limitar resultados
simpletuner jobs list -l 10
```

A opcao `-o` (output) suporta notacao com ponto para acessar campos aninhados nos metadados do job. Por exemplo, `metadata.run_name` extrai o campo `run_name` do objeto `metadata` do job.

</details>

### API de status de GPU

O status de alocacao de GPU esta disponivel via o endpoint de status do sistema:

```
GET /api/system/status?include_allocation=true
```

<details>
<summary>Exemplo de resposta</summary>

```json
{
  "timestamp": 1704067200.0,
  "load_avg_5min": 2.5,
  "memory_percent": 45.2,
  "gpus": [...],
  "gpu_inventory": {
    "backend": "cuda",
    "count": 4,
    "capabilities": {...}
  },
  "gpu_allocation": {
    "allocated_gpus": [0, 1],
    "available_gpus": [2, 3],
    "running_local_jobs": 1,
    "devices": [
      {"index": 0, "name": "A100", "memory_gb": 40, "allocated": true, "job_id": "abc123"},
      {"index": 1, "name": "A100", "memory_gb": 40, "allocated": true, "job_id": "abc123"},
      {"index": 2, "name": "A100", "memory_gb": 40, "allocated": false, "job_id": null},
      {"index": 3, "name": "A100", "memory_gb": 40, "allocated": false, "job_id": null}
    ]
  }
}
```

</details>

Estatisticas da fila tambem incluem info de GPU local:

```
GET /api/queue/stats
```

<details>
<summary>Exemplo de resposta</summary>

```json
{
  "queue_depth": 3,
  "running": 2,
  "max_concurrent": 5,
  "local_gpu_max_concurrent": 6,
  "local_job_max_concurrent": 2,
  "local": {
    "running_jobs": 1,
    "pending_jobs": 0,
    "allocated_gpus": [0, 1],
    "available_gpus": [2, 3],
    "total_gpus": 4,
    "max_concurrent_gpus": 6,
    "max_concurrent_jobs": 2
  }
}
```

</details>

### Limites de concorrencia local

Controle quantos jobs locais e GPUs podem ser usados simultaneamente via o endpoint de concorrencia existente:

```
GET /api/queue/concurrency
POST /api/queue/concurrency
```

O endpoint de concorrencia agora aceita limites de GPU local junto com limites de nuvem:

| Campo | Tipo | Descricao |
|-------|------|-------------|
| `max_concurrent` | int | Maximo de jobs na nuvem rodando (padrao: 5) |
| `user_max_concurrent` | int | Maximo de jobs na nuvem por usuario (padrao: 2) |
| `local_gpu_max_concurrent` | int ou null | Maximo de GPUs para jobs locais (null = ilimitado) |
| `local_job_max_concurrent` | int | Maximo de jobs locais simultaneamente (padrao: 1) |

<details>
<summary>Exemplo</summary>

```bash
# Permitir ate 2 jobs locais usando ate 6 GPUs no total
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{"local_gpu_max_concurrent": 6, "local_job_max_concurrent": 2}'
```

</details>

### API de envio de job local

```
POST /api/queue/submit
```

<details>
<summary>Requisicao e resposta</summary>

**Corpo da requisicao:**

```json
{
  "config_name": "my-training-config",
  "no_wait": false,
  "any_gpu": false
}
```

**Resposta:**

```json
{
  "success": true,
  "job_id": "abc123",
  "status": "running",
  "allocated_gpus": [0, 1],
  "queue_position": null
}
```

</details>

Valores de status:

| Status | Descricao |
|--------|-------------|
| `running` | Job iniciado imediatamente com GPUs alocadas |
| `queued` | Job enfileirado, iniciara quando GPUs estiverem disponiveis |
| `rejected` | GPUs indisponiveis e `no_wait` era true |

### Processamento automatico de jobs

Quando um job termina ou falha, suas GPUs sao liberadas e a fila e processada para iniciar jobs pendentes. Isso acontece automaticamente via hooks do lifecycle do process keeper.

**Comportamento de cancelamento**: Quando um job e cancelado, as GPUs sao liberadas, mas jobs pendentes NAO sao iniciados automaticamente. Isso evita condicoes de corrida durante cancelamento em massa (`simpletuner jobs cancel --all`), onde jobs pendentes iniciariam antes de serem cancelados. Use `POST /api/queue/process` ou reinicie o servidor para disparar o processamento manualmente apos o cancelamento.

## Despacho para workers

Jobs podem ser despachados para workers remotos em vez de rodar nas GPUs locais do orquestrador. Veja [Orquestracao de Workers](experimental/server/WORKERS.md) para configuracao completa de workers.

### Alvos de job

Ao enviar um job, especifique onde ele deve rodar:

| Alvo | Comportamento |
|--------|----------|
| `auto` (padrao) | Tenta workers remotos primeiro, volta para GPUs locais |
| `worker` | Despacha apenas para workers remotos; enfileira se nenhum estiver disponivel |
| `local` | Roda apenas nas GPUs locais do orquestrador |

<details>
<summary>Exemplos</summary>

```bash
# CLI
simpletuner jobs submit my-config --target=worker

# API
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{"config_name": "my-config", "target": "worker"}'
```

</details>

### Selecao de workers

Jobs podem especificar requisitos de labels para combinacao de workers:

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-config",
    "target": "worker",
    "worker_labels": {"gpu_type": "a100*", "location": "us-*"}
  }'
```

Labels suportam padroes glob. O scheduler combina jobs com workers com base em:

1. Requisitos de labels (todos devem corresponder)
2. Requisitos de contagem de GPU
3. Disponibilidade do worker (status IDLE)
4. Ordem FIFO dentro dos workers correspondentes

### Comportamento na inicializacao

Ao iniciar o servidor, o sistema de fila processa automaticamente quaisquer jobs locais pendentes. Se GPUs estiverem disponiveis, jobs enfileirados iniciarao imediatamente sem intervencao manual. Isso garante que jobs enviados antes de um restart continuem processando quando o servidor voltar.

A sequencia de inicializacao:
1. Servidor inicializa o allocator de GPU
2. Jobs locais pendentes sao recuperados da fila
3. Para cada job pendente com GPUs disponiveis, o job e iniciado
4. Jobs que nao podem iniciar (GPUs insuficientes) permanecem enfileirados

Nota: Jobs na nuvem sao tratados pelo scheduler de fila da nuvem separado, que tambem retoma na inicializacao.

## Configuracao

Limites de concorrencia da fila sao configurados via API e persistidos no banco da fila.

**Via Web UI:** Cloud tab -> Queue Panel -> Settings

<details>
<summary>Exemplo de configuracao via API</summary>

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{
    "max_concurrent": 5,
    "user_max_concurrent": 2,
    "team_max_concurrent": 10,
    "enable_fair_share": false
  }'
```

</details>

| Configuracao | Padrao | Descricao |
|---------|---------|-------------|
| `max_concurrent` | 5 | Maximo global de jobs rodando |
| `user_max_concurrent` | 2 | Maximo de jobs rodando por usuario |
| `team_max_concurrent` | 10 | Maximo de jobs rodando por time |
| `enable_fair_share` | false | Habilita limites de concorrencia por time |

### Agendamento fair-share

Quando `enable_fair_share: true`, o scheduler considera a afiliacao de time para evitar que um unico time monopolize recursos.

#### Como funciona

O fair-share adiciona uma terceira camada de controle de concorrencia:

| Camada | Limite | Proposito |
|-------|-------|---------|
| Global | `max_concurrent` | Total de jobs entre todos usuarios/times |
| Por usuario | `user_max_concurrent` | Evita que um usuario consuma todos os slots |
| Por time | `team_max_concurrent` | Evita que um time consuma todos os slots |

Quando um job e considerado para despacho:

1. Checa limite global -> pula se estiver na capacidade
2. Checa limite por usuario -> pula se o usuario estiver na capacidade
3. Se fair-share habilitado E o job tem `team_id`:
   - Checa limite por time -> pula se o time estiver na capacidade

Jobs sem `team_id` nao estao sujeitos aos limites de time.

#### Habilitando fair-share

**Via UI:** Cloud tab -> Queue Panel -> Toggle "Fair-Share Scheduling"

<details>
<summary>Exemplo de API</summary>

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{
    "max_concurrent": 10,
    "user_max_concurrent": 3,
    "team_max_concurrent": 5,
    "enable_fair_share": true
  }'
```

</details>

#### Atribuicao de times

Times sao atribuídos a usuarios no painel de admin. Quando um usuario envia um job, o ID do time dele e anexado a entrada da fila. O scheduler rastreia jobs rodando por time e aplica o limite.

<details>
<summary>Exemplo de cenario</summary>

Com `max_concurrent=6`, `user_max_concurrent=2`, `team_max_concurrent=3`:

| Time | Usuarios | Enviados | Rodando | Bloqueados |
|------|-------|-----------|---------|---------|
| Alpha | Alice, Bob | 4 | 3 (no limite do time) | 1 |
| Beta | Carol | 3 | 2 | 1 (esperando slot global) |

- O time Alpha tem 3 rodando (no `team_max_concurrent`)
- Total rodando e 5 (abaixo de `max_concurrent=6`)
- O job da Carol esta bloqueado porque: 5+1=6, no limite global
- O 4o job da Alice esta bloqueado porque: time em 3/3

Isso garante que nenhum time monopolize a fila mesmo se enviar muitos jobs.

</details>

### Prevencao de starvation

Jobs aguardando por mais tempo que `starvation_threshold_minutes` recebem um boost de prioridade para evitar espera indefinida.

## Fluxo de aprovacao

Jobs podem ser marcados como requerendo aprovacao (ex.: quando o custo estimado excede um limite):

1. Job enviado com `requires_approval: true`
2. Job entra em status `blocked`
3. Admin revisa no painel de fila ou via API
4. Admin aprova ou rejeita
5. Se aprovado, o job vai para `pending` e e agendado normalmente

Veja [Guia Enterprise](experimental/server/ENTERPRISE.md) para configuracao de regras de aprovacao.

## Solucao de problemas

### Jobs travados na fila

<details>
<summary>Passos de debug</summary>

Cheque limites de concorrencia:
```bash
curl http://localhost:8000/api/queue/stats
```

Se `running` for igual a `max_concurrent`, os jobs estao aguardando slots.

</details>

### Fila nao processa

<details>
<summary>Passos de debug</summary>

O processador em background roda a cada 5 segundos. Confira os logs do servidor para erros:
```
Queue scheduler started with 5s processing interval
```

Se nao aparecer, o scheduler pode nao ter iniciado.

</details>

### Job desapareceu da fila

<details>
<summary>Passos de debug</summary>

Confira se foi concluido ou falhou:
```bash
curl "http://localhost:8000/api/queue?include_completed=true"
```

</details>

### Jobs locais aparecem como rodando mas nao treinam

<details>
<summary>Passos de debug</summary>

Se `jobs list` mostra jobs locais como "running" mas nenhum treinamento esta acontecendo:

1. Verifique o status de alocacao de GPU:
   ```bash
   simpletuner jobs status --format json
   ```
   Veja o campo `local.allocated_gpus` - ele deve mostrar quais GPUs estao em uso.

2. Se `allocated_gpus` estiver vazio mas o numero de jobs rodando nao for zero, o estado da fila pode estar inconsistente. Reinicie o servidor para disparar a reconciliacao automatica da fila.

3. Verifique os logs do servidor para erros de alocacao de GPU:
   ```
   Failed to allocate GPUs [0] to job <job_id>
   ```

</details>

### Profundidade da fila mostra contagem errada

<details>
<summary>Explicacao</summary>

A profundidade da fila e o numero de jobs rodando sao calculados separadamente para jobs locais e na nuvem:

- **Jobs locais**: Rastreados via `LocalGPUAllocator` com base no estado de alocacao de GPU
- **Jobs na nuvem**: Rastreados via `QueueScheduler` com base no status do provedor

Use `simpletuner jobs status --format json` para ver o detalhamento:
- `local.running_jobs` - Jobs locais rodando
- `local.pending_jobs` - Jobs locais enfileirados aguardando GPUs
- `running` - Total de jobs rodando (fila da nuvem)
- `queue_depth` - Jobs pendentes na nuvem

</details>

## Veja tambem

- [Orquestracao de Workers](experimental/server/WORKERS.md) - Registro de workers distribuidos e despacho de jobs
- [Tutorial de Treinamento na Nuvem](experimental/cloud/TUTORIAL.md) - Comecando com treinamento na nuvem
- [Guia Enterprise](experimental/server/ENTERPRISE.md) - Setup multi-usuario, aprovacoes, governanca
- [Guia de Operacoes](experimental/cloud/OPERATIONS_TUTORIAL.md) - Deploy em producao
