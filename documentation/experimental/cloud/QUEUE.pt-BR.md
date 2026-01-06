# Sistema de fila

> **Status:** Experimental
> **Disponivel em:** Web UI (aba Cloud)

O sistema de fila gerencia o agendamento de jobs, limites de concorrencia e alocacao justa para jobs de treinamento em nuvem. Ele esta sempre ativo, mesmo em modo de usuario unico, permitindo recursos como agendamento noturno e uso controlado de recursos.

## Visao geral

Quando voce envia um job de treinamento em nuvem, ele e adicionado a fila e processado com base em:

- **Prioridade** - Jobs com prioridade mais alta rodam primeiro
- **Limites de concorrencia** - Limites globais e por usuario evitam exaustao de recursos
- **FIFO dentro da prioridade** - Jobs no mesmo nivel de prioridade rodam na ordem de envio

## Status da fila

Acesse o painel de fila clicando no **icone de fila** na barra de acoes da aba Cloud. O painel mostra:

| Metrica | Descricao |
|--------|-------------|
| **Queued** | Jobs aguardando para rodar |
| **Running** | Jobs em execucao |
| **Max Concurrent** | Limite global de jobs simultaneos |
| **Avg Wait** | Tempo medio que os jobs passam na fila |

## Niveis de prioridade

Jobs recebem prioridade com base no nivel do usuario:

| Nivel do usuario | Prioridade | Valor |
|------------|----------|-------|
| Admin | Urgente | 30 |
| Lead | Alta | 20 |
| Researcher | Normal | 10 |
| Viewer | Baixa | 0 |

Valores mais altos = prioridade maior = processado primeiro.

### Override de prioridade

Leads e admins podem sobrescrever a prioridade de um job para situacoes especificas (ex.: experimentos urgentes).

## Limites de concorrencia

Dois limites controlam quantos jobs podem rodar simultaneamente:

### Limite global (`max_concurrent`)

Maximo de jobs rodando entre todos os usuarios. Padrao: **5 jobs**.

### Limite por usuario (`user_max_concurrent`)

Maximo de jobs que qualquer usuario pode rodar ao mesmo tempo. Padrao: **2 jobs**.

Isso impede que um usuario consuma todos os slots disponiveis.

### Atualizando limites

Admins podem atualizar limites via o painel de fila ou API:

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{"max_concurrent": 10, "user_max_concurrent": 3}'
```

## Ciclo de vida do job na fila

1. **Submitted** - Job criado, adicionado a fila com status `pending`
2. **Pending** - Aguardando um slot (limite de concorrencia)
3. **Running** - Treinamento ativo em GPU na nuvem
4. **Completed/Failed** - Estado terminal, removido da fila ativa

## Endpoints de API

### Listar entradas da fila

```http
GET /api/queue
```

Parametros:
- `status` - Filtrar por status (pending, running, blocked)
- `limit` - Maximo de entradas a retornar (padrao: 50)
- `include_completed` - Incluir jobs finalizados

### Estatisticas da fila

```http
GET /api/queue/stats
```

Retorna:
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

### Meu status na fila

```http
GET /api/queue/me
```

Retorna a posicao atual do usuario na fila, jobs pendentes e jobs em execucao.

**Resposta:**

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
| `running_count` | int | Numero de jobs atualmente em execucao |
| `blocked_count` | int | Numero de jobs aguardando aprovacao |
| `best_position` | int ou null | Posicao do job pendente de maior prioridade do usuario |
| `pending_jobs` | array | Lista de detalhes dos jobs pendentes |
| `running_jobs` | array | Lista de detalhes dos jobs em execucao |

O campo `best_position` indica a posicao na fila do melhor job pendente do usuario (maior prioridade ou enviado mais cedo). Isso ajuda os usuarios a entender quando seu proximo job vai iniciar. Um valor `null` significa que o usuario nao tem jobs pendentes.

### Posicao do job

```http
GET /api/queue/position/{job_id}
```

Retorna a posicao na fila de um job especifico.

### Cancelar job na fila

```http
POST /api/queue/{job_id}/cancel
```

Cancela um job que ainda nao comecou.

### Aprovar job bloqueado

```http
POST /api/queue/{job_id}/approve
```

Apenas admin. Aprova um job que requer aprovacao (ex.: excede limite de custo).

### Rejeitar job bloqueado

```http
POST /api/queue/{job_id}/reject?reason=<reason>
```

Apenas admin. Rejeita um job bloqueado com um motivo.

### Atualizar concorrencia

```http
POST /api/queue/concurrency
```

Body:
```json
{
  "max_concurrent": 10,
  "user_max_concurrent": 3
}
```

### Acionar processamento

```http
POST /api/queue/process
```

Apenas admin. Dispara manualmente o processamento da fila (normalmente automatico).

### Limpar entradas antigas

```http
POST /api/queue/cleanup?days=30
```

Apenas admin. Remove entradas concluídas mais antigas que o numero de dias especificado.
