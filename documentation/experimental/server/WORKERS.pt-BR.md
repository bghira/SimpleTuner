# Orquestracao de workers

A orquestracao de workers do SimpleTuner permite distribuir jobs de treinamento entre varias maquinas com GPU. Workers se registram em um orquestrador central, recebem eventos de despacho em tempo real e reportam status de volta.

## Visao geral

A arquitetura orquestrador/worker permite:

- **Treinamento distribuido** - Rodar jobs em qualquer maquina com GPU, em qualquer lugar
- **Auto-descoberta** - Workers se auto-registram com capacidades de GPU
- **Despacho em tempo real** - Jobs despachados via SSE (Server-Sent Events)
- **Frota mista** - Combine workers efemeros em nuvem com maquinas on-prem persistentes
- **Tolerancia a falhas** - Jobs orfaos sao re-enfileirados automaticamente

## Tipos de worker

| Tipo | Ciclo de vida | Caso de uso |
|------|-----------|----------|
| **Efemero** | Desliga apos concluir o job | Instancias spot em nuvem (RunPod, Vast.ai) |
| **Persistente** | Permanece online entre jobs | GPUs on-prem, instancias reservadas |

## Inicio rapido

### 1. Inicie o orquestrador

Rode o servidor SimpleTuner na sua maquina central:

```bash
simpletuner server --host 0.0.0.0 --port 8001
```

Para producao, habilite SSL:

```bash
simpletuner server --host 0.0.0.0 --port 8001 --ssl
```

### 2. Crie um token de worker

**Via Web UI:** Administration → Workers → Create Worker

**Via API:**

```bash
curl -s -X POST http://localhost:8001/api/admin/workers \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpu-worker-1",
    "worker_type": "persistent",
    "labels": {"location": "datacenter-a", "gpu_type": "a100"}
  }'
```

A resposta inclui o token (exibido apenas uma vez):

```json
{
  "worker_id": "w_abc123",
  "token": "wt_xxxxxxxxxxxx",
  "name": "gpu-worker-1"
}
```

### 3. Inicie o worker

Na maquina com GPU:

```bash
simpletuner worker \
  --orchestrator-url https://orchestrator.example.com:8001 \
  --worker-token wt_xxxxxxxxxxxx \
  --name gpu-worker-1 \
  --persistent
```

Ou via variaveis de ambiente:

```bash
export SIMPLETUNER_ORCHESTRATOR_URL=https://orchestrator.example.com:8001
export SIMPLETUNER_WORKER_TOKEN=wt_xxxxxxxxxxxx
export SIMPLETUNER_WORKER_NAME=gpu-worker-1
export SIMPLETUNER_WORKER_PERSISTENT=true

simpletuner worker
```

O worker vai:

1. Conectar ao orquestrador
2. Reportar capacidades de GPU (auto-detectadas)
3. Entrar no loop de despacho de jobs
4. Enviar heartbeats a cada 30 segundos

### 4. Envie jobs para workers

**Via Web UI:** Configure seu treinamento e clique em **Train in Cloud** → selecione **Worker** como target.

**Via API:**

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H "Content-Type: application/json" \
  -d '{
    "config_name": "my-training-config",
    "target": "worker"
  }'
```

Opcoes de target:

| Target | Comportamento |
|--------|----------|
| `worker` | Despacha apenas para workers remotos |
| `local` | Roda nas GPUs do orquestrador |
| `auto` | Prefere worker se disponivel, fallback para local |

## Referencia de CLI

```
simpletuner worker [OPTIONS]

OPTIONS:
  --orchestrator-url URL   Orchestrator panel URL (or SIMPLETUNER_ORCHESTRATOR_URL)
  --worker-token TOKEN     Authentication token (or SIMPLETUNER_WORKER_TOKEN)
  --name NAME              Worker name (default: hostname)
  --persistent             Stay online between jobs (default: ephemeral)
  -v, --verbose            Enable debug logging
```

### Modo efemero vs persistente

**Efemero (padrao):**
- Worker desliga apos concluir um job
- Ideal para instancias spot em nuvem que cobram por minuto
- Orquestrador limpa workers efemeros offline apos 1 hora

**Persistente (`--persistent`):**
- Worker permanece online aguardando novos jobs
- Reconecta automaticamente se a conexao cair
- Use para GPUs on-prem ou instancias reservadas

## Ciclo de vida do worker

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  CONNECTING │ ──▶ │    IDLE     │ ──▶ │    BUSY     │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  DRAINING   │     │   OFFLINE   │
                    └─────────────┘     └─────────────┘
```

| Status | Descricao |
|--------|-------------|
| `CONNECTING` | Worker estabelecendo conexao |
| `IDLE` | Pronto para receber jobs |
| `BUSY` | Rodando um job no momento |
| `DRAINING` | Finalizando job atual e desligando |
| `OFFLINE` | Desconectado (timeout de heartbeat) |

## Monitoramento de health

O orquestrador monitora a saude dos workers:

- **Intervalo de heartbeat:** 30 segundos (worker → orquestrador)
- **Limite de timeout:** 120 segundos sem heartbeat → marcar offline
- **Loop de health check:** Roda a cada 60 segundos no orquestrador

### Tratamento de falhas

**Worker fica offline durante um job:**

1. Job marcado como falho apos timeout de heartbeat
2. Se houver retries restantes (padrao: 3), job re-enfileirado
3. Proximo worker disponivel pega o job

**Orquestrador reinicia:**

1. Workers reconectam automaticamente
2. Workers reportam quaisquer jobs em andamento
3. Orquestrador reconcilia o estado e retoma

## Matching de GPU

Workers reportam suas capacidades de GPU ao registrar:

```json
{
  "gpu_count": 2,
  "gpu_name": "NVIDIA A100-SXM4-80GB",
  "gpu_vram_gb": 80,
  "accelerator_type": "cuda"
}
```

Jobs podem especificar requisitos de GPU:

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H "Content-Type: application/json" \
  -d '{
    "config_name": "my-config",
    "target": "worker",
    "worker_labels": {"gpu_type": "a100*"}
  }'
```

O scheduler faz matching de jobs para workers com base em:

1. Requisitos de contagem de GPU
2. Matching de labels (suporte a glob)
3. Disponibilidade do worker (status IDLE)

## Labels

Labels fornecem selecao flexivel de workers:

**Atribuir labels na criacao do worker:**

```bash
curl -s -X POST http://localhost:8001/api/admin/workers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "worker-1",
    "labels": {
      "location": "us-west",
      "gpu_type": "a100",
      "team": "nlp"
    }
  }'
```

**Selecionar workers por label:**

```bash
# Match workers with team=nlp
curl -s -X POST http://localhost:8001/api/queue/submit \
  -d '{"config_name": "my-config", "worker_labels": {"team": "nlp"}}'

# Match workers with gpu_type starting with "a100"
curl -s -X POST http://localhost:8001/api/queue/submit \
  -d '{"config_name": "my-config", "worker_labels": {"gpu_type": "a100*"}}'
```

## Operacoes admin

### Listar workers

```bash
curl -s http://localhost:8001/api/admin/workers | jq
```

Resposta:

```json
{
  "workers": [
    {
      "id": "w_abc123",
      "name": "gpu-worker-1",
      "status": "idle",
      "worker_type": "persistent",
      "gpu_count": 2,
      "gpu_name": "A100",
      "labels": {"location": "datacenter-a"},
      "last_heartbeat": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### Drenar um worker

Finalize o job atual de forma graciosa e impeça novos despachos:

```bash
curl -s -X POST http://localhost:8001/api/admin/workers/w_abc123/drain
```

O worker vai:

1. Concluir qualquer job em execucao
2. Entrar em status DRAINING
3. Recusar novos jobs
4. Desconectar apos conclusao do job (efemero) ou permanecer em draining (persistente)

### Rotacionar token

Regenerar o token de autenticacao de um worker:

```bash
curl -s -X POST http://localhost:8001/api/admin/workers/w_abc123/token
```

O token antigo e invalidado imediatamente. Atualize o worker com o novo token.

### Deletar um worker

```bash
curl -s -X DELETE http://localhost:8001/api/admin/workers/w_abc123
```

Funciona apenas se o worker estiver offline.

## Seguranca

### Autenticacao por token

- Workers autenticam via header `X-Worker-Token`
- Tokens sao hashed com SHA-256 antes do armazenamento
- Tokens nunca saem do orquestrador apos a criacao
- Rotacione tokens periodicamente por seguranca

### Seguranca de rede

Para producao:

1. Use a flag `--ssl` ou termine TLS em um reverse proxy
2. Restrinja o registro de workers a redes confiaveis
3. Use regras de firewall para limitar acesso a endpoints `/api/workers/*`

### Audit logging

Todas as acoes de worker sao registradas:

- Tentativas de registro (sucesso/falha)
- Eventos de despacho de job
- Transicoes de status
- Rotacoes de token
- Operacoes admin

Veja [Audit Guide](AUDIT.md) para acesso aos logs.

## Solucao de problemas

### Worker nao conecta

**"Connection refused"**
- Verifique URL e porta do orquestrador
- Verifique regras de firewall permitem conexoes de entrada
- Garanta que o orquestrador esteja rodando com `--host 0.0.0.0`

**"Invalid token"**
- O token pode ter sido rotacionado — solicite um novo
- Verifique espacos em branco na string do token

**"SSL certificate verify failed"**
- Use `--ssl-no-verify` para certs auto-assinados (apenas dev)
- Ou adicione o certificado CA ao trust store do sistema

### Worker fica offline inesperadamente

**Timeout de heartbeat (120s)**
- Verifique estabilidade de rede entre worker e orquestrador
- Procure exaustao de recursos (CPU/memoria) no worker
- Aumente `SIMPLETUNER_HEARTBEAT_TIMEOUT` se a rede for instavel

**Process crash**
- Verifique logs do worker para excecoes Python
- Verifique drivers de GPU (`nvidia-smi`)
- Garanta espaco em disco suficiente para treinamento

### Jobs nao sao despachados para workers

**Sem workers ociosos**
- Verifique status do worker no painel admin
- Garanta que workers estejam conectados e IDLE
- Verifique mismatch de labels entre job e workers

**Requisitos de GPU nao atendidos**
- O job exige mais GPUs do que qualquer worker tem
- Ajuste `--num_processes` no config de treinamento

## Referencia de API

### Endpoints de worker (Worker → Orquestrador)

| Endpoint | Metodo | Descricao |
|----------|--------|-------------|
| `/api/workers/register` | POST | Registrar e reportar capacidades |
| `/api/workers/stream` | GET | Stream SSE para despacho de jobs |
| `/api/workers/heartbeat` | POST | Keepalive periodico |
| `/api/workers/job/{id}/status` | POST | Reportar progresso de job |
| `/api/workers/disconnect` | POST | Notificacao de desligamento gracioso |

### Endpoints admin (requer permissao `admin.workers`)

| Endpoint | Metodo | Descricao |
|----------|--------|-------------|
| `/api/admin/workers` | GET | Listar todos os workers |
| `/api/admin/workers` | POST | Criar token de worker |
| `/api/admin/workers/{id}` | DELETE | Remover worker |
| `/api/admin/workers/{id}/drain` | POST | Drenar worker |
| `/api/admin/workers/{id}/token` | POST | Rotacionar token |

## Veja tambem

- [Enterprise Guide](ENTERPRISE.md) - SSO, cotas, workflows de aprovacao
- [Job Queue](../../JOB_QUEUE.md) - Agendamento de fila e prioridades
- [Cloud Training](../cloud/README.md) - Integracao com provedores de nuvem
- [API Tutorial](../../api/TUTORIAL.md) - Treinamento local via API REST
