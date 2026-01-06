# Audit logging

O sistema de audit logging do SimpleTuner fornece um registro evidenciado contra adulteracao de eventos relevantes para seguranca. Todas as acoes administrativas, eventos de autenticacao e operacoes de job sao registradas com verificacao criptografica de cadeia.

## Visao geral

O audit log captura:
- **Eventos de autenticacao**: Tentativas de login (sucesso/falha), logouts, expiracao de sessao
- **Gerenciamento de usuarios**: Criacao, atualizacao, exclusao, mudancas de permissao
- **Operacoes de API keys**: Criacao, revogacao, uso
- **Gerenciamento de credenciais**: Mudancas de credenciais de provedores
- **Operacoes de job**: Envios, cancelamentos, aprovacoes

## Acessando audit logs

### Web UI

Navegue para a aba **Audit** no painel admin para navegar pelas entradas de audit com opcoes de filtro.

### CLI

```bash
# List recent audit entries
simpletuner auth audit list

# Filter by event type
simpletuner auth audit list --event-type auth.login.failed

# Filter by user
simpletuner auth audit user 123

# View security events only
simpletuner auth audit security

# Get statistics
simpletuner auth audit stats

# Verify chain integrity
simpletuner auth audit verify
```

### Endpoints de API

Todos os endpoints exigem a permissao `admin.audit`.

| Metodo | Endpoint | Descricao |
|--------|----------|-------------|
| GET | `/api/audit` | Listar entradas de audit com filtros |
| GET | `/api/audit/stats` | Obter estatisticas de audit |
| GET | `/api/audit/types` | Listar tipos de evento disponiveis |
| GET | `/api/audit/verify` | Verificar integridade da cadeia |
| GET | `/api/audit/user/{user_id}` | Obter entradas de um usuario |
| GET | `/api/audit/security` | Obter eventos relacionados a seguranca |

## Tipos de evento

### Eventos de autenticacao

| Evento | Descricao |
|-------|-------------|
| `auth.login.success` | Login bem-sucedido |
| `auth.login.failed` | Tentativa de login falhou |
| `auth.logout` | Usuario fez logout |
| `auth.session.expired` | Sessao expirada |
| `auth.api_key.used` | API key foi usada |

### Eventos de gerenciamento de usuario

| Evento | Descricao |
|-------|-------------|
| `user.created` | Novo usuario criado |
| `user.updated` | Detalhes do usuario atualizados |
| `user.deleted` | Usuario deletado |
| `user.password.changed` | Usuario alterou a senha |
| `user.level.changed` | Nivel/papel do usuario alterado |
| `user.permission.changed` | Permissao do usuario alterada |

### Eventos de API key

| Evento | Descricao |
|-------|-------------|
| `api_key.created` | Nova API key criada |
| `api_key.revoked` | API key revogada |

### Eventos de credenciais

| Evento | Descricao |
|-------|-------------|
| `credential.created` | Credencial de provedor adicionada |
| `credential.deleted` | Credencial de provedor removida |
| `credential.used` | Credencial foi usada |

### Eventos de job

| Evento | Descricao |
|-------|-------------|
| `job.submitted` | Job enviado para fila |
| `job.cancelled` | Job cancelado |
| `job.approved` | Aprovacao de job concedida |
| `job.rejected` | Aprovacao de job negada |

## Parametros de consulta

Ao listar entradas de audit, voce pode filtrar por:

| Parametro | Tipo | Descricao |
|-----------|------|-------------|
| `event_type` | string | Filtrar por tipo de evento |
| `actor_id` | int | Filtrar por usuario que realizou a acao |
| `target_type` | string | Filtrar por tipo de recurso alvo |
| `target_id` | string | Filtrar por ID do recurso alvo |
| `since` | ISO date | Timestamp inicial |
| `until` | ISO date | Timestamp final |
| `limit` | int | Maximo de entradas (1-500, padrao 50) |
| `offset` | int | Offset de paginacao |

## Integridade da cadeia

Cada entrada de audit inclui:
- Um hash criptografico do seu conteudo
- Uma referencia ao hash da entrada anterior
- Timestamp de um relogio monotonic

Isso cria uma cadeia de hashes que torna adulteracao detectavel. Use o endpoint de verificacao ou o comando CLI para checar a integridade:

```bash
# Verify entire chain
simpletuner auth audit verify

# Verify specific range
simpletuner auth audit verify --start-id 100 --end-id 200
```

A verificacao checa:
1. O hash de cada entrada corresponde ao conteudo
2. Cada entrada referencia corretamente o hash da entrada anterior
3. Nao ha lacunas na sequencia

## Retencao

Os audit logs sao armazenados no banco de dados do SimpleTuner. Configure a retencao no seu deploy:

```bash
# Environment variable for retention period (days)
SIMPLETUNER_AUDIT_RETENTION_DAYS=365
```

Entradas antigas podem ser arquivadas ou removidas conforme requisitos de compliance.

## Consideracoes de seguranca

- Audit logs sao append-only; entradas nao podem ser modificadas ou deletadas pela API
- A permissao `admin.audit` e necessaria para visualizar logs
- Tentativas de login falhas sao registradas com enderecos IP para monitoramento de seguranca
- Considere encaminhar audit logs para um SIEM em deploys de producao
