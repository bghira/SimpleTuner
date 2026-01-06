# Sistema de Treinamento na Nuvem

> **Status:** Experimental
>
> **Disponivel em:** Web UI (aba Cloud)

O sistema de treinamento na nuvem do SimpleTuner permite rodar jobs de treino em provedores de GPU na nuvem sem configurar sua propria infraestrutura. O sistema e plugavel, permitindo que multiplos provedores sejam adicionados ao longo do tempo.

## Visao geral

O sistema de treinamento na nuvem oferece:

- **Rastreamento unificado de jobs** - acompanhe jobs locais e na nuvem em um unico lugar
- **Empacotamento automatico de dados** - datasets locais sao empacotados e enviados automaticamente
- **Entrega de resultados** - modelos treinados podem ser enviados para HuggingFace, S3 ou baixados localmente
- **Rastreamento de custos** - monitore gastos e custos por provedor com limites de gasto
- **Snapshots de config** - opcionalmente versiona configs de treino com git

## Conceitos chave

Antes de usar treinamento na nuvem, entenda estas tres coisas:

### 1. O que acontece com seus dados

Quando voce envia um job na nuvem:

1. **Datasets sao empacotados** - datasets locais (`type: "local"`) sao zipados e voce ve um resumo
2. **Upload para o provedor** - o zip vai direto para o provedor na nuvem apos consentimento
3. **Treinamento roda** - o modelo pode precisar ser baixado antes das amostras serem treinadas nas GPUs da nuvem
4. **Dados sao apagados** - apos o treino, os dados enviados sao removidos dos servidores do provedor e seu modelo e entregue

**Notas de seguranca:**
- Seu token de API nunca sai da sua maquina
- Arquivos sensiveis (.env, .git, credenciais) sao excluidos automaticamente
- Voce revisa e consente os uploads antes de cada job

### 2. Como voce recebe modelos treinados

O treinamento produz um modelo que precisa ir para algum lugar. Configure um dos seguintes:

| Destino | Setup | Melhor para |
|-------------|-------|----------|
| **HuggingFace Hub** | Defina env `HF_TOKEN`, habilite na aba Publishing | Compartilhar modelos, acesso facil |
| **Download local** | Defina URL de webhook, exponha servidor via ngrok | Privacidade, fluxos locais |
| **Armazenamento S3** | Configure endpoint na aba Publishing | Acesso em equipe, arquivamento |

Veja [Recebendo modelos treinados](TUTORIAL.md#receiving-trained-models) para setup passo a passo.

### 3. Modelo de custos

O Replicate cobra por segundo de GPU:

| Hardware | VRAM | Custo | LoRA tipico (2000 steps) |
|----------|------|------|---------------------------|
| L40S | 48GB | ~$3.50/hr | $5-15 |

**A cobranca comeca** quando o treino inicia e **para** quando termina ou falha.

**Proteja-se:**
- Defina um limite de gasto nas configuracoes Cloud
- Estimativas de custo sao exibidas antes de cada envio
- Cancele jobs em execucao a qualquer momento (voce paga pelo tempo usado)

Veja [Custos](REPLICATE.md#costs) para precos e limites.

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web UI (Cloud Tab)                       │
├─────────────────────────────────────────────────────────────────┤
│  Job List  │  Metrics/Charts  │  Actions/Config  │  Job Details │
└─────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │   Cloud API Routes  │
                    │   /api/cloud/*      │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
┌────────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
│    JobStore     │   │ Upload Service  │   │    Provider     │
│  (Persistence)  │   │ (Data Packaging)│   │   Clients       │
└─────────────────┘   └─────────────────┘   └─────────────────┘
                                                     │
                             ┌───────────────────────┤
                             │                       │
                   ┌─────────▼─────────┐   ┌─────────▼─────────┐
                   │     Replicate     │   │  SimpleTuner.io   │
                   │    Cog Client     │   │   (Coming Soon)   │
                   └───────────────────┘   └───────────────────┘
```

## Provedores suportados

| Provedor | Status | Recursos |
|----------|--------|----------|
| [Replicate](REPLICATE.md) | Suportado | Rastreamento de custo, logs ao vivo, webhooks |
| [Worker Orchestration](../server/WORKERS.md) | Suportado | Workers distribuidos auto-hospedados, qualquer GPU |
| SimpleTuner.io | Em breve | Servico gerenciado pela equipe SimpleTuner |

### Orquestracao de workers

Para treinamento distribuido auto-hospedado em multiplas maquinas, veja o [Guia de Orquestracao de Workers](../server/WORKERS.md). Workers podem rodar em:

- Servidores de GPU on-premise
- VMs na nuvem (qualquer provedor)
- Spot instances (RunPod, Vast.ai, Lambda Labs)

Workers registram-se no seu orquestrador SimpleTuner e recebem jobs automaticamente.

## Fluxo de dados

### Enviando um job

1. **Preparacao de config** - Seu config de treino e serializado
2. **Empacotamento de dados** - Datasets locais (com `type: "local"`) sao zipados
3. **Upload** - O zip e enviado para o hosting de arquivos do Replicate
4. **Envio** - Job e enviado ao provedor na nuvem
5. **Rastreamento** - O status do job e consultado e atualizado em tempo real

### Recebendo resultados

Resultados podem ser entregues via:

1. **HuggingFace Hub** - Envie o modelo treinado para sua conta HuggingFace
2. **Armazenamento S3 compativel** - Envie para qualquer endpoint S3 (AWS, MinIO, etc.)
3. **Download local** - O SimpleTuner inclui um endpoint S3 compativel embutido que recebe uploads localmente

## Privacidade de dados e consentimento

Quando voce envia um job na nuvem, o SimpleTuner pode fazer upload de:

- **Datasets de treinamento** - Imagens/arquivos de datasets com `type: "local"`
- **Configuracao** - Seus parametros de treino (learning rate, configuracoes do modelo, etc.)
- **Captions/metadata** - Quaisquer arquivos de texto associados aos seus datasets

Os dados sao enviados diretamente ao provedor da nuvem (ex.: hosting de arquivos do Replicate). Eles nao passam pelos servidores do SimpleTuner.

### Configuracoes de consentimento

Na aba Cloud, voce pode configurar o comportamento de upload de dados:

| Configuracao | Comportamento |
|---------|----------|
| **Always Ask** | Mostra um dialogo de confirmacao listando datasets antes de cada upload |
| **Always Allow** | Pula a confirmacao para workflows confiaveis |
| **Never Upload** | Desabilita treinamento na nuvem (apenas local) |

## Endpoint S3 local

O SimpleTuner inclui um endpoint S3 compativel embutido para receber modelos treinados:

```
PUT /api/cloud/storage/{bucket}/{key}
GET /api/cloud/storage/{bucket}/{key}
GET /api/cloud/storage/{bucket}  (list objects)
GET /api/cloud/storage  (list buckets)
```

Arquivos sao armazenados em `~/.simpletuner/cloud_outputs/` por padrao.

Voce pode configurar credenciais manualmente; se nao fizer, credenciais efemeras sao geradas automaticamente para cada job, que e o metodo recomendado.

Isso permite o modo "download only", onde voce:
1. Define uma URL de webhook apontando para sua instancia SimpleTuner local
2. O SimpleTuner auto-configura as configuracoes de publicacao S3
3. Modelos treinados sao enviados de volta para sua maquina

**Nota:** Voce precisara expor sua instancia local SimpleTuner via ngrok, cloudflared ou similar para o provedor na nuvem conseguir alcanca-la.

## Adicionando novos provedores

O sistema cloud foi projetado para extensibilidade. Para adicionar um novo provedor:

1. Crie uma nova classe client implementando `CloudTrainerService`:

```python
from .base import CloudTrainerService, CloudJobInfo, CloudJobStatus

class NewProviderClient(CloudTrainerService):
    @property
    def provider_name(self) -> str:
        return "new_provider"

    @property
    def supports_cost_tracking(self) -> bool:
        return True  # or False

    @property
    def supports_live_logs(self) -> bool:
        return True  # or False

    async def validate_credentials(self) -> Dict[str, Any]:
        # Validate API key and return user info
        ...

    async def list_jobs(self, limit: int = 50) -> List[CloudJobInfo]:
        # List recent jobs from the provider
        ...
```
