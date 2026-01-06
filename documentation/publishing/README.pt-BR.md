# Provedores de publicacao

O SimpleTuner agora pode publicar saidas de treinamento em multiplos destinos via `--publishing_config`. Uploads para o Hugging Face continuam controlados por `--push_to_hub`; `publishing_config` e adicional para outros provedores e roda apos a validacao terminar no processo principal.

## Formatos de configuracao
- Aceita JSON inline (`--publishing_config='[{"provider": "s3", ...}]'`), um dict Python passado pelo SDK, ou um caminho para um arquivo JSON.
- Valores sao normalizados para uma lista, seguindo como `--webhook_config` funciona.
- Cada entrada exige uma chave `provider`. O opcional `base_path` prefixa caminhos no destino remoto. Se seu config nao conseguir retornar uma URI, o provedor registra um aviso unico quando consultado.

## Artefato padrao
A publicacao envia o `output_dir` da execucao (pastas e arquivos) usando o nome base do diretorio. Os metadados incluem o id do job atual e o tipo de validacao para que consumidores downstream possam associar uma URI de volta a execucao.

## Provedores
Instale dependencias opcionais dentro da `.venv` do projeto quando usar um provedor.

### S3 compativel e Backblaze B2 (API S3)
- Provedor: `s3` ou `backblaze_b2`
- Dependencia: `pip install boto3`
- Exemplo:
```json
[
  {
    "provider": "s3",
    "bucket": "simpletuner-models",
    "region": "us-east-1",
    "access_key": "AKIA...",
    "secret_key": "SECRET",
    "base_path": "runs/2024",
    "endpoint_url": "https://s3.us-west-004.backblazeb2.com",
    "public_base_url": "https://cdn.example.com/models"
  }
]
```

AVISO: **Nota de seguranca**: Nunca comite credenciais no controle de versao. Use substituicao por variavel de ambiente ou um gerenciador de segredos em deploys de producao.

### Azure Blob Storage
- Provedor: `azure_blob` (alias `azure`)
- Dependencia: `pip install azure-storage-blob`
- Exemplo:
```json
[
  {
    "provider": "azure_blob",
    "connection_string": "DefaultEndpointsProtocol=....",
    "container": "simpletuner",
    "base_path": "models/latest"
  }
]
```

### Dropbox
- Provedor: `dropbox`
- Dependencia: `pip install dropbox`
- Exemplo:
```json
[
  {
    "provider": "dropbox",
    "token": "sl.12345",
    "base_path": "/SimpleTuner/runs"
  }
]
```
Arquivos grandes fazem streaming em sessoes de upload automaticamente; links compartilhados sao criados quando permitido, caso contrario um caminho `dropbox://` e registrado.

## Uso no CLI
```
simpletuner-train \
  --publishing_config=config/publishing.json \
  --push_to_hub=true \
  ...
```
Se voce chamar o SimpleTuner programaticamente, passe uma lista/dict para `publishing_config` e ele sera normalizado para voce.
