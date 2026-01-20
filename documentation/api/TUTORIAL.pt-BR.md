# Tutorial de treinamento via API

## Introducao

Este guia mostra como rodar jobs de treinamento do SimpleTuner **inteiramente via API HTTP** enquanto mantem o setup e o gerenciamento de datasets na linha de comando. Ele espelha a estrutura dos outros tutoriais, mas pula o onboarding da WebUI. Voce ira:

- instalar e iniciar o servidor unificado
- descobrir e baixar o schema OpenAPI
- criar e atualizar ambientes com chamadas REST
- validar, iniciar e monitorar jobs de treinamento via `/api/training`
- seguir duas configuracoes comprovadas: um fine-tune completo PixArt Sigma 900M e uma execucao Flux Kontext LyCORIS LoRA

## Pre-requisitos

- Python 3.10–3.13, Git e `pip`
- SimpleTuner instalado em um virtualenv (`pip install 'simpletuner[cuda]'` ou a variante que corresponda a sua plataforma)
  - CUDA 13 / Blackwell users (NVIDIA B-series GPUs): `pip install 'simpletuner[cuda13]'`
- Acesso aos repositorios Hugging Face necessarios (`huggingface-cli login` antes de baixar modelos gated)
- Datasets preparados localmente com captions (arquivos de caption para PixArt, pastas edit/reference pareadas para Kontext)
- Um shell com `curl` e `jq`

## Inicie o servidor

A partir do seu checkout do SimpleTuner (ou do ambiente onde o pacote esta instalado):

```bash
simpletuner server --port 8001
```

A API fica em `http://localhost:8001`. Deixe o servidor rodando enquanto executa os comandos seguintes em outro terminal.

> **Dica:** Se voce ja tiver um ambiente de configuracao pronto para treinar, pode iniciar o servidor com `--env` para iniciar o treinamento automaticamente assim que o servidor estiver carregado:
>
> ```bash
> simpletuner server --port 8001 --env my-training-config
> ```
>
> Isso valida sua configuracao no startup e inicia o treinamento imediatamente apos o servidor ficar pronto — util para deploys nao assistidos ou scriptados. A opcao `--env` funciona de forma identica a `simpletuner train --env`.

### Configuracao e deploy

Para uso em producao, voce pode configurar o endereco de bind e a porta:

| Opcao | Variavel de ambiente | Padrao | Descricao |
|--------|---------------------|---------|-------------|
| `--host` | `SIMPLETUNER_HOST` | `0.0.0.0` | Endereco para bind do servidor (use `127.0.0.1` atras de reverse proxy) |
| `--port` | `SIMPLETUNER_PORT` | `8001` | Porta para bind do servidor |

<details>
<summary><b>Opcoes de deploy em producao (TLS, Reverse Proxy, Systemd, Docker)</b></summary>

Para deploys em producao, recomenda-se usar um reverse proxy para terminacao TLS.

#### Configuracao Nginx

```nginx
server {
    listen 443 ssl http2;
    server_name training.example.com;

    # TLS configuration (example using Let's Encrypt paths)
    ssl_certificate /etc/letsencrypt/live/training.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/training.example.com/privkey.pem;

    # WebSocket support for SSE streaming (Critical for real-time logs)
    location /api/training/stream {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        # SSE-specific settings
        proxy_buffering off;
        proxy_read_timeout 86400s;
    }

    # Main application
    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        # Large file uploads for datasets
        client_max_body_size 10G;
        proxy_request_buffering off;
    }
}
```

#### Configuracao Caddy

```caddyfile
training.example.com {
    reverse_proxy 127.0.0.1:8001 {
        # SSE streaming support
        flush_interval -1
    }
    # Large file uploads
    request_body {
        max_size 10GB
    }
}
```

#### Servico systemd

```ini
[Unit]
Description=SimpleTuner Training Server
After=network.target

[Service]
Type=simple
User=trainer
WorkingDirectory=/home/trainer/simpletuner-workspace
Environment="SIMPLETUNER_HOST=127.0.0.1"
Environment="SIMPLETUNER_PORT=8001"
ExecStart=/home/trainer/simpletuner-workspace/.venv/bin/simpletuner server
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

#### Docker Compose com Traefik

```yaml
version: '3.8'
services:
  simpletuner:
    image: ghcr.io/bghira/simpletuner:latest
    command: simpletuner server --host 0.0.0.0 --port 8001
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.simpletuner.rule=Host(`training.example.com`)"
      - "traefik.http.services.simpletuner.loadbalancer.server.port=8001"
```
</details>

## Autenticacao

O SimpleTuner suporta autenticacao multi-usuario. Na primeira inicializacao, voce precisa criar uma conta admin.

### Setup inicial

Verifique se o setup e necessario:

```bash
curl -s http://localhost:8001/api/cloud/auth/setup/status | jq
```

Se `needs_setup` for `true`, crie o primeiro admin:

```bash
curl -s -X POST http://localhost:8001/api/cloud/auth/setup/first-admin \
  -H 'Content-Type: application/json' \
  -d '{
    "email": "admin@example.com",
    "username": "admin",
    "password": "your-secure-password"
  }'
```

### API keys

Para acesso via scripts, gere uma API key apos logar:

```bash
# Login first (stores session cookie)
curl -s -X POST http://localhost:8001/api/cloud/auth/login \
  -H 'Content-Type: application/json' \
  -c cookies.txt \
  -d '{"username": "admin", "password": "your-secure-password"}'

# Create an API key
curl -s -X POST http://localhost:8001/api/cloud/auth/api-keys \
  -H 'Content-Type: application/json' \
  -b cookies.txt \
  -d '{"name": "automation-key"}' | jq
```

Use a chave retornada (prefixo `st_`) nas proximas requisicoes:

```bash
curl -s http://localhost:8001/api/training/status \
  -H 'X-API-Key: st_your_key_here'
```

### Gerenciamento de usuarios

Admins podem criar usuarios adicionais via API ou na WebUI (pagina **Manage Users**):

```bash
# Create a new user (requires admin session)
curl -s -X POST http://localhost:8001/api/users \
  -H 'Content-Type: application/json' \
  -b cookies.txt \
  -d '{
    "email": "researcher@example.com",
    "username": "researcher",
    "password": "their-password",
    "level_names": ["researcher"]
  }'
```

> **Nota:** O registro publico vem desabilitado por padrao. Admins podem habilitar na aba **Manage Users → Registration**, mas e recomendado manter desabilitado para deploys privados.

## Descubra a API

O FastAPI serve docs interativas e o schema OpenAPI:

```bash
# FastAPI Swagger UI
python -m webbrowser http://localhost:8001/docs

# ReDoc view
python -m webbrowser http://localhost:8001/redoc

# Download the schema for local inspection
curl -o openapi.json http://localhost:8001/openapi.json
jq '.info' openapi.json
```

Cada endpoint usado neste tutorial esta documentado la sob as tags `configurations` e `training`.

## Caminho rapido: executar sem ambientes

Se voce prefere **pular o gerenciamento de config/ambiente**, pode enviar um treino pontual postando o payload completo estilo CLI diretamente nos endpoints de treinamento:

1. Crie ou reutilize um JSON de dataloader que descreva seu dataset. O trainer so precisa do caminho referenciado por `--data_backend_config`.

   ```bash
   cat <<'JSON' > config/multidatabackend-once.json
   [
     {
       "id": "demo-images",
       "type": "local",
       "dataset_type": "image",
       "instance_data_dir": "/data/datasets/demo",
       "caption_strategy": "textfile",
       "resolution": 1024,
       "resolution_type": "pixel_area"
     },
     {
       "id": "demo-text-embeds",
       "type": "local",
       "dataset_type": "text_embeds",
       "default": true,
       "cache_dir": "/data/cache/text/demo"
     }
   ]
   JSON
   ```

2. Valide a configuracao inline. Forneca todos os argumentos CLI obrigatorios (`--model_family`, `--model_type`, `--pretrained_model_name_or_path`, `--output_dir`, `--data_backend_config` e um entre `--num_train_epochs` ou `--max_train_steps`):

   ```bash
   curl -s -X POST http://localhost:8001/api/training/validate \
     -F __active_tab__=model \
     -F --model_family=pixart_sigma \
     -F --model_type=full \
     -F --model_flavour=900M-1024-v0.6 \
     -F --pretrained_model_name_or_path=terminusresearch/pixart-900m-1024-ft-v0.6 \
     -F --output_dir=/workspace/output/inline-demo \
     -F --data_backend_config=config/multidatabackend-once.json \
     -F --train_batch_size=1 \
     -F --learning_rate=0.0001 \
     -F --max_train_steps=200 \
     -F --num_train_epochs=0
   ```

   Um snippet verde “Configuration Valid” confirma que o trainer aceitou o payload.

3. Inicie o treinamento com os **mesmos** campos de formulario (voce pode adicionar overrides como `--seed` ou `--validation_prompt`):

   ```bash
   curl -s -X POST http://localhost:8001/api/training/start \
     -F __active_tab__=model \
     -F --model_family=pixart_sigma \
     -F --model_type=full \
     -F --model_flavour=900M-1024-v0.6 \
     -F --pretrained_model_name_or_path=terminusresearch/pixart-900m-1024-ft-v0.6 \
     -F --output_dir=/workspace/output/inline-demo \
     -F --data_backend_config=config/multidatabackend-once.json \
     -F --train_batch_size=1 \
     -F --learning_rate=0.0001 \
     -F --max_train_steps=200 \
     -F --num_train_epochs=0 \
     -F --validation_prompt='test shot of <token>'
   ```

O servidor mescla automaticamente as configuracoes enviadas com seus defaults, grava a config resolvida no arquivo ativo e inicia o treinamento. Voce pode reutilizar o mesmo approach para qualquer familia de modelo — as secaoes restantes cobrem um fluxo completo quando voce quer ambientes reutilizaveis.

### Monitorando execucoes ad-hoc

Voce pode acompanhar progresso com os mesmos endpoints de status usados mais adiante no guia:

- Faça polling em `GET /api/training/status` para estado de alto nivel, ID do job ativo e info de stage de startup.
- Busque logs incrementais com `GET /api/training/events?since_index=N` ou faça streaming via WebSocket em `/api/training/events/stream`.

Para updates push, forneca configuracoes de webhook junto aos campos do formulario:

```bash
curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=model \
  -F --model_family=pixart_sigma \
  ... \
  -F --webhook_config='[{"webhook_type":"raw","callback_url":"https://example.com/simpletuner","log_level":"info","ssl_no_verify":false}]' \
  -F --webhook_reporting_interval=10
```

O payload deve ser serializado como JSON em string; o servidor envia atualizacoes do ciclo de vida do job para o `callback_url`. Veja a descricao `--webhook_config` em `documentation/OPTIONS.md` ou o template `config/webhooks.json` para campos suportados.

<details>
<summary><b>Configuracao de webhook para reverse proxies</b></summary>

Ao usar reverse proxy com HTTPS, sua URL de webhook deve ser o endpoint publico:

1.  **Servidor publico:** Use `https://training.example.com/webhook/callback`
2.  **Tunel:** Use ngrok ou cloudflared para dev local.

**Troubleshooting de logs em tempo real (SSE):**
Se `GET /api/training/events` funciona mas o stream trava:
*   **Nginx:** Garanta `proxy_buffering off;` e `proxy_read_timeout` alto (ex.: 86400s).
*   **CloudFlare:** Encerra conexoes longas; use CloudFlare Tunnel ou bypass do proxy para o endpoint de stream.
</details>

### Disparar validacao manual

Se voce quiser forcar uma avaliacao **entre** intervalos de validacao, chame o endpoint:

```bash
curl -s -X POST http://localhost:8001/api/training/validation/run
```

- O servidor responde com o `job_id` ativo.
- O trainer enfileira uma validacao que dispara logo apos a proxima sincronizacao de gradiente (nao interrompe o micro-batch atual).
- A execucao reutiliza seus prompts/configs de validacao para que as imagens aparecam nos streams/logs usuais.
- Para descarregar validacao para um executavel externo em vez do pipeline embutido, defina `--validation_method=external-script` no seu config (ou payload) e aponte `--validation_external_script` para seu script. Voce pode passar contexto de treinamento com placeholders: `{local_checkpoint_path}`, `{global_step}`, `{tracker_run_name}`, `{tracker_project_name}`, `{model_family}`, `{huggingface_path}`, `{remote_checkpoint_path}` (vazio para validacao), mais quaisquer valores `validation_*` (ex.: `validation_num_inference_steps`, `validation_guidance`, `validation_noise_scheduler`). Habilite `--validation_external_background` se quiser que o script rode em fire-and-forget sem bloquear o treinamento.
- Quer disparar automacao imediatamente apos cada checkpoint ser escrito localmente (mesmo enquanto uploads rodam em background)? Configure `--post_checkpoint_script='/opt/hooks/run_eval.sh {local_checkpoint_path} {global_step}'`. Ele usa os mesmos placeholders das hooks de validacao; `{remote_checkpoint_path}` resolve para vazio nesse hook.
- Prefere manter os uploads embutidos do SimpleTuner e passar a URL remota para sua propria ferramenta? Configure `--post_upload_script` em vez disso; ele dispara uma vez por provedor de publicacao/upload para Hugging Face Hub com `{remote_checkpoint_path}` (se fornecido pelo backend) e os mesmos placeholders de contexto. O SimpleTuner nao ingere resultados do seu script, entao registre artefatos/metricas no seu tracker.
  - Exemplo: `--post_upload_script='/opt/hooks/notify.sh {remote_checkpoint_path} {tracker_project_name} {tracker_run_name}'` onde `notify.sh` chama sua API de tracker.
  - Exemplos de trabalho:
    - `simpletuner/examples/external-validation/replicate_post_upload.py` dispara uma inferencia Replicate usando `{remote_checkpoint_path}`, `{model_family}`, `{model_type}`, `{lora_type}` e `{huggingface_path}`.
    - `simpletuner/examples/external-validation/wavespeed_post_upload.py` dispara inferencia WaveSpeed e faz polling de conclusao usando os mesmos placeholders.
    - `simpletuner/examples/external-validation/fal_post_upload.py` dispara inferencia Flux LoRA no fal.ai (requer `FAL_KEY` e `model_family` contendo `flux`).
    - `simpletuner/examples/external-validation/use_second_gpu.py` roda inferencia Flux LoRA em outra GPU sem exigir uploads.

Se nenhum job estiver ativo o endpoint retorna HTTP 400, entao verifique `/api/training/status` antes ao fazer retries por script.

### Disparar checkpoint manual

Para persistir o estado atual do modelo imediatamente (sem esperar o proximo checkpoint agendado), use:

```bash
curl -s -X POST http://localhost:8001/api/training/checkpoint/run
```

- O servidor responde com o `job_id` ativo.
- O trainer salva um checkpoint apos a proxima sincronizacao de gradiente usando as mesmas configuracoes de checkpoints agendados (regras de upload, retencao, etc.).
- Limpeza de rolling e notificacoes de webhook se comportam exatamente como um checkpoint agendado.

Assim como na validacao, o endpoint retorna HTTP 400 se nenhum job estiver rodando.

### Stream de previews de validacao

Modelos que expõem hooks Tiny AutoEncoder (ou equivalente) podem emitir **previews de validacao por passo** enquanto uma imagem/video ainda esta sendo amostrada. Habilite o recurso adicionando as flags CLI ao seu payload:

```bash
curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=validation \
  -F --validation_preview=true \
  -F --validation_preview_steps=4 \
  -F --validation_num_inference_steps=20 \
  …other fields…
```

- `--validation_preview` (padrao `false`) destrava o decoder de preview.
- `--validation_preview_steps` determina com que frequencia emitir frames intermediarios. Com o exemplo acima, voce recebe eventos nos passos 1,5,9,13,17,20 (o primeiro passo sempre e emitido, depois a cada 4 passos).

Cada preview e publicado como evento `validation.image` (veja `simpletuner/helpers/training/validation.py:899-929`). Voce pode consumi-los via webhooks raw, `GET /api/training/events` ou stream SSE em `/api/training/events/stream`. Um payload tipico se parece com:

```json
{
  "type": "validation.image",
  "title": "Validation (step 5/20): night bench",
  "body": "night bench shot of <token>",
  "data": {
    "step": 5,
    "timestep": 563.0,
    "resolution": [1024, 1024],
    "validation_type": "intermediary",
    "prompt": "night bench shot of <token>",
    "step_label": "5/20"
  },
  "images": [
    {"src": "data:image/png;base64,...", "mime_type": "image/png"}
  ]
}
```

Modelos com suporte a video anexam um array `videos` (data URIs GIF com `mime_type: image/gif`). Como esses eventos chegam quase em tempo real, voce pode exibi-los diretamente em dashboards ou enviá-los para Slack/Discord via webhook raw.

## Fluxo comum de API

1. **Criar um ambiente** – `POST /api/configs/environments`
2. **Popular o arquivo de dataloader** – edite `config/<env>/multidatabackend.json`
3. **Atualizar hiperparametros** – `PUT /api/configs/<env>`
4. **Ativar o ambiente** – `POST /api/configs/<env>/activate`
5. **Validar parametros de treinamento** – `POST /api/training/validate`
6. **Iniciar treinamento** – `POST /api/training/start`
7. **Monitorar ou parar o job** – `/api/training/status`, `/api/training/events`, `/api/training/stop`, `/api/training/cancel`

Cada exemplo abaixo segue esse fluxo.

## Opcional: upload de datasets via API (backends locais)

Se o dataset ainda nao estiver na maquina onde o SimpleTuner roda, voce pode envia-lo via HTTP antes de configurar o dataloader. Os endpoints de upload respeitam o `datasets_dir` configurado (definido durante o onboarding da WebUI) e sao voltados para filesystems locais:

1. **Crie uma pasta alvo** no root de datasets:

   ```bash
   DATASETS_DIR=${DATASETS_DIR:-/workspace/simpletuner/datasets}
   curl -s -X POST http://localhost:8001/api/datasets/folders \
     -F parent_path="$DATASETS_DIR" \
     -F folder_name="pixart-upload"
   ```

2. **Envie arquivos ou um ZIP** (imagens + metadados `.txt/.jsonl/.csv` sao aceitos):

   ```bash
   # Upload a zip (automatically extracted on the server)
   curl -s -X POST http://localhost:8001/api/datasets/upload/zip \
     -F target_path="$DATASETS_DIR/pixart-upload" \
     -F file=@/path/to/dataset.zip

   # Or upload individual files
   curl -s -X POST http://localhost:8001/api/datasets/upload \
     -F target_path="$DATASETS_DIR/pixart-upload" \
     -F files[]=@image001.png \
     -F files[]=@image001.txt
   ```

> **Troubleshooting de uploads:** Se uploads grandes falharem com "Entity Too Large" ao usar reverse proxy, aumente o limite de body (ex.: `client_max_body_size 10G;` no Nginx ou `request_body { max_size 10GB }` no Caddy).

Apos o upload terminar, aponte a entrada do `multidatabackend.json` para a nova pasta (por exemplo, `"/data/datasets/pixart-upload"`).

## Exemplo: PixArt Sigma 900M full fine-tune

### 1. Crie o ambiente via REST

```bash
curl -s -X POST http://localhost:8001/api/configs/environments \
  -H 'Content-Type: application/json' \
  -d
```
```json
{
        "name": "pixart-api-demo",
        "model_family": "pixart_sigma",
        "model_flavour": "900M-1024-v0.6",
        "model_type": "full",
        "description": "PixArt 900M API-driven training"
      }
```

Isso cria `config/pixart-api-demo/` e um `multidatabackend.json` inicial.

### 2. Conecte o dataset

Edite o arquivo de dataloader (substitua caminhos pelos seus locais reais de dataset/cache):

```bash
cat <<'JSON' > config/pixart-api-demo/multidatabackend.json
[
  {
    "id": "pixart-camera",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "/data/datasets/pseudo-camera-10k",
    "caption_strategy": "filename",
    "resolution": 1.0,
    "resolution_type": "area",
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "cache_dir_vae": "/data/cache/vae/pixart/pseudo-camera-10k",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square",
    "metadata_backend": "discovery"
  },
  {
    "id": "pixart-text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "/data/cache/text/pixart/pseudo-camera-10k",
    "write_batch_size": 128
  }
]
JSON
```

### 3. Atualize hiperparametros via API

Pegue a config atual, mescle overrides e envie de volta:

```bash
curl -s http://localhost:8001/api/configs/pixart-api-demo \
  | jq '.config + {
      "--output_dir": "/workspace/output/pixart900m",
      "--train_batch_size": 2,
      "--gradient_accumulation_steps": 2,
      "--learning_rate": 0.0001,
      "--optimizer": "adamw_bf16",
      "--lr_scheduler": "cosine",
      "--lr_warmup_steps": 500,
      "--max_train_steps": 1800,
      "--num_train_epochs": 0,
      "--validation_prompt": "a studio portrait of <token> wearing a leather jacket",
      "--validation_guidance": 3.8,
      "--validation_resolution": "1024x1024",
      "--validation_num_inference_steps": 28,
      "--cache_dir_vae": "/data/cache/vae/pixart",
      "--seed": 1337,
      "--resume_from_checkpoint": "latest",
      "--base_model_precision": "bf16",
      "--dataloader_prefetch": true,
      "--report_to": "none",
      "--checkpoints_total_limit": 4,
      "--validation_seed": 12345,
      "--data_backend_config": "pixart-api-demo/multidatabackend.json"
    }' > /tmp/pixart-config.json

jq '{
      "name": "pixart-api-demo",
      "description": "PixArt 900M full tune (API)",
      "tags": ["pixart", "api"],
      "config": .
    }' /tmp/pixart-config.json > /tmp/pixart-update.json

curl -s -X PUT http://localhost:8001/api/configs/pixart-api-demo \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/pixart-update.json
```

### 4. Ative o ambiente

```bash
curl -s -X POST http://localhost:8001/api/configs/pixart-api-demo/activate
```

### 5. Valide antes de iniciar

`validate` consome dados form-encoded. No minimo, garanta que um entre `num_train_epochs` ou `max_train_steps` seja 0:

```bash
curl -s -X POST http://localhost:8001/api/training/validate \
  -F __active_tab__=model \
  -F --num_train_epochs=0
```

Um bloco de sucesso (`Configuration Valid`) significa que o trainer aceita a configuracao mesclada.

### 6. Inicie o treinamento

```bash
curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=model \
  -F --num_train_epochs=0
```

A resposta inclui o job ID. O treinamento roda com os parametros salvos no passo 3.

### 7. Monitorar e parar

```bash
# Query coarse status
curl -s http://localhost:8001/api/training/status | jq

# Stream incremental log events
curl -s 'http://localhost:8001/api/training/events?since_index=0' | jq

# Cancel or stop
curl -s -X POST http://localhost:8001/api/training/stop
curl -s -X POST http://localhost:8001/api/training/cancel -F job_id=<JOB_ID>
```

Notas do PixArt:

- Mantenha o dataset grande o suficiente para o `train_batch_size * gradient_accumulation_steps`
- Defina `HF_ENDPOINT` se precisar de mirror, e autentique antes de baixar `terminusresearch/pixart-900m-1024-ft-v0.6`
- Ajuste `--validation_guidance` entre 3.6 e 4.4 dependendo dos seus prompts

## Exemplo: Flux Kontext LyCORIS LoRA

Kontext compartilha grande parte do pipeline com Flux Dev, mas exige imagens edit/reference pareadas.

### 1. Provisione o ambiente

```bash
curl -s -X POST http://localhost:8001/api/configs/environments \
  -H 'Content-Type: application/json' \
  -d
```
```json
{
        "name": "kontext-api-demo",
        "model_family": "flux",
        "model_flavour": "kontext",
        "model_type": "lora",
        "lora_type": "lycoris",
        "description": "Flux Kontext LoRA via API"
      }
```

### 2. Descreva o dataloader pareado

Kontext precisa de pares edit/reference mais um cache de text-embed:

```bash
cat <<'JSON' > config/kontext-api-demo/multidatabackend.json
[
  {
    "id": "kontext-edit",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "/data/datasets/kontext/edit",
    "conditioning_data": ["kontext-reference"],
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "caption_strategy": "textfile",
    "minimum_image_size": 768,
    "maximum_image_size": 1536,
    "target_downsample_size": 1024,
    "cache_dir_vae": "/data/cache/vae/kontext/edit",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square"
  },
  {
    "id": "kontext-reference",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/data/datasets/kontext/reference",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "/data/cache/vae/kontext/reference"
  },
  {
    "id": "kontext-text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "/data/cache/text/kontext"
  }
]
JSON
```

Garanta que os nomes de arquivo correspondam entre as pastas edit e reference; o SimpleTuner junta embeddings por nome.

### 3. Aplique hiperparametros especificos do Kontext

```bash
curl -s http://localhost:8001/api/configs/kontext-api-demo \
  | jq '.config + {
      "--output_dir": "/workspace/output/kontext",
      "--train_batch_size": 1,
      "--gradient_accumulation_steps": 4,
      "--learning_rate": 0.00001,
      "--optimizer": "optimi-lion",
      "--lr_scheduler": "cosine",
      "--lr_warmup_steps": 200,
      "--max_train_steps": 12000,
      "--num_train_epochs": 0,
      "--validation_prompt": "a cinematic 1024px product photo of <token>",
      "--validation_guidance": 2.5,
      "--validation_resolution": "1024x1024",
      "--validation_num_inference_steps": 30,
      "--cache_dir_vae": "/data/cache/vae/kontext",
      "--seed": 777,
      "--resume_from_checkpoint": "latest",
      "--base_model_precision": "int8-quanto",
      "--dataloader_prefetch": true,
      "--report_to": "wandb",
      "--lora_rank": 16,
      "--lora_dropout": 0.05,
      "--conditioning_multidataset_sampling": "combined",
      "--clip_skip": 2,
      "--data_backend_config": "kontext-api-demo/multidatabackend.json"
    }' > /tmp/kontext-config.json

jq '{
      "name": "kontext-api-demo",
      "description": "Flux Kontext LyCORIS (API)",
      "tags": ["flux", "kontext", "api"],
      "config": .
    }' /tmp/kontext-config.json > /tmp/kontext-update.json

curl -s -X PUT http://localhost:8001/api/configs/kontext-api-demo \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/kontext-update.json
```

### 4. Ative, valide e inicie

```bash
curl -s -X POST http://localhost:8001/api/configs/kontext-api-demo/activate

curl -s -X POST http://localhost:8001/api/training/validate \
  -F __active_tab__=model \
  -F --num_train_epochs=0

curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=model \
  -F --num_train_epochs=0
```

Dicas do Kontext:

- `conditioning_type=reference_strict` mantem recortes alinhados; mude para `reference_loose` se seus datasets diferirem em proporcao
- Quantize para `int8-quanto` para caber em 24GB VRAM a 1024px; precisao total exige GPUs classe Hopper/Blackwell
- Para execucoes multi-node, defina `--accelerate_config` ou `CUDA_VISIBLE_DEVICES` antes de iniciar o servidor

## Enviar jobs locais com fila consciente de GPU

Ao rodar em uma maquina multi-GPU, voce pode enviar jobs locais via API de fila com alocacao de GPU. Jobs entram em fila se as GPUs necessarias estiverem indisponiveis.

### Verificar disponibilidade de GPU

```bash
curl -s "http://localhost:8001/api/system/status?include_allocation=true" | jq '.gpu_allocation'
```

A resposta mostra quais GPUs estao disponiveis:

```json
{
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
```

Voce tambem pode obter estatisticas da fila incluindo info de GPU local:

```bash
curl -s http://localhost:8001/api/queue/stats | jq '.local'
```

### Enviar um job local

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "no_wait": false,
    "any_gpu": false
  }'
```

Opcoes:

| Opcao | Padrao | Descricao |
|--------|---------|-------------|
| `config_name` | required | Nome do ambiente de treinamento a rodar |
| `no_wait` | false | Se true, rejeita imediatamente quando GPUs indisponiveis |
| `any_gpu` | false | Se true, usa quaisquer GPUs disponiveis em vez de IDs configurados |

Resposta:

```json
{
  "success": true,
  "job_id": "abc123",
  "status": "running",
  "allocated_gpus": [0, 1],
  "queue_position": null
}
```

O campo `status` indica o resultado:

- `running` - Job iniciou imediatamente com GPUs alocadas
- `queued` - Job ficou em fila e iniciara quando GPUs ficarem disponiveis
- `rejected` - GPUs indisponiveis e `no_wait` era true

### Configurar limites de concorrencia local

Admins podem limitar quantos jobs locais e GPUs podem ser usados via endpoint de concorrencia de fila:

```bash
# Get current limits
curl -s http://localhost:8001/api/queue/stats | jq '{local_gpu_max_concurrent, local_job_max_concurrent}'

# Update limits (alongside cloud limits)
curl -s -X POST http://localhost:8001/api/queue/concurrency \
  -H 'Content-Type: application/json' \
  -d '{
    "local_gpu_max_concurrent": 6,
    "local_job_max_concurrent": 2
  }'
```

Defina `local_gpu_max_concurrent` como `null` para uso ilimitado de GPU.

### Alternativa CLI

A mesma funcionalidade esta disponivel via CLI:

```bash
# Submit with default queuing behavior
simpletuner jobs submit my-config

# Reject if GPUs unavailable
simpletuner jobs submit my-config --no-wait

# Use any available GPUs
simpletuner jobs submit my-config --any-gpu

# Preview what would happen (dry-run)
simpletuner jobs submit my-config --dry-run
```

## Despachar jobs para workers remotos

Se voce tiver maquinas GPU remotas registradas como workers (veja [Worker Orchestration](../experimental/server/WORKERS.md)), pode despachar jobs para elas via API de fila.

### Verificar workers disponiveis

```bash
curl -s http://localhost:8001/api/admin/workers | jq '.workers[] | {name, status, gpu_name, gpu_count}'
```

### Enviar para um target especifico

```bash
# Prefer remote workers, fall back to local GPUs (default)
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "auto"
  }'

# Force dispatch to remote workers only
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "worker"
  }'

# Run only on orchestrator's local GPUs
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "local"
  }'
```

### Selecionar workers por label

Workers podem ter labels para filtragem (ex.: tipo de GPU, local, time):

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "worker",
    "worker_labels": {"gpu_type": "a100*", "team": "nlp"}
  }'
```

Labels suportam glob patterns (`*` corresponde a qualquer conjunto de caracteres).

## Endpoints uteis em resumo

- `GET /api/configs/` – lista ambientes (use `?config_type=model` para configs de treinamento)
- `GET /api/configs/examples` – enumera templates enviados
- `POST /api/configs/{name}/dataloader` – regenera um arquivo de dataloader se voce quiser defaults
- `GET /api/training/status` – estado de alto nivel, `job_id` ativo e info de startup
- `GET /api/training/events?since_index=N` – stream incremental de logs
- `POST /api/training/checkpoints` – lista checkpoints do output dir do job ativo
- `GET /api/system/status?include_allocation=true` – metricas do sistema com info de alocacao de GPU
- `GET /api/queue/stats` – estatisticas de fila incluindo alocacao de GPU local
- `POST /api/queue/submit` – envia job local ou worker com fila consciente de GPU
- `POST /api/queue/concurrency` – atualiza limites de concorrencia em nuvem e local
- `GET /api/admin/workers` – lista workers registrados e seus status

## Para onde ir depois

- Explore definicoes de opcoes especificas em `documentation/OPTIONS.md`
- Combine essas chamadas REST com `jq`/`yq` ou um cliente Python para automacao
- Conecte WebSockets em `/api/training/events/stream` para dashboards em tempo real
- Reutilize configs exportadas (`GET /api/configs/<env>/export`) para versionar setups funcionais
- **Rode treinamento em GPUs de nuvem** via Replicate—veja o [Cloud Training Tutorial](../experimental/cloud/TUTORIAL.md)

Com esses padroes voce consegue automatizar totalmente o treinamento do SimpleTuner sem tocar na WebUI, enquanto ainda depende do processo CLI consolidado.
