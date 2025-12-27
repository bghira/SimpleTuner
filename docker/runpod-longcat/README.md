# RunPod Template: LongCat Video Full Finetune

Imagem Docker pré-configurada para full finetune do modelo **LongCat Video (13.6B)** usando SimpleTuner, otimizada para 8× A100 80GB ou 8× H200.

## Imagem Docker

```
danielxmed/longcat-video-finetune:latest
```

## Características

- **SimpleTuner** pré-instalado e configurado
- **FSDP2** para distributed training
- Suporte a **AWS S3** para datasets
- Geração automática de **Parquet** para datasets grandes
- Configuração via **secrets** e variáveis de ambiente
- Opção de **auto-start** do treinamento
- TensorBoard integrado

## Configuração do Template no RunPod

### 1. Criar Template

No console do RunPod:
1. Vá para **Templates** → **New Template**
2. Configure:

| Campo | Valor |
|-------|-------|
| **Name** | LongCat Video Full Finetune |
| **Container Image** | `danielxmed/longcat-video-finetune:latest` |
| **Container Disk** | 50 GB (mínimo) |
| **Volume Disk** | 500 GB+ (para cache e checkpoints) |
| **Volume Mount Path** | `/workspace` |

### 2. Configurar Secrets (Obrigatórios)

No RunPod, vá para **Secrets** e crie:

| Secret Name | Descrição |
|-------------|-----------|
| `AWS_BUCKET_NAME` | Nome do bucket S3 com os vídeos |
| `AWS_ACCESS_KEY_ID` | Access key da AWS |
| `AWS_SECRET_ACCESS_KEY` | Secret key da AWS |

### 3. Configurar Environment Variables

No template, adicione as variáveis de ambiente:

#### Obrigatórias (via Secrets)
```
AWS_BUCKET_NAME={{ RUNPOD_SECRET_AWS_BUCKET_NAME }}
AWS_ACCESS_KEY_ID={{ RUNPOD_SECRET_AWS_ACCESS_KEY_ID }}
AWS_SECRET_ACCESS_KEY={{ RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY }}
```

#### Opcionais
| Variável | Default | Descrição |
|----------|---------|-----------|
| `AWS_REGION` | `us-east-1` | Região do bucket |
| `AWS_DATA_PREFIX` | `""` | Prefixo/pasta no bucket |
| `AWS_ENDPOINT_URL` | `null` | Para S3-compatible (R2, MinIO) |
| `USE_PARQUET` | `false` | Usar metadata Parquet |
| `AUTO_START_TRAINING` | `false` | Iniciar treino automaticamente |
| `MODEL_TYPE` | `full` | `full` ou `lora` |
| `LORA_RANK` | - | Se definido, usa LoRA |
| `BASE_MODEL_PRECISION` | `bf16` | `bf16`, `int8-quanto`, `fp8-torchao` |
| `LEARNING_RATE` | `1e-5` | Taxa de aprendizado |
| `MAX_TRAIN_STEPS` | `30000` | Número máximo de steps |
| `TRAIN_BATCH_SIZE` | `1` | Batch size por GPU |
| `GRADIENT_ACCUMULATION_STEPS` | `4` | Gradient accumulation |
| `CHECKPOINTING_STEPS` | `2500` | Salvar checkpoint a cada N steps |
| `NUM_GPUS` | `8` | Número de GPUs |

### 4. Configurar Portas

| Label | Port |
|-------|------|
| HTTP (TensorBoard) | 6006 |
| HTTP (Jupyter) | 8888 |
| TCP (SSH) | 22 |

## Uso

### Modo Interativo (Recomendado para primeira vez)

1. Inicie o Pod com `AUTO_START_TRAINING=false`
2. Conecte via SSH ou Web Terminal
3. Execute:

```bash
# Ver configuração gerada
cat /workspace/config/training_config.json
cat /workspace/config/databackend.json

# (Opcional) Gerar Parquet para startup mais rápido
/workspace/generate_parquet.sh

# Iniciar treinamento
/workspace/start_training.sh

# Em outro terminal, monitorar
/workspace/monitor.sh
```

### Modo Automático

Configure no template:
```
AUTO_START_TRAINING=true
```

O treinamento inicia automaticamente quando o Pod sobe.

### Usando Parquet (Recomendado para 500K+ vídeos)

1. Primeira vez: gere o Parquet
```bash
/workspace/generate_parquet.sh
```

2. Configure `USE_PARQUET=true` no template

3. Nas próximas vezes, o startup será muito mais rápido

## Estrutura de Diretórios

```
/workspace/
├── SimpleTuner/          # Código do SimpleTuner
├── config/               # Configurações geradas
│   ├── training_config.json
│   ├── databackend.json
│   └── metadata.parquet  # (se gerado)
├── cache/
│   ├── vae/              # Cache de VAE latents
│   └── text/             # Cache de text embeddings
├── output/               # Checkpoints salvos
├── logs/                 # TensorBoard logs
├── scripts/              # Scripts auxiliares
├── generate_parquet.sh   # Gera Parquet de metadados
├── start_training.sh     # Inicia treinamento
└── monitor.sh            # Monitora com TensorBoard
```

## Dataset Esperado no S3

```
s3://seu-bucket/
├── video_001.mp4
├── video_001.txt    # Caption
├── video_002.mp4
├── video_002.txt
└── ...
```

### Requisitos dos Vídeos

| Requisito | Valor |
|-----------|-------|
| Formato | MP4, MOV, AVI, WebM |
| Duração mínima | ~3.1s (93 frames @ 30fps) |
| Caption | Arquivo .txt com mesmo nome |

## Exemplos de Configuração

### Full Finetune Padrão (8× A100 80GB)

```
AWS_BUCKET_NAME={{ RUNPOD_SECRET_AWS_BUCKET_NAME }}
AWS_ACCESS_KEY_ID={{ RUNPOD_SECRET_AWS_ACCESS_KEY_ID }}
AWS_SECRET_ACCESS_KEY={{ RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY }}
AUTO_START_TRAINING=false
```

### Full Finetune com Auto-Start

```
AWS_BUCKET_NAME={{ RUNPOD_SECRET_AWS_BUCKET_NAME }}
AWS_ACCESS_KEY_ID={{ RUNPOD_SECRET_AWS_ACCESS_KEY_ID }}
AWS_SECRET_ACCESS_KEY={{ RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY }}
AUTO_START_TRAINING=true
USE_PARQUET=true
```

### LoRA para Prova de Conceito

```
AWS_BUCKET_NAME={{ RUNPOD_SECRET_AWS_BUCKET_NAME }}
AWS_ACCESS_KEY_ID={{ RUNPOD_SECRET_AWS_ACCESS_KEY_ID }}
AWS_SECRET_ACCESS_KEY={{ RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY }}
LORA_RANK=64
BASE_MODEL_PRECISION=int8-quanto
MAX_TRAIN_STEPS=5000
```

### Cloudflare R2 (Egress Grátis)

```
AWS_BUCKET_NAME={{ RUNPOD_SECRET_AWS_BUCKET_NAME }}
AWS_ACCESS_KEY_ID={{ RUNPOD_SECRET_AWS_ACCESS_KEY_ID }}
AWS_SECRET_ACCESS_KEY={{ RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY }}
AWS_ENDPOINT_URL=https://ACCOUNT_ID.r2.cloudflarestorage.com
AWS_REGION=auto
```

## Estimativas de Tempo

### 8× A100 80GB

| Fase | 100K vídeos | 500K vídeos |
|------|-------------|-------------|
| VAE Cache | ~10h | ~50-70h |
| Treino 10K steps | ~40h | ~40h |
| Treino 30K steps | ~120h | ~120h |

### 8× H200

| Fase | 100K vídeos | 500K vídeos |
|------|-------------|-------------|
| VAE Cache | ~6h | ~30-40h |
| Treino 10K steps | ~25h | ~25h |
| Treino 30K steps | ~75h | ~75h |

## Troubleshooting

### OOM (Out of Memory)

```bash
# Use quantização
BASE_MODEL_PRECISION=int8-quanto

# Ou reduza batch
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
```

### Startup Lento

```bash
# Gere Parquet uma vez
/workspace/generate_parquet.sh

# Configure USE_PARQUET=true
```

### Verificar Logs

```bash
# Logs do SimpleTuner
tail -f /workspace/logs/*/events.out.tfevents.*

# TensorBoard
tensorboard --logdir /workspace/logs --port 6006 --bind_all
```

## Build Local (Desenvolvimento)

```bash
cd docker/runpod-longcat
docker build -t longcat-video-finetune:latest .

# Testar
docker run --rm -it \
    -e AWS_BUCKET_NAME=test \
    -e AWS_ACCESS_KEY_ID=test \
    -e AWS_SECRET_ACCESS_KEY=test \
    longcat-video-finetune:latest /bin/bash
```

## Links

- [SimpleTuner](https://github.com/bghira/SimpleTuner)
- [RunPod Docs](https://docs.runpod.io)
- [LongCat Video](https://huggingface.co/LongCat-Video)
