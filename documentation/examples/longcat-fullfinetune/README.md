# LongCat Video Full Finetune - Configuração Pronta

Configuração otimizada para full finetune do modelo LongCat Video (13.6B) em 8× A100 80GB ou 8× H200.

## Especificações

| Item | Valor |
|------|-------|
| **Modelo** | LongCat Video 13.6B |
| **Tipo de treino** | Full Finetune (não LoRA) |
| **Precisão** | bf16 |
| **GPUs** | 8× A100 80GB ou 8× H200 |
| **Distribuição** | FSDP2 (PyTorch Fully Sharded Data Parallel v2) |
| **Dataset** | 523,325 vídeos de 5s via AWS S3 |
| **Formato** | video.mp4 + video.txt (caption) |

## Estrutura de Arquivos

```
documentation/examples/longcat-fullfinetune/
├── training_config.json           # Configuração principal de treino
├── databackend.json               # Configuração do dataset S3
├── databackend-with-parquet.json  # Versão otimizada com Parquet
├── databackend-local.json         # Versão para dados locais
├── README.md                      # Este arquivo
└── scripts/
    ├── generate_metadata_parquet.py  # Gera Parquet de metadados (opcional)
    └── launch_training.sh            # Script de lançamento
```

## Pré-requisitos

1. **Hardware**: 8× A100 80GB ou 8× H200
2. **Storage**: ~10TB de NVMe para cache de VAE
3. **Rede**: Acesso ao bucket S3 com os vídeos
4. **Software**: SimpleTuner instalado com CUDA

## Setup Rápido

### 1. Configure as Credenciais AWS

Edite `databackend.json` e preencha:

```json
{
  "aws_bucket_name": "SEU_BUCKET",
  "aws_region_name": "sua-regiao",
  "aws_access_key_id": "SUA_ACCESS_KEY",
  "aws_secret_access_key": "SUA_SECRET_KEY",
  "aws_data_prefix": "pasta/dos/videos/"
}
```

Ou exporte como variáveis de ambiente:

```bash
export AWS_ACCESS_KEY_ID="sua_key"
export AWS_SECRET_ACCESS_KEY="sua_secret"
```

### 2. Ajuste os Caminhos de Cache

Edite `databackend.json` e ajuste:

```json
{
  "cache_dir_vae": "/mnt/nvme/cache/vae/longcat"
}
```

E o cache de text embeds:

```json
{
  "cache_dir": "/mnt/nvme/cache/text/longcat"
}
```

**Importante**: Use storage NVMe rápido para o cache!

### 3. (Opcional) Gere o Parquet de Metadados

Para datasets muito grandes, gerar um Parquet de metadados acelera o startup:

```bash
cd documentation/examples/longcat-fullfinetune/scripts

# Instala dependências
pip install boto3 opencv-python pandas tqdm pyarrow

# Gera o Parquet (pode levar algumas horas para 500K+ vídeos)
python generate_metadata_parquet.py \
    --bucket SEU_BUCKET \
    --prefix "pasta/dos/videos/" \
    --output ../metadata.parquet \
    --region us-east-1 \
    --workers 32
```

Depois, copie `databackend-with-parquet.json` para `databackend.json` ou edite o arquivo para usar o Parquet:

```json
{
  "metadata_backend": "parquet",
  "caption_strategy": "parquet",
  "parquet": {
    "path": "documentation/examples/longcat-fullfinetune/metadata.parquet",
    "filename_column": "filename",
    "caption_column": "caption",
    "width_column": "width",
    "height_column": "height"
  },
  "skip_file_discovery": "vae,aspect,metadata"
}
```

### 4. Lance o Treinamento

```bash
# Torna o script executável
chmod +x documentation/examples/longcat-fullfinetune/scripts/launch_training.sh

# Dry run para verificar configuração
./documentation/examples/longcat-fullfinetune/scripts/launch_training.sh --dry-run

# Inicia o treinamento
./documentation/examples/longcat-fullfinetune/scripts/launch_training.sh
```

Ou manualmente:

```bash
accelerate launch \
    --num_processes=8 \
    --mixed_precision=bf16 \
    -m simpletuner.train \
    --config documentation/examples/longcat-fullfinetune/training_config.json
```

## Fases do Treinamento

O treinamento acontece em duas fases:

### Fase 1: Pré-processamento (Cache de VAE)

Na primeira execução, o SimpleTuner irá:

1. Listar todos os vídeos no S3
2. Baixar cada vídeo
3. Redimensionar e cortar para 480p, 93 frames
4. Encodar pelo VAE e salvar cache em `.pt`
5. Encodar captions pelo text encoder

**Tempo estimado**: 50-70 horas para 500K vídeos (paralelizado em 8 GPUs)

### Fase 2: Treinamento

Após o cache estar pronto:

- Lê latents do cache local (rápido)
- Treina o transformer completo com FSDP2
- Salva checkpoints a cada 2500 steps

**Tempo estimado**: ~15-20 segundos por step

## Ajustes Opcionais

### Se ficar sem VRAM

Descomente no `training_config.json`:

```json
{
  "musubi_blocks_to_swap": 4,
  "musubi_block_swap_device": "cpu"
}
```

Isso faz streaming de blocos para CPU RAM, reduzindo VRAM.

### Ajustar Learning Rate

Para um domínio muito diferente do pré-treino:

```json
{
  "learning_rate": 5e-6,
  "lr_warmup_steps": 1000
}
```

### Treinar por mais tempo

```json
{
  "max_train_steps": 50000
}
```

### Validação mais frequente

```json
{
  "validation_every_n_steps": 1000,
  "num_validation_videos": 4
}
```

## Monitoramento

### TensorBoard

```bash
tensorboard --logdir logs/longcat-fullfinetune --port 6006
```

### Checkpoints

Salvos em `output/longcat-fullfinetune/` a cada 2500 steps.

Para resumir de um checkpoint:

```bash
./scripts/launch_training.sh --resume output/longcat-fullfinetune/checkpoint-5000
```

## Estimativas de Tempo (8× A100 80GB)

| Fase | Dataset 100K | Dataset 500K |
|------|-------------|--------------|
| VAE Cache | ~10h | ~50-70h |
| Treino 10K steps | ~40h | ~40h |
| Treino 30K steps | ~120h | ~120h |

## Troubleshooting

### OOM (Out of Memory)

1. Reduza `train_batch_size` para 1
2. Aumente `gradient_accumulation_steps` para 8
3. Habilite `musubi_blocks_to_swap`
4. Use `base_model_precision: "int8-quanto"`

### S3 Download Lento

1. Use máquinas na mesma região do bucket
2. Considere Cloudflare R2 (egress grátis)
3. Faça download local antes e use `type: "local"`

### Startup Lento

1. Gere o Parquet de metadados
2. Configure `skip_file_discovery: "vae,aspect,metadata"`
3. Configure `preserve_data_backend_cache: true`

## Próximos Passos Após o Treino

### Converter para Formato Diffusers

O modelo treinado já estará em formato compatível com Diffusers em:
`output/longcat-fullfinetune/`

### Inference

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "output/longcat-fullfinetune",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

video = pipe(
    prompt="Your prompt here",
    num_frames=93,
    guidance_scale=4.0,
    num_inference_steps=40
).frames
```

## Contato

Se tiver problemas, abra uma issue no repositório SimpleTuner:
https://github.com/bghira/SimpleTuner/issues
