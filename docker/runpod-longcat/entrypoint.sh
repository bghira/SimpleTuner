#!/bin/bash
# =============================================================================
# RunPod Entrypoint: LongCat Video Full Finetune
# =============================================================================
set -e

echo "============================================="
echo "  LongCat Video Full Finetune - RunPod"
echo "============================================="
echo ""

# -----------------------------------------------------------------------------
# 1. Verifica variáveis obrigatórias
# -----------------------------------------------------------------------------
echo "[1/6] Verificando configuração..."

# Verifica secrets obrigatórios (podem vir como RUNPOD_SECRET_ ou diretamente)
AWS_BUCKET_NAME="${AWS_BUCKET_NAME:-${RUNPOD_SECRET_AWS_BUCKET_NAME}}"
AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-${RUNPOD_SECRET_AWS_ACCESS_KEY_ID}}"
AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-${RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY}}"

if [ -z "$AWS_BUCKET_NAME" ]; then
    echo "ERRO: AWS_BUCKET_NAME não definido!"
    echo "Configure o secret AWS_BUCKET_NAME no template do RunPod"
    exit 1
fi

if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "ERRO: AWS_ACCESS_KEY_ID não definido!"
    echo "Configure o secret AWS_ACCESS_KEY_ID no template do RunPod"
    exit 1
fi

if [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "ERRO: AWS_SECRET_ACCESS_KEY não definido!"
    echo "Configure o secret AWS_SECRET_ACCESS_KEY no template do RunPod"
    exit 1
fi

echo "  ✓ Credenciais AWS configuradas"

# -----------------------------------------------------------------------------
# 2. Configura variáveis com defaults
# -----------------------------------------------------------------------------
echo "[2/6] Aplicando configurações..."

# AWS
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DATA_PREFIX="${AWS_DATA_PREFIX:-}"
export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-}"

# Treinamento
export USE_PARQUET="${USE_PARQUET:-false}"
export AUTO_START_TRAINING="${AUTO_START_TRAINING:-false}"
export MODEL_TYPE="${MODEL_TYPE:-full}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-30000}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-2500}"
export VALIDATION_EVERY_N_STEPS="${VALIDATION_EVERY_N_STEPS:-2500}"
export NUM_GPUS="${NUM_GPUS:-8}"

# LoRA (se definido, usa LoRA em vez de full)
export LORA_RANK="${LORA_RANK:-}"

# Precisão
export BASE_MODEL_PRECISION="${BASE_MODEL_PRECISION:-bf16}"

# Diretórios
export CONFIG_DIR="${CONFIG_DIR:-/workspace/config}"
export CACHE_DIR="${CACHE_DIR:-/workspace/cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output}"
export SIMPLETUNER_DIR="${SIMPLETUNER_DIR:-/workspace/SimpleTuner}"

echo "  ✓ Bucket: $AWS_BUCKET_NAME"
echo "  ✓ Região: $AWS_REGION"
echo "  ✓ Prefixo: ${AWS_DATA_PREFIX:-<raiz do bucket>}"
echo "  ✓ Tipo de treino: ${LORA_RANK:+LoRA rank $LORA_RANK}${LORA_RANK:-Full Finetune}"
echo "  ✓ Precisão: $BASE_MODEL_PRECISION"
echo "  ✓ Learning Rate: $LEARNING_RATE"
echo "  ✓ Max Steps: $MAX_TRAIN_STEPS"
echo "  ✓ GPUs: $NUM_GPUS"
echo "  ✓ Use Parquet: $USE_PARQUET"
echo "  ✓ Auto Start: $AUTO_START_TRAINING"

# -----------------------------------------------------------------------------
# 3. Gera arquivos de configuração
# -----------------------------------------------------------------------------
echo "[3/6] Gerando arquivos de configuração..."

# Determina model_type
if [ -n "$LORA_RANK" ]; then
    EFFECTIVE_MODEL_TYPE="lora"
    LORA_CONFIG='"lora_rank": '$LORA_RANK','
else
    EFFECTIVE_MODEL_TYPE="full"
    LORA_CONFIG=""
fi

# Gera training_config.json
cat > "${CONFIG_DIR}/training_config.json" << EOF
{
  "model_type": "${EFFECTIVE_MODEL_TYPE}",
  "model_family": "longcat_video",
  "model_flavour": "final",
  "pretrained_model_name_or_path": null,

  "base_model_precision": "${BASE_MODEL_PRECISION}",
  "attention_implementation": "sdpa",

  "output_dir": "${OUTPUT_DIR}",
  "data_backend_config": "${CONFIG_DIR}/databackend.json",

  ${LORA_CONFIG}
  "train_batch_size": ${TRAIN_BATCH_SIZE},
  "gradient_accumulation_steps": ${GRADIENT_ACCUMULATION_STEPS},
  "gradient_checkpointing": true,

  "learning_rate": ${LEARNING_RATE},
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 500,
  "max_train_steps": ${MAX_TRAIN_STEPS},
  "checkpointing_steps": ${CHECKPOINTING_STEPS},
  "checkpoints_total_limit": 5,

  "optimizer": "adamw_bf16",
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "adam_weight_decay": 0.01,
  "adam_epsilon": 1e-8,
  "max_grad_norm": 1.0,

  "fsdp_enable": true,
  "fsdp_version": 2,
  "fsdp_state_dict_type": "SHARDED_STATE_DICT",
  "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
  "fsdp_cpu_ram_efficient_loading": true,

  "validation_prompt": "A professional video showing the main subject in high quality, smooth motion, natural lighting",
  "validation_negative_prompt": "",
  "validation_resolution": "480x832",
  "validation_num_video_frames": 93,
  "validation_num_inference_steps": 40,
  "validation_guidance": 4.0,
  "validation_every_n_steps": ${VALIDATION_EVERY_N_STEPS},
  "num_validation_videos": 2,

  "seed": 42,
  "mixed_precision": "bf16",
  "report_to": "tensorboard",
  "logging_dir": "/workspace/logs",

  "dataloader_num_workers": 4,
  "allow_tf32": true
}
EOF

echo "  ✓ training_config.json gerado"

# Gera databackend.json
if [ "$USE_PARQUET" = "true" ] && [ -f "${CONFIG_DIR}/metadata.parquet" ]; then
    # Versão com Parquet
    cat > "${CONFIG_DIR}/databackend.json" << EOF
[
  {
    "id": "longcat-video-dataset",
    "type": "aws",
    "dataset_type": "video",

    "aws_bucket_name": "${AWS_BUCKET_NAME}",
    "aws_region_name": "${AWS_REGION}",
    "aws_endpoint_url": ${AWS_ENDPOINT_URL:+\"$AWS_ENDPOINT_URL\"}${AWS_ENDPOINT_URL:-null},
    "aws_access_key_id": "${AWS_ACCESS_KEY_ID}",
    "aws_secret_access_key": "${AWS_SECRET_ACCESS_KEY}",
    "aws_data_prefix": "${AWS_DATA_PREFIX}",

    "metadata_backend": "parquet",
    "caption_strategy": "parquet",
    "parquet": {
      "path": "${CONFIG_DIR}/metadata.parquet",
      "filename_column": "filename",
      "caption_column": "caption",
      "width_column": "width",
      "height_column": "height"
    },

    "resolution": 480,
    "resolution_type": "pixel_area",
    "minimum_image_size": 256,
    "maximum_image_size": 1024,

    "crop": true,
    "crop_style": "center",
    "crop_aspect": "preserve",

    "cache_dir_vae": "${CACHE_DIR}/vae",
    "preserve_data_backend_cache": true,
    "skip_file_discovery": "vae,aspect,metadata",

    "video": {
      "bucket_strategy": "aspect_ratio",
      "min_frames": 93,
      "max_frames": 93,
      "frame_interval": 4
    },

    "repeats": 0
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "${CACHE_DIR}/text",
    "write_batch_size": 128
  }
]
EOF
    echo "  ✓ databackend.json gerado (modo Parquet)"
else
    # Versão direta do S3
    cat > "${CONFIG_DIR}/databackend.json" << EOF
[
  {
    "id": "longcat-video-dataset",
    "type": "aws",
    "dataset_type": "video",

    "aws_bucket_name": "${AWS_BUCKET_NAME}",
    "aws_region_name": "${AWS_REGION}",
    "aws_endpoint_url": ${AWS_ENDPOINT_URL:+\"$AWS_ENDPOINT_URL\"}${AWS_ENDPOINT_URL:-null},
    "aws_access_key_id": "${AWS_ACCESS_KEY_ID}",
    "aws_secret_access_key": "${AWS_SECRET_ACCESS_KEY}",
    "aws_data_prefix": "${AWS_DATA_PREFIX}",

    "caption_strategy": "textfile",

    "resolution": 480,
    "resolution_type": "pixel_area",
    "minimum_image_size": 256,
    "maximum_image_size": 1024,

    "crop": true,
    "crop_style": "center",
    "crop_aspect": "preserve",

    "cache_dir_vae": "${CACHE_DIR}/vae",
    "preserve_data_backend_cache": true,

    "video": {
      "bucket_strategy": "aspect_ratio",
      "min_frames": 93,
      "max_frames": 93,
      "frame_interval": 4
    },

    "repeats": 0
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "${CACHE_DIR}/text",
    "write_batch_size": 128
  }
]
EOF
    echo "  ✓ databackend.json gerado (modo S3 direto)"
fi

# -----------------------------------------------------------------------------
# 4. Cria scripts de conveniência
# -----------------------------------------------------------------------------
echo "[4/6] Criando scripts de conveniência..."

# Script para gerar Parquet
cat > /workspace/generate_parquet.sh << 'SCRIPT'
#!/bin/bash
echo "Gerando Parquet de metadados..."
python /workspace/scripts/generate_metadata_parquet.py \
    --bucket "$AWS_BUCKET_NAME" \
    --prefix "$AWS_DATA_PREFIX" \
    --output "${CONFIG_DIR}/metadata.parquet" \
    --region "$AWS_REGION" \
    ${AWS_ENDPOINT_URL:+--endpoint-url "$AWS_ENDPOINT_URL"} \
    --workers 32

if [ -f "${CONFIG_DIR}/metadata.parquet" ]; then
    echo "✓ Parquet gerado com sucesso!"
    echo "Execute: USE_PARQUET=true para usar no próximo treino"
else
    echo "✗ Erro ao gerar Parquet"
    exit 1
fi
SCRIPT
chmod +x /workspace/generate_parquet.sh

# Script para iniciar treino
cat > /workspace/start_training.sh << 'SCRIPT'
#!/bin/bash
echo "============================================="
echo "  Iniciando Treinamento LongCat Video"
echo "============================================="

cd "${SIMPLETUNER_DIR}"

# Configura ambiente
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# Detecta NVLink
if nvidia-smi topo -m 2>/dev/null | grep -q "NV"; then
    echo "NVLink detectado"
    export NCCL_P2P_LEVEL=NVL
fi

echo "Iniciando com ${NUM_GPUS} GPUs..."
echo ""

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    -m simpletuner.train \
    --config "${CONFIG_DIR}/training_config.json" \
    "$@"
SCRIPT
chmod +x /workspace/start_training.sh

# Script para monitorar
cat > /workspace/monitor.sh << 'SCRIPT'
#!/bin/bash
echo "Monitorando treinamento..."
echo "TensorBoard: http://localhost:6006"
echo ""
tensorboard --logdir /workspace/logs --port 6006 --bind_all &
watch -n 5 nvidia-smi
SCRIPT
chmod +x /workspace/monitor.sh

echo "  ✓ Scripts criados em /workspace/"
echo "    - generate_parquet.sh"
echo "    - start_training.sh"
echo "    - monitor.sh"

# -----------------------------------------------------------------------------
# 5. Exibe informações do sistema
# -----------------------------------------------------------------------------
echo "[5/6] Informações do sistema..."

echo "  Python: $(python --version 2>&1 | cut -d' ' -f2)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.version.cuda)')"

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "  GPUs: ${GPU_COUNT}x ${GPU_NAME} (${GPU_MEM})"
fi

# -----------------------------------------------------------------------------
# 6. Ação final
# -----------------------------------------------------------------------------
echo "[6/6] Finalizando setup..."

if [ "$USE_PARQUET" = "true" ] && [ ! -f "${CONFIG_DIR}/metadata.parquet" ]; then
    echo ""
    echo "  ⚠ USE_PARQUET=true mas metadata.parquet não existe"
    echo "  Execute: /workspace/generate_parquet.sh"
    echo ""
fi

if [ "$AUTO_START_TRAINING" = "true" ]; then
    echo ""
    echo "============================================="
    echo "  AUTO_START_TRAINING=true"
    echo "  Iniciando treinamento automaticamente..."
    echo "============================================="
    echo ""

    # Inicia TensorBoard em background
    tensorboard --logdir /workspace/logs --port 6006 --bind_all &

    # Inicia treinamento
    exec /workspace/start_training.sh
else
    echo ""
    echo "============================================="
    echo "  Setup concluído!"
    echo "============================================="
    echo ""
    echo "  Próximos passos:"
    echo "    1. (Opcional) Gerar Parquet: /workspace/generate_parquet.sh"
    echo "    2. Iniciar treino: /workspace/start_training.sh"
    echo "    3. Monitorar: /workspace/monitor.sh"
    echo ""
    echo "  Arquivos de configuração:"
    echo "    - ${CONFIG_DIR}/training_config.json"
    echo "    - ${CONFIG_DIR}/databackend.json"
    echo ""
    echo "  Diretórios:"
    echo "    - Cache VAE: ${CACHE_DIR}/vae"
    echo "    - Cache Text: ${CACHE_DIR}/text"
    echo "    - Output: ${OUTPUT_DIR}"
    echo "    - Logs: /workspace/logs"
    echo ""

    # Mantém container rodando (para JupyterLab/SSH do RunPod)
    if [ -f /start.sh ]; then
        exec /start.sh
    else
        # Fallback: mantém rodando
        tail -f /dev/null
    fi
fi
