#!/bin/bash
# =============================================================================
# LongCat Video Full Finetune - Launch Script
# =============================================================================
#
# Este script configura e lança o treinamento do LongCat Video com FSDP2
# em 8 GPUs (A100 80GB ou H200).
#
# Uso:
#   ./launch_training.sh [--dry-run] [--resume CHECKPOINT_PATH]
#
# =============================================================================

set -e

# Diretório base do SimpleTuner
SIMPLETUNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CONFIG_DIR="${SIMPLETUNER_DIR}/documentation/examples/longcat-fullfinetune"

# Número de GPUs
NUM_GPUS=${NUM_GPUS:-8}

# Verifica se é dry-run
DRY_RUN=""
RESUME_ARG=""
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN="--dry-run"
            echo "[DRY RUN] Apenas mostrando configuração, não vai treinar"
            ;;
        --resume)
            shift
            RESUME_ARG="--resume_from_checkpoint $1"
            echo "[RESUME] Continuando do checkpoint: $1"
            ;;
    esac
done

echo "============================================="
echo "LongCat Video Full Finetune"
echo "============================================="
echo "SimpleTuner dir: ${SIMPLETUNER_DIR}"
echo "Config dir: ${CONFIG_DIR}"
echo "Número de GPUs: ${NUM_GPUS}"
echo "============================================="

# Verifica se as configurações existem
if [ ! -f "${CONFIG_DIR}/training_config.json" ]; then
    echo "ERRO: training_config.json não encontrado em ${CONFIG_DIR}"
    exit 1
fi

if [ ! -f "${CONFIG_DIR}/databackend.json" ]; then
    echo "ERRO: databackend.json não encontrado em ${CONFIG_DIR}"
    exit 1
fi

# Verifica credenciais AWS
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "AVISO: Variáveis AWS_ACCESS_KEY_ID e/ou AWS_SECRET_ACCESS_KEY não definidas"
    echo "       Defina-as ou edite databackend.json com suas credenciais"
fi

# Cria diretórios de cache se não existirem
echo "Criando diretórios de cache..."
mkdir -p /mnt/nvme/cache/vae/longcat
mkdir -p /mnt/nvme/cache/text/longcat
mkdir -p "${SIMPLETUNER_DIR}/output/longcat-fullfinetune"
mkdir -p "${SIMPLETUNER_DIR}/logs/longcat-fullfinetune"

# Configurações de ambiente para performance
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# Se houver NVLink, habilita
if nvidia-smi topo -m 2>/dev/null | grep -q "NV"; then
    echo "NVLink detectado, habilitando otimizações..."
    export NCCL_P2P_LEVEL=NVL
fi

echo ""
echo "Iniciando treinamento com ${NUM_GPUS} GPUs..."
echo "============================================="

cd "${SIMPLETUNER_DIR}"

# Lança o treinamento
# Usando accelerate para multi-GPU com FSDP2
if [ -n "$DRY_RUN" ]; then
    echo "[DRY RUN] Comando que seria executado:"
    echo ""
    echo "accelerate launch \\"
    echo "    --num_processes=${NUM_GPUS} \\"
    echo "    --mixed_precision=bf16 \\"
    echo "    --dynamo_backend=no \\"
    echo "    -m simpletuner.train \\"
    echo "    --config ${CONFIG_DIR}/training_config.json \\"
    echo "    ${RESUME_ARG}"
else
    accelerate launch \
        --num_processes=${NUM_GPUS} \
        --mixed_precision=bf16 \
        --dynamo_backend=no \
        -m simpletuner.train \
        --config "${CONFIG_DIR}/training_config.json" \
        ${RESUME_ARG}
fi

echo ""
echo "============================================="
echo "Treinamento finalizado!"
echo "============================================="
