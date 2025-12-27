#!/bin/bash
# =============================================================================
# Build and Push LongCat Video Finetune Docker Image
# =============================================================================
#
# Uso:
#   ./build_and_push.sh [--no-push]
#
# Requer:
#   - Docker instalado
#   - Login no Docker Hub: docker login
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="danielxmed/longcat-video-finetune"
TAG="latest"

echo "============================================="
echo "  Build: ${IMAGE_NAME}:${TAG}"
echo "============================================="

cd "$SCRIPT_DIR"

# Build para linux/amd64 (compatível com RunPod)
echo ""
echo "[1/3] Building Docker image..."
docker build \
    --platform linux/amd64 \
    -t "${IMAGE_NAME}:${TAG}" \
    .

echo ""
echo "[2/3] Tagging image..."
docker tag "${IMAGE_NAME}:${TAG}" "${IMAGE_NAME}:$(date +%Y%m%d)"

if [ "$1" != "--no-push" ]; then
    echo ""
    echo "[3/3] Pushing to Docker Hub..."
    docker push "${IMAGE_NAME}:${TAG}"
    docker push "${IMAGE_NAME}:$(date +%Y%m%d)"

    echo ""
    echo "============================================="
    echo "  Imagem publicada!"
    echo "  ${IMAGE_NAME}:${TAG}"
    echo "  ${IMAGE_NAME}:$(date +%Y%m%d)"
    echo "============================================="
else
    echo ""
    echo "[3/3] Skipping push (--no-push)"
    echo ""
    echo "============================================="
    echo "  Build concluído (local apenas)"
    echo "  ${IMAGE_NAME}:${TAG}"
    echo "============================================="
fi
