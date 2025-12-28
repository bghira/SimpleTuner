#!/bin/bash
# =============================================================================
# Build and Push LongCat Video Finetune Docker Image
# =============================================================================
#
# Usage:
#   ./build_and_push.sh [--no-push]
#
# Requirements:
#   - Docker installed
#   - Docker Hub login: docker login
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

# Build for linux/amd64 (RunPod compatible)
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
    echo "  Image published!"
    echo "  ${IMAGE_NAME}:${TAG}"
    echo "  ${IMAGE_NAME}:$(date +%Y%m%d)"
    echo "============================================="
else
    echo ""
    echo "[3/3] Skipping push (--no-push)"
    echo ""
    echo "============================================="
    echo "  Build complete (local only)"
    echo "  ${IMAGE_NAME}:${TAG}"
    echo "============================================="
fi
