#!/bin/bash

# Source directory where the models are stored
SOURCE_DIR="/home/ubuntu/enwik8-usw2/sd35lora/kanji_lora1/datasets/models"

# Target directory for symlinks
TARGET_DIR="/home/ubuntu/enwik8-usw2/sd35lora/ComfyUI/ComfyUI/models/loras"

# Ensure target directory exists or create it
mkdir -p "${TARGET_DIR}"

# Iterate over each checkpoint directory
for CHECKPOINT_DIR in ${SOURCE_DIR}/checkpoint-*; do
    # Check if it's indeed a directory
    if [ -d "${CHECKPOINT_DIR}" ]; then
        # Extract the checkpoint number from the directory name
        CHECKPOINT_NAME=$(basename ${CHECKPOINT_DIR})
        
        # Define the source file path
        SOURCE_FILE="${CHECKPOINT_DIR}/pytorch_lora_weights.safetensors"
        
        # Define the symlink name with 'lora' added before 'safetensors'
        LINK_NAME="${TARGET_DIR}/${CHECKPOINT_NAME}_lora.safetensors"
        
        # Check if the source file exists
        if [ -f "${SOURCE_FILE}" ]; then
            # Create a symlink in the target directory
            echo "Creating symlink from ${SOURCE_FILE} to ${LINK_NAME}"
            ln -s "${SOURCE_FILE}" "${LINK_NAME}"
            echo "Symlink created for ${CHECKPOINT_NAME}"
        else
            echo "File not found: ${SOURCE_FILE}"
        fi
    else
        echo "Not a directory: ${CHECKPOINT_DIR}"
    fi
done

echo "Symlinking complete."