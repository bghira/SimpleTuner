#!/bin/bash

accelerate launch --mixed_precision='bf16' --num_processes=2 --num_machines=1 --dynamo_backend='no' train_sdxl.py \
	--pretrained_model_name_or_path='ptx0/sdxl-base' \
	--resume_from_checkpoint='latest' \
	--learning_rate='4e-7' \
	--seed 42 \
	--instance_data_dir='/notebooks/aggregate' \
	--seen_state_path='/notebooks/SimpleTuner/seen_state.json' \
	--print_filenames \
	--mixed_precision='bf16' \
	--vae_dtype='bf16' \
	--allow_tf32 \
	--train_batch=10 \
	--validation_prompt='ethnographic photography of teddy bear at a picnic' \
	--num_validation_images=1 \
	--resolution=256 \
	--validation_resolution=512 \
	--checkpointing_steps=50 --checkpoints_total_limit=2 \
	--validation_steps=100 \
	--report_to='wandb' \
	--use_original_images=true \
	--enable_xformers_memory_efficient_attention --use_8bit_adam --use_ema \
	--gradient_checkpointing --gradient_accumulation_steps=15 \
