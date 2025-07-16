# SDXL LCM Distillation Quickstart (SimpleTuner)

In this example, we'll be training a **4-8 step SDXL student** using **LCM (Latent Consistency Model) distillation** from a pre-trained SDXL teacher model.

> **NOTE**: Other models can be used as a basis, SDXL is merely used to illustrate the configuration concepts for LCM.

LCM enables:
* Ultra-fast inference (4-8 steps vs 25-50)
* Consistency across timesteps
* High-quality outputs with minimal steps

## 📦 Installation

Follow the standard SimpleTuner installation [guide](/INSTALL.md):

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U poetry pip

# Necessary on some systems
poetry config virtualenvs.create false

# Install based on your system:
poetry install              # Linux with NVIDIA
# poetry install -C install/apple  # MacOS
# poetry install -C install/rocm   # Linux with ROCm
```

For container environments (Vast, RunPod, etc.):
```bash
apt -y install nvidia-cuda-toolkit libgl1-mesa-glx
```

---

## 📁 Configuration

Create your `config/config.json` for SDXL LCM:

```json
{
  "model_type": "lora",
  "model_family": "sdxl",
  "output_dir": "/home/user/output/sdxl-lcm",
  "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
  
  "distillation_method": "lcm",
  "distillation_config": {
    "lcm": {
      "num_ddim_timesteps": 50,
      "w_min": 1.0,
      "w_max": 12.0,
      "loss_type": "l2",
      "huber_c": 0.001,
      "timestep_scaling_factor": 10.0
    }
  },
  
  "resolution": 1024,
  "resolution_type": "pixel",
  "validation_resolution": "1024x1024,1280x768,768x1280",
  "aspect_bucket_rounding": 64,
  "minimum_image_size": 0.5,
  "maximum_image_size": 1.0,
  
  "learning_rate": 1e-4,
  "lr_scheduler": "constant_with_warmup",
  "lr_warmup_steps": 1000,
  "max_train_steps": 10000,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "gradient_checkpointing": true,
  "mixed_precision": "bf16",
  
  "lora_type": "standard",
  "lora_rank": 64,
  "lora_alpha": 64,
  "lora_dropout": 0.0,
  
  "validation_steps": 250,
  "validation_num_inference_steps": 4,
  "validation_guidance": 0.0,
  "validation_prompt": "A portrait of a woman with flowers in her hair, highly detailed, professional photography",
  "validation_negative_prompt": "blurry, low quality, distorted, amateur",
  
  "checkpointing_steps": 500,
  "checkpoints_total_limit": 5,
  "resume_from_checkpoint": "latest",
  
  "optimizer": "adamw_bf16",
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "adam_weight_decay": 1e-2,
  "adam_epsilon": 1e-8,
  "max_grad_norm": 1.0,
  
  "seed": 42,
  "push_to_hub": true,
  "hub_model_id": "your-username/sdxl-lcm-distilled",
  "report_to": "wandb",
  "tracker_project_name": "sdxl-lcm-distillation",
  "tracker_run_name": "sdxl-lcm-4step"
}
```

### Key LCM Configuration Options:

- **`num_ddim_timesteps`**: Number of timesteps in the DDIM solver (50-100 typical)
- **`w_min/w_max`**: Guidance scale range for training (1.0-12.0 for SDXL)
- **`loss_type`**: Use "l2" or "huber" (huber is more robust to outliers)
- **`timestep_scaling_factor`**: Scaling for boundary conditions (default 10.0)
- **`validation_num_inference_steps`**: Test with your target step count (4-8)
- **`validation_guidance`**: Set to 0.0 for LCM (no CFG at inference)

### For Quantized Training (Lower VRAM):

Add these options to reduce memory usage:
```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "optimizer": "optimi-lion"
}
```

---

## 🎬 Dataset Configuration

Create `multidatabackend.json` in your output directory:

```json
[
  {
    "id": "your-dataset-name",
    "type": "local",
    "crop": false,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1.0,
    "minimum_image_size": 0.5,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
    "cache_dir_vae": "cache/vae/sdxl/your-dataset",
    "instance_data_dir": "/path/to/your/dataset",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery"
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/sdxl/your-dataset",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

> **Important**: LCM distillation requires diverse, high-quality data. A minimum of 10k+ images is recommended for good results.

---

## 🚀 Training

1. **Login to services** (if using hub features):
   ```bash
   huggingface-cli login
   wandb login
   ```

2. **Start training**:
   ```bash
   bash train.sh
   ```

3. **Monitor progress**:
   - Watch for decreasing LCM loss
   - Validation images should maintain quality at 4-8 steps
   - Training typically takes 5k-10k steps

---

## 📊 Expected Results

| Metric | Expected Value | Notes |
| ------ | -------------- | ----- |
| LCM Loss | < 0.1 | Should decrease steadily |
| Validation Quality | Good at 4 steps | May need guidance=0 |
| Training Time | 5-10 hours | On single A100 |
| Final Inference | 4-8 steps | vs 25-50 for base SDXL |

---

## 🧩 Troubleshooting

| Problem | Solution |
| ------- | -------- |
| **OOM errors** | Reduce batch size, enable gradient checkpointing, use int8 quantization |
| **Blurry outputs** | Increase `num_ddim_timesteps`, check data quality, reduce learning rate |
| **Slow convergence** | Increase learning rate to 2e-4, ensure diverse dataset |
| **Validation looks bad** | Use `validation_guidance: 0.0`, check if using correct scheduler |
| **Artifacts at low steps** | Normal for <4 steps, try training longer or adjusting `w_min/w_max` |

---

## 🔧 Advanced Tips

1. **Multi-resolution training**: SDXL benefits from training on multiple aspects:
   ```json
   "validation_resolution": "1024x1024,1280x768,768x1280,1152x896,896x1152"
   ```

2. **Progressive training**: Start with more timesteps, then reduce:
   - Week 1: Train with `validation_num_inference_steps: 8`
   - Week 2: Fine-tune with `validation_num_inference_steps: 4`

3. **Scheduler for inference**: After training, use the LCM scheduler:
   ```python
   from diffusers import LCMScheduler
   scheduler = LCMScheduler.from_pretrained(
       "stabilityai/stable-diffusion-xl-base-1.0", 
       subfolder="scheduler"
   )
   ```

4. **Combining with ControlNet**: LCM works well with ControlNet for guided generation at low steps.

---

## 📚 Additional Resources

- [LCM Paper](https://arxiv.org/abs/2310.04378)
- [Diffusers LCM Docs](https://huggingface.co/docs/diffusers/using-diffusers/inference_with_lcm)
- [More SimpleTuner Docs](/documentation/quickstart/SDXL.md)

---

## 🎯 Next Steps

After successful LCM distillation:
1. Test your model with various prompts at 4-8 steps
2. Try LCM-LoRA on different base models
3. Experiment with even fewer steps (2-3) for specific use cases
4. Consider fine-tuning on domain-specific data