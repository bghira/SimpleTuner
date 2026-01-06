# Quickstart de Destilacao DCM (SimpleTuner)

Neste exemplo, vamos treinar um **student de 4 steps** usando **destilacao DCM** a partir de um modelo teacher flow-matching grande como o [Wan 2.1 T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B).

O DCM suporta:

* **Modo semantic**: flow-matching padrao com CFG embutido.
* **Modo fine**: supervisao adversarial baseada em GAN (experimental).

---

## Requisitos de hardware

| Modelo     | Batch Size | VRAM minima | Notas                                  |
| --------- | ---------- | -------- | -------------------------------------- |
| Wan 1.3B  | 1          | 12 GB    | GPU nivel A5000 / 3090+                |
| Wan 14B   | 1          | 24 GB    | Mais lento; use `--offload_during_startup` |
| Modo fine | 1          | +10%     | Discriminador roda por GPU             |

> AVISO: Macs e Apple silicon sao lentos e nao recomendados. Voce tera tempos de 10 min/step mesmo em modo semantic.

---

## Instalacao

Mesmos passos do guia Wan:

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.12 -m venv .venv && source .venv/bin/activate

# Instalar com deteccao automatica de plataforma
pip install -e .
```

**Nota:** O setup.py detecta sua plataforma (CUDA/ROCm/Apple) e instala as dependencias apropriadas.

---

## Configuracao

Edite seu `config/config.json`:

```json
{
    "aspect_bucket_rounding": 2,
    "attention_mechanism": "diffusers",
    "base_model_precision": "int8-quanto",
    "caption_dropout_probability": 0.1,
    "checkpoint_step_interval": 100,
    "checkpoints_total_limit": 5,
    "compress_disk_cache": true,
    "data_backend_config": "config/wan/multidatabackend.json",
    "delete_problematic_images": false,
    "disable_benchmark": false,
    "disable_bucket_pruning": true,
    "distillation_method": "dcm",
    "distillation_config": {
      "mode": "semantic",
      "euler_steps": 100
    },
    "ema_update_interval": 2,
    "ema_validation": "ema_only",
    "flow_schedule_shift": 17,
    "grad_clip_method": "value",
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": true,
    "hub_model_id": "wan-disney-DCM-distilled",
    "ignore_final_epochs": true,
    "learning_rate": 1e-4,
    "lora_alpha": 128,
    "lora_rank": 128,
    "lora_type": "standard",
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 400000,
    "lycoris_config": "config/wan/lycoris_config.json",
    "max_grad_norm": 0.01,
    "max_train_steps": 400000,
    "minimum_image_size": 0,
    "mixed_precision": "bf16",
    "model_family": "wan",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "adamw_bf16",
    "output_dir": "output/wan",
    "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "prodigy_steps": 100000,
    "push_checkpoints_to_hub": true,
    "push_to_hub": true,
    "quantize_via": "cpu",
    "report_to": "wandb",
    "resolution": 480,
    "resolution_type": "pixel_area",
    "resume_from_checkpoint": "latest",
    "seed": 42,
    "text_encoder_1_precision": "int8-quanto",
    "tracker_project_name": "lora-training",
    "tracker_run_name": "wan-AdamW-DCM",
    "train_batch_size": 2,
    "use_ema": false,
    "vae_batch_size": 1,
    "validation_guidance": 1.0,
    "validation_negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    "validation_num_inference_steps": 8,
    "validation_num_video_frames": 16,
    "validation_prompt": "A black and white animated scene unfolds featuring a distressed upright cow with prominent horns and expressive eyes, suspended by its legs from a hook on a static background wall. A smaller Mickey Mouse-like character enters, standing near a wooden bench, initiating interaction between the two. The cow's posture changes as it leans, stretches, and falls, while the mouse watches with a concerned expression, its face a mixture of curiosity and worry, in a world devoid of color.",
    "validation_prompt_library": false,
    "validation_resolution": "832x480",
    "validation_seed": 42,
    "validation_step_interval": 4,
    "webhook_config": "config/wan/webhook.json"
}
```

### Opcional:

* Para **modo fine**, basta mudar `"mode": "fine"`.
  - Este modo e experimental no SimpleTuner e requer passos extras para uso, que nao estao detalhados neste guia.

---

## Dataset e dataloader

Reutilize o dataset Disney e o JSON `data_backend_config` do quickstart do Wan.

> **Nota**: Este dataset e inadequado para destilacao, e necessario um conjunto **muito** mais diverso e volumoso para ter sucesso.

Garanta:

* `num_frames`: 75–81
* `resolution`: 480
* `crop`: false (deixe videos sem crop)
* `repeats`: 0 por enquanto

---

## Notas

* **Modo semantic** e estavel e recomendado para a maioria dos casos.
* **Modo fine** adiciona realismo, mas precisa de mais steps e tuning, e o suporte atual do SimpleTuner para isso nao e bom.

---

## Solucao de problemas

| Problema                      | Solucao                                                                  |
| ---------------------------- | -------------------------------------------------------------------- |
| **Resultados borrados**       | Use mais euler_steps, ou aumente `multiphase`                       |
| **Validacao degradando**  | Use `validation_guidance: 1.0`                                       |
| **OOM em modo fine**         | Reduza `train_batch_size`, reduza niveis de precisao, ou use GPU maior |
| **Modo fine nao converge** | Nao use modo fine; ele nao e muito bem testado no SimpleTuner      |
