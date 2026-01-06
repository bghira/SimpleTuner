# Quickstart de Destilacao DMD (SimpleTuner)

Neste exemplo, vamos treinar um **student de 3 steps** usando **DMD (Distribution Matching Distillation)** a partir de um modelo teacher flow-matching grande como o [Wan 2.1 T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B).

Recursos do DMD:

* **Generator (Student)**: aprende a corresponder ao teacher em menos steps
* **Fake Score Transformer**: discrimina entre saidas do teacher e do student
* **Simulacao multi-step**: modo opcional de consistencia treino-inferencia

---

## Requisitos de hardware

AVISO: O DMD e intensivo em memoria devido ao fake score transformer, que exige manter uma segunda copia completa do modelo base em memoria.

Recomenda-se tentar LCM ou DCM para o modelo Wan 14B em vez de DMD se voce nao tiver a VRAM necessaria.

Uma NVIDIA B200 pode ser necessaria ao destilar o modelo 14B sem suporte de atencao esparsa.

Usar student training com LoRA pode reduzir bastante os requisitos, mas ainda e pesado.

---

## Instalacao

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
    "checkpoint_step_interval": 200,
    "checkpoints_total_limit": 3,
    "compress_disk_cache": true,
    "data_backend_config": "config/wan/multidatabackend.json",
    "delete_problematic_images": false,
    "disable_benchmark": false,
    "disable_bucket_pruning": true,
    "distillation_method": "dmd",
    "distillation_config": {
        "dmd_denoising_steps": "1000,757,522",
        "generator_update_interval": 1,
        "real_score_guidance_scale": 3.0,
        "fake_score_lr": 1e-5,
        "fake_score_weight_decay": 0.01,
        "fake_score_betas": [0.9, 0.999],
        "fake_score_eps": 1e-8,
        "fake_score_grad_clip": 1.0,
        "fake_score_guidance_scale": 0.0,
        "min_timestep_ratio": 0.02,
        "max_timestep_ratio": 0.98,
        "num_frame_per_block": 3,
        "independent_first_frame": false,
        "same_step_across_blocks": false,
        "last_step_only": false,
        "num_training_frames": 21,
        "context_noise": 0,
        "ts_schedule": true,
        "ts_schedule_max": false,
        "min_score_timestep": 0,
        "timestep_shift": 1.0
    },
    "ema_update_interval": 5,
    "ema_validation": "ema_only",
    "flow_schedule_shift": 5,
    "grad_clip_method": "value",
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": true,
    "hub_model_id": "wan-disney-DMD-3step",
    "ignore_final_epochs": true,
    "learning_rate": 2e-5,
    "lora_alpha": 128,
    "lora_rank": 128,
    "lora_type": "standard",
    "lr_scheduler": "cosine_with_min_lr",
    "lr_warmup_steps": 100,
    "max_grad_norm": 1.0,
    "max_train_steps": 4000,
    "minimum_image_size": 0,
    "mixed_precision": "bf16",
    "model_family": "wan",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "adamw_bf16",
    "output_dir": "output/wan-dmd",
    "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "push_checkpoints_to_hub": true,
    "push_to_hub": true,
    "quantize_via": "cpu",
    "report_to": "wandb",
    "resolution": 480,
    "resolution_type": "pixel_area",
    "resume_from_checkpoint": "latest",
    "seed": 1000,
    "text_encoder_1_precision": "int8-quanto",
    "tracker_project_name": "dmd-training",
    "tracker_run_name": "wan-DMD-3step",
    "train_batch_size": 1,
    "use_ema": true,
    "vae_batch_size": 1,
    "validation_guidance": 1.0,
    "validation_negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    "validation_num_inference_steps": 3,
    "validation_num_video_frames": 121,
    "validation_prompt": "A black and white animated scene unfolds featuring a distressed upright cow with prominent horns and expressive eyes, suspended by its legs from a hook on a static background wall. A smaller Mickey Mouse-like character enters, standing near a wooden bench, initiating interaction between the two. The cow's posture changes as it leans, stretches, and falls, while the mouse watches with a concerned expression, its face a mixture of curiosity and worry, in a world devoid of color.",
    "validation_prompt_library": "config/wan/validation_prompts_dmd.json",
    "validation_resolution": "1280x704",
    "validation_seed": 42,
    "validation_step_interval": 200,
    "webhook_config": "config/wan/webhook.json"
}
```

### Principais configuracoes do DMD:

* **`dmd_denoising_steps`** - Timesteps alvo para a simulacao backward (padrao: `1000,757,522` para student de 3 steps).
* **`generator_update_interval`** - Executa o replay caro do generator a cada _N_ steps do trainer. Aumente para trocar qualidade por velocidade.
* **`fake_score_lr` / `fake_score_weight_decay` / `fake_score_betas`** - Hiperparametros do otimizador do fake score transformer.
* **`fake_score_guidance_scale`** - CFG opcional na rede de fake score (padrao desligado).
* **`num_frame_per_block`, `same_step_across_blocks`, `last_step_only`** - Controlam como blocos temporais sao escalonados durante self-forcing rollout.
* **`num_training_frames`** - Maximo de frames gerados durante a simulacao backward (valores maiores melhoram fidelidade com custo de memoria).
* **`min_timestep_ratio`, `max_timestep_ratio`, `timestep_shift`** - Moldam a janela de amostragem KL. Combine com o flow schedule do teacher se fugir dos padroes.

---

## Dataset e dataloader

Para o DMD funcionar bem, voce precisa de **dados diversos e de alta qualidade**:

```json
{
  "dataset_type": "video",
  "cache_dir": "cache/wan-dmd",
  "resolution_type": "pixel_area",
  "crop": false,
  "num_frames": 121,
  "frame_interval": 1,
  "resolution": 480,
  "minimum_image_size": 0,
  "repeats": 0
}
```

> **Nota**: O dataset Disney e **inadequado** para DMD. **NAO use!** Ele e fornecido apenas para fins ilustrativos.

Voce precisa de:
> - Alto volume (minimo de 10k videos)
> - Conteudo diverso (estilos, movimentos, sujeitos diferentes)
> - Alta qualidade (sem artefatos de compressao)

Esses podem ser gerados a partir do modelo pai.

---

## Dicas de treinamento

1. **Mantenha o intervalo do generator baixo**: Comece com `"generator_update_interval": 1`. Aumente apenas se precisar de throughput e puder tolerar atualizacoes mais ruidosas.
2. **Monitore ambas as losses**: Acompanhe `dmd_loss` e `fake_score_loss` no wandb
3. **Frequencia de validacao**: DMD converge rapido, valide frequentemente
4. **Gestao de memoria**:
   - Use `gradient_checkpointing`
   - Reduza `train_batch_size` para 1
   - Considere `base_model_precision: "int8-quanto"`

---

## DMD vs DCM

| Recurso | DCM | DMD |
|---------|-----|-----|
| Uso de memoria | Menor | Maior (modelo fake score) |
| Tempo de treino | Maior | Menor (4k steps tipico) |
| Qualidade | Boa | Excelente |
| Steps de inferencia | 4-8+ | 3-8 |
| Estabilidade | Estavel | Exige tuning |

---

## Solucao de problemas

| Problema | Solucao |
|---------|-----|
| **Erros de OOM** | Reduza `num_training_frames`, diminua `generator_update_interval` ou reduza batch size |
| **Fake score nao aprende** | Aumente `fake_score_lr` ou use scheduler diferente |
| **Generator overfitting** | Aumente `generator_update_interval` para 10 |
| **Qualidade ruim em 3 steps** | Tente "1000,500" para 2 steps primeiro |
| **Treinamento instavel** | Reduza learning rates, verifique qualidade dos dados |

---

## Opcoes avancadas

Para quem quer experimentar:

```json
"distillation_config": {
    "dmd_denoising_steps": "1000,666,333",
    "generator_update_interval": 4,
    "fake_score_guidance_scale": 1.2,
    "num_training_frames": 28,
    "same_step_across_blocks": true,
    "timestep_shift": 7.0
}
```

> AVISO: Recomenda-se usar a implementacao original do FastVideo para DMD em projetos com recursos limitados, pois ela suporta sequence-parallel e video-sparse attention (VSA) para uso de runtime muito mais eficiente.
