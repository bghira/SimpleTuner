# Quickstart de Destilacao LCM SDXL (SimpleTuner)

Neste exemplo, vamos treinar um **student SDXL de 4-8 steps** usando destilacao **LCM (Latent Consistency Model)** a partir de um modelo teacher SDXL pretreinado.

> **NOTA**: Outros modelos podem ser usados como base; o SDXL e usado apenas para ilustrar os conceitos de configuracao do LCM.

O LCM permite:
* Inferencia ultra-rapida (4-8 steps vs 25-50)
* Consistencia entre timesteps
* Saidas de alta qualidade com poucos steps

## Instalacao

Siga o guia de instalacao padrao do SimpleTuner [aqui](../INSTALL.md):

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.12 -m venv .venv && source .venv/bin/activate

# Instalar com deteccao automatica de plataforma
pip install -e .
```

**Nota:** O setup.py detecta sua plataforma (CUDA/ROCm/Apple) e instala as dependencias apropriadas.

Para ambientes de container (Vast, RunPod, etc.):
```bash
apt -y install nvidia-cuda-toolkit
```

---

## Configuracao

Crie seu `config/config.json` para SDXL LCM:

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

  "validation_step_interval": 250,
  "validation_num_inference_steps": 4,
  "validation_guidance": 0.0,
  "validation_prompt": "A portrait of a woman with flowers in her hair, highly detailed, professional photography",
  "validation_negative_prompt": "blurry, low quality, distorted, amateur",

  "checkpoint_step_interval": 500,
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

### Opcoes principais de configuracao do LCM:

- **`num_ddim_timesteps`**: Numero de timesteps no solver DDIM (50-100 tipico)
- **`w_min/w_max`**: Faixa de guidance scale para treino (1.0-12.0 para SDXL)
- **`loss_type`**: Use "l2" ou "huber" (huber e mais robusto a outliers)
- **`timestep_scaling_factor`**: Escala para condicoes de contorno (padrao 10.0)
- **`validation_num_inference_steps`**: Teste com sua contagem alvo de steps (4-8)
- **`validation_guidance`**: Defina como 0.0 para LCM (sem CFG na inferencia)

### Para treinamento quantizado (menos VRAM):

Adicione estas opcoes para reduzir uso de memoria:
```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "optimizer": "optimi-lion"
}
```

---

## Configuracao do dataset

Crie `multidatabackend.json` no seu diretorio de output:

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

> **Importante**: A destilacao LCM exige dados diversos e de alta qualidade. Recomenda-se um minimo de 10k+ imagens para bons resultados.

---

## Treinamento

1. **Login nos servicos** (se usar recursos do hub):
   ```bash
   huggingface-cli login
   wandb login
   ```

2. **Inicie o treinamento**:
   ```bash
   bash train.sh
   ```

3. **Monitore o progresso**:
   - Acompanhe a perda LCM diminuindo
   - Imagens de validacao devem manter qualidade em 4-8 steps
   - O treinamento tipicamente leva 5k-10k steps

---

## Resultados esperados

| Metrica | Valor esperado | Notas |
| ------ | -------------- | ----- |
| LCM Loss | < 0.1 | Deve diminuir de forma constante |
| Qualidade de validacao | Boa em 4 steps | Pode exigir guidance=0 |
| Tempo de treinamento | 5-10 horas | Em um unico A100 |
| Inferencia final | 4-8 steps | vs 25-50 no SDXL base |

---

## Solucao de problemas

| Problema | Solucao |
| ------- | -------- |
| **Erros de OOM** | Reduza batch size, habilite gradient checkpointing, use quantizacao int8 |
| **Saidas borradas** | Aumente `num_ddim_timesteps`, verifique qualidade dos dados, reduza a taxa de aprendizado |
| **Convergencia lenta** | Aumente learning rate para 2e-4, garanta dataset diverso |
| **Validacao ruim** | Use `validation_guidance: 0.0`, verifique se esta usando o scheduler correto |
| **Artefatos com poucos steps** | Normal para <4 steps; tente treinar mais tempo ou ajustar `w_min/w_max` |

---

## Dicas avancadas

1. **Treinamento multi-resolucao**: SDXL se beneficia de treinamento em multiplos aspectos:
   ```json
   "validation_resolution": "1024x1024,1280x768,768x1280,1152x896,896x1152"
   ```

2. **Treinamento progressivo**: Comece com mais timesteps, depois reduza:
   - Semana 1: Treine com `validation_num_inference_steps: 8`
   - Semana 2: Fine-tune com `validation_num_inference_steps: 4`

3. **Scheduler para inferencia**: Apos o treinamento, use o scheduler LCM:
   ```python
   from diffusers import LCMScheduler
   scheduler = LCMScheduler.from_pretrained(
       "stabilityai/stable-diffusion-xl-base-1.0",
       subfolder="scheduler"
   )
   ```

4. **Combinando com ControlNet**: LCM funciona bem com ControlNet para geracao guiada em poucos steps.

---

## Recursos adicionais

- [Paper LCM](https://arxiv.org/abs/2310.04378)
- [Docs LCM do Diffusers](https://huggingface.co/docs/diffusers/using-diffusers/inference_with_lcm)
- [Mais docs do SimpleTuner](../quickstart/SDXL.md)

---

## Proximos passos

Apos uma destilacao LCM bem-sucedida:
1. Teste o modelo com varios prompts em 4-8 steps
2. Tente LCM-LoRA em diferentes modelos base
3. Experimente ainda menos steps (2-3) para casos de uso especificos
4. Considere fine-tuning em dados de dominio especifico
