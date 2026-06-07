# Quickstart de continuacao AnyFlow

Este guia mostra como continuar o objetivo de treino AnyFlow em um dataset Wan downstream. Para a visao geral da implementacao, veja [AnyFlow](/documentation/experimental/ANYFLOW.pt-BR.md).

Os checkpoints publicos NVIDIA AnyFlow sao pipelines Diffusers completos com pesos completos do transformer, nao adapters LoRA. Nao aponte `init_lora` para esses repositorios. Use `init_lora` apenas quando voce tiver um arquivo ou repositorio LoRA compativel com SimpleTuner.

## Qual checkpoint usar

Use os checkpoints AnyFlow T2V bidirecionais como transformer pretreinado:

- `nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers`
- `nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers`

Mantenha o checkpoint Wan original como fonte para text encoder, tokenizer, VAE e scheduler:

- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- `Wan-AI/Wan2.1-T2V-14B-Diffusers`

Os checkpoints FAR (`nvidia/AnyFlow-FAR-*`) usam uma arquitetura causal AnyFlow e nao sao o alvo deste quickstart do SimpleTuner.

## Config de exemplo

Comece com a config normal do Wan quickstart e altere os campos de modelo e destilacao:

```json
{
  "model_family": "wan",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_vae_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_transformer_model_name_or_path": "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_transformer_subfolder": "transformer",
  "data_backend_config": "config/wan/multidatabackend.json",
  "output_dir": "output/wan-anyflow-lora",
  "lora_rank": 32,
  "lora_alpha": 32,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "learning_rate": 0.0001,
  "max_train_steps": 1000,
  "distillation_method": "anyflow",
  "distillation_config": {
    "anyflow": {
      "target_mode": "online_teacher",
      "teacher_rollout_steps": 1,
      "r_timestep_sampler": "uniform",
      "min_interval_ratio": 0.02,
      "gate_value": 0.25,
      "deltatime_type": "r",
      "loss_weight": 1.0
    }
  }
}
```

Execute o treino no diretorio do SimpleTuner:

```bash
simpletuner train
```

O LoRA resultante continua a partir do transformer AnyFlow destilado e mantem o objetivo AnyFlow ativo durante o fine-tuning downstream.

## Se voce tiver um LoRA AnyFlow

Se um LoRA AnyFlow extraido for publicado separadamente, use o checkpoint Wan original e carregue o adapter com `init_lora`:

```json
{
  "model_family": "wan",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_vae_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "init_lora": "your-org/anyflow-wan21-1.3b-lora",
  "lora_rank": 32,
  "lora_alpha": 32,
  "distillation_method": "anyflow"
}
```

O rank LoRA e os target modules devem corresponder ao adapter publicado. Um checkpoint transformer completo nao e um valor valido para `init_lora`.

## Sobre extrair um LoRA

Extrair um LoRA de um transformer AnyFlow completo e possivel em principio, mas isso e um projeto de conversao. SimpleTuner inclui scripts experimentais:

```bash
python scripts/extract_peft_lora.py \
  Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers \
  output/anyflow-wan21-1.3b-r32.safetensors \
  --rank 32
```

Para LyCORIS/LoCon, use `scripts/extract_lycoris_adapter.py` com os mesmos argumentos e `--algo locon`.

A conversao carrega o transformer Wan base e o transformer AnyFlow correspondentes, calcula deltas dos pesos lineares, fatora esses deltas em matrizes LoRA de baixo rank, salva um adapter compativel e valida o resultado.

Isso e aproximado e depende do rank. O target padrao corresponde aos defaults PEFT Wan do SimpleTuner (`to_q,to_k,to_v,to_out.0`). Use `--target-modules all-linear` apenas se a config downstream tambem mirar os mesmos modules.

## Limites atuais

- A licenca publica dos modelos NVIDIA AnyFlow e noncommercial; confira a model card upstream antes de publicar adapters derivados.
- A validacao AnyFlow esta conectada pelo hook de scheduler do distiller para pipelines FlowMap-capable registradas. Caminhos custom ou external de validacao ainda precisam passar `r_timestep` ou `timestep_r` ao componente do modelo.
- Continuacao full-rank com online teacher ainda precisa de wiring separado de student e teacher. Por enquanto, LoRA continuation e o caminho suportado.
