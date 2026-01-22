# Guia de Início Rápido do LongCat-Image

LongCat-Image é um modelo bilíngue (zh/en) de texto-para-imagem com 6B que usa flow matching e o encoder de texto Qwen-2.5-VL. Este guia mostra configuração, preparação de dados e a execução do primeiro treino/validação com o SimpleTuner.

---

## 1) Requisitos de hardware (o que esperar)

- VRAM: 16–24 GB cobre LoRA 1024px com `int8-quanto` ou `fp8-torchao`. Execuções full bf16 podem precisar de ~24 GB.
- RAM do sistema: ~32 GB normalmente é suficiente.
- Apple MPS: suportado para inferência/prévia; já fazemos downcast de pos-ids para float32 no MPS para evitar problemas de dtype.

---

## 2) Pré-requisitos (passo a passo)

1. Python 3.10–3.13 verificado:
   ```bash
   python --version
   ```
2. (Linux/CUDA) Em imagens recém criadas, instale o toolchain padrão:
   ```bash
   apt -y update
   apt -y install build-essential nvidia-cuda-toolkit
   ```
3. Instale o SimpleTuner com os extras corretos para seu backend:
   ```bash
   pip install "simpletuner[cuda]"   # CUDA
   pip install "simpletuner[cuda13]" # CUDA 13 / Blackwell (NVIDIA B-series GPUs)
   pip install "simpletuner[mps]"    # Apple Silicon
   pip install "simpletuner[cpu]"    # CPU-only
   ```
4. A quantização é integrada (`int8-quanto`, `int4-quanto`, `fp8-torchao`) e não precisa de installs manuais extras em setups normais.

---

## 3) Configuração do ambiente

### Web UI (mais guiada)
```bash
simpletuner server
```
Acesse http://localhost:8001 e escolha a família de modelo `longcat_image`.

### Base de CLI (config/config.json)

```jsonc
{
  "model_type": "lora",
  "model_family": "longcat_image",
  "model_flavour": "final",                // options: final, dev
  "pretrained_model_name_or_path": null,   // auto-selected from flavour; override with a local path if needed
  "base_model_precision": "int8-quanto",   // good default; fp8-torchao also works
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 16,
  "learning_rate": 1e-4,
  "validation_resolution": "1024x1024",
  "validation_guidance": 4.5,
  "validation_num_inference_steps": 30
}
```

**Padrões-chave para manter**
- O scheduler de flow matching é automático; não são necessárias flags de schedule especiais.
- Buckets de aspecto permanecem alinhados a 64 pixels; não reduza `aspect_bucket_alignment`.
- Comprimento máximo de tokens 512 (Qwen-2.5-VL).

Economias opcionais de memória (escolha conforme seu hardware):
- `--enable_group_offload --group_offload_type block_level --group_offload_blocks_per_group 1`
- Reduza `lora_rank` (4–8) e/ou use precisão base `int8-quanto`.
- Se a validação der OOM, diminua `validation_resolution` ou steps primeiro.

### Criação rápida do config (uma vez)
```bash
cp config/config.json.example config/config.json
```
Edite os campos acima (model_family, flavour, precision, paths). Aponte `output_dir` e caminhos de dataset para seu armazenamento.

### Iniciar treinamento (CLI)
```bash
simpletuner train --config config/config.json
```
ou inicie a WebUI e comece uma execução pela página Jobs após selecionar o mesmo config.

---

## 4) Pontos do dataloader (o que fornecer)

- Pastas de imagens com legendas padrão (textfile/JSON/CSV) funcionam. Inclua zh/en se quiser manter força bilíngue.
- Mantenha bordas de buckets na grade de 64px. Se treinar multi-aspect, liste várias resoluções (ex.: `1024x1024,1344x768`).
- O VAE é KL com shift+scale; caches usam o fator de escala embutido automaticamente.

---

## 5) Validação e inferência

- Guidance: 4–6 é um bom começo; deixe o prompt negativo vazio.
- Steps: ~30 para checagens de velocidade; 40–50 para melhor qualidade.
- A prévia de validação funciona sem ajustes; latentes são desempacotados antes da decodificação para evitar incompatibilidades de canais.

Exemplo (CLI validate):
```bash
simpletuner validate \
  --model_family longcat_image \
  --model_flavour final \
  --validation_resolution 1024x1024 \
  --validation_num_inference_steps 30 \
  --validation_guidance 4.5
```

---

## 6) Solução de problemas

- **Erros de float64 no MPS**: tratados internamente; mantenha seu config em float32/bf16.
- **Incompatibilidade de canais em prévias**: corrigido ao desempacotar latentes antes de decodificar (incluído no código deste guia).
- **OOM**: diminua `validation_resolution`, reduza `lora_rank`, ative group offload, ou mude para `int8-quanto` / `fp8-torchao`.
- **Tokenização lenta**: Qwen-2.5-VL limita em 512 tokens; evite prompts muito longos.

---

## 7) Seleção de flavour
- `final`: release principal (melhor qualidade).
- `dev`: checkpoint intermediário para experimentos/fine-tunes.
