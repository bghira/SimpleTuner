# Guia de Início Rápido do LongCat-Video

LongCat-Video é um modelo bilíngue (zh/en) de texto-para-vídeo e imagem-para-vídeo com 13.6B, que usa flow matching, o encoder de texto Qwen-2.5-VL e o VAE da Wan. Este guia mostra configuração, preparação de dados e a execução do primeiro treino/validação com o SimpleTuner.

---

## 1) Requisitos de hardware (o que esperar)

- Transformer 13.6B + Wan VAE: espere mais VRAM do que modelos de imagem; comece com `train_batch_size=1`, gradient checkpointing e LoRA com ranks baixos.
- RAM do sistema: mais de 32 GB ajuda para clipes multi-frame; mantenha datasets em armazenamento rápido.
- Apple MPS: suportado para prévias; codificações posicionais são convertidas para float32 automaticamente.

---

## 2) Pré-requisitos

1. Verifique Python 3.12 (o SimpleTuner vem com `.venv` por padrão):
   ```bash
   python --version
   ```
2. Instale o SimpleTuner com o backend que corresponde ao seu hardware:
   ```bash
   pip install "simpletuner[cuda]"   # NVIDIA
   pip install "simpletuner[mps]"    # Apple Silicon
   pip install "simpletuner[cpu]"    # CPU-only
   ```
3. A quantização é integrada (`int8-quanto`, `int4-quanto`, `fp8-torchao`) e não precisa de installs manuais extras em setups normais.

---

## 3) Configuração do ambiente

### Web UI
```bash
simpletuner server
```
Abra http://localhost:8001 e escolha a família de modelo `longcat_video`.

### Base de CLI (config/config.json)

```jsonc
{
  "model_type": "lora",
  "model_family": "longcat_video",
  "model_flavour": "final",
  "pretrained_model_name_or_path": null,      // auto-selected from flavour
  "base_model_precision": "bf16",             // int8-quanto/fp8-torchao also work for LoRA
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 8,
  "learning_rate": 1e-4,
  "validation_resolution": "480x832",
  "validation_num_video_frames": 93,
  "validation_num_inference_steps": 40,
  "validation_guidance": 4.0
}
```

**Padrões-chave para manter**
- Scheduler de flow matching com shift `12.0` é automático; não precisa de flags de ruído personalizadas.
- Buckets de aspecto permanecem alinhados a 64 pixels; `aspect_bucket_alignment` é forçado para 64.
- Comprimento máximo de tokens 512 (Qwen-2.5-VL); o pipeline adiciona negativos vazios automaticamente quando o CFG está ativado e nenhum prompt negativo é fornecido.
- Frames precisam satisfazer `(num_frames - 1)` divisível pelo stride temporal do VAE (padrão 4). Os 93 frames padrão já atendem isso.

Economias opcionais de VRAM:
- Reduza `lora_rank` (4–8) e use precisão base `int8-quanto`.
- Ative group offload: `--enable_group_offload --group_offload_type block_level --group_offload_blocks_per_group 1`.
- Diminua `validation_resolution`, frames ou steps primeiro se as prévias derem OOM.
- Padrões de atenção: em CUDA, LongCat-Video usa automaticamente o kernel block-sparse Triton incluído quando disponível e recua para o dispatcher padrão caso contrário. Não precisa de toggle. Se você quiser xFormers especificamente, defina `attention_implementation: "xformers"` no seu config/CLI.

### Iniciar treinamento (CLI)
```bash
simpletuner train --config config/config.json
```
Ou inicie a Web UI e envie um job com o mesmo config.

---

## 4) Orientações do dataloader

- Use datasets de vídeo com legendas; cada amostra deve fornecer frames (ou um clipe curto) e uma legenda de texto. `dataset_type: video` é tratado automaticamente via `VideoToTensor`.
- Mantenha dimensões de frame na grade de 64px (por exemplo, 480x832, buckets 720p). Altura/largura devem ser divisíveis pelo stride do VAE da Wan (16px com os ajustes embutidos) e por 64 para bucketing.
- Para execuções de imagem-para-vídeo, inclua uma imagem de condicionamento por amostra; ela é colocada no primeiro frame latente e mantida fixa durante a amostragem.
- LongCat-Video é 30 fps por padrão. Os 93 frames padrão são ~3,1 s; se você mudar a contagem de frames, mantenha `(frames - 1) % 4 == 0` e lembre que a duração escala com o fps.

### Estratégia de buckets de vídeo

Na seção `video` do seu dataset, você pode configurar como os vídeos são agrupados:
- `bucket_strategy`: `aspect_ratio` (padrão) agrupa por proporção espacial. `resolution_frames` agrupa por formato `WxH@F` (ex.: `480x832@93`) para datasets com resolução/duração mistas.
- `frame_interval`: ao usar `resolution_frames`, arredonde a contagem de frames para este intervalo (ex.: 4 para combinar com o stride temporal do VAE).

---

## 5) Validação e inferência

- Guidance: 3.5–5.0 funciona bem; prompts negativos vazios são gerados automaticamente quando CFG está ligado.
- Steps: 35–45 para checagens de qualidade; menos para prévias rápidas.
- Frames: 93 por padrão (alinha com o stride temporal do VAE de 4).
- Precisa de mais folga para prévias ou treino? Defina `musubi_blocks_to_swap` (tente 4–8) e, opcionalmente, `musubi_block_swap_device` para streamar os últimos blocos do transformer a partir da CPU durante o forward/backward. Isso adiciona overhead de transferência, mas reduz picos de VRAM.

- A validação roda a partir dos campos `validation_*` no seu config ou pela aba de prévia da WebUI após iniciar `simpletuner server`. Use esses caminhos para checagens rápidas em vez de um subcomando CLI separado.
- Para validação orientada por dataset (incluindo I2V), defina `validation_using_datasets: true` e aponte `eval_dataset_id` para sua divisão de validação. Se essa divisão estiver marcada com `is_i2v` e tiver frames de condicionamento vinculados, o pipeline mantém o primeiro frame fixo automaticamente.
- As prévias de latentes são desempacotadas antes da decodificação para evitar incompatibilidades de canais.

---

## 6) Solução de problemas

- **Erros de altura/largura**: garanta que ambas sejam divisíveis por 16 e permaneçam na grade de 64px.
- **Avisos de float64 no MPS**: tratados internamente; mantenha a precisão em bf16/float32.
- **OOM**: diminua resolução/frames de validação primeiro, reduza `lora_rank`, ative group offload, ou mude para `int8-quanto`/`fp8-torchao`.
- **Negativos em branco com CFG**: se você omitir um prompt negativo, o pipeline insere um vazio automaticamente.

---

## 7) Variações

- `final`: release principal do LongCat-Video (texto-para-vídeo + imagem-para-vídeo em um único checkpoint).
