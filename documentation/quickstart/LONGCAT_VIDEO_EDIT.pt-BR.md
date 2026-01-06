# Guia de Início Rápido do LongCat-Video Edit (Image-to-Video)

Este guia mostra como treinar e validar o fluxo imagem-para-vídeo do LongCat-Video. Você não precisa trocar de flavour; o mesmo checkpoint `final` cobre texto-para-vídeo e imagem-para-vídeo. A diferença vem dos seus datasets e das configurações de validação.

---

## 1) Diferenças do modelo vs LongCat-Video base

|                               | Base (text2video) | Edit / I2V |
| ----------------------------- | ----------------- | ---------- |
| Flavour                       | `final`           | `final` (mesmos pesos) |
| Condicionamento               | nenhum            | **requer frame de condicionamento** (primeiro latente mantido fixo) |
| Encoder de texto              | Qwen-2.5-VL       | Qwen-2.5-VL (mesmo) |
| Pipeline                      | TEXT2IMG          | IMG2VIDEO |
| Entradas de validação         | apenas prompt     | prompt **e** imagem de condicionamento |
| Buckets / stride              | buckets de 64px, `(frames-1)%4==0` | igual |

**Padrões principais herdados**
- Flow matching com shift `12.0`.
- Buckets de aspecto forçados a 64px.
- Encoder de texto Qwen-2.5-VL; negativos vazios adicionados automaticamente quando CFG está ligado.
- Frames padrão: 93 (satisfaz `(frames-1)%4==0`).

---

## 2) Mudanças no config (CLI/WebUI)

```jsonc
{
  "model_family": "longcat_video",
  "model_flavour": "final",
  "model_type": "lora",
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 8,
  "learning_rate": 1e-4,
  "validation_resolution": "480x832",
  "validation_num_video_frames": 93,
  "validation_num_inference_steps": 40,
  "validation_guidance": 4.0,
  "validation_using_datasets": true,
  "eval_dataset_id": "longcat-video-val"
}
```

Mantenha `aspect_bucket_alignment` em 64. O primeiro frame latente guarda a imagem inicial; não altere isso. Fique com 93 frames (já atende a regra `(frames - 1) % 4 == 0`) a menos que você tenha um forte motivo para mudar.

Setup rápido:
```bash
cp config/config.json.example config/config.json
```
Preencha `model_family`, `model_flavour`, `output_dir`, `data_backend_config` e `eval_dataset_id`. Mantenha os padrões acima a menos que saiba que precisa de valores diferentes.

Opções de atenção em CUDA:
- Em CUDA, o LongCat-Video prefere automaticamente o kernel block-sparse Triton incluído quando presente e recua para o dispatcher padrão caso contrário. Não é necessário toggle manual.
- Para forçar xFormers, defina `attention_implementation: "xformers"` no seu config/CLI.

---

## 3) Dataloader: pare clipes com frames iniciais

- Crie dois datasets:
  - **Clipes**: os vídeos alvo + legendas (instruções de edição). Marque `is_i2v: true` e defina `conditioning_data` para o ID do dataset de frame inicial.
  - **Frames iniciais**: uma imagem por clipe, com os mesmos nomes de arquivo, sem legendas.
- Mantenha ambos na grade de 64px (ex.: 480x832). Altura/largura devem ser divisíveis por 16. Contagem de frames deve atender `(frames - 1) % 4 == 0`; 93 já é válido.
- Use caches VAE separados para clipes vs frames iniciais.

Exemplo de `multidatabackend.json`:
```jsonc
[
  {
    "id": "longcat-video-train",
    "type": "local",
    "dataset_type": "video",
    "is_i2v": true,
    "instance_data_dir": "/data/video-clips",
    "caption_strategy": "textfile",
    "resolution": 480,
    "cache_dir_vae": "/cache/vae/longcat/video",
    "conditioning_data": ["longcat-video-cond"]
  },
  {
    "id": "longcat-video-cond",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/data/video-start-frames",
    "caption_strategy": null,
    "resolution": 480,
    "cache_dir_vae": "/cache/vae/longcat/video-cond"
  }
]
```

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

---

## 4) Especificidades de validação

- Adicione uma pequena divisão de validação com a mesma estrutura pareada do treino. Defina `validation_using_datasets: true` e aponte `eval_dataset_id` para essa divisão (por exemplo, `longcat-video-val`) para que a validação obtenha o frame inicial automaticamente.
- Prévias na WebUI: inicie `simpletuner server`, escolha LongCat-Video edit e envie o frame inicial + prompt.
- Guidance: 3.5–5.0 funciona; negativos vazios são preenchidos automaticamente quando CFG está ligado.
- Para prévias ou treino com pouca VRAM, defina `musubi_blocks_to_swap` (comece com 4–8) e, opcionalmente, `musubi_block_swap_device` para streamar os últimos blocos do transformer da CPU durante o forward/backward. Isso troca algum throughput por menor pico de VRAM.
- O frame de condicionamento fica fixo durante a amostragem; apenas os frames posteriores são denoised.

---

## 5) Início de treinamento (CLI)

Após o config e o dataloader estarem definidos:
```bash
simpletuner train --config config/config.json
```
Garanta que os frames de condicionamento estejam presentes nos dados de treino para que o pipeline consiga construir latentes de condicionamento.

---

## 6) Solução de problemas

- **Imagem de condicionamento ausente**: forneça um dataset de condicionamento via `conditioning_data` com nomes de arquivo correspondentes; defina `eval_dataset_id` para o ID da sua divisão de validação.
- **Erros de altura/largura**: mantenha as dimensões divisíveis por 16 e na grade de 64px.
- **Primeiro frame deriva**: reduza o guidance (3.5–4.0) ou diminua steps.
- **OOM**: diminua resolução/frames de validação, reduza `lora_rank`, ative group offload, ou use `int8-quanto`/`fp8-torchao`.
