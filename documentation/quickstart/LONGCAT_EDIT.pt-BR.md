# Guia de Início Rápido do LongCat-Image Edit

Esta é a variante de edição/img2img do LongCat-Image. Leia [LONGCAT_IMAGE.md](../quickstart/LONGCAT_IMAGE.md) primeiro; este arquivo só lista o que muda para o flavour edit.

---

## 1) Diferenças do modelo vs LongCat-Image base

|                               | Base (text2img) | Edit |
| ----------------------------- | --------------- | ---- |
| Flavour                       | `final` / `dev` | `edit` |
| Condicionamento               | nenhum          | **requer latentes de condicionamento (imagem de referência)** |
| Encoder de texto              | Qwen-2.5-VL     | Qwen-2.5-VL **com contexto de visão** (a codificação do prompt precisa da imagem de referência) |
| Pipeline                      | TEXT2IMG        | IMG2IMG/EDIT |
| Entradas de validação         | apenas prompt   | prompt **e** referência |

---

## 2) Mudanças no config (CLI/WebUI)

```jsonc
{
  "model_type": "lora",
  "model_family": "longcat_image",
  "model_flavour": "edit",
  "base_model_precision": "int8-quanto",      // fp8-torchao also fine; helps fit 16–24 GB
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "learning_rate": 5e-5,
  "validation_guidance": 4.5,
  "validation_num_inference_steps": 40,
  "validation_resolution": "768x768"
}
```

Mantenha `aspect_bucket_alignment` em 64. Não desative latentes de condicionamento; o pipeline de edição espera por eles.

Criação rápida do config:
```bash
cp config/config.json.example config/config.json
```
Depois defina `model_family`, `model_flavour`, caminhos de dataset e output_dir.

---

## 3) Dataloader: pares de edição + referência

Use dois datasets alinhados: **imagens de edição** (legenda = instrução de edição) e **imagens de referência**. O `conditioning_data` do dataset de edição deve apontar para o ID do dataset de referência. Os nomes dos arquivos precisam corresponder 1-para-1.

```jsonc
[
  {
    "id": "edit-images",
    "type": "local",
    "instance_data_dir": "/data/edits",
    "caption_strategy": "textfile",
    "resolution": 768,
    "cache_dir_vae": "/cache/vae/longcat/edit",
    "conditioning_data": ["ref-images"]
  },
  {
    "id": "ref-images",
    "type": "local",
    "instance_data_dir": "/data/refs",
    "caption_strategy": null,
    "resolution": 768,
    "cache_dir_vae": "/cache/vae/longcat/ref"
  }
]
```

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

Notas:
- Buckets de aspecto: mantenha na grade de 64px.
- Legendas das referências são opcionais; se presentes, substituem as legendas de edição (geralmente indesejado).
- Caches VAE de edição e referência devem ser caminhos separados.
- Se você vir cache misses ou erros de shape, limpe os caches VAE de ambos os datasets e regenere.

---

## 4) Especificidades de validação

- A validação precisa de imagens de referência para produzir latentes de condicionamento. Aponte a divisão de validação de `edit-images` para `ref-images` via `conditioning_data`.
- Guidance: 4–6 funciona bem; mantenha o prompt negativo vazio.
- Callbacks de prévia são suportados; latentes são desempacotados para os decoders automaticamente.
- Se a validação falhar por falta de latentes de condicionamento, verifique se o dataloader de validação inclui entradas de edição e referência com nomes de arquivo correspondentes.

---

## 5) Comandos de inferência / validação

Validação rápida via CLI:
```bash
simpletuner validate \
  --model_family longcat_image \
  --model_flavour edit \
  --validation_resolution 768x768 \
  --validation_guidance 4.5 \
  --validation_num_inference_steps 40
```

WebUI: escolha o pipeline **Edit**, forneça a imagem de origem e a instrução de edição.

---

## 6) Início de treinamento (CLI)

Após o config e o dataloader estarem definidos:
```bash
simpletuner train --config config/config.json
```
Garanta que o dataset de referência esteja presente durante o treino para que latentes de condicionamento possam ser computados ou carregados do cache.

---

## 7) Solução de problemas

- **Latentes de condicionamento ausentes**: garanta que o dataset de referência esteja ligado via `conditioning_data` e que os nomes dos arquivos correspondam.
- **Erros de dtype no MPS**: o pipeline faz downgrade automático de pos-ids para float32 no MPS; mantenha o restante em float32/bf16.
- **Incompatibilidade de canais em prévias**: as prévias fazem un-patchify dos latentes antes da decodificação (mantenha esta versão do SimpleTuner).
- **OOM durante edição**: diminua resolução/steps de validação, reduza `lora_rank`, ative group offload e prefira `int8-quanto`/`fp8-torchao`.
