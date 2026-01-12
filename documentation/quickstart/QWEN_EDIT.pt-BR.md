# Guia de Início Rápido do Qwen Image Edit

Este guia cobre os flavours **edit** do Qwen Image suportados pelo SimpleTuner:

- `edit-v1` – uma imagem de referência por exemplo de treino. A referência é codificada pelo encoder de texto Qwen2.5-VL e cacheada como **conditioning image embeds**.
- `edit-v2` (“edit plus”) – até três imagens de referência por amostra, codificadas em latentes do VAE em tempo real.

Ambas as variantes herdam a maior parte do [guia base do Qwen Image](./QWEN_IMAGE.md); as seções abaixo focam no que é *diferente* ao fazer fine-tuning dos checkpoints de edição.

---

## 1. Checklist de hardware

O modelo base continua com **20 B parâmetros**:

| Requisito | Recomendação |
|-------------|----------------|
| GPU VRAM    | 24 G mínimo (com quantização int8/nf4) • 40 G+ fortemente recomendado |
| Precisão   | `mixed_precision=bf16`, `base_model_precision=int8-quanto` (ou `nf4-bnb`) |
| Batch size  | Deve permanecer `train_batch_size=1`; use gradient accumulation para o batch efetivo |

Todos os pré-requisitos de treino do [guia Qwen Image](./QWEN_IMAGE.md) continuam valendo (Python ≥ 3.10, imagem CUDA 12.x, etc.).

---

## 2. Destaques de configuração

Dentro de `config/config.json`:

<details>
<summary>Ver exemplo de config</summary>

```jsonc
{
  "model_type": "lora",
  "model_family": "qwen_image",
  "model_flavour": "edit-v1",      // ou "edit-v2"
  "train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "validation_resolution": "1024x1024",
  "validation_guidance": 4.0,
  "validation_num_inference_steps": 30,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "base_model_precision": "int8-quanto",
  "quantize_via": "cpu",
  "quantize_activations": false,
  "flow_schedule_shift": 1.73,
  "data_backend_config": "config/qwen_edit/multidatabackend.json"
}
```
</details>

- EMA roda na CPU por padrão e é seguro deixar habilitado, a menos que você precise de checkpoints mais rápidos.
- `validation_resolution` deve ser reduzida (ex.: `768x768`) em placas de 24 G.
- `match_target_res` pode ser adicionado em `model_kwargs` para `edit-v2` se você quiser que as imagens de controle herdem a resolução alvo em vez do empacotamento padrão de 1 MP:

<details>
<summary>Ver exemplo de config</summary>

```jsonc
"model_kwargs": {
  "match_target_res": true
}
```
</details>

### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz viés de exposição e melhora a qualidade ao permitir que o modelo gere suas próprias entradas durante o treinamento.

> ⚠️ Esses recursos aumentam o overhead computacional do treinamento.

---

</details>

## 3. Layout do dataloader

Ambos os flavours esperam **datasets pareados**: uma imagem edit, legenda opcional e uma ou mais imagens de controle/referência que compartilham **os mesmos nomes de arquivo**.

Para detalhes dos campos, veja [`conditioning_type`](../DATALOADER.md#conditioning_type) e [`conditioning_data`](../DATALOADER.md#conditioning_data). Se você fornecer múltiplos datasets de condicionamento, escolha como eles são amostrados com `conditioning_multidataset_sampling` em [OPTIONS](../OPTIONS.md#--conditioning_multidataset_sampling).

### 3.1 edit-v1 (uma imagem de controle)

O dataset principal deve referenciar um dataset de condicionamento **e** um cache de conditioning-image-embeds:

<details>
<summary>Ver exemplo de config</summary>

```jsonc
[
  {
    "id": "qwen-edit-images",
    "type": "local",
    "instance_data_dir": "/datasets/qwen-edit/images",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "conditioning_data": ["qwen-edit-reference"],
    "conditioning_image_embeds": "qwen-edit-ref-embeds",
    "cache_dir_vae": "cache/vae/qwen-edit-images"
  },
  {
    "id": "qwen-edit-reference",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/qwen-edit/reference",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/qwen-edit-reference"
  },
  {
    "id": "qwen-edit-ref-embeds",
    "type": "local",
    "dataset_type": "conditioning_image_embeds",
    "cache_dir": "cache/conditioning_image_embeds/qwen-edit"
  }
]
```
</details>

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

- `conditioning_type=reference_strict` garante que os recortes correspondam à imagem edit. Use `reference_loose` apenas se a referência puder ter proporção diferente.
- A entrada `conditioning_image_embeds` armazena os tokens visuais do Qwen2.5-VL gerados para cada referência. Se omitido, o SimpleTuner cria um cache padrão em `cache/conditioning_image_embeds/<dataset_id>`.

### 3.2 edit-v2 (multi-controle)

Para `edit-v2`, liste cada dataset de controle em `conditioning_data`. Cada entrada fornece um frame de controle adicional. Você **não** precisa de cache de conditioning-image-embeds porque os latentes são computados em tempo real.

<details>
<summary>Ver exemplo de config</summary>

```jsonc
[
  {
    "id": "qwen-edit-plus-images",
    "type": "local",
    "instance_data_dir": "/datasets/qwen-edit-plus/images",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "conditioning_data": [
      "qwen-edit-plus-reference-a",
      "qwen-edit-plus-reference-b",
      "qwen-edit-plus-reference-c"
    ],
    "cache_dir_vae": "cache/vae/qwen-edit-plus/images"
  },
  {
    "id": "qwen-edit-plus-reference-a",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/qwen-edit-plus/reference_a",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/qwen-edit-plus/ref_a"
  },
  {
    "id": "qwen-edit-plus-reference-b",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/qwen-edit-plus/reference_b",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/qwen-edit-plus/ref_b"
  },
  {
    "id": "qwen-edit-plus-reference-c",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/qwen-edit-plus/reference_c",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/qwen-edit-plus/ref_c"
  }
]
```
</details>

Use quantos datasets de controle você tiver (1–3). O SimpleTuner mantém o alinhamento por amostra comparando os nomes de arquivo.

---

## 4. Rodando o trainer

O teste mais rápido é usar um dos presets de exemplo:

```bash
simpletuner train example=qwen_image.edit-v1-lora
# ou
simpletuner train example=qwen_image.edit-v2-lora
```

Ao executar manualmente:

```bash
simpletuner train \
  --config config/config.json \
  --data config/qwen_edit/multidatabackend.json
```

### Dicas

- Mantenha `caption_dropout_probability` em `0.0` a menos que você tenha motivo para treinar sem a instrução de edição.
- Em treinos longos, reduza a cadência de validação (`validation_step_interval`) para que validações de edição não dominem o tempo de execução.
- Checkpoints de edição do Qwen não vêm com guidance head; `validation_guidance` normalmente fica entre **3.5–4.5**.

---

## 5. Prévias de validação

Se você quiser ver a imagem de referência ao lado do output de validação, armazene seus pares edit/referência de validação em um dataset dedicado (mesmo layout do split de treino) e defina:

<details>
<summary>Ver exemplo de config</summary>

```jsonc
{
  "eval_dataset_id": "qwen-edit-val"
}
```
</details>

O SimpleTuner vai reutilizar as imagens de condicionamento desse dataset durante a validação.

---

### Troubleshooting

- **`ValueError: Control tensor list length does not match batch size`** – garanta que cada dataset de condicionamento contém arquivos para *todas* as imagens edit. Pastas vazias ou nomes de arquivo incompatíveis disparam esse erro.
- **Sem memória durante a validação** – reduza `validation_resolution`, `validation_num_inference_steps`, ou quantize mais (`base_model_precision=int2-quanto`) antes de tentar novamente.
- **Erros de cache não encontrado** ao usar `edit-v1` – confira se o campo `conditioning_image_embeds` do dataset principal corresponde a uma entrada existente de cache.

---

Você está pronto para adaptar o quickstart base do Qwen Image ao treino de edição. Para opções completas de configuração (cache do encoder de texto, amostragem multi-backend etc.), reutilize as orientações de [FLUX_KONTEXT.md](./FLUX_KONTEXT.md) – o fluxo de pareamento de dataset é o mesmo, apenas a família do modelo muda para `qwen_image`.
