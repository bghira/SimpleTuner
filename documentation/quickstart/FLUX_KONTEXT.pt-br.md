# Mini guia de Início Rápido do Kontext [dev]

> Nota: Kontext compartilha 90% do fluxo de treino com Flux, então este arquivo só lista o que *difere*. Quando um passo **não** é mencionado aqui, siga as [instruções](../quickstart/FLUX.md).

---

## 1. Visão geral do modelo

|                                                  | Flux-dev             | Kontext-dev                                |
| ------------------------------------------------ | -------------------- | ------------------------------------------ |
| Licença                                          | Não comercial        | Não comercial                              |
| Guidance                                         | Destilado (CFG ~ 1)  | Destilado (CFG ~ 1)                        |
| Variantes disponíveis                            | *dev*, schnell, [pro] | *dev*, [pro, max]                          |
| Comprimento da sequência T5                      | 512 dev, 256 schnell | 512 dev                                    |
| Tempo típico de inferência 1024 px<br>(4090 @ CFG 1) | ~20 s              | **~80 s**                                  |
| VRAM para LoRA 1024 px @ int8-quanto             | 18 G                 | **24 G**                                   |

Kontext mantém o backbone transformer do Flux, mas introduz **condicionamento por referência pareada**.

Há dois modos `conditioning_type` disponíveis para o Kontext:

* `conditioning_type=reference_loose` (✅ estável) - a referência pode diferir em proporção/tamanho do edit.
  - Ambos os datasets são escaneados para metadados, agrupados por buckets de aspecto e recortados de forma independente, o que pode aumentar bastante o tempo de inicialização.
  - Isso pode ser um problema em setups onde você quer garantir o alinhamento da imagem editada e da referência, como em um dataloader que usa uma única imagem por nome de arquivo.
* `conditioning_type=reference_strict` (✅ estável) - a referência é pré-transformada exatamente como o recorte do edit.
  - É assim que você deve configurar seus datasets se precisa de alinhamento perfeito entre recortes/buckets de aspecto das imagens de edit e referência.
  - Antes exigia `--vae_cache_ondemand` e mais VRAM, mas não exige mais.
  - Duplica os metadados de recorte/bucket do dataset de origem na inicialização, então você não precisa fazer isso manualmente.

Para definições de campo, veja [`conditioning_type`](../DATALOADER.md#conditioning_type) e [`conditioning_data`](../DATALOADER.md#conditioning_data). Para controlar como múltiplos conjuntos de condicionamento são amostrados, use `conditioning_multidataset_sampling` conforme descrito em [OPTIONS](../OPTIONS.md#--conditioning_multidataset_sampling).

---

## 2. Requisitos de hardware

* **RAM do sistema**: a quantização ainda precisa de 50 GB.
* **GPU**: 3090 (24 G) é o mínimo realista para treino 1024 px **com int8-quanto**.
  * Sistemas Hopper H100/H200 com Flash Attention 3 podem habilitar `--fuse_qkv_projections` para acelerar bastante o treinamento.
  * Se você treinar em 512 px, dá para caber em uma placa de 12 G, mas espere batches lentos (o comprimento da sequência continua grande).

---

## 3. Diferenças rápidas de configuração

Abaixo está o menor conjunto de mudanças que você precisa em `config/config.json` comparado com a configuração típica de treino do Flux.

<details>
<summary>Ver exemplo de config</summary>

```jsonc
{
  "model_family": "flux",
  "model_flavour": "kontext",                      // <-- change this from "dev" to "kontext"
  "base_model_precision": "int8-quanto",           // fits on 24 G at 1024 px
  "gradient_checkpointing": true,
  "fuse_qkv_projections": false,                    // <-- use this to speed up training on Hopper H100/H200 systems. WARNING: requires flash-attn manually installed.
  "lora_rank": 16,
  "learning_rate": 1e-5,
  "optimizer": "optimi-lion",                      // <-- use Lion for faster results, and adamw_bf16 for slower but possibly more stable results.
  "max_train_steps": 10000,
  "validation_guidance": 2.5,                       // <-- kontext really does best with a guidance value of 2.5
  "validation_resolution": "1024x1024",
  "conditioning_multidataset_sampling": "random"   // <-- setting this to "combined" when you have two conditioning datasets defined will show them simultaneously instead of switching between them.
}
```
</details>

### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz viés de exposição e melhora a qualidade ao permitir que o modelo gere suas próprias entradas durante o treinamento.

> ⚠️ Esses recursos aumentam o overhead computacional do treinamento.

</details>

### Trecho do dataloader (multi-data-backend)

Se você curou manualmente um dataset de pares de imagens, pode configurá-lo usando dois diretórios separados: um para as imagens editadas e outro para as imagens de referência.

O campo `conditioning_data` no dataset de edição deve apontar para o `id` do dataset de referência.

<details>
<summary>Ver exemplo de config</summary>

```jsonc
[
  {
    "id": "my-edited-images",
    "type": "local",
    "cache_dir_vae": "/cache/vae/flux/kontext/edited-images",   // <-- where VAE outputs are stored
    "instance_data_dir": "/datasets/edited-images",            // <-- use absolute paths
    "conditioning_data": [
      "my-reference-images"                                     // <-- this should be your "id" of the reference set
                                                                // you could specify a second set to alternate between or combine them, e.g. ["reference-images", "reference-images2"]
    ],
    "resolution": 1024,
    "caption_strategy": "textfile"                             // <-- these captions should contain the edit instructions
  },
  {
    "id": "my-reference-images",
    "type": "local",
    "cache_dir_vae": "/cache/vae/flux/kontext/ref-images",      // <-- where VAE outputs are stored. must be different from other dataset VAE paths.
    "instance_data_dir": "/datasets/reference-images",         // <-- use absolute paths
    "conditioning_type": "reference_strict",                   // <-- if this is set to reference_loose, the images are cropped independently of the edit images
    "resolution": 1024,
    "caption_strategy": null,                                   // <-- no captions needed for references, but if available, will be used INSTEAD of the edit captions
                                                                // NOTE: you cannot define separate conditioning captions when using conditioning_multidataset_sampling=combined.
                                                                // Only the edit datasets' captions will be used.
  }
]
```
</details>

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

*Cada imagem editada **deve** ter nomes de arquivo e extensões correspondentes 1-para-1 em ambas as pastas de dataset. O SimpleTuner vai anexar automaticamente a embedding de referência ao condicionamento da edição.*

Há um exemplo preparado do dataset demo [Kontext Max derived](https://huggingface.co/datasets/terminusresearch/KontextMax-Edit-smol), que contém imagens de referência e edição junto com seus textfiles de legenda, disponível para consulta para entender melhor como configurá-lo.

### Configurando uma divisão de validação dedicada

Aqui está um exemplo de configuração que usa um conjunto de treino com 200.000 amostras e um conjunto de validação com apenas algumas.

No seu `config.json` você vai querer adicionar:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "eval_dataset_id": "edited-images",
}
```
</details>

Para o seu `multidatabackend.json`, `edited-images` e `reference-images` devem conter dados de validação com o mesmo layout de uma divisão de treino usual.

<details>
<summary>Ver exemplo de config</summary>

```json
[
    {
        "id": "edited-images",
        "disabled": false,
        "type": "local",
        "instance_data_dir": "/datasets/edit/edited-images",
        "minimum_image_size": 1024,
        "maximum_image_size": 1536,
        "target_downsample_size": 1024,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "caption_strategy": "textfile",
        "cache_dir_vae": "cache/vae/flux-edit",
        "vae_cache_clear_each_epoch": false,
        "conditioning_data": ["reference-images"]
    },
    {
        "id": "reference-images",
        "disabled": false,
        "type": "local",
        "instance_data_dir": "/datasets/edit/reference-images",
        "minimum_image_size": 1024,
        "maximum_image_size": 1536,
        "target_downsample_size": 1024,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "caption_strategy": null,
        "cache_dir_vae": "cache/vae/flux-ref",
        "vae_cache_clear_each_epoch": false,
        "conditioning_type": "reference_strict"
    },
    {
        "id": "subjects200k-left",
        "disabled": false,
        "type": "huggingface",
        "dataset_name": "Yuanshi/Subjects200K",
        "caption_strategy": "huggingface",
        "metadata_backend": "huggingface",
        "resolution": 512,
        "resolution_type": "pixel_area",
        "conditioning_data": ["subjects200k-right"],
        "huggingface": {
            "caption_column": "description.description_0",
            "image_column": "image",
            "composite_image_config": {
                "enabled": true,
                "image_count": 2,
                "select_index": 0
            }
        }
    },
    {
        "id": "subjects200k-right",
        "disabled": false,
        "type": "huggingface",
        "dataset_type": "conditioning",
        "conditioning_type": "reference_strict",
        "source_dataset_id": "subjects200k-left",
        "dataset_name": "Yuanshi/Subjects200K",
        "caption_strategy": "huggingface",
        "metadata_backend": "huggingface",
        "resolution": 512,
        "resolution_type": "pixel_area",
        "huggingface": {
            "caption_column": "description.description_1",
            "image_column": "image",
            "composite_image_config": {
                "enabled": true,
                "image_count": 2,
                "select_index": 1
            }
        }
    },
    {
        "id": "text-embed-cache",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/flux"
    }
]
```
</details>

### Geração automática de pares referência-edição

Se você não tem pares pré-existentes de referência e edição, o SimpleTuner pode gerá-los automaticamente a partir de um único dataset. Isso é especialmente útil para treinar modelos para:
- Aprimoramento de imagem / super-resolução
- Remoção de artefatos JPEG
- Desfoque reverso
- Outras tarefas de restauração

#### Exemplo: dataset de treino para desfoque reverso

<details>
<summary>Ver exemplo de config</summary>

```jsonc
[
  {
    "id": "high-quality-images",
    "type": "local",
    "instance_data_dir": "/path/to/sharp-images",
    "resolution": 1024,
    "caption_strategy": "textfile",
    "conditioning": [
      {
        "type": "superresolution",
        "blur_radius": 3.0,
        "blur_type": "gaussian",
        "add_noise": true,
        "noise_level": 0.02,
        "captions": ["enhance sharpness", "deblur", "increase clarity", "sharpen image"]
      }
    ]
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/kontext"
  }
]
```
</details>

Esta configuração vai:
1. Criar versões borradas (estas se tornam as imagens de "referência") a partir das imagens nítidas de alta qualidade
2. Usar as imagens nítidas originais como alvo de perda do treino
3. Treinar o Kontext para melhorar/desfocar a imagem de referência de baixa qualidade

> **NOTA**: Você não pode definir `captions` em um dataset de condicionamento quando usa `conditioning_multidataset_sampling=combined`. As legendas do dataset de edição serão usadas no lugar.

#### Exemplo: remoção de artefatos JPEG

<details>
<summary>Ver exemplo de config</summary>

```jsonc
[
  {
    "id": "pristine-images",
    "type": "local",
    "instance_data_dir": "/path/to/pristine-images",
    "resolution": 1024,
    "caption_strategy": "textfile",
    "conditioning": [
      {
        "type": "jpeg_artifacts",
        "quality_mode": "range",
        "quality_range": [10, 30],
        "compression_rounds": 2,
        "captions": ["remove compression artifacts", "restore quality", "fix jpeg artifacts"]
      }
    ]
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/kontext"
  }
]
```
</details>

#### Notas importantes

1. **A geração acontece na inicialização**: As versões degradadas são criadas automaticamente quando o treinamento começa
2. **Cache**: As imagens geradas são salvas, então execuções futuras não vão regenerá-las
3. **Estratégia de legenda**: O campo `captions` no config de condicionamento fornece prompts específicos de tarefa que funcionam melhor do que descrições genéricas de imagem
4. **Desempenho**: Esses geradores baseados em CPU (blur, JPEG) são rápidos e usam múltiplos processos
5. **Espaço em disco**: Garanta espaço suficiente para as imagens geradas, pois elas podem ser grandes. Infelizmente, ainda não há a capacidade de criá-las sob demanda.

Para mais tipos de condicionamento e configurações avançadas, veja a [documentação do ControlNet](../CONTROLNET.md).

---

## 4. Dicas de treino específicas do Kontext

1. **Sequências mais longas -> passos mais lentos.** Espere ~0,4 it/s em uma única 4090 em 1024 px, LoRA rank 1, bf16 + int8.
2. **Explore até encontrar as configurações corretas.** Não há muito conhecimento sobre o fine-tuning do Kontext; por segurança, fique em `1e-5` (Lion) ou `5e-4` (AdamW).
3. **Observe picos de VRAM durante o cache de VAE.** Se houver OOM, adicione `--offload_during_startup=true`, reduza sua `resolution` ou habilite VAE tiling via `config.json`.
4. **Você pode treiná-lo sem imagens de referência, mas não atualmente via SimpleTuner.** Hoje, algumas coisas são meio hardcoded para exigir imagens condicionais, mas você pode fornecer datasets normais ao lado dos pares de edição para permitir que ele aprenda sujeitos e semelhança.
5. **Re-destilação de guidance.** Como o Flux-dev, o Kontext-dev é destilado por CFG; se você precisa de diversidade, treine novamente com `validation_guidance_real > 1` e use um nó de Adaptive-Guidance na inferência. Isso vai levar muito mais tempo para convergir e exigirá uma LoRA de rank alto ou uma Lycoris LoKr para funcionar.
6. **Treino full-rank provavelmente é perda de tempo.** Kontext foi projetado para treino com rank baixo, e o treino full-rank provavelmente não dará resultados melhores do que uma Lycoris LoKr, que normalmente supera uma LoRA padrão com menos esforço na busca dos melhores parâmetros. Se quiser tentar mesmo assim, você terá que usar DeepSpeed.
7. **Você pode usar duas ou mais imagens de referência no treino.** Por exemplo, se você tem imagens sujeito-sujeito-cena para inserir dois sujeitos em uma mesma cena, forneça todas as imagens relevantes como referências. Apenas garanta que os nomes de arquivo correspondam em todas as pastas.

---

## 5. Armadilhas na inferência

- Combine os níveis de precisão de treino e inferência; treino em int8 funciona melhor com inferência em int8 e assim por diante.
- Vai ser muito lento porque duas imagens passam pelo sistema ao mesmo tempo. Espere 80 s por edição 1024 px em uma 4090.

---

## 6. Tabela rápida de troubleshooting

| Sintoma                               | Causa provável                 | Correção rápida                                         |
| ------------------------------------- | ------------------------------ | ------------------------------------------------------ |
| OOM durante a quantização             | Pouca RAM do **sistema**       | Use `quantize_via=cpu`                                 |
| Imagem de referência ignorada / sem edição aplicada | Dataloader emparelhado errado | Garanta nomes de arquivo idênticos e o campo `conditioning_data` |
| Artefatos de grade quadrada           | Edições de baixa qualidade dominam | Faça um dataset de maior qualidade, reduza LR, evite Lion |

---

## 7. Leitura adicional

Para opções avançadas de ajuste (LoKr, quantização NF4, DeepSpeed etc.), consulte [o quickstart original do Flux](../quickstart/FLUX.md) - todos os flags funcionam da mesma forma, a menos que indicado acima.
