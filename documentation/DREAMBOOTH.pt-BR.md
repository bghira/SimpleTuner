# Dreambooth (treinamento de um unico assunto)

## Contexto

O termo Dreambooth refere-se a uma tecnica desenvolvida pelo Google para injetar assuntos ao ajusta-los em um modelo usando um pequeno conjunto de imagens de alta qualidade ([paper](https://dreambooth.github.io))

No contexto de fine-tuning, o Dreambooth adiciona novas tecnicas para ajudar a evitar o colapso do modelo devido, por exemplo, a overfitting ou artefatos.

### Imagens de regularizacao

Imagens de regularizacao geralmente sao geradas pelo proprio modelo que voce esta treinando, usando um token que se pareca com sua classe.

Elas nao **precisam** ser imagens sinteticas geradas pelo modelo, mas isso possivelmente tem desempenho melhor do que usar dados reais (ex.: fotografias de pessoas reais).

Exemplo: se voce estiver treinando com imagens de um sujeito masculino, seus dados de regularizacao seriam fotografias ou amostras geradas de forma sintetica de sujeitos masculinos aleatorios.

> 🟢 Imagens de regularizacao podem ser configuradas como um dataset separado, permitindo que se misturem uniformemente aos seus dados de treinamento.

### Treinamento com token raro

Um conceito de valor duvidoso do paper original era fazer uma busca reversa no vocabulario do tokenizer do modelo para encontrar uma string "rara" com pouquissimo treinamento associado.

Desde entao, a ideia evoluiu e foi debatida, com um grupo oposto decidindo treinar com o nome de uma celebridade que seja parecido o suficiente, ja que isso requer menos compute.

> 🟡 O treinamento com token raro e suportado no SimpleTuner, mas nao ha ferramenta disponivel para ajudar voce a encontrar um.

### Perda de preservacao de prior

O modelo contem algo chamado "prior" que, em teoria, poderia ser preservado durante o treinamento Dreambooth. Em experimentos com Stable Diffusion, no entanto, isso nao pareceu ajudar - o modelo apenas overfita no proprio conhecimento.

> 🟢 ([#1031](https://github.com/bghira/SimpleTuner/issues/1031)) A perda de preservacao de prior e suportada no SimpleTuner ao treinar adaptadores LyCORIS definindo `is_regularisation_data` naquele dataset.

### Perda com mascara

Mascaras de imagem podem ser definidas em pares com dados de imagem. As partes escuras da mascara fazem com que os calculos de loss ignorem essas partes da imagem.

Existe um [script](/scripts/toolkit/datasets/masked_loss/generate_dataset_masks.py) de exemplo para gerar essas mascaras, dado um input_dir e output_dir:

```bash
python generate_dataset_masks.py --input_dir /images/input \
                      --output_dir /images/output \
                      --text_input "person"
```

No entanto, isso nao possui funcionalidades avancadas como desfoque de padding da mascara.

Ao definir seu dataset de mascaras de imagem:

- Cada imagem deve ter uma mascara. Use uma imagem totalmente branca se voce nao quiser mascarar.
- Defina `dataset_type=conditioning` na pasta de dados de condicionamento (mascara)
- Defina `conditioning_type=mask` no seu dataset de mascara
- Defina `conditioning_data=` para o `id` do dataset de condicionamento no seu dataset de imagens

```json
[
    {
        "id": "dreambooth-data",
        "type": "local",
        "dataset_type": "image",
        "conditioning_data": "dreambooth-conditioning",
        "instance_data_dir": "/training/datasets/test_datasets/dreambooth",
        "cache_dir_vae": "/training/cache/vae/sdxl/dreambooth-data",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "an dreambooth",
        "metadata_backend": "discovery",
        "resolution": 1024,
        "minimum_image_size": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution_type": "pixel_area"
    },
    {
        "id": "dreambooth-conditioning",
        "type": "local",
        "dataset_type": "conditioning",
        "instance_data_dir": "/training/datasets/test_datasets/dreambooth_mask",
        "resolution": 1024,
        "minimum_image_size": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution_type": "pixel_area",
        "conditioning_type": "mask"
    },
    {
        "id": "an example backend for text embeds.",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "/training/cache/text/sdxl-base/masked_loss"
    }
]
```

## Configuracao

Seguir o [tutorial](TUTORIAL.md) e obrigatorio antes de voce continuar para a configuracao especifica do Dreambooth.

Para ajuste do DeepFloyd, e recomendado visitar [esta pagina](DEEPFLOYD.md) para dicas especificas relacionadas a configuracao desse modelo.

### Treinamento de modelo quantizado (apenas LoRA/LyCORIS)

Testado em sistemas Apple e NVIDIA, o Hugging Face Optimum-Quanto pode ser usado para reduzir a precisao e os requisitos de VRAM.

Dentro da sua venv do SimpleTuner:

```bash
pip install optimum-quanto
```

Os niveis de precisao disponiveis dependem do seu hardware e de suas capacidades.

- int2-quanto, int4-quanto, **int8-quanto** (recomendado)
- fp8-quanto, fp8-torchao (apenas para CUDA >= 8.9, ex.: 4090 ou H100)
- nf4-bnb (necessario para usuarios com pouca VRAM)

Dentro do seu config.json, os seguintes valores devem ser modificados ou adicionados:
```json
{
    "base_model_precision": "int8-quanto",
    "text_encoder_1_precision": "no_change",
    "text_encoder_2_precision": "no_change",
    "text_encoder_3_precision": "no_change"
}
```

Dentro do nosso config de dataloader `multidatabackend-dreambooth.json`, ele ficara algo assim:

```json
[
    {
        "id": "subjectname-data-512px",
        "type": "local",
        "instance_data_dir": "/training/datasets/subjectname",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "subjectname",
        "cache_dir_vae": "/training/vae_cache/subjectname",
        "repeats": 100,
        "crop": false,
        "resolution": 512,
        "resolution_type": "pixel_area",
        "minimum_image_size": 192
    },
    {
        "id": "subjectname-data-1024px",
        "type": "local",
        "instance_data_dir": "/training/datasets/subjectname",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "subjectname",
        "cache_dir_vae": "/training/vae_cache/subjectname-1024px",
        "repeats": 100,
        "crop": false,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "minimum_image_size": 768
    },
    {
        "id": "regularisation-data",
        "type": "local",
        "instance_data_dir": "/training/datasets/regularisation",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "a picture of a man",
        "cache_dir_vae": "/training/vae_cache/regularisation",
        "repeats": 0,
        "resolution": 512,
        "resolution_type": "pixel_area",
        "minimum_image_size": 192,
        "is_regularisation_data": true
    },
    {
        "id": "regularisation-data-1024px",
        "type": "local",
        "instance_data_dir": "/training/datasets/regularisation",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "a picture of a man",
        "cache_dir_vae": "/training/vae_cache/regularisation-1024px",
        "repeats": 0,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "minimum_image_size": 768,
        "is_regularisation_data": true
    },
    {
        "id": "textembeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/training/text_cache/sdxl_base"
    }
]
```

Alguns valores-chave foram ajustados para facilitar o treinamento de um unico assunto:

- Agora temos dois datasets configurados duas vezes, totalizando quatro datasets. Dados de regularizacao sao opcionais, e o treinamento pode funcionar melhor sem eles. Voce pode remover esse dataset da lista se desejar.
- A resolucao esta definida em 512px e 1024px com bucketing misto, o que pode ajudar a melhorar a velocidade de treinamento e a convergencia
- O tamanho minimo de imagem esta definido como 192px ou 768px, o que nos permite ampliar algumas imagens menores, o que pode ser necessario para datasets com algumas imagens importantes porem de baixa resolucao.
- `caption_strategy` agora e `instanceprompt`, o que significa que usaremos o valor `instance_prompt` para cada imagem do dataset como sua caption.
  - **Nota:** Usar o instance prompt e o metodo tradicional do treinamento Dreambooth, mas captions curtas podem funcionar melhor. Se voce perceber que o modelo falha em generalizar, pode valer a pena tentar usar captions.

### Consideracoes sobre o dataset de regularizacao

Para um dataset de regularizacao:

- Defina `repeats` bem alto no seu assunto Dreambooth para que a contagem de imagens nos dados do Dreambooth seja multiplicada `repeats` vezes e supere a contagem de imagens do seu conjunto de regularizacao
  - Se seu conjunto de regularizacao tem 1000 imagens e voce tem 10 imagens no seu conjunto de treinamento, voce vai querer um valor de repeats de pelo menos 100 para ter resultados rapidos
- `minimum_image_size` foi aumentado para garantir que nao introduzimos muitos artefatos de baixa qualidade
- Da mesma forma, usar captions mais descritivas pode ajudar a evitar o esquecimento. Mudar de `instanceprompt` para `textfile` ou outras estrategias exigira criar arquivos `.txt` para cada imagem.
- Quando `is_regularisation_data` (ou 🇺🇸 `is_regularization_data` com z, para usuarios americanos) esta definido, os dados desse conjunto serao alimentados no modelo base para obter uma predicao que pode ser usada como alvo de loss para o modelo aluno LyCORIS.
  - Nota: atualmente isso so funciona em um adaptador LyCORIS.

## Selecionando um instance prompt

Como mencionado anteriormente, o foco original do Dreambooth era a selecao de tokens raros para treinar.

Alternativamente, pode-se usar o nome real do sujeito ou o nome de uma celebridade "parecida o suficiente".

Depois de varios experimentos de treinamento, parece que uma celebridade "parecida o suficiente" e a melhor escolha, especialmente se pedir o nome real da pessoa no prompt acabar parecendo diferente.

# Scheduled Sampling (Rollout)

Ao treinar em datasets pequenos como no Dreambooth, modelos podem rapidamente overfitar para o ruido "perfeito" adicionado durante o treinamento. Isso leva a **exposure bias**: o modelo aprende a denoisar entradas perfeitas, mas falha quando encontra suas proprias saidas levemente imperfeitas durante a inferencia.

**Scheduled Sampling (Rollout)** resolve isso ao ocasionalmente deixar o modelo gerar seus proprios latentes ruidosos por alguns passos durante o loop de treinamento. Em vez de treinar em ruido gaussiano puro + sinal, ele treina em amostras de "rollout" que contem os proprios erros anteriores do modelo. Isso ensina o modelo a se corrigir, levando a uma geracao de assunto mais robusta e estavel.

> 🟢 Este recurso e experimental, mas altamente recomendado para datasets pequenos onde overfitting ou "frying" e comum.
> ⚠️ Habilitar rollout aumenta os requisitos de compute, pois o modelo deve realizar etapas extras de inferencia durante o loop de treinamento.

Para habilitar, adicione estas chaves ao seu `config.json`:

```json
{
  "scheduled_sampling_max_step_offset": 10,
  "scheduled_sampling_probability": 1.0,
  "scheduled_sampling_ramp_steps": 1000,
  "scheduled_sampling_sampler": "unipc"
}
```

*   `scheduled_sampling_max_step_offset`: Quantos passos gerar. Um valor pequeno (ex.: 5-10) geralmente e suficiente.
*   `scheduled_sampling_probability`: Com que frequencia aplicar essa tecnica (0.0 a 1.0).
*   `scheduled_sampling_ramp_steps`: Aumenta a probabilidade nos primeiros N passos para evitar desestabilizar o treinamento inicial.

# Exponential moving average (EMA)

Um segundo modelo pode ser treinado em paralelo ao seu checkpoint, quase de graca - apenas a memoria do sistema resultante (por padrao) e consumida, e nao mais VRAM.

Aplicar `use_ema=true` no seu arquivo de configuracao habilitara esse recurso.

# Rastreamento de CLIP score

Se voce quiser habilitar avaliacoes para pontuar o desempenho do modelo, veja [este documento](evaluation/CLIP_SCORES.md) para informacoes sobre configuracao e interpretacao dos CLIP scores.

# Perda de avaliacao estavel

Se voce quiser usar loss MSE estavel para pontuar o desempenho do modelo, veja [este documento](evaluation/EVAL_LOSS.md) para informacoes sobre configuracao e interpretacao da loss de avaliacao.

# Previews de validacao

O SimpleTuner suporta streaming de previews intermediarios de validacao durante a geracao usando modelos Tiny AutoEncoder. Esse recurso permite ver suas imagens de validacao sendo geradas passo a passo em tempo real via callbacks de webhook, em vez de esperar pela geracao completa.

## Habilitando previews de validacao

Para habilitar previews de validacao, adicione o seguinte ao seu `config.json`:

```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```

## Requisitos

- Familia de modelos com suporte a Tiny AutoEncoder (Flux, SDXL, SD3, etc.)
- Configuracao de webhook para receber as imagens de preview
- Validacao deve estar habilitada (`validation_disable` nao pode estar definido como true)

## Opcoes de configuracao

- `--validation_preview`: Habilita/desabilita o recurso de preview (padrao: false)
- `--validation_preview_steps`: Controla com que frequencia os previews sao decodificados durante a amostragem (padrao: 1)
  - Defina como 1 para receber um preview em cada passo de amostragem
  - Defina como valores maiores (ex.: 3 ou 5) para reduzir o overhead da decodificacao do Tiny AutoEncoder

## Exemplo

Com `validation_num_inference_steps=20` e `validation_preview_steps=5`, voce recebera imagens de preview nos passos 5, 10, 15 e 20 durante cada geracao de validacao.

# Ajuste do refiner

Se voce e fa do refiner do SDXL, pode achar que ele faz com que suas geracoes "estraguem" os resultados do seu modelo treinado com Dreambooth.

O SimpleTuner suporta o treinamento do refiner do SDXL usando LoRA e full rank.

Isso exige algumas consideracoes:
- As imagens devem ser puramente de alta qualidade
- Os embeddings de texto nao podem ser compartilhados com o modelo base
- Os embeddings do VAE **podem** ser compartilhados com o modelo base

Voce precisara atualizar `cache_dir` na configuracao do seu dataloader, `multidatabackend.json`:

```json
[
    {
        "id": "textembeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/training/text_cache/sdxl_refiner"
    }
]
```

Se voce quiser direcionar um score estetico especifico com seus dados, voce pode adicionar isto ao `config/config.json`:

```bash
"--data_aesthetic_score": 5.6,
```

Atualize **5.6** para o score que voce gostaria de atingir. O padrao e **7.0**.

> ⚠️ Ao treinar o refiner do SDXL, seus prompts de validacao serao ignorados. Em vez disso, imagens aleatorias dos seus datasets serao refinadas.
