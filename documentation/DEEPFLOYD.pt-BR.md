# DeepFloyd IF

> 🤷🏽‍♂️ Treinar DeepFloyd requer pelo menos 24G de VRAM para uma LoRA. Este guia foca no modelo base de 400M parametros, embora a variante XL de 4.3B possa ser treinada usando as mesmas diretrizes.

## Contexto

Na primavera de 2023, a StabilityAI lançou um modelo de difusao em cascata por pixel chamado DeepFloyd.
![](https://tripleback.net/public/deepfloyd.png)

Comparando rapidamente com Stable Diffusion XL:
- Codificador de texto
  - SDXL usa dois codificadores CLIP, "OpenCLIP G/14" e "OpenAI CLIP-L/14"
  - DeepFloyd usa um unico modelo transformer auto-supervisionado, o T5 XXL do Google
- Contagem de parametros
  - DeepFloyd vem em varias densidades: 400M, 900M e 4.3B de parametros. Cada unidade maior e sucessivamente mais cara de treinar.
  - SDXL tem apenas um, ~3B de parametros.
  - O codificador de texto do DeepFloyd tem 11B de parametros sozinho, tornando a configuracao mais pesada em torno de 15.3B de parametros.
- Quantidade de modelos
  - DeepFloyd roda em **tres** etapas: 64px -> 256px -> 1024px
    - Cada etapa completa totalmente seu objetivo de denoise
  - SDXL roda em **duas** etapas, incluindo seu refiner, de 1024px -> 1024px
    - Cada etapa completa apenas parcialmente seu objetivo de denoise
- Design
  - Os tres modelos do DeepFloyd aumentam a resolucao e os detalhes finos
  - Os dois modelos do SDXL cuidam dos detalhes finos e da composicao

Para ambos os modelos, a primeira etapa define a maior parte da composicao da imagem (onde itens grandes / sombras aparecem).

## Avaliacao do modelo

Aqui esta o que voce pode esperar ao usar DeepFloyd para treinamento ou inferencia.

### Estetica

Quando comparado ao SDXL ou Stable Diffusion 1.x/2.x, a estetica do DeepFloyd fica em algum lugar entre Stable Diffusion 2.x e SDXL.


### Desvantagens

Este nao e um modelo popular, por varios motivos:

- O requisito de VRAM para inferencia e mais pesado do que outros modelos
- Os requisitos de VRAM para treinamento superam outros modelos
  - Um ajuste completo do u-net exigindo mais de 48G de VRAM
  - LoRA em rank-32, batch-4 precisa de ~24G de VRAM
  - Os objetos de cache de embeddings de texto sao ENORMES (varios Megabytes cada, vs centenas de Kilobytes para os embeddings CLIP duplos do SDXL)
  - Os objetos de cache de embeddings de texto sao LENTOS PARA CRIAR, cerca de 9-10 por segundo atualmente em uma A6000 nao-Ada.
- A estetica padrao e pior do que outros modelos (como tentar treinar SD 1.5 puro)
- Ha **tres** modelos para ajustar ou carregar no sistema durante a inferencia (quatro se contar o codificador de texto)
- As promessas da StabilityAI nao corresponderam a realidade do uso do modelo (superestimado)
- A licenca do DeepFloyd-IF e restritiva para uso comercial.
  - Isso nao impactou os pesos do NovelAI, que foram vazados de forma ilicita. A natureza da licenca comercial parece uma desculpa conveniente, considerando os outros, maiores problemas.

### Vantagens

No entanto, o DeepFloyd realmente tem vantagens que muitas vezes sao ignoradas:

- Em inferencia, o codificador de texto T5 demonstra um forte entendimento do mundo
- Pode ser treinado nativamente com captions muito longas
- A primeira etapa tem area de ~64x64 pixels e pode ser treinada com multiplas proporcoes
  - A natureza de baixa resolucao dos dados de treinamento significa que o DeepFloyd foi _o unico modelo_ capaz de treinar em _TODO_ o LAION-A (poucas imagens sao menores que 64x64 no LAION)
- Cada etapa pode ser ajustada de forma independente, com foco em objetivos diferentes
  - A primeira etapa pode ser ajustada focando em qualidades de composicao, e as etapas posteriores sao ajustadas para melhores detalhes ampliados
- Treina muito rapido apesar do maior consumo de memoria de treinamento
  - Treina mais rapido em termos de throughput - uma alta taxa de amostras por hora e observada no ajuste da etapa 1
  - Aprende mais rapido do que um modelo equivalente a CLIP, talvez em detrimento de quem esta acostumado a treinar modelos CLIP
    - Em outras palavras, voce precisara ajustar suas expectativas de learning rates e cronogramas de treinamento
- Nao ha VAE, as amostras de treinamento sao reduzidas diretamente para o tamanho alvo e os pixels sao consumidos pelo U-net
- Ele suporta LoRAs de ControlNet e muitos outros truques que funcionam em u-nets CLIP lineares tipicos.

## Fine-tuning de uma LoRA

> ⚠️ Devido aos requisitos de compute para backprop completo do u-net mesmo no menor modelo 400M do DeepFloyd, isso nao foi testado. LoRA sera usado neste documento, embora o ajuste completo do u-net tambem deva funcionar.

Estas instrucoes assumem familiaridade basica com o SimpleTuner. Para iniciantes, e recomendado comecar com um modelo mais bem suportado, como [Kwai Kolors](quickstart/KOLORS.md).

No entanto, se voce realmente deseja treinar DeepFloyd, e necessario usar a opcao de configuracao `model_flavour` para indicar qual modelo voce esta treinando.

### config.json

```bash
"model_family": "deepfloyd",

# Valores possiveis:
# - i-medium-400m
# - i-large-900m
# - i-xlarge-4.3b
# - ii-medium-450m
# - ii-large-1.2b
"model_flavour": "i-medium-400m",

# DoRA ainda nao foi muito testado. E novo e experimental.
"use_dora": false,
# Bitfit nao foi testado quanto a eficacia no DeepFloyd.
# Provavelmente funciona, mas nao ha ideia do resultado.
"use_bitfit": false,

# Maior learning rate a usar.
"learning_rate": 4e-5,
# Para cronogramas que decaem ou oscilam, este sera o LR final ou o fundo do vale.
"lr_end": 4e-6,
```

- O `model_family` e deepfloyd
- O `model_flavour` aponta para Stage I ou II
- `resolution` agora e `64` e `resolution_type` e `pixel`
- `attention_mechanism` pode ser definido como `xformers`, mas usuarios AMD e Apple nao conseguirao definir isso, exigindo mais VRAM.
  - **Nota** ~~Apple MPS atualmente tem um bug que impede o ajuste do DeepFloyd de funcionar de qualquer forma.~~ A partir do Pytorch 2.6 ou algum momento anterior, o stage I e II treinam no Apple MPS.

Para validacoes mais detalhadas, o valor de `validation_resolution` pode ser definido como:

- `validation_resolution=64` resultara em uma imagem quadrada 64x64.
- `validation_resolution=96x64` resultara em uma imagem widescreen 3:2.
- `validation_resolution=64,96,64x96,96x64` resultara em quatro imagens sendo geradas para cada validacao:
  - 64x64
  - 96x96
  - 64x96
  - 96x64

### multidatabackend_deepfloyd.json

Agora vamos configurar o dataloader para treinamento DeepFloyd. Isso sera quase identico a configuracao de SDXL ou datasets de modelos legados, com foco em parametros de resolucao.

```json
[
    {
        "id": "primary-dataset",
        "type": "local",
        "instance_data_dir": "/training/data/primary-dataset",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "random",
        "resolution": 64,
        "resolution_type": "pixel",
        "minimum_image_size": 64,
        "maximum_image_size": 256,
        "target_downsample_size": 128,
        "prepend_instance_prompt": false,
        "instance_prompt": "Your Subject Trigger Phrase or Word",
        "caption_strategy": "instanceprompt",
        "repeats": 1
    },
    {
        "id": "an example backend for text embeds.",
        "dataset_type": "text_embeds",
        "default": true,
        "disable": false,
        "type": "local",
        "cache_dir": "/training/cache/deepfloyd/text/dreambooth"
    }
]
```

Acima esta uma configuracao basica de Dreambooth para DeepFloyd:

- Os valores de `resolution` e `resolution_type` estao definidos como `64` e `pixel`, respectivamente
- O valor de `minimum_image_size` foi reduzido para 64 pixels para garantir que nao ampliemos imagens menores acidentalmente
- O valor de `maximum_image_size` foi definido como 256 pixels para garantir que imagens grandes nao sejam recortadas em uma proporcao maior que 4:1, o que pode resultar em perda catastrofica do contexto da cena
- O valor de `target_downsample_size` foi definido como 128 pixels para que quaisquer imagens maiores que `maximum_image_size` de 256 pixels sejam primeiro redimensionadas para 128 pixels antes do recorte

Nota: as imagens sao reduzidas em 25% por vez para evitar saltos extremos no tamanho da imagem que causem uma media incorreta dos detalhes da cena.

## Rodando inferencia

Atualmente, o DeepFloyd nao tem scripts de inferencia dedicados no toolkit do SimpleTuner.

Fora o processo de validacoes embutido, voce pode consultar [este documento do Hugging Face](https://huggingface.co/docs/diffusers/v0.23.1/en/training/dreambooth#if) que contem um pequeno exemplo para rodar inferencia depois:

```py
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", use_safetensors=True)
pipe.load_lora_weights("<lora weights path>")
pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, variance_type="fixed_small")
```

> ⚠️ Note que o primeiro valor para `DiffusionPipeline.from_pretrained(...)` esta definido como `IF-I-M-v1.0`, mas voce deve atualizar isso para usar o caminho do modelo base em que treinou sua LoRA.

> ⚠️ Note que nem todas as recomendacoes do Hugging Face se aplicam ao SimpleTuner. Por exemplo, podemos ajustar a LoRA do DeepFloyd stage I com apenas 22G de VRAM vs 28G nos scripts de dreambooth do Diffusers, graças ao pre-cache eficiente e estados do otimizador puro-bf16.

## Fine-tuning do modelo de super-resolucao stage II

O modelo stage II do DeepFloyd recebe entradas em torno de imagens 64x64 (ou 96x64) e retorna a imagem ampliada resultante usando a configuracao `VALIDATION_RESOLUTION`.

As imagens de avaliacao sao coletadas automaticamente dos seus datasets, de modo que `--num_eval_images` especifica quantas imagens ampliadas selecionar de cada dataset. As imagens atualmente sao selecionadas aleatoriamente - mas permanecerao as mesmas em cada sessao.

Mais alguns checks estao em vigor para garantir que voce nao rode acidentalmente com tamanhos incorretos definidos.

Para treinar o stage II, basta seguir as etapas acima, usando `deepfloyd-stage2-lora` no lugar de `deepfloyd-lora` para `MODEL_TYPE`:

```bash
export MODEL_TYPE="deepfloyd-stage2-lora"
```
