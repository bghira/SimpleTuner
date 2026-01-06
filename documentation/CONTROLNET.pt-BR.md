# Guia de treinamento do ControlNet

## Contexto

Modelos ControlNet sao capazes de muitas tarefas, que dependem dos dados de condicionamento fornecidos no treinamento.

No inicio, eram muito intensivos em recursos para treinar, mas agora podemos usar PEFT LoRA ou Lycoris para treinar as mesmas tarefas com muito menos recursos.

Exemplo (retirado do model card do ControlNet no Diffusers):

![example](https://tripleback.net/public/controlnet-example-1.png)

A esquerda, voce pode ver o "mapa de bordas canny" fornecido como entrada de condicionamento. A direita, estao as saidas que o modelo ControlNet guiou a partir do modelo base SDXL.

Quando o modelo e usado dessa forma, o prompt quase nao lida com a composicao, apenas preenche os detalhes.

## Como e treinar um ControlNet

No começo, ao treinar um ControlNet, ele nao tem nenhuma indicacao de controle:

![example](https://tripleback.net/public/controlnet-example-2.png)
(_ControlNet treinado por apenas 4 passos em um modelo Stable Diffusion 2.1_)

O prompt do antelope ainda tem a maior parte do controle sobre a composicao, e a entrada de condicionamento do ControlNet e ignorada.

Com o tempo, a entrada de controle deve ser respeitada:

![example](https://tripleback.net/public/controlnet-example-3.png)
(_ControlNet treinado por apenas 100 passos em um modelo Stable Diffusion XL_)

Nesse ponto, alguns indícios de influencia do ControlNet comecaram a aparecer, mas os resultados eram extremamente inconsistentes.

Muito mais do que 100 passos sera necessario para isso funcionar!

## Exemplo de configuracao do dataloader

A configuracao do dataloader continua bem proxima de uma configuracao tipica de dataset texto-para-imagem:

- Os dados de imagem principais sao o conjunto `antelope-data`
  - A chave `conditioning_data` agora esta definida e deve ser o valor `id` do conjunto de dados de condicionamento que pareia com esse conjunto.
  - `dataset_type` deve ser `image` para o conjunto base
- Um dataset secundario e configurado, chamado `antelope-conditioning`
  - O nome nao e importante - adicionar `-data` e `-conditioning` e feito neste exemplo apenas para fins ilustrativos.
  - `dataset_type` deve ser `conditioning`, indicando ao treinador que isso sera usado para avaliacao e para treinamento com entrada condicionada.
- Ao treinar SDXL, as entradas de condicionamento nao sao codificadas pelo VAE, mas sim passadas diretamente para o modelo durante o treinamento como valores de pixel. Isso significa que nao gastamos mais tempo processando embeddings do VAE no inicio do treinamento!
- Ao treinar Flux, SD3, Auraflow, HiDream ou outros modelos MMDiT, as entradas de condicionamento sao codificadas em latentes, e essas serao computadas sob demanda durante o treinamento.
- Embora tudo esteja explicitamente rotulado como `-controlnet` aqui, voce pode reutilizar os mesmos embeddings de texto que usou para o ajuste completo/LoRA. As entradas do ControlNet nao modificam os embeddings do prompt.
- Ao usar aspect bucketing e recorte aleatorio, as amostras de condicionamento serao recortadas da mesma forma que as amostras de imagem principal, entao nao ha com o que se preocupar.

```json
[
    {
        "id": "antelope-data",
        "type": "local",
        "dataset_type": "image",
        "conditioning_data": "antelope-conditioning",
        "instance_data_dir": "datasets/animals/antelope-data",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "an antelope",
        "metadata_backend": "discovery",
        "minimum_image_size": 512,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "cache_dir_vae": "cache/vae/sdxl/antelope-data",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "cache_file_suffix": "controlnet-sdxl"
    },
    {
        "id": "antelope-conditioning",
        "type": "local",
        "dataset_type": "conditioning",
        "instance_data_dir": "datasets/animals/antelope-conditioning",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "an antelope",
        "metadata_backend": "discovery",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution": 1024,
        "minimum_image_size": 512,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "resolution_type": "pixel_area",
        "cache_file_suffix": "controlnet-sdxl"
    },
    {
        "id": "an example backend for text embeds.",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/sdxl-base/controlnet"
    }
]
```

## Gerando entradas de imagem de condicionamento

Como o suporte a ControlNet e novo no SimpleTuner, no momento temos apenas uma opcao disponivel para gerar o seu conjunto de treinamento:

- [create_canny_edge.py](/scripts/toolkit/datasets/controlnet/create_canny_edge.py)
  - Um exemplo extremamente basico de como gerar um conjunto de treinamento para o modelo Canny.
  - Voce precisara modificar os valores `input_dir` e `output_dir` no script

Isso leva cerca de 30 segundos para um dataset pequeno com menos de 100 imagens.

## Modificando sua configuracao para treinar modelos ControlNet

Apenas configurar o dataloader nao sera suficiente para iniciar o treinamento de modelos ControlNet.

Dentro de `config/config.json`, voce precisa definir os seguintes valores:

```bash
"model_type": 'lora',
"controlnet": true,

# Voce talvez precise reduzir TRAIN_BATCH_SIZE e RESOLUTION mais do que o normal
"train_batch_size": 1
```

Sua configuracao ficara algo assim no final:

```json
{
    "aspect_bucket_rounding": 2,
    "caption_dropout_probability": 0.1,
    "checkpoint_step_interval": 100,
    "checkpoints_total_limit": 5,
    "controlnet": true,
    "data_backend_config": "config/controlnet-sdxl/multidatabackend.json",
    "disable_benchmark": false,
    "gradient_checkpointing": true,
    "hub_model_id": "simpletuner-controlnet-sdxl-lora-test",
    "learning_rate": 3e-5,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 100,
    "max_train_steps": 1000,
    "minimum_image_size": 0,
    "mixed_precision": "bf16",
    "model_family": "sdxl",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "bnb-lion8bit",
    "output_dir": "output/controlnet-sdxl/models",
    "push_checkpoints_to_hub": true,
    "push_to_hub": true,
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "resume_from_checkpoint": "latest",
    "seed": 42,
    "train_batch_size": 1,
    "use_ema": false,
    "vae_cache_ondemand": true,
    "validation_guidance": 4.2,
    "validation_guidance_rescale": 0.0,
    "validation_num_inference_steps": 20,
    "validation_resolution": "1024x1024",
    "validation_seed": 42,
    "validation_step_interval": 10,
    "validation_torch_compile": false
}
```

## Inferencia nos modelos ControlNet resultantes

Um exemplo SDXL e fornecido aqui para inferencia em um modelo ControlNet **completo** (nao ControlNet LoRA):

```py
# Atualize estes valores:
base_model = "stabilityai/stable-diffusion-xl-base-1.0"         # Este e o modelo que voce usou como `--pretrained_model_name_or_path`
controlnet_model_path = "diffusers/controlnet-canny-sdxl-1.0"   # Este e o caminho para o checkpoint ControlNet resultante
# controlnet_model_path = "/path/to/controlnet/checkpoint-100"

# Deixe o resto como esta:
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2

prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = 'low quality, bad quality, sketches'

image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")

controlnet_conditioning_scale = 0.5  # recomendado para boa generalizacao

controlnet = ControlNetModel.from_pretrained(
    controlnet_model_path,
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

images = pipe(
    prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images

images[0].save(f"hug_lab.png")
```
(_Codigo de demonstracao adaptado do [exemplo de SDXL ControlNet no Hugging Face](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0)_)


## Geracao automatica de dados e condicionamento

O SimpleTuner pode gerar datasets de condicionamento automaticamente durante a inicializacao, eliminando a necessidade de pre-processamento manual. Isso e especialmente util para:
- Treinamento de super-resolucao
- Remocao de artefatos JPEG
- Geracao guiada por profundidade
- Deteccao de bordas (Canny)

### Como funciona

Em vez de criar manualmente datasets de condicionamento, voce pode especificar um array `conditioning` na configuracao do seu dataset principal. O SimpleTuner vai:
1. Gerar as imagens de condicionamento na inicializacao
2. Criar datasets separados com metadados apropriados
3. Vincula-los automaticamente ao seu dataset principal

### Consideracoes de performance

Alguns geradores vao rodar mais devagar se forem limitados por CPU e seu sistema tiver dificuldades com tarefas de CPU, enquanto outros podem exigir recursos de GPU e, portanto, rodar no processo principal, o que pode aumentar o tempo de inicializacao.

**Geradores baseados em CPU (rapidos):**
- `superresolution` - Operacoes de blur e ruido
- `jpeg_artifacts` - Simulacao de compressao
- `random_masks` - Geracao de mascaras
- `canny` - Deteccao de bordas

**Geradores baseados em GPU (mais lentos):**
- `depth` / `depth_midas` - Requer carregar modelos transformer
- `segmentation` - Modelos de segmentacao semantica
- `optical_flow` - Estimativa de movimento

Geradores baseados em GPU rodam no processo principal e podem aumentar significativamente o tempo de inicializacao para datasets grandes.

### Exemplo: dataset de condicionamento multi-tarefa

Aqui esta um exemplo completo que gera multiplos tipos de condicionamento a partir de um unico dataset de origem:

```json
[
  {
    "id": "multitask-training",
    "type": "local",
    "instance_data_dir": "/datasets/high-quality-images",
    "caption_strategy": "filename",
    "resolution": 512,
    "conditioning": [
      {
        "type": "superresolution",
        "blur_radius": 2.0,
        "noise_level": 0.02,
        "captions": ["enhance image quality", "increase resolution", "sharpen"]
      },
      {
        "type": "jpeg_artifacts",
        "quality_range": [20, 40],
        "captions": ["remove compression", "fix jpeg artifacts"]
      },
      {
        "type": "canny",
        "low_threshold": 50,
        "high_threshold": 150
      }
    ]
  },
  {
    "id": "text-embed-cache",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/sdxl"
  }
]
```

Essa configuracao vai:
1. Carregar suas imagens de alta qualidade de `/datasets/high-quality-images`
2. Gerar tres datasets de condicionamento automaticamente
3. Usar captions especificas para tarefas de super-resolucao e JPEG
4. Usar as captions originais das imagens para o dataset de bordas Canny

#### Estrategias de caption para datasets gerados

Voce tem duas opcoes para captionar dados de condicionamento gerados:

1. **Usar captions da origem** (padrao): omita o campo `captions`
2. **Captions personalizadas**: forneca uma string ou um array de strings

Para treinamento especifico de tarefas (como "melhorar" ou "remover artefatos"), captions personalizadas geralmente funcionam melhor do que as descricoes originais das imagens.

### Otimizacao do tempo de inicializacao

Para datasets grandes, a geracao de condicionamento pode demorar bastante. Para otimizar:

1. **Gere uma vez**: os dados de condicionamento sao armazenados em cache e nao serao gerados novamente se ja existirem
2. **Use geradores de CPU**: eles podem usar multiplos processos para gerar mais rapido
3. **Desative tipos nao usados**: gere apenas o que voce precisa para o seu treinamento
4. **Pre-gerar**: voce pode rodar com `--skip_file_discovery=true` para pular a descoberta e geracao dos dados de condicionamento
5. **Evite varreduras de disco**: voce pode usar `preserve_data_backend_cache=True` em qualquer configuracao de dataset grande para evitar revarrer o disco em busca de dados de condicionamento existentes. Isso vai acelerar significativamente o tempo de inicializacao, especialmente para datasets grandes.

O processo de geracao mostra barras de progresso e suporta retomada se for interrompido.
