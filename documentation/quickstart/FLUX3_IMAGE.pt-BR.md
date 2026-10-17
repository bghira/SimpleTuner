# Guia rápido do FLUX.3 Image Klein

Este guia mostra como treinar uma LoRA do FLUX.3 Image Klein com suas imagens. Uma LoRA é um pequeno adaptador que ensina um tema ou estilo ao modelo e mantém os pesos originais separados.

Comece com `model_family: "flux3_image"`, `model_flavour: "klein-base"` e `model_type: "lora"`. Isso carrega [black-forest-labs/flux-3-image-klein-base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base). Visite a página, aceite as condições de acesso e leia a [licença FLUX [Kom]munity](https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/LICENSE) antes do download.

## Instalação

Você precisa do Python 3.13, Git e espaço livre para o modelo, imagens e cache. Os comandos abaixo instalam o SimpleTuner para uma GPU NVIDIA CUDA. Consulte o [guia de instalação](../INSTALL.pt-BR.md) para outras plataformas e CUDA 13.

```bash
git clone https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[cuda]'
hf auth login
```

Execute os demais comandos na pasta `SimpleTuner`, com o ambiente virtual ativo. No `hf auth login`, informe um token do Hugging Face com acesso de leitura ao modelo.

## Escolha uma receita para sua GPU

A memória GPU (VRAM) determina quantas imagens você pode processar juntas. Escolha o exemplo para sua capacidade. As receitas Base de 24 e 32 GB usam resolução 512; as maiores também incluem datasets de 1024.

| Memória GPU | Exemplo Base | Imagens por lote em 512 / 1024 |
| --- | --- | --- |
| 24 GB | `flux3_image-24g.peft-lora` | 3 / — |
| 32 GB | `flux3_image-32g.peft-lora` | 4 / — |
| 48 GB | `flux3_image-48g.peft-lora` | 9 / 2 |
| 80 GB | `flux3_image-80g.peft-lora` | 18 / 5 |
| 144 GB | `flux3_image-144g.peft-lora` | 36 / 10 |

Para a variante destilada `klein`, escolha uma destas receitas. Use o nome completo nos comandos de cópia de configuração abaixo. A escala corresponde à área de uma imagem quadrada; as proporções são preservadas.

| Memória da GPU | Exemplo destilado | Escala de treinamento | Imagens por lote |
| --- | --- | --- | ---: |
| 24 GB | `flux3_image-klein-24g.peft-lora` | 768px | 1 |
| 80 GB | `flux3_image-klein-80g.peft-lora` | 1024px | 4 |

Os lotes são pontos de partida. Legendas longas e referências exigem mais memória; reduza `train_batch_size` se ela acabar. As receitas usam LoRA de rank 32, taxa `1e-4` e REPA, que compara características de imagem durante o treino. Seus dataloaders combinam conceito e regularização para preservar o conhecimento geral do modelo.

## Prepare imagens e legendas

Crie `datasets/my-concept` dentro do SimpleTuner e coloque suas imagens nela. Para cada imagem, crie um arquivo de texto com o mesmo nome e sua legenda:

```text
datasets/my-concept/
  image01.png
  image01.txt
  image02.jpg
  image02.txt
```

Para uma LoRA de tema, escolha uma palavra distinta como `TOK` e use-a nas legendas. Descreva o tema, cenário e pose. Por exemplo, `image01.txt` pode conter:

```text
Uma foto de TOK, uma pequena coruja de cerâmica com penas azuis, sobre uma mesa de madeira.
```

## Configure o treinamento

Copie a receita escolhida. Estes comandos usam `flux3_image-48g.peft-lora`; substitua esse nome completo pela sua receita Base ou destilada:

```bash
mkdir -p config/examples
cp -R simpletuner/examples/flux3_image-48g.peft-lora config/examples/
cp simpletuner/examples/flux3_image-prompts.json config/examples/
cp config/examples/flux3_image-48g.peft-lora/config.json config/config.json
```

Abra `config/config.json` em um editor de texto. Para a primeira execução com suas imagens locais, altere estes valores na configuração existente:

```json
{
  "data_backend_config": "config/dataloader.json",
  "output_dir": "output/my-flux3-lora",
  "train_batch_size": 1
}
```

Um lote de 1 permite começar com poucos dados. Aumente após a primeira execução funcionar, quando houver imagens e memória suficientes. `max_train_steps` controla a duração. Edite `config/examples/flux3_image-prompts.json` com prompts que incluam sua palavra; eles geram imagens de progresso.

Salve o conteúdo abaixo como `config/dataloader.json`. Ele informa onde estão as imagens, como ler as legendas e onde armazenar o cache:

```json
[
  {
    "id": "my-concept",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/my-concept",
    "caption_strategy": "textfile",
    "resolution": 0.589824,
    "resolution_type": "area",
    "crop": false,
    "cache_dir_vae": "cache/vae/flux3-image/my-concept",
    "text_embeds": "text"
  },
  {
    "id": "text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux3-image/my-concept"
  }
]
```

Neste dataset local, `resolution: 0.589824` com `resolution_type: "area"` equivale à área de 768×768. Use `1.048576` para o dataset em escala 1024 da receita destilada de 80 GB.

`resolution` controla o tamanho de treino; as dimensões são alinhadas a múltiplos de 16. O cache guarda latentes e embeddings reutilizáveis para agilizar próximas execuções. Separe os caches por família de modelo. Consulte o [guia de dados](../DATALOADER.pt-BR.md) para adicionar datasets ou regularização.

## Opcional: expanda legendas com o prompt upsampler

Adicione este ajuste a `config/config.json` para expandir legendas curtas e prompts de validação com o [upsampler do FLUX.3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup):

```json
{
  "flux3_prompt_upsample": true
}
```

A opção vem desativada. Ao ativá-la, o SimpleTuner baixa o modelo Qwen3.5-9B separado (cerca de 19 GB de pesos) e reescreve prompts positivos antes de armazenar seus embeddings. Para edições, usa as imagens de referência pareadas. Prompts negativos, legendas vazias de dropout e arquivos originais são preservados.

Reserve mais RAM e tempo de pré-processamento. O upsampler e o codificador de texto se alternam na GPU, e os embeddings são reutilizados no treino. O SimpleTuner separa os embeddings expandidos dos originais e reutiliza os resultados quando as mesmas referências são selecionadas novamente.

## Inicie o treinamento

Na pasta SimpleTuner, execute:

```bash
simpletuner train
```

Na primeira execução, o SimpleTuner baixa o modelo, lê suas imagens e cria os caches antes de treinar. A compilação pode aumentar o tempo inicial. Mantenha o processo ativo durante essas etapas. Imagens de progresso e checkpoints ficam em `output/my-flux3-lora`; use as imagens para avaliar o aprendizado do tema.

Para `klein-base`, use 50 passos de validação e guidance 3.5. Para o destilado, selecione `model_flavour: "klein"`: ele carrega `black-forest-labs/flux-3-image-klein` e usa 16 passos com guidance 1. Aceite o acesso a esse repositório antes de iniciar.

## Treinamento de edição e referências

As duas variantes aceitam referências opcionais. O dataset alvo contém as edições desejadas e suas instruções; os datasets de condicionamento contêm as imagens de origem ou referência. Salve este JSON separadamente, configure `data_backend_config` e comece com `train_batch_size: 1`:

```json
[
  {
    "id": "edits",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/edits",
    "caption_strategy": "textfile",
    "resolution": 512,
    "resolution_type": "pixel",
    "conditioning_data": ["references"],
    "cache_dir_vae": "cache/vae/flux3-image/edits",
    "text_embeds": "text"
  },
  {
    "id": "references",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "datasets/references",
    "conditioning_type": "reference_loose",
    "resolution": 512,
    "resolution_type": "pixel",
    "cache_dir_vae": "cache/vae/flux3-image/references"
  },
  {
    "id": "text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux3-image-edit"
  }
]
```

Os nomes devem coincidir: por exemplo, `edits/sample.png`, `edits/sample.txt` e `references/sample.png`. `reference_loose` permite tamanhos e proporções diferentes; `reference_strict` mantém recortes alinhados. Adicione IDs em `conditioning_data` para até dez referências por amostra; use `conditioning_multidataset_sampling: "combined"` para fornecê-las juntas. Sem `conditioning_data`, o treino é texto para imagem. Separe caches de referências, alvos e outras famílias.

Com `conditioning_multidataset_sampling: "random"`, cada amostra de treinamento usa um conjunto de referências e a legenda desse conjunto. Defina `caption_strategy: "textfile"` em cada conjunto de referências e escreva a instrução de edição no arquivo `.txt` correspondente, como `references/sample.txt`.

## Gere uma imagem com sua LoRA

Localize o diretório de checkpoint que contém `pytorch_lora_weights.safetensors`. Substitua `output/my-flux3-lora/checkpoint-1000` por esse diretório e use a mesma variante do treinamento. O exemplo usa base e guidance CFG-Zero-Star:

```python
import torch
from simpletuner.helpers.models.flux3_image.pipeline import Flux3ImagePipeline

pipe = Flux3ImagePipeline.from_pretrained(
    "black-forest-labs/flux-3-image-klein-base", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("output/my-flux3-lora/checkpoint-1000")
image = pipe(
    "A photo of TOK, a small ceramic owl, sitting on a wooden shelf",
    num_inference_steps=50,
    guidance_scale=3.5,
    use_cfg_zero_star=True,
).images[0]
image.save("my-flux3-image.png")
```

## LoRA assistente

O treinamento LoRA com `model_flavour: "klein"` carrega automaticamente o [assistente publicado](https://huggingface.co/SimpleTuner/flux-3-klein-image-training-assistant) como adaptador congelado. `assistant_lora_strength` usa `1.0` durante o treinamento; `assistant_lora_inference_strength` usa `0.0` na validação. Defina `disable_assistant_lora: true` para desativá-lo. Para outro adaptador, configure `assistant_lora_path` e, se necessário, `assistant_lora_weight_name`.

Para treinar seu próprio assistente, selecione `flux3_image-assistant-lora.peft-lora`. Essa receita desativa o assistente existente e aprende com amostras geradas pelo modelo. Depois, indique o diretório salvo em `assistant_lora_path` na sua configuração LoRA destilada.

Veja outros ajustes na [referência de opções](../OPTIONS.pt-BR.md). Ative TwinFlow com `twinflow_enabled: true` e `use_ema: true`; EMA mantém o modelo professor usado nesse método.
