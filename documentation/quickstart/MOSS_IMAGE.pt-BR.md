# Guia rápido do MOSS-Image

Treine um LoRA do MOSS-Image com suas próprias imagens. Um LoRA é um pequeno adaptador que ensina um assunto ou estilo ao modelo e mantém o checkpoint original separado.

Comece com `model_family: "moss_image"`, `model_flavour: "base"` e `model_type: "lora"` para usar [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base). Escolha `model_flavour: "sft"` para [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), refinado com dados selecionados. Ambos compartilham um transformador de imagens de 5B, o codificador de texto Qwen3.5-9B e um VAE FLUX.2. Os checkpoints usam [Apache-2.0](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/main/LICENSE). O codificador de texto permanece congelado durante o treinamento.

## Escolher um preset de VRAM

Escolha a receita de acordo com a memória da GPU. Todos os níveis treinam com área equivalente a 1024×1024, preservam as proporções e mantêm LoRA BF16 de rank 32 e REPA.

| Memória da GPU | Exemplo Base | Exemplo SFT | Imagens por lote | `gradient_checkpointing_interval` |
| --- | --- | --- | ---: | ---: |
| 24 GB | `moss_image-24g.peft-lora` | `moss_image-sft-24g.peft-lora` | 1 | 1 |
| 32 GB | `moss_image-32g.peft-lora` | `moss_image-sft-32g.peft-lora` | 1 | 2 |
| 48 GB | `moss_image-48g.peft-lora` | `moss_image-sft-48g.peft-lora` | 2 | 2 |
| 80 GB | `moss_image-80g.peft-lora` | `moss_image-sft-80g.peft-lora` | 4 | 2 |
| 144 GB | `moss_image-144g.peft-lora` | `moss_image-sft-144g.peft-lora` | 8 | 2 |

Os exemplos padrão `moss_image.peft-lora` e `moss_image-sft.peft-lora` usam os ajustes de 48 GB. Os lotes são pontos de partida: reduza `train_batch_size` para legendas mais longas, recursos adicionais ou datasets com menos imagens por bucket de proporção que o tamanho do lote. O nível de 144 GB é destinado a GPUs da classe H200.

## Instalar o SimpleTuner

Use Python 3.13 e uma GPU NVIDIA para a receita CUDA abaixo. Consulte a [instalação](../INSTALL.md) para outras plataformas e versões do CUDA.

```bash
git clone https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[cuda,attention-cute]'
```

## Preparar imagens e legendas

Crie `datasets/moss-image` e adicione imagens com arquivos `.txt` de mesmo nome. Use imagens nítidas com detalhes suficientes para treinar em cerca de um megapixel. Uma imagem quadrada nessa escala tem 1024×1024; imagens largas e altas mantêm suas proporções.

```text
datasets/moss-image/
  image01.png
  image01.txt
  image02.jpg
  image02.txt
```

Escolha uma palavra como `TOK` e inclua-a em todas as legendas daquele assunto. Descreva o assunto visível, composição, luz, ambiente e estilo. O projeto recomenda começar em inglês. Copie o texto visível com precisão, inclusive quebras de linha. Exemplo: `A photograph of TOK, a small blue ceramic owl on a wooden table beside a window.`

## Configurar o treinamento

Copie a receita Base de 48 GB e seus prompts. Substitua `48g` pelo nível escolhido. Para SFT, substitua `moss_image-48g.peft-lora` por `moss_image-sft-48g.peft-lora` nos comandos.

```bash
mkdir -p config/examples datasets/moss-image
cp -R simpletuner/examples/moss_image-48g.peft-lora config/examples/
cp simpletuner/examples/moss_image-prompts.json config/examples/
cp config/examples/moss_image-48g.peft-lora/config.json config/config.json
```

A configuração usa LoRA BF16 de rank 32, taxa de aprendizado `1e-4`, gradient checkpointing, REPA, 4.000 passos do otimizador e escala 1024. Abra `config/config.json` e altere estas entradas na configuração existente:

```json
{
  "output_dir": "output/my-moss-lora",
  "validation_prompt": "A studio photograph of TOK on a wooden table, soft window light, a pale gray background.",
  "user_prompt_library": null
}
```

O `dataloader.json` copiado aponta para `datasets/moss-image`, lê as legendas correspondentes e armazena latentes e embeddings em cache. `resolution: 1.048576` com `resolution_type: "area"` equivale à área de 1024×1024. As dimensões são alinhadas a múltiplos de 16. Consulte o [guia do dataloader](../DATALOADER.md) para mais datasets e imagens de regularização.

## Treinar e comparar prévias

Execute `simpletuner train` na pasta SimpleTuner com o ambiente virtual ativo. A primeira execução baixa o modelo e o codificador REPA e cria os caches. A receita salva checkpoints e imagens de validação a cada 250 passos. Compare o aprendizado do assunto, a composição e os detalhes; ajuste `max_train_steps` para seu dataset.

```bash
simpletuner train
```

## Configurações de amostragem

Comece com **50 passos**, **CFG 3** e validação em **1024×1024**. `moss_image_validation_shift` usa **5** na amostragem; o treinamento usa a grade original sem deslocamento. `moss_image_uncond_skip_layers` usa **[22]**, ignorando essa camada no ramo negativo do CFG. Use `[]` para CFG padrão. O desvio adicional no ramo positivo usa `validation_guidance_skip_layers`, `validation_guidance_skip_scale` e os limites de início e fim em [OPTIONS](../OPTIONS.md). Na geração Python, `use_cfg_zero_star=True` ativa CFG-Zero*.

## Prévias durante a amostragem

Adicione os ajustes abaixo para ver quadros de progresso durante o denoising na WebUI. As prévias usam TAEF2; as imagens finais usam o VAE completo. A CLI pode enviar prévias por um webhook configurado.

```json
{
  "validation_preview": true,
  "validation_preview_steps": 5
}
```

## Memória e opções de treinamento

A receita define `train_batch_size` para o nível de VRAM escolhido. Reduza para datasets pequenos ou quando legendas e recursos adicionais precisarem de mais memória. `gradient_checkpointing_backend` aceita `torch`, `unsloth`, `torch-ffn` e `unsloth-ffn`; o intervalo e `gradient_checkpointing_segment_stride` controlam a cobertura. Musubi move blocos entre GPU e CPU com `musubi_blocks_to_swap`. RamTorch usa `ramtorch: true` e permite descarga seletiva com `ramtorch_transformer_percent`. A descarga usa RAM e adiciona transferências; configure uma abordagem de cada vez. Consulte [OPTIONS](../OPTIONS.md).

A instalação CUDA inclui atenção CuTE. Durante o treinamento, o padrão `attention_mechanism: "kohaku-fa-auto"` seleciona CuTE em GPUs Ada e Hopper e SDPA nativo em outras arquiteturas. Ative `gradient_checkpointing_offload_attention: true` para descarregar os tensores de atenção salvos para backward; `gradient_checkpointing_offload_prefetch: true` ativa a pré-busca das transferências. Consulte [OPTIONS](../OPTIONS.md) para outros backends de atenção.

A receita ativa [REPA](../experimental/IMAGE_REPA.md). Para alinhar características de uma camada posterior a uma anterior, adicione [LayerSync](../experimental/LAYERSYNC.md):

```json
{
  "layersync_enabled": true,
  "layersync_student_block": 20,
  "layersync_teacher_block": 8,
  "layersync_lambda": 0.2
}
```

Para pós-treinamento, siga [AnyFlow](../quickstart/ANYFLOW.md) ou [TwinFlow](../distillation/TWINFLOW.md). Mantenha `lora_dropout: 0.0` no AnyFlow. Ao carregar seu LoRA em Python, passe os ajustes do treinamento para `load_lora_weights`, por exemplo `flowmap_deltatime_type="r", flowmap_gate_value=0.25`.

## Melhoria de prompts e layouts

MOSS-Image aceita legendas detalhadas, tags e layouts JSON. A [skill de melhoria de prompts](https://github.com/OpenMOSS/MOSS-Image/tree/main/skills/moss-image-prompt-enhancer) ajuda um agente a prepará-los. Use o resultado como legenda ou prompt. O [guia do projeto](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md) inclui recomendações de legendagem.

Um layout descreve a cena, o texto e suas posições. As coordenadas são `[y_min, x_min, y_max, x_max]` em pixels, com origem no centro da tela. Em 1024×1024 vão de −512 a +512. Serialize com `json.dumps(layout, ensure_ascii=False)` e passe a string como prompt; use dimensões iguais às do layout.

Para expandir legendas para JSON nativo durante o pré-processamento, adicione estas opções à configuração de treino:

```json
{
  "moss_image_auto_json": true,
  "moss_image_prompt_upsample": true
}
```

O modelo padrão é `Qwen/Qwen3.5-27B`, um dos VLMs recomendados pelo projeto. Use uma GPU de 80 GB e reserve RAM adicional. O SimpleTuner usa cada imagem e as dimensões de seu bucket para gerar descrições e caixas em pixels centrados, depois armazena os embeddings. Os prompts de validação também são expandidos; os vazios e negativos ficam intactos. Para encapsular legendas existentes em JSON sem carregar outro modelo, ative somente `moss_image_auto_json`.

```json
{
  "summary": {"caption": "A cream travel poster with the title EXPLORE above a blue mountain."},
  "items": [
    {"label": "text", "text": "EXPLORE", "caption": "Large navy title", "bbox": [-416, -384, -288, 384]},
    {"label": "object", "caption": "Blue mountain", "bbox": [-192, -288, 384, 288]}
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

## Gerar imagens com seu LoRA

O adaptador final é `pytorch_lora_weights.safetensors` na pasta de saída. Carregue-o com o mesmo checkpoint Base ou SFT usado no treinamento. Inclua a palavra do assunto e descreva a cena:

```python
import torch
from simpletuner.helpers.models.moss_image.pipeline import MOSSImagePipeline

pipe = MOSSImagePipeline.from_pretrained(
    "OpenMOSS-Team/MOSS-Image-Base", torch_dtype=torch.bfloat16
)
pipe.load_lora_weights("output/my-moss-lora")
pipe.to("cuda")
image = pipe(
    "A studio photograph of TOK on a wooden table, soft window light, a pale gray background.",
    height=1024, width=1024, num_inference_steps=50, guidance_scale=3.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("my-moss-image.png")
```
