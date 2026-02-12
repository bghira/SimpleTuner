# T-LoRA (LoRA dependente de timestep)

## Contexto

O fine-tuning padrão com LoRA aplica uma adaptação de baixo rank fixa uniformemente em todos os timesteps de difusao. Quando os dados de treinamento sao limitados (especialmente personalizacao de imagem unica), isso leva ao overfitting — o modelo memoriza padroes de ruido em timesteps de alto ruido onde pouca informacao semantica existe.

**T-LoRA** ([Soboleva et al., 2025](https://arxiv.org/abs/2507.05964)) resolve isso ajustando dinamicamente o numero de ranks LoRA ativos com base no timestep de difusao atual:

- **Alto ruido** (inicio da remocao de ruido, $t \to T$): menos ranks estao ativos, impedindo o modelo de memorizar padroes de ruido nao informativos.
- **Baixo ruido** (final da remocao de ruido, $t \to 0$): mais ranks estao ativos, permitindo que o modelo capture detalhes finos do conceito.

O suporte a T-LoRA do SimpleTuner e construido sobre a biblioteca [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) e requer uma versao do LyCORIS que inclua o modulo `lycoris.modules.tlora`.

> 🟡 **Experimental:** T-LoRA com modelos de video pode produzir resultados inferiores porque a compressao temporal mistura frames entre os limites dos timesteps.

## Configuracao rapida

### 1. Defina sua configuracao de treinamento

No seu `config.json`, use LyCORIS com um arquivo de configuracao T-LoRA separado:

```json
{
    "model_type": "lora",
    "lora_type": "lycoris",
    "lycoris_config": "config/lycoris_tlora.json",
    "validation_lycoris_strength": 1.0
}
```

### 2. Crie a configuracao LyCORIS T-LoRA

Crie `config/lycoris_tlora.json`:

```json
{
    "algo": "tlora",
    "multiplier": 1.0,
    "linear_dim": 64,
    "linear_alpha": 32,
    "apply_preset": {
        "target_module": ["Attention", "FeedForward"]
    }
}
```

Isso e tudo que voce precisa para comecar o treinamento. As secoes abaixo cobrem ajustes opcionais e inferencia.

## Referencia de configuracao

### Campos obrigatorios

| Campo | Tipo | Descricao |
|-------|------|-----------|
| `algo` | string | Deve ser `"tlora"` |
| `multiplier` | float | Multiplicador de intensidade do LoRA. Mantenha em `1.0` a menos que voce saiba o que esta fazendo |
| `linear_dim` | int | Rank do LoRA. Isso se torna `max_rank` no cronograma de mascaramento |
| `linear_alpha` | int | Fator de escala do LoRA (separado de `tlora_alpha`) |

### Campos opcionais

| Campo | Tipo | Padrao | Descricao |
|-------|------|--------|-----------|
| `tlora_min_rank` | int | `1` | Minimo de ranks ativos no nivel de ruido mais alto |
| `tlora_alpha` | float | `1.0` | Expoente do cronograma de mascaramento. `1.0` e linear; valores acima de `1.0` deslocam mais capacidade para os passos de detalhes finos |
| `apply_preset` | object | — | Direcionamento de modulos via `target_module` e `module_algo_map` |

### Alvos de modulos especificos por modelo

Para a maioria dos modelos, os alvos genericos `["Attention", "FeedForward"]` funcionam. Para Flux 2 (Klein), use os nomes de classe personalizados:

```json
{
    "algo": "tlora",
    "multiplier": 1.0,
    "linear_dim": 64,
    "linear_alpha": 32,
    "apply_preset": {
        "target_module": [
            "Flux2Attention", "Flux2FeedForward", "Flux2ParallelSelfAttention"
        ]
    }
}
```

Consulte a [documentacao do LyCORIS](../LYCORIS.md) para a lista completa de alvos de modulos por modelo.

## Parametros de ajuste

### `linear_dim` (rank)

Rank mais alto = mais parametros e expressividade, mas mais propenso a overfitting com dados limitados. O artigo original do T-LoRA usa rank 64 para personalizacao de imagem unica em SDXL.

### `tlora_min_rank`

Controla o piso para ativacao de rank no timestep mais ruidoso. Aumentar isso permite que o modelo aprenda estrutura mais grosseira, mas reduz o beneficio contra overfitting. Comece com o padrao de `1` e aumente apenas se a convergencia estiver muito lenta.

### `tlora_alpha` (expoente do cronograma)

Controla a forma da curva do cronograma de mascaramento:

- `1.0` — interpolacao linear entre `min_rank` e `max_rank`
- `> 1.0` — mascaramento mais agressivo em alto ruido; a maioria dos ranks so ativa perto do final da remocao de ruido
- `< 1.0` — mascaramento mais suave; os ranks ativam mais cedo

<details>
<summary>Visualizacao do cronograma (rank vs. timestep)</summary>

Com `linear_dim=64`, `tlora_min_rank=1`, para um scheduler de 1000 passos:

```
alpha=1.0 (linear):
  t=0   (limpo)  → 64 ranks ativos
  t=250 (25%)    → 48 ranks ativos
  t=500 (50%)    → 32 ranks ativos
  t=750 (75%)    → 16 ranks ativos
  t=999 (ruido)  →  1 rank ativo

alpha=2.0 (quadratico — enviesado para detalhes):
  t=0   (limpo)  → 64 ranks ativos
  t=250 (25%)    → 60 ranks ativos
  t=500 (50%)    → 48 ranks ativos
  t=750 (75%)    → 20 ranks ativos
  t=999 (ruido)  →  1 rank ativo

alpha=0.5 (sqrt — enviesado para estrutura):
  t=0   (limpo)  → 64 ranks ativos
  t=250 (25%)    → 55 ranks ativos
  t=500 (50%)    → 46 ranks ativos
  t=750 (75%)    → 33 ranks ativos
  t=999 (ruido)  →  1 rank ativo
```

</details>

## Inferencia com pipelines do SimpleTuner

Os pipelines incluidos no SimpleTuner possuem suporte integrado a T-LoRA. Durante a validacao, os parametros de mascaramento do treinamento sao automaticamente reutilizados em cada passo de remocao de ruido — nenhuma configuracao extra e necessaria.

Para inferencia independente fora do treinamento, voce pode importar o pipeline do SimpleTuner diretamente e definir o atributo `_tlora_config`. Isso garante que o mascaramento por passo corresponda ao que foi usado no treinamento do modelo.

### Exemplo SDXL

```py
import torch
from lycoris import create_lycoris_from_weights

# Use o pipeline SDXL incluso no SimpleTuner (com suporte T-LoRA integrado)
from simpletuner.helpers.models.sdxl.pipeline import StableDiffusionXLPipeline
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
dtype = torch.bfloat16
device = "cuda"

# Carregar componentes do pipeline
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)

# Carregar e aplicar pesos LyCORIS T-LoRA
lora_path = "path/to/pytorch_lora_weights.safetensors"
wrapper, _ = create_lycoris_from_weights(1.0, lora_path, unet)
wrapper.merge_to()

unet.to(device)

pipe = StableDiffusionXLPipeline(
    scheduler=scheduler,
    vae=vae,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    unet=unet,
)

# Habilitar mascaramento T-LoRA na inferencia — deve corresponder a configuracao de treinamento
pipe._tlora_config = {
    "max_rank": 64,      # linear_dim da sua configuracao lycoris
    "min_rank": 1,       # tlora_min_rank (padrao 1)
    "alpha": 1.0,        # tlora_alpha (padrao 1.0)
}

with torch.inference_mode():
    image = pipe(
        prompt="a sks dog riding a surfboard",
        width=1024,
        height=1024,
        num_inference_steps=25,
        guidance_scale=5.0,
    ).images[0]

image.save("tlora_output.png")
```

### Exemplo Flux

```py
import torch
from lycoris import create_lycoris_from_weights

# Use o pipeline Flux incluso no SimpleTuner (com suporte T-LoRA integrado)
from simpletuner.helpers.models.flux.pipeline import FluxPipeline
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16
device = "cuda"

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2")
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype)

# Carregar e aplicar pesos LyCORIS T-LoRA
lora_path = "path/to/pytorch_lora_weights.safetensors"
wrapper, _ = create_lycoris_from_weights(1.0, lora_path, transformer)
wrapper.merge_to()

transformer.to(device)

pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=transformer,
)

# Habilitar mascaramento T-LoRA na inferencia
pipe._tlora_config = {
    "max_rank": 64,
    "min_rank": 1,
    "alpha": 1.0,
}

with torch.inference_mode():
    image = pipe(
        prompt="a sks dog riding a surfboard",
        width=1024,
        height=1024,
        num_inference_steps=25,
        guidance_scale=3.5,
    ).images[0]

image.save("tlora_flux_output.png")
```

> **Nota:** Voce deve usar o pipeline incluso no SimpleTuner (ex: `simpletuner.helpers.models.flux.pipeline.FluxPipeline`), nao o pipeline padrao do Diffusers. Apenas os pipelines inclusos contem a logica de mascaramento T-LoRA por passo.

### Por que nao apenas usar `merge_to()` e pular o mascaramento?

`merge_to()` incorpora os pesos do LoRA no modelo base permanentemente — isso e necessario para que os parametros do LoRA estejam ativos durante o forward pass. No entanto, o T-LoRA foi **treinado** com mascaramento de rank dependente de timestep: certos ranks foram zerados dependendo do nivel de ruido. Sem reaplicar esse mesmo mascaramento durante a inferencia, todos os ranks disparam em cada timestep, produzindo imagens super-saturadas ou com aparencia de queimado.

Definir `_tlora_config` no pipeline instrui o loop de remocao de ruido a aplicar a mascara correta antes de cada forward pass do modelo e limpa-la depois.

<details>
<summary>Como o mascaramento funciona internamente</summary>

Em cada passo de remocao de ruido, o pipeline chama:

```python
from simpletuner.helpers.training.lycoris import apply_tlora_inference_mask, clear_tlora_mask

_tlora_cfg = getattr(self, "_tlora_config", None)
if _tlora_cfg:
    apply_tlora_inference_mask(
        timestep=int(t),
        max_timestep=self.scheduler.config.num_train_timesteps,
        max_rank=_tlora_cfg["max_rank"],
        min_rank=_tlora_cfg["min_rank"],
        alpha=_tlora_cfg["alpha"],
    )
try:
    noise_pred = self.unet(...)  # ou self.transformer(...)
finally:
    if _tlora_cfg:
        clear_tlora_mask()
```

`apply_tlora_inference_mask` calcula uma mascara binaria de formato `(1, max_rank)` usando a formula:

$$r = \left\lfloor\left(\frac{T - t}{T}\right)^\alpha \cdot (R_{\max} - R_{\min})\right\rfloor + R_{\min}$$

onde $T$ e o timestep maximo do scheduler, $R_{\max}$ e `linear_dim`, e $R_{\min}$ e `tlora_min_rank`. Os primeiros $r$ elementos da mascara sao definidos como `1.0` e o restante como `0.0`. Essa mascara e entao definida globalmente em todos os modulos T-LoRA via `set_timestep_mask()` do LyCORIS.

Apos o forward pass ser concluido, `clear_tlora_mask()` remove o estado da mascara para que nao vaze para operacoes subsequentes.

</details>

<details>
<summary>Como o SimpleTuner passa a configuracao durante a validacao</summary>

Durante o treinamento, o dicionario de configuracao T-LoRA (`max_rank`, `min_rank`, `alpha`) e armazenado no objeto Accelerator. Quando a validacao e executada, `validation.py` copia essa configuracao para o pipeline:

```python
# setup_pipeline()
if getattr(self.accelerator, "_tlora_active", False):
    self.model.pipeline._tlora_config = self.accelerator._tlora_config

# clean_pipeline()
if hasattr(self.model.pipeline, "_tlora_config"):
    del self.model.pipeline._tlora_config
```

Isso e totalmente automatico — nenhuma configuracao do usuario e necessaria para que as imagens de validacao usem o mascaramento correto.

</details>

## Origem: o artigo T-LoRA

<details>
<summary>Detalhes do artigo e algoritmo</summary>

**T-LoRA: Single Image Diffusion Model Customization Without Overfitting**
Vera Soboleva, Aibek Alanov, Andrey Kuznetsov, Konstantin Sobolev
[arXiv:2507.05964](https://arxiv.org/abs/2507.05964) — Aceito na AAAI 2026

O artigo introduz duas inovacoes complementares:

### 1. Mascaramento de rank dependente de timestep

A ideia central e que timesteps de difusao mais altos (entradas mais ruidosas) sao mais propensos a overfitting do que timesteps mais baixos. Em alto ruido, o latente contem principalmente ruido aleatorio com pouco sinal semantico — treinar um adaptador de rank completo nisso ensina o modelo a memorizar padroes de ruido em vez de aprender o conceito alvo.

O T-LoRA resolve isso com um cronograma de mascaramento dinamico que restringe o rank LoRA ativo com base no timestep atual.

### 2. Parametrizacao de pesos ortogonais (opcional)

O artigo tambem propoe inicializar os pesos do LoRA via decomposicao SVD dos pesos originais do modelo, impondo ortogonalidade atraves de uma perda de regularizacao. Isso garante independencia entre os componentes do adaptador.

A integracao do SimpleTuner com LyCORIS foca no componente de mascaramento por timestep, que e o principal fator de reducao de overfitting. A inicializacao ortogonal faz parte da implementacao independente do T-LoRA, mas nao e atualmente utilizada pelo algoritmo `tlora` do LyCORIS.

### Citacao

```bibtex
@misc{soboleva2025tlorasingleimagediffusion,
      title={T-LoRA: Single Image Diffusion Model Customization Without Overfitting},
      author={Vera Soboleva and Aibek Alanov and Andrey Kuznetsov and Konstantin Sobolev},
      year={2025},
      eprint={2507.05964},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.05964},
}
```

</details>

## Erros comuns

- **Esqueceu `_tlora_config` durante a inferencia:** As imagens ficam super-saturadas ou com aparencia de queimado. Todos os ranks disparam em cada timestep em vez de seguir o cronograma de mascaramento treinado.
- **Usando o pipeline padrao do Diffusers:** Os pipelines padrao nao contem a logica de mascaramento T-LoRA. Voce deve usar os pipelines inclusos no SimpleTuner.
- **Incompatibilidade de `linear_dim`:** O `max_rank` em `_tlora_config` deve corresponder ao `linear_dim` usado durante o treinamento, ou as dimensoes da mascara estarao incorretas.
- **Modelos de video:** A compressao temporal mistura frames entre os limites dos timesteps, o que pode enfraquecer o sinal de mascaramento dependente de timestep. Os resultados podem ser inferiores.
- **SDXL + modulos FeedForward:** Treinar modulos FeedForward com LyCORIS no SDXL pode causar perda NaN — isso e um problema geral do LyCORIS, nao especifico do T-LoRA. Consulte a [documentacao do LyCORIS](../LYCORIS.md#potential-problems) para detalhes.
