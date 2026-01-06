# Mixture-of-Experts

O SimpleTuner permite dividir a tarefa de treinamento em duas, de modo que as etapas de self-attention e cross-attention da inferencia possam ser divididas entre dois conjuntos totalmente diferentes de pesos.

Neste exemplo, usaremos o esforco colaborativo da SegMind com o Hugging Face, [SSD-1B](https://huggingface.co/segmind/ssd-1b), para criar dois novos modelos que treinam de forma mais confiavel e apresentam melhores detalhes finos do que um unico modelo.

Graças ao tamanho pequeno do SSD-1B, e possivel treinar em hardware ainda mais leve. Como estamos iniciando o modelo a partir dos pesos pretreinados deles, precisamos respeitar a licenca Apache 2.0 - mas isso e relativamente simples. Voce pode ate usar os pesos resultantes em um contexto comercial!

Quando SDXL 0.9 e 1.0 foram introduzidos, ambos continham um modelo base completo com um refiner de schedule dividido.

- O modelo base foi treinado em steps 999 a 0
  - O modelo base tem mais de 3B parametros e funciona totalmente sozinho.
- O modelo refiner foi treinado em steps 199 a 0
  - O refiner tambem tem mais de 3B parametros, um desperdicio aparentemente desnecessario de recursos. Ele nao funciona sozinho sem deformacoes substanciais e um viés para saidas cartunizadas.

Vamos ver como podemos melhorar essa situacao.


## O modelo base, "Stage One"

A primeira parte de um mixture-of-experts e conhecida como modelo base. Como mencionado no caso do SDXL, ele e treinado nos 1000 timesteps - mas nao precisa ser. A configuracao abaixo treinara o modelo base em apenas 650 steps do total de 1000, economizando tempo e treinando de forma mais confiavel.

### Configuracao do ambiente

Definir os valores abaixo no seu `config/config.env` vai nos colocar no caminho certo:

```bash
# Garanta que estes nao estejam definidos incorretamente.
export USE_BITFIT=false
export USE_DORA=false
# lora poderia ser usado aqui, mas o conceito nao foi explorado.
export MODEL_TYPE="full"
export MODEL_FAMILY="sdxl"
export MODEL_NAME="segmind/SSD-1B"
# O modelo original da Segmind usou taxa de aprendizado 1e-5, que
# provavelmente e alta demais para o batch size que a maioria consegue.
export LEARNING_RATE=4e-7

# Queremos isso o mais alto possivel dentro do que voce tolera.
# - Se o treinamento for muito lento, garanta que CHECKPOINT_STEPS e VALIDATION_STEPS
#   estejam baixos o suficiente para gerar um checkpoint a cada poucas horas.
# - Os modelos originais da Segmind usaram batch size 32 com 4 acumulacoes.
export TRAIN_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=1

# Se voce esta em uma maquina forte que nao usa toda a VRAM no treino, defina como "false" e o treino ficara mais rapido.
export USE_GRADIENT_CHECKPOINTING=true

# Habilitar treinamento do modelo de primeiro estagio
export TRAINER_EXTRA_ARGS="--refiner_training --refiner_training_strength=0.35 --refiner_training_invert_schedule"

# Opcionalmente reparametrize para v-prediction/zero-terminal SNR. prediction_type 'sample' pode ser usado para x-prediction.
# Isso vai parecer ruim no inicio ate cerca de 1500-2500 steps, mas pode valer a pena.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --prediction_type=v_prediction --rescale_betas_zero_snr --training_scheduler_timestep_spacing=trailing"
```

### Configuracao do dataloader

Nenhuma consideracao especial e necessaria para configuracao do dataloader. Veja o [guia de configuracao do dataloader](DATALOADER.md) para mais informacoes sobre esse passo.

### Validacao

Atualmente, o SimpleTuner nao aciona o modelo de segundo estagio durante avaliacoes do estagio um.

Trabalhos futuros suportarao isso como uma opcao, caso um modelo de estagio dois ja exista ou esteja sendo treinado em paralelo.

---

## O modelo refiner, "Stage Two"

### Comparacao com treinamento do refiner do SDXL

- Diferente do refiner do SDXL, ao usar Segmind SSD-1B para ambos os estagios, os text embeds **podem** ser compartilhados entre os dois jobs de treinamento
  - O refiner do SDXL usa um layout de text embed diferente do modelo base do SDXL.
- Os VAE embeds **podem** ser compartilhados entre os jobs de treinamento, assim como no refiner do SDXL. Ambos os modelos usam o mesmo layout de entrada.
- Nenhum score estetico e usado para os modelos Segmind; em vez disso eles usam os mesmos microconditioning inputs do SDXL, por exemplo, coordenadas de crop
- O treinamento e bem mais rapido, pois o modelo e menor e os text embeds podem ser reutilizados do treinamento do stage one

### Configuracao do ambiente

Atualize os valores abaixo no seu `config/config.env` para trocar o treinamento para o seu modelo de stage two. Pode ser conveniente manter uma copia da configuracao do modelo base.

```bash
# Atualize o OUTPUT_DIR para que nao sobrescrevamos os checkpoints do stage one.
export OUTPUT_DIR="/some/new/path"

# Vamos trocar --refiner_training_invert_schedule por --validation_using_datasets
# - Treinar o fim do modelo em vez do inicio
# - Validar usando imagens como entrada para denoising parcial e avaliar melhorias de detalhe fino
export TRAINER_EXTRA_ARGS="--refiner_training --refiner_training_strength=0.35 --validation_using_datasets"

# Nao altere estes valores se voce os definiu no stage one. Garanta a mesma parametrizacao em ambos os modelos!
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --prediction_type=v_prediction --rescale_betas_zero_snr --training_scheduler_timestep_spacing=trailing"
```

### Formato do dataset

As imagens devem ser puramente de alta qualidade - remova quaisquer datasets que voce considere questionaveis em termos de artefatos de compressao ou outros erros.

Fora isso, a mesma configuracao de dataloader pode ser usada entre os dois jobs de treinamento.

Se voce quiser um dataset de demonstracao, [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) e uma boa escolha com licenca permissiva.

### Validacao

O treinamento do refiner de stage two seleciona automaticamente imagens de cada um dos seus conjuntos de treino e usa essas imagens como entrada para denoising parcial na validacao.

## Acompanhamento de CLIP score

Se voce quiser habilitar avaliacoes para pontuar o desempenho do modelo, veja [este documento](evaluation/CLIP_SCORES.md) para informacoes sobre configuracao e interpretacao de CLIP scores.

# Perda de avaliacao estavel

Se voce quiser usar perda MSE estavel para pontuar o desempenho do modelo, veja [este documento](evaluation/EVAL_LOSS.md) para informacoes sobre configuracao e interpretacao de perda de avaliacao.

## Juntando tudo na inferencia

Se voce quiser plugar os dois modelos para experimentar em um script simples, isto vai te colocar no caminho:

```py
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, UniPCMultistepScheduler
from torch import float16, cuda
from torch.backends import mps

# Para um training_refiner_strength de .35, defina a forca do modelo base como 0.65.
# Formula: 1 - training_refiner_strength
training_refiner_strength = 0.35
base_model_power = 1 - training_refiner_strength
# Reduza isto para menor qualidade, mas mais velocidade.
num_inference_steps = 40
# Atualize estes para seus caminhos locais ou do hub do Hugging Face.
stage_1_model_id = 'bghira/terminus-xl-velocity-v2'
stage_2_model_id = 'bghira/terminus-xl-refiner'
torch_device = 'cuda' if cuda.is_available() else 'mps' if mps.is_available() else 'cpu'

pipe = StableDiffusionXLPipeline.from_pretrained(stage_1_model_id, add_watermarker=False, torch_dtype=float16).to(torch_device)
pipe.scheduler = UniPCMultistepScheduler.from_pretrained(stage_1_model_id, subfolder="scheduler")
img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(stage_2_model_id).to(device=torch_device, dtype=float16)
img2img_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(stage_1_model_id, subfolder="scheduler")

prompt = "An astronaut riding a green horse"

# Importante: mude para True se voce reparametrizou os modelos.
use_zsnr = True

image = pipe(
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    denoising_end=base_model_power,
    guidance_scale=9.2,
    guidance_rescale=0.7 if use_zsnr else 0.0,
    output_type="latent",
).images
image = img2img_pipe(
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    denoising_start=base_model_power,
    guidance_scale=9.2,
    guidance_rescale=0.7 if use_zsnr else 0.0,
    image=image,
).images[0]
image.save('demo.png', format="PNG")
```

Algumas experimentacoes que voce pode fazer:
- Brinque com valores como `base_model_power` ou `num_inference_steps`, que devem ser os mesmos para ambos os pipelines.
- Voce tambem pode ajustar `guidance_scale` e `guidance_rescale` de forma diferente em cada estagio. Isso impacta contraste e realismo.
- Use prompts diferentes entre o modelo base e o refiner para mudar o foco de guidance para detalhes finos.
