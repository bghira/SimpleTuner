## Qwen Image 2.1

Qwen Image 2.1 é o padrão (`model_flavour: "v2.1"`), usando `Qwen/Qwen-Image-2.1`. Ele tem um transformer de 32 blocos, um codificador de texto Qwen3-VL e um VAE de 64 canais com compressão espacial de 16×.

O Qwen Image 2.1 usa o [VAE com correção de textura de Ollin](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1) por padrão na validação e em outras operações de decodificação VAE. Apenas o decodificador foi ajustado; o codificador permanece inalterado, portanto os latentes de treinamento existentes da versão 2.1 continuam compatíveis. A revisão do VAE é fixada independentemente do modelo base. Para reproduzir saídas com o VAE original, defina `pretrained_vae_model_name_or_path: "Qwen/Qwen-Image-2.1"`. Substituições explícitas do VAE e variantes anteriores mantêm o comportamento existente.

O exemplo `qwen_image.peft-lora` treina com `RareConcepts/Domokun` em 512px, usando o gatilho `🟫`. Comece com BF16 (`base_model_precision: "no_change"`) e use checkpointing de gradientes quando faltar memória. O exemplo usa caches de latentes e texto separados para a versão 2.1; não reutilize caches de versões anteriores.

```bash
simpletuner train example=qwen_image.peft-lora
```

Para validação, use `validation_guidance: 1.0`, `validation_guidance_real: 1.0` e `validation_num_inference_steps: 40`. Mantenha o gatilho nos prompts de validação para verificar se o conceito foi aprendido.

Qwen Image 2.1 decodifica imagens individuais sem manter caches temporais de características que não são utilizados. Na decodificação isolada de uma imagem de 2048×2048 em H200 com BF16, isso reduziu o pico de memória alocada de 26.87 para 15.28 GiB com saída idêntica. A decodificação em blocos economiza mais memória, mas pode introduzir linhas de cor ao limitar o contexto espacial; remover os caches sem uso não corrige essas linhas.

As versões anteriores continuam disponíveis: `v1.0` seleciona Qwen-Image, `v2.0` seleciona Qwen-Image-2512 e as variantes `edit-*` mantêm os checkpoints existentes. Seus adaptadores e caches de latentes não são intercambiáveis com os da versão 2.1.

A receita Domokun de 250 steps serve para medir throughput, não como receita de convergência confiável. Um checkpoint anterior gerou Domokun reconhecível após recarregar, mas novas execuções de 250 steps não reproduziram o resultado. Os controles mantendo máscaras de padding, desativando compilação e restaurando a expressão RoPE anterior também falharam. Os latentes em cache decodificam o personagem correto. A causa da deterioração continua sem resolução; as tabelas de tempo não demonstram qualidade equivalente entre backends de atenção.

### Configurações por VRAM

Os exemplos usam BF16, LoRA de rank 32, Optimi Lion e compilação regional a 512px, sem checkpoint de gradientes. A primeira execução inclui compilação; compare os passos após o aquecimento. Os limites de 24 GB e 32 GB foram verificados na L40S, não em placas separadas dessas capacidades.

| VRAM | Exemplo | Lote do dataset | Pico de VRAM (GiB) | Passo aquecido (s) |
| --- | --- | --- | --- | --- |
| 24 GB | `qwen_image-2.1-24g.peft-lora` | 1 | 20.6 | 0.238 |
| 32 GB | `qwen_image-2.1-32g.peft-lora` | 2 | 26.5 | 0.390 |
| 48 GB | `qwen_image-2.1-48g.peft-lora` | 2 | 26.5 | 0.390 |
| 80 GB | `qwen_image-2.1-80g.peft-lora` | 10 | 71.8 | 0.639 |
| 144 GB | `qwen_image-2.1-144g.peft-lora` | 20 | 128.4 | 1.223 |

Medições na L40S (configurações de 24/32/48 GB), H100 (80 GB) e H200 (144 GB), com 20 passos e os cinco primeiros excluídos do tempo. O pico de VRAM inclui a preparação. São resultados a 512px para cada lote, não garantias para imagens maiores ou prompts mais longos.

A configuração de 48 GB também usa lote 2: na L40S ele teve melhor rendimento por imagem que os lotes 3, 4 e 5. O lote 5 coube em 43.3 GiB, mas levou 0.991 s/passo, contra 0.390 s/passo com lote 2.

```bash
simpletuner train example=qwen_image-2.1-48g.peft-lora
```

Use o arquivo de dataset incluído em cada exemplo: ele define explicitamente o tamanho do lote. Inicie um novo treino ao alterar o lote ou as configurações do dataset; não reutilize um checkpoint de estado incompatível.

Para reduzir a VRAM, habilite `gradient_checkpointing: true` e `gradient_checkpointing_interval: 2`. Agora isso aplica checkpoint a grupos de dois blocos contíguos. Consulte as [medições de checkpoint e atenção do Qwen Image 2.1](../experimental/SEGMENTED_CHECKPOINTING.pt-BR.md#qwen-image-21); o resultado anterior com blocos alternados foi substituído. BF16 cabe nesses presets sem um checkpoint int8.

O caminho texto-para-imagem evita montar sequências com controle dependente dos valores dos tensores, permitindo captura sem quebras de grafo. RoPE com aritmética real permite ao Inductor fundir normalização e rotação; os epílogos de modulação, residual e MLP também são compilados. Os kernels existentes de GEMM CuTe ConvRot para Hopper e de RoPE do LTX apenas para inferência não são usados nestes exemplos de treino.


### Piloto experimental de AnyFlow

Os três exemplos `qwen_image-2.1-anyflow-stage*.peft-lora` testam se a destilação de intervalos pode preservar o comportamento da base ao introduzir `🟫`. São experimentais; o model card do Qwen não confirma que a destilação de guidance causou a deterioração anterior.

Todas as etapas usam 1024px, BF16, AdamW, rank 32 e batch 4. A etapa 1 desativa o checkpointing de gradientes no H200; as etapas 2 e 3 usam grupos contíguos de dois blocos. A carga difere dos presets de desempenho a 512px acima. Instale `webshart` antes de executar.

Estes exemplos de AnyFlow usam explicitamente `grad_clip_method: "value"` com `max_grad_norm: 0.01`: cada elemento do gradiente é limitado a ±0.01. Isso não limita a norma global. Para testar o clipping por norma em 1.0, defina tanto `grad_clip_method: "norm"` quanto `max_grad_norm: 1.0`.

1. **Etapa 1:** 10.000 atualizações forward de AnyFlow a `1e-5`. CC12M usa `webshart/cc12m-structured-captions` com `caption_key: "long_caption"`; e621 usa `webshart/e621-2024-webp-4Mpixel-webshart-indices`. Cada dataset de conhecimentos prévios fica limitado a 4.096 imagens aceitas, com peso 0.49 cada. `RareConcepts/Domokun` usa o gatilho `🟫`, peso 0.02 e `repeats: 0`.
2. **Etapa 2:** 2.000 atualizações DMD on-policy a `2e-6`, co-treinando o objetivo forward com a mesma mistura. É DMD de AnyFlow, não DPO com pares de preferência. Incluir o personagem nas duas etapas fornece exemplos reais; o professor congelado sozinho não pode ensiná-lo.
3. **Etapa 3:** 100 atualizações a `5e-7`, com peso 0.5 para Domokun e 0.25 para cada um dos dois datasets de regularização, limitados a 64 imagens cada. Todos os intervalos usam `r=t` e o alvo flow bruto, mantendo o embedder de intervalos durante um breve ajuste supervisionado. Os batches de regularização usam a previsão da base com o adapter desativado.

Revise os checkpoints de 2.000, 5.000 e 10.000 atualizações antes de estender a etapa 1 para 20.000. A etapa 2 tem um orçamento de 2.000 atualizações; pare antes se as imagens de validação comparáveis piorarem. A contagem de passos, sozinha, não determina um modelo final.

A compilação dinâmica está habilitada para captions de comprimentos variados. `schedule_shift: 2.000802574061872` corresponde ao scheduler publicado a 1024px (4.096 tokens latentes); recalcule ao alterar a resolução.

```bash
simpletuner train example=qwen_image-2.1-anyflow-stage1.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage2.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage3.peft-lora
```

As etapas 2 e 3 carregam o adapter final anterior com `init_lora` e iniciam novos estados do otimizador e dataloader. Os pesos controlam a seleção entre datasets ainda disponíveis, sem garantir porcentagens finais. O piloto limitado não cobre os corpora completos. O campo textual `long_caption` funciona com o seletor de captions Webshart existente e não exige suporte a captions como objetos JSON nativos.

As etapas 1 e 2 usam `diffusion_target: "base_prediction"` e `fuse_guidance_scale: 1.0`: o ramo diffusion preserva o campo condicional congelado, enquanto os outros ramos aprendem com os dados. É uma hipótese de preservação, não uma garantia de aprendizado do personagem. Compare os prompts de personagem e de conhecimentos prévios em 4, 16 e 40 passos de inferência com uma referência da base em 40 passos; a validação automática usa 4. Consulte [AnyFlow](../experimental/ANYFLOW.pt-BR.md) para os objetivos e requisitos de checkpoints.

Piloto concluído: a etapa 1 chegou a 10.000 atualizações e a etapa 2 a 2.000. Após recarregar os adaptadores, as imagens de 40 passos da raposa e dos prompts do personagem continuaram coerentes, mas estes últimos produziram pessoas em vez de Domokun. As imagens de quatro passos permaneceram borradas ou ruidosas. Outra comparação na L40S usou os dois checkpoints em 1024px, seed 42 e CFG 1/2/4/6, com prompt negativo vazio. Asserções de chamadas confirmaram duas passagens por passo acima de CFG 1. As amostras revisadas da raposa e da praia não foram recuperadas: aumentar CFG intensificou saturação e artefatos. Esses resultados não identificam a causa; a etapa 3 ainda não foi validada.

Duas continuações da etapa 1 a partir do mesmo checkpoint adicionaram 1.000 atualizações cada, comparando limites de clipping por valor de 0.01 e 1.0. Após recarregar, as imagens da raposa com quatro passos continuaram ruidosas em ambas; as imagens de praia com 40 passos ainda mostraram pessoas. O clipping ocorreu em 65/1.000 atualizações com 0.01 e em 0/1.000 com 1.0. Ambas usaram o otimizador sem correção, e os gradientes iniciais diferiram antes de o clipping ser aplicado; pequenas diferenças não podem ser atribuídas apenas ao limite.

<a id="qwen21-optimizer-correction"></a>

Os pilotos da etapa 1 e do assistente descritos aqui foram executados antes da correção do helper de soma estocástica do AdamW BF16: ele calculava `other + alpha * input` em vez de `input + alpha * other`. Com β₁ = 0,9, o primeiro momento seguia `m = 0.09 * m + g` em vez de `m = 0.9 * m + 0.1 * g`. Testes de regressão com aritmética exata cobrem CPU, MPS e CUDA. As observações das imagens continuam válidas para esses checkpoints, mas não são um teste isolado da arquitetura ou do objetivo de destilação; o treinamento precisa ser verificado novamente com o otimizador corrigido.

Repetir a continuação de 1.000 atualizações com o otimizador corrigido, o mesmo checkpoint inicial e clipping por valor de 1.0 não recuperou as imagens revisadas da raposa ou da praia com quatro passos. Com 40 passos a raposa permaneceu coerente, enquanto o prompt de praia ainda gerou uma pessoa. Isso testa a recuperação do checkpoint existente, não o treinamento do zero com o otimizador corrigido; a causa da falha geral continua sem solução.

### Configuração do Qwen Image anterior (v1.0 / v2.0)

> 🆕 Procurando os checkpoints de edição? Veja o [guia de Início Rápido do Qwen Image Edit](./QWEN_EDIT.md) para instruções de treino com referência pareada.

Neste exemplo, vamos treinar um LoRA para o Qwen Image, um modelo visão-linguagem de 20B parâmetros. Devido ao tamanho, precisaremos de técnicas agressivas de otimização de memória.

Uma GPU de 24GB é o mínimo absoluto e, mesmo assim, você precisará de quantização extensa e configuração cuidadosa. 40GB+ é fortemente recomendado para uma experiência mais tranquila.

Ao treinar em 24G, as validações vão dar OOM a menos que você use resolução menor ou quantização agressiva além de int8.

### Requisitos de hardware

Qwen Image é um modelo de 20B parâmetros com um encoder de texto sofisticado que, sozinho, consome ~16GB de VRAM antes de quantização. O modelo usa um VAE customizado com 16 canais latentes.

**Limitações importantes:**
- **Não suportado em AMD ROCm ou MacOS** devido à falta de flash attention eficiente
- Batch size > 1 não funciona corretamente no momento; use gradient accumulation em vez disso
- TREAD (Text-Representation Enhanced Adversarial Diffusion) ainda não é suportado

### Pré-requisitos

Certifique-se de que você tem Python instalado; o SimpleTuner funciona bem com 3.10 até 3.12.

Você pode verificar executando:

```bash
python --version
```

Se você não tem o Python 3.12 instalado no Ubuntu, pode tentar o seguinte:

```bash
apt -y install python3.13 python3.13-venv
```

#### Dependências da imagem de contêiner

Para Vast, RunPod e TensorDock (entre outros), o seguinte funciona em uma imagem CUDA 12.2-12.8 para habilitar a compilação de extensões CUDA:

```bash
apt -y install nvidia-cuda-toolkit
```

### Instalação

Instale o SimpleTuner via pip:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

Para instalação manual ou setup de desenvolvimento, veja a [documentação de instalação](../INSTALL.md).

### Configurando o ambiente

Para rodar o SimpleTuner, você precisará configurar um arquivo de configuração, os diretórios de dataset e modelo, e um arquivo de configuração do dataloader.

#### Arquivo de configuração

Um script experimental, `configure.py`, pode permitir que você pule esta seção inteiramente por meio de uma configuração interativa passo a passo. Ele contém alguns recursos de segurança que ajudam a evitar armadilhas comuns.

**Nota:** Isso não configura seu dataloader. Você ainda precisará fazer isso manualmente depois.

Para executá-lo:

```bash
simpletuner configure
```

> ⚠️ Para usuários localizados em países onde o Hugging Face Hub não é facilmente acessível, você deve adicionar `HF_ENDPOINT=https://hf-mirror.com` ao seu `~/.bashrc` ou `~/.zshrc` dependendo de qual `$SHELL` seu sistema usa.

Se você preferir configurar manualmente:

Copie `config/config.json.example` para `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Lá, você provavelmente precisará modificar as seguintes variáveis:

- `model_type` - Defina como `lora`.
- `lora_type` - Defina como `standard` para PEFT LoRA ou `lycoris` para LoKr.
- `model_family` - Defina como `qwen_image`.
- `model_flavour` - Defina como `v1.0`.
- `output_dir` - Defina como o diretório onde você quer armazenar seus checkpoints e imagens de validação. É recomendado usar um caminho completo aqui.
- `train_batch_size` - Ajuste de acordo com a VRAM disponível. Os overrides atuais de Qwen no SimpleTuner suportam batch sizes maiores que 1.
- `gradient_accumulation_steps` - Configure em 2-8 se quiser um batch efetivo maior sem aumentar a VRAM por passo.
- `validation_resolution` - Defina `1024x1024` ou menos por restrições de memória.
  - 24G não aguenta validações 1024x1024 atualmente - você precisará reduzir o tamanho
  - Outras resoluções podem ser especificadas usando vírgulas: `1024x1024,768x768,512x512`
- `validation_guidance` - Use um valor em torno de 3.0-4.0 para bons resultados.
- `validation_num_inference_steps` - Use algo em torno de 30.
- `use_ema` - Definir como `true` ajuda a obter resultados mais suaves, mas usa mais memória.

- `optimizer` - Use `optimi-lion` para bons resultados, ou `adamw-bf16` se tiver memória de sobra.
- `mixed_precision` - Deve ser definido como `bf16` para o Qwen Image.
- `gradient_checkpointing` - **Obrigatório** habilitar (`true`) para uso de memória aceitável.
- `base_model_precision` - **Fortemente recomendado** definir `int8-quanto` ou `nf4-bnb` para placas de 24GB.
- `quantize_via` - Defina como `cpu` para evitar OOM durante a quantização em GPUs menores.
- `quantize_activations` - Mantenha como `false` para preservar qualidade de treino.

Configurações de otimização de memória para GPUs 24GB:
- `lora_rank` - Use 8 ou menos.
- `lora_alpha` - Igual ao valor de lora_rank.
- `flow_schedule_shift` - Defina como 1.73 (ou experimente entre 1.0-3.0).

Seu config.json vai ficar mais ou menos assim para um setup mínimo:

<details>
<summary>Ver exemplo de config</summary>

```json
{
    "model_type": "lora",
    "model_family": "qwen_image",
    "model_flavour": "v1.0",
    "lora_type": "standard",
    "lora_rank": 8,
    "lora_alpha": 8,
    "output_dir": "output/models-qwen_image",
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "validation_resolution": "1024x1024",
    "validation_guidance": 4.0,
    "validation_num_inference_steps": 30,
    "validation_seed": 42,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_step_interval": 100,
    "vae_batch_size": 1,
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 1000,
    "max_grad_norm": 0.01,
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant_with_warmup",
    "learning_rate": "1e-4",
    "gradient_checkpointing": "true",
    "base_model_precision": "int2-quanto",
    "quantize_via": "cpu",
    "quantize_activations": false,
    "flow_schedule_shift": 1.73,
    "disable_benchmark": false,
    "data_backend_config": "config/qwen_image/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "aspect_bucket_rounding": 2
}
```
</details>

> ℹ️ Usuários multi-GPU podem consultar [este documento](../OPTIONS.md#environment-configuration-variables) para informações sobre como configurar o número de GPUs a usar.

> ⚠️ **Crítico para GPUs 24GB**: O encoder de texto sozinho usa ~16GB de VRAM. Com quantização `int2-quanto` ou `nf4-bnb`, isso pode ser reduzido significativamente.

Para um sanity check rápido com uma configuração conhecida:

**Opção 1 (Recomendado - pip install):**
```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
simpletuner train example=qwen_image.peft-lora
```

**Opção 2 (Método Git clone):**
```bash
simpletuner train env=examples/qwen_image.peft-lora
```

**Opção 3 (Método legado - ainda funciona):**
```bash
ENV=examples/qwen_image.peft-lora ./train.sh
```

### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz viés de exposição e melhora a qualidade ao permitir que o modelo gere suas próprias entradas durante o treinamento.

> ⚠️ Esses recursos aumentam o overhead computacional do treinamento.

#### Prompts de validação

Dentro de `config/config.json` está o "prompt de validação principal", que normalmente é o instance_prompt principal que você está treinando para seu único sujeito ou estilo. Além disso, um arquivo JSON pode ser criado contendo prompts extras para rodar durante validações.

O arquivo de exemplo `config/user_prompt_library.json.example` contém o seguinte formato:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

Os nicknames são o nome do arquivo para a validação, então mantenha-os curtos e compatíveis com seu sistema de arquivos.

Para apontar o trainer para essa biblioteca de prompts, adicione ao seu config.json:
```json
  "validation_prompt_library": "config/user_prompt_library.json",
```

Um conjunto de prompts diversos ajudará a determinar se o modelo está aprendendo corretamente:

```json
{
    "anime_style": "a breathtaking anime-style portrait with vibrant colors and expressive features",
    "chef_cooking": "a high-quality, detailed photograph of a sous-chef immersed in culinary creation",
    "portrait": "a lifelike and intimate portrait showcasing unique personality and charm",
    "cinematic": "a cinematic, visually stunning photo with dramatic and captivating presence",
    "elegant": "an elegant and timeless portrait exuding grace and sophistication",
    "adventurous": "a dynamic and adventurous photo captured in an exciting moment",
    "mysterious": "a mysterious and enigmatic portrait shrouded in shadows and intrigue",
    "vintage": "a vintage-style portrait evoking the charm and nostalgia of a bygone era",
    "artistic": "an artistic and abstract representation blending creativity with visual storytelling",
    "futuristic": "a futuristic and cutting-edge portrayal set against advanced technology"
}
```

#### Rastreamento de score CLIP

Se você quiser habilitar avaliações para pontuar o desempenho do modelo, veja [este documento](../evaluation/CLIP_SCORES.md) para informações sobre como configurar e interpretar scores CLIP.

#### Perda de avaliação estável

Se você quiser usar perda MSE estável para pontuar o desempenho do modelo, veja [este documento](../evaluation/EVAL_LOSS.md) para informações sobre como configurar e interpretar avaliação de loss.

#### Prévias de validação

SimpleTuner suporta streaming de prévias intermediárias de validação durante a geração usando modelos Tiny AutoEncoder. Isso permite ver imagens de validação sendo geradas passo a passo em tempo real via callbacks de webhook.

Para habilitar:
```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```

**Requisitos:**
- Configuração de webhook
- Validação habilitada

Defina `validation_preview_steps` para um valor maior (por exemplo, 3 ou 5) para reduzir o overhead do Tiny AutoEncoder. Com `validation_num_inference_steps=20` e `validation_preview_steps=5`, você receberá imagens de prévia nos steps 5, 10, 15 e 20.

#### Ajuste de schedule de flow

Qwen Image, como modelo de flow-matching, suporta shift do schedule de timesteps para controlar quais partes do processo de geração são treinadas.

O parâmetro `flow_schedule_shift` controla isso:
- Valores baixos (0.1-1.0): foco em detalhes finos
- Valores médios (1.0-3.0): treino equilibrado (recomendado)
- Valores altos (3.0-6.0): foco em grandes aspectos composicionais

##### Auto-shift
Você pode habilitar o shift dependente de resolução com `--flow_schedule_auto_shift`, que usa valores de shift maiores para imagens maiores e menores para imagens menores. Isso pode dar resultados estáveis, mas potencialmente medianos.

##### Especificação manual
Um valor de `--flow_schedule_shift` de 1.73 é recomendado como ponto de partida para Qwen Image, embora você possa precisar experimentar baseado no seu dataset e objetivos.

#### Considerações sobre o dataset

É crucial ter um dataset substancial para treinar seu modelo. Existem limitações no tamanho do dataset, e você precisa garantir que seu dataset seja grande o suficiente para treinar de forma eficaz.

> ℹ️ Com poucas imagens, você pode ver a mensagem **no images detected in dataset** - aumentar o valor de `repeats` vai superar essa limitação.

> ⚠️ **Importante**: Devido às limitações atuais, mantenha `train_batch_size` em 1 e use `gradient_accumulation_steps` para simular batch sizes maiores.

Crie um documento `--data_backend_config` (`config/multidatabackend.json`) contendo isto:

```json
[
  {
    "id": "pseudo-camera-10k-qwen",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/pseudo-camera-10k",
    "instance_data_dir": "datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery",
    "repeats": 0,
    "is_regularisation_data": true
  },
  {
    "id": "dreambooth-subject",
    "type": "local",
    "crop": false,
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/qwen_image",
    "disabled": false,
    "write_batch_size": 16
  }
]
```

> ℹ️ Use `caption_strategy=textfile` se você tiver arquivos `.txt` contendo legendas.
> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).
> ℹ️ Observe o `write_batch_size` reduzido para text embeds para evitar OOM.

Depois, crie um diretório `datasets`:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

Isso vai baixar cerca de 10k amostras de fotografias para o diretório `datasets/pseudo-camera-10k`, que será criado automaticamente.

Suas imagens de Dreambooth devem ir para o diretório `datasets/dreambooth-subject`.

#### Login no WandB e Huggingface Hub

Você vai querer fazer login no WandB e no HF Hub antes de iniciar o treinamento, especialmente se estiver usando `--push_to_hub` e `--report_to=wandb`.

Se você pretende enviar itens para um repositório Git LFS manualmente, também deve executar `git config --global credential.helper store`.

Execute os seguintes comandos:

```bash
wandb login
```

e

```bash
huggingface-cli login
```

Siga as instruções para fazer login em ambos os serviços.

</details>

### Executando o treinamento

A partir do diretório do SimpleTuner, basta executar:

```bash
./train.sh
```

Isso vai iniciar o cache em disco das embeddings de texto e saídas do VAE.

Para mais informações, veja os documentos do [dataloader](../DATALOADER.md) e do [tutorial](../TUTORIAL.md).

### Dicas de otimização de memória

#### Configuração de menor VRAM (mínimo 24GB)

A configuração de menor VRAM do Qwen Image exige cerca de 24GB:

- SO: Ubuntu Linux 24
- GPU: um único dispositivo NVIDIA CUDA (mínimo 24GB)
- Memória do sistema: 64GB+ recomendado
- Precisão do modelo base:
  - Para sistemas NVIDIA: `int2-quanto` ou `nf4-bnb` (obrigatório para placas de 24GB)
  - `int4-quanto` pode funcionar, mas pode ter qualidade menor
- Otimizador: `optimi-lion` ou `bnb-lion8bit-paged` para eficiência de memória
- Resolução: comece com 512px ou 768px, suba para 1024px se a memória permitir
- Batch size: 1 (obrigatório devido às limitações atuais)
- Gradient accumulation steps: 2-8 para simular batches maiores
- Habilite `--gradient_checkpointing` (obrigatório)
- Use `--quantize_via=cpu` para evitar OOM na inicialização
- Use um rank LoRA pequeno (1-8)
- Definir a variável de ambiente `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ajuda a minimizar uso de VRAM

**NOTA**: O pré-cache de embeddings do VAE e saídas do encoder de texto vai usar memória significativa. Habilite `offload_during_startup=true` se você encontrar OOM.

### Rodando inferência no LoRA depois

Como o Qwen Image é um modelo mais novo, aqui está um exemplo funcional de inferência:

<details>
<summary>Mostrar exemplo de inferência em Python</summary>

```python
import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration

model_id = 'Qwen/Qwen-Image'
adapter_id = 'your-username/your-lora-name'

# Load the pipeline
pipeline = QwenImagePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

# Load LoRA weights
pipeline.load_lora_weights(adapter_id)

# Optional: quantize the model to save VRAM
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

# Move to device
pipeline.to('cuda' if torch.cuda.is_available() else 'cpu')

# Generate an image
prompt = "Your test prompt here"
negative_prompt = 'ugly, cropped, blurry, low-quality, mediocre average'

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=4.0,
    generator=torch.Generator(device='cuda').manual_seed(42),
    width=1024,
    height=1024,
).images[0]

image.save("output.png", format="PNG")
```
</details>

### Notas e dicas de troubleshooting

#### Limitações de batch size

Builds antigos do Qwen no diffusers tinham problemas com batch size > 1 por causa do padding dos embeddings de texto e do mascaramento de atenção. Os overrides atuais de Qwen no SimpleTuner corrigem os dois pontos, então batches maiores funcionam se a sua VRAM permitir.
- Aumente `train_batch_size` somente depois de confirmar a folga de memória.
- Se ainda aparecerem artefatos em uma instalação antiga, atualize e regenere quaisquer text embeds antigos.

#### Quantização

- `int2-quanto` oferece a economia de memória mais agressiva, mas pode impactar a qualidade
- `nf4-bnb` oferece bom equilíbrio entre memória e qualidade
- `int4-quanto` é uma opção intermediária
- Evite `int8` a menos que você tenha 40GB+ de VRAM

#### Taxas de aprendizado

Para treino de LoRA:
- LoRAs pequenas (rank 1-8): use learning rates em torno de 1e-4
- LoRAs maiores (rank 16-32): use learning rates em torno de 5e-5
- Com otimizador Prodigy: comece em 1.0 e deixe adaptar

#### Artefatos de imagem

Se você encontrar artefatos:
- Reduza sua taxa de aprendizado
- Aumente gradient accumulation steps
- Garanta que suas imagens são de alta qualidade e bem pré-processadas
- Considere usar resoluções menores inicialmente

#### Treino com múltiplas resoluções

Comece o treino em resoluções menores (512px ou 768px) para acelerar o aprendizado inicial, depois faça fine-tune em 1024px. Habilite `--flow_schedule_auto_shift` ao treinar em diferentes resoluções.

### Limitações de plataforma

**Não suportado em:**
- AMD ROCm (falta implementação eficiente de flash attention)
- Apple Silicon/MacOS (limitações de memória e atenção)
- GPUs de consumidor com menos de 24GB de VRAM

### Problemas conhecidos atuais

1. Batch size > 1 não funciona corretamente (use gradient accumulation)
2. TREAD ainda não é suportado
3. Alto uso de memória do encoder de texto (~16GB antes da quantização)
4. Problemas de manuseio de comprimento de sequência ([issue upstream](https://github.com/huggingface/diffusers/issues/12075))

Para ajuda adicional e troubleshooting, consulte a [documentação do SimpleTuner](/documentation) ou entre no Discord da comunidade.

<a id="assistant-lora"></a>

### Treinar uma LoRA auxiliar a partir de legendas

`qwen_image-2.1-assistant-lora.peft-lora` é um ponto de partida experimental para L40S: BF16, lote 1, checkpointing com intervalo 2, rank 32 e AdamW BF16 a `1e-4`. O orçamento é de 1.000 atualizações, com validação e salvamento a cada 50. Substitua as doze legendas de teste por textos diversos antes de um treino efetivo; a convergência ainda não foi validada.

`grad_clip_method: "norm"`, `max_grad_norm: 1.0`.

Defina `distillation_method: assistant_lora`. O backend pré-calcula os embeddings de texto. Cada lote gera novos latentes do modelo base com o adaptador desativado, 40 passos nativos e CFG 1. O pipeline privado compartilha o transformer sem carregar VAE, processador ou codificador de texto. Depois restaura o adaptador e realiza o treino normal de remoção de ruído. Os latentes finais não são armazenados. Atualmente, somente texto para imagem com Qwen Image 2.1 é compatível.

`distillation_config.assistant_lora` aceita `num_inference_steps` (padrão 40), `resolutions` (lista não vazia de `[largura, altura]`, padrão `[[1024, 1024]]`) e `seed` (42). As dimensões devem ser múltiplas de 32. As resoluções alternam por lote e as sementes avançam por amostra, incluindo lotes incompletos. Os checkpoints preservam esses contadores. Retome sem mudar dados, lote, acumulação ou topologia distribuída. Cache de texto sob demanda não é compatível.

Para testar a execução, use 8 atualizações e 2 passos do professor; volte a 40 antes de avaliar imagens. Revise nas atualizações 100, 250, 500 e 1.000, comparando os mesmos prompts e sementes do modelo base. A utilidade da LoRA auxiliar precisa de outro teste de treino de conceitos.

O treino LoRA do Qwen Image 2.1 carrega por padrão o [assistente v2](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v2), congelado durante o treino e desativado na validação. Use `disable_assistant_lora: true` para desativá-lo, ou `assistant_lora_path` para escolher outro adaptador. As versões anteriores mantêm seu comportamento. Ao treinar um novo assistente, mantenha `disable_assistant_lora: true`, como nos exemplos correspondentes.

Use este como único método de destilação; não há suporte para combiná-lo com outros destiladores.

Datasets de legendas exigem `dataloader_prefetch: false` para que o cursor do checkpoint corresponda às legendas consumidas. A retomada rejeita mudanças nos identificadores/textos, lote, repetições, embaralhamento, semente, acumulação ou configuração distribuída. Os checkpoints do auxiliar também rejeitam mudanças na semente de geração, na lista de resoluções ou no número de passos de inferência do professor.

<a id="assistant-lora-multires"></a>

#### Experimento de assistente com quatro resoluções base e buckets de proporção

`qwen_image-2.1-assistant-lora-multires.peft-lora` percorre 12 buckets de proporção nas resoluções base 512, 1024, 1536 e 2048. Cada base inclui um bucket quadrado, um retrato 4:7 e uma paisagem 7:4; cada lote usa um bucket, com exposição igual por base e proporção durante um ciclo completo. Mantém os 40 passos do professor, BF16 com lote 1, checkpointing a cada 2 blocos, rank 32, AdamW BF16 e 1.000 atualizações do exemplo base, compartilhando seu backend de captions e seus prompts de validação. Substitua as captions de teste pelo mesmo conjunto diverso usado na referência. Comece com um adaptador e um otimizador novos no novo diretório de saída; `resume_from_checkpoint: ""` desativa a retomada. Preserve os pesos da primeira execução para comparação.

O experimento testa se a exposição a vários tamanhos melhora o assistente; não estabelece uma exigência de resolução nativa nem um ganho de qualidade. A execução de 1.000 atualizações na L40S completou os 12 buckets, com validação em 1024×1024 e 2048×2048. As imagens finais da raposa e do retrato permaneceram coerentes, com as conhecidas linhas de cor da decodificação do VAE com tiling. O professor gera alvos latentes sem decodificação pelo VAE. O benefício no treinamento posterior de conceitos continua sem verificação.

Assistant LoRA usa o mesmo mínimo de 32 entradas de cache Dynamo, restrito à execução, que AnyFlow para as variantes do professor, aluno e validação. Limites maiores definidos pelo usuário são preservados, e o limite original é restaurado ao sair. Monitore recompilações ao adicionar resoluções ou captions mais longas.

Um teste posterior de Domokun treinou dois adaptadores novos por 250 atualizações cada a 2048px, batch 1, LR `1e-5` e clipping por valor de 1.0. A intensidade do assistente no treinamento foi 0 no controle e 1 na outra execução; ambas desativaram o assistente na inferência. Os pesos iniciais e as imagens iniciais de validação eram idênticos. Com 40 passos de inferência a 1024px, ambos os resultados finais ainda geraram pessoas para os prompts do personagem e mantiveram raposas/retratos coerentes; nenhum benefício do assistente foi demonstrado. As duas execuções e a preparação do assistente usaram o otimizador sem correção descrito [acima](#qwen21-optimizer-correction).

<a id="assistant-lora-offline"></a>

#### LoRA auxiliar com imagens geradas reutilizáveis

`qwen_image-2.1-assistant-lora-offline.peft-lora` treina com [10.000 imagens geradas](https://huggingface.co/datasets/webshart/qwen-image-2.1-generated-images) via Webshart. O dataset usa prompts `long_caption` do CC12M, 40 passos nativos do professor, CFG 1 e decodificação VAE da imagem inteira. Doze backends cobrem buckets quadrados, verticais e horizontais nas resoluções base 512, 1024, 1536 e 2048, com pesos de amostragem iguais e sem repetições.

Esta receita inicia um treinamento novo com `adamw_bf16` corrigido, LR `1e-4`, `grad_clip_method: "norm"`, `max_grad_norm: 1.0`, BF16 com lote 1, rank 32 e checkpointing a cada 2 blocos. São 1.000 atualizações, salvamento a cada 50 e validação a cada 100 em 1024. Gere prévias adicionais em 2048px antes de publicar. O tiling do VAE fica desativado e a codificação usa lote 1. `vae_cache_ondemand: true` codifica e armazena as imagens conforme são amostradas, evitando codificar todas as 10.000 imagens antes de 1.000 atualizações. O treinamento comum com imagens substitui a geração online do professor; omita `distillation_method` e mantenha `disable_assistant_lora: true` ao criar o auxiliar.

O dataset pode ser reutilizado. Os PNGs exigem nova codificação VAE, portanto os alvos não são idênticos aos latentes finais do professor. Inspecione as validações antes de publicar e teste o benefício do auxiliar em outro treinamento de conceito. Ao trocar a receita de captions por esta, comece do zero sem retomar estados do otimizador ou do dataset.

O [assistente v1 substituto](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v1) concluiu esta receita de 1.000 atualizações na L40S, com revisão final das imagens em 1024 e 2048. Uma comparação equivalente de Domokun com 1.000 atualizações em 2048, LR `1e-4` e clipping de norma 1.0 manteve o assistente congelado no treinamento e desativado na validação. Tanto o controle quanto o treino assistido ainda geraram pessoas para os dois prompts do personagem. O controle introduziu traços fortes de Domokun em um prompt não relacionado de raposa; o treino assistido preservou uma raposa reconhecível. Ambos preservaram um retrato coerente. Isso é evidência limitada de menor contaminação entre conceitos, não de aprendizado bem-sucedido do personagem ou de benefício geral de qualidade; a avaliação usa uma semente de treinamento e quatro prompts.

```bash
simpletuner train example=qwen_image-2.1-assistant-lora-offline.peft-lora
```
