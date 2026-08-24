# MiniMax Music 3 Quickstart

Este guia configura o SimpleTuner para treinamento LoRA do MiniMax Music 3.

## Visão geral

MiniMax Music 3 é um modelo de geração musical condicionado por legenda e letras. O layout Diffusers usa um modelo de linguagem Qwen3 autoregressivo para condicionamento de texto/áudio, um transformer flow-matching sobre latents DAV de 128 canais, e um decoder/vocoder para áudio de validação.

O SimpleTuner oferece suporte a:

- treinamento LoRA, LyCORIS e full-rank do transformer
- VAECache a partir de áudio bruto usando o autoencoder original `dav.pth`
- caption, lyrics e duration vindos dos metadados do dataset de áudio
- validação com `validation_prompt`, `validation_lyrics`, `validation_audio_duration` e bibliotecas de prompts
- importação/exportação de LoRA ComfyUI MiniMax Music com `lora_format: "comfyui"`
- AnyFlow, TwinFlow, CREPA self-flow e LayerSync

## Requisitos de hardware

MiniMax Music 3 tem um flow transformer de 2.4B e um modelo Qwen3 AR de 8B para condicionamento.

- **Mínimo:** GPU NVIDIA com 24GB+ de VRAM para LoRA conservador.
- **Recomendado:** 48GB+ de VRAM, ou offload para CPU/RAM para ranks maiores, clipes mais longos e validação frequente.
- **Mac:** MPS pode funcionar para partes do stack, mas CUDA é o alvo prático para treinamento e validação.

Comece com `base_model_precision: "int8-quanto"`, `text_encoder_1_precision: "int8-quanto"` e `gradient_checkpointing: true`. Se o text encoder ainda for o gargalo, use offload do text encoder antes de aumentar o LoRA rank.

## Pré-requisitos

Instale o SimpleTuner e o FFmpeg para carregar áudio:

```bash
pip install simpletuner
```

Para instalação manual ou ambiente de desenvolvimento, veja a [documentação de instalação](../INSTALL.md).

## Configuração

Crie uma pasta dedicada:

```bash
mkdir -p config/minimaxmusic-training-demo
```

Crie `config/minimaxmusic-training-demo/config.json`:

<details>
<summary>Ver exemplo de configuração</summary>

```json
{
  "model_family": "minimaxmusic",
  "model_type": "lora",
  "model_flavour": "music3",
  "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-Music3",
  "pretrained_vae_model_name_or_path": "SimpleTuner/MiniMax-Music-3-Encoder",
  "resolution": 512,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "int8-quanto",
  "gradient_checkpointing": true,
  "lora_rank": 64,
  "lora_format": "comfyui",
  "optimizer": "adamw_bf16",
  "learning_rate": 0.00005,
  "train_batch_size": 1,
  "vae_batch_size": 1,
  "data_backend_config": "config/minimaxmusic-training-demo/multidatabackend.json",
  "validation_prompt": "bright synth pop with clean vocal melody and crisp percussion",
  "validation_lyrics": "[verse]\nturning sparks into a skyline\n[chorus]\nwe keep singing through the night",
  "validation_audio_duration": 30,
  "validation_guidance": 1.7,
  "validation_num_inference_steps": 30,
  "validation_steps": 50,
  "validation_disable_unconditional": true
}
```
</details>

Templates prontos estão disponíveis em:

- `simpletuner/examples/minimaxmusic-music3.peft-lora`
- `simpletuner/examples/minimaxmusic-audio.json`
- `simpletuner/examples/minimaxmusic-prompts.json`

Execute o exemplo:

```bash
simpletuner train example=minimaxmusic-music3.peft-lora
```

## VAECache

O cache de áudio bruto do MiniMax Music 3 usa o audio autoencoder DAV. O repositório VAE recomendado do SimpleTuner é `SimpleTuner/MiniMax-Music-3-Encoder`, com o componente convertido em `audio_vae/` para carregamento no estilo Diffusers.

O repositório upstream `MiniMaxAI/MiniMax-Music3` também inclui o `dav.pth` original, e o SimpleTuner pode carregá-lo diretamente. Se usar um diretório Diffusers convertido localmente, mantenha `dav.pth` na raiz do checkpoint ou aponte `pretrained_vae_model_name_or_path` para um caminho ou repositório Hub que contenha `dav.pth` ou um subdiretório `audio_vae/`. Um subdiretório `vocoder/` sozinho serve para decode de validação, mas não para VAE caching de áudio bruto.

## Dataset

MiniMax Music 3 exige um dataset **audio** e um backend de cache **text embeds**.

```json
[
  {
    "id": "minimaxmusic-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "Yi3852/ACEStep-Songs",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "audio": {
      "bucket_strategy": "duration",
      "duration_interval": 3.0,
      "max_duration_seconds": 30
    },
    "cache_dir_vae": "cache/vae/{model_family}/minimaxmusic-demo-data"
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/{model_family}"
  }
]
```

Para arquivos locais, use `.txt` para a descrição e `.lyrics` para a letra:

```text
datasets/minimaxmusic-audio/
├── track_01.wav
├── track_01.txt
└── track_01.lyrics
```

## Validação

- **`validation_prompt`**: descrição musical ou tags.
- **`validation_lyrics`**: letra cantada. Use string vazia para validação instrumental.
- **`validation_audio_duration`**: duração do clipe em segundos.
- **`validation_guidance`**: escala CFG. Comece entre `1.5` e `2.0`.
- **`validation_num_inference_steps`**: passos de sampling. Comece por volta de `30`.
- **`validation_steps`**: frequência de renderização da validação.
- **`validation_prompt_library`**: use `"audio"` para a biblioteca integrada de caption + lyrics musical.
- **`user_prompt_library`**: caminho para uma biblioteca JSON. As entradas podem usar `prompt` ou `caption`, alem de `lyrics` multiline opcional.

## Treinamento

```bash
simpletuner train env=minimaxmusic-training-demo
```

Para começar de um LoRA MiniMax Music 3 existente:

```bash
simpletuner train env=minimaxmusic-training-demo --init_lora=/path/to/adapter.safetensors --init_lora_step=0
```

Se o adapter estiver em formato ComfyUI nativo, mantenha `lora_format: "comfyui"` na configuração. O SimpleTuner converte durante o treinamento e exporta no mesmo formato.

## Recursos avançados

MiniMax Music 3 usa o caminho de treinamento flow-matching do SimpleTuner, então AnyFlow, TwinFlow, CREPA self-flow e LayerSync estão disponíveis. Comece com LoRA padrão e ative um recurso avançado por vez.

## Treinamento do modelo de linguagem (estágio AR)

O modelo de linguagem Qwen3 que planeja os códigos semânticos do MiniMax Music 3 pode ser treinado no lugar do DiT musical — útil para palavras-gatilho estilo dreambooth que vinculam um estilo musical a uma palavra-chave.

Veja [fiona crapple](https://huggingface.co/terminusresearch/minimax-music3-lm-lora-fiona-crapple) para um exemplo completo de treinamento de LM LoRA produzido com este modo, incluindo configurações, checkpoints e comparações de áudio.

```json
{
  "minimax_music_train_component": "language_model",
  "minimax_music_lm_max_frames": 0,
  "minimax_music_lm_window_mode": "prefix"
}
```

Requisitos e diferenças em relação ao treinamento do DiT:

- Cada amostra do dataset deve fornecer `prompt` (ou `tags`), `lyrics` e o metadado `audio_tokens_path` apontando para um arquivo `.pt` de códigos RVQ brutos por codebook com formato `[frames, codebooks]` (códigos semânticos `< 16384`, residuais `< audio_vocab_size`, sem offsets de vocabulário). Exporte-os com `precompute_rvq_codes.py --raw-codes` do repositório dedicado `minimax-music3-latent-replanner`.
- A perda é entropia cruzada de próximo token sobre o codebook semântico, mascarada às posições de áudio; o depth decoder RVQ permanece congelado e fornece os embeddings de entrada dos códigos residuais.
- Apenas LoRA PEFT padrão é suportado e `lora_format: "comfyui"` é rejeitado. Os checkpoints salvam `pytorch_lora_weights.safetensors` com chaves de adaptador prefixadas com `language_model.`.
- O áudio de validação no treinador fica desabilitado neste modo; renderize a partir dos checkpoints salvos com a pilha de geração padrão.
- Não há cache de VAE nem de text embeds neste modo — o treinamento lê os tokens diretamente, então `cache_dir_vae` e backends de text embeds não são usados.
- Coloque sua palavra-gatilho (por exemplo `"fiona crapple"`) no campo caption/`prompt` de cada amostra; mantenha as letras inalteradas.
- Para execuções curtas com limite de frames, use `minimax_music_lm_window_mode: "random"` para amostrar janelas RVQ posicionadas em vez de treinar sempre introduções. Janelas aleatórias adicionam início/fim/duração ao prompt e omitem letras completas, a menos que a amostra forneça `lyrics_window`.
- Não deixe o treinamento com janelas recortadas ensinar cada recorte como um clipe terminado. Se as saídas fizerem fade ou resolverem repetidamente nas bordas do recorte, inspecione os rótulos e alvos do recorte: janelas internas devem ser supervisionadas como janelas internas, e o comportamento de fim de áudio só deve ser ensinado em finais reais de música.
- Para treinar a estrutura da música, use `minimax_music_lm_window_mode: "continuation"`. Ele amostra uma janela-alvo, mantém todos os tokens de áudio desde o início da faixa até essa janela como contexto causal e mascara a perda do contexto anterior. Isso usa mais memória que um recorte aleatório isolado, mas evita ensinar cada trecho como abertura de música.
- Use otimizadores agressivos com cuidado em datasets pequenos de áudio para LM. Prodigy pode passar muito do ponto com learning rates altos, e Lion pode se sobreadaptar dentro dos primeiros mil passos; use AdamW como baseline antes de testar otimizadores mais rápidos.
- **Preservação de prior**: adicione um segundo backend de áudio com `is_regularisation_data: true` contendo instrumentais ou músicas não relacionadas (letras vazias são permitidas). Nesses lotes a perda mira a distribuição de próximo token do modelo base congelado em vez dos códigos reais, mantendo o LoRA cirúrgico: captions de regularização continuam prevendo exatamente como o modelo base faria, reduzindo bastante o vazamento de estilo.

### Como configurar datasets de estilo e cantor

Adaptação de estilo musical e adaptação de identidade vocal precisam de designs de dataset diferentes. Não trate o nome de um cantor como atalho para uma caption musical detalhada.

#### Estilos musicais

Estilos musicais são mais tolerantes. Um conjunto variado de 24 ou mais faixas pode ser suficiente para um adapter útil quando o objetivo é gênero, arranjo ou produção, e não um timbre vocal específico.

- Otimize para diversidade sem sair do estilo-alvo. Inclua tempos, combinações de instrumentos, escolhas de produção, climas e subgêneros próximos que um usuário poderia pedir na inferência.
- Dê várias captions de estilo completas para cada amostra de áudio. Um trigger sozinho comprime o dataset em uma associação média e não ensina os controles necessários para reproduzir sua amplitude.
- Trate o timbre vocal como incidental. Use vários vocalistas ou material instrumental para que uma voz não vire acidentalmente parte do estilo aprendido.
- Observe colapso com prompts de validação fixos e várias forças de checkpoint. Adapters de estilo costumam ficar úteis antes de precisarem de muitos passos.

Com `caption_strategy: "textfile"` e o padrão `disable_multiline_split: false`, cada linha não vazia em um sidecar `.txt` é uma caption candidata separada. O SimpleTuner escolhe uma candidata sempre que amostra aquele áudio; ele não combina todas as linhas em uma caption agrupada. Workflows DiT cacheiam cada caption distinta separadamente, enquanto o treinamento LM tokeniza a caption selecionada online e não usa cache de text embeddings. Por exemplo:

```text
rock artístico sincopado, bateria seca, guitarra angular, mudanças dinâmicas abruptas
metal alternativo melódico, harmonias em camadas, baixo inquieto, andamento teatral
rock progressivo tenso, acentos de métrica irregular, verso esparso, refrão explosivo
```

Isso é aumento de captions, não um prompt multilinha: o modelo vê uma dessas linhas para um exemplo de treinamento.

#### Identidade do cantor

Identidade do cantor é muito menos tolerante. Construa um adapter por cantor e remova toda faixa ou seção que contenha outro vocalista, incluindo duetos, versos alternados, backing leads e participações. Etiquetas de letra como `[Verse: ...]` ou `[Chorus: ...]` não separam vozes de forma confiável.

- Coloque o mesmo trigger único do cantor em cada caption candidata, seguido de uma descrição de estilo completa e variada. Um trigger em uma linha e descrições em linhas separadas está errado porque só uma linha é escolhida por vez.
- Um dataset de cantor estreito e de um único gênero geralmente aprende o cantor dentro daquele arranjo, não uma identidade vocal portátil. O delta de identidade fica entrelaçado com o gênero, a instrumentação, a mixagem e a estrutura de música com que sempre coocorre, então o trigger pode funcionar apenas dentro do domínio. Controle vocal entre gêneros exige variedade real de gênero e arranjo no dataset do cantor.
- Mantenha as letras fiéis, mas não dependa de etiquetas de seção para ensinar identidade. A associação entre áudio e caption carrega o sinal útil.
- Para um corpus muito pequeno, contrapartes instrumentais podem fornecer preservação de prior. Seis faixas vocais cuidadosamente isoladas podem funcionar quando pareadas com regularização construída a partir dessas faixas.

```text
vocalista_xyz, rock alternativo esparso, bateria seca, verso tenso, refrão explosivo
vocalista_xyz, art metal melódico, guitarra em camadas, groove médio, vocal próximo
vocalista_xyz, rock acústico de câmara, percussão manual, abertura suave, elevação dramática
```

Um workflow prático de regularização usa Demucs para remover vocais:

```bash
python -m demucs --two-stems=vocals path/to/track.wav
```

Coloque cada `no_vocals.wav` resultante em um backend de áudio separado com uma caption `.txt` apenas de estilo, sem trigger de cantor, e um sidecar `.lyrics` contendo `[Instrumental]`. Defina `is_regularisation_data: true` nesse backend. Lotes de regularização miram o planner base congelado, ajudando o adapter a separar "esta música" de "este cantor" em vez de reescrever todo o estilo ao redor de um corpus vocal minúsculo.

Para um corpus maior e diverso de um único cantor, comece sem esse branch de regularização e adicione-o apenas se a validação mostrar vazamento de estilo ou dano ao modelo base. A regularização pode retardar a aquisição de identidade quando o dataset vocal já fornece cobertura suficiente. Uma explicação plausível é que o sinal extra de preservação dilui ainda mais um gradiente de identidade já diverso, mas trate isso como hipótese de ajuste, não como regra geral.

## Solução de problemas

- **`VAE caching requires the original dav.pth checkpoint`**: use `SimpleTuner/MiniMax-Music-3-Encoder` ou `MiniMaxAI/MiniMax-Music3`, mantenha `dav.pth` na raiz do checkpoint local ou aponte `pretrained_vae_model_name_or_path` para um local que o contenha.
- **Lyrics ausentes**: confirme que os metadados têm `lyrics`, ou coloque arquivos `.lyrics` ao lado dos áudios ao usar `caption_strategy: "textfile"`.
- **OOM no text embedding ou validação**: reduza `validation_audio_duration`, use int8 no text encoder ou habilite offload do text encoder.

## Experimentos relacionados ao MiniMax Music 3

- [Encoders RVQ abertos](https://huggingface.co/SimpleTuner/open-rvq-encoder-minimax-music3)
- [Integração de áudio de referência RVQ](https://github.com/bghira/minimax-music3-rvq-reference-audio)
- [LoRA do LM Fiona Crapple](https://huggingface.co/terminusresearch/minimax-music3-lm-lora-fiona-crapple)
- [Refinador latente](https://github.com/bghira/minimax-music3-latent-refiner) e [pesos v0.10](https://huggingface.co/terminusresearch/minimax-music3-latent-refiner-v0.10)
- [Replanejador latente](https://github.com/bghira/minimax-music3-latent-replanner) e [registro experimental](https://huggingface.co/terminusresearch/minimax-music3-replanner-experiment)
