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

## Solução de problemas

- **`VAE caching requires the original dav.pth checkpoint`**: use `SimpleTuner/MiniMax-Music-3-Encoder` ou `MiniMaxAI/MiniMax-Music3`, mantenha `dav.pth` na raiz do checkpoint local ou aponte `pretrained_vae_model_name_or_path` para um local que o contenha.
- **Lyrics ausentes**: confirme que os metadados têm `lyrics`, ou coloque arquivos `.lyrics` ao lado dos áudios ao usar `caption_strategy: "textfile"`.
- **OOM no text embedding ou validação**: reduza `validation_audio_duration`, use int8 no text encoder ou habilite offload do text encoder.
