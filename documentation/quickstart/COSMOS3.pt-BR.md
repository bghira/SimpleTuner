# Quickstart do Cosmos3

Treine um LyCORIS LoKr para NVIDIA Cosmos3.

## Notas do modelo

- `model_family`: `cosmos3`
- `model_flavour` padrao: `nano`
- Flavours suportados:

| Flavour | Modelo Hub | Notas |
| --- | --- | --- |
| `nano` | `nvidia/Cosmos3-Nano` | modelo omni 16B |
| `super` | `nvidia/Cosmos3-Super` | modelo omni 65B |
| `super-t2i` | `nvidia/Cosmos3-Super-Text2Image` | modelo texto-para-imagem 65B |
| `super-i2v` | `nvidia/Cosmos3-Super-Image2Video` | modelo imagem-para-video 65B, video sem audio |

- Cosmos3 usa IDs do tokenizer diretamente.
- Prompts positivos sao convertidos para captions JSON do Cosmos3 durante a tokenizacao.
- Prompts negativos nao sao convertidos para JSON.
- Nao adicione um backend `text_embeds`.
- Amostras de imagem e video usam o cache VAE normal.
- Amostras de video + audio usam o cache VAE normal e um cache VAE de audio.
- `super-i2v` requer latentes de condicionamento.
- Datasets de acao e politica nao sao cobertos aqui.

## Componentes

O SimpleTuner usa componentes Cosmos3 separados por padrao:

| Flavour | Reasoner | Generator |
| --- | --- | --- |
| `nano` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-nano` | `SimpleTuner/cosmos3-component-generation-layers-bf16-nano` |
| `super` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-super` | `SimpleTuner/cosmos3-component-generation-layers-bf16-super` |
| `super-t2i` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-super-t2i` | `SimpleTuner/cosmos3-component-generation-layers-bf16-super-t2i` |
| `super-i2v` | `SimpleTuner/cosmos3-component-reasoning-layers-bf16-super-i2v` | `SimpleTuner/cosmos3-component-generation-layers-bf16-super-i2v` |

- Mantenha `cosmos3_reasoner_component: auto`.
- Mantenha `cosmos3_generator_component: auto`.
- As saidas do reasoner sao cacheadas pelo caminho de text embeds.
- `text_cache_disable: true` executa o reasoner congelado durante o treinamento.

## Hardware

- Comece com `model_flavour: nano`.
- Use `mixed_precision: bf16`.
- Comece com `base_model_precision: no_change`.
- Mantenha `train_batch_size: 1`.
- Ative `gradient_checkpointing`.
- Use os componentes generator separados para reduzir memoria ao carregar o transformer.

Group offload opcional:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream
```

- Streams sao apenas CUDA.
- Nao combine com `--enable_model_cpu_offload`.
- Adicione `--group_offload_to_disk_path /fast-ssd/simpletuner-offload` quando faltar RAM do sistema.

## Instalacao

```bash
pip install 'simpletuner[cuda]'

# Usuarios CUDA 13 / Blackwell
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

Instalacao de desenvolvimento:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Configs de exemplo

| Exemplo | Flavour | Dataset | Midia | Backend |
| --- | --- | --- | --- | --- |
| `cosmos3-image.lycoris-lokr` | `nano` | `RareConcepts/Domokun` | imagem | `multidatabackend-cosmos3-domokun-512px.json` |
| `cosmos3-video.lycoris-lokr` | `nano` | `sayakpaul/video-dataset-disney-organized` | video | `multidatabackend-cosmos3-disney-video-480p+49f.json` |
| `cosmos3-video-audio.lycoris-lokr` | `nano` | `bghira/Synchronised-Drumming-Gemini3Captions` | video + audio | `multidatabackend-cosmos3-drumming-video-audio-480p+49f.json` |
| `cosmos3-super-i2v.lycoris-lokr` | `super-i2v` | `sayakpaul/video-dataset-disney-organized` | imagem-para-video | `multidatabackend-cosmos3-disney-i2v-480p+49f.json` |

## Campos obrigatorios

- `model_family`: `cosmos3`
- `model_type`: `lora`
- `lora_type`: `lycoris`
- `base_model_precision`: `no_change`
- `mixed_precision`: `bf16`
- `train_batch_size`: `1`
- `gradient_checkpointing`: `true`

## Notas dos datasets

### Imagem

- Dataset: [`RareConcepts/Domokun`](https://huggingface.co/datasets/RareConcepts/Domokun)
- Backend: `config/examples/multidatabackend-cosmos3-domokun-512px.json`
- Tipo de backend: `huggingface`
- Estrategia de captions: `instanceprompt`
- Cache de text embeds: apenas cache do reasoner
- Cache de audio: nao usado

### Video

- Dataset: [`sayakpaul/video-dataset-disney-organized`](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized)
- Backend: `config/examples/multidatabackend-cosmos3-disney-video-480p+49f.json`
- Tipo de backend: `huggingface`
- Colunas: `video`, `prompt`
- Cache de text embeds: apenas cache do reasoner
- Cache de audio: nao usado

### Video com audio

- Dataset: [`bghira/Synchronised-Drumming-Gemini3Captions`](https://huggingface.co/datasets/bghira/Synchronised-Drumming-Gemini3Captions)
- Backend: `config/examples/multidatabackend-cosmos3-drumming-video-audio-480p+49f.json`
- Tipo de backend: `local`
- Arquivos: videos `.mpeg` com captions `.txt` adjacentes
- Cache de text embeds: apenas cache do reasoner
- Cache de audio: gerado a partir do backend de video

Baixe o dataset antes do treino:

```bash
huggingface-cli download \
  --repo-type=dataset \
  bghira/Synchronised-Drumming-Gemini3Captions \
  --local-dir datasets/Synchronised-Drumming-Gemini3Captions
```

Bloco do backend de audio:

```json
"audio": {
  "auto_split": true,
  "sample_rate": 16000,
  "channels": 1,
  "duration_interval": 3.0,
  "allow_zero_audio": false
}
```

### Super-I2V

- Dataset: [`sayakpaul/video-dataset-disney-organized`](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized)
- Backend: `config/examples/multidatabackend-cosmos3-disney-i2v-480p+49f.json`
- Tipo de backend: `huggingface`
- Colunas: `video`, `prompt`
- `video.is_i2v`: `true`
- Tipo de condicionamento: `reference_strict`
- Latentes de condicionamento: obrigatorios
- Cache de audio: nao usado

O backend I2V marca o dataset de video com:

```json
"video": {
  "num_frames": 49,
  "min_frames": 49,
  "is_i2v": true
}
```

O SimpleTuner cria o backend de referencia estrita pareado a partir dessa flag.

## Executar

```bash
simpletuner train example=cosmos3-image.lycoris-lokr
simpletuner train example=cosmos3-video.lycoris-lokr
simpletuner train example=cosmos3-video-audio.lycoris-lokr
simpletuner train example=cosmos3-super-i2v.lycoris-lokr
```

## Validacao

- Exemplo de imagem: `validation_resolution: 512x512`
- Exemplos de video: `validation_resolution: 768x432`
- Exemplos de video: `validation_num_video_frames: 49`
- Exemplo Super-I2V: usa entradas de validacao de condicionamento.

## Referencias

- [Pipeline Cosmos3 no Diffusers](https://huggingface.co/docs/diffusers/en/api/pipelines/cosmos3)
- [Colecao NVIDIA Cosmos3](https://huggingface.co/collections/nvidia/cosmos3)
