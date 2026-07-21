## Quickstart do Cosmos3

Treine um LyCORIS LoKr para NVIDIA Cosmos3.

## Notas do modelo

- `model_family`: `cosmos3`
- Flavour padrao: `nano`
- Flavours:
  - `nano`: `nvidia/Cosmos3-Nano`, 16B
  - `super`: `nvidia/Cosmos3-Super`, 65B
  - `super-t2i`: `nvidia/Cosmos3-Super-Text2Image`, 65B
- O Cosmos3 consome IDs do tokenizer diretamente no SimpleTuner.
- Prompts positivos sao convertidos para captions JSON estruturados do Cosmos3 durante a tokenizacao.
- Prompts negativos nao sao convertidos para JSON.
- Nao adicione um backend `text_embeds`.
- Estes exemplos nao adicionam backends `image_embeds`.
- Amostras de imagem e video usam o cache VAE normal.
- Video com audio usa o cache VAE normal e um cache VAE de audio.
- Datasets de acao e politica nao sao cobertos aqui.

## Hardware

- Comece com `model_flavour: nano`.
- Use `mixed_precision: bf16`.
- Comece com `base_model_precision: no_change`.
- Mantenha `train_batch_size: 1`.
- Ative `gradient_checkpointing`.

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

| Exemplo | Dataset | Midia | Backend |
| --- | --- | --- | --- |
| `cosmos3-image.lycoris-lokr` | `RareConcepts/Domokun` | imagem | `multidatabackend-cosmos3-domokun-512px.json` |
| `cosmos3-video.lycoris-lokr` | `sayakpaul/video-dataset-disney-organized` | video | `multidatabackend-cosmos3-disney-video-480p+49f.json` |
| `cosmos3-video-audio.lycoris-lokr` | `bghira/Synchronised-Drumming-Gemini3Captions` | video + audio | `multidatabackend-cosmos3-drumming-video-audio-480p+49f.json` |

## Campos obrigatorios

- `model_family`: `cosmos3`
- `model_flavour`: `nano`
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
- Cache de text embeds: nao usado

### Video

- Dataset: [`sayakpaul/video-dataset-disney-organized`](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized)
- Backend: `config/examples/multidatabackend-cosmos3-disney-video-480p+49f.json`
- Tipo de backend: `huggingface`
- Colunas: `video`, `prompt`
- Cache de text embeds: nao usado
- Cache de audio: nao usado

### Video com audio

- Dataset: [`bghira/Synchronised-Drumming-Gemini3Captions`](https://huggingface.co/datasets/bghira/Synchronised-Drumming-Gemini3Captions)
- Backend: `config/examples/multidatabackend-cosmos3-drumming-video-audio-480p+49f.json`
- Tipo de backend: `local`
- Arquivos: videos `.mpeg` com captions `.txt` adjacentes
- Cache de text embeds: nao usado
- Cache de audio: gerado automaticamente a partir do backend de video

Baixe o dataset antes do treino:

```bash
huggingface-cli download \
  --repo-type=dataset \
  bghira/Synchronised-Drumming-Gemini3Captions \
  --local-dir datasets/Synchronised-Drumming-Gemini3Captions
```

O backend inclui:

```json
"audio": {
  "auto_split": true,
  "sample_rate": 16000,
  "channels": 1,
  "duration_interval": 3.0,
  "allow_zero_audio": false
}
```

O SimpleTuner injeta um dataset de audio a partir desse bloco e salva latentes de audio em um cache VAE separado.

## Executar

```bash
simpletuner train example=cosmos3-image.lycoris-lokr
simpletuner train example=cosmos3-video.lycoris-lokr
simpletuner train example=cosmos3-video-audio.lycoris-lokr
```

## Validacao

- Exemplo de imagem: `validation_resolution: 512x512`
- Exemplos de video: `validation_resolution: 768x432`
- Exemplos de video: `validation_num_video_frames: 49`
- A validacao de geracao de audio pode precisar de configuracoes de validacao baseadas em dataset.

## Referencias

- [Pipeline Cosmos3 no Diffusers](https://huggingface.co/docs/diffusers/en/api/pipelines/cosmos3)
- [Colecao NVIDIA Cosmos3](https://huggingface.co/collections/nvidia/cosmos3)
