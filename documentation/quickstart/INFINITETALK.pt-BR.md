# Guia rápido do InfiniteTalk

InfiniteTalk é um modelo de vídeo dirigido por áudio baseado no Wan 2.1 I2V 14B. O SimpleTuner carrega a base Wan e aplica o projetor e a atenção de áudio oficiais nos 40 blocos.

Esta integração treina o modelo oficial de um locutor. O modo com vários locutores exige vários áudios sincronizados e máscaras; o dataloader atual representa um áudio por vídeo.

## Requisitos

- GPU NVIDIA com bf16
- 64 GB de RAM; 96 GB ou mais para RamTorch ou carga sem quantização
- `ffmpeg`
- Vídeos a 25 fps com áudio alinhado

```bash
python -m venv .venv
source .venv/bin/activate
pip install 'simpletuner[cuda]'
```

Os exemplos autorizam o kernel Hub fixado `kernels-community/flash-attn3` com `trust_remote_code: true`. Remova a opção ao escolher um backend local ou integrado.

## Perfis iniciais

| VRAM | Quadros | Pesos | Residência | Exemplo |
| --- | ---: | --- | --- | --- |
| 24 GB | 17 | bf16 | RamTorch, todos os blocos | `infinitetalk-14b-480p-24gb.peft-lora` |
| 32 GB | 17 | int8 TorchAO | troca de 20 blocos | `infinitetalk-14b-480p-32gb.peft-lora` |
| 48 GB | 33 | bf16 | troca de 24 blocos | `infinitetalk-14b-480p-48gb.peft-lora` |
| 80 GB | 49 | bf16 | residente | `infinitetalk-14b-480p-80gb.peft-lora` |

## Dados

Coloque a legenda ao lado do vídeo: `clip-001.mp4` e `clip-001.txt`. As configurações incluídas extraem áudio mono de 16 kHz:

```json
"audio": {"auto_split": true, "sample_rate": 16000, "channels": 1}
```

Regras:

- Use 25 fps.
- Use `4k + 1` quadros: 17, 33 ou 49.
- O áudio deve cobrir exatamente o intervalo do clipe.
- Não combine recorte temporal aleatório com áudio completo.
- Clipes sem áudio são rejeitados.

## Treinamento

```bash
simpletuner train \
  --config simpletuner/examples/infinitetalk-14b-480p-80gb.peft-lora/config.json
```

```json
{
  "model_family": "infinitetalk",
  "model_flavour": "single-14b-480p",
  "pretrained_model_name_or_path": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
  "framerate": 25
}
```

Para reduzir memória: diminua os quadros, aumente `musubi_blocks_to_swap`, use int8 TorchAO e depois RamTorch. InfiniteTalk não aceita TREAD nem paralelismo de contexto porque a atenção de áudio depende dos limites exatos de cada quadro.

A validação precisa de imagem e áudio. A validação integrada usa CFG de texto e mantém áudio nas duas ramificações; use o projeto oficial para CFG separado de texto/áudio.

LoRA, LyCORIS, treinamento completo, quantização para adaptadores, checkpointing, troca de blocos, RamTorch, chunking FFN, CREPA e LayerSync são suportados. Vários locutores não são suportados.

Fontes: [código](https://github.com/MeiGen-AI/InfiniteTalk), [relatório](https://arxiv.org/abs/2508.14033), [pesos](https://huggingface.co/MeiGen-AI/InfiniteTalk).
