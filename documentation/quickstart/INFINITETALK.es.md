# Inicio rápido de InfiniteTalk

InfiniteTalk es un modelo de vídeo controlado por audio basado en Wan 2.1 I2V 14B. SimpleTuner carga el modelo Wan y superpone el proyector de audio y la atención de audio oficiales en los 40 bloques.

Esta integración entrena el modelo oficial de un solo hablante. El modo multihablante necesita varios audios sincronizados y máscaras de hablante; el cargador actual representa un audio por vídeo.

## Requisitos

- GPU NVIDIA con bf16
- 64 GB de RAM; 96 GB o más para RamTorch o carga sin cuantizar
- `ffmpeg`
- Vídeos a 25 fps con audio alineado

```bash
python -m venv .venv
source .venv/bin/activate
pip install 'simpletuner[cuda]'
```

Los ejemplos autorizan el kernel Hub fijado `kernels-community/flash-attn3` con `trust_remote_code: true`. Elimínalo si eliges un backend local o integrado.

## Perfiles iniciales

| VRAM | Fotogramas | Pesos | Residencia | Ejemplo |
| --- | ---: | --- | --- | --- |
| 24 GB | 17 | bf16 | RamTorch, todos los bloques | `infinitetalk-14b-480p-24gb.peft-lora` |
| 32 GB | 17 | int8 TorchAO | intercambio de 20 bloques | `infinitetalk-14b-480p-32gb.peft-lora` |
| 48 GB | 33 | bf16 | intercambio de 24 bloques | `infinitetalk-14b-480p-48gb.peft-lora` |
| 80 GB | 49 | bf16 | residente | `infinitetalk-14b-480p-80gb.peft-lora` |

## Datos

Coloca el texto junto al vídeo: `clip-001.mp4` y `clip-001.txt`. Las configuraciones incluidas extraen audio mono a 16 kHz con:

```json
"audio": {"auto_split": true, "sample_rate": 16000, "channels": 1}
```

Reglas de alineación:

- Usa 25 fps.
- Usa `4k + 1` fotogramas: 17, 33 o 49.
- El audio debe cubrir exactamente el intervalo del clip.
- No combines un recorte temporal aleatorio con el audio completo.
- Los clips sin audio se rechazan.

## Entrenamiento

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

El orden recomendado para reducir memoria es: menos fotogramas, más `musubi_blocks_to_swap`, int8 TorchAO y por último RamTorch. InfiniteTalk no admite TREAD ni paralelismo de contexto porque la atención de audio depende de límites exactos por fotograma.

La validación necesita imagen y audio. La validación integrada aplica CFG de texto y mantiene el audio en ambas ramas; para comparar calidad final y CFG separado de texto/audio, usa la implementación oficial.

LoRA, LyCORIS, entrenamiento completo, cuantización para adaptadores, checkpointing, intercambio de bloques, RamTorch, chunking FFN, CREPA y LayerSync están soportados. El entrenamiento multihablante no está soportado.

Fuentes: [código](https://github.com/MeiGen-AI/InfiniteTalk), [informe](https://arxiv.org/abs/2508.14033), [pesos](https://huggingface.co/MeiGen-AI/InfiniteTalk).
