# InfiniteTalk क्विकस्टार्ट

InfiniteTalk, Wan 2.1 I2V 14B पर आधारित ऑडियो-चालित वीडियो मॉडल है। SimpleTuner Wan बेस लोड करता है और आधिकारिक ऑडियो प्रोजेक्टर तथा सभी 40 ब्लॉकों में ऑडियो अटेंशन जोड़ता है।

यह एकल-वक्ता मॉडल को ट्रेन करता है। बहु-वक्ता मोड के लिए कई सिंक्रोनाइज़्ड ऑडियो स्ट्रीम और स्पीकर मास्क चाहिए; मौजूदा dataloader प्रति वीडियो एक ऑडियो स्ट्रीम रखता है।

## आवश्यकताएँ

- bf16 समर्थित NVIDIA GPU
- 64 GB RAM; RamTorch या बिना quantization के लिए 96 GB या अधिक
- `ffmpeg`
- 25 fps पर ऑडियो-संरेखित वीडियो

```bash
python -m venv .venv
source .venv/bin/activate
pip install 'simpletuner[cuda]'
```

उदाहरण `trust_remote_code: true` से पिन किए गए `kernels-community/flash-attn3` Hub kernel को अधिकृत करते हैं। स्थानीय या built-in backend चुनने पर इसे हटाएँ।

## शुरुआती प्रोफाइल

| VRAM | फ्रेम | वज़न | रेजिडेंसी | उदाहरण |
| --- | ---: | --- | --- | --- |
| 24 GB | 17 | bf16 | सभी ब्लॉक RamTorch | `infinitetalk-14b-480p-24gb.peft-lora` |
| 32 GB | 17 | int8 TorchAO | 20 ब्लॉक स्वैप | `infinitetalk-14b-480p-32gb.peft-lora` |
| 48 GB | 33 | bf16 | 24 ब्लॉक स्वैप | `infinitetalk-14b-480p-48gb.peft-lora` |
| 80 GB | 49 | bf16 | रेजिडेंट | `infinitetalk-14b-480p-80gb.peft-lora` |

## डेटा

वीडियो और कैप्शन साथ रखें: `clip-001.mp4` और `clip-001.txt`। दिए गए कॉन्फ़िग 16 kHz मोनो ऑडियो निकालते हैं:

```json
"audio": {"auto_split": true, "sample_rate": 16000, "channels": 1}
```

- 25 fps उपयोग करें।
- फ्रेम संख्या `4k + 1` हो: 17, 33 या 49।
- ऑडियो उसी वीडियो अंतराल को कवर करे।
- रैंडम temporal crop को पूरे ट्रैक के ऑडियो से न जोड़ें।
- बिना ऑडियो वाले क्लिप अस्वीकार होते हैं।

## ट्रेनिंग

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

मेमोरी घटाने का क्रम: कम फ्रेम, अधिक `musubi_blocks_to_swap`, int8 TorchAO, फिर RamTorch। ऑडियो अटेंशन सटीक फ्रेम सीमाओं पर निर्भर है, इसलिए TREAD और context parallelism समर्थित नहीं हैं।

वैलिडेशन के लिए इमेज और ऑडियो चाहिए। अंतर्निहित वैलिडेशन text CFG करता है और दोनों शाखाओं में ऑडियो रखता है; अलग text/audio CFG के लिए आधिकारिक प्रोजेक्ट उपयोग करें।

LoRA, LyCORIS, full training, adapter quantization, checkpointing, block swap, RamTorch, FFN chunking, CREPA और LayerSync समर्थित हैं। बहु-वक्ता ट्रेनिंग समर्थित नहीं है।

स्रोत: [कोड](https://github.com/MeiGen-AI/InfiniteTalk), [रिपोर्ट](https://arxiv.org/abs/2508.14033), [वज़न](https://huggingface.co/MeiGen-AI/InfiniteTalk)।
