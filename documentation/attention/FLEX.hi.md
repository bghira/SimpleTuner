# FlexAttention गाइड

**FlexAttention को CUDA डिवाइस की आवश्यकता है।**

FlexAttention PyTorch का block-level attention kernel है जो PyTorch 2.5.0 में आया था। यह SDPA गणना को एक प्रोग्रामेबल लूप के रूप में लिखता है ताकि आप CUDA लिखे बिना masking रणनीतियाँ व्यक्त कर सकें। Diffusers इसे नए `attention_backend` dispatcher के जरिए एक्सपोज़ करता है, और SimpleTuner उस dispatcher को `--attention_mechanism=flex` से जोड़ देता है।

> ⚠️ FlexAttention अभी भी upstream में “prototype” लेबल के साथ है। ड्राइवर, CUDA वर्ज़न, या PyTorch बिल्ड बदलने पर recompile की उम्मीद रखें।

## प्रीरिक्विज़िट्स

1. **Ampere+ GPU** – NVIDIA SM80 (A100), Ada (4090/L40S), या Hopper (H100/H200) समर्थित हैं। पुराने कार्ड kernel registration के दौरान capability check में फेल हो जाते हैं।
2. **Compiler toolchain** – kernels `nvcc` के साथ रनटाइम पर कम्पाइल होते हैं। wheel से मेल खाता `cuda-nvcc` इंस्टॉल करें (वर्तमान रिलीज़ के लिए CUDA 12.8) और सुनिश्चित करें कि `nvcc` आपके `$PATH` में हो।

## कर्नेल्स बनाना

`torch.nn.attention.flex_attention` का पहला import CUDA एक्सटेंशन को PyTorch के lazy cache में बनाता है। आप build errors जल्दी देखने के लिए इसे पहले से चला सकते हैं:

```bash
python - <<'PY'
import torch
from torch.nn.attention import flex_attention

assert torch.__version__ >= "2.5.0", torch.__version__
flex_attention.build_flex_attention_kernels()  # no-op when already compiled
print("FlexAttention kernels installed at", flex_attention.kernel_root)
PY
```

- यदि आपको `AttributeError: flex_attention has no attribute build_flex_attention_kernels` दिखे, तो PyTorch अपग्रेड करें – यह helper 2.5.0+ में है।
- Cache `~/.cache/torch/kernels` के नीचे होता है। यदि CUDA अपग्रेड किया है और rebuild चाहिए, तो इसे हटा दें।

## SimpleTuner में FlexAttention सक्षम करना

कर्नेल्स बनने के बाद, `config.json` के जरिए backend चुनें:

```json
{
  "attention_mechanism": "flex"
}
```

क्या उम्मीद करें:

- केवल dispatch-enabled transformer blocks (Flux, Wan 2.2, LTXVideo, QwenImage, आदि) Diffusers के `attention_backend` से गुजरते हैं। क्लासिक SD/SDXL UNets अभी भी सीधे PyTorch SDPA को कॉल करते हैं, इसलिए वहाँ FlexAttention का असर नहीं होगा।
- FlexAttention फिलहाल BF16/FP16 टेन्सर्स सपोर्ट करता है। यदि आप FP32 या FP8 वेट्स चलाते हैं तो `ValueError: Query, key, and value must be either bfloat16 or float16` मिलेगा।
- backend केवल `is_causal=False` को सम्मान देता है। मास्क देने पर वह kernel के अपेक्षित block mask में बदलता है, लेकिन arbitrary ragged masks अभी समर्थित नहीं हैं (upstream व्यवहार के अनुरूप)।

## ट्रबलशूटिंग चेकलिस्ट

| Symptom | Fix |
| --- | --- |
| `RuntimeError: Flex Attention backend 'flex' is not usable because of missing package` | PyTorch बिल्ड < 2.5 है या CUDA शामिल नहीं है। नया CUDA wheel इंस्टॉल करें। |
| `Could not compile flex_attention kernels` | सुनिश्चित करें कि `nvcc` आपके torch wheel के अपेक्षित CUDA वर्ज़न (12.1+) से मेल खाता है। यदि installer headers नहीं ढूँढ पा रहा हो तो `export CUDA_HOME=/usr/local/cuda-12.4` सेट करें। |
| `ValueError: Query, key, and value must be on a CUDA device` | FlexAttention केवल CUDA के लिए है। Apple/ROCm रन में backend सेटिंग हटाएँ। |
| Training कभी backend पर स्विच नहीं होता | सुनिश्चित करें कि आप ऐसा मॉडल फैमिली उपयोग कर रहे हैं जो पहले से Diffusers की `dispatch_attention_fn` उपयोग करता है (Flux/Wan/LTXVideo)। स्टैंडर्ड SD UNets आपके चुने backend के बावजूद PyTorch SDPA ही उपयोग करेंगे। |

और गहरे internals एवं API flags के लिए upstream डॉक्यूमेंटेशन देखें: [PyTorch FlexAttention docs](https://pytorch.org/docs/stable/nn.attention.html#flexattention)।
