# ConvRot / Hadamard SDNQ

SimpleTuner SDNQ के Hadamard path के जरिए ConvRot-style rotation उपलब्ध कराता है। यह बड़े PEFT jobs के लिए उपयोगी है, जहां frozen base model int8 में चले और LoRA या LyCORIS adapters bf16 जैसे mixed-precision dtype में trainable रहें।

SimpleTuner arbitrary ConvRot sidecar buffers को अलग feature की तरह consume नहीं करता। Common path में original model weights load करें, फिर model load के बाद SimpleTuner trained component को SDNQ से quantize करेगा। Single-file quantized transformer weights support करने वाले model loaders compatible INT8 ConvRot transformer safetensors भी load कर सकते हैं और उन्हें SDNQ Hadamard के जरिए चला सकते हैं।

## Quick Setup

```json
{
  "base_model_precision": "int8-sdnq",
  "gradient_checkpointing": true,
  "sdnq_use_hadamard": true,
  "sdnq_hadamard_group_size": 256,
  "sdnq_group_size": -1,
  "sdnq_use_quantized_matmul": true,
  "sdnq_compile_mode": "compile"
}
```

बड़े models के लिए `quantize_via` को `cpu` पर रखें, जब तक model guide कुछ और न कहे। CPU quantization setup के दौरान accelerator memory peak को कम करता है।

## Options क्या करती हैं

- `base_model_precision: int8-sdnq` trained base component के लिए post-load SDNQ int8 quantization चुनता है।
- `sdnq_use_hadamard: true` Hadamard rotation path enable करता है।
- `sdnq_hadamard_group_size: 256` SDNQ का rotation block size सेट करता है। ConvRot के लिए `256` इस्तेमाल करें; छोटे blocks QuaRot-style path चुनते हैं।
- `sdnq_group_size: -1` static full-row weight scales इस्तेमाल करता है। यह dynamic grouped path से बचता है, जो मुख्य रूप से full fine-tuning के लिए है और training के दौरान weights requantize कर सकता है।
- `sdnq_use_quantized_matmul: true` SDNQ int8 matmul path active रखता है।
- `sdnq_compile_mode: compile` जहां SDNQ support करता है, quantization helpers और kernels compile करता है।
- `gradient_checkpointing: true` PEFT workloads में SDNQ को lower-overhead training path इस्तेमाल करने देता है। SimpleTuner इसे SDNQ को `use_grad_ckpt=True` के रूप में पास करता है; gradient checkpointing enabled होने पर इस SDNQ flag को false करने से सिर्फ quantized backward inputs save करने का extra काम जुड़ता है, जिन्हें checkpointing तुरंत discard कर देता है।

## PEFT Behavior

Base transformer SDNQ से quantize होता है। Adapter weights trainable रहते हैं और normal mixed-precision dtype इस्तेमाल करते हैं, आम तौर पर bf16।

कुछ models training से पहले fixed helper adapters load करते हैं। Z-Image Turbo में assistant LoRA है। SimpleTuner उस assistant adapter को SDNQ quantization के बाद तक defer करता है, ताकि SDNQ PEFT wrapper proxy weights के बजाय original transformer modules देखे।

## Requirements And Limits

- SimpleTuner supported install targets के लिए SDNQ training dependency install और configure करता है।
- यह preset बड़े models की LoRA और LyCORIS training के लिए है। SDNQ Hadamard के साथ full fine-tuning को अलग validation चाहिए।
- शुरुआती steps धीमे हो सकते हैं, क्योंकि setup और early training में SDNQ और Torch kernels compile करते हैं।
- Validation और inference quantized base model plus active adapter इस्तेमाल करते हैं, training की तरह।
- ConvRot quantization damage कम कर सकता है, लेकिन यह guarantee नहीं है कि INT8 हर model पर BF16 या FP8 जैसा होगा। Long run शुरू करने से पहले loss curve और generated samples दोनों validate करें।
- SDNQ ConvRot के साथ standalone inference इस training guide के scope से बाहर है। Direct SDNQ inference APIs के लिए [SDNQ upstream documentation](https://github.com/Disty0/sdnq) follow करें, क्योंकि वह API SimpleTuner training configuration से ज्यादा अक्सर बदलती है।

## Measured Results

ये model-specific SimpleTuner trainer measurements हैं, synthetic GEMM-only results नहीं। `Loop s/step` wrapper का train-loop wall time per step है। `Mean step` पहले पांच warmup steps को exclude करता है।

| Model | GPU | Steps | Weight path | Loop s/step | Mean step | p50 | p95 | Peak allocated VRAM |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Z-Image Turbo LoRA | H100 80GB | 1000 | SDNQ Hadamard post-load quantization | 1.107 | 1.087 | 1.071 | 1.109 | 9.70 GiB |
| Z-Image Turbo LoRA | L40S | 1000 | SDNQ Hadamard post-load quantization | 1.026 | 1.018 | 1.002 | 1.040 | 9.66 GiB |
| Z-Image Turbo LoRA | L40S | 1000 | baseline SDNQ Hadamard path | 1.131 | 1.072 | 1.055 | 1.102 | 9.66 GiB |
| Krea 2 Raw LoRA | H100 80GB | 100 | `lilcheaty/Krea2-INT8-ConvRot` transformer weights, diffusers attention | 0.787 | 0.399 | 0.397 | 0.411 | 32.15 GiB |
| Krea 2 Raw LoRA | L40S | 100 | `lilcheaty/Krea2-INT8-ConvRot` transformer weights, cuDNN attention | 0.945 | 0.794 | 0.793 | 0.799 | 31.89 GiB |
| Mage-Flow LoRA, square crop | H100 80GB | 100 | SDNQ INT8 vanilla post-load quantization | 1.113 | 0.277 | 0.276 | 0.286 | 20.12 GiB |
| Mage-Flow LoRA, square crop | H100 80GB | 100 | SDNQ ConvRot 256 post-load quantization | 0.436 | 0.299 | 0.297 | 0.308 | 20.15 GiB |

Warm-cache L40S Z-Image comparison में current path train-loop wall time से 10.3% और measured train-step mean से 5.2% baseline SDNQ Hadamard path से तेज था। Krea 2 rows Hugging Face INT8 ConvRot transformer-weight path को real 100-step training runs में verify करती हैं। Mage-Flow rows दिखाती हैं कि model-specific validation क्यों जरूरी है: square crop ने shape-compile churn का बड़ा हिस्सा हटाया, ConvRot ने vanilla INT8 की तुलना में total train-loop time कम किया, लेकिन warmed measured step vanilla INT8 से थोड़ा धीमा था।

## Example Models

SimpleTuner में Z-Image Turbo, Krea 2, FLUX.2, Cosmos 3, और LTXVideo 2.3 के लिए SDNQ Hadamard examples हैं। ये examples `sdnq_group_size: -1` इस्तेमाल करते हैं क्योंकि यह PEFT workload के लिए dynamic grouped training default से बेहतर fit हुआ।
