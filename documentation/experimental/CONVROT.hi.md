# ConvRot / Hadamard SDNQ

SimpleTuner SDNQ के Hadamard path के जरिए ConvRot-style rotation उपलब्ध कराता है। यह बड़े PEFT jobs के लिए उपयोगी है, जहां frozen base model int8 में चले और LoRA या LyCORIS adapters bf16 जैसे mixed-precision dtype में trainable रहें।

यह external ConvRot checkpoint buffers को सीधे load नहीं करता। Original model weights load करें, फिर model load के बाद SimpleTuner trained component को SDNQ से quantize करेगा।

## Quick Setup

```json
{
  "base_model_precision": "int8-sdnq",
  "gradient_checkpointing": true,
  "sdnq_use_hadamard": true,
  "sdnq_hadamard_group_size": 128,
  "sdnq_group_size": -1,
  "sdnq_use_quantized_matmul": true,
  "sdnq_compile_mode": "compile"
}
```

बड़े models के लिए `quantize_via` को `cpu` पर रखें, जब तक model guide कुछ और न कहे। CPU quantization setup के दौरान accelerator memory peak को कम करता है।

## Options क्या करती हैं

- `base_model_precision: int8-sdnq` trained base component के लिए post-load SDNQ int8 quantization चुनता है।
- `sdnq_use_hadamard: true` Hadamard rotation path enable करता है।
- `sdnq_hadamard_group_size: 128` SDNQ का rotation block size सेट करता है।
- `sdnq_group_size: -1` static full-row weight scales इस्तेमाल करता है। यह dynamic grouped path से बचता है, जो मुख्य रूप से full fine-tuning के लिए है और training के दौरान weights requantize कर सकता है।
- `sdnq_use_quantized_matmul: true` SDNQ int8 matmul path active रखता है।
- `sdnq_compile_mode: compile` जहां SDNQ support करता है, quantization helpers और kernels compile करता है।
- `gradient_checkpointing: true` PEFT workloads में SDNQ को lower-overhead training path इस्तेमाल करने देता है। SimpleTuner इसे SDNQ को `use_grad_ckpt=True` के रूप में पास करता है; gradient checkpointing enabled होने पर इस SDNQ flag को false करने से सिर्फ quantized backward inputs save करने का extra काम जुड़ता है, जिन्हें checkpointing तुरंत discard कर देता है।

## PEFT Behavior

Base transformer SDNQ से quantize होता है। Adapter weights trainable रहते हैं और normal mixed-precision dtype इस्तेमाल करते हैं, आम तौर पर bf16।

कुछ models training से पहले fixed helper adapters load करते हैं। Z-Image Turbo में assistant LoRA है। SimpleTuner उस assistant adapter को SDNQ quantization के बाद तक defer करता है, ताकि SDNQ PEFT wrapper proxy weights के बजाय original transformer modules देखे।

## Requirements And Limits

- Hadamard support वाला SDNQ build इस्तेमाल करें। H100 verification में upstream SDNQ `0.2.3` इस्तेमाल हुआ; PyPI `0.2.2` में वही bf16 Hadamard fix नहीं है।
- यह preset बड़े models की LoRA और LyCORIS training के लिए है। SDNQ Hadamard के साथ full fine-tuning को अलग validation चाहिए।
- शुरुआती steps धीमे हो सकते हैं, क्योंकि setup और early training में SDNQ और Torch kernels compile करते हैं।
- Validation और inference quantized base model plus active adapter इस्तेमाल करते हैं, training की तरह।

## Example Models

SimpleTuner में Z-Image Turbo, Krea 2, FLUX.2, Cosmos 3, और LTXVideo 2.3 के लिए SDNQ Hadamard examples हैं। ये examples `sdnq_group_size: -1` इस्तेमाल करते हैं क्योंकि यह PEFT workload के लिए dynamic grouped training default से बेहतर fit हुआ।
