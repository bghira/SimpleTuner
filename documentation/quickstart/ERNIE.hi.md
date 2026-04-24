# ERNIE-Image [base / turbo] क्विकस्टार्ट

यह गाइड ERNIE-Image LoRA प्रशिक्षण के लिए है। ERNIE-Image Baidu की single-stream flow-matching transformer family है, और SimpleTuner इसमें `base` और `turbo` दोनों flavours को सपोर्ट करता है।

## हार्डवेयर आवश्यकताएँ

ERNIE छोटा मॉडल नहीं है। इसे दूसरे बड़े single-stream transformers की तरह प्लान करें:

- int8 quantization + bf16 LoRA के साथ 24 GB या उससे अधिक GPU सबसे यथार्थवादी लक्ष्य है
- 16 GB भी RamTorch और aggressive offload के साथ चल सकता है, लेकिन गति धीमी होगी
- multi-GPU, FSDP2, और CPU/RAM offload अतिरिक्त मदद देते हैं

Apple GPU पर प्रशिक्षण की सिफारिश नहीं की जाती।

## इंस्टॉलेशन

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

अधिक जानकारी के लिए [installation documentation](../INSTALL.md) देखें।

## वातावरण सेटअप

### WebUI

```bash
simpletuner server
```

इसके बाद training wizard में ERNIE family चुनें।

### कमांड लाइन

शुरू करने का सबसे आसान तरीका शामिल example का उपयोग करना है:

- example config: `simpletuner/examples/ernie.peft-lora/config.json`
- runnable local env: `config/ernie-example/config.json`

चलाएँ:

```bash
simpletuner train --env ernie-example
```

यदि आप इसे manually configure करना चाहते हैं:

- `model_type`: `lora`
- `model_family`: `ernie`
- `model_flavour`: `base` या `turbo`
- `pretrained_model_name_or_path`:
  - `base`: `baidu/ERNIE-Image`
  - `turbo`: `baidu/ERNIE-Image-Turbo`
- `resolution`: शुरुआत `512` से करें
- `train_batch_size`: `1`
- `ramtorch`: `true`
- `ramtorch_text_encoder`: `true`
- `gradient_checkpointing`: `true`

उदाहरण config में:

- `max_train_steps: 100`
- `optimizer: optimi-lion`
- `learning_rate: 1e-4`
- `validation_guidance: 4.0`
- `validation_num_inference_steps: 20`

### Turbo के लिए Assistant LoRA

ERNIE Turbo में assistant LoRA support मौजूद है, लेकिन अभी default adapter path नहीं है।

- supported flavour: `turbo`
- default weight filename: `pytorch_lora_weights.safetensors`
- जो आपको देना होगा: `assistant_lora_path`

अगर आपके पास custom assistant adapter है:

```json
{
  "assistant_lora_path": "your-org/your-ernie-turbo-assistant-lora",
  "assistant_lora_weight_name": "pytorch_lora_weights.safetensors"
}
```

अगर उपयोग नहीं करना है:

```json
{
  "disable_assistant_lora": true
}
```

### Dataset और captions

उदाहरण env यह उपयोग करता है:

- `dataset_name`: `RareConcepts/Domokun`
- `caption_strategy`: `instanceprompt`
- `instance_prompt`: `🟫`

यह smoke test के लिए ठीक है, लेकिन ERNIE आम तौर पर एक-token trigger की तुलना में वास्तविक text captions पर बेहतर प्रतिक्रिया देता है। वास्तविक training के लिए अधिक descriptive captions बेहतर हैं।

### उन्नत सुविधाएँ

ERNIE में ये features भी उपलब्ध हैं:

- TREAD
- LayerSync
- REPA / CREPA-style hidden state capture
- turbo के लिए assistant LoRA loading

पहले base training को स्थिर रूप से चलाएँ, फिर advanced features जोड़ें।
