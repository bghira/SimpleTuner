# SimpleTuner Training Script विकल्प

## Overview

यह गाइड SimpleTuner के `train.py` स्क्रिप्ट में उपलब्ध command‑line विकल्पों का user‑friendly विवरण देती है। ये विकल्प उच्च स्तर का customization देते हैं, जिससे आप मॉडल को अपनी आवश्यकताओं के अनुसार ट्रेन कर सकते हैं।

### JSON Configuration file format

अपेक्षित JSON फ़ाइल‑नाम `config.json` है और key नाम नीचे दिए `--arguments` जैसे ही हैं। JSON फ़ाइल में अग्रणी `--` आवश्यक नहीं है, लेकिन चाहें तो रख सकते हैं।

Ready‑to‑run उदाहरण ढूँढ रहे हैं? [simpletuner/examples/README.md](/simpletuner/examples/README.md) में curated presets देखें।

### Easy configure script (***RECOMMENDED***)

`simpletuner configure` कमांड से `config.json` फ़ाइल को अधिकतर आदर्श डिफ़ॉल्ट सेटिंग्स के साथ सेट किया जा सकता है।

#### मौजूदा कॉन्फ़िगरेशन बदलना

`configure` कमांड एक ही argument स्वीकार कर सकता है, एक compatible `config.json`, जिससे आप training setup को इंटरैक्टिव तरीके से बदल सकते हैं:

```bash
simpletuner configure config/foo/config.json
```

यहाँ `foo` आपका config environment है — या यदि आप config environments उपयोग नहीं कर रहे हैं, तो `config/config.json` का उपयोग करें।

<img width="1484" height="560" alt="image" src="https://github.com/user-attachments/assets/67dec8d8-3e41-42df-96e6-f95892d2814c" />

> ⚠️ जिन देशों में Hugging Face Hub आसानी से उपलब्ध नहीं है, वहाँ के उपयोगकर्ताओं को अपने सिस्टम के `$SHELL` के अनुसार `~/.bashrc` या `~/.zshrc` में `HF_ENDPOINT=https://hf-mirror.com` जोड़ना चाहिए।

---

## 🌟 Core Model Configuration

### `--model_type`

- **What**: यह चुनता है कि LoRA या full fine‑tune बनाया जाएगा।
- **Choices**: lora, full.
- **Default**: lora
  - यदि lora उपयोग हो, तो `--lora_type` तय करता है कि PEFT या LyCORIS उपयोग हो रहा है। कुछ मॉडल (PixArt) केवल LyCORIS adapters के साथ काम करते हैं।

### `--model_family`

- **What**: यह निर्धारित करता है कि कौन‑सा model architecture train किया जा रहा है।
- **Choices**: pixart_sigma, flux, sd3, sdxl, kolors, legacy

### `--lora_format`

- **What**: load/save के लिए LoRA checkpoint key format चुनता है।
- **Choices**: `diffusers` (डिफ़ॉल्ट), `comfyui`
- **Notes**:
  - `diffusers` standard PEFT/Diffusers layout है।
  - `comfyui` keys को ComfyUI‑style में convert करता है (`diffusion_model.*` के साथ `lora_A/lora_B` और `.alpha` tensors)। Flux, Flux2, Lumina2, और Z‑Image ComfyUI inputs को auto‑detect करेंगे भले ही यह `diffusers` पर हो, लेकिन saving के लिए ComfyUI output force करने के लिए `comfyui` सेट करें।

### `--fuse_qkv_projections`

- **What**: मॉडल के attention blocks में QKV projections को fuse करता है ताकि hardware का अधिक कुशल उपयोग हो।
- **Note**: केवल NVIDIA H100 या H200 पर उपलब्ध है जब Flash Attention 3 मैन्युअली install हो।

### `--offload_during_startup`

- **What**: VAE caching चलने के दौरान text encoder weights को CPU पर offload करता है।
- **Why**: HiDream और Wan 2.1 जैसे बड़े मॉडल्स में VAE cache लोड करते समय OOM हो सकता है। यह विकल्प training quality को प्रभावित नहीं करता, लेकिन बहुत बड़े text encoders या धीमे CPUs के साथ, कई datasets पर startup time काफ़ी बढ़ सकता है। इसी कारण यह डिफ़ॉल्ट रूप से disabled है।
- **Tip**: विशेष रूप से memory‑constrained systems के लिए नीचे दिए group offloading फीचर के साथ पूरक है।

### `--offload_during_save`

- **What**: `save_hooks.py` checkpoints तैयार करते समय पूरे pipeline को अस्थायी रूप से CPU पर ले जाता है ताकि सभी FP8/quantized weights device से बाहर लिखे जाएँ।
- **Why**: fp8‑quanto weights सेव करने पर VRAM उपयोग अचानक बढ़ सकता है (उदा., `state_dict()` serialization के दौरान)। यह विकल्प training के लिए मॉडल को accelerator पर रखता है, लेकिन save trigger होने पर थोड़ी देर के लिए offload करता है ताकि CUDA OOM से बचा जा सके।
- **Tip**: केवल तब सक्षम करें जब saving OOM errors दे रही हो; loader बाद में मॉडल वापस ले आता है ताकि training seamless रहे।

### `--delete_model_after_load`

- **What**: मॉडल files को HuggingFace cache से delete करता है जब वे memory में load हो जाते हैं।
- **Why**: उन सेटअप्स में disk usage घटाता है जहाँ GB के हिसाब से billing होती है। मॉडल VRAM/RAM में लोड हो जाने के बाद on‑disk cache अगली run तक आवश्यक नहीं रहता। यह भार storage से network bandwidth पर शिफ्ट करता है अगली रन में।
- **Notes**:
  - यदि validation सक्षम है तो VAE **delete नहीं** होगा, क्योंकि validation images बनाने के लिए इसकी आवश्यकता है।
  - Text encoders को data backend factory के startup पूरा होने के बाद delete किया जाता है (embed caching के बाद)।
  - Transformer/UNet मॉडल्स load होने के तुरंत बाद delete होते हैं।
  - Multi‑node setups में, हर node पर केवल local‑rank 0 deletion करता है। Shared network storage पर race conditions संभालने के लिए deletion failures silently ignore की जाती हैं।
  - यह saved training checkpoints पर **प्रभाव नहीं** डालता — केवल pre‑trained base model cache पर लागू होता है।

### `--trust_remote_code`

- **What**: जब checkpoint upstream custom classes पर निर्भर हो, तब Transformers और tokenizer को model repository से custom Python code चलाने की अनुमति देता है।
- **Default**: `False`
- **Why**: ACE-Step v1.5 checkpoints के लिए आवश्यक है, क्योंकि upstream repository में custom `AutoModel` और tokenizer code शामिल है।
- **Warning**: इसे केवल उन्हीं model repositories के लिए सक्षम करें जिन पर आप भरोसा करते हैं।

### `--enable_group_offload`

- **What**: diffusers की grouped module offloading सक्षम करता है ताकि forward passes के बीच model blocks को CPU (या disk) पर stage किया जा सके।
- **Why**: बड़े transformers (Flux, Wan, Auraflow, LTXVideo, Cosmos2Image) पर peak VRAM usage को बहुत कम करता है, खासकर CUDA streams के साथ, और performance पर न्यूनतम प्रभाव पड़ता है।
- **Notes**:
  - `--enable_model_cpu_offload` के साथ mutually exclusive — प्रति run एक ही strategy चुनें।
  - diffusers **v0.33.0** या नया required है।

### `--group_offload_type`

- **Choices**: `block_level` (डिफ़ॉल्ट), `leaf_level`
- **What**: layers को कैसे group किया जाए नियंत्रित करता है। `block_level` VRAM बचत और throughput के बीच संतुलन रखता है, जबकि `leaf_level` अधिक CPU transfers की कीमत पर अधिक बचत देता है।

### `--group_offload_blocks_per_group`

- **What**: `block_level` उपयोग करते समय, एक offload group में कितने transformer blocks bundle किए जाएँ।
- **Default**: `1`
- **Why**: इस संख्या को बढ़ाने से transfer frequency कम होती है (तेज़), लेकिन अधिक parameters accelerator पर resident रहते हैं (अधिक VRAM)।

### `--group_offload_use_stream`

- **What**: host/device transfers को compute के साथ overlap करने के लिए dedicated CUDA stream उपयोग करता है।
- **Default**: `False`
- **Notes**:
  - non‑CUDA backends (Apple MPS, ROCm, CPU) पर स्वतः CPU‑style transfers पर fallback करता है।
  - NVIDIA GPUs पर training करते समय, और copy engine capacity spare हो, तो अनुशंसित।

### `--group_offload_to_disk_path`

- **What**: directory path जहाँ grouped parameters को RAM की बजाय disk पर spill किया जाएगा।
- **Why**: अत्यंत tight CPU RAM बजट के लिए उपयोगी (जैसे बड़े NVMe drive वाला workstation)।
- **Tip**: तेज़ local SSD उपयोग करें; network filesystems training को काफी धीमा कर देंगे।

### `--musubi_blocks_to_swap`

- **What**: LongCat‑Video, Wan, LTXVideo, Kandinsky5‑Video, Qwen‑Image, Flux, Flux.2, Cosmos2Image, और HunyuanVideo के लिए Musubi block swap — आख़िरी N transformer blocks को CPU पर रखें और forward के दौरान प्रति block weights stream करें।
- **Default**: `0` (disabled)
- **Notes**: Musubi‑style weight offload; throughput लागत पर VRAM कम करता है और gradients सक्षम होने पर skip हो जाता है।

### `--musubi_block_swap_device`

- **What**: swapped transformer blocks को स्टोर करने के लिए device string (उदा. `cpu`, `cuda:0`)।
- **Default**: `cpu`
- **Notes**: केवल तब उपयोग होता है जब `--musubi_blocks_to_swap` > 0 हो।

### `--ramtorch`

- **What**: `nn.Linear` layers को RamTorch CPU‑streamed implementations से replace करता है।
- **Why**: Linear weights को CPU memory में साझा करता है और उन्हें accelerator पर stream करता है ताकि VRAM pressure कम हो।
- **Notes**:
  - CUDA या ROCm आवश्यक है (Apple/MPS पर समर्थित नहीं)।
  - `--enable_group_offload` के साथ mutually exclusive।
  - `--set_grads_to_none` स्वतः सक्षम करता है।

### `--ramtorch_target_modules`

- **What**: Comma‑separated glob patterns जो यह सीमित करते हैं कि कौन‑से Linear modules को RamTorch में बदला जाए।
- **Default**: यदि कोई pattern नहीं दिया गया, तो सभी Linear layers convert होती हैं।
- **Notes**:
  - `fnmatch` glob syntax का उपयोग करके fully qualified module names या class names को match करता है।
  - किसी block के अंदर की layers को match करने के लिए patterns में trailing `.*` wildcard शामिल होना चाहिए। उदाहरण के लिए, `transformer_blocks.0.*` block 0 के अंदर सभी layers को match करता है, जबकि `transformer_blocks.*` सभी transformer blocks को match करता है। `transformer_blocks.0` जैसा bare name बिना `.*` के भी काम करेगा (यह automatically expand होता है), लेकिन clarity के लिए explicit wildcard form recommended है।
  - उदाहरण: `"transformer_blocks.*,single_transformer_blocks.0.*,single_transformer_blocks.1.*"`

### `--ramtorch_text_encoder`

- **What**: सभी text encoder Linear layers पर RamTorch replacements लागू करता है।
- **Default**: `False`

### `--ramtorch_vae`

- **What**: VAE mid‑block Linear layers के लिए experimental RamTorch conversion।
- **Default**: `False`
- **Notes**: VAE up/down convolutional blocks unchanged रहते हैं।

### `--ramtorch_controlnet`

- **What**: ControlNet training के दौरान ControlNet Linear layers पर RamTorch replacements लागू करता है।
- **Default**: `False`

### `--ramtorch_transformer_percent`

- **What**: RamTorch के साथ offload करने के लिए transformer Linear layers का प्रतिशत (0-100)।
- **Default**: `100` (सभी eligible layers)
- **Why**: VRAM बचत और performance के बीच संतुलन के लिए partial offloading की अनुमति देता है। कम values GPU पर अधिक layers रखती हैं जिससे training तेज होती है जबकि memory usage भी कम होती है।
- **Notes**: Layers module traversal order की शुरुआत से select की जाती हैं। `--ramtorch_target_modules` के साथ combine किया जा सकता है।

### `--ramtorch_text_encoder_percent`

- **What**: RamTorch के साथ offload करने के लिए text encoder Linear layers का प्रतिशत (0-100)।
- **Default**: `100` (सभी eligible layers)
- **Why**: जब `--ramtorch_text_encoder` enabled हो तब text encoders की partial offloading की अनुमति देता है।
- **Notes**: केवल तब लागू होता है जब `--ramtorch_text_encoder` enabled हो।

### `--ramtorch_disable_sync_hooks`

- **What**: RamTorch layers के बाद add किए गए CUDA synchronization hooks को disable करता है।
- **Default**: `False` (sync hooks enabled)
- **Why**: Sync hooks RamTorch के ping-pong buffering system में race conditions को fix करते हैं जो non-deterministic outputs का कारण बन सकते हैं। Disable करने से performance बेहतर हो सकता है लेकिन incorrect results का risk है।
- **Notes**: केवल तब disable करें जब sync hooks में समस्या हो या उनके बिना test करना हो।

### `--ramtorch_disable_extensions`

- **What**: केवल Linear layers पर RamTorch apply करता है, Embedding/RMSNorm/LayerNorm/Conv को skip करता है।
- **Default**: `True` (extensions disabled)
- **Why**: SimpleTuner RamTorch को Linear layers से आगे बढ़ाकर Embedding, RMSNorm, LayerNorm, और Conv layers को include करता है। इन extensions को disable करके केवल Linear layers offload करने के लिए इसका उपयोग करें।
- **Notes**: VRAM savings कम हो सकती है लेकिन extended layer types की समस्याओं को debug करने में मदद कर सकता है।

### `--pretrained_model_name_or_path`

- **What**: pretrained model का path या <https://huggingface.co/models> से उसका identifier.
- **Why**: उस base model को निर्दिष्ट करने के लिए जिससे training शुरू होगी। Repository से specific versions चुनने के लिए `--revision` और `--variant` उपयोग करें। यह SDXL, Flux, और SD3.x के लिए single‑file `.safetensors` paths भी सपोर्ट करता है।

### `--pretrained_t5_model_name_or_path`

- **What**: pretrained T5 model का path या <https://huggingface.co/models> से उसका identifier.
- **Why**: PixArt ट्रेन करते समय आप अपने T5 weights के लिए कोई specific source चुनना चाह सकते हैं ताकि base model switch करने पर बार‑बार download न करना पड़े।

### `--pretrained_gemma_model_name_or_path`

- **What**: pretrained Gemma model का path या <https://huggingface.co/models> से उसका identifier.
- **Why**: Gemma‑based models (जैसे LTX-2, Sana, Lumina2) ट्रेन करते समय आप base diffusion model path बदले बिना Gemma weights का source specify कर सकते हैं।

### `--max_grounding_entities`
- GLIGEN-style spatial annotations के लिए प्रति image grounding entities की अधिकतम संख्या। Default: 0 (disabled)। सामान्य मान: 4-16।

### `--pretrained_grounding_model_name_or_path`
- प्रति entity image feature extraction के लिए वैकल्पिक pretrained model। Default: None।

### `--custom_text_encoder_intermediary_layers`

- **What**: FLUX.2 models के लिए text encoder से extract होने वाली hidden state layers को override करें।
- **Format**: Layer indices का JSON array, जैसे `[10, 20, 30]`
- **Default**: सेट न होने पर model-specific defaults उपयोग होते हैं:
  - FLUX.2-dev (Mistral-3): `[10, 20, 30]`
  - FLUX.2-klein (Qwen3): `[9, 18, 27]`
- **Why**: Custom alignment या research के लिए विभिन्न text encoder hidden state combinations के साथ experiment करने की सुविधा देता है।
- **Note**: यह option experimental है और केवल FLUX.2 models पर लागू होता है। Layer indices बदलने से cached text embeddings invalid हो जाएंगे और regenerate करने होंगे। Layers की संख्या model की expected input (3 layers) से match होनी चाहिए।

### `--gradient_checkpointing`

- **What**: Training के दौरान gradients layerwise compute होकर accumulate होते हैं ताकि peak VRAM कम हो, लेकिन training धीमी होती है।

### `--gradient_checkpointing_interval`

- **What**: हर *n* blocks पर checkpoint करें, जहाँ *n* शून्य से बड़ा मान है। 1 का मान `--gradient_checkpointing` enabled जैसा है, और 2 हर दूसरे block पर checkpoint करेगा।
- **Note**: यह विकल्प फिलहाल केवल SDXL और Flux में समर्थित है। SDXL इसमें hackish implementation उपयोग करता है।

### `--gradient_checkpointing_backend`

- **Choices**: `torch`, `unsloth`
- **What**: Gradient checkpointing के लिए implementation चुनें।
  - `torch` (default): Standard PyTorch checkpointing जो backward pass के दौरान activations को recalculate करता है। ~20% time overhead।
  - `unsloth`: Recalculate करने के बजाय activations को asynchronously CPU पर offload करता है। ~30% अधिक memory बचत केवल ~2% overhead के साथ। Fast PCIe bandwidth आवश्यक है।
- **Note**: केवल `--gradient_checkpointing` enabled होने पर प्रभावी। `unsloth` backend के लिए CUDA आवश्यक है।

### `--refiner_training`

- **What**: custom mixture‑of‑experts मॉडल श्रृंखला training सक्षम करता है। इन विकल्पों पर अधिक जानकारी के लिए [Mixture-of-Experts](MIXTURE_OF_EXPERTS.md) देखें।

## Precision

### `--quantize_via`

- **Choices**: `cpu`, `accelerator`, `pipeline`
  - `accelerator` पर यह मध्यम रूप से तेज़ हो सकता है लेकिन Flux जैसे बड़े मॉडल के लिए 24G cards पर OOM का जोखिम रहता है।
  - `cpu` पर quantisation में लगभग 30 seconds लगते हैं। (**Default**)
  - `pipeline` Diffusers को `--quantization_config` या pipeline‑capable presets (उदा. `nf4-bnb`, `int8-torchao`, `fp8-torchao`, `int8-quanto`, या `.gguf` checkpoints) के साथ quantization delegate करता है।

### `--base_model_precision`

- **What**: model precision घटाएँ और कम memory में training करें। तीन समर्थित quantisation backends हैं: BitsAndBytes (pipeline), TorchAO (pipeline या manual), और Optimum Quanto (pipeline या manual)।

#### Diffusers pipeline presets

- `nf4-bnb` Diffusers के माध्यम से 4‑bit NF4 BitsAndBytes config के साथ लोड होता है (CUDA only)। `bitsandbytes` और BnB support वाली diffusers build आवश्यक है।
- `int4-torchao`, `int8-torchao`, और `fp8-torchao` Diffusers के माध्यम से TorchAoConfig का उपयोग करते हैं (CUDA)। `torchao` और recent diffusers/transformers आवश्यक है।
- `int8-quanto`, `int4-quanto`, `int2-quanto`, `fp8-quanto`, और `fp8uz-quanto` Diffusers के माध्यम से QuantoConfig का उपयोग करते हैं। Diffusers FP8-NUZ को float8 weights पर map करता है; NUZ variant के लिए manual quanto quantization उपयोग करें।
- `.gguf` checkpoints auto‑detect होकर उपलब्ध होने पर `GGUFQuantizationConfig` के साथ लोड होते हैं। GGUF support के लिए recent diffusers/transformers install करें।

#### Optimum Quanto

Hugging Face द्वारा प्रदान की गई optimum‑quanto लाइब्रेरी सभी समर्थित प्लेटफ़ॉर्म्स पर मजबूत समर्थन देती है।

- `int8-quanto` सबसे व्यापक रूप से compatible है और संभवतः सर्वोत्तम परिणाम देता है
  - RTX4090 और संभवतः अन्य GPUs पर सबसे तेज़ training
  - CUDA devices पर int8, int4 के लिए hardware‑accelerated matmul उपयोग करता है
    - int4 अभी भी बहुत धीमा है
  - `TRAINING_DYNAMO_BACKEND=inductor` (`torch.compile()`) के साथ काम करता है
- `fp8uz-quanto` CUDA और ROCm devices के लिए experimental fp8 variant है।
  - Instinct या नई architecture वाले AMD silicon पर बेहतर supported
  - 4090 पर training में `int8-quanto` से थोड़ा तेज़ हो सकता है, लेकिन inference में नहीं (1 second धीमा)
  - `TRAINING_DYNAMO_BACKEND=inductor` (`torch.compile()`) के साथ काम करता है
- `fp8-quanto` फिलहाल fp8 matmul उपयोग नहीं करेगा, Apple systems पर काम नहीं करता।
  - CUDA या ROCm पर hardware fp8 matmul अभी नहीं है, इसलिए संभवतः int8 से काफी धीमा होगा
    - fp8 GEMM के लिए MARLIN kernel उपयोग करता है
  - dynamo के साथ असंगत, इस संयोजन की कोशिश होने पर dynamo स्वतः disabled हो जाएगा।

#### TorchAO

PyTorch की एक नई लाइब्रेरी, AO हमें linears और 2D convolutions (उदा. unet style models) को quantised counterparts से बदलने देती है।
<!-- Additionally, it provides an experimental CPU offload optimiser that essentially provides a simpler reimplementation of DeepSpeed. -->

- `int8-torchao` memory consumption को Quanto के precision levels जैसा कम कर देता है
  - लिखते समय, Apple MPS पर Quanto (9s/iter) की तुलना में थोड़ा धीमा (11s/iter)
  - `torch.compile` उपयोग न करने पर CUDA devices पर `int8-quanto` जैसी speed और memory, ROCm पर speed profile अज्ञात
  - `torch.compile` उपयोग करने पर `int8-quanto` से धीमा
- `fp8-torchao` केवल Hopper (H100, H200) या नए (Blackwell B200) accelerators के लिए उपलब्ध है

##### Optimisers

TorchAO में सामान्य 4bit और 8bit optimisers हैं: `ao-adamw8bit`, `ao-adamw4bit`

यह Hopper (H100 या बेहतर) users के लिए दो optimisers भी देता है: `ao-adamfp8`, और `ao-adamwfp8`

#### SDNQ (SD.Next Quantization Engine)

[SDNQ](https://github.com/disty0/sdnq) एक training‑optimized quantization लाइब्रेरी है जो सभी प्लेटफ़ॉर्म्स पर काम करती है: AMD (ROCm), Apple (MPS), और NVIDIA (CUDA)। यह stochastic rounding और quantized optimizer states के साथ memory‑efficient quantized training प्रदान करती है।

##### Recommended Precision Levels

**Full finetuning के लिए** (model weights update होती हैं):
- `uint8-sdnq` - memory बचत और training quality का सबसे अच्छा संतुलन
- `uint16-sdnq` - अधिकतम quality के लिए उच्च precision (उदा. Stable Cascade)
- `int16-sdnq` - signed 16‑bit विकल्प
- `fp16-sdnq` - quantized FP16, SDNQ लाभों के साथ अधिकतम precision

**LoRA training के लिए** (base model weights frozen):
- `int8-sdnq` - signed 8‑bit, अच्छा general purpose विकल्प
- `int6-sdnq`, `int5-sdnq` - lower precision, छोटा memory footprint
- `uint5-sdnq`, `uint4-sdnq`, `uint3-sdnq`, `uint2-sdnq` - aggressive compression

**Note:** `int7-sdnq` उपलब्ध है लेकिन अनुशंसित नहीं (धीमा और int8 से बहुत छोटा नहीं)।

**Important:** 5‑bit से नीचे SDNQ गुणवत्ता बनाए रखने के लिए स्वतः SVD (Singular Value Decomposition) को 8 steps के साथ सक्षम करता है। SVD quantize करने में अधिक समय लेता है और non‑deterministic है, इसलिए Disty0 HuggingFace पर pre‑quantized SVD मॉडल देता है। SVD training के दौरान compute overhead जोड़ता है, इसलिए full finetuning में (जहाँ weights सक्रिय रूप से update होती हैं) इससे बचें।

**Key features:**
- Cross‑platform: AMD, Apple, और NVIDIA हार्डवेयर पर समान रूप से काम करता है
- Training‑optimized: stochastic rounding से quantization error accumulation कम होता है
- Memory efficient: quantized optimizer state buffers सपोर्ट करता है
- Decoupled matmul: weight precision और matmul precision स्वतंत्र हैं (INT8/FP8/FP16 matmul उपलब्ध)

##### SDNQ Optimisers

SDNQ में अतिरिक्त memory बचत के लिए optional quantized state buffers वाले optimizers शामिल हैं:

- `sdnq-adamw` - AdamW with quantized state buffers (uint8, group_size=32)
- `sdnq-adamw+no_quant` - बिना quantized states के AdamW (comparison के लिए)
- `sdnq-adafactor` - quantized state buffers के साथ Adafactor
- `sdnq-came` - quantized state buffers के साथ CAME optimizer
- `sdnq-lion` - quantized state buffers के साथ Lion optimizer
- `sdnq-muon` - quantized state buffers के साथ Muon optimizer
- `sdnq-muon+quantized_matmul` - zeropower computation में INT8 matmul के साथ Muon

सभी SDNQ optimizers डिफ़ॉल्ट रूप से stochastic rounding उपयोग करते हैं और custom settings के लिए `--optimizer_config` के साथ कॉन्फ़िगर किए जा सकते हैं, जैसे `use_quantized_buffers=false` जिससे state quantization बंद हो जाती है।

**Muon‑specific options:**
- `use_quantized_matmul` - zeropower_via_newtonschulz5 में INT8/FP8/FP16 matmul सक्षम करें
- `quantized_matmul_dtype` - matmul precision: `int8` (consumer GPUs), `fp8` (datacenter), `fp16`
- `zeropower_dtype` - zeropower computation के लिए precision (जब `use_quantized_matmul=True` हो तब ignore)
- Muon बनाम AdamW fallback के लिए अलग values सेट करने हेतु args को `muon_` या `adamw_` prefix करें

**Pre‑quantized models:** Disty0 pre‑quantized uint4 SVD models [huggingface.co/collections/Disty0/sdnq](https://huggingface.co/collections/Disty0/sdnq) पर देता है। इन्हें सामान्य रूप से लोड करें, फिर SDNQ import करने के बाद `convert_sdnq_model_to_training()` से convert करें (register करने के लिए loading से पहले SDNQ import होना चाहिए)।

**Note on checkpointing:** SDNQ training models को training resumption के लिए native PyTorch format (`.pt`) और inference के लिए safetensors format दोनों में सेव किया जाता है। Proper training resumption के लिए native format आवश्यक है क्योंकि SDNQ की `SDNQTensor` class custom serialization उपयोग करती है।

**Disk space tip:** disk space बचाने के लिए आप केवल quantized weights रख सकते हैं और inference के लिए ज़रूरत पड़ने पर SDNQ की [dequantize_sdnq_training.py](https://github.com/Disty0/sdnq/blob/main/scripts/dequantize_sdnq_training.py) script से dequantize कर सकते हैं।

### `--quantization_config`

- **What**: `--quantize_via=pipeline` उपयोग करते समय Diffusers `quantization_config` overrides का JSON object या file path.
- **How**: inline JSON (या file) स्वीकार करता है जिसमें per‑component entries हों। Keys में `unet`, `transformer`, `text_encoder`, या `default` शामिल हो सकते हैं।
- **Examples**:

```json
{
  "unet": {"load_in_4bit": true, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16"},
  "text_encoder": {"quant_type": {"group_size": 128}}
}
```

यह उदाहरण UNet पर 4‑bit NF4 BnB और text encoder पर TorchAO int4 सक्षम करता है।

#### Torch Dynamo

WebUI से `torch.compile()` सक्षम करने के लिए **Hardware → Accelerate (advanced)** पर जाएँ और **Torch Dynamo Backend** को अपने पसंदीदा compiler (उदा. *inductor*) पर सेट करें। अतिरिक्त toggles आपको optimisation **mode** चुनने, **dynamic shape** guards सक्षम करने, या **regional compilation** opt‑in करने देते हैं ताकि बहुत गहरे transformer models पर cold starts तेज़ हो सकें।

वही कॉन्फ़िगरेशन `config/config.env` में इस तरह व्यक्त की जा सकती है:

```bash
TRAINING_DYNAMO_BACKEND=inductor
```

आप इसे वैकल्पिक रूप से `--dynamo_mode=max-autotune` या UI में उपलब्ध अन्य Dynamo flags के साथ pair कर सकते हैं ताकि finer control मिले।

ध्यान दें कि training के पहले कई steps सामान्य से धीमे होंगे क्योंकि compilation background में हो रही होती है।

Settings को `config.json` में persist करने के लिए समान keys जोड़ें:

```json
{
  "dynamo_backend": "inductor",
  "dynamo_mode": "max-autotune",
  "dynamo_fullgraph": false,
  "dynamo_dynamic": false,
  "dynamo_use_regional_compilation": true
}
```

यदि आप Accelerate defaults inherit करना चाहते हैं तो संबंधित entries छोड़ दें (उदा., `dynamo_mode` न दें ताकि automatic selection उपयोग हो)।

### `--attention_mechanism`

Alternative attention mechanisms समर्थित हैं, जिनके compatibility स्तर या trade‑offs अलग होते हैं:

- `diffusers` PyTorch के native SDPA kernels उपयोग करता है और डिफ़ॉल्ट है।
- `xformers` Meta के [xformers](https://github.com/facebookresearch/xformers) attention kernel (training + inference) सक्षम करता है, जब underlying मॉडल `enable_xformers_memory_efficient_attention` expose करता है।
- `flash-attn`, `flash-attn-2`, `flash-attn-3`, और `flash-attn-3-varlen` Diffusers के नए `attention_backend` helper के जरिए FlashAttention v1/2/3 kernels में route करते हैं। संबंधित `flash-attn` / `flash-attn-interface` wheels install करें और ध्यान दें कि FA3 फिलहाल Hopper GPUs की मांग करता है।
- `flex` PyTorch 2.5 का FlexAttention backend चुनता है (CUDA पर FP16/BF16)। आपको Flex kernels अलग से compile/install करने होंगे — देखें [documentation/attention/FLEX.md](attention/FLEX.md)।
- `cudnn`, `native-efficient`, `native-flash`, `native-math`, `native-npu`, और `native-xla` `torch.nn.attention.sdpa_kernel` द्वारा expose किए गए matching SDPA backend चुनते हैं। ये तब उपयोगी हैं जब आपको determinism (`native-math`), CuDNN SDPA kernel, या vendor‑native accelerators (NPU/XLA) चाहिए।
- `sla` [Sparse–Linear Attention (SLA)](https://github.com/thu-ml/SLA) सक्षम करता है, जो fine‑tunable sparse/linear hybrid kernel देता है और training तथा validation दोनों में बिना अतिरिक्त gating के उपयोग किया जा सकता है।
  - SLA package install करें (उदा. `pip install -e ~/src/SLA`) इस backend को चुनने से पहले।
  - SimpleTuner SLA के learned projection weights हर checkpoint में `sla_attention.pt` में सेव करता है; resume और inference के लिए इस फ़ाइल को बाकी checkpoint के साथ रखें।
  - क्योंकि backbone को SLA के mixed sparse/linear behavior के अनुसार ट्यून किया गया है, inference समय पर भी SLA आवश्यक होगा। focused गाइड के लिए `documentation/attention/SLA.md` देखें।
  - यदि जरूरत हो तो SLA runtime defaults override करने के लिए `--sla_config '{"topk":0.15,"blkq":32,"tie_feature_map_qk":false}'` (JSON या Python dict syntax) उपयोग करें।
- `sageattention`, `sageattention-int8-fp16-triton`, `sageattention-int8-fp16-cuda`, और `sageattention-int8-fp8-cuda` संबंधित [SageAttention](https://github.com/thu-ml/SageAttention) kernels को wrap करते हैं। ये inference‑oriented हैं और accidental training से बचने के लिए `--sageattention_usage` के साथ उपयोग होने चाहिए।
  - सरल शब्दों में, SageAttention inference के compute requirement को कम करता है

> ℹ️ Flash/Flex/PyTorch backend selectors Diffusers के `attention_backend` dispatcher पर निर्भर हैं, इसलिए वे वर्तमान में transformer‑style models में अधिक लाभ देते हैं जो पहले से इस code path को उपयोग करते हैं (Flux, Wan 2.x, LTXVideo, QwenImage, आदि)। Classic SD/SDXL UNets अभी सीधे PyTorch SDPA उपयोग करते हैं।

`--sageattention_usage` के जरिए SageAttention के साथ training सक्षम करना सावधानी से करना चाहिए, क्योंकि यह QKV linears के लिए अपनी custom CUDA implementations से gradients track या propagate नहीं करता।

- इससे ये layers पूरी तरह untrained रह जाते हैं, जो model collapse या छोटे training runs में हल्का सुधार कर सकते हैं।

---

## 📰 Publishing

### `--push_to_hub`

- **What**: यदि दिया गया, तो training पूरी होने पर आपका मॉडल [Huggingface Hub](https://huggingface.co) पर upload होगा। `--push_checkpoints_to_hub` उपयोग करने पर हर intermediary checkpoint भी push होगा।

### `--push_to_hub_background`

- **What**: Hugging Face Hub पर background worker से uploads करता है ताकि checkpoint pushes training loop को pause न करें।
- **Why**: Hub uploads asynchronous रहते हुए training और validation चलती रहती है। Run समाप्त होने से पहले final uploads की प्रतीक्षा की जाती है ताकि failures surface हों।

### `--webhook_config`

- **What**: webhook targets (उदा. Discord, custom endpoints) के लिए कॉन्फ़िगरेशन ताकि real‑time training events मिल सकें।
- **Why**: बाहरी tools और dashboards के साथ training runs मॉनिटर करने देता है और मुख्य training चरणों पर notifications भेजता है।
- **Notes**: webhook payloads में `job_id` फ़ील्ड `SIMPLETUNER_JOB_ID` environment variable सेट करके भरी जा सकती है:
  ```bash
  export SIMPLETUNER_JOB_ID="my-training-run-name"
  python train.py
  ```
यह उन monitoring tools के लिए उपयोगी है जो कई training runs से webhooks प्राप्त करते हैं ताकि यह पता चले कि किस config ने event भेजा। यदि SIMPLETUNER_JOB_ID सेट नहीं है, तो webhook payloads में job_id null होगा।

### `--publishing_config`

- **What**: non‑Hugging Face publishing targets (S3‑compatible storage, Backblaze B2, Azure Blob Storage, Dropbox) को वर्णित करने वाला optional JSON/dict/file path.
- **Why**: `--webhook_config` parsing को mirror करता है ताकि artifacts को Hub के बाहर भी fan out किया जा सके। Publishing validation के बाद main process पर current `output_dir` का उपयोग करके चलता है।
- **Notes**: Providers `--push_to_hub` के additive हैं। provider SDKs (जैसे `boto3`, `azure-storage-blob`, `dropbox`) को `.venv` में install करें जब आप इन्हें enable करें। पूर्ण उदाहरणों के लिए `documentation/publishing/README.md` देखें।

### `--hub_model_id`

- **What**: Huggingface Hub मॉडल और local results directory का नाम।
- **Why**: यह मान `--output_dir` द्वारा निर्दिष्ट स्थान के अंतर्गत directory नाम के रूप में उपयोग होता है। यदि `--push_to_hub` दिया गया है, तो यही Huggingface Hub पर मॉडल का नाम होगा।

### `--modelspec_comment`

- **What**: safetensors फ़ाइल metadata में `modelspec.comment` के रूप में embedded text
- **Default**: None (disabled)
- **Notes**:
  - बाहरी model viewers (ComfyUI, model info tools) में दिखाई देता है
  - string या strings की array (newlines से जुड़ी) स्वीकार करता है
  - environment variable substitution के लिए `{env:VAR_NAME}` placeholders support करता है
  - metadata लिखते समय `{current_step}`, `{current_epoch}`, और `{timestamp}` support करता है
  - `{timestamp}` UTC ISO 8601 value का उपयोग करता है
  - प्रत्येक checkpoint save के समय current config value उपयोग करता है

**Example (string)**:
```json
"modelspec_comment": "मेरे custom dataset v2.1 पर trained"
```

**Example (array multi-line के लिए)**:
```json
"modelspec_comment": [
  "Training run: experiment-42",
  "Dataset: custom-portraits-v2",
  "Notes: {env:TRAINING_NOTES}"
]
```

### `--disable_benchmark`

- **What**: step 0 पर base model के लिए होने वाली startup validation/benchmark को disable करता है। ये outputs आपकी trained model validation images के बाएँ हिस्से में stitched होते हैं।

## 📂 Data Storage and Management

### `--data_backend_config`

- **What**: आपके SimpleTuner dataset कॉन्फ़िगरेशन का path.
- **Why**: अलग‑अलग storage माध्यमों पर कई datasets को एक training session में जोड़ा जा सकता है।
- **Notes**:
  - `config.json` और `config.toml` से load की गई string values `{env:VAR_NAME}` support करती हैं
  - referenced `multidatabackend.json` के अंदर की string values भी `{env:VAR_NAME}` support करती हैं
- **Example**: उदाहरण कॉन्फ़िगरेशन के लिए [multidatabackend.json.example](/multidatabackend.json.example) देखें, और data loader कॉन्फ़िगर करने के लिए [यह दस्तावेज़](DATALOADER.md) देखें।

### `--override_dataset_config`

- **What**: दिया जाने पर, SimpleTuner cached config और current values के बीच अंतर को ignore करेगा।
- **Why**: किसी dataset पर SimpleTuner पहली बार चलने पर यह dataset की हर चीज़ की जानकारी वाला cache document बनाता है, जिसमें dataset config भी शामिल होता है, जैसे इसके "crop" और "resolution" संबंधित मान। इन्हें मनमाने रूप से या गलती से बदलने पर training jobs बेतरतीब रूप से crash हो सकते हैं, इसलिए इस parameter का उपयोग न करने और बदलाव किसी अन्य तरीके से करने की सिफ़ारिश है।

### `--data_backend_sampling`

- **What**: कई data backends के साथ sampling अलग strategies से की जा सकती है।
- **Options**:
  - `uniform` - v0.9.8.1 और पहले का behavior जहाँ dataset length consider नहीं होती, केवल manual probability weightings ली जाती थीं।
  - `auto-weighting` - डिफ़ॉल्ट behavior जहाँ dataset length का उपयोग करके सभी datasets को समान रूप से sample किया जाता है, ताकि पूरे data distribution पर uniform sampling बनी रहे।
    - यह तब आवश्यक है जब आपके datasets अलग‑अलग sizes के हों और आप चाहते हों कि मॉडल उन्हें समान रूप से सीखे।
    - लेकिन Dreambooth images को regularisation set के विरुद्ध सही तरह sample करने के लिए `repeats` को manually adjust करना **आवश्यक** है

### `--vae_cache_scan_behaviour`

- **What**: integrity scan check का व्यवहार कॉन्फ़िगर करता है।
- **Why**: dataset के लिए गलत settings training के कई चरणों पर लागू हो सकते हैं, जैसे यदि आप गलती से dataset से `.json` cache files delete कर दें और data backend config को aspect‑crops की जगह square images के लिए switch कर दें। इससे data cache inconsistent हो जाता है, जिसे `multidatabackend.json` में `scan_for_errors` को `true` सेट करके ठीक किया जा सकता है। जब यह scan चलता है, तो यह `--vae_cache_scan_behaviour` के मान के अनुसार inconsistency को resolve करता है: `recreate` (डिफ़ॉल्ट) offending cache entry को हटाता है ताकि वह फिर से बनाई जा सके, और `sync` bucket metadata को वास्तविक training sample के अनुरूप अपडेट करता है। अनुशंसित मान: `recreate`.

### `--dataloader_prefetch`

- **What**: batches को ahead‑of‑time retrieve करता है।
- **Why**: विशेष रूप से बड़े batch sizes के साथ, samples disk (यहाँ तक कि NVMe) से लोड होने पर training "pause" होती है, जिससे GPU utilisation metrics प्रभावित होते हैं। Dataloader prefetch सक्षम करने पर पूरे batches का buffer भरकर रखा जाता है ताकि वे तुरंत load हो सकें।

> ⚠️ यह वास्तव में केवल H100 या बेहतर GPU पर कम resolution में उपयोगी है जहाँ I/O bottleneck बनता है। अधिकांश अन्य उपयोग मामलों में यह अनावश्यक जटिलता है।

### `--dataloader_prefetch_qlen`

- **What**: memory में रखे गए batches की संख्या बढ़ाता या घटाता है।
- **Why**: dataloader prefetch के साथ, डिफ़ॉल्ट रूप से प्रति GPU/process 10 entries memory में रखी जाती हैं। यह बहुत अधिक या बहुत कम हो सकता है। इस मान को बदलकर batches की संख्या समायोजित की जा सकती है।

### `--compress_disk_cache`

- **What**: VAE और text embed caches को disk पर compress करता है।
- **Why**: DeepFloyd, SD3, और PixArt में उपयोग होने वाला T5 encoder बहुत बड़े text embeds बनाता है जो छोटे या redundant captions के लिए mostly empty space होते हैं। `--compress_disk_cache` सक्षम करने से space उपयोग 75% तक घट सकता है, औसतन 40% बचत के साथ।

> ⚠️ आपको मौजूदा cache directories मैन्युअल रूप से हटाने होंगे ताकि trainer उन्हें compression के साथ फिर से बना सके।

---

## 🌈 Image और Text Processing

कई सेटिंग्स [dataloader config](DATALOADER.md) में सेट होती हैं, लेकिन ये global रूप से लागू होंगी।

### `--resolution_type`

- **What**: यह SimpleTuner को बताता है कि `area` size calculations उपयोग करनी हैं या `pixel` edge calculations। `pixel_area` का hybrid approach भी समर्थित है, जो `area` measurements के लिए megapixel की बजाय pixel उपयोग करने देता है।
- **Options**:
  - `resolution_type=pixel_area`
    - `resolution` का मान 1024 होने पर यह internally efficient aspect bucketing के लिए सटीक area measurement में मैप होगा।
    - `1024` के लिए उदाहरण आकार: 1024x1024, 1216x832, 832x1216
  - `resolution_type=pixel`
    - dataset की सभी images का छोटा edge इस resolution तक resize होगा, जिससे resulting images बड़े हो सकते हैं और VRAM उपयोग बढ़ सकता है।
    - `1024` के लिए उदाहरण आकार: 1024x1024, 1766x1024, 1024x1766
  - `resolution_type=area`
    - **Deprecated**. इसकी जगह `pixel_area` उपयोग करें।

### `--resolution`

- **What**: input image resolution, pixel edge length में व्यक्त
- **Default**: 1024
- **Note**: यदि dataset में resolution सेट नहीं है, तो यही global default उपयोग होगा।

### `--validation_resolution`

- **What**: output image resolution, pixels में; या `widthxheight` फ़ॉर्मैट में, जैसे `1024x1024`। Multiple resolutions को comma से अलग कर सकते हैं।
- **Why**: validation के दौरान बनने वाली सभी images इसी resolution पर होंगी। यह तब उपयोगी है जब मॉडल अलग resolution पर train हो रहा हो।

### `--validation_method`

- **What**: validation runs कैसे execute हों, यह चुनें।
- **Options**: `simpletuner-local` (डिफ़ॉल्ट) built‑in pipeline चलाता है; `external-script` user‑provided executable चलाता है।
- **Why**: training को local pipeline work में रोके बिना validation को external system में हैंड‑ऑफ करने देता है।

### `--validation_external_script`

- **What**: `--validation_method=external-script` होने पर चलाया जाने वाला executable। यह shell‑style splitting उपयोग करता है, इसलिए command string को ठीक से quote करें।
- **Placeholders**: आप training context पास करने के लिए इन tokens को embed कर सकते हैं (`.format` के साथ)। Missing values खाली string से replace होती हैं जब तक कि उल्लेख न किया गया हो:
  - `{local_checkpoint_path}` → `output_dir` के अंतर्गत latest checkpoint directory (कम से कम एक checkpoint आवश्यक)।
  - `{global_step}` → current global step.
  - `{tracker_run_name}` → `--tracker_run_name` का मान.
  - `{tracker_project_name}` → `--tracker_project_name` का मान.
  - `{model_family}` → `--model_family` का मान.
  - `{model_type}` / `{lora_type}` → model type और LoRA flavour.
  - `{huggingface_path}` → `--hub_model_id` का मान (यदि सेट हो)।
  - `{remote_checkpoint_path}` → last upload का remote URL (validation hook के लिए empty)।
  - कोई भी `validation_*` config value (उदा., `validation_num_inference_steps`, `validation_guidance`, `validation_noise_scheduler`).
- **Example**: `--validation_external_script="/opt/tools/validate.sh {local_checkpoint_path} {global_step}"`

### `--validation_external_background`

- **What**: सेट होने पर `--validation_external_script` background में launch होता है (fire‑and‑forget)।
- **Why**: external script का इंतज़ार किए बिना training चलती रहती है; इस मोड में exit codes check नहीं होते।

### `--post_upload_script`

- **What**: हर publishing provider और Hugging Face Hub upload (final model और checkpoint uploads) के बाद optional executable चलता है। यह asynchronous चलता है ताकि training block न हो।
- **Placeholders**: `--validation_external_script` जैसे replacements, साथ ही `{remote_checkpoint_path}` (provider द्वारा लौटाया गया URI) ताकि आप published URL downstream systems को भेज सकें।
- **Notes**:
  - scripts हर provider/upload पर चलती हैं; errors लॉग होते हैं लेकिन training रोकते नहीं।
  - जब कोई remote upload न हो तब भी scripts invoke होती हैं, ताकि आप local automation (उदा., दूसरे GPU पर inference) चला सकें।
  - SimpleTuner आपकी script के परिणाम ingest नहीं करता — metrics या images रिकॉर्ड करने के लिए सीधे अपने tracker पर लॉग करें।
- **Example**:
  ```bash
  --post_upload_script='/opt/hooks/notify.sh {remote_checkpoint_path} {tracker_project_name} {tracker_run_name}'
  ```
  जहाँ `/opt/hooks/notify.sh` आपके tracking system को post कर सकता है:
  ```bash
  #!/usr/bin/env bash
  REMOTE="$1"
  PROJECT="$2"
  RUN="$3"
  curl -X POST "https://tracker.internal/api/runs/${PROJECT}/${RUN}/artifacts" \
       -H "Content-Type: application/json" \
       -d "{\"remote_uri\":\"${REMOTE}\"}"
  ```
- **Working samples**:
  - `simpletuner/examples/external-validation/replicate_post_upload.py` एक Replicate hook दिखाता है जो `{remote_checkpoint_path}`, `{model_family}`, `{model_type}`, `{lora_type}`, और `{huggingface_path}` consume करके uploads के बाद inference ट्रिगर करता है।
  - `simpletuner/examples/external-validation/wavespeed_post_upload.py` वही placeholders के साथ WaveSpeed hook दिखाता है और WaveSpeed की async polling उपयोग करता है।
  - `simpletuner/examples/external-validation/fal_post_upload.py` fal.ai Flux LoRA hook दिखाता है (`FAL_KEY` आवश्यक)।
  - `simpletuner/examples/external-validation/use_second_gpu.py` secondary GPU पर Flux LoRA inference चलाता है और remote uploads के बिना भी काम करता है।

### `--post_checkpoint_script`

- **What**: हर checkpoint directory disk पर लिखे जाने के तुरंत बाद चलाया जाने वाला executable (uploads शुरू होने से पहले)। main process पर asynchronous चलता है।
- **Placeholders**: `--validation_external_script` जैसे replacements, जिनमें `{local_checkpoint_path}`, `{global_step}`, `{tracker_run_name}`, `{tracker_project_name}`, `{model_family}`, `{model_type}`, `{lora_type}`, `{huggingface_path}` और कोई भी `validation_*` config value शामिल हैं। `{remote_checkpoint_path}` इस hook के लिए खाली होता है।
- **Notes**:
  - Scheduled, manual, और rolling checkpoints पर यह तब fire होता है जब local save पूरा हो जाए।
  - Local automation (दूसरे volume पर copy, eval jobs चलाना) के लिए उपयोगी है, uploads के खत्म होने का इंतज़ार किए बिना।
- **Example**:
  ```bash
  --post_checkpoint_script='/opt/hooks/run_eval.sh {local_checkpoint_path} {global_step}'
  ```


### `--validation_adapter_path`

- **What**: scheduled validations चलाते समय अस्थायी रूप से एक single LoRA adapter लोड करता है।
- **Formats**:
  - Hugging Face repo: `org/repo` या `org/repo:weight_name.safetensors` (डिफ़ॉल्ट `pytorch_lora_weights.safetensors`).
  - Local file या directory path जो safetensors adapter की ओर इशारा करे।
- **Notes**:
  - `--validation_adapter_config` के साथ mutually exclusive; दोनों देने पर error आता है।
  - adapter केवल validation runs के लिए attach होता है (baseline training weights untouched रहते हैं)।

### `--validation_adapter_name`

- **What**: `--validation_adapter_path` से लोड किए गए अस्थायी adapter के लिए optional identifier.
- **Why**: logs/Web UI में adapter run को label करने और कई adapters sequentially test होने पर predictable names सुनिश्चित करने के लिए।

### `--validation_adapter_strength`

- **What**: अस्थायी adapter enable होने पर strength multiplier (डिफ़ॉल्ट `1.0`).
- **Why**: training state बदले बिना validation के दौरान हल्का/तेज़ LoRA scaling sweep करने देता है; शून्य से बड़ा कोई भी मान स्वीकार होता है।

### `--validation_adapter_mode`

- **Choices**: `adapter_only`, `comparison`, `none`
- **What**:
  - `adapter_only`: केवल अस्थायी adapter के साथ validations चलाएँ।
  - `comparison`: base‑model और adapter‑enabled samples दोनों जनरेट करें ताकि side‑by‑side समीक्षा हो सके।
  - `none`: adapter attach करना छोड़ दें (CLI flags हटाए बिना फीचर disable करने के लिए उपयोगी)।

### `--validation_adapter_config`

- **What**: multiple validation adapter combinations को iterate करने के लिए JSON file या inline JSON.
- **Format**: entries की array या `runs` array वाला object। हर entry में शामिल हो सकता है:
  - `label`: logs/UI में दिखने वाला friendly name.
  - `path`: Hugging Face repo ID या local path (`--validation_adapter_path` जैसा format)।
  - `adapter_name`: प्रति adapter optional identifier.
  - `strength`: optional scalar override.
  - `adapters`/`paths`: एक ही run में multiple adapters लोड करने के लिए objects/strings की array.
- **Notes**:
  - यह प्रदान होने पर single‑adapter options (`--validation_adapter_path`, `--validation_adapter_name`, `--validation_adapter_strength`, `--validation_adapter_mode`) UI में ignore/disable हो जाते हैं।
  - हर run को एक‑एक करके load किया जाता है और अगला शुरू होने से पहले पूरी तरह detach किया जाता है।

### `--validation_preview`

- **What**: Tiny AutoEncoders का उपयोग करके diffusion sampling के दौरान intermediate validation previews stream करता है
- **Default**: False
- **Why**: validation images के generate होने के दौरान real‑time preview सक्षम करता है; lightweight Tiny AutoEncoder models द्वारा decode होकर webhook callbacks से भेजे जाते हैं। इससे आप पूरी generation का इंतज़ार किए बिना step‑by‑step progress देख सकते हैं।
- **Notes**:
  - केवल Tiny AutoEncoder support वाले model families पर उपलब्ध (उदा., Flux, SDXL, SD3)
  - preview images प्राप्त करने के लिए webhook configuration आवश्यक
  - previews कितनी बार decode हों, यह नियंत्रित करने के लिए `--validation_preview_steps` उपयोग करें

### `--validation_preview_steps`

- **What**: validation previews decode और stream करने का interval
- **Default**: 1
- **Why**: validation sampling के दौरान intermediate latents कितनी बार decode हों, यह नियंत्रित करता है। उच्च मान (उदा. 3) Tiny AutoEncoder का overhead घटाता है क्योंकि हर N steps पर decode होता है।
- **Example**: `--validation_num_inference_steps=20` और `--validation_preview_steps=5` के साथ, generation process के दौरान 4 preview images मिलेंगी (steps 5, 10, 15, 20 पर)।

### `--evaluation_type`

- **What**: validations के दौरान generated images की CLIP evaluation सक्षम करें।
- **Why**: CLIP scores validation prompt के साथ generated image features की दूरी निकालते हैं। इससे prompt adherence में सुधार का संकेत मिल सकता है, लेकिन सार्थक परिणामों के लिए बड़ी संख्या में validation prompts चाहिए।
- **Options**: "none" या "clip"
- **Scheduling**: step‑based scheduling के लिए `--eval_steps_interval` या epoch‑based scheduling के लिए `--eval_epoch_interval` उपयोग करें (fractions जैसे `0.5` प्रति epoch कई बार चलेंगे)। यदि दोनों सेट हों, trainer warning लॉग करेगा और दोनों schedules चलाएगा।

### `--eval_loss_disable`

- **What**: validation के दौरान evaluation loss गणना disable करें।
- **Why**: जब eval dataset कॉन्फ़िगर हो, loss स्वतः गणना होता है। यदि CLIP evaluation सक्षम है, तो दोनों चलेंगे। यह flag eval loss को disable करने देता है जबकि CLIP evaluation चालू रहता है।

### `--validation_using_datasets`

- **What**: pure text-to-image generation के बजाय training datasets से images validation के लिए use करें।
- **Why**: image-to-image (img2img) या image-to-video (i2v) validation mode enable करता है जहाँ model training images को conditioning inputs के रूप में use करता है। उपयोगी है:
  - Edit/inpainting models test करने के लिए जिन्हें input images चाहिए
  - Model image structure को कितना preserve करता है evaluate करने के लिए
  - Dual text-to-image AND image-to-image workflows support करने वाले models के लिए (जैसे, Flux2, LTXVideo2)
  - **I2V video models** (HunyuanVideo, WAN, Kandinsky5Video): image dataset से images को video generation validation के लिए first-frame conditioning input के रूप में use करें
- **Notes**:
  - Model में `IMG2IMG` या `IMG2VIDEO` pipeline registered होना चाहिए
  - `--eval_dataset_id` के साथ combine कर सकते हैं specific dataset से images लेने के लिए
  - i2v models के लिए, यह training में use होने वाली complex conditioning dataset pairing setup के बिना simple image dataset validation के लिए use करने देता है
  - Flux Kontext validation के लिए इस flag का उपयोग नहीं करता; इसे disabled रखें और edit dataset चुनने के लिए `--eval_dataset_id` उपयोग करें, जबकि Kontext paired reference dataset अपने आप load कर लेता है
  - Denoising strength normal validation timestep settings से control होती है

### `--eval_dataset_id`

- **What**: Evaluation/validation image sourcing के लिए specific dataset ID।
- **Why**: `--validation_using_datasets` या conditioning-based validation use करते समय, यह control करता है कौन सा dataset input images provide करे:
  - इस option के बिना, images सभी training datasets से randomly select होती हैं
  - इस option के साथ, केवल specified dataset validation inputs के लिए use होता है
- **Notes**:
  - Dataset ID आपके dataloader config में configured dataset से match होना चाहिए
  - Dedicated eval dataset use करके consistent evaluation maintain करने के लिए useful
  - Conditioning models के लिए, dataset का conditioning data (यदि हो) भी use होगा
  - Flux Kontext के लिए validation dataset चुनने का सही तरीका यही है; `--validation_using_datasets` enable न करें

---

## Conditioning और Validation Modes को समझना

SimpleTuner conditioning inputs (reference images, control signals, आदि) use करने वाले models के लिए तीन मुख्य paradigms support करता है:

### 1. Models जो Conditioning REQUIRE करते हैं

कुछ models conditioning inputs के बिना function नहीं कर सकते:

- **Flux Kontext**: Edit-style training के लिए हमेशा reference images चाहिए
- **ControlNet training**: Control signal images require करता है

इन models के लिए, conditioning dataset mandatory है। WebUI conditioning options को required दिखाएगी, और training इनके बिना fail होगी।
Flux Kontext validation भी इसी conditioning-based path पर रहती है। Validation के लिए edit dataset चुनने हेतु `--eval_dataset_id` उपयोग करें और `--validation_using_datasets` disabled रखें।

### 2. Models जो Optional Conditioning SUPPORT करते हैं

कुछ models text-to-image AND image-to-image दोनों modes में operate कर सकते हैं:

- **Flux2**: Optional reference images के साथ dual T2I/I2I training support करता है
- **LTXVideo2**: Optional first-frame conditioning के साथ T2V और I2V (image-to-video) दोनों support करता है
- **LongCat-Video**: Optional frame conditioning support करता है
- **HunyuanVideo i2v**: First-frame conditioning के साथ I2V support करता है (flavours: `i2v-480p`, `i2v-720p`, आदि)
- **WAN i2v**: First-frame conditioning के साथ I2V support करता है
- **Kandinsky5Video i2v**: First-frame conditioning के साथ I2V support करता है

इन models के लिए, आप conditioning datasets ADD कर सकते हैं पर जरूरी नहीं। WebUI conditioning options को optional दिखाएगी।

**I2V Validation Shortcut**: i2v video models के लिए, आप `--validation_using_datasets` को image dataset (via `--eval_dataset_id` specified) के साथ use कर सकते हैं validation conditioning images directly प्राप्त करने के लिए, training में use होने वाली full conditioning dataset pairing setup की जरूरत के बिना।

### 3. Validation Modes

| Mode | Flag | Behavior |
|------|------|----------|
| **Text-to-Image/Video** | (default) | केवल text prompts से generate |
| **Dataset-based (img2img)** | `--validation_using_datasets` | Datasets से images partially denoise |
| **Dataset-based (i2v)** | `--validation_using_datasets` | i2v video models के लिए, images को first-frame conditioning के रूप में use |
| **Conditioning-based** | (auto जब conditioning configured हो) | Validation के दौरान conditioning inputs use |

**Modes combine करना**: जब model conditioning support करता है AND `--validation_using_datasets` enabled है:
- Validation system datasets से images लेता है
- यदि उन datasets में conditioning data है, तो automatically use होता है
- `--eval_dataset_id` use करें control करने के लिए कौन सा dataset inputs provide करे

**I2V models के साथ `--validation_using_datasets`**: i2v video models (HunyuanVideo, WAN, Kandinsky5Video) के लिए, यह flag enable करने पर validation के लिए simple image dataset use कर सकते हैं। Images validation videos generate करने के लिए first-frame conditioning inputs के रूप में use होती हैं, complex conditioning dataset pairing setup की जरूरत के बिना।

**Flux Kontext के साथ `--validation_using_datasets`**: यह flag enable न करें। Kontext edit-only है और validation अपनी सामान्य paired image + conditioning dataset path से चलाता है। इसके बजाय edit dataset चुनने के लिए `--eval_dataset_id` उपयोग करें।

### Conditioning Data Types

Different models different conditioning data expect करते हैं:

| Type | Models | Dataset Setting |
|------|--------|-----------------|
| `conditioning` | ControlNet, Control | Dataset config में `type: conditioning` |
| `image` | Flux Kontext | `type: image` (standard image dataset) |
| `latents` | Flux, Flux2 | Conditioning automatically VAE-encoded होता है |

---

### `--caption_strategy`

- **What**: image captions derive करने की रणनीति। **Choices**: `textfile`, `filename`, `parquet`, `instanceprompt`
- **Why**: training images के captions कैसे बनाए जाएँ, यह तय करता है।
  - `textfile` image के समान फ़ाइल‑नाम वाली `.txt` फ़ाइल के contents उपयोग करेगा
  - `filename` फ़ाइल‑नाम को कुछ cleanup करके caption के रूप में उपयोग करेगा।
  - `parquet` dataset में parquet फ़ाइल होने पर `caption` column उपयोग करेगा जब तक `parquet_caption_column` न दिया गया हो। सभी captions मौजूद होने चाहिए जब तक `parquet_fallback_caption_column` न दिया गया हो।
  - `instanceprompt` dataset config में `instance_prompt` मान को हर image के prompt के रूप में उपयोग करेगा।

### `--conditioning_multidataset_sampling` {#--conditioning_multidataset_sampling}

- **What**: multiple conditioning datasets से sampling कैसे की जाए। **Choices**: `combined`, `random`
- **Why**: multiple conditioning datasets (उदा., multiple reference images या control signals) के साथ training करते समय यह तय करता है कि उन्हें कैसे उपयोग किया जाए:
  - `combined` conditioning inputs को stitch करके training में एक साथ दिखाता है। यह multi‑image compositing tasks के लिए उपयोगी है।
  - `random` हर sample के लिए एक conditioning dataset रैंडम रूप से चुनता है, training के दौरान conditions बदलते हुए।
- **Note**: `combined` उपयोग करने पर आप conditioning datasets पर अलग `captions` परिभाषित नहीं कर सकते; source dataset के captions ही उपयोग होते हैं।
- **See also**: multiple conditioning datasets कॉन्फ़िगर करने के लिए [DATALOADER.md](DATALOADER.md#conditioning_data) देखें।

---

## 🎛 Training Parameters

### `--num_train_epochs`

- **What**: training epochs की संख्या (कितनी बार सभी images देखी जाती हैं)। इसे 0 सेट करने पर `--max_train_steps` को प्राथमिकता मिलती है।
- **Why**: image repeats की संख्या तय करता है, जो training duration को प्रभावित करता है। अधिक epochs आम तौर पर overfitting का कारण बनते हैं, लेकिन आपके concepts सीखने के लिए आवश्यक हो सकते हैं। उचित मान 5 से 50 के बीच हो सकता है।

### `--max_train_steps`

- **What**: इतने training steps के बाद training बंद होती है। 0 सेट करने पर `--num_train_epochs` को प्राथमिकता मिलती है।
- **Why**: training को छोटा करने के लिए उपयोगी।

### `--ignore_final_epochs`

- **What**: अंतिम गिने गए epochs को ignore करके `--max_train_steps` को प्राथमिकता देता है।
- **Why**: dataloader length बदलने पर epoch calculation बदल जाती है और training जल्दी खत्म हो सकती है। यह विकल्प अंतिम epochs को ignore करके `--max_train_steps` तक training जारी रखता है।

### `--learning_rate`

- **What**: संभावित warmup के बाद initial learning rate।
- **Why**: learning rate gradient updates के लिए एक तरह का "step size" है — बहुत अधिक होने पर solution से आगे निकल जाते हैं, बहुत कम होने पर ideal solution तक नहीं पहुँचते। `full` tune के लिए न्यूनतम मान `1e-7` और अधिकतम `1e-6` तक हो सकता है, जबकि `lora` tuning के लिए न्यूनतम `1e-5` और अधिकतम `1e-3` तक हो सकता है। उच्च learning rate उपयोग करने पर EMA network और warmup लाभदायक होते हैं — देखें `--use_ema`, `--lr_warmup_steps`, और `--lr_scheduler`।

### `--lr_scheduler`

- **What**: समय के साथ learning rate कैसे scale हो।
- **Choices**: constant, constant_with_warmup, cosine, cosine_with_restarts, **polynomial** (अनुशंसित), linear
- **Why**: loss landscape को explore करने के लिए learning rate को समय‑समय पर बदलना उपयोगी है। cosine schedule डिफ़ॉल्ट है, जिससे training दो extremes के बीच smooth तरीके से चलती है। constant learning rate में अक्सर बहुत ऊँचा या बहुत कम मान चुन लिया जाता है, जिससे divergence (बहुत ऊँचा) या local minima में फँसना (बहुत कम) होता है। polynomial schedule warmup के साथ सबसे अच्छा रहता है, जहाँ यह धीरे‑धीरे `learning_rate` तक पहुँचता है और फिर धीरे‑धीरे `--lr_end` के पास पहुँचता है।

### `--optimizer`

- **What**: training के लिए optimizer।
- **Choices**: adamw_bf16, ao-adamw8bit, ao-adamw4bit, ao-adamfp8, ao-adamwfp8, adamw_schedulefree, adamw_schedulefree+aggressive, adamw_schedulefree+no_kahan, optimi-stableadamw, optimi-adamw, optimi-lion, optimi-radam, optimi-ranger, optimi-adan, optimi-adam, optimi-sgd, soap, bnb-adagrad, bnb-adagrad8bit, bnb-adam, bnb-adam8bit, bnb-adamw, bnb-adamw8bit, bnb-adamw-paged, bnb-adamw8bit-paged, bnb-lion, bnb-lion8bit, bnb-lion-paged, bnb-lion8bit-paged, bnb-ademamix, bnb-ademamix8bit, bnb-ademamix-paged, bnb-ademamix8bit-paged, prodigy

> Note: कुछ optimisers non‑NVIDIA hardware पर उपलब्ध नहीं हो सकते।

### `--optimizer_config`

- **What**: optimizer settings को fine‑tune करें।
- **Why**: optimizers में बहुत सारे settings होते हैं, हर एक के लिए CLI argument देना व्यावहारिक नहीं है। इसलिए आप comma‑separated सूची देकर default settings override कर सकते हैं।
- **Example**: **prodigy** optimizer के लिए `d_coef` सेट करना: `--optimizer_config=d_coef=0.1`

> Note: Optimizer betas dedicated parameters `--optimizer_beta1`, `--optimizer_beta2` से override किए जाते हैं।

### `--train_batch_size`

- **What**: training data loader के लिए batch size।
- **Why**: model memory consumption, convergence quality, और training speed प्रभावित करता है। बड़ा batch size सामान्यतः बेहतर परिणाम देता है, लेकिन बहुत बड़ा batch size overfitting या destabilized training का कारण बन सकता है और training अवधि भी बढ़ा सकता है। प्रयोग ज़रूरी है, लेकिन सामान्यतः लक्ष्य यह है कि training speed घटाए बिना VRAM अधिकतम उपयोग में हो।

### `--gradient_accumulation_steps`

- **What**: backward/update pass करने से पहले accumulate किए जाने वाले steps की संख्या; यह memory बचाने के लिए काम को कई batches में बाँट देता है, लेकिन training runtime बढ़ता है।
- **Why**: बड़े models या datasets को संभालने में उपयोगी।

> Note: gradient accumulation steps उपयोग करते समय किसी भी optimizer के लिए fused backward pass enable न करें।

### `--allow_dataset_oversubscription` {#--allow_dataset_oversubscription}

- **What**: dataset effective batch size से छोटा होने पर `repeats` स्वतः adjust करता है।
- **Why**: multi‑GPU कॉन्फ़िगरेशन के लिए न्यूनतम requirements पूरी न होने पर training failure को रोकता है।
- **How it works**:
  - **effective batch size** की गणना करता है: `train_batch_size × num_gpus × gradient_accumulation_steps`
  - यदि किसी aspect bucket में effective batch size से कम samples हैं, तो `repeats` स्वतः बढ़ाता है
  - केवल तब लागू होता है जब dataset config में `repeats` explicitly सेट न हो
  - adjustment और reasoning दिखाने के लिए warning लॉग करता है
- **Use cases**:
  - कई GPUs के साथ छोटे datasets (< 100 images)
  - datasets फिर से कॉन्फ़िगर किए बिना अलग batch sizes के साथ experimentation
  - full dataset इकट्ठा करने से पहले prototyping
- **Example**: 25 images, 8 GPUs, और `train_batch_size=4` के साथ effective batch size 32 होता है। यह flag `repeats=1` स्वतः सेट करेगा ताकि 50 samples (25 × 2) मिलें।
- **Note**: यह dataloader कॉन्फ़िग में manually‑set `repeats` values को override **नहीं** करेगा। `--disable_bucket_pruning` की तरह, यह flag बिना surprising behavior के सुविधा देता है।

Multi‑GPU training के लिए dataset sizing पर अधिक विवरण [DATALOADER.md](DATALOADER.md#automatic-dataset-oversubscription) में देखें।

---

## 🛠 Advanced Optimizations

### `--use_ema`

- **What**: मॉडल के training जीवनकाल में weights का exponential moving average रखना, मॉडल को समय‑समय पर खुद में back‑merge करने जैसा है।
- **Why**: अधिक system resources और थोड़ा अधिक runtime खर्च करके training stability बेहतर होती है।

### `--ema_device`

- **Choices**: `cpu`, `accelerator`; default: `cpu`
- **What**: EMA weights updates के बीच कहाँ रखी जाएँ।
- **Why**: EMA को accelerator पर रखने से updates तेज़ होते हैं लेकिन VRAM लागत बढ़ती है। CPU पर रखने से memory दबाव कम होता है, लेकिन `--ema_cpu_only` सेट न होने पर weights को शटल करना पड़ता है।

### `--ema_cpu_only`

- **What**: `--ema_device=cpu` होने पर EMA weights को updates के लिए accelerator पर वापस ले जाने से रोकता है।
- **Why**: बड़े EMAs के लिए host‑to‑device transfer समय और VRAM उपयोग बचाता है। `--ema_device=accelerator` होने पर इसका प्रभाव नहीं है क्योंकि weights पहले से accelerator पर हैं।

### `--ema_foreach_disable`

- **What**: EMA updates के लिए `torch._foreach_*` kernels का उपयोग disable करता है।
- **Why**: कुछ back‑ends या hardware combinations में foreach ops समस्याग्रस्त होते हैं। इन्हें disable करने पर scalar implementation उपयोग होती है, जिससे updates थोड़ा धीमे हो जाते हैं।

### `--ema_update_interval`

- **What**: EMA shadow parameters कितनी बार update हों, यह कम करता है।
- **Why**: हर step पर update करना कई workflows के लिए आवश्यक नहीं। उदाहरण के लिए, `--ema_update_interval=100` हर 100 optimizer steps पर EMA update करेगा, जिससे `--ema_device=cpu` या `--ema_cpu_only` के साथ overhead घटता है।

### `--ema_decay`

- **What**: EMA updates लागू करते समय smoothing factor नियंत्रित करता है।
- **Why**: उच्च मान (उदा. `0.999`) EMA को धीरे प्रतिक्रिया देने देते हैं लेकिन बहुत स्थिर weights देते हैं। कम मान (उदा. `0.99`) नए training signals के साथ तेज़ adapt होते हैं।

### `--snr_gamma`

- **What**: min‑SNR weighted loss factor उपयोग करता है।
- **Why**: Minimum SNR gamma loss factor को schedule में timestep की स्थिति के अनुसार weigh करता है। बहुत noisy timesteps का योगदान कम होता है और कम‑noise timesteps का योगदान बढ़ता है। मूल पेपर द्वारा अनुशंसित मान **5** है, लेकिन आप **1** से **20** तक मान उपयोग कर सकते हैं (आमतौर पर 20 को max माना जाता है; 20 से ऊपर बदलाव कम होता है)। **1** सबसे मजबूत प्रभाव देता है।

### `--use_soft_min_snr`

- **What**: loss landscape पर अधिक gradual weighting के साथ मॉडल ट्रेन करता है।
- **Why**: pixel diffusion models training में विशिष्ट loss weighting schedule के बिना degrade हो सकते हैं। DeepFloyd में soft‑min‑snr‑gamma लगभग अनिवार्य पाया गया। Latent diffusion models में आपको सफलता मिल सकती है, लेकिन छोटे प्रयोगों में इससे blurry results होने की संभावना दिखी।

### `--diff2flow_enabled`

- **What**: epsilon या v‑prediction models के लिए Diffusion‑to‑Flow bridge सक्षम करता है।
- **Why**: model architecture बदले बिना standard diffusion objectives वाले मॉडल्स को flow‑matching targets (noise - latents) उपयोग करने देता है।
- **Note**: Experimental फीचर।

### `--diff2flow_loss`

- **What**: native prediction loss की बजाय Flow Matching loss पर training।
- **Why**: `--diff2flow_enabled` के साथ enabled होने पर, loss को model के native target (epsilon या velocity) की जगह flow target (noise - latents) के खिलाफ compute करता है।
- **Note**: `--diff2flow_enabled` आवश्यक है।

### `--scheduled_sampling_max_step_offset`

- **What**: training के दौरान "roll out" होने वाले steps की अधिकतम संख्या।
- **Why**: Scheduled Sampling (Rollout) सक्षम करता है, जहाँ मॉडल training के दौरान कुछ steps के लिए अपने inputs खुद generate करता है। इससे मॉडल अपनी गलतियाँ सुधारना सीखता है और exposure bias घटता है।
- **Default**: 0 (disabled). सक्षम करने के लिए सकारात्मक integer (उदा., 5 या 10) दें।

### `--scheduled_sampling_strategy`

- **What**: rollout offset चुनने की रणनीति।
- **Choices**: `uniform`, `biased_early`, `biased_late`.
- **Default**: `uniform`.
- **Why**: rollout लंबाइयों का distribution नियंत्रित करता है। `uniform` समान रूप से sample करता है; `biased_early` छोटे rollouts को प्राथमिकता देता है; `biased_late` लंबे rollouts को प्राथमिकता देता है।

### `--scheduled_sampling_probability`

- **What**: किसी sample के लिए non‑zero rollout offset लागू होने की संभावना।
- **Default**: 0.0.
- **Why**: scheduled sampling कितनी बार लागू हो, यह नियंत्रित करता है। 0.0 इसे disable करता है चाहे `max_step_offset` > 0 हो। 1.0 हर sample पर लागू करता है।

### `--scheduled_sampling_prob_start`

- **What**: ramp की शुरुआत में scheduled sampling की initial probability।
- **Default**: 0.0.

### `--scheduled_sampling_prob_end`

- **What**: ramp के अंत में scheduled sampling की final probability।
- **Default**: 0.5.

### `--scheduled_sampling_ramp_steps`

- **What**: `prob_start` से `prob_end` तक probability बढ़ाने के steps की संख्या।
- **Default**: 0 (कोई ramp नहीं)।

### `--scheduled_sampling_start_step`

- **What**: scheduled sampling ramp शुरू करने का global step।
- **Default**: 0.0.

### `--scheduled_sampling_ramp_shape`

- **What**: probability ramp का आकार।
- **Choices**: `linear`, `cosine`.
- **Default**: `linear`.

### `--scheduled_sampling_sampler`

- **What**: rollout generation steps के लिए उपयोग किया जाने वाला solver।
- **Choices**: `unipc`, `euler`, `dpm`.
- **Default**: `unipc`.

### `--scheduled_sampling_order`

- **What**: rollout के लिए solver का order।
- **Default**: 2.

### `--scheduled_sampling_reflexflow`

- **What**: flow‑matching models के लिए scheduled sampling के दौरान ReflexFlow‑style enhancements (anti‑drift + frequency‑compensated weighting) सक्षम करें।
- **Why**: directional regularization और bias‑aware loss weighting जोड़कर flow‑matching models में exposure bias घटाता है।
- **Default**: `--scheduled_sampling_max_step_offset` > 0 होने पर flow‑matching models के लिए auto‑enable; `--scheduled_sampling_reflexflow=false` से override करें।

### `--scheduled_sampling_reflexflow_alpha`

- **What**: exposure bias से निकले frequency‑compensation weight का scaling factor।
- **Default**: 1.0.
- **Why**: flow‑matching models में rollout के दौरान बड़े exposure bias वाले क्षेत्रों को अधिक weight देता है।

### `--scheduled_sampling_reflexflow_beta1`

- **What**: ReflexFlow anti‑drift (directional) regularizer का weight।
- **Default**: 10.0.
- **Why**: flow‑matching models में scheduled sampling उपयोग करते समय मॉडल को target clean sample के साथ अपनी predicted direction align करने के लिए कितना मजबूती से प्रोत्साहित किया जाए, यह नियंत्रित करता है।

### `--scheduled_sampling_reflexflow_beta2`

- **What**: ReflexFlow frequency‑compensation (loss reweighting) term का weight।
- **Default**: 1.0.
- **Why**: reweighted flow‑matching loss को scale करता है, जैसा ReflexFlow paper में β₂ knob के रूप में बताया गया है।

---

## 🎯 CREPA (Cross-frame Representation Alignment)

CREPA एक regularization तकनीक है जो video diffusion models की fine‑tuning में temporal consistency सुधारती है, adjacent frames से pretrained visual features के साथ hidden states align करके। यह पेपर ["Cross-Frame Representation Alignment for Fine-Tuning Video Diffusion Models"](https://arxiv.org/abs/2506.09229) पर आधारित है।

### `--crepa_enabled`

- **What**: training के दौरान CREPA regularization सक्षम करें।
- **Why**: पड़ोसी frames के DINOv2 features के साथ DiT hidden states align करके वीडियो frames में semantic consistency बढ़ाता है।
- **Default**: `false`
- **Note**: केवल Transformer-आधारित diffusion models (DiT शैली) पर लागू। UNet models (SDXL, SD1.5, Kolors) के लिए U-REPA का उपयोग करें।

### `--crepa_block_index`

- **What**: alignment के लिए किस transformer block के hidden states उपयोग हों।
- **Why**: पेपर CogVideoX के लिए block 8 और Hunyuan Video के लिए block 10 सुझाता है। शुरुआती blocks अक्सर बेहतर काम करते हैं क्योंकि वे DiT का "encoder" हिस्सा होते हैं।
- **Required**: हाँ, जब CREPA enabled हो।

### `--crepa_lambda`

- **What**: मुख्य training loss के मुकाबले CREPA alignment loss का weight।
- **Why**: alignment regularization training को कितना प्रभावित करे, यह नियंत्रित करता है। पेपर CogVideoX के लिए 0.5 और Hunyuan Video के लिए 1.0 उपयोग करता है।
- **Default**: `0.5`

### `--crepa_adjacent_distance`

- **What**: neighbor frame alignment के लिए दूरी `d`।
- **Why**: पेपर की Equation 6 के अनुसार, $K = \{f-d, f+d\}$ बताता है कि किन neighboring frames से align करना है। `d=1` होने पर हर frame अपने immediate neighbors से align करता है।
- **Default**: `1`

### `--crepa_adjacent_tau`

- **What**: exponential distance weighting के लिए temperature coefficient।
- **Why**: $e^{-|k-f|/\tau}$ के जरिए alignment weight कितनी जल्दी decay हो, यह नियंत्रित करता है। कम मान immediate neighbors पर अधिक जोर देता है।
- **Default**: `1.0`

### `--crepa_cumulative_neighbors`

- **What**: adjacent mode की जगह cumulative mode उपयोग करें।
- **Why**:
  - **Adjacent mode (डिफ़ॉल्ट)**: केवल exact दूरी `d` वाले frames से align करता है (पेपर के $K = \{f-d, f+d\}$ जैसा)
  - **Cumulative mode**: दूरी 1 से `d` तक सभी frames से align करता है, smoother gradients देता है
- **Default**: `false`

### `--crepa_normalize_neighbour_sum`

- **What**: neighbor‑sum alignment को per‑frame weight sum से normalize करें।
- **Why**: `crepa_alignment_score` को [-1, 1] में रखता है और loss scale को अधिक literal बनाता है। यह पेपर की Eq. (6) से experimental deviation है।
- **Default**: `false`

### `--crepa_normalize_by_frames`

- **What**: alignment loss को frames की संख्या से normalize करें।
- **Why**: video length के बावजूद loss scale consistent रहता है। Disable करने पर लंबे videos को stronger alignment signal मिलता है।
- **Default**: `true`

### `--crepa_spatial_align`

- **What**: जब DiT और encoder के token counts अलग हों तो spatial interpolation उपयोग करें।
- **Why**: DiT hidden states और DINOv2 features की spatial resolutions अलग हो सकती हैं। सक्षम होने पर bilinear interpolation उन्हें spatially align करता है; disabled होने पर global pooling fallback होता है।
- **Default**: `true`

### `--crepa_model`

- **What**: feature extraction के लिए कौन‑सा pretrained encoder उपयोग हो।
- **Why**: पेपर DINOv2‑g (ViT‑Giant) उपयोग करता है। `dinov2_vitb14` जैसे छोटे variants कम memory लेते हैं।
- **Default**: `dinov2_vitg14`
- **Choices**: `dinov2_vitg14`, `dinov2_vitb14`, `dinov2_vits14`

### `--crepa_encoder_frames_batch_size`

- **What**: external feature encoder parallel में कितने frames प्रोसेस करे। 0 या negative होने पर पूरे batch के सभी frames एक साथ प्रोसेस होते हैं। यदि संख्या divisor नहीं है, तो remainder छोटे batch के रूप में संभाला जाएगा।
- **Why**: DINO‑like encoders image models हैं, इसलिए वे VRAM बचाने के लिए frames को sliced batches में प्रोसेस कर सकते हैं, गति की कीमत पर।
- **Default**: `-1`

### `--crepa_use_backbone_features`

- **What**: external encoder skip करें और diffusion model के अंदर student block को teacher block के साथ align करें।
- **Why**: जब backbone के पास पहले से मजबूत semantic layer हो, तब DINOv2 लोड करने से बचता है।
- **Default**: `false`

### `--crepa_feature_source`

- **What**: चुनता है कि CREPA अपना teacher signal कहां से ले।
- **Why**: classic external encoder path के लिए `encoder`, internal block-to-block alignment के लिए `backbone`, और Self-Flow के cleaner EMA teacher view के लिए `self_flow` उपयोग करें।
- **Choices**: `encoder`, `backbone`, `self_flow`
- **Default**: `encoder`

### `--crepa_self_flow`

- **What**: Self-Flow mode enable करने वाला legacy boolean alias।
- **Why**: पुराने configs अब भी इसका उपयोग कर सकते हैं, लेकिन नए configs को `crepa_feature_source=self_flow` उपयोग करना चाहिए।
- **Default**: `false`
- **Note**: यह `crepa_use_backbone_features` और किसी दूसरे mode वाले `crepa_feature_source` के साथ conflict करता है।

### `--crepa_self_flow_mask_ratio`

- **What**: Self-Flow में alternate timestep पाने वाले tokens का fraction।
- **Why**: यह cleaner और noisier tokens के बीच information asymmetry नियंत्रित करता है। ऊंचे values self-supervised signal को मजबूत करते हैं, लेकिन training unstable भी कर सकते हैं।
- **Default**: `0.1`
- **Range**: `0.0` से `0.5`

### `--crepa_teacher_block_index`

- **What**: backbone features या Self-Flow उपयोग करते समय teacher block index।
- **Why**: external encoder के बिना earlier student block को later teacher block से align करने देता है। Self-Flow में यह explicitly जरूरी है ताकि EMA teacher deeper semantic layer से पढ़े।
- **Default**: backbone mode में नहीं दिया गया तो `crepa_block_index` उपयोग होगा; Self-Flow mode में यह required है।

### `--crepa_encoder_image_size`

- **What**: encoder के लिए input resolution।
- **Why**: DINOv2 models अपने training resolution पर बेहतर काम करते हैं। giant model 518x518 उपयोग करता है।
- **Default**: `518`

### `--crepa_scheduler`

- **What**: training के दौरान CREPA coefficient decay का schedule।
- **Why**: जैसे-जैसे training आगे बढ़े, CREPA regularization strength को कम करने देता है, deep encoder features पर overfitting रोकता है।
- **Options**: `constant`, `linear`, `cosine`, `polynomial`
- **Default**: `constant`

### `--crepa_warmup_steps`

- **What**: CREPA weight को 0 से `crepa_lambda` तक linearly ramp करने के लिए steps की संख्या।
- **Why**: gradual warmup CREPA regularization शुरू होने से पहले early training को stabilize करने में मदद कर सकता है।
- **Default**: `0`

### `--crepa_decay_steps`

- **What**: decay के लिए कुल steps (warmup के बाद)। 0 सेट करने पर पूरी training run पर decay होगा।
- **Why**: decay phase की duration नियंत्रित करता है। warmup पूरा होने के बाद decay शुरू होता है।
- **Default**: `0` (`max_train_steps` उपयोग होगा)

### `--crepa_lambda_end`

- **What**: decay पूरा होने के बाद final CREPA weight।
- **Why**: 0 सेट करने पर training के अंत में CREPA प्रभावी रूप से disable हो जाता है, text2video के लिए उपयोगी जहाँ CREPA artifacts पैदा कर सकता है।
- **Default**: `0.0`

### `--crepa_power`

- **What**: polynomial decay के लिए power factor। 1.0 = linear, 2.0 = quadratic, आदि।
- **Why**: higher values शुरुआत में तेज decay करते हैं जो अंत की ओर धीमा हो जाता है।
- **Default**: `1.0`

### `--crepa_cutoff_step`

- **What**: hard cutoff step जिसके बाद CREPA disable हो जाता है।
- **Why**: model temporal alignment पर converge होने के बाद CREPA disable करने के लिए उपयोगी।
- **Default**: `0` (कोई step-based cutoff नहीं)

### `--crepa_similarity_threshold`

- **What**: similarity EMA threshold जिस पर CREPA cutoff trigger होता है।
- **Why**: जब alignment score (`crepa_alignment_score`) का exponential moving average इस मान तक पहुँचता है, तो deep encoder features पर overfitting रोकने के लिए CREPA disable हो जाता है। text2video training के लिए विशेष रूप से उपयोगी। `crepa_normalize_neighbour_sum` enable न होने पर alignment score 1.0 से ऊपर जा सकता है।
- **Default**: None (disabled)

### `--crepa_similarity_ema_decay`

- **What**: similarity tracking के लिए exponential moving average decay factor।
- **Why**: higher values smoother tracking देते हैं (0.99 ≈ 100-step window), lower values changes पर तेज react करते हैं।
- **Default**: `0.99`

### `--crepa_threshold_mode`

- **What**: similarity threshold पहुँचने पर व्यवहार।
- **Options**: `permanent` (threshold hit होने पर CREPA permanently off रहता है), `recoverable` (similarity गिरने पर CREPA फिर से enable होता है)
- **Default**: `permanent`

### Example Configuration

```toml
# Enable CREPA for video fine-tuning
crepa_enabled = true
crepa_block_index = 8          # Adjust based on your model
crepa_lambda = 0.5
crepa_adjacent_distance = 1
crepa_adjacent_tau = 1.0
crepa_cumulative_neighbors = false
crepa_normalize_neighbour_sum = false
crepa_normalize_by_frames = true
crepa_spatial_align = true
crepa_model = "dinov2_vitg14"
crepa_encoder_frames_batch_size = -1
crepa_use_backbone_features = false
# crepa_teacher_block_index = 16
crepa_encoder_image_size = 518

# CREPA Scheduling (optional)
# crepa_scheduler = "cosine"           # Decay type: constant, linear, cosine, polynomial
# crepa_warmup_steps = 100             # Warmup before CREPA kicks in
# crepa_decay_steps = 1000             # Steps for decay (0 = entire training)
# crepa_lambda_end = 0.0               # Final weight after decay
# crepa_cutoff_step = 5000             # Hard cutoff step (0 = disabled)
# crepa_similarity_threshold = 0.9    # Similarity-based cutoff
# crepa_threshold_mode = "permanent"   # permanent or recoverable
```

---

## 🎯 U-REPA (UNet Representation Alignment)

U-REPA UNet आधारित diffusion models (SDXL, SD1.5, Kolors) के लिए regularization तकनीक है। यह UNet mid-block features को pretrained vision features के साथ align करता है और relative similarity structure रखने के लिए manifold loss जोड़ता है।

### `--urepa_enabled`

- **What**: training के दौरान U-REPA enable करें।
- **Why**: frozen vision encoder के साथ UNet mid-block features का representation alignment जोड़ता है।
- **Default**: `false`
- **Note**: केवल UNet models (SDXL, SD1.5, Kolors) पर लागू।

### `--urepa_lambda`

- **What**: मुख्य training loss के मुकाबले U-REPA alignment loss का weight।
- **Why**: regularization की strength नियंत्रित करता है।
- **Default**: `0.5`

### `--urepa_manifold_weight`

- **What**: manifold loss का weight (alignment loss के मुकाबले)।
- **Why**: relative similarity structure पर ज़ोर देता है (paper default 3.0)।
- **Default**: `3.0`

### `--urepa_model`

- **What**: frozen vision encoder के लिए torch hub identifier।
- **Why**: default DINOv2 ViT-G/14; छोटे मॉडल (जैसे `dinov2_vits14`) तेज़ होते हैं।
- **Default**: `dinov2_vitg14`

### `--urepa_encoder_image_size`

- **What**: vision encoder preprocessing के लिए input resolution।
- **Why**: encoder की native resolution रखें (DINOv2 ViT-G/14 के लिए 518; ViT-S/14 के लिए 224)।
- **Default**: `518`

### `--urepa_use_tae`

- **What**: full VAE की जगह Tiny AutoEncoder उपयोग करें।
- **Why**: तेज़ और कम VRAM, लेकिन decoded image quality कम।
- **Default**: `false`

### `--urepa_scheduler`

- **What**: training के दौरान U-REPA coefficient decay schedule।
- **Why**: training बढ़ने के साथ regularization strength कम करने में मदद।
- **Options**: `constant`, `linear`, `cosine`, `polynomial`
- **Default**: `constant`

### `--urepa_warmup_steps`

- **What**: U-REPA weight को 0 से `urepa_lambda` तक linearly बढ़ाने के steps।
- **Why**: शुरुआती training को stabilize करता है।
- **Default**: `0`

### `--urepa_decay_steps`

- **What**: decay के लिए कुल steps (warmup के बाद)। 0 मतलब पूरे training में decay।
- **Why**: decay phase की duration नियंत्रित करता है।
- **Default**: `0` (`max_train_steps`)

### `--urepa_lambda_end`

- **What**: decay के बाद final U-REPA weight।
- **Why**: 0 रखने पर training के अंत में U-REPA effectively disable हो जाता है।
- **Default**: `0.0`

### `--urepa_power`

- **What**: polynomial decay power। 1.0 = linear, 2.0 = quadratic आदि।
- **Why**: बड़ा मान शुरुआत में तेज़ decay और अंत में धीमा करता है।
- **Default**: `1.0`

### `--urepa_cutoff_step`

- **What**: इस step के बाद U-REPA बंद।
- **Why**: alignment converge होने के बाद बंद करने के लिए।
- **Default**: `0` (no cutoff)

### `--urepa_similarity_threshold`

- **What**: similarity EMA threshold जिस पर U-REPA cutoff ट्रिगर हो।
- **Why**: `urepa_similarity` का EMA इस मान तक पहुंचते ही U-REPA disable होता है, overfitting रोकने के लिए।
- **Default**: None (disabled)

### `--urepa_similarity_ema_decay`

- **What**: similarity tracking के लिए EMA decay factor।
- **Why**: बड़ा मान smooth (0.99 ≈ 100-step window), छोटा मान तेज़ प्रतिक्रिया।
- **Default**: `0.99`

### `--urepa_threshold_mode`

- **What**: threshold पहुंचने पर व्यवहार।
- **Options**: `permanent` (एक बार बंद तो हमेशा बंद), `recoverable` (similarity गिरने पर फिर enable)
- **Default**: `permanent`

### Example Configuration

```toml
# UNet fine-tuning के लिए U-REPA enable करें (SDXL, SD1.5, Kolors)
urepa_enabled = true
urepa_lambda = 0.5
urepa_manifold_weight = 3.0
urepa_model = "dinov2_vitg14"
urepa_encoder_image_size = 518
urepa_use_tae = false

# U-REPA Scheduling (optional)
# urepa_scheduler = "cosine"           # Decay type: constant, linear, cosine, polynomial
# urepa_warmup_steps = 100             # U-REPA शुरू होने से पहले warmup
# urepa_decay_steps = 1000             # Decay steps (0 = पूरे training में)
# urepa_lambda_end = 0.0               # Decay के बाद final weight
# urepa_cutoff_step = 5000             # Hard cutoff step (0 = disabled)
# urepa_similarity_threshold = 0.9     # Similarity-based cutoff
# urepa_threshold_mode = "permanent"   # permanent या recoverable
```

---

## 🔄 Checkpointing and Resumption

### `--checkpoint_step_interval` (alias: `--checkpointing_steps`)

- **What**: training state checkpoints कितने steps पर सेव हों (steps में interval)।
- **Why**: training resume और inference के लिए उपयोगी। हर *n* iterations पर Diffusers filesystem layout में `.safetensors` format का partial checkpoint सेव होगा।

---

## 🔁 LayerSync (Hidden State Self-Alignment)

LayerSync एक "student" layer को उसी transformer के एक मजबूत "teacher" layer से match करने के लिए प्रोत्साहित करता है, hidden tokens पर cosine similarity का उपयोग करके।

### `--layersync_enabled`

- **What**: एक ही मॉडल के अंदर दो transformer blocks के बीच LayerSync hidden‑state alignment सक्षम करें।
- **Notes**: hidden‑state buffer allocate करता है; required flags missing हों तो startup पर error देता है।
- **Default**: `false`

### `--layersync_student_block`

- **What**: student anchor के रूप में उपयोग होने वाला transformer block index।
- **Indexing**: LayerSync पेपर‑style 1‑based depths या 0‑based layer ids स्वीकार करता है; implementation पहले `idx-1` आज़माता है, फिर `idx`।
- **Required**: हाँ, जब LayerSync enabled हो।

### `--layersync_teacher_block`

- **What**: teacher target के रूप में उपयोग होने वाला transformer block index (student से गहरा हो सकता है)।
- **Indexing**: student block की तरह ही 1‑based‑first, फिर 0‑based fallback।
- **Default**: omit होने पर student block ही उपयोग होता है ताकि loss self‑similarity बन जाए।

### `--layersync_lambda`

- **What**: student और teacher hidden states के बीच LayerSync cosine alignment loss (negative cosine similarity) का weight।
- **Effect**: base loss के ऊपर जोड़ा गया auxiliary regularizer scale करता है; उच्च मान student tokens को teacher tokens से अधिक alignment के लिए push करते हैं।
- **Upstream name**: मूल LayerSync codebase में `--reg-weight`.
- **Required**: LayerSync enabled होने पर > 0 होना चाहिए (अन्यथा training abort होती है)।
- **Default**: LayerSync enabled होने पर `0.2` (reference repo से मेल), अन्यथा `0.0`.

Upstream option mapping (LayerSync → SimpleTuner):
- `--encoder-depth` → `--layersync_student_block` (upstream जैसा 1‑based depth या 0‑based layer index स्वीकार करता है)
- `--gt-encoder-depth` → `--layersync_teacher_block` (1‑based preferred; omit होने पर student पर डिफ़ॉल्ट)
- `--reg-weight` → `--layersync_lambda`

> Notes: LayerSync हमेशा similarity से पहले teacher hidden state detach करता है, reference implementation से मेल खाते हुए। यह उन मॉडलों पर निर्भर करता है जो transformer hidden states expose करते हैं (SimpleTuner के अधिकांश transformer backbones) और hidden‑state buffer के लिए per‑step memory जोड़ता है; VRAM tight हो तो disable करें।

### `--checkpoint_epoch_interval`

- **What**: हर N पूर्ण epochs पर checkpointing चलाएँ।
- **Why**: step‑based checkpoints को पूरक करता है, ताकि multi‑dataset sampling के कारण step counts बदलने पर भी epoch boundaries पर state capture हो सके।

### `--resume_from_checkpoint`

- **What**: training resume करना है या नहीं और कहाँ से। `latest`, local checkpoint नाम/पथ, या S3/R2 URI स्वीकार करता है।
- **Why**: saved state से training जारी रखने देता है, manual रूप से या latest उपलब्ध checkpoint से।
- **Remote resume**: पूरा URI (`s3://bucket/jobs/.../checkpoint-100`) या bucket-relative key (`jobs/.../checkpoint-100`) दें। `latest` केवल local `output_dir` पर काम करता है。
- **Requirements**: remote resume के लिए publishing_config में S3 entry (bucket + credentials) चाहिए जो checkpoint read कर सके।
- **Notes**: remote checkpoints में `checkpoint_manifest.json` होना चाहिए (हाल की SimpleTuner runs से generated)। एक checkpoint में `unet` और वैकल्पिक रूप से `unet_ema` subfolder होता है। `unet` को किसी भी Diffusers layout SDXL मॉडल में रखा जा सकता है, जिससे इसे सामान्य मॉडल की तरह उपयोग किया जा सकता है।

> ℹ️ PixArt, SD3, या Hunyuan जैसे transformer मॉडल `transformer` और `transformer_ema` subfolder नाम उपयोग करते हैं।

### `--disk_low_threshold`

- **What**: checkpoint saves से पहले आवश्यक न्यूनतम खाली disk space।
- **Why**: disk full errors से training crash को रोकता है, कम space का जल्दी पता लगाकर configured action लेता है।
- **Format**: size string जैसे `100G`, `50M`, `1T`, `500K`, या plain bytes।
- **Default**: None (feature disabled)

### `--disk_low_action`

- **What**: disk space threshold से कम होने पर लिया जाने वाला action।
- **Choices**: `stop`, `wait`, `script`
- **Default**: `stop`
- **Behavior**:
  - `stop`: error message के साथ training तुरंत समाप्त करता है।
  - `wait`: space उपलब्ध होने तक हर 30 seconds में loop करता है। अनिश्चित काल तक प्रतीक्षा कर सकता है।
  - `script`: space खाली करने के लिए `--disk_low_script` द्वारा specified script चलाता है।

### `--disk_low_script`

- **What**: disk space कम होने पर चलाने के लिए cleanup script का path।
- **Why**: disk space कम होने पर automated cleanup (जैसे पुराने checkpoints हटाना, cache clear करना) की अनुमति देता है।
- **Notes**: केवल `--disk_low_action=script` होने पर उपयोग होता है। script executable होना चाहिए। यदि script fail होती है या पर्याप्त space खाली नहीं करती, training error के साथ रुक जाएगी।
- **Default**: None

---

## 📊 Logging and Monitoring

### `--logging_dir`

- **What**: TensorBoard logs के लिए directory।
- **Why**: training progress और performance metrics मॉनिटर करने देता है।

### `--report_to`

- **What**: results और logs रिपोर्ट करने के लिए platform निर्दिष्ट करता है।
- **Why**: TensorBoard, wandb, या comet_ml जैसी platforms के साथ integration सक्षम करता है। multiple trackers पर रिपोर्ट करने के लिए comma से अलग values उपयोग करें;
- **Choices**: wandb, tensorboard, comet_ml

## Environment configuration variables

ऊपर दिए गए विकल्प अधिकतर `config.json` पर लागू होते हैं — लेकिन कुछ entries `config.env` में सेट करनी पड़ती हैं।

- `TRAINING_NUM_PROCESSES` को सिस्टम में GPUs की संख्या पर सेट करें। अधिकांश उपयोग‑मामलों में इससे DistributedDataParallel (DDP) training सक्षम हो जाती है। यदि आप `config.env` उपयोग नहीं करना चाहते, तो `config.json` में `num_processes` उपयोग करें।
- `TRAINING_DYNAMO_BACKEND` डिफ़ॉल्ट रूप से `no` है, लेकिन इसे किसी भी समर्थित torch.compile backend (उदा. `inductor`, `aot_eager`, `cudagraphs`) पर सेट किया जा सकता है और `--dynamo_mode`, `--dynamo_fullgraph`, या `--dynamo_use_regional_compilation` के साथ finer tuning के लिए जोड़ा जा सकता है
- `SIMPLETUNER_LOG_LEVEL` डिफ़ॉल्ट रूप से `INFO` है, लेकिन issue reports के लिए `debug.log` में अधिक जानकारी जोड़ने हेतु इसे `DEBUG` पर सेट किया जा सकता है
- `VENV_PATH` को आपके python virtual env की लोकेशन पर सेट किया जा सकता है यदि वह सामान्य `.venv` लोकेशन में नहीं है
- `ACCELERATE_EXTRA_ARGS` को unset छोड़ा जा सकता है, या इसमें `--multi_gpu` या FSDP‑specific flags जैसे अतिरिक्त arguments जोड़े जा सकते हैं

---

यह एक बेसिक overview है ताकि आप शुरुआत कर सकें। पूर्ण options सूची और अधिक विस्तृत स्पष्टीकरण के लिए, कृपया पूरी specification देखें:

```
usage: train.py [-h] --model_family
                {kolors,auraflow,omnigen,flux,deepfloyd,cosmos2image,sana,qwen_image,pixart_sigma,sdxl,sd1x,sd2x,wan,hidream,sd3,lumina2,ltxvideo}
                [--model_flavour MODEL_FLAVOUR] [--controlnet [CONTROLNET]]
                [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH]
                --output_dir OUTPUT_DIR [--logging_dir LOGGING_DIR]
                --model_type {full,lora} [--seed SEED]
                [--resolution RESOLUTION]
                [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                [--prediction_type {epsilon,v_prediction,sample,flow_matching}]
                [--pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH]
                [--vae_dtype {default,fp32,fp16,bf16}]
                [--vae_cache_ondemand [VAE_CACHE_ONDEMAND]]
                [--accelerator_cache_clear_interval ACCELERATOR_CACHE_CLEAR_INTERVAL]
                [--aspect_bucket_rounding {1,2,3,4,5,6,7,8,9}]
                [--base_model_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_1_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_2_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_3_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_4_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--gradient_checkpointing_interval GRADIENT_CHECKPOINTING_INTERVAL]
                [--offload_during_startup [OFFLOAD_DURING_STARTUP]]
                [--quantize_via {cpu,accelerator,pipeline}]
                [--quantization_config QUANTIZATION_CONFIG]
                [--fuse_qkv_projections [FUSE_QKV_PROJECTIONS]]
                [--control [CONTROL]]
                [--controlnet_custom_config CONTROLNET_CUSTOM_CONFIG]
                [--controlnet_model_name_or_path CONTROLNET_MODEL_NAME_OR_PATH]
                [--tread_config TREAD_CONFIG]
                [--pretrained_transformer_model_name_or_path PRETRAINED_TRANSFORMER_MODEL_NAME_OR_PATH]
                [--pretrained_transformer_subfolder PRETRAINED_TRANSFORMER_SUBFOLDER]
                [--pretrained_unet_model_name_or_path PRETRAINED_UNET_MODEL_NAME_OR_PATH]
                [--pretrained_unet_subfolder PRETRAINED_UNET_SUBFOLDER]
                [--pretrained_t5_model_name_or_path PRETRAINED_T5_MODEL_NAME_OR_PATH]
                [--pretrained_gemma_model_name_or_path PRETRAINED_GEMMA_MODEL_NAME_OR_PATH]
                [--revision REVISION] [--variant VARIANT]
                [--base_model_default_dtype {bf16,fp32}]
                [--unet_attention_slice [UNET_ATTENTION_SLICE]]
                [--num_train_epochs NUM_TRAIN_EPOCHS]
                [--max_train_steps MAX_TRAIN_STEPS]
                [--train_batch_size TRAIN_BATCH_SIZE]
                [--learning_rate LEARNING_RATE] --optimizer
                {adamw_bf16,ao-adamw8bit,ao-adamw4bit,ao-adamfp8,ao-adamwfp8,adamw_schedulefree,adamw_schedulefree+aggressive,adamw_schedulefree+no_kahan,optimi-stableadamw,optimi-adamw,optimi-lion,optimi-radam,optimi-ranger,optimi-adan,optimi-adam,optimi-sgd,soap,prodigy}
                [--optimizer_config OPTIMIZER_CONFIG]
                [--lr_scheduler {linear,sine,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                [--lr_warmup_steps LR_WARMUP_STEPS]
                [--checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT]
                [--gradient_checkpointing [GRADIENT_CHECKPOINTING]]
                [--train_text_encoder [TRAIN_TEXT_ENCODER]]
                [--text_encoder_lr TEXT_ENCODER_LR]
                [--lr_num_cycles LR_NUM_CYCLES] [--lr_power LR_POWER]
                [--use_soft_min_snr [USE_SOFT_MIN_SNR]] [--use_ema [USE_EMA]]
                [--ema_device {accelerator,cpu}]
                [--ema_cpu_only [EMA_CPU_ONLY]]
                [--ema_update_interval EMA_UPDATE_INTERVAL]
                [--ema_foreach_disable [EMA_FOREACH_DISABLE]]
                [--ema_decay EMA_DECAY] [--lora_rank LORA_RANK]
                [--lora_alpha LORA_ALPHA] [--lora_type {standard,lycoris}]
                [--lora_dropout LORA_DROPOUT]
                [--lora_init_type {default,gaussian,loftq,olora,pissa}]
                [--peft_lora_mode {standard,singlora}]
                [--peft_lora_target_modules PEFT_LORA_TARGET_MODULES]
                [--singlora_ramp_up_steps SINGLORA_RAMP_UP_STEPS]
                [--init_lora INIT_LORA] [--lycoris_config LYCORIS_CONFIG]
                [--init_lokr_norm INIT_LOKR_NORM]
                [--flux_lora_target {mmdit,context,context+ffs,all,all+ffs,ai-toolkit,tiny,nano,controlnet,all+ffs+embedder,all+ffs+embedder+controlnet}]
                [--use_dora [USE_DORA]]
                [--resolution_type {pixel,area,pixel_area}]
                --data_backend_config DATA_BACKEND_CONFIG
                [--caption_strategy {filename,textfile,instance_prompt,parquet}]
                [--conditioning_multidataset_sampling {combined,random}]
                [--instance_prompt INSTANCE_PROMPT]
                [--parquet_caption_column PARQUET_CAPTION_COLUMN]
                [--parquet_filename_column PARQUET_FILENAME_COLUMN]
                [--ignore_missing_files [IGNORE_MISSING_FILES]]
                [--vae_cache_scan_behaviour {recreate,sync}]
                [--vae_enable_slicing [VAE_ENABLE_SLICING]]
                [--vae_enable_tiling [VAE_ENABLE_TILING]]
                [--vae_enable_patch_conv [VAE_ENABLE_PATCH_CONV]]
                [--vae_batch_size VAE_BATCH_SIZE]
                [--caption_dropout_probability CAPTION_DROPOUT_PROBABILITY]
                [--tokenizer_max_length TOKENIZER_MAX_LENGTH]
                [--validation_step_interval VALIDATION_STEP_INTERVAL]
                [--validation_epoch_interval VALIDATION_EPOCH_INTERVAL]
                [--disable_benchmark [DISABLE_BENCHMARK]]
                [--validation_prompt VALIDATION_PROMPT]
                [--num_validation_images NUM_VALIDATION_IMAGES]
                [--num_eval_images NUM_EVAL_IMAGES]
                [--eval_steps_interval EVAL_STEPS_INTERVAL]
                [--eval_epoch_interval EVAL_EPOCH_INTERVAL]
                [--eval_timesteps EVAL_TIMESTEPS]
                [--eval_dataset_pooling [EVAL_DATASET_POOLING]]
                [--evaluation_type {none,clip}]
                [--pretrained_evaluation_model_name_or_path PRETRAINED_EVALUATION_MODEL_NAME_OR_PATH]
                [--validation_guidance VALIDATION_GUIDANCE]
                [--validation_num_inference_steps VALIDATION_NUM_INFERENCE_STEPS]
                [--validation_on_startup [VALIDATION_ON_STARTUP]]
                [--validation_using_datasets [VALIDATION_USING_DATASETS]]
                [--validation_torch_compile [VALIDATION_TORCH_COMPILE]]
                [--validation_guidance_real VALIDATION_GUIDANCE_REAL]
                [--validation_no_cfg_until_timestep VALIDATION_NO_CFG_UNTIL_TIMESTEP]
                [--validation_negative_prompt VALIDATION_NEGATIVE_PROMPT]
                [--validation_randomize [VALIDATION_RANDOMIZE]]
                [--validation_seed VALIDATION_SEED]
                [--validation_disable [VALIDATION_DISABLE]]
                [--validation_prompt_library [VALIDATION_PROMPT_LIBRARY]]
                [--user_prompt_library USER_PROMPT_LIBRARY]
                [--eval_dataset_id EVAL_DATASET_ID]
                [--validation_stitch_input_location {left,right}]
                [--validation_guidance_rescale VALIDATION_GUIDANCE_RESCALE]
                [--validation_disable_unconditional [VALIDATION_DISABLE_UNCONDITIONAL]]
                [--validation_guidance_skip_layers VALIDATION_GUIDANCE_SKIP_LAYERS]
                [--validation_guidance_skip_layers_start VALIDATION_GUIDANCE_SKIP_LAYERS_START]
                [--validation_guidance_skip_layers_stop VALIDATION_GUIDANCE_SKIP_LAYERS_STOP]
                [--validation_guidance_skip_scale VALIDATION_GUIDANCE_SKIP_SCALE]
                [--validation_lycoris_strength VALIDATION_LYCORIS_STRENGTH]
                [--validation_noise_scheduler {ddim,ddpm,euler,euler-a,unipc,dpm++,perflow}]
                [--validation_num_video_frames VALIDATION_NUM_VIDEO_FRAMES]
                [--validation_audio_only [VALIDATION_AUDIO_ONLY]]
                [--validation_resolution VALIDATION_RESOLUTION]
                [--validation_seed_source {cpu,gpu}]
                [--i_know_what_i_am_doing [I_KNOW_WHAT_I_AM_DOING]]
                [--flow_sigmoid_scale FLOW_SIGMOID_SCALE]
                [--flux_fast_schedule [FLUX_FAST_SCHEDULE]]
                [--flow_use_uniform_schedule [FLOW_USE_UNIFORM_SCHEDULE]]
                [--flow_use_beta_schedule [FLOW_USE_BETA_SCHEDULE]]
                [--flow_beta_schedule_alpha FLOW_BETA_SCHEDULE_ALPHA]
                [--flow_beta_schedule_beta FLOW_BETA_SCHEDULE_BETA]
                [--flow_schedule_shift FLOW_SCHEDULE_SHIFT]
                [--flow_schedule_auto_shift [FLOW_SCHEDULE_AUTO_SHIFT]]
                [--flux_guidance_mode {constant,random-range}]
                [--flux_attention_masked_training [FLUX_ATTENTION_MASKED_TRAINING]]
                [--flux_guidance_value FLUX_GUIDANCE_VALUE]
                [--flux_guidance_min FLUX_GUIDANCE_MIN]
                [--flux_guidance_max FLUX_GUIDANCE_MAX]
                [--t5_padding {zero,unmodified}]
                [--sd3_clip_uncond_behaviour {empty_string,zero}]
                [--sd3_t5_uncond_behaviour {empty_string,zero}]
                [--soft_min_snr_sigma_data SOFT_MIN_SNR_SIGMA_DATA]
                [--mixed_precision {no,fp16,bf16,fp8}]
                [--attention_mechanism {diffusers,xformers,flash-attn,flash-attn-2,flash-attn-3,flash-attn-3-varlen,flex,cudnn,native-efficient,native-flash,native-math,native-npu,native-xla,sla,sageattention,sageattention-int8-fp16-triton,sageattention-int8-fp16-cuda,sageattention-int8-fp8-cuda}]
                [--sageattention_usage {training,inference,training+inference}]
                [--disable_tf32 [DISABLE_TF32]]
                [--set_grads_to_none [SET_GRADS_TO_NONE]]
                [--noise_offset NOISE_OFFSET]
                [--noise_offset_probability NOISE_OFFSET_PROBABILITY]
                [--input_perturbation INPUT_PERTURBATION]
                [--input_perturbation_steps INPUT_PERTURBATION_STEPS]
                [--lr_end LR_END] [--lr_scale [LR_SCALE]]
                [--lr_scale_sqrt [LR_SCALE_SQRT]]
                [--ignore_final_epochs [IGNORE_FINAL_EPOCHS]]
                [--freeze_encoder_before FREEZE_ENCODER_BEFORE]
                [--freeze_encoder_after FREEZE_ENCODER_AFTER]
                [--freeze_encoder_strategy {before,between,after}]
                [--layer_freeze_strategy {none,bitfit}]
                [--fully_unload_text_encoder [FULLY_UNLOAD_TEXT_ENCODER]]
                [--save_text_encoder [SAVE_TEXT_ENCODER]]
                [--text_encoder_limit TEXT_ENCODER_LIMIT]
                [--prepend_instance_prompt [PREPEND_INSTANCE_PROMPT]]
                [--only_instance_prompt [ONLY_INSTANCE_PROMPT]]
                [--data_aesthetic_score DATA_AESTHETIC_SCORE]
                [--delete_unwanted_images [DELETE_UNWANTED_IMAGES]]
                [--delete_problematic_images [DELETE_PROBLEMATIC_IMAGES]]
                [--disable_bucket_pruning [DISABLE_BUCKET_PRUNING]]
                [--disable_segmented_timestep_sampling [DISABLE_SEGMENTED_TIMESTEP_SAMPLING]]
                [--preserve_data_backend_cache [PRESERVE_DATA_BACKEND_CACHE]]
                [--override_dataset_config [OVERRIDE_DATASET_CONFIG]]
                [--cache_dir CACHE_DIR] [--cache_dir_text CACHE_DIR_TEXT]
                [--cache_dir_vae CACHE_DIR_VAE]
                [--compress_disk_cache [COMPRESS_DISK_CACHE]]
                [--aspect_bucket_disable_rebuild [ASPECT_BUCKET_DISABLE_REBUILD]]
                [--keep_vae_loaded [KEEP_VAE_LOADED]]
                [--skip_file_discovery SKIP_FILE_DISCOVERY]
                [--data_backend_sampling {uniform,auto-weighting}]
                [--image_processing_batch_size IMAGE_PROCESSING_BATCH_SIZE]
                [--write_batch_size WRITE_BATCH_SIZE]
                [--read_batch_size READ_BATCH_SIZE]
                [--enable_multiprocessing [ENABLE_MULTIPROCESSING]]
                [--max_workers MAX_WORKERS]
                [--aws_max_pool_connections AWS_MAX_POOL_CONNECTIONS]
                [--torch_num_threads TORCH_NUM_THREADS]
                [--dataloader_prefetch [DATALOADER_PREFETCH]]
                [--dataloader_prefetch_qlen DATALOADER_PREFETCH_QLEN]
                [--aspect_bucket_worker_count ASPECT_BUCKET_WORKER_COUNT]
                [--aspect_bucket_alignment {8,16,24,32,64}]
                [--minimum_image_size MINIMUM_IMAGE_SIZE]
                [--maximum_image_size MAXIMUM_IMAGE_SIZE]
                [--target_downsample_size TARGET_DOWNSAMPLE_SIZE]
                [--max_upscale_threshold MAX_UPSCALE_THRESHOLD]
                [--metadata_update_interval METADATA_UPDATE_INTERVAL]
                [--debug_aspect_buckets [DEBUG_ASPECT_BUCKETS]]
                [--debug_dataset_loader [DEBUG_DATASET_LOADER]]
                [--print_filenames [PRINT_FILENAMES]]
                [--print_sampler_statistics [PRINT_SAMPLER_STATISTICS]]
                [--timestep_bias_strategy {earlier,later,range,none}]
                [--timestep_bias_begin TIMESTEP_BIAS_BEGIN]
                [--timestep_bias_end TIMESTEP_BIAS_END]
                [--timestep_bias_multiplier TIMESTEP_BIAS_MULTIPLIER]
                [--timestep_bias_portion TIMESTEP_BIAS_PORTION]
                [--training_scheduler_timestep_spacing {leading,linspace,trailing}]
                [--inference_scheduler_timestep_spacing {leading,linspace,trailing}]
                [--loss_type {l2,huber,smooth_l1}]
                [--huber_schedule {snr,exponential,constant}]
                [--huber_c HUBER_C] [--snr_gamma SNR_GAMMA]
                [--masked_loss_probability MASKED_LOSS_PROBABILITY]
                [--hidream_use_load_balancing_loss [HIDREAM_USE_LOAD_BALANCING_LOSS]]
                [--hidream_load_balancing_loss_weight HIDREAM_LOAD_BALANCING_LOSS_WEIGHT]
                [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2]
                [--optimizer_beta1 OPTIMIZER_BETA1]
                [--optimizer_beta2 OPTIMIZER_BETA2]
                [--optimizer_cpu_offload_method {none}]
                [--gradient_precision {unmodified,fp32}]
                [--adam_weight_decay ADAM_WEIGHT_DECAY]
                [--adam_epsilon ADAM_EPSILON] [--prodigy_steps PRODIGY_STEPS]
                [--max_grad_norm MAX_GRAD_NORM]
                [--grad_clip_method {value,norm}]
                [--optimizer_offload_gradients [OPTIMIZER_OFFLOAD_GRADIENTS]]
                [--fuse_optimizer [FUSE_OPTIMIZER]]
                [--optimizer_release_gradients [OPTIMIZER_RELEASE_GRADIENTS]]
                [--push_to_hub [PUSH_TO_HUB]]
                [--push_to_hub_background [PUSH_TO_HUB_BACKGROUND]]
                [--push_checkpoints_to_hub [PUSH_CHECKPOINTS_TO_HUB]]
                [--publishing_config PUBLISHING_CONFIG]
                [--hub_model_id HUB_MODEL_ID]
                [--model_card_private [MODEL_CARD_PRIVATE]]
                [--model_card_safe_for_work [MODEL_CARD_SAFE_FOR_WORK]]
                [--model_card_note MODEL_CARD_NOTE]
                [--modelspec_comment MODELSPEC_COMMENT]
                [--report_to {tensorboard,wandb,comet_ml,all,none}]
                [--checkpoint_step_interval CHECKPOINT_STEP_INTERVAL]
                [--checkpoint_epoch_interval CHECKPOINT_EPOCH_INTERVAL]
                [--checkpointing_rolling_steps CHECKPOINTING_ROLLING_STEPS]
                [--checkpointing_use_tempdir [CHECKPOINTING_USE_TEMPDIR]]
                [--checkpoints_rolling_total_limit CHECKPOINTS_ROLLING_TOTAL_LIMIT]
                [--tracker_run_name TRACKER_RUN_NAME]
                [--tracker_project_name TRACKER_PROJECT_NAME]
                [--tracker_image_layout {gallery,table}]
                [--enable_watermark [ENABLE_WATERMARK]]
                [--framerate FRAMERATE]
                [--seed_for_each_device [SEED_FOR_EACH_DEVICE]]
                [--snr_weight SNR_WEIGHT]
                [--rescale_betas_zero_snr [RESCALE_BETAS_ZERO_SNR]]
                [--webhook_config WEBHOOK_CONFIG]
                [--webhook_reporting_interval WEBHOOK_REPORTING_INTERVAL]
                [--distillation_method {lcm,dcm,dmd,perflow}]
                [--distillation_config DISTILLATION_CONFIG]
                [--ema_validation {none,ema_only,comparison}]
                [--local_rank LOCAL_RANK] [--ltx_train_mode {t2v,i2v}]
                [--ltx_i2v_prob LTX_I2V_PROB]
                [--ltx_partial_noise_fraction LTX_PARTIAL_NOISE_FRACTION]
                [--ltx_protect_first_frame [LTX_PROTECT_FIRST_FRAME]]
                [--offload_param_path OFFLOAD_PARAM_PATH]
                [--offset_noise [OFFSET_NOISE]]
                [--quantize_activations [QUANTIZE_ACTIVATIONS]]
                [--refiner_training [REFINER_TRAINING]]
                [--refiner_training_invert_schedule [REFINER_TRAINING_INVERT_SCHEDULE]]
                [--refiner_training_strength REFINER_TRAINING_STRENGTH]
                [--sdxl_refiner_uses_full_range [SDXL_REFINER_USES_FULL_RANGE]]
                [--sana_complex_human_instruction SANA_COMPLEX_HUMAN_INSTRUCTION]

The following SimpleTuner command-line options are available:

options:
  -h, --help            show this help message and exit
  --model_family {kolors,auraflow,omnigen,flux,deepfloyd,cosmos2image,sana,qwen_image,pixart_sigma,sdxl,sd1x,sd2x,wan,hidream,sd3,lumina2,ltxvideo,ace_step,heartmula}
                        The base model architecture family to train
  --model_flavour MODEL_FLAVOUR
                        Specific variant of the selected model family.
                        ACE-Step flavours `base`, `v15-turbo`, `v15-base`,
                        और `v15-sft` हैं। v1.5 flavours अब training और
                        built-in validation audio generation को support करते
                        हैं, और upstream repository के लिए
                        `--trust_remote_code` आवश्यक है।
  --controlnet [CONTROLNET]
                        Train ControlNet (full or LoRA) branches alongside the
                        primary network.
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                        Optional override of the model checkpoint. Leave blank
                        to use the default path for the selected model
                        flavour.
  --output_dir OUTPUT_DIR
                        Directory where model checkpoints and logs will be
                        saved
  --logging_dir LOGGING_DIR
                        Directory for TensorBoard logs
  --model_type {full,lora}
                        Choose between full model training or LoRA adapter
                        training
  --seed SEED           Seed used for deterministic training behaviour
  --resolution RESOLUTION
                        Resolution for training images
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        Select checkpoint to resume training from
  --prediction_type {epsilon,v_prediction,sample,flow_matching}
                        The parameterization type for the diffusion model
  --pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH
                        Path to pretrained VAE model
  --vae_dtype {default,fp32,fp16,bf16}
                        Precision for VAE encoding/decoding. Lower precision
                        saves memory.
  --vae_cache_ondemand [VAE_CACHE_ONDEMAND]
                        Process VAE latents during training instead of
                        precomputing them
  --vae_cache_disable [VAE_CACHE_DISABLE]
                        Implicitly enables on-demand caching and disables
                        writing embeddings to disk.
  --accelerator_cache_clear_interval ACCELERATOR_CACHE_CLEAR_INTERVAL
                        Clear the cache from VRAM every X steps to prevent
                        memory leaks
  --aspect_bucket_rounding {1,2,3,4,5,6,7,8,9}
                        Number of decimal places to round aspect ratios to for
                        bucket creation
  --base_model_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for loading the base model. Lower precision
                        saves memory.
  --text_encoder_1_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_2_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_3_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_4_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --gradient_checkpointing_interval GRADIENT_CHECKPOINTING_INTERVAL
                        Checkpoint every N transformer blocks
  --offload_during_startup [OFFLOAD_DURING_STARTUP]
                        Offload text encoders to CPU during VAE caching
  --quantize_via {cpu,accelerator,pipeline}
                        Where to perform model quantization
  --quantization_config QUANTIZATION_CONFIG
                        JSON or file path describing Diffusers quantization
                        config for pipeline quantization
  --fuse_qkv_projections [FUSE_QKV_PROJECTIONS]
                        Enables Flash Attention 3 when supported; otherwise
                        falls back to PyTorch SDPA.
  --control [CONTROL]   Enable channel-wise control style training
  --controlnet_custom_config CONTROLNET_CUSTOM_CONFIG
                        Custom configuration for ControlNet models
  --controlnet_model_name_or_path CONTROLNET_MODEL_NAME_OR_PATH
                        Path to ControlNet model weights to preload
  --tread_config TREAD_CONFIG
                        Configuration for TREAD training method
  --pretrained_transformer_model_name_or_path PRETRAINED_TRANSFORMER_MODEL_NAME_OR_PATH
                        Path to pretrained transformer model
  --pretrained_transformer_subfolder PRETRAINED_TRANSFORMER_SUBFOLDER
                        Subfolder containing transformer model weights
  --pretrained_unet_model_name_or_path PRETRAINED_UNET_MODEL_NAME_OR_PATH
                        Path to pretrained UNet model
  --pretrained_unet_subfolder PRETRAINED_UNET_SUBFOLDER
                        Subfolder containing UNet model weights
  --pretrained_t5_model_name_or_path PRETRAINED_T5_MODEL_NAME_OR_PATH
                        Path to pretrained T5 model
  --pretrained_gemma_model_name_or_path PRETRAINED_GEMMA_MODEL_NAME_OR_PATH
                        Path to pretrained Gemma model
  --revision REVISION   Git branch/tag/commit for model version
  --variant VARIANT     Model variant (e.g., fp16, bf16)
  --base_model_default_dtype {bf16,fp32}
                        Default precision for quantized base model weights
  --unet_attention_slice [UNET_ATTENTION_SLICE]
                        Enable attention slicing for SDXL UNet
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Number of times to iterate through the entire dataset
  --max_train_steps MAX_TRAIN_STEPS
                        Maximum number of training steps (0 = use epochs
                        instead)
  --train_batch_size TRAIN_BATCH_SIZE
                        Number of samples processed per forward/backward pass
                        (per device).
  --learning_rate LEARNING_RATE
                        Base learning rate for training
  --optimizer {adamw_bf16,ao-adamw8bit,ao-adamw4bit,ao-adamfp8,ao-adamwfp8,adamw_schedulefree,adamw_schedulefree+aggressive,adamw_schedulefree+no_kahan,optimi-stableadamw,optimi-adamw,optimi-lion,optimi-radam,optimi-ranger,optimi-adan,optimi-adam,optimi-sgd,soap,prodigy}
                        Optimization algorithm for training
  --optimizer_config OPTIMIZER_CONFIG
                        Comma-separated key=value pairs forwarded to the
                        selected optimizer
  --lr_scheduler {linear,sine,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
                        How learning rate changes during training
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of steps to accumulate gradients
  --lr_warmup_steps LR_WARMUP_STEPS
                        Number of steps to gradually increase LR from 0
  --checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT
                        Maximum number of checkpoints to keep on disk
  --gradient_checkpointing [GRADIENT_CHECKPOINTING]
                        Trade compute for memory during training
  --train_text_encoder [TRAIN_TEXT_ENCODER]
                        Also train the text encoder (CLIP) model
  --text_encoder_lr TEXT_ENCODER_LR
                        Separate learning rate for text encoder
  --lr_num_cycles LR_NUM_CYCLES
                        Number of cosine annealing cycles
  --lr_power LR_POWER   Power for polynomial decay scheduler
  --use_soft_min_snr [USE_SOFT_MIN_SNR]
                        Use soft clamping instead of hard clamping for Min-SNR
  --use_ema [USE_EMA]   Maintain an exponential moving average copy of the
                        model during training.
  --ema_device {accelerator,cpu}
                        Where to keep the EMA weights in-between updates.
  --ema_cpu_only [EMA_CPU_ONLY]
                        Keep EMA weights exclusively on CPU even when
                        ema_device would normally move them.
  --ema_update_interval EMA_UPDATE_INTERVAL
                        Update EMA weights every N optimizer steps
  --ema_foreach_disable [EMA_FOREACH_DISABLE]
                        Fallback to standard tensor ops instead of
                        torch.foreach updates.
  --ema_decay EMA_DECAY
                        Smoothing factor for EMA updates (closer to 1.0 =
                        slower drift).
  --lora_rank LORA_RANK
                        Dimension of LoRA update matrices
  --lora_alpha LORA_ALPHA
                        Scaling factor for LoRA updates
  --lora_type {standard,lycoris}
                        LoRA implementation type
  --lora_dropout LORA_DROPOUT
                        LoRA dropout randomly ignores neurons during training.
                        This can help prevent overfitting.
  --lora_init_type {default,gaussian,loftq,olora,pissa}
                        The initialization type for the LoRA model
  --peft_lora_mode {standard,singlora}
                        PEFT LoRA training mode
  --peft_lora_target_modules PEFT_LORA_TARGET_MODULES
                        JSON array (or path to a JSON file) listing PEFT
                        LoRA target module names. Overrides preset targets.
  --singlora_ramp_up_steps SINGLORA_RAMP_UP_STEPS
                        Number of ramp-up steps for SingLoRA
  --slider_lora_target [SLIDER_LORA_TARGET]
                        Route LoRA training to slider-friendly targets
                        (self-attn + conv/time embeddings). Only affects
                        standard PEFT LoRA.
  --init_lora INIT_LORA
                        Specify an existing LoRA or LyCORIS safetensors file
                        to initialize the adapter
  --lycoris_config LYCORIS_CONFIG
                        Path to LyCORIS configuration JSON file
  --init_lokr_norm INIT_LOKR_NORM
                        Perturbed normal initialization for LyCORIS LoKr
                        layers
  --flux_lora_target {mmdit,context,context+ffs,all,all+ffs,ai-toolkit,tiny,nano,controlnet,all+ffs+embedder,all+ffs+embedder+controlnet}
                        Which layers to train in Flux models
  --use_dora [USE_DORA]
                        Enable DoRA (Weight-Decomposed LoRA)
  --resolution_type {pixel,area,pixel_area}
                        How to interpret the resolution value
  --data_backend_config DATA_BACKEND_CONFIG
                        Select a saved dataset configuration (managed in
                        Datasets & Environments tabs)
  --caption_strategy {filename,textfile,instance_prompt,parquet}
                        How to load captions for images
  --conditioning_multidataset_sampling {combined,random}
                        How to sample from multiple conditioning datasets
  --instance_prompt INSTANCE_PROMPT
                        Instance prompt for training
  --parquet_caption_column PARQUET_CAPTION_COLUMN
                        Column name containing captions in parquet files
  --parquet_filename_column PARQUET_FILENAME_COLUMN
                        Column name containing image paths in parquet files
  --ignore_missing_files [IGNORE_MISSING_FILES]
                        Continue training even if some files are missing
  --vae_cache_scan_behaviour {recreate,sync}
                        How to scan VAE cache for missing files
  --vae_enable_slicing [VAE_ENABLE_SLICING]
                        Enable VAE attention slicing for memory efficiency
  --vae_enable_tiling [VAE_ENABLE_TILING]
                        Enable VAE tiling for large images
  --vae_enable_patch_conv [VAE_ENABLE_PATCH_CONV]
                        Enable patch-based 3D conv for HunyuanVideo VAE to
                        reduce peak VRAM (slight slowdown)
  --vae_batch_size VAE_BATCH_SIZE
                        Batch size for VAE encoding during caching
  --caption_dropout_probability CAPTION_DROPOUT_PROBABILITY
                        Caption dropout will randomly drop captions and, for
                        SDXL, size conditioning inputs based on this
                        probability
  --tokenizer_max_length TOKENIZER_MAX_LENGTH
                        Override the tokenizer sequence length (advanced).
  --validation_step_interval VALIDATION_STEP_INTERVAL
                        Run validation every N training steps (deprecated alias: --validation_steps)
  --validation_epoch_interval VALIDATION_EPOCH_INTERVAL
                        Run validation every N training epochs
  --disable_benchmark [DISABLE_BENCHMARK]
                        Skip generating baseline comparison images before
                        training starts
  --validation_prompt VALIDATION_PROMPT
                        Prompt to use for validation images
  --num_validation_images NUM_VALIDATION_IMAGES
                        Number of images to generate per validation
  --num_eval_images NUM_EVAL_IMAGES
                        Number of images to generate for evaluation metrics
  --eval_steps_interval EVAL_STEPS_INTERVAL
                        Run evaluation every N training steps
  --eval_epoch_interval EVAL_EPOCH_INTERVAL
                        Run evaluation every N training epochs (decimals run
                        multiple times per epoch)
  --eval_timesteps EVAL_TIMESTEPS
                        Number of timesteps for evaluation
  --eval_dataset_pooling [EVAL_DATASET_POOLING]
                        Combine evaluation metrics from all datasets into a
                        single chart
  --evaluation_type {none,clip}
                        Type of evaluation metrics to compute
  --pretrained_evaluation_model_name_or_path PRETRAINED_EVALUATION_MODEL_NAME_OR_PATH
                        Path to pretrained model for evaluation metrics
  --validation_guidance VALIDATION_GUIDANCE
                        CFG guidance scale for validation images
  --validation_num_inference_steps VALIDATION_NUM_INFERENCE_STEPS
                        Number of diffusion steps for validation renders
  --validation_on_startup [VALIDATION_ON_STARTUP]
                        Run validation on the base model before training
                        starts
  --validation_using_datasets [VALIDATION_USING_DATASETS]
                        Use random images from training datasets for
                        validation
  --validation_torch_compile [VALIDATION_TORCH_COMPILE]
                        Use torch.compile() on validation pipeline for speed
  --validation_guidance_real VALIDATION_GUIDANCE_REAL
                        CFG value for distilled models (e.g., FLUX schnell)
  --validation_no_cfg_until_timestep VALIDATION_NO_CFG_UNTIL_TIMESTEP
                        Skip CFG for initial timesteps (Flux only)
  --validation_negative_prompt VALIDATION_NEGATIVE_PROMPT
                        Negative prompt for validation images
  --validation_randomize [VALIDATION_RANDOMIZE]
                        Use random seeds for each validation
  --validation_seed VALIDATION_SEED
                        Fixed seed for reproducible validation images
  --validation_disable [VALIDATION_DISABLE]
                        Completely disable validation image generation
  --validation_prompt_library [VALIDATION_PROMPT_LIBRARY]
                        Use SimpleTuner's built-in prompt library
  --user_prompt_library USER_PROMPT_LIBRARY
                        Path to custom JSON prompt library
  --eval_dataset_id EVAL_DATASET_ID
                        Specific dataset to use for evaluation metrics
  --validation_stitch_input_location {left,right}
                        Where to place input image in img2img validations
  --validation_guidance_rescale VALIDATION_GUIDANCE_RESCALE
                        CFG rescale value for validation
  --validation_disable_unconditional [VALIDATION_DISABLE_UNCONDITIONAL]
                        Disable unconditional image generation during
                        validation
  --validation_guidance_skip_layers VALIDATION_GUIDANCE_SKIP_LAYERS
                        JSON list of transformer layers to skip during
                        classifier-free guidance
  --validation_guidance_skip_layers_start VALIDATION_GUIDANCE_SKIP_LAYERS_START
                        Starting layer index to skip guidance
  --validation_guidance_skip_layers_stop VALIDATION_GUIDANCE_SKIP_LAYERS_STOP
                        Ending layer index to skip guidance
  --validation_guidance_skip_scale VALIDATION_GUIDANCE_SKIP_SCALE
                        Scale guidance strength when applying layer skipping
  --validation_lycoris_strength VALIDATION_LYCORIS_STRENGTH
                        Strength multiplier for LyCORIS validation
  --validation_noise_scheduler {ddim,ddpm,euler,euler-a,unipc,dpm++,perflow}
                        Noise scheduler for validation
  --validation_num_video_frames VALIDATION_NUM_VIDEO_FRAMES
                        Number of frames for video validation
  --validation_audio_only [VALIDATION_AUDIO_ONLY]
                        Disable video generation during validation and emit
                        audio only
  --validation_resolution VALIDATION_RESOLUTION
                        Override resolution for validation images (pixels or
                        megapixels)
  --validation_seed_source {cpu,gpu}
                        Source device used to generate validation seeds
  --i_know_what_i_am_doing [I_KNOW_WHAT_I_AM_DOING]
                        Unlock experimental overrides and bypass built-in
                        safety limits.
  --flow_sigmoid_scale FLOW_SIGMOID_SCALE
                        Scale factor for sigmoid timestep sampling for flow-
                        matching models.
  --flux_fast_schedule [FLUX_FAST_SCHEDULE]
                        Use experimental fast schedule for Flux training
  --flow_use_uniform_schedule [FLOW_USE_UNIFORM_SCHEDULE]
                        Use uniform schedule instead of sigmoid for flow-
                        matching
  --flow_use_beta_schedule [FLOW_USE_BETA_SCHEDULE]
                        Use beta schedule instead of sigmoid for flow-matching
  --flow_beta_schedule_alpha FLOW_BETA_SCHEDULE_ALPHA
                        Alpha value for beta schedule (default: 2.0)
  --flow_beta_schedule_beta FLOW_BETA_SCHEDULE_BETA
                        Beta value for beta schedule (default: 2.0)
  --flow_schedule_shift FLOW_SCHEDULE_SHIFT
                        Shift the noise schedule for flow-matching models
  --flow_schedule_auto_shift [FLOW_SCHEDULE_AUTO_SHIFT]
                        Auto-adjust schedule shift based on image resolution
  --flux_guidance_mode {constant,random-range}
                        Guidance mode for Flux training
  --flux_attention_masked_training [FLUX_ATTENTION_MASKED_TRAINING]
                        Enable attention masked training for Flux models
  --flux_guidance_value FLUX_GUIDANCE_VALUE
                        Guidance value for constant mode
  --flux_guidance_min FLUX_GUIDANCE_MIN
                        Minimum guidance value for random-range mode
  --flux_guidance_max FLUX_GUIDANCE_MAX
                        Maximum guidance value for random-range mode
  --t5_padding {zero,unmodified}
                        Padding behavior for T5 text encoder
  --sd3_clip_uncond_behaviour {empty_string,zero}
                        How SD3 handles unconditional prompts
  --sd3_t5_uncond_behaviour {empty_string,zero}
                        How SD3 T5 handles unconditional prompts
  --soft_min_snr_sigma_data SOFT_MIN_SNR_SIGMA_DATA
                        Sigma data for soft min SNR weighting
  --mixed_precision {no,fp16,bf16,fp8}
                        Precision for training computations
  --attention_mechanism {diffusers,xformers,flash-attn,flash-attn-2,flash-attn-3,flash-attn-3-varlen,flex,cudnn,native-efficient,native-flash,native-math,native-npu,native-xla,sla,sageattention,sageattention-int8-fp16-triton,sageattention-int8-fp16-cuda,sageattention-int8-fp8-cuda}
                        Attention computation backend
  --sageattention_usage {training,inference,training+inference}
                        When to use SageAttention
  --disable_tf32 [DISABLE_TF32]
                        Force IEEE FP32 precision (disables TF32) using
                        PyTorch's fp32_precision controls when available
  --set_grads_to_none [SET_GRADS_TO_NONE]
                        Set gradients to None instead of zero
  --noise_offset NOISE_OFFSET
                        Add noise offset to training
  --noise_offset_probability NOISE_OFFSET_PROBABILITY
                        Probability of applying noise offset
  --input_perturbation INPUT_PERTURBATION
                        Add additional noise only to the inputs fed to the
                        model during training
  --input_perturbation_steps INPUT_PERTURBATION_STEPS
                        Only apply input perturbation over the first N steps
                        with linear decay
  --lr_end LR_END       A polynomial learning rate will end up at this value
                        after the specified number of warmup steps
  --lr_scale [LR_SCALE]
                        Scale the learning rate by the number of GPUs,
                        gradient accumulation steps, and batch size
  --lr_scale_sqrt [LR_SCALE_SQRT]
                        If using --lr_scale, use the square root of (number of
                        GPUs * gradient accumulation steps * batch size)
  --ignore_final_epochs [IGNORE_FINAL_EPOCHS]
                        When provided, the max epoch counter will not
                        determine the end of the training run
  --freeze_encoder_before FREEZE_ENCODER_BEFORE
                        When using 'before' strategy, we will freeze layers
                        earlier than this
  --freeze_encoder_after FREEZE_ENCODER_AFTER
                        When using 'after' strategy, we will freeze layers
                        later than this
  --freeze_encoder_strategy {before,between,after}
                        When freezing the text encoder, we can use the
                        'before', 'between', or 'after' strategy
  --layer_freeze_strategy {none,bitfit}
                        When freezing parameters, we can use the 'none' or
                        'bitfit' strategy
  --fully_unload_text_encoder [FULLY_UNLOAD_TEXT_ENCODER]
                        If set, will fully unload the text_encoder from memory
                        when not in use
  --save_text_encoder [SAVE_TEXT_ENCODER]
                        If set, will save the text encoder after training
  --text_encoder_limit TEXT_ENCODER_LIMIT
                        When training the text encoder, we want to limit how
                        long it trains for to avoid catastrophic loss
  --prepend_instance_prompt [PREPEND_INSTANCE_PROMPT]
                        When determining the captions from the filename,
                        prepend the instance prompt as an enforced keyword
  --only_instance_prompt [ONLY_INSTANCE_PROMPT]
                        Use the instance prompt instead of the caption from
                        filename
  --data_aesthetic_score DATA_AESTHETIC_SCORE
                        Since currently we do not calculate aesthetic scores
                        for data, we will statically set it to one value. This
                        is only used by the SDXL Refiner
  --delete_unwanted_images [DELETE_UNWANTED_IMAGES]
                        If set, will delete images that are not of a minimum
                        size to save on disk space for large training runs
  --delete_problematic_images [DELETE_PROBLEMATIC_IMAGES]
                        If set, any images that error out during load will be
                        removed from the underlying storage medium
  --disable_bucket_pruning [DISABLE_BUCKET_PRUNING]
                        When training on very small datasets, you might not
                        care that the batch sizes will outpace your image
                        count. Setting this option will prevent SimpleTuner
                        from deleting your bucket lists that do not meet the
                        minimum image count requirements. Use at your own
                        risk, it may end up throwing off your statistics or
                        epoch tracking
  --disable_segmented_timestep_sampling [DISABLE_SEGMENTED_TIMESTEP_SAMPLING]
                        By default, the timestep schedule is divided into
                        roughly `train_batch_size` number of segments, and
                        then each of those are sampled from separately. This
                        improves the selection distribution, but may not be
                        desired in certain training scenarios, eg. when
                        limiting the timestep selection range
  --preserve_data_backend_cache [PRESERVE_DATA_BACKEND_CACHE]
                        For very large cloud storage buckets that will never
                        change, enabling this option will prevent the trainer
                        from scanning it at startup, by preserving the cache
                        files that we generate. Be careful when using this,
                        as, switching datasets can result in the preserved
                        cache being used, which would be problematic.
                        Currently, cache is not stored in the dataset itself
                        but rather, locally. This may change in a future
                        release
  --override_dataset_config [OVERRIDE_DATASET_CONFIG]
                        When provided, the dataset's config will not be
                        checked against the live backend config
  --cache_dir CACHE_DIR
                        The directory where the downloaded models and datasets
                        will be stored
  --cache_dir_text CACHE_DIR_TEXT
                        This is the path to a local directory that will
                        contain your text embed cache
  --cache_dir_vae CACHE_DIR_VAE
                        This is the path to a local directory that will
                        contain your VAE outputs
  --compress_disk_cache [COMPRESS_DISK_CACHE]
                        If set, will gzip-compress the disk cache for Pytorch
                        files. This will save substantial disk space, but may
                        slow down the training process
  --aspect_bucket_disable_rebuild [ASPECT_BUCKET_DISABLE_REBUILD]
                        When using a randomised aspect bucket list, the VAE
                        and aspect cache are rebuilt on each epoch. With a
                        large and diverse enough dataset, rebuilding the
                        aspect list may take a long time, and this may be
                        undesirable. This option will not override
                        vae_cache_clear_each_epoch. If both options are
                        provided, only the VAE cache will be rebuilt
  --keep_vae_loaded [KEEP_VAE_LOADED]
                        If set, will keep the VAE loaded in memory. This can
                        reduce disk churn, but consumes VRAM during the
                        forward pass
  --skip_file_discovery SKIP_FILE_DISCOVERY
                        Comma-separated values of which stages to skip
                        discovery for. Skipping any stage will speed up
                        resumption, but will increase the risk of errors, as
                        missing images or incorrectly bucketed images may not
                        be caught. Valid options: aspect, vae, text, metadata
  --data_backend_sampling {uniform,auto-weighting}
                        When using multiple data backends, the sampling
                        weighting can be set to 'uniform' or 'auto-weighting'
  --image_processing_batch_size IMAGE_PROCESSING_BATCH_SIZE
                        When resizing and cropping images, we do it in
                        parallel using processes or threads. This defines how
                        many images will be read into the queue before they
                        are processed
  --write_batch_size WRITE_BATCH_SIZE
                        When using certain storage backends, it is better to
                        batch smaller writes rather than continuous
                        dispatching. In SimpleTuner, write batching is
                        currently applied during VAE caching, when many small
                        objects are written. This mostly applies to S3, but
                        some shared server filesystems may benefit as well.
                        Default: 64
  --read_batch_size READ_BATCH_SIZE
                        Used by the VAE cache to prefetch image data. This is
                        the number of images to read ahead
  --enable_multiprocessing [ENABLE_MULTIPROCESSING]
                        If set, will use processes instead of threads during
                        metadata caching operations
  --max_workers MAX_WORKERS
                        How many active threads or processes to run during VAE
                        caching
  --aws_max_pool_connections AWS_MAX_POOL_CONNECTIONS
                        When using AWS backends, the maximum number of
                        connections to keep open to the S3 bucket at a single
                        time
  --torch_num_threads TORCH_NUM_THREADS
                        The number of threads to use for PyTorch operations.
                        This is not the same as the number of workers
  --dataloader_prefetch [DATALOADER_PREFETCH]
                        When provided, the dataloader will read-ahead and
                        attempt to retrieve latents, text embeds, and other
                        metadata ahead of the time when the batch is required,
                        so that it can be immediately available
  --dataloader_prefetch_qlen DATALOADER_PREFETCH_QLEN
                        Set the number of prefetched batches
  --aspect_bucket_worker_count ASPECT_BUCKET_WORKER_COUNT
                        The number of workers to use for aspect bucketing.
                        This is a CPU-bound task, so the number of workers
                        should be set to the number of CPU threads available.
                        If you use an I/O bound backend, an even higher value
                        may make sense. Default: 12
  --aspect_bucket_alignment {8,16,24,32,64}
                        When training diffusion models, the image sizes
                        generally must align to a 64 pixel interval
  --minimum_image_size MINIMUM_IMAGE_SIZE
                        The minimum resolution for both sides of input images
  --maximum_image_size MAXIMUM_IMAGE_SIZE
                        When cropping images that are excessively large, the
                        entire scene context may be lost, eg. the crop might
                        just end up being a portion of the background. To
                        avoid this, a maximum image size may be provided,
                        which will result in very-large images being
                        downsampled before cropping them. This value uses
                        --resolution_type to determine whether it is a pixel
                        edge or megapixel value
  --target_downsample_size TARGET_DOWNSAMPLE_SIZE
                        When using --maximum_image_size, very-large images
                        exceeding that value will be downsampled to this
                        target size before cropping
  --max_upscale_threshold MAX_UPSCALE_THRESHOLD
                        Limit upscaling of small images to prevent quality
                        degradation (opt-in). When set, filters out aspect
                        buckets requiring upscaling beyond this threshold.
                        For example, 0.2 allows up to 20% upscaling. Default
                        (None) allows unlimited upscaling. Must be between 0
                        and 1.
  --metadata_update_interval METADATA_UPDATE_INTERVAL
                        When generating the aspect bucket indicies, we want to
                        save it every X seconds
  --debug_aspect_buckets [DEBUG_ASPECT_BUCKETS]
                        If set, will print excessive debugging for aspect
                        bucket operations
  --debug_dataset_loader [DEBUG_DATASET_LOADER]
                        If set, will print excessive debugging for data loader
                        operations
  --print_filenames [PRINT_FILENAMES]
                        If any image files are stopping the process eg. due to
                        corruption or truncation, this will help identify
                        which is at fault
  --print_sampler_statistics [PRINT_SAMPLER_STATISTICS]
                        If provided, will print statistics about the dataset
                        sampler. This is useful for debugging
  --timestep_bias_strategy {earlier,later,range,none}
                        Strategy for biasing timestep sampling
  --timestep_bias_begin TIMESTEP_BIAS_BEGIN
                        Beginning of timestep bias range
  --timestep_bias_end TIMESTEP_BIAS_END
                        End of timestep bias range
  --timestep_bias_multiplier TIMESTEP_BIAS_MULTIPLIER
                        Multiplier for timestep bias probability
  --timestep_bias_portion TIMESTEP_BIAS_PORTION
                        Portion of training steps to apply timestep bias
  --training_scheduler_timestep_spacing {leading,linspace,trailing}
                        Timestep spacing for training scheduler
  --inference_scheduler_timestep_spacing {leading,linspace,trailing}
                        Timestep spacing for inference scheduler
  --loss_type {l2,huber,smooth_l1}
                        Loss function for training
  --huber_schedule {snr,exponential,constant}
                        Schedule for Huber loss transition threshold
  --huber_c HUBER_C     Transition point between L2 and L1 regions for Huber
                        loss
  --snr_gamma SNR_GAMMA
                        SNR weighting gamma value (0 = disabled)
  --masked_loss_probability MASKED_LOSS_PROBABILITY
                        Probability of applying masked loss weighting per
                        batch
  --hidream_use_load_balancing_loss [HIDREAM_USE_LOAD_BALANCING_LOSS]
                        Apply experimental load balancing loss when training
                        HiDream models.
  --hidream_load_balancing_loss_weight HIDREAM_LOAD_BALANCING_LOSS_WEIGHT
                        Strength multiplier for HiDream load balancing loss.
  --adam_beta1 ADAM_BETA1
                        First moment decay rate for Adam optimizers
  --adam_beta2 ADAM_BETA2
                        Second moment decay rate for Adam optimizers
  --optimizer_beta1 OPTIMIZER_BETA1
                        First moment decay rate for optimizers
  --optimizer_beta2 OPTIMIZER_BETA2
                        Second moment decay rate for optimizers
  --optimizer_cpu_offload_method {none}
                        Method for CPU offloading optimizer states
  --gradient_precision {unmodified,fp32}
                        Precision for gradient computation
  --adam_weight_decay ADAM_WEIGHT_DECAY
                        L2 regularisation strength for Adam-family optimizers.
  --adam_epsilon ADAM_EPSILON
                        Small constant added for numerical stability.
  --prodigy_steps PRODIGY_STEPS
                        Number of steps Prodigy should spend adapting its
                        learning rate.
  --max_grad_norm MAX_GRAD_NORM
                        Gradient clipping threshold to prevent exploding
                        gradients.
  --grad_clip_method {value,norm}
                        Strategy for applying max_grad_norm during clipping.
  --optimizer_offload_gradients [OPTIMIZER_OFFLOAD_GRADIENTS]
                        Move optimizer gradients to CPU to save GPU memory.
  --fuse_optimizer [FUSE_OPTIMIZER]
                        Enable fused kernels when offloading to reduce memory
                        overhead.
  --optimizer_release_gradients [OPTIMIZER_RELEASE_GRADIENTS]
                        Free gradient tensors immediately after optimizer step
                        when using Optimi optimizers.
  --push_to_hub [PUSH_TO_HUB]
                        Automatically upload the trained model to your Hugging
                        Face Hub repository.
  --push_to_hub_background [PUSH_TO_HUB_BACKGROUND]
                        Run Hub uploads in a background worker so training is
                        not blocked while pushing.
  --push_checkpoints_to_hub [PUSH_CHECKPOINTS_TO_HUB]
                        Upload intermediate checkpoints to the same Hugging
                        Face repository during training.
  --publishing_config PUBLISHING_CONFIG
                        Optional JSON/file path describing additional
                        publishing targets (S3/Backblaze B2/Azure Blob/Dropbox).
  --hub_model_id HUB_MODEL_ID
                        If left blank, SimpleTuner derives a name from the
                        project settings when pushing to Hub.
  --model_card_private [MODEL_CARD_PRIVATE]
                        Create the Hugging Face repository as private instead
                        of public.
  --model_card_safe_for_work [MODEL_CARD_SAFE_FOR_WORK]
                        Remove the default NSFW warning from the generated
                        model card on Hugging Face Hub.
  --model_card_note MODEL_CARD_NOTE
                        Optional note that appears at the top of the generated
                        model card.
  --modelspec_comment MODELSPEC_COMMENT
                        Text embedded in safetensors file metadata as
                        modelspec.comment, visible in external model viewers.
  --report_to {tensorboard,wandb,comet_ml,all,none}
                        Where to log training metrics
  --checkpoint_step_interval CHECKPOINT_STEP_INTERVAL
                        Save model checkpoint every N steps (deprecated alias: --checkpointing_steps)
  --checkpoint_epoch_interval CHECKPOINT_EPOCH_INTERVAL
                        Save model checkpoint every N epochs
  --checkpointing_rolling_steps CHECKPOINTING_ROLLING_STEPS
                        Rolling checkpoint window size for continuous
                        checkpointing
  --checkpointing_use_tempdir [CHECKPOINTING_USE_TEMPDIR]
                        Use temporary directory for checkpoint files before
                        final save
  --checkpoints_rolling_total_limit CHECKPOINTS_ROLLING_TOTAL_LIMIT
                        Maximum number of rolling checkpoints to keep
  --tracker_run_name TRACKER_RUN_NAME
                        Name for this training run in tracking platforms
  --tracker_project_name TRACKER_PROJECT_NAME
                        Project name in tracking platforms
  --tracker_image_layout {gallery,table}
                        How validation images are displayed in trackers
  --enable_watermark [ENABLE_WATERMARK]
                        Add invisible watermark to generated images
  --framerate FRAMERATE
                        Framerate for video model training
  --seed_for_each_device [SEED_FOR_EACH_DEVICE]
                        Use a unique deterministic seed per GPU instead of
                        sharing one seed across devices.
  --snr_weight SNR_WEIGHT
                        Weight factor for SNR-based loss scaling
  --rescale_betas_zero_snr [RESCALE_BETAS_ZERO_SNR]
                        Rescale betas for zero terminal SNR
  --webhook_config WEBHOOK_CONFIG
                        Path to webhook configuration file
  --webhook_reporting_interval WEBHOOK_REPORTING_INTERVAL
                        Interval for webhook reports (seconds)
  --distillation_method {lcm,dcm,dmd,perflow}
                        Method for model distillation
  --distillation_config DISTILLATION_CONFIG
                        Path to distillation configuration file
  --ema_validation {none,ema_only,comparison}
                        Control how EMA weights are used during validation
                        runs.
  --local_rank LOCAL_RANK
                        Local rank for distributed training
  --ltx_train_mode {t2v,i2v}
                        Training mode for LTX models
  --ltx_i2v_prob LTX_I2V_PROB
                        Probability of using image-to-video training for LTX
  --ltx_partial_noise_fraction LTX_PARTIAL_NOISE_FRACTION
                        Fraction of noise to add for LTX partial training
  --ltx_protect_first_frame [LTX_PROTECT_FIRST_FRAME]
                        Protect the first frame from noise in LTX training
  --offload_param_path OFFLOAD_PARAM_PATH
                        Path to offloaded parameter files
  --offset_noise [OFFSET_NOISE]
                        Enable offset-noise training
  --quantize_activations [QUANTIZE_ACTIVATIONS]
                        Quantize model activations during training
  --refiner_training [REFINER_TRAINING]
                        Enable refiner model training mode
  --refiner_training_invert_schedule [REFINER_TRAINING_INVERT_SCHEDULE]
                        Invert the noise schedule for refiner training
  --refiner_training_strength REFINER_TRAINING_STRENGTH
                        Strength of refiner training
  --sdxl_refiner_uses_full_range [SDXL_REFINER_USES_FULL_RANGE]
                        Use full timestep range for SDXL refiner
  --sana_complex_human_instruction SANA_COMPLEX_HUMAN_INSTRUCTION
                        Complex human instruction for Sana model training
```
