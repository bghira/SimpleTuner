# Z-Image [base / turbo] क्विकस्टार्ट

इस उदाहरण में, हम Z‑Image Turbo LoRA प्रशिक्षण करेंगे। Z‑Image एक 6B flow‑matching transformer है (Flux के लगभग आधे आकार का) जिसमें base और turbo flavours हैं। Turbo को assistant adapter चाहिए; SimpleTuner इसे स्वतः लोड कर सकता है।

## हार्डवेयर आवश्यकताएँ

Z‑Image को Flux से कम मेमोरी चाहिए, लेकिन मजबूत GPUs का लाभ मिलता है। जब आप rank‑16 LoRA के हर घटक (MLP, projections, transformer blocks) को ट्रेन करते हैं, तो आमतौर पर यह उपयोग करता है:

- बेस मॉडल quantise न करने पर ~32‑40G VRAM
- int8 + bf16 base/LoRA weights पर quantise करने पर ~16‑24G VRAM
- NF4 + bf16 base/LoRA weights पर quantise करने पर ~10–12G VRAM

इसके अलावा, Ramtorch और group offload से VRAM उपयोग और कम किया जा सकता है। Multi‑GPU उपयोगकर्ताओं के लिए, FSDP2 कई छोटे GPUs पर चलाने देगा।

आपको चाहिए:

- **पूर्ण न्यूनतम** एक **3080 10G** (आक्रामक quantisation/offload के साथ)
- **यथार्थवादी न्यूनतम** एक 3090/4090 या V100/A6000
- **आदर्श रूप से** कई 4090, A6000, L40S, या बेहतर

Apple GPUs पर प्रशिक्षण अनुशंसित नहीं है।

### मेमोरी ऑफ़लोडिंग (वैकल्पिक)

Grouped module offloading transformer weights bottleneck होने पर VRAM दबाव काफी कम करता है। इसे `TRAINER_EXTRA_ARGS` (या WebUI Hardware page) में निम्न flags जोड़कर सक्षम करें:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Streams केवल CUDA पर प्रभावी हैं; SimpleTuner ROCm, MPS और CPU पर इन्हें स्वतः बंद कर देता है।
- इसे अन्य CPU offload रणनीतियों के साथ **न** मिलाएँ।
- Group offload Quanto quantisation के साथ संगत नहीं है।
- Disk offload के लिए तेज़ लोकल SSD/NVMe लक्ष्य चुनें।

## पूर्वापेक्षाएँ

सुनिश्चित करें कि Python इंस्टॉल है; SimpleTuner 3.10 से 3.12 के साथ अच्छा काम करता है।

आप इसे चलाकर जांच सकते हैं:

```bash
python --version
```

यदि आपके Ubuntu पर Python 3.12 इंस्टॉल नहीं है, तो आप यह प्रयास कर सकते हैं:

```bash
apt -y install python3.13 python3.13-venv
```

### Container image dependencies

Vast, RunPod, और TensorDock (आदि) के लिए, CUDA 12.x इमेज पर CUDA extensions कम्पाइल करने हेतु यह काम करेगा:

```bash
apt -y install nvidia-cuda-toolkit-12-8
```

## इंस्टॉलेशन

pip के जरिए SimpleTuner इंस्टॉल करें:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
```

मैनुअल इंस्टॉलेशन या डेवलपमेंट सेटअप के लिए, [installation documentation](../INSTALL.md) देखें।

### AMD ROCm follow‑up steps

AMD MI300X को उपयोगी बनाने के लिए निम्न चलाना आवश्यक है:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

## वातावरण सेटअप

### Web interface method

SimpleTuner WebUI सेटअप को सरल बनाता है। सर्वर चलाने के लिए:

```bash
simpletuner server
```

यह डिफ़ॉल्ट रूप से पोर्ट 8001 पर वेब सर्वर बनाता है, जिसे आप http://localhost:8001 पर खोल सकते हैं।

### Manual / command‑line method

command‑line टूल्स से SimpleTuner चलाने के लिए, आपको एक configuration फ़ाइल, dataset और model directories, तथा एक dataloader configuration फ़ाइल सेट करनी होगी।

#### Configuration file

एक प्रयोगात्मक स्क्रिप्ट, `configure.py`, इंटरैक्टिव step‑by‑step कॉन्फ़िगरेशन के जरिए इस सेक्शन को पूरी तरह स्किप करने में मदद कर सकती है। इसमें कुछ सुरक्षा फीचर्स हैं जो सामान्य pitfalls से बचाते हैं।

**नोट:** यह आपके dataloader को कॉन्फ़िगर नहीं करता। आपको उसे बाद में मैन्युअली करना होगा।

इसे चलाने के लिए:

```bash
simpletuner configure
```

> ⚠️ जिन देशों में Hugging Face Hub आसानी से उपलब्ध नहीं है, वहाँ `HF_ENDPOINT=https://hf-mirror.com` को अपने `~/.bashrc` या `~/.zshrc` में जोड़ें, यह आपके सिस्टम के `$SHELL` पर निर्भर करता है।

यदि आप मैन्युअल कॉन्फ़िगर करना पसंद करते हैं:

`config/config.json.example` को `config/config.json` में कॉपी करें:

```bash
cp config/config.json.example config/config.json
```

फिर, आपको संभवतः निम्न वेरिएबल्स बदलने होंगे:

- `model_type` - इसे `lora` पर सेट करें।
- `model_family` - इसे `z-image` पर सेट करें।
- `model_flavour` - `turbo` पर सेट करें (या v2 assistant adapter के लिए `turbo-ostris-v2`); base flavour वर्तमान में उपलब्ध नहीं checkpoint पर इंगित करता है।
- `pretrained_model_name_or_path` - इसे `TONGYI-MAI/Z-Image-Turbo` पर सेट करें।
- `output_dir` - इसे उस डायरेक्टरी पर सेट करें जहाँ आप अपने checkpoints और validation images रखना चाहते हैं। यहाँ full path उपयोग करने की सलाह है।
- `train_batch_size` - 1 पर रखें, खासकर यदि आपका डेटासेट छोटा है।
- `validation_resolution` - Z‑Image 1024px है; `1024x1024` या multi‑aspect buckets उपयोग करें: `1024x1024,1280x768,2048x2048`.
- `validation_guidance` - Z‑Image Turbo के लिए low guidance (0–1) सामान्य है, लेकिन base flavour को 4–6 की रेंज चाहिए।
- `validation_num_inference_steps` - Turbo को केवल 8 चाहिए, जबकि Base लगभग 50‑100 पर भी ठीक चल सकता है।
- यदि आप LoRA का आकार काफी कम करना चाहते हैं, तो `--lora_rank=4` उपयोग करें। इससे VRAM उपयोग कम हो सकता है।
- Turbo के लिए assistant adapter दें (नीचे देखें) या उसे स्पष्ट रूप से बंद करें।

- `gradient_accumulation_steps` - runtime को linearly बढ़ाता है; VRAM relief चाहिए तो उपयोग करें।
- `optimizer` - शुरुआती उपयोगकर्ताओं को adamw_bf16 अनुशंसित है, हालांकि अन्य adamw/lion variants भी ठीक हैं।
- `mixed_precision` - आधुनिक GPUs पर `bf16`; अन्यथा `fp16`.
- `gradient_checkpointing` - लगभग हर स्थिति में true रखें।
- `gradient_checkpointing_interval` - बड़े GPUs पर हर _n_ blocks पर checkpoint करने के लिए 2+ पर सेट किया जा सकता है।

### उन्नत प्रयोगात्मक विशेषताएँ

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में प्रयोगात्मक फीचर्स शामिल हैं जो प्रशिक्षण की स्थिरता और प्रदर्शन को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बढ़ाता है, क्योंकि यह प्रशिक्षण के दौरान मॉडल को अपने इनपुट्स खुद जनरेट करने देता है।

> ⚠️ ये फीचर्स प्रशिक्षण के कंप्यूटेशनल ओवरहेड को बढ़ाते हैं।

</details>

### Assistant LoRA (Turbo)

Turbo को assistant adapter चाहिए:

- `assistant_lora_path`: `ostris/zimage_turbo_training_adapter`
- `assistant_lora_weight_name`:
  - `turbo`: `zimage_turbo_training_adapter_v1.safetensors`
  - `turbo-ostris-v2`: `zimage_turbo_training_adapter_v2.safetensors`

SimpleTuner turbo flavours के लिए इन्हें auto‑fill करता है जब तक आप override न करें। यदि गुणवत्ता गिरावट स्वीकार्य हो, तो `--disable_assistant_lora` से बंद करें।

### Validation prompts

`config/config.json` के अंदर "primary validation prompt" होता है, जो आमतौर पर आपके single subject या style के लिए मुख्य instance_prompt होता है। इसके अतिरिक्त, एक JSON फ़ाइल बनाई जा सकती है जिसमें वैलिडेशन के दौरान चलाने के लिए अतिरिक्त प्रॉम्प्ट्स हों।

उदाहरण config फ़ाइल `config/user_prompt_library.json.example` का फ़ॉर्मैट:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```
</details>

nicknames validation के लिए फ़ाइलनाम होते हैं, इसलिए इन्हें छोटा और फ़ाइलसिस्टम‑अनुकूल रखें।

ट्रेनर को इस prompt library की ओर इंगित करने के लिए, `config.json` के अंत में नया लाइन जोड़कर इसे TRAINER_EXTRA_ARGS में जोड़ें:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```
</details>

विविध प्रॉम्प्ट्स का सेट यह निर्धारित करने में मदद करेगा कि मॉडल प्रशिक्षण के दौरान collapse तो नहीं हो रहा। इस उदाहरण में `<token>` शब्द को अपने subject नाम (instance_prompt) से बदलें।

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
    "anime_<token>": "a breathtaking anime-style portrait of <token>, capturing her essence with vibrant colors and expressive features",
    "chef_<token>": "a high-quality, detailed photograph of <token> as a sous-chef, immersed in the art of culinary creation",
    "just_<token>": "a lifelike and intimate portrait of <token>, showcasing her unique personality and charm",
    "cinematic_<token>": "a cinematic, visually stunning photo of <token>, emphasizing her dramatic and captivating presence",
    "elegant_<token>": "an elegant and timeless portrait of <token>, exuding grace and sophistication",
    "adventurous_<token>": "a dynamic and adventurous photo of <token>, captured in an exciting, action-filled moment",
    "mysterious_<token>": "a mysterious and enigmatic portrait of <token>, shrouded in shadows and intrigue",
    "vintage_<token>": "a vintage-style portrait of <token>, evoking the charm and nostalgia of a bygone era",
    "artistic_<token>": "an artistic and abstract representation of <token>, blending creativity with visual storytelling",
    "futuristic_<token>": "a futuristic and cutting-edge portrayal of <token>, set against a backdrop of advanced technology",
    "woman": "a beautifully crafted portrait of a woman, highlighting her natural beauty and unique features",
    "man": "a powerful and striking portrait of a man, capturing his strength and character",
    "boy": "a playful and spirited portrait of a boy, capturing youthful energy and innocence",
    "girl": "a charming and vibrant portrait of a girl, emphasizing her bright personality and joy",
    "family": "a heartwarming and cohesive family portrait, showcasing the bonds and connections between loved ones"
}
```
</details>

> ℹ️ Z‑Image एक flow‑matching मॉडल है और छोटे, बहुत मिलते‑जुलते prompts लगभग वही इमेज उत्पन्न करेंगे। लंबे, अधिक वर्णनात्मक prompts उपयोग करें।

### CLIP score ट्रैकिंग

यदि आप मॉडल प्रदर्शन स्कोर करने के लिए evaluations सक्षम करना चाहते हैं, तो CLIP scores को कॉन्फ़िगर और इंटरप्रेट करने के लिए [यह दस्तावेज़](../evaluation/CLIP_SCORES.md) देखें।

### स्थिर evaluation loss

यदि आप मॉडल प्रदर्शन स्कोर करने के लिए stable MSE loss उपयोग करना चाहते हैं, तो evaluation loss को कॉन्फ़िगर और इंटरप्रेट करने के लिए [यह दस्तावेज़](../evaluation/EVAL_LOSS.md) देखें।

### Validation previews

SimpleTuner Tiny AutoEncoder मॉडलों का उपयोग करके generation के दौरान intermediate validation previews स्ट्रीम करने का समर्थन करता है। इससे आप webhook callbacks के जरिए real‑time में step‑by‑step validation images देख सकते हैं।

सक्रिय करने के लिए:
<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```
</details>

**आवश्यकताएँ:**
- Webhook configuration
- Validation सक्षम होना

`validation_preview_steps` को ऊँचा मान (जैसे 3 या 5) रखें ताकि Tiny AutoEncoder का ओवरहेड कम हो।

### Flow schedule shifting (flow matching)

Z‑Image जैसे flow‑matching मॉडलों में "shift" पैरामीटर होता है जो timestep schedule के प्रशिक्षित हिस्से को शिफ्ट करता है। resolution‑आधारित auto‑shift सुरक्षित डिफ़ॉल्ट है। manually shift बढ़ाने से सीखने का झुकाव coarse features की ओर जाता है; घटाने पर fine details की ओर। Turbo मॉडल के लिए इन मानों को बदलना नुकसान पहुँचा सकता है।

### Quantised model training

TorchAO या अन्य quantisation precision और VRAM आवश्यकताओं को कम कर सकते हैं — Optimum Quanto अब लगभग end‑of‑life पर है, लेकिन उपलब्ध है।

`config.json` उपयोगकर्ताओं के लिए:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
  "base_model_precision": "int8-torchao",
  "lora_rank": 16,
  "max_grad_norm": 1.0,
  "base_model_default_dtype": "bf16"
```
</details>

### डेटासेट विचार

> ⚠️ प्रशिक्षण के लिए इमेज गुणवत्ता महत्वपूर्ण है; Z‑Image artifacts को जल्दी absorb कर लेता है। अंत में उच्च गुणवत्ता डेटा पर final pass आवश्यक हो सकता है।

डेटासेट को पर्याप्त बड़ा रखें (कम से कम `train_batch_size * gradient_accumulation_steps`, और `vae_batch_size` से अधिक)। यदि **no images detected in dataset** दिखे तो `repeats` बढ़ाएँ।

उदाहरण multi‑backend config (`config/multidatabackend.json`):

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
[
  {
    "id": "pseudo-camera-10k-zimage",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/zimage/pseudo-camera-10k",
    "instance_data_dir": "datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery",
    "repeats": 0,
    "is_regularisation_data": true
  },
  {
    "id": "dreambooth-subject",
    "type": "local",
    "crop": false,
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/zimage/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "dreambooth-subject-512",
    "type": "local",
    "crop": false,
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/zimage/dreambooth-subject-512",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/zimage",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> `caption_strategy` विकल्प और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

512px और 1024px datasets को साथ चलाना समर्थित है और convergence सुधार सकता है।

datasets डायरेक्टरी बनाएँ:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

### WandB और Huggingface Hub में लॉग‑इन

प्रशिक्षण से पहले लॉग‑इन करें, खासकर यदि आप `--push_to_hub` और `--report_to=wandb` उपयोग कर रहे हैं:

```bash
wandb login
huggingface-cli login
```

### प्रशिक्षण रन निष्पादित करना

SimpleTuner डायरेक्टरी से, प्रशिक्षण शुरू करने के लिए आपके पास कई विकल्प हैं:

**विकल्प 1 (अनुशंसित - pip install):**

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
simpletuner train
```

**विकल्प 2 (Git clone विधि):**

```bash
simpletuner train
```

**विकल्प 3 (Legacy विधि - अभी भी काम करता है):**

```bash
./train.sh
```

इससे text embed और VAE आउटपुट कैशिंग डिस्क पर शुरू होगी।

अधिक जानकारी के लिए [dataloader](../DATALOADER.md) और [tutorial](../TUTORIAL.md) दस्तावेज़ देखें।

## Multi‑GPU Configuration

SimpleTuner में WebUI के माध्यम से **automatic GPU detection** है। Onboarding के दौरान आप कॉन्फ़िगर करेंगे:

- **Auto Mode**: सभी detected GPUs को optimal settings के साथ स्वतः उपयोग करता है
- **Manual Mode**: विशिष्ट GPUs चुनें या custom process count सेट करें
- **Disabled Mode**: single GPU training

WebUI आपके hardware को detect करके `--num_processes` और `CUDA_VISIBLE_DEVICES` स्वतः सेट करता है।

Manual configuration या advanced setups के लिए, installation guide में [Multi‑GPU Training section](../INSTALL.md#multiple-gpu-training) देखें।

## Inference टिप्स

### Guidance settings

Z‑Image flow‑matching है; कम guidance मान (0–1 के आसपास) गुणवत्ता और diversity बनाए रखते हैं। यदि आप उच्च guidance vectors के साथ ट्रेन करते हैं, तो सुनिश्चित करें कि आपका inference pipeline CFG सपोर्ट करे और batch CFG के साथ धीमी generation या उच्च VRAM उपयोग की अपेक्षा करें।

## नोट्स और समस्या‑समाधान टिप्स

### सबसे कम VRAM कॉन्फ़िग

- GPU: एक NVIDIA CUDA डिवाइस (10–12G) आक्रामक quantisation/offload के साथ
- System memory: ~32–48G
- Base model precision: `nf4-bnb` या `int8`
- Optimiser: Lion 8Bit Paged, `bnb-lion8bit-paged` या adamw variants
- Resolution: 512px (1024px के लिए अधिक VRAM चाहिए)
- Batch size: 1, zero gradient accumulation steps
- DeepSpeed: बंद/कॉन्फ़िगर नहीं
- यदि startup पर <=16G कार्ड्स में OOM हो, तो `--quantize_via=cpu` उपयोग करें
- `--gradient_checkpointing` सक्षम करें
- Ramtorch या group offload सक्षम करें

Pre‑caching चरण में मेमोरी खत्म हो सकती है; text encoder quantisation और VAE tiling को `--text_encoder_precision=int8-torchao` और `--vae_enable_tiling=true` से सक्षम किया जा सकता है। startup पर `--offload_during_startup=true` से और मेमोरी बचाई जा सकती है, जिससे text encoder या VAE में से केवल एक लोड रहता है, दोनों नहीं।

### Quantisation

- 16G कार्ड पर यह मॉडल ट्रेन करने के लिए अक्सर न्यूनतम 8bit quantisation चाहिए होती है।
- मॉडल को 8bit पर quantize करने से सामान्यतः प्रशिक्षण प्रभावित नहीं होता और बड़े batch sizes संभव होते हैं।
- **int8** को hardware acceleration का लाभ मिलता है; **nf4-bnb** VRAM और कम करता है लेकिन अधिक संवेदनशील है।
- LoRA को बाद में लोड करते समय **आदर्श रूप से** वही base model precision उपयोग करें जिस पर आपने प्रशिक्षण किया था।

### Aspect bucketing

- केवल square crops पर प्रशिक्षण सामान्यतः काम करता है, लेकिन multi‑aspect buckets generalisation सुधार सकते हैं।
- प्राकृतिक aspect buckets shapes को bias कर सकते हैं; यदि व्यापक कवरेज चाहिए तो random cropping मददगार हो सकता है।
- इमेज डायरेक्टरी dataset को कई बार परिभाषित करके configs मिश्रित करने से अच्छा generalisation मिला है।

### Learning rates

#### LoRA (--lora_type=standard)

- बड़े transformers पर कम learning rates अक्सर बेहतर व्यवहार करते हैं।
- बहुत ऊँचे ranks से पहले modest ranks (4–16) से शुरू करें।
- यदि मॉडल destabilise हो तो `max_grad_norm` घटाएँ; learning stalls हो तो बढ़ाएँ।

#### LoKr (--lora_type=lycoris)

- ऊँचे learning rates (जैसे AdamW के साथ `1e-3`, Lion के साथ `2e-4`) अच्छे काम कर सकते हैं; आवश्यकता अनुसार ट्यून करें।
- base model को संरक्षित रखने के लिए regularisation datasets को `is_regularisation_data` से चिह्नित करें।

### Image artifacts

Z‑Image खराब image artifacts को जल्दी absorb कर लेता है। साफ़‑सफाई के लिए उच्च‑गुणवत्ता डेटा पर final pass की आवश्यकता हो सकती है। यदि learning rate बहुत ऊँचा हो, डेटा कम गुणवत्ता का हो, या aspect handling गलत हो, तो grid artifacts आ सकते हैं।

### Training custom fine‑tuned Z‑Image models

कुछ fine‑tuned checkpoints में पूरा directory structure नहीं होता। आवश्यकता होने पर ये फ़ील्ड सही सेट करें:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
    "model_family": "z-image",
    "pretrained_model_name_or_path": "TONGYI-MAI/Z-Image-Base",
    "pretrained_transformer_model_name_or_path": "your-custom-transformer",
    "pretrained_vae_model_name_or_path": "TONGYI-MAI/Z-Image-Base",
    "pretrained_transformer_subfolder": "none"
}
```
</details>

## Troubleshooting

- Startup पर OOM: group offload सक्षम करें (Quanto के साथ नहीं), LoRA rank घटाएँ, या quantize करें (`--base_model_precision int8`/`nf4`).
- Blurry outputs: `validation_num_inference_steps` बढ़ाएँ (जैसे 24–28) या guidance को 1.0 की ओर बढ़ाएँ।
- Artifacts/overfitting: rank या learning rate घटाएँ, अधिक विविध prompts जोड़ें, या प्रशिक्षण छोटा करें।
- Assistant adapter समस्याएँ: turbo को adapter path/weight चाहिए; गुणवत्ता हानि स्वीकार्य हो तभी disable करें।
- धीमी validations: validation resolutions या steps घटाएँ; flow matching जल्दी converge करता है।
