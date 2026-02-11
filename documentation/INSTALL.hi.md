# सेटअप

जो उपयोगकर्ता Docker या किसी अन्य कंटेनर ऑर्केस्ट्रेशन प्लेटफ़ॉर्म का उपयोग करना चाहते हैं, वे पहले [यह दस्तावेज़](DOCKER.md) देखें।

## इंस्टॉलेशन

Windows 10 या उससे नए संस्करण पर काम करने वाले उपयोगकर्ताओं के लिए, Docker और WSL पर आधारित इंस्टॉलेशन गाइड [यह दस्तावेज़](DOCKER.md) में उपलब्ध है।

### Pip इंस्टॉलेशन विधि

आप pip का उपयोग करके SimpleTuner को आसानी से इंस्टॉल कर सकते हैं, जो अधिकांश उपयोगकर्ताओं के लिए अनुशंसित है:

```bash
# for CUDA
pip install 'simpletuner[cuda]'
# for CUDA 13 / Blackwell (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
# for ROCm
pip install 'simpletuner[rocm]' --extra-index-url https://download.pytorch.org/whl/rocm7.1
# for Apple Silicon
pip install 'simpletuner[apple]'
# for CPU-only (not recommended)
pip install 'simpletuner[cpu]'
# for JPEG XL support (optional)
pip install 'simpletuner[jxl]'

# development requirements (optional, only for submitting PRs or running tests)
pip install 'simpletuner[dev]'
```

### Git रिपॉज़िटरी विधि

लोकल डेवलपमेंट या टेस्टिंग के लिए, आप SimpleTuner रिपॉज़िटरी को क्लोन कर सकते हैं और python venv सेट अप कर सकते हैं:

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git

cd SimpleTuner

# if python --version shows 3.11 or 3.12, you may want to upgrade to 3.13.
python3.13 -m venv .venv

source .venv/bin/activate
```

> ℹ️ आप अपनी कस्टम venv path का उपयोग `config/config.env` फ़ाइल में `export VENV_PATH=/path/to/.venv` सेट करके कर सकते हैं।

**नोट:** हम यहाँ `release` ब्रांच इंस्टॉल कर रहे हैं; `main` ब्रांच में प्रयोगात्मक फीचर्स हो सकते हैं जो बेहतर परिणाम या कम मेमोरी उपयोग दे सकते हैं।

ऑटोमैटिक प्लेटफ़ॉर्म डिटेक्शन के साथ SimpleTuner इंस्टॉल करें:

```bash
# Basic installation (auto-detects CUDA/ROCm/Apple)
pip install -e .

# With JPEG XL support
pip install -e .[jxl]
```

**नोट:** setup.py आपके प्लेटफ़ॉर्म (CUDA/ROCm/Apple) को स्वतः पहचानता है और उपयुक्त dependencies इंस्टॉल करता है।

#### NVIDIA Hopper / Blackwell फॉलो‑अप चरण

वैकल्पिक रूप से, Hopper (या उससे नए) उपकरण `torch.compile` का उपयोग करते समय बेहतर inference और प्रशिक्षण प्रदर्शन के लिए FlashAttention3 का लाभ उठा सकते हैं।

आपको अपने SimpleTuner डायरेक्टरी से, venv सक्रिय रखते हुए, निम्न कमांड्स चलाने होंगे:

```bash
git clone https://github.com/Dao-AILab/flash-attention
pushd flash-attention
  pushd hopper
    python setup.py install
  popd
popd
```

> ⚠️ SimpleTuner में flash_attn बिल्ड का प्रबंधन अभी कमजोर रूप से समर्थित है। अपडेट्स पर यह टूट सकता है, जिससे आपको समय‑समय पर यह बिल्ड प्रक्रिया फिर से चलानी पड़ सकती है।

#### AMD ROCm फॉलो‑अप चरण

AMD MI300X को उपयोग योग्य बनाने के लिए निम्न चरण चलाने होंगे:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
  python3 -m pip install --upgrade pip
  python3 -m pip install .
popd
```

> ℹ️ **ROCm acceleration defaults**: जब SimpleTuner HIP‑सक्षम PyTorch बिल्ड का पता लगाता है तो यह स्वतः `PYTORCH_TUNABLEOP_ENABLED=1` एक्सपोर्ट करता है (यदि आपने पहले से सेट नहीं किया है), ताकि TunableOp kernels उपलब्ध हों। MI300/gfx94x डिवाइसों पर हम डिफ़ॉल्ट रूप से `HIPBLASLT_ALLOW_TF32=1` भी सेट करते हैं, जिससे हाथ से वातावरण सेट किए बिना hipBLASLt की TF32 paths सक्षम हो जाती हैं।

### सभी प्लेटफ़ॉर्म

- 2a. **विकल्प एक (अनुशंसित)**: `simpletuner configure` चलाएँ
- 2b. **विकल्प दो**: `config/config.json.example` को `config/config.json` में कॉपी करें और विवरण भरें।

> ⚠️ जिन देशों में Hugging Face Hub आसानी से उपलब्ध नहीं है, वहाँ के उपयोगकर्ताओं को अपने सिस्टम के `$SHELL` के अनुसार `~/.bashrc` या `~/.zshrc` में `HF_ENDPOINT=https://hf-mirror.com` जोड़ना चाहिए।

#### मल्टी‑GPU प्रशिक्षण {#multiple-gpu-training}

SimpleTuner अब WebUI के जरिए **स्वचालित GPU पहचान और कॉन्फ़िगरेशन** शामिल करता है। पहली बार लोड होने पर, आपको एक onboarding चरण के माध्यम से गाइड किया जाएगा जो आपके GPUs पहचानता है और Accelerate को स्वतः कॉन्फ़िगर करता है।

##### WebUI ऑटो‑डिटेक्शन (अनुशंसित)

जब आप पहली बार WebUI लॉन्च करते हैं या `simpletuner configure` का उपयोग करते हैं, तो आपको "Accelerate GPU Defaults" onboarding चरण मिलेगा जो:

1. सिस्टम पर उपलब्ध सभी GPUs को **स्वचालित रूप से पहचानता** है
2. नाम, मेमोरी, और डिवाइस IDs सहित **GPU विवरण दिखाता** है
3. मल्टी‑GPU प्रशिक्षण के लिए **उत्तम सेटिंग्स की सिफ़ारिश करता** है
4. **तीन कॉन्फ़िगरेशन मोड प्रदान करता है:**

   - **ऑटो मोड** (अनुशंसित): सभी detected GPUs को optimal process count के साथ उपयोग करता है
   - **मैन्युअल मोड**: विशिष्ट GPUs चुनें या कस्टम process count सेट करें
   - **Disabled मोड**: केवल single GPU प्रशिक्षण

**यह कैसे काम करता है:**
- सिस्टम CUDA/ROCm के जरिए आपके GPU हार्डवेयर का पता लगाता है
- उपलब्ध डिवाइसों के आधार पर optimal `--num_processes` गणना करता है
- विशिष्ट GPUs चुने जाने पर `CUDA_VISIBLE_DEVICES` स्वतः सेट करता है
- भविष्य के प्रशिक्षण रन के लिए आपकी प्राथमिकताएँ सेव करता है

##### मैन्युअल कॉन्फ़िगरेशन

यदि WebUI का उपयोग नहीं कर रहे हैं, तो आप अपने `config.json` में GPU visibility सीधे नियंत्रित कर सकते हैं:

```json
{
  "accelerate_visible_devices": [0, 1, 2],
  "num_processes": 3
}
```

यह प्रशिक्षण को GPUs 0, 1, और 2 तक सीमित करेगा और 3 प्रक्रियाएँ लॉन्च करेगा।

3. यदि आप `--report_to='wandb'` (डिफ़ॉल्ट) का उपयोग कर रहे हैं, तो निम्न चरण आपकी सांख्यिकी रिपोर्ट करने में मदद करेंगे:

```bash
wandb login
```

प्रिंटेड निर्देशों का पालन करें, अपनी API key ढूँढें और कॉन्फ़िगर करें।

इसके बाद, आपके प्रशिक्षण सत्र और validation डेटा Weights & Biases पर उपलब्ध होंगे।

> ℹ️ यदि आप Weights & Biases या Tensorboard रिपोर्टिंग पूरी तरह बंद करना चाहते हैं, तो `--report-to=none` का उपयोग करें।


4. simpletuner के साथ प्रशिक्षण शुरू करें; logs `debug.log` में लिखे जाएंगे

```bash
simpletuner train
```

> ⚠️ इस बिंदु पर, यदि आपने `simpletuner configure` का उपयोग किया है, तो आप तैयार हैं! यदि नहीं - ये कमांड्स काम करेंगे, लेकिन आगे की कॉन्फ़िगरेशन आवश्यक है। अधिक जानकारी के लिए [ट्यूटोरियल](TUTORIAL.md) देखें।

### यूनिट टेस्ट चलाएँ

यह सुनिश्चित करने के लिए कि इंस्टॉलेशन सफलतापूर्वक पूरा हुआ है, यूनिट टेस्ट चलाएँ:

```bash
python -m unittest discover tests/
```

## उन्नत: कई कॉन्फ़िगरेशन वातावरण

जो उपयोगकर्ता कई मॉडल ट्रेन करते हैं या अलग‑अलग डेटासेट/सेटिंग्स के बीच जल्दी स्विच करना चाहते हैं, स्टार्टअप पर दो environment variables देखे जाते हैं।

इनका उपयोग करने के लिए:

```bash
simpletuner train env=default config_backend=env
```

- `env` का डिफ़ॉल्ट `default` होता है, जो इस गाइड द्वारा कॉन्फ़िगर किए गए सामान्य `SimpleTuner/config/` डायरेक्टरी की ओर इशारा करता है
  - `simpletuner train env=pixart` उपयोग करने पर `SimpleTuner/config/pixart` डायरेक्टरी में `config.env` खोजा जाएगा
- `config_backend` का डिफ़ॉल्ट `env` होता है, जो इसी गाइड द्वारा कॉन्फ़िगर की गई सामान्य `config.env` फ़ाइल का उपयोग करता है
  - समर्थित विकल्प: `env`, `json`, `toml`, या `cmd` (यदि आप `train.py` मैन्युअली चलाते हैं)
  - `simpletuner train config_backend=json` उपयोग करने पर `SimpleTuner/config/config.json` खोजेगा, न कि `config.env`
  - इसी तरह, `config_backend=toml` `config.env` का उपयोग करेगा

आप `config/config.env` बना सकते हैं जिसमें इनमें से एक या दोनों मान हों:

```bash
ENV=default
CONFIG_BACKEND=json
```

इन्हें अगले रन पर याद रखा जाएगा। ध्यान दें कि इन्हें ऊपर बताए गए multiGPU विकल्पों के अतिरिक्त जोड़ा जा सकता है।

## प्रशिक्षण डेटा

एक सार्वजनिक डेटासेट [Hugging Face Hub पर उपलब्ध](https://huggingface.co/datasets/bghira/pseudo-camera-10k) है जिसमें लगभग 10k छवियाँ हैं, और कैप्शन फ़ाइल‑नाम के रूप में हैं, जो SimpleTuner के लिए उपयोग‑तैयार हैं।

आप छवियों को एक ही फ़ोल्डर में रख सकते हैं या उन्हें सबडायरेक्टरीज़ में सुव्यवस्थित कर सकते हैं।

### छवि चयन दिशानिर्देश

**गुणवत्ता आवश्यकताएँ:**
- JPEG artifacts या धुंधली छवियाँ नहीं - आधुनिक मॉडल इन्हें सीख लेते हैं
- ग्रेनी CMOS sensor noise से बचें (यह सभी generated छवियों में दिखेगा)
- watermark, badge, या signatures नहीं (ये सीख लिए जाएंगे)
- movie frames सामान्यतः काम नहीं करते क्योंकि compression होता है (इसके बजाय production stills उपयोग करें)

**तकनीकी विनिर्देश:**
- छवियाँ आदर्श रूप से 64 से विभाज्य हों (resize किए बिना reuse संभव होता है)
- संतुलित क्षमताओं के लिए square और non‑square छवियों का मिश्रण रखें
- सर्वोत्तम परिणामों के लिए विविध, उच्च‑गुणवत्ता डेटासेट का उपयोग करें

### कैप्शनिंग

SimpleTuner में bulk‑renaming के लिए [captioning scripts](/scripts/toolkit/README.md) उपलब्ध हैं। समर्थित कैप्शन फ़ॉर्मैट:
- फ़ाइल‑नाम को कैप्शन के रूप में (डिफ़ॉल्ट)
- `--caption_strategy=textfile` के साथ text files
- JSONL, CSV, या advanced metadata files

**अनुशंसित कैप्शनिंग टूल्स:**
- **InternVL2**: सर्वोत्तम गुणवत्ता लेकिन धीमा (छोटे डेटासेट)
- **BLIP3**: अच्छा instruction following वाला हल्का विकल्प
- **Florence2**: सबसे तेज़ लेकिन कुछ को आउटपुट पसंद नहीं आते

### प्रशिक्षण बैच आकार

आपका अधिकतम बैच आकार VRAM और रेज़ोल्यूशन पर निर्भर करता है:
```
vram use = batch size * resolution + base_requirements
```

**मुख्य सिद्धांत:**
- VRAM समस्याओं के बिना संभव हो तो सबसे बड़ा बैच आकार उपयोग करें
- उच्च रेज़ोल्यूशन = अधिक VRAM = कम बैच आकार
- यदि 128x128 पर बैच आकार 1 भी काम नहीं करता, तो हार्डवेयर अपर्याप्त है

#### मल्टी‑GPU डेटासेट आवश्यकताएँ

मल्टी‑GPU के साथ प्रशिक्षण करते समय, आपका डेटासेट **effective batch size** के लिए पर्याप्त बड़ा होना चाहिए:
```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

**उदाहरण:** 4 GPUs और `train_batch_size=4` के साथ, आपको प्रति aspect bucket कम से कम 16 सैंपल चाहिए।

**छोटे डेटासेट के समाधान:**
- repeats को auto‑adjust करने के लिए `--allow_dataset_oversubscription` का उपयोग करें
- अपने dataloader कॉन्फ़िग में `repeats` मैन्युअली सेट करें
- बैच आकार या GPU संख्या कम करें

पूरे विवरण के लिए [DATALOADER.md](DATALOADER.md#multi-gpu-training-and-dataset-sizing) देखें।

## Hugging Face Hub पर प्रकाशित करना

ट्रेनिंग पूरी होने पर मॉडलों को Hub पर स्वतः push करने के लिए `config/config.json` में जोड़ें:

```json
{
  "push_to_hub": true,
  "hub_model_name": "your-model-name"
}
```

ट्रेनिंग से पहले लॉगिन करें:
```bash
huggingface-cli login
```

## डिबगिंग

`config/config.env` में विस्तृत लॉगिंग सक्षम करें:

```bash
export SIMPLETUNER_LOG_LEVEL=DEBUG
export SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG
```

प्रोजेक्ट रूट में `debug.log` फ़ाइल बनाई जाएगी जिसमें सभी लॉग एंट्रीज़ होंगी।
