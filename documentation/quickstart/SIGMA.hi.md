## PixArt Sigma क्विकस्टार्ट

इस उदाहरण में, हम SimpleTuner टूलकिट का उपयोग करके PixArt Sigma मॉडल को प्रशिक्षित करेंगे और `full` मॉडल टाइप का उपयोग करेंगे, क्योंकि यह छोटा मॉडल होने के कारण VRAM में फिट हो सकता है।

### पूर्वापेक्षाएँ

सुनिश्चित करें कि Python इंस्टॉल है; SimpleTuner 3.10 से 3.12 के साथ अच्छा काम करता है।

आप इसे चलाकर जांच सकते हैं:

```bash
python --version
```

यदि आपके Ubuntu पर python 3.12 इंस्टॉल नहीं है, तो आप यह कोशिश कर सकते हैं:

```bash
apt -y install python3.13 python3.13-venv
```

#### Container image dependencies

Vast, RunPod, और TensorDock (आदि) के लिए, CUDA 12.2‑12.8 इमेज पर CUDA extensions कम्पाइल करने के लिए यह काम करेगा:

```bash
apt -y install nvidia-cuda-toolkit
```

### इंस्टॉलेशन

pip के जरिए SimpleTuner इंस्टॉल करें:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

मैनुअल इंस्टॉलेशन या डेवलपमेंट सेटअप के लिए, [installation documentation](../INSTALL.md) देखें।

#### AMD ROCm follow-up steps

AMD MI300X को उपयोगी बनाने के लिए निम्न चलाना आवश्यक है:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

### वातावरण सेटअप

SimpleTuner चलाने के लिए, आपको एक configuration फ़ाइल, dataset और model directories, तथा एक dataloader configuration फ़ाइल सेट करनी होगी।

#### Configuration file

एक प्रयोगात्मक स्क्रिप्ट `configure.py` इंटरैक्टिव step‑by‑step कॉन्फ़िगरेशन के जरिए इस सेक्शन को पूरी तरह स्किप करने में मदद कर सकती है। इसमें कुछ सुरक्षा फीचर्स हैं जो सामान्य pitfalls से बचाते हैं।

**नोट:** यह आपके dataloader को कॉन्फ़िगर नहीं करता। आपको वह बाद में मैनुअली करना होगा।

इसे चलाने के लिए:

```bash
simpletuner configure
```
> ⚠️ जिन देशों में Hugging Face Hub आसानी से उपलब्ध नहीं है, वहाँ `HF_ENDPOINT=https://hf-mirror.com` को अपने `~/.bashrc` या `~/.zshrc` में जोड़ें, यह आपके सिस्टम के `$SHELL` पर निर्भर करता है।

यदि आप मैनुअल कॉन्फ़िगर करना पसंद करते हैं:

`config/config.json.example` को `config/config.json` में कॉपी करें:

```bash
cp config/config.json.example config/config.json
```

वहाँ आपको निम्न वेरिएबल्स बदलने होंगे:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "model_type": "full",
  "use_bitfit": false,
  "pretrained_model_name_or_path": "pixart-alpha/pixart-sigma-xl-2-1024-ms",
  "model_family": "pixart_sigma",
  "output_dir": "/home/user/output/models",
  "validation_resolution": "1024x1024,1280x768",
  "validation_guidance": 3.5
}
```
</details>

- `pretrained_model_name_or_path` - इसे `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` पर सेट करें।
- `MODEL_TYPE` - इसे `full` पर सेट करें।
- `USE_BITFIT` - इसे `false` पर सेट करें।
- `MODEL_FAMILY` - इसे `pixart_sigma` पर सेट करें।
- `OUTPUT_DIR` - इसे उस डायरेक्टरी पर सेट करें जहाँ आप अपने checkpoints और validation images रखना चाहते हैं। यहाँ full path उपयोग करने की सलाह है।
- `VALIDATION_RESOLUTION` - PixArt Sigma 1024px या 2048px मॉडल फॉर्मैट में आता है, इसलिए इस उदाहरण के लिए इसे सावधानी से `1024x1024` रखें।
  - साथ ही, PixArt को multi‑aspect buckets पर fine‑tune किया गया था, और अन्य resolutions को कॉमा से अलग कर के दिया जा सकता है: `1024x1024,1280x768`
- `VALIDATION_GUIDANCE` - PixArt को बहुत कम मान लाभ देता है। इसे `3.6` से `4.4` के बीच सेट करें।
- `pixart_validation_pipeline_mode` - सामान्य validation के लिए `trained-stage` रखें। v0.7 split pipeline, जिसमें 900M MoE-style stage split भी शामिल है, validate करने के लिए `full-pipeline` उपयोग करें: stage 1 `1 - refiner_training_strength` तक latents के रूप में चलता है, फिर stage 2 उसी boundary से resume करता है।
  - यदि आप केवल एक stage train करते हैं, तो validation में उपयोग होने वाले fixed peer-stage checkpoint को override करने के लिए `pixart_validation_stage1_model` या `pixart_validation_stage2_model` सेट करें।

Mac M‑series मशीन पर कुछ अतिरिक्त सेटिंग्स:

- `mixed_precision` को `no` पर सेट करें।

> 💡 **टिप:** बड़े डेटासेट्स में जहाँ डिस्क स्पेस चिंता का विषय हो, आप `--vae_cache_disable` का उपयोग करके VAE एन्कोडिंग ऑनलाइन कर सकते हैं, जिससे डिस्क पर कैश नहीं बनेगा।

#### Dataset considerations

अपने मॉडल को प्रशिक्षित करने के लिए पर्याप्त बड़ा डेटासेट होना महत्वपूर्ण है। डेटासेट आकार पर सीमाएँ हैं, और आपको सुनिश्चित करना होगा कि आपका डेटासेट पर्याप्त बड़ा हो। ध्यान दें कि न्यूनतम डेटासेट आकार `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS` है। यदि डेटासेट बहुत छोटा है तो ट्रेनर उसे discover नहीं कर पाएगा।

आपके डेटासेट के अनुसार, आपको डेटासेट डायरेक्टरी और dataloader configuration फ़ाइल अलग तरीके से सेट करनी होगी। इस उदाहरण में हम [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) डेटासेट उपयोग करेंगे।

अपने `/home/user/simpletuner/config` डायरेक्टरी में, एक multidatabackend.json बनाएँ:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
[
  {
    "id": "pseudo-camera-10k-pixart",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "random",
    "resolution": 1.0,
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
    "cache_dir_vae": "cache/vae/pixart/pseudo-camera-10k",
    "instance_data_dir": "/home/user/simpletuner/datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery"
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/pixart/pseudo-camera-10k",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> `caption_strategy` विकल्प और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

फिर, एक `datasets` डायरेक्टरी बनाएँ:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
popd
```

यह लगभग 10k फोटोग्राफ सैंपल्स को आपकी `datasets/pseudo-camera-10k` डायरेक्टरी में डाउनलोड करेगा, जो अपने‑आप बना दी जाएगी।

#### WandB और Huggingface Hub में लॉग‑इन

प्रशिक्षण शुरू करने से पहले WandB और HF Hub में लॉग‑इन करना बेहतर है, खासकर अगर आप `push_to_hub: true` और `--report_to=wandb` उपयोग कर रहे हैं।

यदि आप Git LFS रिपॉज़िटरी में मैन्युअली आइटम्स push करने वाले हैं, तो `git config --global credential.helper store` भी चलाएँ।

निम्न कमांड चलाएँ:

```bash
wandb login
```

और

```bash
huggingface-cli login
```

निर्देशों का पालन करके दोनों सेवाओं में लॉग‑इन करें।

### प्रशिक्षण रन निष्पादित करना

SimpleTuner डायरेक्टरी से, बस यह चलाएँ:

```bash
bash train.sh
```

इससे text embeds और VAE आउटपुट कैशिंग डिस्क पर शुरू होगी।

अधिक जानकारी के लिए [dataloader](../DATALOADER.md) और [tutorial](../TUTORIAL.md) दस्तावेज़ देखें।

### CLIP score ट्रैकिंग

यदि आप मॉडल प्रदर्शन को स्कोर करने के लिए evaluations सक्षम करना चाहते हैं, तो CLIP scores कॉन्फ़िगर करने और समझने के लिए [इस दस्तावेज़](../evaluation/CLIP_SCORES.md) को देखें।
