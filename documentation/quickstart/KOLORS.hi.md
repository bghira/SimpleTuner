## Kwai Kolors क्विकस्टार्ट

इस उदाहरण में, हम SimpleTuner टूलकिट का उपयोग करके Kwai Kolors मॉडल प्रशिक्षण करेंगे और `lora` मॉडल टाइप का उपयोग करेंगे।

Kolors का आकार लगभग SDXL जितना है, इसलिए आप `full` प्रशिक्षण आज़मा सकते हैं, लेकिन उसके बदलाव इस क्विकस्टार्ट में नहीं बताए गए हैं।

### पूर्वापेक्षाएँ

सुनिश्चित करें कि Python इंस्टॉल है; SimpleTuner 3.10 से 3.12 के साथ अच्छा काम करता है।

आप इसे चलाकर जांच सकते हैं:

```bash
python --version
```

यदि आपके Ubuntu पर Python 3.12 इंस्टॉल नहीं है, तो आप यह प्रयास कर सकते हैं:

```bash
apt -y install python3.13 python3.13-venv
```

#### Container image dependencies

Vast, RunPod, और TensorDock (आदि) के लिए, CUDA 12.2‑12.8 इमेज पर CUDA extensions कम्पाइल करने हेतु यह काम करेगा:

```bash
apt -y install nvidia-cuda-toolkit
```

### इंस्टॉलेशन

pip के जरिए SimpleTuner इंस्टॉल करें:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
```

मैनुअल इंस्टॉलेशन या डेवलपमेंट सेटअप के लिए, [installation documentation](../INSTALL.md) देखें।

### वातावरण सेटअप

SimpleTuner चलाने के लिए, आपको एक configuration फ़ाइल, dataset और model directories, तथा एक dataloader configuration फ़ाइल सेट करनी होगी।

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

#### AMD ROCm follow‑up steps

AMD MI300X को उपयोगी बनाने के लिए निम्न चलाना आवश्यक है:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

फिर, आपको निम्न वेरिएबल्स बदलने होंगे:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "model_type": "lora",
  "model_family": "kolors",
  "pretrained_model_name_or_path": "Kwai-Kolors/Kolors-diffusers",
  "output_dir": "/home/user/output/models",
  "validation_resolution": "1024x1024,1280x768",
  "validation_guidance": 3.4,
  "use_gradient_checkpointing": true,
  "learning_rate": 1e-4
}
```
</details>

- `pretrained_model_name_or_path` - इसे `Kwai-Kolors/Kolors-diffusers` पर सेट करें।
- `MODEL_TYPE` - इसे `lora` पर सेट करें।
- `USE_DORA` - यदि आप DoRA ट्रेन करना चाहते हैं तो `true` सेट करें।
- `MODEL_FAMILY` - इसे `kolors` पर सेट करें।
- `OUTPUT_DIR` - इसे उस डायरेक्टरी पर सेट करें जहाँ आप अपने checkpoints और validation images रखना चाहते हैं। यहाँ full path उपयोग करने की सलाह है।
- `VALIDATION_RESOLUTION` - इस उदाहरण के लिए इसे `1024x1024` पर सेट करें।
  - साथ ही, Kolors को multi‑aspect buckets पर fine‑tune किया गया था, और अन्य resolutions को कॉमा से अलग कर के दिया जा सकता है: `1024x1024,1280x768`
- `VALIDATION_GUIDANCE` - inference में परीक्षण के लिए जिस मान के साथ आप सहज हों, वही रखें। इसे `4.2` से `6.4` के बीच सेट करें।
- `USE_GRADIENT_CHECKPOINTING` - यदि आपके पास बहुत VRAM नहीं है, तो इसे `true` रखें।
- `LEARNING_RATE` - low‑rank नेटवर्क्स के लिए `1e-4` सामान्य है, लेकिन यदि "burning" या early overtraining दिखे तो `1e-5` अधिक conservative हो सकता है।

यदि आप Mac M‑series मशीन उपयोग कर रहे हैं तो कुछ अतिरिक्त सेटिंग्स:

- `mixed_precision` को `no` पर सेट करें।
- `attention_mechanism` को `diffusers` पर सेट करें, क्योंकि `xformers` और अन्य मान शायद काम नहीं करेंगे।

#### Quantised model training

Apple और NVIDIA सिस्टम्स पर टेस्ट किया गया है; Hugging Face Optimum‑Quanto विशेष रूप से ChatGLM 6B (text encoder) की precision और VRAM आवश्यकताओं को कम करने के लिए उपयोग किया जा सकता है।

`config.json` के लिए:
<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "optimizer": "adamw_bf16"
}
```
</details>

`config.env` उपयोगकर्ताओं (deprecated) के लिए:

```bash
# choices: int8-quanto, int4-quanto, int2-quanto, fp8-quanto
# int8-quanto was tested with a single subject dreambooth LoRA.
# fp8-quanto does not work on Apple systems. you must use int levels.
# int2-quanto is pretty extreme and gets the whole rank-1 LoRA down to about 13.9GB VRAM.
# may the gods have mercy on your soul, should you push things Too Far.
export TRAINER_EXTRA_ARGS="--base_model_precision=int8-quanto"

# Maybe you want the text encoders to remain full precision so your text embeds are cake.
# We unload the text encoders before training, so, that's not an issue during training time - only during pre-caching.
# Alternatively, you can go ham on quantisation here and run them in int4 or int8 mode, because no one can stop you.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --text_encoder_1_precision=no_change"

# When you're quantising the model, --base_model_default_dtype is set to bf16 by default. This setup requires adamw_bf16, but saves the most memory.
# adamw_bf16 only supports bf16 training, but any other optimiser will support both bf16 or fp32 training precision.
export OPTIMIZER="adamw_bf16"
```

#### उन्नत प्रयोगात्मक विशेषताएँ

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में प्रयोगात्मक फीचर्स शामिल हैं जो प्रशिक्षण की स्थिरता और प्रदर्शन को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बढ़ाता है, क्योंकि यह प्रशिक्षण के दौरान मॉडल को अपने इनपुट्स खुद जनरेट करने देता है।
*   **[Diff2Flow](../experimental/DIFF2FLOW.md):** Flow Matching objective के साथ Kolors को प्रशिक्षित करने देता है, जिससे generation की straightness और गुणवत्ता संभावित रूप से बेहतर हो सकती है।

> ⚠️ ये फीचर्स प्रशिक्षण के कंप्यूटेशनल ओवरहेड को बढ़ाते हैं।

</details>

#### डेटासेट विचार

अपने मॉडल को प्रशिक्षित करने के लिए पर्याप्त बड़ा डेटासेट होना महत्वपूर्ण है। डेटासेट आकार पर सीमाएँ हैं, और आपको सुनिश्चित करना होगा कि आपका डेटासेट पर्याप्त बड़ा हो। ध्यान दें कि न्यूनतम डेटासेट आकार `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS` है। यदि डेटासेट बहुत छोटा है, तो ट्रेनर उसे discover नहीं कर पाएगा।

आपके डेटासेट के अनुसार, आपको डेटासेट डायरेक्टरी और dataloader configuration फ़ाइल अलग तरीके से सेट करनी होगी। इस उदाहरण में हम [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) डेटासेट उपयोग करेंगे।

अपने `OUTPUT_DIR` डायरेक्टरी में, एक multidatabackend.json बनाएँ:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
[
  {
    "id": "pseudo-camera-10k-kolors",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "random",
    "resolution": 1.0,
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
    "cache_dir_vae": "cache/vae/kolors/pseudo-camera-10k",
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
    "cache_dir": "cache/text/kolors/pseudo-camera-10k",
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
huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=datasets/pseudo-camera-10k
```

यह लगभग 10k फोटोग्राफ सैंपल्स को आपकी `datasets/pseudo-camera-10k` डायरेक्टरी में डाउनलोड करेगा, जो अपने‑आप बन जाएगी।

#### WandB और Huggingface Hub में लॉग‑इन

प्रशिक्षण शुरू करने से पहले WandB और HF Hub में लॉग‑इन करना बेहतर है, खासकर यदि आप `push_to_hub: true` और `--report_to=wandb` उपयोग कर रहे हैं।

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

इससे text embed और VAE आउटपुट कैशिंग डिस्क पर शुरू होगी।

अधिक जानकारी के लिए [dataloader](../DATALOADER.md) और [tutorial](../TUTORIAL.md) दस्तावेज़ देखें।

### CLIP score ट्रैकिंग

यदि आप मॉडल प्रदर्शन स्कोर करने के लिए evaluations सक्षम करना चाहते हैं, तो CLIP scores को कॉन्फ़िगर और इंटरप्रेट करने के लिए [यह दस्तावेज़](../evaluation/CLIP_SCORES.md) देखें।

# स्थिर evaluation loss

यदि आप मॉडल प्रदर्शन स्कोर करने के लिए stable MSE loss उपयोग करना चाहते हैं, तो evaluation loss को कॉन्फ़िगर और इंटरप्रेट करने के लिए [यह दस्तावेज़](../evaluation/EVAL_LOSS.md) देखें।

#### Validation previews

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

`validation_preview_steps` को ऊँचा मान (जैसे 3 या 5) रखें ताकि Tiny AutoEncoder का ओवरहेड कम हो। `validation_num_inference_steps=20` और `validation_preview_steps=5` के साथ, आपको steps 5, 10, 15, और 20 पर preview images मिलेंगी।
