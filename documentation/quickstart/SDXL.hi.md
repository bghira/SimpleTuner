## Stable Diffusion XL क्विकस्टार्ट

इस उदाहरण में, हम SimpleTuner टूलकिट का उपयोग करके Stable Diffusion XL मॉडल प्रशिक्षण करेंगे और `lora` मॉडल टाइप का उपयोग करेंगे।

आधुनिक, बड़े मॉडलों की तुलना में SDXL आकार में काफी छोटा है, इसलिए `full` प्रशिक्षण संभव हो सकता है, लेकिन इसके लिए LoRA की तुलना में अधिक VRAM और अन्य हाइपर‑पैरामीटर समायोजन चाहिए होंगे।

### पूर्वापेक्षाएँ

सुनिश्चित करें कि Python इंस्टॉल है; SimpleTuner 3.10 से 3.12 के साथ अच्छा काम करता है (AMD ROCm मशीनों को 3.12 चाहिए होगा)।

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
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

मैनुअल इंस्टॉलेशन या डेवलपमेंट सेटअप के लिए, [installation documentation](../INSTALL.md) देखें।

### वातावरण सेटअप

SimpleTuner चलाने के लिए, आपको एक configuration फ़ाइल, dataset और model directories, तथा एक dataloader configuration फ़ाइल सेट करनी होगी।

#### Configuration file

एक प्रयोगात्मक स्क्रिप्ट, `configure.py`, इंटरैक्टिव step‑by‑step कॉन्फ़िगरेशन के जरिए इस सेक्शन को पूरी तरह स्किप करने में मदद कर सकती है। इसमें कुछ सुरक्षा फीचर्स हैं जो सामान्य pitfalls से बचाते हैं।

**नोट:** यह आपके dataloader को **पूरी तरह** कॉन्फ़िगर नहीं करता। आपको उसे बाद में मैन्युअली करना होगा।

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
  "model_family": "sdxl",
  "model_flavour": "base-1.0",
  "output_dir": "/home/user/output/models",
  "validation_resolution": "1024x1024,1280x768",
  "validation_guidance": 3.4,
  "use_gradient_checkpointing": true,
  "learning_rate": 1e-4
}
```
</details>

- `model_family` - इसे `sdxl` पर सेट करें।
- `model_flavour` - इसे `base-1.0` पर सेट करें, या `pretrained_model_name_or_path` से किसी अन्य मॉडल पर पॉइंट करें।
- `model_type` - इसे `lora` पर सेट करें।
- `use_dora` - यदि आप DoRA ट्रेन करना चाहते हैं तो `true` सेट करें।
- `output_dir` - इसे उस डायरेक्टरी पर सेट करें जहाँ आप अपने checkpoints और validation images रखना चाहते हैं। यहाँ full path उपयोग करने की सलाह है।
- `validation_resolution` - इस उदाहरण के लिए इसे `1024x1024` पर सेट करें।
  - साथ ही, Stable Diffusion XL को multi‑aspect buckets पर fine‑tune किया गया था, और अन्य resolutions को कॉमा से अलग कर के दिया जा सकता है: `1024x1024,1280x768`
- `validation_guidance` - inference में परीक्षण के लिए जिस मान के साथ आप सहज हों, वही रखें। इसे `4.2` से `6.4` के बीच सेट करें।
- `sdxl_validation_pipeline_mode` - सामान्य validation के लिए `trained-stage` रखें। SDXL base/refiner split से validate करने के लिए `full-pipeline` उपयोग करें: stage 1 latent output के साथ `1 - refiner_training_strength` तक चलता है, फिर stage 2 उसी boundary से resume करता है।
  - केवल एक stage train करते समय, `sdxl_validation_stage1_model` और `sdxl_validation_stage2_model` peer stage के रूप में उपयोग होने वाले fixed base/refiner checkpoint को override कर सकते हैं।
- `use_gradient_checkpointing` - यदि आपके पास बहुत VRAM नहीं है, तो इसे `true` रखें।
- `learning_rate` - low‑rank नेटवर्क्स के लिए `1e-4` सामान्य है, लेकिन यदि "burning" या early overtraining दिखे तो `1e-5` अधिक conservative हो सकता है।

यदि आप Mac M‑series मशीन उपयोग कर रहे हैं तो कुछ अतिरिक्त सेटिंग्स:

- `mixed_precision` को `no` पर सेट करें।
  - यह pytorch 2.4 में सही था, लेकिन शायद 2.6+ में bf16 अब उपयोग किया जा सकता है
- `attention_mechanism` को `xformers` पर सेट किया जा सकता है, लेकिन यह अब लगभग obsolete है।

#### Quantised model training

Apple और NVIDIA सिस्टम्स पर टेस्ट किया गया है; Hugging Face Optimum‑Quanto Unet की precision और VRAM आवश्यकताओं को घटा सकता है, लेकिन यह SD3/Flux जैसे Diffusion Transformer मॉडलों जितना अच्छा काम नहीं करता, इसलिए अनुशंसित नहीं है।

यदि आपके संसाधन बहुत सीमित हैं, तब भी इसे उपयोग किया जा सकता है।

`config.json` के लिए:
<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "optimizer": "optimi-lion"
}
```
</details>

#### उन्नत प्रयोगात्मक विशेषताएँ

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में प्रयोगात्मक फीचर्स शामिल हैं जो प्रशिक्षण की स्थिरता और प्रदर्शन को काफी बेहतर कर सकते हैं, खासकर छोटे डेटासेट्स या SDXL जैसी पुरानी आर्किटेक्चर के लिए।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बढ़ाता है, क्योंकि यह प्रशिक्षण के दौरान मॉडल को अपने इनपुट्स खुद जनरेट करने देता है।
*   **[Diff2Flow](../experimental/DIFF2FLOW.md):** Flow Matching objective के साथ SDXL को प्रशिक्षित करने देता है, जिससे generation की straightness और गुणवत्ता संभावित रूप से बेहतर हो सकती है।

> ⚠️ ये फीचर्स प्रशिक्षण के कंप्यूटेशनल ओवरहेड को बढ़ाते हैं।

</details>

#### डेटासेट विचार

अपने मॉडल को प्रशिक्षित करने के लिए पर्याप्त बड़ा डेटासेट होना महत्वपूर्ण है। डेटासेट आकार पर सीमाएँ हैं, और आपको सुनिश्चित करना होगा कि आपका डेटासेट पर्याप्त बड़ा हो। ध्यान दें कि न्यूनतम डेटासेट आकार `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS` है। यदि डेटासेट बहुत छोटा है, तो ट्रेनर उसे discover नहीं कर पाएगा।

> 💡 **टिप:** बड़े डेटासेट्स में जहाँ डिस्क स्पेस चिंता का विषय हो, आप `--vae_cache_disable` का उपयोग करके बिना डिस्क कैश के online VAE encoding कर सकते हैं। यह `--vae_cache_ondemand` उपयोग करने पर भी implicitly सक्षम रहता है, लेकिन `--vae_cache_disable` जोड़ने से सुनिश्चित होता है कि डिस्क पर कुछ भी न लिखा जाए।

आपके डेटासेट के अनुसार, आपको डेटासेट डायरेक्टरी और dataloader configuration फ़ाइल अलग तरीके से सेट करनी होगी। इस उदाहरण में हम [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) डेटासेट उपयोग करेंगे।

अपने `OUTPUT_DIR` डायरेक्टरी में, एक multidatabackend.json बनाएँ:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
[
  {
    "id": "pseudo-camera-10k-sdxl",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "random",
    "resolution": 1.0,
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
    "cache_dir_vae": "cache/vae/sdxl/pseudo-camera-10k",
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
    "cache_dir": "cache/text/sdxl/pseudo-camera-10k",
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
