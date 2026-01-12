## Stable Diffusion 3

इस उदाहरण में, हम SimpleTuner टूलकिट का उपयोग करके Stable Diffusion 3 मॉडल प्रशिक्षण करेंगे और `lora` मॉडल टाइप का उपयोग करेंगे।

### पूर्वापेक्षाएँ

सुनिश्चित करें कि Python इंस्टॉल है; SimpleTuner 3.10 से 3.12 के साथ अच्छा काम करता है।

आप इसे चलाकर जांच सकते हैं:

```bash
python --version
```

यदि आपके Ubuntu पर Python 3.12 इंस्टॉल नहीं है, तो आप यह प्रयास कर सकते हैं:

```bash
apt -y install python3.12 python3.12-venv
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
```

मैनुअल इंस्टॉलेशन या डेवलपमेंट सेटअप के लिए, [installation documentation](../INSTALL.md) देखें।

#### AMD ROCm follow‑up steps

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

फिर, आपको निम्न वेरिएबल्स बदलने होंगे:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "model_type": "lora",
  "model_family": "sd3",
  "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3.5-large",
  "output_dir": "/home/user/outputs/models",
  "validation_resolution": "1024x1024,1280x768",
  "validation_guidance": 3.0,
  "validation_prompt": "your main test prompt here",
  "user_prompt_library": "config/user_prompt_library.json"
}
```
</details>

- `pretrained_model_name_or_path` - इसे `stabilityai/stable-diffusion-3.5-large` पर सेट करें। ध्यान दें कि इस मॉडल को डाउनलोड करने के लिए आपको Huggingface पर लॉग‑इन और एक्सेस की आवश्यकता होगी।
  - यदि आप पुराने SD3.0 Medium (2B) को ट्रेन करना चाहते हैं, तो `stabilityai/stable-diffusion-3-medium-diffusers` उपयोग करें।
- `MODEL_TYPE` - इसे `lora` पर सेट करें।
- `MODEL_FAMILY` - इसे `sd3` पर सेट करें।
- `OUTPUT_DIR` - इसे उस डायरेक्टरी पर सेट करें जहाँ आप अपने checkpoints और validation images रखना चाहते हैं। यहाँ full path उपयोग करने की सलाह है।
- `VALIDATION_RESOLUTION` - SD3 1024px मॉडल है, इसलिए इसे `1024x1024` पर सेट करें।
  - साथ ही, SD3 को multi‑aspect buckets पर fine‑tune किया गया था, और अन्य resolutions को कॉमा से अलग कर के दिया जा सकता है: `1024x1024,1280x768`
- `VALIDATION_GUIDANCE` - SD3 को बहुत कम मान लाभ देता है। इसे `3.0` पर सेट करें।

यदि आप Mac M‑series मशीन उपयोग कर रहे हैं तो:

- `mixed_precision` को `no` पर सेट करें।

### उन्नत प्रयोगात्मक विशेषताएँ

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में प्रयोगात्मक फीचर्स शामिल हैं जो प्रशिक्षण की स्थिरता और प्रदर्शन को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बढ़ाता है, क्योंकि यह प्रशिक्षण के दौरान मॉडल को अपने इनपुट्स खुद जनरेट करने देता है।

> ⚠️ ये फीचर्स प्रशिक्षण के कंप्यूटेशनल ओवरहेड को बढ़ाते हैं।

#### Quantised model training

Apple और NVIDIA सिस्टम्स पर टेस्ट किया गया है; Hugging Face Optimum‑Quanto base SDXL training की आवश्यकताओं से काफी कम precision/VRAM पर SD3 को चला सकता है।

> ⚠️ यदि आप JSON config फ़ाइल उपयोग कर रहे हैं, तो `config.env` की बजाय `config.json` में यह फ़ॉर्मैट उपयोग करें:

```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "text_encoder_3_precision": "no_change",
  "optimizer": "adamw_bf16"
}
```

`config.env` उपयोगकर्ताओं (deprecated) के लिए:

```bash
</details>

# choices: int8-quanto, int4-quanto, int2-quanto, fp8-quanto
# int8-quanto was tested with a single subject dreambooth LoRA.
# fp8-quanto does not work on Apple systems. you must use int levels.
# int2-quanto is pretty extreme and gets the whole rank-1 LoRA down to about 13.9GB VRAM.
# may the gods have mercy on your soul, should you push things Too Far.
export TRAINER_EXTRA_ARGS="--base_model_precision=int8-quanto"

# Maybe you want the text encoders to remain full precision so your text embeds are cake.
# We unload the text encoders before training, so, that's not an issue during training time - only during pre-caching.
# Alternatively, you can go ham on quantisation here and run them in int4 or int8 mode, because no one can stop you.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --text_encoder_1_precision=no_change --text_encoder_2_precision=no_change"

# When you're quantising the model, --base_model_default_dtype is set to bf16 by default. This setup requires adamw_bf16, but saves the most memory.
# adamw_bf16 only supports bf16 training, but any other optimiser will support both bf16 or fp32 training precision.
export OPTIMIZER="adamw_bf16"
```

#### डेटासेट विचार

अपने मॉडल को प्रशिक्षित करने के लिए पर्याप्त बड़ा डेटासेट होना महत्वपूर्ण है। डेटासेट आकार पर सीमाएँ हैं, और आपको सुनिश्चित करना होगा कि आपका डेटासेट पर्याप्त बड़ा हो। ध्यान दें कि न्यूनतम डेटासेट आकार `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS` के साथ‑साथ `VAE_BATCH_SIZE` से भी अधिक होना चाहिए। यदि डेटासेट बहुत छोटा है, तो वह उपयोग योग्य नहीं होगा।

आपके डेटासेट के अनुसार, आपको डेटासेट डायरेक्टरी और dataloader configuration फ़ाइल अलग तरीके से सेट करनी होगी। इस उदाहरण में हम [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) डेटासेट उपयोग करेंगे।

अपने `/home/user/simpletuner/config` डायरेक्टरी में, एक multidatabackend.json बनाएँ:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
[
  {
    "id": "pseudo-camera-10k-sd3",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 0,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "/home/user/simpletuner/output/cache/vae/sd3/pseudo-camera-10k",
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
    "cache_dir": "cache/text/sd3/pseudo-camera-10k",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> `caption_strategy` विकल्प और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

फिर, `datasets` डायरेक्टरी बनाएँ:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
popd
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

## नोट्स और समस्या‑समाधान टिप्स

### Skip‑layer guidance (SD3.5 Medium)

StabilityAI SD 3.5 Medium inference पर SLG (Skip‑layer guidance) सक्षम करने की सलाह देता है। यह training परिणामों को प्रभावित नहीं करता, केवल validation sample गुणवत्ता को प्रभावित करता है।

`config.json` के लिए अनुशंसित मान:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "--validation_guidance_skip_layers": [7, 8, 9],
  "--validation_guidance_skip_layers_start": 0.01,
  "--validation_guidance_skip_layers_stop": 0.2,
  "--validation_guidance_skip_scale": 2.8,
  "--validation_guidance": 4.0,
  "--flow_use_uniform_schedule": true,
  "--flow_schedule_auto_shift": true
}
```
</details>

- `..skip_scale` बताता है कि skip‑layer guidance के दौरान positive prompt prediction को कितना scale करना है। 2.8 का डिफ़ॉल्ट मान base मॉडल के skip मान `7, 8, 9` के लिए सुरक्षित है, लेकिन यदि आप अधिक layers skip करते हैं तो इसे बढ़ाना होगा (हर अतिरिक्त layer पर दोगुना)।
- `..skip_layers` बताता है कि negative prompt prediction के दौरान कौन‑सी layers skip करनी हैं।
- `..skip_layers_start` यह निर्धारित करता है कि inference pipeline के किस हिस्से में skip‑layer guidance लागू होनी चाहिए।
- `..skip_layers_stop` कुल inference steps के किस हिस्से के बाद SLG बंद हो जाएगा, यह सेट करता है।

SLG कम steps के लिए लागू कर सकते हैं जिससे प्रभाव कमजोर होगा या inference गति में कम कमी आएगी।

ऐसा लगता है कि LoRA या LyCORIS मॉडल के व्यापक प्रशिक्षण में इन मानों को बदलने की जरूरत होगी, हालांकि यह स्पष्ट नहीं कि कैसे।

**Inference में कम CFG उपयोग करना आवश्यक है।**

### मॉडल अस्थिरता

SD 3.5 Large 8B मॉडल में training के दौरान संभावित अस्थिरताएँ होती हैं:

- उच्च `--max_grad_norm` मान मॉडल को जोखिम भरे weight updates खोजने देते हैं
- Learning rate बहुत संवेदनशील हो सकता है; `1e-5` StableAdamW के साथ काम करता है, लेकिन `4e-5` अस्थिर हो सकता है
- बड़े batch sizes **बहुत** मदद करते हैं
- quantisation बंद करने या शुद्ध fp32 training से स्थिरता प्रभावित नहीं होती

SD3.5 के साथ आधिकारिक training code जारी नहीं हुआ, जिससे डेवलपर्स को [SD3.5 repository](https://github.com/stabilityai/sd3.5) के contents के आधार पर training loop का अनुमान लगाना पड़ा।

SimpleTuner के SD3.5 समर्थन में कुछ बदलाव किए गए:
- quantisation से अधिक layers को बाहर रखना
- T5 padding space को डिफ़ॉल्ट रूप से अब zero न करना (`--t5_padding`)
- unconditional predictions के लिए encoded blank captions उपयोग करने (`empty_string`, **default**) या zeros (`zero`) चुनने के लिए switch (`--sd3_clip_uncond_behaviour` और `--sd3_t5_uncond_behaviour`); इसे छेड़ना अनुशंसित नहीं है
- SD3.5 training loss function को upstream StabilityAI/SD3.5 repo के अनुसार अपडेट करना
- SD3 के static 1024px मान से मेल खाने के लिए डिफ़ॉल्ट `--flow_schedule_shift` को 3 पर अपडेट करना
  - StabilityAI ने `--flow_schedule_shift=1` को `--flow_use_uniform_schedule` के साथ उपयोग करने का दस्तावेज़ दिया है
  - Community सदस्यों ने बताया है कि multi‑aspect या multi‑resolution training में `--flow_schedule_auto_shift` बेहतर काम करता है
- hard‑coded tokenizer sequence length limit को **154** तक बढ़ाना, और इसे **77** tokens तक वापस करने का विकल्प देना ताकि डिस्क स्पेस/compute बच सके (गुणवत्ता में कमी के बदले)

#### Stable configuration values

ये विकल्प SD3.5 को अधिक समय तक स्थिर रखने में मदद करते हैं:
- optimizer=adamw_bf16
- flow_schedule_shift=1
- learning_rate=1e-4
- batch_size=4 * 3 GPUs
- max_grad_norm=0.1
- base_model_precision=int8-quanto
- No loss masking or dataset regularisation, as their contribution to this instability is unknown
- `validation_guidance_skip_layers=[7,8,9]`

### सबसे कम VRAM कॉन्फ़िग

- OS: Ubuntu Linux 24
- GPU: एक NVIDIA CUDA डिवाइस (10G, 12G)
- System memory: लगभग 50G सिस्टम मेमोरी
- Base model precision: `nf4-bnb`
- Optimiser: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolution: 512px
- Batch size: 1, zero gradient accumulation steps
- DeepSpeed: बंद/कॉन्फ़िगर नहीं
- PyTorch: 2.5

### SageAttention

`--attention_mechanism=sageattention` उपयोग करने पर validation समय पर inference तेज़ हो सकता है।

**नोट**: यह हर मॉडल कॉन्फ़िगरेशन के साथ संगत नहीं है, लेकिन कोशिश करने लायक है।

### Masked loss

यदि आप किसी subject या style को ट्रेन कर रहे हैं और इनमें से किसी को mask करना चाहते हैं, तो Dreambooth गाइड के [masked loss training](../DREAMBOOTH.md#masked-loss) सेक्शन देखें।

### Regularisation data

regularisation datasets के बारे में अधिक जानकारी के लिए Dreambooth गाइड के [इस सेक्शन](../DREAMBOOTH.md#prior-preservation-loss) और [इस सेक्शन](../DREAMBOOTH.md#regularisation-dataset-considerations) को देखें।

### Quantised training

SD3 और अन्य मॉडलों के लिए quantisation कॉन्फ़िगरेशन हेतु Dreambooth गाइड के [इस सेक्शन](../DREAMBOOTH.md#quantised-model-training-loralycoris-only) को देखें।

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
