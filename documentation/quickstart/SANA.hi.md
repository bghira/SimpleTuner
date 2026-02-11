## NVLabs Sana क्विकस्टार्ट

इस उदाहरण में, हम NVLabs Sana मॉडल का full‑rank प्रशिक्षण करेंगे।

### हार्डवेयर आवश्यकताएँ

Sana बहुत हल्का है और 24G कार्ड पर शायद full gradient checkpointing की भी जरूरत न पड़े, जिसका मतलब है कि यह बहुत जल्दी ट्रेन होता है!

- **पूर्ण न्यूनतम** लगभग 12G VRAM है, हालांकि यह गाइड आपको पूरी तरह वहाँ तक पहुँचाने में मदद नहीं कर सकती
- **यथार्थवादी न्यूनतम** एक 3090 या V100 GPU है
- **आदर्श रूप से** कई 4090, A6000, L40S, या बेहतर

Sana की आर्किटेक्चर अन्य SimpleTuner‑trainable मॉडलों से अलग है;

- शुरुआत में, अन्य मॉडलों के विपरीत, Sana को fp16 प्रशिक्षण चाहिए था और bf16 पर क्रैश होता था
  - NVIDIA के मॉडल लेखकों ने bf16‑compatible weights प्रदान किए ताकि fine‑tuning संभव हो सके
- bf16/fp16 समस्याओं के कारण quantisation इस परिवार में अधिक संवेदनशील हो सकता है
- Sana के head_dim आकार के कारण SageAttention अभी काम नहीं करता (असमर्थित है)
- Sana training का loss value बहुत ऊँचा होता है और इसे अन्य मॉडलों की तुलना में बहुत कम learning rate चाहिए हो सकता है (जैसे `1e-5` के आसपास)
- प्रशिक्षण में NaN values आ सकती हैं, और इसका कारण स्पष्ट नहीं है

Gradient checkpointing VRAM बचा सकता है, लेकिन training धीमी होती है। 4090 + 5800X3D पर परीक्षण परिणामों का चार्ट:

![image](https://github.com/user-attachments/assets/310bf099-a077-4378-acf4-f60b4b82fdc4)

SimpleTuner का Sana modeling code `--gradient_checkpointing_interval` निर्दिष्ट करने देता है ताकि हर _n_ blocks पर checkpoint हो और ऊपर चार्ट जैसे परिणाम मिलें।

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
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
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

वहाँ आपको संभवतः निम्न वेरिएबल्स बदलने होंगे:

- `model_type` - इसे `full` पर सेट करें।
- `model_family` - इसे `sana` पर सेट करें।
- `pretrained_model_name_or_path` - इसे `terminusresearch/sana-1.6b-1024px` पर सेट करें।
- `output_dir` - इसे उस डायरेक्टरी पर सेट करें जहाँ आप अपने checkpoints और validation images रखना चाहते हैं। यहाँ full path उपयोग करने की सलाह है।
- `train_batch_size` - 24G कार्ड पर full gradient checkpointing के साथ यह 6 तक हो सकता है।
- `validation_resolution` - Sana का यह checkpoint 1024px मॉडल है; इसे `1024x1024` या अन्य समर्थित resolutions पर सेट करें।
  - अन्य resolutions को कॉमा से अलग कर सकते हैं: `1024x1024,1280x768,2048x2048`
- `validation_guidance` - inference पर जिस मान के साथ आप सहज हों, वही रखें।
- `validation_num_inference_steps` - सर्वोत्तम गुणवत्ता के लिए लगभग 50 रखें, लेकिन परिणाम अच्छे हों तो कम भी स्वीकार्य है।
- `use_ema` - इसे `true` सेट करने से मुख्य trained checkpoint के साथ अधिक स्मूद परिणाम मिलते हैं।

- `optimizer` - आप कोई भी optimiser उपयोग कर सकते हैं जिसे आप जानते हों, लेकिन इस उदाहरण में हम `optimi-adamw` इस्तेमाल करेंगे।
- `mixed_precision` - सबसे कुशल training के लिए `bf16` अनुशंसित है, या बेहतर परिणामों के लिए `no` (लेकिन मेमोरी अधिक खपत होगी और धीमा रहेगा)।
  - यहाँ `fp16` की सिफारिश नहीं है, लेकिन कुछ Sana finetunes के लिए जरूरी हो सकता है (और इसे सक्षम करने में अन्य समस्याएँ आती हैं)
- `gradient_checkpointing` - इसे बंद करने पर सबसे तेज़ होगा, लेकिन batch sizes सीमित होंगे। सबसे कम VRAM उपयोग के लिए इसे सक्षम रखना आवश्यक है।
- `gradient_checkpointing_interval` - यदि `gradient_checkpointing` आपके GPU पर ज्यादा लगता है, तो इसे 2 या अधिक पर सेट करें ताकि हर _n_ blocks पर ही checkpoint हो। 2 का मतलब आधे blocks, 3 का मतलब एक‑तिहाई।

Multi‑GPU उपयोगकर्ता उपयोग किए जाने वाले GPU की संख्या कॉन्फ़िगर करने के लिए [इस दस्तावेज़](../OPTIONS.md#environment-configuration-variables) को देखें।

### उन्नत प्रयोगात्मक विशेषताएँ

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में प्रयोगात्मक फीचर्स शामिल हैं जो प्रशिक्षण की स्थिरता और प्रदर्शन को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बढ़ाता है, क्योंकि यह प्रशिक्षण के दौरान मॉडल को अपने इनपुट्स खुद जनरेट करने देता है।

> ⚠️ ये फीचर्स प्रशिक्षण के कंप्यूटेशनल ओवरहेड को बढ़ाते हैं।

#### वैलिडेशन प्रॉम्प्ट्स

`config/config.json` के अंदर "primary validation prompt" होता है, जो आमतौर पर आपके single subject या style के लिए मुख्य instance_prompt होता है। इसके अतिरिक्त, एक JSON फ़ाइल बनाई जा सकती है जिसमें वैलिडेशन के दौरान चलाने के लिए अतिरिक्त प्रॉम्प्ट्स हों।

उदाहरण config फ़ाइल `config/user_prompt_library.json.example` का फ़ॉर्मैट:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

nicknames validation के लिए फ़ाइलनाम होते हैं, इसलिए इन्हें छोटा और फ़ाइलसिस्टम‑अनुकूल रखें।

ट्रेनर को इस prompt library की ओर इंगित करने के लिए, `config.json` के अंत में नया लाइन जोड़कर इसे TRAINER_EXTRA_ARGS में जोड़ें:
```json
  "--user_prompt_library": "config/user_prompt_library.json",
```

विविध प्रॉम्प्ट्स का सेट यह निर्धारित करने में मदद करेगा कि मॉडल प्रशिक्षण के दौरान collapse तो नहीं हो रहा। इस उदाहरण में `<token>` शब्द को अपने subject नाम (instance_prompt) से बदलें।

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

> ℹ️ Sana एक अजीब text encoder कॉन्फ़िगरेशन उपयोग करता है, जिसका अर्थ है कि छोटे prompts संभवतः बहुत खराब दिख सकते हैं।

#### CLIP score ट्रैकिंग

यदि आप मॉडल प्रदर्शन स्कोर करने के लिए evaluations सक्षम करना चाहते हैं, तो CLIP scores को कॉन्फ़िगर और इंटरप्रेट करने के लिए [यह दस्तावेज़](../evaluation/CLIP_SCORES.md) देखें।

</details>

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

#### Sana time schedule shifting

Sana, Flux, और SD3 जैसे flow‑matching मॉडलों में "shift" नाम का गुण होता है जो हमें एक साधारण decimal value से timestep schedule के प्रशिक्षित हिस्से को शिफ्ट करने देता है।

##### Auto‑shift

आम तौर पर अनुशंसित तरीका यह है कि हाल के कई कार्यों का पालन करते हुए resolution‑dependent timestep shift सक्षम किया जाए, `--flow_schedule_auto_shift`, जो बड़े images के लिए उच्च shift मान और छोटे images के लिए कम shift मान उपयोग करता है। यह स्थिर लेकिन संभवतः औसत प्रशिक्षण परिणाम दे सकता है।

##### Manual specification

_Discord के General Awareness का इन उदाहरणों के लिए धन्यवाद_

`--flow_schedule_shift` का मान 0.1 (बहुत कम) रखने पर केवल इमेज के सूक्ष्म विवरण प्रभावित होते हैं:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

`--flow_schedule_shift` का मान 4.0 (बहुत अधिक) रखने पर बड़े compositional features और संभवतः मॉडल का colour space प्रभावित हो सकता है:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)

#### डेटासेट विचार

> ⚠️ Sana के लिए training इमेज गुणवत्ता अधिकांश अन्य मॉडलों से अधिक महत्वपूर्ण है, क्योंकि यह आपके images के artifacts को *पहले* अवशोषित करेगा, और फिर concept/subject सीखेगा।

अपने मॉडल को प्रशिक्षित करने के लिए पर्याप्त बड़ा डेटासेट होना महत्वपूर्ण है। डेटासेट आकार पर सीमाएँ हैं, और आपको सुनिश्चित करना होगा कि आपका डेटासेट पर्याप्त बड़ा हो। ध्यान दें कि न्यूनतम डेटासेट आकार `train_batch_size * gradient_accumulation_steps` के साथ-साथ `vae_batch_size` से भी अधिक होना चाहिए। यदि डेटासेट बहुत छोटा है, तो वह उपयोग योग्य नहीं होगा।

> ℹ️ बहुत कम images होने पर आपको **no images detected in dataset** संदेश दिख सकता है — `repeats` मान बढ़ाना इस सीमा को पार करेगा।

आपके डेटासेट के अनुसार, आपको डेटासेट डायरेक्टरी और dataloader configuration फ़ाइल अलग तरीके से सेट करनी होगी। इस उदाहरण में हम [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) डेटासेट उपयोग करेंगे।

एक `--data_backend_config` (`config/multidatabackend.json`) दस्तावेज़ बनाएँ जिसमें यह हो:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
[
  {
    "id": "pseudo-camera-10k-sana",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/sana/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/sana/dreambooth-subject",
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
    "cache_dir_vae": "cache/vae/sana/dreambooth-subject-512",
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
    "cache_dir": "cache/text/sana",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> `caption_strategy` विकल्प और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

> ℹ️ 512px और 1024px datasets को साथ चलाना समर्थित है, और इससे Sana के लिए बेहतर convergence हो सकता है।

फिर, `datasets` डायरेक्टरी बनाएँ:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

यह लगभग 10k फोटोग्राफ सैंपल्स को आपकी `datasets/pseudo-camera-10k` डायरेक्टरी में डाउनलोड करेगा, जो अपने‑आप बन जाएगी।

आपकी Dreambooth images को `datasets/dreambooth-subject` डायरेक्टरी में जाना चाहिए।

#### WandB और Huggingface Hub में लॉग‑इन

प्रशिक्षण शुरू करने से पहले WandB और HF Hub में लॉग‑इन करना बेहतर है, खासकर यदि आप `--push_to_hub` और `--report_to=wandb` उपयोग कर रहे हैं।

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

SimpleTuner डायरेक्टरी से, प्रशिक्षण शुरू करने के लिए आपके पास कई विकल्प हैं:

**विकल्प 1 (अनुशंसित - pip install):**
```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
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

## नोट्स और समस्या‑समाधान टिप्स

### सबसे कम VRAM कॉन्फ़िग

वर्तमान में सबसे कम VRAM उपयोग निम्न के साथ संभव है:

- OS: Ubuntu Linux 24
- GPU: एक NVIDIA CUDA डिवाइस (10G, 12G)
- System memory: लगभग 50G सिस्टम मेमोरी
- Base model precision: `nf4-bnb`
- Optimiser: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolution: 1024px
- Batch size: 1, gradient accumulation steps शून्य
- DeepSpeed: बंद/कॉन्फ़िगर नहीं
- <=16G कार्ड्स पर startup के दौरान outOfMemory त्रुटि से बचने के लिए `--quantize_via=cpu` उपयोग करें।
- `--gradient_checkpointing` सक्षम करें

**नोट**: VAE embeds और text encoder outputs की pre‑caching अधिक मेमोरी ले सकती है और फिर भी OOM हो सकता है। ऐसा हो तो text encoder quantisation सक्षम किया जा सकता है। VAE tiling फिलहाल Sana के लिए काम नहीं कर सकता। बड़े डेटासेट्स में जहाँ डिस्क स्पेस चिंता का विषय हो, आप `--vae_cache_disable` का उपयोग करके बिना डिस्क कैश के online encoding कर सकते हैं।

4090 पर गति लगभग 1.4 iterations per second थी।

### Masked loss

यदि आप किसी subject या style को ट्रेन कर रहे हैं और इनमें से किसी को mask करना चाहते हैं, तो Dreambooth गाइड के [masked loss training](../DREAMBOOTH.md#masked-loss) सेक्शन देखें।

### Quantisation

अभी तक अच्छी तरह से परीक्षण नहीं हुआ है।

### Learning rates

#### LoRA (--lora_type=standard)

*समर्थित नहीं है।*

#### LoKr (--lora_type=lycoris)
- LoKr के लिए हल्के learning rates बेहतर हैं (`1e-4` AdamW के साथ, `2e-5` Lion के साथ)
- अन्य algo के लिए अधिक exploration की आवश्यकता है।
- `is_regularisation_data` का Sana के साथ प्रभाव अज्ञात है (टेस्ट नहीं हुआ)

### Image artifacts

Sana का image artifacts पर response अज्ञात है।

यह फिलहाल ज्ञात नहीं है कि कोई सामान्य training artifacts बनेंगे या उनके कारण क्या होंगे।

यदि कोई image quality समस्या आए, तो Github पर issue खोलें।

### Aspect bucketing

इस मॉडल का aspect bucketed डेटा पर response अज्ञात है। experimentation मददगार होगी।
