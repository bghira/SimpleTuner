## Auraflow क्विकस्टार्ट

इस उदाहरण में, हम Auraflow के लिए एक Lycoris LoKr प्रशिक्षित करेंगे।

इस मॉडल के लिए full fine‑tuning में 6B पैरामीटर के कारण बहुत VRAM लगेगा, और इसे चलाने के लिए आपको [DeepSpeed](../DEEPSPEED.md) की आवश्यकता होगी।

### हार्डवेयर आवश्यकताएँ

Auraflow v0.3 एक 6B पैरामीटर MMDiT के रूप में रिलीज़ हुआ था, जो टेक्स्ट के encoded प्रतिनिधित्व के लिए Pile T5 और latent इमेज प्रतिनिधित्व के लिए 4ch SDXL VAE का उपयोग करता है।

यह मॉडल inference में कुछ धीमा है, लेकिन training अच्छी गति से करता है।

### मेमोरी ऑफ़लोडिंग (वैकल्पिक)

Auraflow को नया grouped offloading path काफी फायदा देता है। यदि आपके पास एक ही 24G (या उससे छोटा) GPU है, तो training flags में यह जोड़ें:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Streams गैर‑CUDA backends पर अपने‑आप बंद हो जाते हैं, इसलिए यह कमांड ROCm और MPS पर भी सुरक्षित है।
- इसे `--enable_model_cpu_offload` के साथ न मिलाएँ।
- Disk offloading throughput घटाकर host RAM दबाव कम करता है; बेहतर परिणाम के लिए लोकल SSD रखें।

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

- `model_type` - इसे `lora` पर सेट करें।
- `lora_type` - इसे `lycoris` पर सेट करें।
- `model_family` - इसे `auraflow` पर सेट करें।
- `model_flavour` - इसे `pony` पर सेट करें, या डिफ़ॉल्ट मॉडल उपयोग करने के लिए खाली छोड़ दें।
- `output_dir` - इसे उस डायरेक्टरी पर सेट करें जहाँ आप अपने checkpoints और validation images रखना चाहते हैं। यहाँ full path उपयोग करने की सलाह है।
- `train_batch_size` - 24G कार्ड के लिए 1 से 4 तक काम करना चाहिए।
- `validation_resolution` - इसे `1024x1024` या Auraflow की अन्य समर्थित resolutions में से किसी पर सेट करें।
  - अन्य resolutions को कॉमा से अलग कर सकते हैं: `1024x1024,1280x768,1536x1536`
  - ध्यान दें कि Auraflow के positional embeds थोड़े अजीब हैं और multi‑scale images (multiple base resolutions) के साथ training का परिणाम अनिश्चित है।
- `validation_guidance` - Auraflow inference पर जिसे आप चुनते हैं वही रखें; 3.5‑4.0 के आसपास का कम मान अधिक यथार्थवादी परिणाम देता है।
- `validation_num_inference_steps` - लगभग 30‑50 रखें।
- `use_ema` - इसे `true` सेट करने से मुख्य trained checkpoint के साथ अधिक स्मूद परिणाम मिलते हैं।

- `optimizer` - आप कोई भी optimiser उपयोग कर सकते हैं जिसे आप जानते हों, लेकिन इस उदाहरण में हम `optimi-lion` इस्तेमाल करेंगे।
  - Pony Flow के लेखक सबसे कम issues और स्थिर प्रशिक्षण के लिए `adamw_bf16` की सलाह देते हैं।
  - हम इस डेमो में Lion उपयोग कर रहे हैं ताकि मॉडल तेजी से ट्रेन होता दिखे, लेकिन लंबे रन के लिए `adamw_bf16` सुरक्षित विकल्प है।
- `learning_rate` - Lion optimiser के साथ Lycoris LoKr के लिए `4e-5` एक अच्छा शुरुआती मान है।
  - यदि आप `adamw_bf16` चुनते हैं, तो LR लगभग 10x बड़ा (`2.5e-4`) रखें।
  - छोटे Lycoris/LoRA ranks के लिए **उच्च learning rates** चाहिए और बड़े Lycoris/LoRA के लिए **कम learning rates**।
- `mixed_precision` - सर्वाधिक कुशल training के लिए `bf16` अनुशंसित है, या बेहतर परिणाम के लिए `no` (लेकिन मेमोरी अधिक खपत होगी और धीमा रहेगा)।
- `gradient_checkpointing` - इसे बंद करने पर सबसे तेज़ होगा, लेकिन batch sizes सीमित होंगे। सबसे कम VRAM उपयोग के लिए इसे सक्षम रखना आवश्यक है।

इन विकल्पों का प्रभाव अभी अज्ञात है।

अंत में आपका config.json लगभग ऐसा दिखेगा:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
    "validation_torch_compile": "false",
    "validation_step_interval": 200,
    "validation_seed": 42,
    "validation_resolution": "1024x1024",
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_num_inference_steps": "20",
    "validation_guidance": 2.0,
    "validation_guidance_rescale": "0.0",
    "vae_cache_ondemand": true,
    "vae_batch_size": 1,
    "train_batch_size": 1,
    "tracker_run_name": "eval_loss_test1",
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "output_dir": "output/models-auraflow",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "model_type": "lora",
    "model_family": "auraflow",
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 10000,
    "max_grad_norm": 0.01,
    "lycoris_config": "config/lycoris_config.json",
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant",
    "lora_type": "lycoris",
    "learning_rate": "4e-5",
    "gradient_checkpointing": "true",
    "grad_clip_method": "value",
    "eval_steps_interval": 100,
    "disable_benchmark": false,
    "data_backend_config": "config/auraflow/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "base_model_precision": "int8-quanto",
    "aspect_bucket_rounding": 2
}
```
</details>

> ℹ️ Multi‑GPU उपयोगकर्ता उपयोग किए जाने वाले GPU की संख्या कॉन्फ़िगर करने के लिए [इस दस्तावेज़](../OPTIONS.md#environment-configuration-variables) को देखें।

और एक सरल `config/lycoris_config.json` फ़ाइल:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 10000,
    "linear_alpha": 1,
    "factor": 16,
    "apply_preset": {
        "target_module": [
            "Attention",
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 8
            },
        }
    }
}
```
</details>

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

> ℹ️ Auraflow डिफ़ॉल्ट रूप से 128 tokens तक जाता है और फिर truncate करता है।

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

#### Flow schedule shifting

OmniGen, Sana, Flux, और SD3 जैसे flow‑matching मॉडलों में "shift" नाम का एक गुण होता है, जो हमें एक साधारण decimal value से timestep schedule के प्रशिक्षित हिस्से को शिफ्ट करने देता है।

##### Auto‑shift

आम तौर पर अनुशंसित तरीका यह है कि हाल के कई कार्यों का पालन करते हुए resolution‑dependent timestep shift सक्षम किया जाए, `--flow_schedule_auto_shift`, जो बड़े images के लिए उच्च shift मान और छोटे images के लिए कम shift मान उपयोग करता है। यह स्थिर लेकिन संभवतः औसत प्रशिक्षण परिणाम दे सकता है।

##### Manual specification

_Discord के General Awareness का इन उदाहरणों के लिए धन्यवाद_

`--flow_schedule_shift` का मान 0.1 (बहुत कम) रखने पर केवल इमेज के सूक्ष्म विवरण प्रभावित होते हैं:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

`--flow_schedule_shift` का मान 4.0 (बहुत अधिक) रखने पर बड़े compositional features और संभवतः मॉडल का colour space प्रभावित हो सकता है:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### डेटासेट विचार

अपने मॉडल को प्रशिक्षित करने के लिए पर्याप्त बड़ा डेटासेट होना महत्वपूर्ण है। डेटासेट आकार पर सीमाएँ हैं, और आपको सुनिश्चित करना होगा कि आपका डेटासेट पर्याप्त बड़ा हो। ध्यान दें कि न्यूनतम डेटासेट आकार `train_batch_size * gradient_accumulation_steps` के साथ-साथ `vae_batch_size` से भी अधिक होना चाहिए। यदि डेटासेट बहुत छोटा है, तो वह उपयोग योग्य नहीं होगा।

> ℹ️ बहुत कम images होने पर आपको **no images detected in dataset** संदेश दिख सकता है — `repeats` मान बढ़ाना इस सीमा को पार करेगा।

आपके डेटासेट के अनुसार, आपको डेटासेट डायरेक्टरी और dataloader configuration फ़ाइल अलग तरीके से सेट करनी होगी। इस उदाहरण में हम [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) डेटासेट उपयोग करेंगे।

एक `--data_backend_config` (`config/multidatabackend.json`) दस्तावेज़ बनाएँ जिसमें यह हो:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
[
  {
    "id": "pseudo-camera-10k-auraflow",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/auraflow/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/auraflow/dreambooth-subject",
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
    "cache_dir": "cache/text/auraflow",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> ℹ️ यदि आपके पास captions वाली `.txt` फ़ाइलें हैं तो `caption_strategy=textfile` उपयोग करें।
> `caption_strategy` विकल्प और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

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

### बाद में LoKr पर inference चलाना

क्योंकि यह नया मॉडल है, उदाहरण को काम करने के लिए कुछ समायोजन चाहिए। यहाँ एक काम करने वाला उदाहरण है:

<details>
<summary>Show Python inference example</summary>

```py
import torch
from helpers.models.auraflow.pipeline import AuraFlowPipeline
from helpers.models.auraflow.transformer import AuraFlowTransformer2DModel
from lycoris import create_lycoris_from_weights
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

model_id = 'terminusresearch/auraflow-v0.3'
adapter_repo_id = 'bghira/auraflow-photo-1mp-Prodigy'
adapter_filename = 'pytorch_lora_weights.safetensors'

def download_adapter(repo_id: str):
    import os
    from huggingface_hub import hf_hub_download
    adapter_filename = "pytorch_lora_weights.safetensors"
    cache_dir = os.environ.get('HF_PATH', os.path.expanduser('~/.cache/huggingface/hub/models'))
    cleaned_adapter_path = repo_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    path_to_adapter = os.path.join(cache_dir, cleaned_adapter_path)
    path_to_adapter_file = os.path.join(path_to_adapter, adapter_filename)
    os.makedirs(path_to_adapter, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id, filename=adapter_filename, local_dir=path_to_adapter
    )

    return path_to_adapter_file

adapter_file_path = download_adapter(repo_id=adapter_repo_id)
transformer = AuraFlowTransformer2DModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, subfolder="transformer")
pipeline = AuraFlowPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    transformer=transformer,
)
lora_scale = 1.0
wrapper, _ = create_lycoris_from_weights(lora_scale, adapter_file_path, pipeline.transformer)
wrapper.merge_to()

prompt = "Place your test prompt here."
negative_prompt = 'ugly, cropped, blurry, low-quality, mediocre average'

## Optional: quantise the model to save on vram.
## Note: The model was quantised during training, and so it is recommended to do the same during inference time.
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

pipeline.to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') # the pipeline is already in its target precision level
t5_embeds, negative_t5_embeds, attention_mask, negative_attention_mask = pipeline.encode_prompt(
    prompt=prompt, prompt_2=prompt, prompt_3=prompt, prompt_4=prompt, num_images_per_prompt=1
)
# We'll nuke the text encoders to save memory.
pipeline.text_encoder.to("meta")
pipeline.text_encoder_2.to("meta")
pipeline.text_encoder_3.to("meta")
model_output = pipeline(
    prompt_embeds=t5_embeds,
    prompt_attention_mask=attention_mask,
    negative_prompt_embeds=negative_t5_embeds,
    negative_prompt_attention_mask=negative_attention_mask,
    num_inference_steps=30,
    generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(42),
    width=1024,
    height=1024,
    guidance_scale=3.2,
).images[0]

model_output.save("output.png", format="PNG")

```
</details>

## नोट्स और समस्या‑समाधान टिप्स

### सबसे कम VRAM कॉन्फ़िग

सबसे कम VRAM वाला Auraflow कॉन्फ़िग लगभग 20‑22G है:

- OS: Ubuntu Linux 24
- GPU: एक NVIDIA CUDA डिवाइस (10G, 12G)
- सिस्टम मेमोरी: लगभग 50G सिस्टम मेमोरी (कम‑ज्यादा हो सकती है)
- Base model precision:
  - Apple और AMD सिस्टम के लिए, `int8-quanto` (या `fp8-torchao`, `int8-torchao`—सबकी मेमोरी प्रोफ़ाइल समान)
    - `int4-quanto` भी काम करता है, लेकिन accuracy/परिणाम कमजोर हो सकते हैं
  - NVIDIA सिस्टम के लिए, `nf4-bnb` अच्छा काम करता है, लेकिन `int8-quanto` से धीमा होगा
- Optimiser: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolution: 1024px
- Batch size: 1, gradient accumulation steps शून्य
- DeepSpeed: बंद/कॉन्फ़िगर नहीं
- PyTorch: 2.7+
- स्टार्टअप पर <=16G कार्ड्स में outOfMemory त्रुटि से बचने के लिए `--quantize_via=cpu` उपयोग करें।
- `--gradient_checkpointing` सक्षम करें
- बहुत छोटा LoRA या Lycoris कॉन्फ़िग उपयोग करें (जैसे LoRA rank 1 या Lokr factor 25)

**नोट**: VAE embeds और text encoder outputs की pre‑caching अधिक मेमोरी ले सकती है और फिर भी OOM हो सकता है। VAE tiling और slicing डिफ़ॉल्ट रूप से सक्षम हैं। यदि OOM दिखे, तो `offload_during_startup=true` सक्षम करना पड़ सकता है; अन्यथा संभव है कि संसाधन पर्याप्त न हों।

Pytorch 2.7 और CUDA 12.8 के साथ NVIDIA 4090 पर गति लगभग 3 iterations per second थी।

### Masked loss

यदि आप किसी subject या style को ट्रेन कर रहे हैं और इनमें से किसी को mask करना चाहते हैं, तो Dreambooth गाइड के [masked loss training](../DREAMBOOTH.md#masked-loss) सेक्शन देखें।

### Quantisation

Auraflow `int4` तक अच्छी तरह प्रतिक्रिया देता है, हालांकि यदि आप `bf16` नहीं चला सकते, तो गुणवत्ता और स्थिरता के लिए `int8` sweet spot रहेगा।

### Learning rates

#### LoRA (--lora_type=standard)

*समर्थित नहीं है।*

#### LoKr (--lora_type=lycoris)
- LoKr के लिए हल्के learning rates बेहतर हैं (`adamw` के साथ `1e-4`, Lion के साथ `2e-5`)
- अन्य algo के लिए अधिक exploration की आवश्यकता है।
- `is_regularisation_data` का Auraflow के साथ प्रभाव अज्ञात है (टेस्ट नहीं हुआ, लेकिन शायद ठीक हो?)

### इमेज artifacts

Auraflow का image artifacts पर response अज्ञात है, हालांकि यह Flux VAE का उपयोग करता है और fine‑details में समान सीमाएँ हैं।

यदि कोई image quality समस्या आए, तो Github पर issue खोलें।

### Aspect bucketing

मॉडल के patch embed implementation में कुछ सीमाएँ हैं, जिनके कारण कुछ resolutions त्रुटि पैदा कर सकते हैं।

प्रयोग‑आधारित जांच और विस्तृत bug reports मददगार होंगे।

### Full‑rank tuning

Auraflow के साथ DeepSpeed बहुत अधिक system memory उपयोग करेगा, और full tuning संभवतः आपके उम्मीद के अनुसार concepts सीखने या model collapse से बचने में काम न करे।

Full‑rank tuning के बजाय Lycoris LoKr अनुशंसित है, क्योंकि यह अधिक स्थिर है और कम मेमोरी footprint रखता है।
