# Mixture-of-Experts

SimpleTuner प्रशिक्षण कार्य को दो हिस्सों में बाँटने की सुविधा देता है, ताकि inference के self‑attention और cross‑attention चरणों को पूरी तरह अलग‑अलग वेट्स सेट्स के बीच प्रभावी रूप से बाँटा जा सके।

इस उदाहरण में, हम SegMind और Hugging Face की सहयोगी पहल [SSD-1B](https://huggingface.co/segmind/ssd-1b) का उपयोग करेंगे ताकि दो नए मॉडल बनाए जा सकें, जो एक अकेले मॉडल की तुलना में अधिक विश्वसनीय रूप से ट्रेन हों और बेहतर fine details दें।

SSD-1B मॉडल के छोटे आकार के कारण, हल्के हार्डवेयर पर भी ट्रेनिंग संभव है। चूँकि हम Segmind के pretrained weights से शुरू कर रहे हैं, हमें उनके Apache 2.0 लाइसेंस का पालन करना होगा — लेकिन यह अपेक्षाकृत सरल है। आप इन वेट्स को commercial सेटिंग में भी उपयोग कर सकते हैं!

जब SDXL 0.9 और 1.0 पेश किए गए, दोनों में split‑schedule refiner के साथ एक full base model था।

- base model को 999 से 0 steps पर ट्रेन किया गया
  - base model 3B से अधिक पैरामीटर्स का है और पूरी तरह standalone काम करता है।
- refiner model को 199 से 0 steps पर ट्रेन किया गया
  - refiner model भी 3B से अधिक पैरामीटर्स का है, जो संसाधनों की अनावश्यक बर्बादी जैसा लगता है। यह अकेले अपने दम पर काम नहीं करता और बिना पर्याप्त deforming के आउटपुट्स कार्टून‑जैसे लगते हैं।

आइए देखें कि हम इस स्थिति को कैसे बेहतर बना सकते हैं।


## Base मॉडल, "Stage One"

mixture‑of‑experts का पहला हिस्सा base model कहलाता है। जैसा कि SDXL के मामले में बताया गया, इसे सभी 1000 timesteps पर ट्रेन किया जाता है — लेकिन इसकी ज़रूरत नहीं है। निम्न कॉन्फ़िगरेशन base model को कुल 1000 में से सिर्फ़ 650 steps पर ट्रेन करेगा, समय बचाएगा और अधिक विश्वसनीय रूप से ट्रेन करेगा।

### Environment कॉन्फ़िगरेशन

`config/config.env` में निम्न मान सेट करने से शुरुआत हो जाएगी:

```bash
# Ensure these aren't incorrectly set.
export USE_BITFIT=false
export USE_DORA=false
# lora could be used here instead, but the concept hasn't been explored.
export MODEL_TYPE="full"
export MODEL_FAMILY="sdxl"
export MODEL_NAME="segmind/SSD-1B"
# The original Segmind model used a learning rate of 1e-5, which is
# probably too high for whatever batch size most users can pull off.
export LEARNING_RATE=4e-7

# We really want this as high as you can tolerate.
# - If training is very slow, ensure your CHECKPOINT_STEPS and VALIDATION_STEPS
#   are set low enough that you'll get a checkpoint every couple hours.
# - The original Segmind models used a batch size of 32 with 4 accumulations.
export TRAIN_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=1

# If you are running on a beefy machine that doesn't fully utilise its VRAM during training, set this to "false" and your training will go faster.
export USE_GRADIENT_CHECKPOINTING=true

# Enable first stage model training
export TRAINER_EXTRA_ARGS="--refiner_training --refiner_training_strength=0.35 --refiner_training_invert_schedule"

# Optionally reparameterise it to v-prediction/zero-terminal SNR. 'sample' prediction_type can be used instead for x-prediction.
# This will start out looking pretty terrible until about 1500-2500 steps have passed, but it could be very worthwhile.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --prediction_type=v_prediction --rescale_betas_zero_snr --training_scheduler_timestep_spacing=trailing"
```

### Dataloader कॉन्फ़िगरेशन

Dataloader कॉन्फ़िगरेशन के लिए कोई विशेष बदलाव आवश्यक नहीं है। इस चरण के लिए [dataloader config गाइड](DATALOADER.md) देखें।

### Validation

फिलहाल, SimpleTuner stage one evaluations के दौरान stage two मॉडल को शामिल नहीं करता।

भविष्य में इसे एक विकल्प के रूप में सपोर्ट किया जाएगा, जब stage two मॉडल पहले से मौजूद हो या साथ‑साथ ट्रेन किया जा रहा हो।

---

## Refiner मॉडल, "Stage Two"

### SDXL refiner ट्रेनिंग से तुलना

- SDXL refiner के विपरीत, Segmind SSD‑1B को दोनों स्टेज के लिए उपयोग करने पर टेक्स्ट embeds **shared** किए जा सकते हैं
  - SDXL refiner का text embed layout SDXL base model से अलग होता है।
- VAE embeds **shared** किए जा सकते हैं, ठीक SDXL refiner की तरह। दोनों मॉडल समान input layout उपयोग करते हैं।
- Segmind मॉडल्स में aesthetic score उपयोग नहीं होता; वे SDXL की तरह वही microconditioning inputs उपयोग करते हैं, जैसे crop coordinates
- प्रशिक्षण काफ़ी तेज़ होता है, क्योंकि मॉडल छोटा है और text embeds stage one से reuse हो सकते हैं

### Environment कॉन्फ़िगरेशन

stage two मॉडल पर प्रशिक्षण स्विच करने के लिए `config/config.env` में निम्न मान अपडेट करें। base मॉडल कॉन्फ़िगरेशन की एक कॉपी रखना उपयोगी हो सकता है।

```bash
# Update your OUTPUT_DIR value, so that we don't overwrite the stage one model checkpoints.
export OUTPUT_DIR="/some/new/path"

# We'll swap --refiner_training_invert_schedule for --validation_using_datasets
# - Train the end of the model instead of the beginning
# - Validate using images as input for partial denoising to evaluate fine detail improvements
export TRAINER_EXTRA_ARGS="--refiner_training --refiner_training_strength=0.35 --validation_using_datasets"

# Don't update these values if you've set them on the stage one. Be sure to use the same parameterisation for both models!
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --prediction_type=v_prediction --rescale_betas_zero_snr --training_scheduler_timestep_spacing=trailing"
```

### Dataset फ़ॉर्मैट

इमेजेस केवल उच्च‑गुणवत्ता वाली हों — compression artifacts या अन्य त्रुटियों वाले datasets को हटा दें।

इसके अलावा, दोनों प्रशिक्षण जॉब्स के बीच बिल्कुल वही dataloader कॉन्फ़िगरेशन उपयोग किया जा सकता है।

यदि आपको demonstration dataset चाहिए, तो permissive लाइसेंसिंग वाला [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) एक अच्छा विकल्प है।

### Validation

Stage two refiner training validation समय पर प्रत्येक training set से इमेजेस स्वतः चुनेगा और उन्हें partial denoising के लिए इनपुट की तरह उपयोग करेगा।

## CLIP score tracking

यदि आप मॉडल प्रदर्शन को स्कोर करने के लिए evaluations सक्षम करना चाहते हैं, तो CLIP scores कॉन्फ़िगर और interpret करने के लिए [यह दस्तावेज़](evaluation/CLIP_SCORES.md) देखें।

# स्थिर evaluation loss

यदि आप मॉडल प्रदर्शन को स्कोर करने के लिए स्थिर MSE loss उपयोग करना चाहते हैं, तो evaluation loss को कॉन्फ़िगर और interpret करने के लिए [यह दस्तावेज़](evaluation/EVAL_LOSS.md) देखें।

## Inference समय पर सब कुछ जोड़ना

यदि आप दोनों मॉडलों को एक साथ जोड़कर एक सरल स्क्रिप्ट में प्रयोग करना चाहते हैं, तो यह शुरुआत के लिए मदद करेगा:

```py
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, UniPCMultistepScheduler
from torch import float16, cuda
from torch.backends import mps

# For a training_refiner_strength of .35, you'll set the base model strength to 0.65.
# Formula: 1 - training_refiner_strength
training_refiner_strength = 0.35
base_model_power = 1 - training_refiner_strength
# Reduce this for lower quality but speed-up.
num_inference_steps = 40
# Update these to your local or hugging face hub paths.
stage_1_model_id = 'bghira/terminus-xl-velocity-v2'
stage_2_model_id = 'bghira/terminus-xl-refiner'
torch_device = 'cuda' if cuda.is_available() else 'mps' if mps.is_available() else 'cpu'

pipe = StableDiffusionXLPipeline.from_pretrained(stage_1_model_id, add_watermarker=False, torch_dtype=float16).to(torch_device)
pipe.scheduler = UniPCMultistepScheduler.from_pretrained(stage_1_model_id, subfolder="scheduler")
img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(stage_2_model_id).to(device=torch_device, dtype=float16)
img2img_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(stage_1_model_id, subfolder="scheduler")

prompt = "An astronaut riding a green horse"

# Important: update this to True if you reparameterised the models.
use_zsnr = True

image = pipe(
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    denoising_end=base_model_power,
    guidance_scale=9.2,
    guidance_rescale=0.7 if use_zsnr else 0.0,
    output_type="latent",
).images
image = img2img_pipe(
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    denoising_start=base_model_power,
    guidance_scale=9.2,
    guidance_rescale=0.7 if use_zsnr else 0.0,
    image=image,
).images[0]
image.save('demo.png', format="PNG")
```

कुछ प्रयोग जो आप कर सकते हैं:
- यहाँ `base_model_power` या `num_inference_steps` जैसे मानों के साथ खेलें, जिन्हें दोनों pipelines में समान होना चाहिए।
- आप `guidance_scale`, `guidance_rescale` के साथ भी प्रयोग कर सकते हैं; इन्हें प्रत्येक स्टेज के लिए अलग सेट किया जा सकता है। ये contrast और realism पर असर डालते हैं।
- base और refiner मॉडल्स के बीच अलग prompts उपयोग करके fine details के लिए guidance focus बदलें।
