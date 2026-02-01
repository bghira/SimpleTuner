# Dataloader कॉन्फ़िगरेशन फ़ाइल

यह `multidatabackend.example.json` के रूप में dataloader कॉन्फ़िगरेशन फ़ाइल का सबसे बेसिक उदाहरण है।

```json
[
  {
    "id": "something-special-to-remember-by",
    "type": "local",
    "instance_data_dir": "/path/to/data/tree",
    "crop": true,
    "crop_style": "center",
    "crop_aspect": "square",
    "resolution": 1024,
    "minimum_image_size": 768,
    "maximum_image_size": 2048,
    "minimum_aspect_ratio": 0.50,
    "maximum_aspect_ratio": 3.00,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "prepend_instance_prompt": false,
    "instance_prompt": "something to label every image",
    "only_instance_prompt": false,
    "caption_strategy": "textfile",
    "cache_dir_vae": "/path/to/vaecache",
    "repeats": 0
  },
  {
    "id": "an example backend for text embeds.",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "aws",
    "aws_bucket_name": "textembeds-something-yummy",
    "aws_region_name": null,
    "aws_endpoint_url": "https://foo.bar/",
    "aws_access_key_id": "wpz-764e9734523434",
    "aws_secret_access_key": "xyz-sdajkhfhakhfjd",
    "aws_data_prefix": "",
    "cache_dir": ""
  }
]
```

## Configuration Options

### `id`

- **Description:** डेटासेट का unique identifier। एक बार सेट होने के बाद इसे स्थिर रहना चाहिए, क्योंकि यह डेटासेट को उसके state tracking entries से जोड़ता है।

### `disabled`

- **Values:** `true` | `false`
- **Description:** `true` होने पर यह डेटासेट प्रशिक्षण के दौरान पूरी तरह skip किया जाता है। बिना कॉन्फ़िगरेशन हटाए अस्थायी रूप से किसी डेटासेट को बाहर रखने के लिए उपयोगी।
- **Note:** `disable` स्पेलिंग भी स्वीकार करता है।

### `dataset_type`

- **Values:** `image` | `video` | `audio` | `text_embeds` | `image_embeds` | `conditioning_image_embeds` | `conditioning`
- **Description:** `image`, `video`, और `audio` datasets मुख्य training samples रखते हैं। `text_embeds` में text encoder cache के outputs होते हैं, `image_embeds` में VAE latents (जब मॉडल VAE उपयोग करता है) होते हैं, और `conditioning_image_embeds` cached conditioning image embeddings (जैसे Wan 2.2 I2V के लिए CLIP vision features) स्टोर करते हैं। जब किसी dataset को `conditioning` के रूप में चिन्हित किया जाता है, तो उसे [conditioning_data option](#conditioning_data) के जरिए `image` dataset से जोड़ा जा सकता है।
- **Note:** Text और image embed datasets की परिभाषा image datasets से अलग होती है। Text embed dataset केवल text embed objects स्टोर करता है। Image dataset training data स्टोर करता है।
- **Note:** images और video को **एक ही** dataset में न मिलाएं। इन्हें अलग करें।

### `default`

- **केवल `dataset_type=text_embeds` पर लागू**
- `true` होने पर यह text embed dataset वह स्थान होगा जहाँ SimpleTuner validation prompt embeds जैसे text embed cache स्टोर करेगा। क्योंकि ये image data से pair नहीं होते, इनके लिए एक विशिष्ट लोकेशन चाहिए।

### `cache_dir`

- **केवल `dataset_type=text_embeds` और `dataset_type=image_embeds` पर लागू**
- **Description:** इस dataset के लिए embed cache files कहाँ स्टोर होंगी। `text_embeds` के लिए यही text encoder outputs लिखने की जगह है। `image_embeds` के लिए यही VAE latents स्टोर करने की जगह है।
- **Note:** यह `cache_dir_vae` से अलग है, जो primary image/video datasets पर VAE cache की जगह बताने के लिए सेट होता है।

### `write_batch_size`

- **केवल `dataset_type=text_embeds` पर लागू**
- **Description:** एक batch operation में कितने text embeds लिखे जाएँ। उच्च मान write throughput बढ़ा सकते हैं लेकिन अधिक memory लेते हैं।
- **Default:** trainer के `--write_batch_size` argument (आमतौर पर 128) पर fallback करता है।

### `text_embeds`

- **केवल `dataset_type=image` पर लागू**
- यदि unset हो, तो `default` text_embeds dataset उपयोग होगा। यदि किसी `text_embeds` dataset का `id` सेट किया गया है, तो वही उपयोग होगा। किसी image dataset को विशिष्ट text embed dataset से जोड़ने देता है।

### `image_embeds`

- **केवल `dataset_type=image` पर लागू**
- यदि unset हो, तो VAE outputs image backend पर स्टोर होंगे। अन्यथा, आप इसे किसी `image_embeds` dataset के `id` पर सेट कर सकते हैं और VAE outputs वहाँ स्टोर होंगे। Image data के साथ image_embed dataset को जोड़ने देता है।

### `conditioning_image_embeds`

- **`dataset_type=image` और `dataset_type=video` पर लागू**
- जब कोई मॉडल `requires_conditioning_image_embeds` रिपोर्ट करता है, तो cached conditioning image embeddings (उदाहरण: Wan 2.2 I2V के लिए CLIP vision features) स्टोर करने हेतु इसे किसी `conditioning_image_embeds` dataset के `id` पर सेट करें। यदि unset हो, तो SimpleTuner डिफ़ॉल्ट रूप से cache को `cache/conditioning_image_embeds/<dataset_id>` में लिखता है, जिससे यह VAE cache से टकराता नहीं है।
- जिन मॉडलों को इन embeds की आवश्यकता होती है, उन्हें अपनी primary pipeline के माध्यम से image encoder expose करना चाहिए। यदि मॉडल encoder प्रदान नहीं कर सकता, तो preprocessing जल्दी fail होगा, empty files silently generate करने की जगह।

#### `cache_dir_conditioning_image_embeds`

- **conditioning image embed cache destination के लिए वैकल्पिक override.**
- जब आप cache को किसी विशेष filesystem स्थान पर pin करना चाहते हैं या dedicated remote backend (`dataset_type=conditioning_image_embeds`) रखना चाहते हैं, तब इसे सेट करें। न सेट करने पर ऊपर बताया गया cache path स्वतः उपयोग होता है।

#### `conditioning_image_embed_batch_size`

- **conditioning image embeds बनाते समय batch size के लिए वैकल्पिक override.**
- यदि स्पष्ट रूप से न दिया जाए, तो `conditioning_image_embed_batch_size` trainer argument या VAE batch size डिफ़ॉल्ट होगा।

### Audio dataset configuration (`dataset_type=audio`)

Audio backends एक dedicated `audio` block सपोर्ट करते हैं ताकि metadata और bucket गणना duration‑aware रहे। उदाहरण:

```json
"audio": {
  "max_duration_seconds": 90,
  "channels": 2,
  "bucket_strategy": "duration",
  "duration_interval": 15,
  "truncation_mode": "beginning"
}
```

- **`bucket_strategy`** – फिलहाल `duration` डिफ़ॉल्ट है और clips को समान अंतराल वाले buckets में truncate करता है ताकि per‑GPU sampling batch गणना का सम्मान करे।
- **`duration_interval`** – seconds में bucket rounding (unset होने पर डिफ़ॉल्ट **3**)। `15` के साथ, 77s clip 75s bucket में जाएगा। यह single long clips को अन्य ranks को starve करने से रोकता है और truncation को समान interval पर मजबूर करता है।
- **`max_duration_seconds`** – इससे लंबे clips metadata discovery के दौरान पूरी तरह skip किए जाते हैं ताकि अत्यधिक लंबे tracks अनपेक्षित रूप से buckets न भरें।
- **`truncation_mode`** – bucket interval पर snap करते समय clip का कौन‑सा हिस्सा रखा जाए। विकल्प: `beginning`, `end`, या `random` (डिफ़ॉल्ट: `beginning`)।
- **`audio_only`** – केवल ऑडियो ट्रेनिंग मोड (LTX-2): वीडियो फाइलों के बिना केवल ऑडियो जेनरेशन ट्रेन करता है। वीडियो latents स्वचालित रूप से शून्य हो जाते हैं और वीडियो loss मास्क हो जाता है।
- **`target_resolution`** – केवल ऑडियो मोड के लिए लक्ष्य वीडियो resolution (latent dimensions की गणना के लिए उपयोग)।
- standard audio settings (channel count, cache directory) सीधे `simpletuner.helpers.data_backend.factory` द्वारा बनाए गए runtime audio backend पर मैप होते हैं। Padding जानबूझकर नहीं किया जाता—clips truncate होते हैं ताकि behavior ACE-Step जैसे diffusion trainers के साथ consistent रहे।

### Audio Captions (Hugging Face)
Hugging Face audio datasets के लिए, आप यह निर्दिष्ट कर सकते हैं कि कौन‑से columns caption (prompt) बनाएँ और कौन‑सा column lyrics रखता है:
```json
"config": {
    "audio_caption_fields": ["prompt", "tags"],
    "lyrics_column": "lyrics"
}
```
*   `audio_caption_fields`: कई columns को जोड़कर text prompt बनाता है (text encoder द्वारा उपयोग किया जाता है)।
*   `lyrics_column`: lyrics के column को निर्दिष्ट करता है (lyric encoder द्वारा उपयोग किया जाता है)।

Metadata discovery के दौरान loader प्रत्येक file के लिए `sample_rate`, `num_samples`, `num_channels`, और `duration_seconds` रिकॉर्ड करता है। CLI में bucket reports अब **samples** में बोलते हैं न कि **images** में, और empty‑dataset diagnostics active `bucket_strategy`/`duration_interval` (और किसी भी `max_duration_seconds` सीमा) को सूचीबद्ध करेंगे ताकि आप logs में जाए बिना intervals tune कर सकें।

### `type`

- **Values:** `aws` | `local` | `csv` | `huggingface`
- **Description:** इस dataset के लिए storage backend (local, csv या cloud) तय करता है।

### `conditioning_type`

- **Values:** `controlnet` | `mask` | `reference_strict` | `reference_loose`
- **Description:** `conditioning` dataset के लिए conditioning का प्रकार बताता है।
  - **controlnet**: ControlNet conditioning inputs for control signal training.
  - **mask**: inpainting training के लिए binary masks.
  - **reference_strict**: strict pixel alignment वाले reference images (Qwen Edit जैसे edit models के लिए)।
  - **reference_loose**: loose alignment वाले reference images.

### `source_dataset_id`

- **केवल `dataset_type=conditioning` पर लागू** और `conditioning_type` `reference_strict`, `reference_loose`, या `mask` हो
- **Description:** conditioning dataset को उसके source image/video dataset से pixel alignment के लिए जोड़ता है। सेट होने पर SimpleTuner source dataset से metadata duplicate करता है ताकि conditioning images उनके targets से align हों।
- **Note:** strict alignment modes के लिए आवश्यक; loose alignment के लिए वैकल्पिक।

### `conditioning_data`

- **Values:** conditioning dataset का `id` या `id` values का array
- **Description:** [ControlNet guide](CONTROLNET.md) में बताए अनुसार, `image` dataset को उसके ControlNet या image mask data के साथ इस विकल्प से pair किया जा सकता है।
- **Note:** यदि आपके पास कई conditioning datasets हैं, तो आप उन्हें `id` values के array के रूप में दे सकते हैं। Flux Kontext ट्रेन करते समय, यह conditions के बीच random switching या multi‑image compositing tasks के लिए inputs stitch करने की अनुमति देता है।

### `instance_data_dir` / `aws_data_prefix`

- **Local:** filesystem पर डेटा का path.
- **AWS:** bucket में डेटा का S3 prefix.

### `caption_strategy`

- **textfile** requires कि image.png के साथ उसी डायरेक्टरी में image.txt हो, जिसमें एक या अधिक captions नई पंक्तियों से अलग किए गए हों। ये image+text pairs **एक ही डायरेक्टरी में** होने चाहिए।
- **instanceprompt** requires कि `instance_prompt` का मान दिया जाए, और यह हर image के caption के लिए **केवल** यही मान उपयोग करेगा।
- **filename** फ़ाइल‑नाम का cleaned‑up संस्करण caption के रूप में उपयोग करेगा, जैसे underscores को spaces से बदलकर।
- **parquet** captions को parquet table से खींचता है जिसमें बाकी image metadata होता है। इसे कॉन्फ़िगर करने के लिए `parquet` फ़ील्ड का उपयोग करें। देखें [Parquet caption strategy](#parquet-caption-strategy-json-lines-datasets)।

`textfile` और `parquet` दोनों multi‑captions सपोर्ट करते हैं:
- textfiles नई पंक्तियों पर split होते हैं। हर नई पंक्ति अलग caption होगी।
- parquet tables में field iterable type हो सकता है।

### `disable_multiline_split`

- जब `true` पर सेट किया जाता है, तो caption text files को newlines द्वारा multiple caption variants में split होने से रोकता है।
- उपयोगी जब आपके captions में intentional line breaks हों जिन्हें एक single caption के रूप में संरक्षित रखना हो।
- Default: `false` (captions newlines द्वारा split होते हैं)

### `caption_shuffle`

Data augmentation के लिए tag-based captions के deterministic shuffled variants generate करता है। यह model को सिखाता है कि tag order महत्वपूर्ण नहीं है और specific tag sequences पर overfitting कम करता है।

**Configuration:**

```json
{
  "caption_shuffle": {
    "enable": true,
    "count": 3,
    "seed": 42,
    "split_on": "comma",
    "position_start": 1,
    "include_original": true
  }
}
```

**Parameters:**

- `enable` (bool): Caption shuffling enable करना है या नहीं। Default: `false`
- `count` (int): प्रति caption generate करने के लिए shuffled variants की संख्या। Default: `1`
- `seed` (int): Deterministic shuffling के लिए seed। यदि specify नहीं किया गया, तो global `--seed` value उपयोग होता है।
- `split_on` (string): Captions को tags में split करने के लिए delimiter। Options: `comma`, `space`, `period`। Default: `comma`
- `position_start` (int): पहले N tags को उनकी original position में रखें (subject/style tags को पहले रखने के लिए उपयोगी)। Default: `0`
- `include_original` (bool): Shuffled variants के साथ unshuffled original caption include करना है या नहीं। Default: `true`

**Example:**

`split_on: "comma"`, `position_start: 1`, `count: 2` के साथ:

- Original: `"dog, running, park, sunny day"`
- Result: `["dog, running, park, sunny day", "dog, park, sunny day, running", "dog, sunny day, running, park"]`

पहला tag "dog" fixed रहता है जबकि बाकी tags shuffle होते हैं।

**Notes:**

- Shuffling text embed pre-caching के दौरान apply होता है, इसलिए सभी variants एक बार में calculate होते हैं।
- Training के दौरान, प्रति sample एक variant randomly select होता है।
- यदि caption में `position_start + 2` से कम tags हैं, तो shuffling skip होता है (shuffle करने के लिए कुछ meaningful नहीं)।
- जब `include_original: false` लेकिन shuffling possible नहीं है, तो warning के साथ original include होता है।

### `metadata_backend`

- **Values:** `discovery` | `parquet` | `huggingface`
- **Description:** dataset preparation के दौरान SimpleTuner image dimensions और अन्य metadata कैसे खोजेगा।
  - **discovery** (डिफ़ॉल्ट): वास्तविक image files को स्कैन करके dimensions पढ़ता है। किसी भी storage backend पर काम करता है लेकिन बड़े datasets के लिए धीमा हो सकता है।
  - **parquet**: parquet/JSONL फ़ाइल में `width_column` और `height_column` से dimensions पढ़ता है, file access skip करता है। देखें [Parquet caption strategy](#parquet-caption-strategy-json-lines-datasets)।
  - **huggingface**: Hugging Face datasets से metadata उपयोग करता है। देखें [Hugging Face Datasets Support](#hugging-face-datasets-support)।
- **Note:** `parquet` उपयोग करते समय `parquet` block में `width_column` और `height_column` कॉन्फ़िगर करना आवश्यक है। इससे बड़े datasets के लिए startup काफ़ी तेज़ हो जाता है।

### `metadata_update_interval`

- **Values:** Integer (seconds)
- **Description:** प्रशिक्षण के दौरान dataset metadata को कितनी बार refresh करना है (seconds में)। लंबे प्रशिक्षण में बदलते datasets के लिए उपयोगी।
- **Default:** trainer के `--metadata_update_interval` argument पर fallback करता है।

### Cropping Options

- `crop`: image cropping सक्षम/अक्षम करता है।
- `crop_style`: cropping शैली चुनता है (`random`, `center`, `corner`, `face`)।
- `crop_aspect`: cropping aspect चुनता है (`closest`, `random`, `square` या `preserve`)।
- `crop_aspect_buckets`: जब `crop_aspect` `closest` या `random` हो, तो इस सूची से bucket चुना जाएगा। डिफ़ॉल्ट रूप से सभी buckets उपलब्ध हैं (unlimited upscaling अनुमति)। जरूरत हो तो `max_upscale_threshold` से upscaling सीमित करें।

### `resolution`

- **resolution_type=area:** final image size megapixel count पर निर्भर है — यहाँ 1.05 का मान 1024^2 (1024x1024) कुल pixel area के आसपास aspect buckets देगा, ~1_050_000 pixels।
- **resolution_type=pixel_area:** `area` जैसा, लेकिन megapixels की जगह pixels में मापा जाता है। यहाँ 1024 का मान 1024^2 (1024x1024) कुल pixel area के आसपास aspect buckets देगा, ~1_050_000 pixels।
- **resolution_type=pixel:** final image size का छोटा edge इस मान पर निर्धारित होगा।

> **NOTE**: images upscale, downscale या crop हों, यह `minimum_image_size`, `maximum_target_size`, `target_downsample_size`, `crop`, और `crop_aspect` के मानों पर निर्भर करता है।

### `minimum_image_size`

- जिन images का आकार इस मान से नीचे गिरता है, उन्हें training से **exclude** किया जाएगा।
- जब `resolution` megapixels में मापा जाए (`resolution_type=area`), तो यह भी megapixels में होना चाहिए (उदा. 1024x1024 **area** से छोटे images को exclude करने के लिए `1.05` megapixels)।
- जब `resolution` pixels में मापा जाए, तो यहाँ भी वही unit उपयोग करें (उदा. छोटे edge length में 1024px से छोटे images को exclude करने के लिए `1024`)।
- **Recommendation**: `minimum_image_size` को `resolution` के बराबर रखें, जब तक आप खराब upscaling वाले images पर training का जोखिम नहीं लेना चाहते।

### `minimum_aspect_ratio`

- **Description:** image का न्यूनतम aspect ratio। यदि image का aspect ratio इस मान से कम है, तो उसे training से exclude किया जाएगा।
- **Note**: यदि exclude होने वाली images की संख्या बहुत अधिक है, तो startup में समय अधिक लग सकता है क्योंकि trainer उन्हें scan और bucket करने की कोशिश करेगा यदि वे bucket lists में नहीं हैं।

> **Note**: एक बार dataset के लिए aspect और metadata lists बन जाने के बाद, `skip_file_discovery="vae aspect metadata"` उपयोग करने से startup पर dataset स्कैन नहीं होगा और बहुत समय बच जाएगा।

### `maximum_aspect_ratio`

- **Description:** image का अधिकतम aspect ratio। यदि image का aspect ratio इस मान से अधिक है, तो उसे training से exclude किया जाएगा।
- **Note**: यदि exclude होने वाली images की संख्या बहुत अधिक है, तो startup में समय अधिक लग सकता है क्योंकि trainer उन्हें scan और bucket करने की कोशिश करेगा यदि वे bucket lists में नहीं हैं।

> **Note**: एक बार dataset के लिए aspect और metadata lists बन जाने के बाद, `skip_file_discovery="vae aspect metadata"` उपयोग करने से startup पर dataset स्कैन नहीं होगा और बहुत समय बच जाएगा।

### `conditioning`

- **Values:** conditioning कॉन्फ़िगरेशन objects की array
- **Description:** source images से conditioning datasets स्वतः बनाता है। प्रत्येक conditioning type एक अलग dataset बनाता है जिसे ControlNet training या अन्य conditioning tasks में उपयोग किया जा सकता है।
- **Note:** निर्दिष्ट होने पर SimpleTuner `{source_id}_conditioning_{type}` जैसे IDs के साथ conditioning datasets स्वतः बनाएगा।

हर conditioning object में शामिल हो सकता है:
- `type`: generate होने वाला conditioning type (required)
- `params`: type‑specific parameters (optional)
- `captions`: generated dataset के लिए caption strategy (optional)
  - `false` हो सकता है (कोई captions नहीं)
  - एक string (सभी images के लिए instance prompt के रूप में)
  - strings की array (हर image के लिए randomly चुनी जाती है)
  - यदि छोड़ा जाए, तो source dataset के captions उपयोग होंगे

#### Available Conditioning Types

##### `superresolution`
Super‑resolution training के लिए images के low‑quality संस्करण बनाता है:
```json
{
  "type": "superresolution",
  "blur_radius": 2.5,
  "blur_type": "gaussian",
  "add_noise": true,
  "noise_level": 0.03,
  "jpeg_quality": 85,
  "downscale_factor": 2
}
```

##### `jpeg_artifacts`
Artifact removal training के लिए JPEG compression artifacts बनाता है:
```json
{
  "type": "jpeg_artifacts",
  "quality_mode": "range",
  "quality_range": [10, 30],
  "compression_rounds": 1,
  "enhance_blocks": false
}
```

##### `depth` / `depth_midas`
DPT models से depth maps बनाता है:
```json
{
  "type": "depth_midas",
  "model_type": "DPT"
}
```
**Note:** Depth generation के लिए GPU चाहिए और यह main process में चलता है, जो CPU‑based generators की तुलना में धीमा हो सकता है।

##### `random_masks` / `inpainting`
Inpainting training के लिए random masks बनाता है:
```json
{
  "type": "random_masks",
  "mask_types": ["rectangle", "circle", "brush", "irregular"],
  "min_coverage": 0.1,
  "max_coverage": 0.5,
  "output_mode": "mask"
}
```

##### `canny` / `edges`
Canny edge detection maps बनाता है:
```json
{
  "type": "canny",
  "low_threshold": 100,
  "high_threshold": 200
}
```

इन conditioning datasets के उपयोग के लिए अधिक विवरण [ControlNet guide](CONTROLNET.md) में देखें।

#### Examples

##### Video dataset

Video dataset (उदा. mp4) वीडियो फ़ाइलों का फ़ोल्डर होना चाहिए और captions स्टोर करने के सामान्य तरीकों का उपयोग करना चाहिए।

```json
[
  {
    "id": "disney-black-and-white",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/ltxvideo/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 125,
        "min_frames": 125
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/ltxvideo",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

- `video` subsection में हम निम्न keys सेट कर सकते हैं:
  - `num_frames` (optional, int) यह बताता है कि हम कितने frames पर training करेंगे।
    - 25 fps पर 125 frames = 5 seconds का वीडियो, standard output। यही आपका target होना चाहिए।
  - `min_frames` (optional, int) training के लिए स्वीकार्य न्यूनतम video length निर्धारित करता है।
    - यह कम से कम `num_frames` के बराबर होना चाहिए। इसे सेट न करने पर यह बराबर माना जाएगा।
  - `max_frames` (optional, int) training के लिए स्वीकार्य अधिकतम video length निर्धारित करता है।
  - `is_i2v` (optional, bool) यह तय करता है कि dataset पर i2v training की जाएगी या नहीं।
    - LTX के लिए यह डिफ़ॉल्ट रूप से True होता है। आप इसे disable कर सकते हैं।
  - `bucket_strategy` (optional, string) यह तय करता है कि videos को buckets में कैसे समूहित किया जाए:
    - `aspect_ratio` (डिफ़ॉल्ट): केवल spatial aspect ratio के आधार पर buckets (उदा. `1.78`, `0.75`)। image datasets जैसा व्यवहार।
    - `resolution_frames`: resolution और frame count के आधार पर `WxH@F` फ़ॉर्मैट में buckets (उदा. `1920x1080@125`)। अलग resolutions और durations वाले datasets के लिए उपयोगी।
  - `frame_interval` (optional, int) जब `bucket_strategy: "resolution_frames"` उपयोग हो, तो frame counts को इस मान के निकटतम गुणज तक नीचे round किया जाता है। इसे आपके मॉडल के required frame count factor पर सेट करें (कुछ models को `num_frames - 1` का किसी मान से divisible होना आवश्यक होता है)।

**Automatic Frame Count Adjustment:** SimpleTuner automatically आपके videos की frame counts को model-specific constraints को पूरा करने के लिए adjust करता है। उदाहरण के लिए, LTX-2 को frame counts की आवश्यकता है जो `frames % 8 == 1` (जैसे 49, 57, 65, 73, 81, आदि) को satisfy करते हैं। यदि आपके videos में अलग frame counts हैं (जैसे 119 frames), तो वे automatically nearest valid frame count (जैसे 113 frames) तक trim कर दिए जाते हैं। adjustment के बाद `min_frames` से कम हो जाने वाले videos को warning message के साथ skip किया जाता है। यह automatic adjustment training errors को prevent करता है और किसी configuration की आवश्यकता नहीं है।

**Note:** `bucket_strategy: "resolution_frames"` के साथ `num_frames` सेट करने पर आपको केवल एक frame bucket मिलेगा और `num_frames` से छोटे videos discard होंगे। यदि आप कम discards के साथ multiple frame buckets चाहते हैं, तो `num_frames` unset रखें।

Mixed‑resolution video datasets के लिए `resolution_frames` bucketing का उदाहरण:

```json
{
  "id": "mixed-resolution-videos",
  "type": "local",
  "dataset_type": "video",
  "resolution": 720,
  "resolution_type": "pixel_area",
  "instance_data_dir": "datasets/videos",
  "video": {
      "bucket_strategy": "resolution_frames",
      "frame_interval": 25,
      "min_frames": 25,
      "max_frames": 250
  }
}
```

यह कॉन्फ़िगरेशन `1280x720@100`, `1920x1080@125`, `640x480@75` जैसी buckets बनाएगा। Videos को उनकी training resolution और frame count के आधार पर group किया जाता है (निकटतम 25 frames पर rounded)।


##### Configuration
```json
    "minimum_image_size": 1024,
    "resolution": 1024,
    "resolution_type": "pixel"
```
##### Outcome
- जिन images का छोटा edge **1024px** से कम है, वे पूरी तरह training से exclude होंगी।
- `768x1024` या `1280x768` जैसी images exclude होंगी, लेकिन `1760x1024` और `1024x1024` नहीं।
- कोई image upsample नहीं होगी, क्योंकि `minimum_image_size` `resolution` के बराबर है

##### Configuration
```json
    "minimum_image_size": 1024,
    "resolution": 1024,
    "resolution_type": "pixel_area" # different from the above configuration, which is 'pixel'
```
##### Outcome
- image का total area (width * height) यदि minimum area (1024 * 1024) से कम है तो उसे training से exclude किया जाएगा।
- `1280x960` जैसी images exclude **नहीं** होंगी क्योंकि `(1280 * 960)` `(1024 * 1024)` से बड़ा है
- कोई image upsample नहीं होगी, क्योंकि `minimum_image_size` `resolution` के बराबर है

##### Configuration
```json
    "minimum_image_size": 0, # or completely unset, not present in the config
    "resolution": 1024,
    "resolution_type": "pixel",
    "crop": false
```

##### Outcome
- images का छोटा edge 1024px पर लाते हुए उनका aspect ratio बनाए रखा जाएगा
- size के आधार पर कोई image exclude नहीं होगी
- छोटी images को naive `PIL.resize` methods से upsample किया जाएगा, जो अच्छे नहीं दिखते
  - upscaling से बचने की सिफ़ारिश है जब तक कि training से पहले आप अपनी पसंद के upscaler से इसे हाथ से न करें

### `maximum_image_size` and `target_downsample_size`

Images cropping से पहले resize नहीं होतीं **जब तक** `maximum_image_size` और `target_downsample_size` दोनों सेट न हों। यानी, `4096x4096` image सीधे `1024x1024` target तक crop होगी, जो अनचाहा हो सकता है।

- `maximum_image_size` वह threshold बताता है जिसके बाद resizing शुरू होगी। यदि images इससे बड़ी हैं तो cropping से पहले downsample होंगी।
- `target_downsample_size` बताता है कि resample के बाद और crop से पहले image कितनी बड़ी होगी।

#### Examples

##### Configuration
```json
    "resolution_type": "pixel_area",
    "resolution": 1024,
    "maximum_image_size": 1536,
    "target_downsample_size": 1280,
    "crop": true,
    "crop_aspect": "square"
```

##### Outcome
- जिन images का pixel area `(1536 * 1536)` से बड़ा है, उन्हें resize करके उनका pixel area लगभग `(1280 * 1280)` किया जाएगा, जबकि उनका original aspect ratio बना रहेगा
- final image size को random‑cropped करके `(1024 * 1024)` pixel area बनाया जाएगा
- उदा. 20 megapixel datasets पर training के लिए उपयोगी, जिन्हें cropping से पहले पर्याप्त रूप से resize करना जरूरी हो ताकि scene context का भारी नुकसान न हो (जैसे किसी व्यक्ति की तस्वीर को केवल टाइल वॉल या blurry background के छोटे हिस्से तक crop कर देना)

### `max_upscale_threshold`

डिफ़ॉल्ट रूप से SimpleTuner छोटी images को target resolution तक upscale करता है, जिससे गुणवत्ता गिर सकती है। `max_upscale_threshold` विकल्प इस upscaling व्यवहार को सीमित करने देता है।

- **Default**: `null` (unlimited upscaling अनुमति)
- **When set**: उन aspect buckets को फ़िल्टर करता है जिनमें दिए गए threshold से अधिक upscaling चाहिए
- **Value range**: 0 और 1 के बीच (उदा. `0.2` = अधिकतम 20% upscaling की अनुमति)
- **Applies to**: जब `crop_aspect` `closest` या `random` हो, तब aspect bucket selection पर लागू

#### Examples

##### Configuration
```json
    "resolution": 1024,
    "resolution_type": "pixel",
    "crop": true,
    "crop_aspect": "random",
    "crop_aspect_buckets": [1.0, 0.5, 2.0],
    "max_upscale_threshold": null
```

##### Outcome
- सभी aspect buckets चयन के लिए उपलब्ध हैं
- 256x256 image को 1024x1024 तक upscale किया जा सकता है (4x scaling)
- बहुत छोटी images के लिए quality degradation हो सकती है

##### Configuration
```json
    "resolution": 1024,
    "resolution_type": "pixel",
    "crop": true,
    "crop_aspect": "random",
    "crop_aspect_buckets": [1.0, 0.5, 2.0],
    "max_upscale_threshold": 0.2
```

##### Outcome
- केवल वे aspect buckets उपलब्ध होंगे जिन्हें ≤20% upscaling चाहिए
- 256x256 image को 1024x1024 तक बढ़ाने के लिए (4x = 300% upscaling) कोई bucket उपलब्ध नहीं होगा
- 850x850 image सभी buckets उपयोग कर सकती है क्योंकि 1024/850 ≈ 1.2 (20% upscaling)
- खराब upscaling वाली images को बाहर रखकर training quality बनाए रखने में मदद करता है

---

### `prepend_instance_prompt`

- सक्षम होने पर, सभी captions की शुरुआत में `instance_prompt` मान जोड़ा जाएगा।

### `only_instance_prompt`

- `prepend_instance_prompt` के अतिरिक्त, पूरे dataset के सभी captions को एक ही phrase या trigger word से बदल देता है।

### `repeats`

- यह बताता है कि epoch के दौरान dataset के सभी samples कितनी बार देखे जाते हैं। छोटे datasets का प्रभाव बढ़ाने या VAE cache objects का अधिकतम उपयोग करने के लिए उपयोगी।
- यदि आपके पास 1000 images वाला dataset और 100 images वाला dataset है, तो छोटे dataset को 1000 total images के बराबर लाने के लिए आपको `repeats` `9` **या उससे अधिक** देना चाहिए।

> ℹ️ यह मान Kohya scripts के समान विकल्प से अलग व्यवहार करता है, जहाँ मान 1 का अर्थ no repeats होता है। **SimpleTuner में 0 का अर्थ no repeats है**। Kohya config value से 1 घटाएँ ताकि SimpleTuner के समतुल्य मान मिले; इसी कारण `(dataset_length + repeats * dataset_length)` से **9** प्राप्त होता है।

#### Multi-GPU Training and Dataset Sizing

Multiple GPUs के साथ training करते समय, आपका dataset **effective batch size** को समाहित करने लायक होना चाहिए, जिसकी गणना इस प्रकार है:

```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

उदाहरण के लिए, 4 GPUs, `train_batch_size=4`, और `gradient_accumulation_steps=1` के साथ, हर aspect bucket में (repeats लागू होने के बाद) कम से कम **16 samples** चाहिए।

**Important:** यदि आपका dataset कॉन्फ़िगरेशन zero usable batches बनाता है तो SimpleTuner error उठाएगा। error message में दिखाया जाएगा:
- वर्तमान कॉन्फ़िगरेशन मान (batch size, GPU count, repeats)
- किन aspect buckets में samples कम हैं
- प्रत्येक bucket के लिए न्यूनतम required repeats
- सुझाए गए solutions

##### Automatic Dataset Oversubscription

यदि आपका dataset effective batch size से छोटा है और `repeats` स्वतः समायोजित करना चाहते हैं, तो `--allow_dataset_oversubscription` फ़्लैग का उपयोग करें ([OPTIONS.md](OPTIONS.md#--allow_dataset_oversubscription) में दस्तावेज़ित)।

सक्षम होने पर, SimpleTuner:
- training के लिए न्यूनतम repeats की गणना करेगा
- आवश्यकता पूरी करने के लिए `repeats` स्वतः बढ़ाएगा
- adjustment दिखाने के लिए warning लॉग करेगा
- **manually‑set repeats का सम्मान करेगा** — यदि आपने dataset config में `repeats` स्पष्ट रूप से सेट किया है, तो auto adjustment skip होगा

यह विशेष रूप से उपयोगी है जब:
- छोटे datasets (< 100 images) पर training हो
- छोटे datasets के साथ उच्च GPU counts उपयोग हों
- अलग batch sizes के साथ experiment कर रहे हों बिना datasets फिर से कॉन्फ़िगर किए

**Example scenario:**
- Dataset: 25 images
- Configuration: 8 GPUs, `train_batch_size=4`, `gradient_accumulation_steps=1`
- Effective batch size: 32 samples चाहिए
- Oversubscription के बिना: Error आएगा
- `--allow_dataset_oversubscription` के साथ: repeats स्वतः 1 पर सेट होंगे (25 × 2 = 50 samples)

### `max_num_samples`

- **विवरण:** Dataset को अधिकतम samples की संख्या तक सीमित करता है। सेट करने पर, पूर्ण dataset से निर्दिष्ट आकार का एक deterministic random subset चुना जाता है।
- **उपयोग का मामला:** बड़े regularization datasets के लिए उपयोगी जहाँ आप छोटे training sets को overwhelm न करने के लिए डेटा का केवल एक हिस्सा उपयोग करना चाहते हैं।
- **Deterministic selection:** Random selection dataset `id` को seed के रूप में उपयोग करता है, जिससे reproducibility के लिए training sessions में समान subset चुना जाना सुनिश्चित होता है।
- **डिफ़ॉल्ट:** `null` (कोई सीमा नहीं, सभी samples उपयोग होते हैं)

#### उदाहरण
```json
{
  "id": "regularization-data",
  "max_num_samples": 1000,
  ...
}
```

यह dataset से 1000 samples को deterministically select करेगा, जिसमें हर बार training चलाने पर समान selection उपयोग होगी।

### `start_epoch` / `start_step`

- यह schedule करता है कि dataset sampling कब शुरू होगी।
- `start_epoch` (डिफ़ॉल्ट: `1`) epoch number से gate करता है; `start_step` (डिफ़ॉल्ट: `0`) optimizer step (gradient accumulation के बाद) से gate करता है। Samples drawn होने से पहले दोनों conditions पूरी होनी चाहिए।
- कम से कम एक dataset में `start_epoch<=1` **और** `start_step<=1` होना चाहिए; अन्यथा startup पर कोई डेटा उपलब्ध नहीं होगा और training error देगी।
- जिन datasets की start condition कभी पूरी नहीं होती (उदा., `start_epoch` `--num_train_epochs` से आगे), उन्हें skip किया जाएगा और model card में नोट किया जाएगा।
- जब scheduled datasets mid‑run में active होते हैं, तो progress‑bar step estimates approximate हो जाते हैं क्योंकि epoch length बढ़ सकती है।

### `end_epoch` / `end_step`

- यह schedule करता है कि dataset sampling कब **बंद** होगी (`start_epoch`/`start_step` का पूरक)।
- `end_epoch` (डिफ़ॉल्ट: `null` = कोई सीमा नहीं) इस epoch के बाद sampling बंद कर देता है; `end_step` (डिफ़ॉल्ट: `null` = कोई सीमा नहीं) इस optimizer step के बाद sampling बंद कर देता है।
- कोई भी condition समाप्त होने पर dataset बंद हो जाएगा; वे स्वतंत्र रूप से काम करते हैं।
- **Curriculum learning** workflows के लिए उपयोगी जहाँ आप चाहते हैं:
  - पहले low-resolution data पर train करें, फिर high-resolution data पर switch करें।
  - एक निश्चित बिंदु के बाद regularisation data को धीरे-धीरे हटाएं।
  - एक single config file में multi-stage training बनाएं।

**उदाहरण: Curriculum Learning**
```json
[
  {
    "id": "lowres-512",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "/data/512",
    "end_step": 300
  },
  {
    "id": "highres-1024",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "/data/1024",
    "start_step": 300
  }
]
```

इस उदाहरण में, 512px dataset steps 1-300 के लिए उपयोग होता है, फिर 1024px dataset step 300 से आगे संभाल लेता है।

### `is_regularisation_data`

- इसे `is_regularization_data` भी लिखा जा सकता है
- LyCORIS adapters के लिए parent‑teacher training सक्षम करता है ताकि prediction target दिए गए dataset के लिए base model के परिणाम को प्राथमिकता दे।
  - Standard LoRA फिलहाल समर्थित नहीं हैं।

### `delete_unwanted_images`

- **Values:** `true` | `false`
- **Description:** सक्षम होने पर, size या aspect ratio filters में fail होने वाली images (उदा. `minimum_image_size` से नीचे या `minimum_aspect_ratio`/`maximum_aspect_ratio` से बाहर) dataset directory से स्थायी रूप से delete कर दी जाती हैं।
- **Warning:** यह destructive है और undo नहीं किया जा सकता। सावधानी से उपयोग करें।
- **Default:** trainer के `--delete_unwanted_images` argument पर fallback करता है (डिफ़ॉल्ट: `false`)।

### `delete_problematic_images`

- **Values:** `true` | `false`
- **Description:** सक्षम होने पर, VAE encoding के दौरान fail होने वाली images (corrupted files, unsupported formats, आदि) dataset directory से स्थायी रूप से delete कर दी जाती हैं।
- **Warning:** यह destructive है और undo नहीं किया जा सकता। सावधानी से उपयोग करें।
- **Default:** trainer के `--delete_problematic_images` argument पर fallback करता है (डिफ़ॉल्ट: `false`)।

### Filtering Statistics देखना

जब SimpleTuner आपके dataset को process करता है, यह track करता है कि कितनी files filter हुईं और क्यों। ये statistics dataset के cache file (`aspect_ratio_bucket_indices_*.json`) में store होती हैं और WebUI में देखी जा सकती हैं।

**Track की जाने वाली Statistics:**
- **total_processed**: Process की गई files की संख्या
- **too_small**: `minimum_image_size` से नीचे होने के कारण filter की गई files
- **too_long**: duration limits से अधिक होने के कारण filter की गई files (audio/video)
- **metadata_missing**: missing metadata के कारण skip की गई files
- **not_found**: जो files locate नहीं हो सकीं
- **already_exists**: cache में पहले से मौजूद files (reprocess नहीं हुईं)
- **other**: अन्य कारणों से filter की गई files

**WebUI में देखना:**

WebUI file browser में datasets browse करते समय, किसी existing dataset वाली directory select करने पर filtering statistics दिखाई देंगी (यदि उपलब्ध हों)। यह diagnose करने में मदद करता है कि आपके dataset में expected से कम usable samples क्यों हैं।

**Filtered files का Troubleshooting:**

यदि बहुत सी files `too_small` के रूप में filter हो रही हैं:
1. अपनी `minimum_image_size` setting check करें — यह `resolution` और `resolution_type` से match होनी चाहिए
2. `resolution_type=pixel` के लिए, `minimum_image_size` minimum shorter edge length है
3. `resolution_type=area` या `pixel_area` के लिए, `minimum_image_size` minimum total area है

अधिक details के लिए नीचे [Troubleshooting](#filtered-datasets-का-troubleshooting) section देखें।

### `slider_strength`

- **Values:** कोई भी float मान (positive, negative, या zero)
- **Description:** dataset को slider LoRA training के लिए चिह्नित करता है, जो contrastive "opposites" सीखकर controllable concept adapters बनाता है।
  - **Positive values** (उदा. `0.5`): "कॉनसेप्ट का अधिक" — brighter eyes, stronger smile, आदि।
  - **Negative values** (उदा. `-0.5`): "कॉनसेप्ट का कम" — dimmer eyes, neutral expression, आदि।
  - **Zero या omitted**: neutral examples जो किसी दिशा में concept push नहीं करते।
- **Note:** जब datasets में `slider_strength` values होती हैं, SimpleTuner batches को fixed cycle में rotate करता है: positive → negative → neutral। हर group के भीतर standard backend probabilities लागू रहती हैं।
- **See also:** slider LoRA training सेटअप के लिए [SLIDER_LORA.md](SLIDER_LORA.md) देखें।

### `vae_cache_clear_each_epoch`

- सक्षम होने पर, हर dataset repeat cycle के अंत में filesystem से सभी VAE cache objects delete कर दिए जाते हैं। बड़े datasets के लिए यह resource‑intensive हो सकता है, लेकिन `crop_style=random` और/या `crop_aspect=random` के साथ इसे सक्षम रखना बेहतर है ताकि हर image से crops की पूरी रेंज sample हो सके।
- वास्तव में, random bucketing या crops उपयोग करने पर यह विकल्प **डिफ़ॉल्ट रूप से सक्षम** होता है।

### `vae_cache_disable`

- **Values:** `true` | `false`
- **Description:** सक्षम होने पर (command‑line argument `--vae_cache_disable` के जरिए), यह विकल्प on‑demand VAE caching सक्षम करता है लेकिन generated embeddings को disk पर लिखना बंद कर देता है। यह बड़े datasets के लिए उपयोगी है जहाँ disk space चिंता का विषय है या लिखना व्यावहारिक नहीं है।
- **Note:** यह trainer‑level argument है, per‑dataset कॉन्फ़िगरेशन विकल्प नहीं, लेकिन यह dataloader के VAE cache के साथ interaction को प्रभावित करता है।

### `skip_file_discovery`

- आप शायद इसे कभी सेट नहीं करना चाहेंगे — यह केवल बहुत बड़े datasets के लिए उपयोगी है।
- यह parameter values की comma या space separated सूची स्वीकार करता है, जैसे `vae metadata aspect text`, ताकि loader configuration के एक या अधिक चरणों के लिए file discovery skip की जा सके।
- यह commandline option `--skip_file_discovery` के बराबर है।
- यह तब उपयोगी है जब आपके पास ऐसे datasets हों जिन्हें trainer को हर startup पर scan करने की आवश्यकता नहीं है, जैसे उनके latents/embeds पहले से पूरी तरह cached हों। इससे startup और training resume तेज़ होते हैं।

### `preserve_data_backend_cache`

- आप शायद इसे कभी सेट नहीं करना चाहेंगे — यह केवल बहुत बड़े AWS datasets के लिए उपयोगी है।
- `skip_file_discovery` की तरह, इसे startup पर अनावश्यक, लंबी, और महँगी filesystem scans रोकने के लिए सेट किया जा सकता है।
- यह boolean मान लेता है, और `true` होने पर generated filesystem list cache file launch पर हटाई नहीं जाती।
- यह बहुत बड़े और धीमे storage systems (जैसे S3 या local SMR spinning hard drives) के लिए उपयोगी है, जिनका response time बहुत धीमा होता है।
- इसके अतिरिक्त, S3 पर backend listing लागत बढ़ा सकती है और इससे बचना चाहिए।

> ⚠️ **दुर्भाग्य से, यदि डेटा सक्रिय रूप से बदल रहा है तो इसे सेट नहीं किया जा सकता।** ट्रेनर pool में जोड़े गए नए डेटा को नहीं देख पाएगा; इसके लिए पूर्ण scan फिर से करना पड़ेगा।

### `hash_filenames`

- VAE cache entries की filenames हमेशा hashed होती हैं। यह user‑configurable नहीं है और लंबे filenames वाले datasets को बिना path length issues के उपयोग करने में मदद करता है। आपके कॉन्फ़िगरेशन में कोई भी `hash_filenames` सेटिंग ignore की जाएगी।

## Captions फ़िल्टर करना

### `caption_filter_list`

- **केवल text embed datasets के लिए।** यह एक JSON सूची, txt फ़ाइल का path, या JSON document का path हो सकता है। Filter strings simple terms हो सकते हैं जिन्हें सभी captions से हटाया जाए, या वे regular expressions हो सकते हैं। इसके अतिरिक्त, sed‑style `s/search/replace/` entries captions में strings को हटाने की बजाय _replace_ करने के लिए उपयोग की जा सकती हैं।

#### Example filter list

पूरा उदाहरण [यहाँ](/config/caption_filter_list.txt.example) उपलब्ध है। इसमें BLIP (all common variety), LLaVA, और CogVLM द्वारा लौटाई गई सामान्य repetitive और negative strings होती हैं।

यह एक छोटा उदाहरण है, जिसे नीचे समझाया गया है:

```
arafed
this .* has a
^this is the beginning of the string
s/this/will be found and replaced/
```

क्रम से, ये लाइनों का व्यवहार:

- `arafed ` (आख़िर में space के साथ) उस caption से हटाया जाएगा जहाँ यह मिलता है। अंत में space रखने से caption बेहतर दिखेगा क्योंकि double‑spaces नहीं बचेंगे। यह आवश्यक नहीं है, लेकिन अच्छा दिखता है।
- `this .* has a` एक regular expression है जो "this ... has a" वाले किसी भी भाग को हटाएगा; `.*` का मतलब है "जो भी मिले" जब तक कि "has a" न मिल जाए।
- `^this is the beginning of the string` phrase को हटाएगा, लेकिन केवल तब जब यह caption की शुरुआत में हो।
- `s/this/will be found and replaced/` caption में "this" के पहले instance को "will be found and replaced" से बदल देगा।

> ❗Regular expressions को debug और test करने के लिए [regex 101](https://regex101.com) का उपयोग करें।

# Advanced techniques

## Advanced Example Configuration

```json
[
  {
    "id": "something-special-to-remember-by",
    "type": "local",
    "instance_data_dir": "/path/to/data/tree",
    "crop": false,
    "crop_style": "random|center|corner|face",
    "crop_aspect": "square|preserve|closest|random",
    "crop_aspect_buckets": [0.33, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    "resolution": 1.0,
    "resolution_type": "area|pixel",
    "minimum_image_size": 1.0,
    "prepend_instance_prompt": false,
    "instance_prompt": "something to label every image",
    "only_instance_prompt": false,
    "caption_strategy": "filename|instanceprompt|parquet|textfile",
    "disable_multiline_split": false,
    "cache_dir_vae": "/path/to/vaecache",
    "vae_cache_clear_each_epoch": true,
    "probability": 1.0,
    "repeats": 0,
    "start_epoch": 1,
    "start_step": 0,
    "text_embeds": "alt-embed-cache",
    "image_embeds": "vae-embeds-example",
    "conditioning_image_embeds": "conditioning-embeds-example"
  },
  {
    "id": "another-special-name-for-another-backend",
    "type": "aws",
    "aws_bucket_name": "something-yummy",
    "aws_region_name": null,
    "aws_endpoint_url": "https://foo.bar/",
    "aws_access_key_id": "wpz-764e9734523434",
    "aws_secret_access_key": "xyz-sdajkhfhakhfjd",
    "aws_data_prefix": "",
    "cache_dir_vae": "s3prefix/for/vaecache",
    "vae_cache_clear_each_epoch": true,
    "repeats": 0
  },
  {
      "id": "vae-embeds-example",
      "type": "local",
      "dataset_type": "image_embeds",
      "disabled": false,
  },
  {
      "id": "conditioning-embeds-example",
      "type": "local",
      "dataset_type": "conditioning_image_embeds",
      "disabled": false
  },
  {
    "id": "an example backend for text embeds.",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "aws",
    "aws_bucket_name": "textembeds-something-yummy",
    "aws_region_name": null,
    "aws_endpoint_url": "https://foo.bar/",
    "aws_access_key_id": "wpz-764e9734523434",
    "aws_secret_access_key": "xyz-sdajkhfhakhfjd",
    "aws_data_prefix": "",
    "cache_dir": ""
  },
  {
    "id": "alt-embed-cache",
    "dataset_type": "text_embeds",
    "default": false,
    "type": "local",
    "cache_dir": "/path/to/textembed_cache"
  }
]
```

## CSV URL list से सीधे training

**Note: आपके CSV में आपकी images के captions शामिल होने चाहिए।**

> ⚠️ यह advanced **और** experimental फीचर है, और इसमें समस्याएँ आ सकती हैं। यदि ऐसा हो, तो कृपया [issue](https://github.com/bghira/simpletuner/issues) खोलें!

डेटा को URL list से मैन्युअल डाउनलोड करने के बजाय, आप इसे सीधे trainer में plug कर सकते हैं।

**Note:** image data को मैन्युअल डाउनलोड करना हमेशा बेहतर होता है। local disk space बचाने का एक और तरीका [cloud storage के साथ local encoder caches](#local-cache-with-cloud-dataset) उपयोग करना हो सकता है।

### Advantages

- डेटा सीधे डाउनलोड करने की जरूरत नहीं
- SimpleTuner के caption toolkit से URL list को सीधे caption किया जा सकता है
- disk space बचता है, क्योंकि केवल image embeds (यदि लागू हों) और text embeds स्टोर होते हैं

### Disadvantages

- हर image डाउनलोड कर उसके metadata collect करने के लिए महँगा और संभावित रूप से धीमा aspect bucket scan चाहिए
- डाउनलोड की गई images on‑disk cached होती हैं, जो बहुत बड़ी हो सकती हैं। यह सुधार का क्षेत्र है क्योंकि इस संस्करण में cache management बहुत बेसिक है, write‑only/delete‑never
- यदि dataset में invalid URLs बहुत हैं, तो resume के समय भी समय बर्बाद हो सकता है क्योंकि अभी bad samples **कभी** URL list से हटाए नहीं जाते
  - **Suggestion:** पहले URL validation task चलाएँ और bad samples हटाएँ।

### Configuration

आवश्यक keys:

- `type: "csv"`
- `csv_caption_column`
- `csv_cache_dir`
- `caption_strategy: "csv"`

```json
[
    {
        "id": "csvtest",
        "type": "csv",
        "csv_caption_column": "caption",
        "csv_file": "/Volumes/ml/dataset/test_list.csv",
        "csv_cache_dir": "/Volumes/ml/cache/csv/test",
        "cache_dir_vae": "/Volumes/ml/cache/vae/sdxl",
        "caption_strategy": "csv",
        "image_embeds": "image-embeds",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "resolution_type": "pixel",
        "minimum_image_size": 0,
        "disabled": false,
        "skip_file_discovery": "",
        "preserve_data_backend_cache": false
    },
    {
      "id": "image-embeds",
      "type": "local"
    },
    {
        "id": "text-embeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/Volumes/ml/cache/text/sdxl",
        "disabled": false,
        "preserve_data_backend_cache": false,
        "skip_file_discovery": "",
        "write_batch_size": 128
    }
]
```

## Parquet caption strategy / JSON Lines datasets

> ⚠️ यह advanced फीचर है और अधिकांश users के लिए आवश्यक नहीं होगा।

जब आप सैकड़ों हजारों या लाखों images वाले बहुत बड़े dataset पर मॉडल train करते हैं, तो metadata को txt files की बजाय parquet database में स्टोर करना सबसे तेज़ होता है — खासकर जब training data S3 bucket पर हो।

Parquet caption strategy आपको अपनी सभी files को उनके `id` मान के आधार पर नाम देने देता है, और captions को config मान के जरिए बदलने देता है, बजाय कई text files अपडेट करने या captions बदलने के लिए files rename करने के।

यहाँ [photo-concept-bucket](https://huggingface.co/datasets/bghira/photo-concept-bucket) dataset के captions और data का उपयोग करने वाला एक उदाहरण dataloader कॉन्फ़िगरेशन है:

```json
{
  "id": "photo-concept-bucket",
  "type": "local",
  "instance_data_dir": "/models/training/datasets/photo-concept-bucket-downloads",
  "caption_strategy": "parquet",
  "metadata_backend": "parquet",
  "parquet": {
    "path": "photo-concept-bucket.parquet",
    "filename_column": "id",
    "caption_column": "cogvlm_caption",
    "fallback_caption_column": "tags",
    "width_column": "width",
    "height_column": "height",
    "identifier_includes_extension": false
  },
  "resolution": 1.0,
  "minimum_image_size": 0.75,
  "maximum_image_size": 2.0,
  "target_downsample_size": 1.5,
  "prepend_instance_prompt": false,
  "instance_prompt": null,
  "only_instance_prompt": false,
  "disable": false,
  "cache_dir_vae": "/models/training/vae_cache/photo-concept-bucket",
  "probability": 1.0,
  "skip_file_discovery": "",
  "preserve_data_backend_cache": false,
  "vae_cache_clear_each_epoch": true,
  "repeats": 1,
  "crop": true,
  "crop_aspect": "closest",
  "crop_style": "random",
  "crop_aspect_buckets": [1.0, 0.75, 1.23],
  "resolution_type": "area"
}
```

इस कॉन्फ़िगरेशन में:

- `caption_strategy` को `parquet` पर सेट किया गया है।
- `metadata_backend` को `parquet` पर सेट किया गया है।
- एक नया सेक्शन `parquet` परिभाषित किया जाना चाहिए:
  - `path` parquet या JSONL फ़ाइल का path है।
  - `filename_column` table में filenames रखने वाले column का नाम है। यहाँ हम numeric `id` column का उपयोग कर रहे हैं (अनुशंसित)।
  - `caption_column` captions रखने वाले column का नाम है। यहाँ `cogvlm_caption` column का उपयोग किया गया है। LAION datasets में यह TEXT field होगा।
  - `width_column` और `height_column` strings, int, या single‑entry Series जैसे प्रकार वाले columns हो सकते हैं, जो image के वास्तविक dimensions बताते हैं। इससे dataset preparation time बेहतर होता है क्योंकि हमें dimensions खोजने के लिए वास्तविक images access नहीं करनी पड़तीं।
  - `fallback_caption_column` एक वैकल्पिक column नाम है जिसमें fallback captions होते हैं। ये primary caption field खाली होने पर उपयोग होते हैं। यहाँ `tags` column उपयोग किया गया है।
  - `identifier_includes_extension` को `true` सेट करें जब filename column में image extension शामिल हो। अन्यथा extension `.png` मान लिया जाएगा। Table के filename column में extensions शामिल करना अनुशंसित है।

> ⚠️ Parquet support capability केवल captions पढ़ने तक सीमित है। आपको अपनी image samples के लिए अलग से data source populate करना होगा, जहाँ filenames "{id}.png" हों। ideas के लिए [scripts/toolkit/datasets](scripts/toolkit/datasets) डायरेक्टरी के scripts देखें।

अन्य dataloader कॉन्फ़िगरेशन की तरह:

- `prepend_instance_prompt` और `instance_prompt` सामान्य रूप से व्यवहार करते हैं।
- प्रशिक्षण runs के बीच किसी sample के caption को अपडेट करने पर नया embed cache होगा, लेकिन पुराना (orphaned) unit हटेगा नहीं।
- जब किसी dataset में image मौजूद नहीं है, तो उसका filename caption के रूप में उपयोग होगा और error emit होगा।

## Cloud dataset के साथ local cache

महंगे local NVMe storage का अधिकतम उपयोग करने के लिए, आप image files (png, jpg) को S3 bucket पर रख सकते हैं, और local storage को text encoder(s) और VAE (यदि लागू हों) से निकाले गए feature maps cache करने के लिए उपयोग कर सकते हैं।

इस उदाहरण कॉन्फ़िगरेशन में:

- Image data S3‑compatible bucket पर स्टोर होता है
- VAE data /local/path/to/cache/vae पर स्टोर होता है
- Text embeds /local/path/to/cache/textencoder पर स्टोर होते हैं

> ⚠️ `resolution` और `crop` जैसे अन्य dataset options कॉन्फ़िगर करना न भूलें

```json
[
    {
        "id": "data",
        "type": "aws",
        "aws_bucket_name": "text-vae-embeds",
        "aws_endpoint_url": "https://storage.provider.example",
        "aws_access_key_id": "exampleAccessKey",
        "aws_secret_access_key": "exampleSecretKey",
        "aws_region_name": null,
        "cache_dir_vae": "/local/path/to/cache/vae/",
        "caption_strategy": "parquet",
        "metadata_backend": "parquet",
        "parquet": {
            "path": "train.parquet",
            "caption_column": "caption",
            "filename_column": "filename",
            "width_column": "width",
            "height_column": "height",
            "identifier_includes_extension": true
        },
        "preserve_data_backend_cache": false,
        "image_embeds": "vae-embed-storage"
    },
    {
        "id": "vae-embed-storage",
        "type": "local",
        "dataset_type": "image_embeds"
    },
    {
        "id": "text-embed-storage",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/local/path/to/cache/textencoder/",
        "write_batch_size": 128
    }
]
```

**Note:** `image_embeds` dataset में data paths सेट करने के विकल्प नहीं होते। ये image backend पर `cache_dir_vae` के जरिए कॉन्फ़िगर किए जाते हैं।

### Hugging Face Datasets Support

SimpleTuner अब Hugging Face Hub से सीधे datasets लोड करना सपोर्ट करता है, बिना पूरे dataset को local डाउनलोड किए। यह experimental फीचर निम्न के लिए आदर्श है:

- Hugging Face पर होस्ट किए गए बड़े datasets
- built‑in metadata और quality assessments वाले datasets
- local storage आवश्यकताओं के बिना तेज़ experimentation

इस फीचर पर विस्तृत दस्तावेज़ के लिए [यह दस्तावेज़](HUGGINGFACE_DATASETS.md) देखें।

Hugging Face dataset का उपयोग करने का एक बेसिक उदाहरण: अपने dataloader configuration में `"type": "huggingface"` सेट करें:

```json
{
  "id": "my-hf-dataset",
  "type": "huggingface",
  "dataset_name": "username/dataset-name",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "caption_column": "caption",
  "image_column": "image"
}
```

## Custom aspect ratio‑to‑resolution mapping

जब SimpleTuner पहली बार चलता है, तो यह resolution‑specific aspect mapping lists बनाता है जो decimal aspect‑ratio मान को target pixel size से जोड़ते हैं।

आप custom mapping बना सकते हैं जो trainer को उसके अपने calculations की बजाय आपकी चुनी हुई target resolution के अनुसार adjust करने पर मजबूर करे। यह functionality आपके जोखिम पर है, क्योंकि गलत कॉन्फ़िगरेशन से नुकसान हो सकता है।

Custom mapping बनाने के लिए:

- उदाहरण (नीचे) के अनुसार एक फ़ाइल बनाएँ
- फ़ाइल का नाम `aspect_ratio_map-{resolution}.json` फ़ॉर्मैट में रखें
  - `resolution=1.0` / `resolution_type=area` के लिए mapping filename `aspect_resolution_map-1.0.json` होगा
- फ़ाइल को `--output_dir` में रखें
  - यही वह स्थान है जहाँ आपके checkpoints और validation images मिलेंगे
- किसी अतिरिक्त कॉन्फ़िगरेशन flags या options की आवश्यकता नहीं है। सही नाम और स्थान होने पर यह स्वतः discover और उपयोग हो जाएगी।

### Example mapping configuration

यह SimpleTuner द्वारा generated एक example aspect ratio mapping है। आपको इसे manual कॉन्फ़िगर करने की आवश्यकता नहीं है, क्योंकि trainer इसे स्वतः बनाएगा। लेकिन resulting resolutions पर पूरी तरह नियंत्रण के लिए, ये mappings modification के लिए शुरुआती बिंदु के रूप में दी जाती हैं।

- dataset में 1 मिलियन से अधिक images थीं
- dataloader `resolution` `1.0` पर सेट था
- dataloader `resolution_type` `area` पर सेट था

यह सबसे सामान्य कॉन्फ़िगरेशन है और 1 megapixel मॉडल के लिए trainable aspect buckets की सूची है।

```json
{
    "0.07": [320, 4544],    "0.38": [640, 1664],    "0.88": [960, 1088],    "1.92": [1472, 768],    "3.11": [1792, 576],    "5.71": [2560, 448],
    "0.08": [320, 3968],    "0.4": [640, 1600],     "0.89": [1024, 1152],   "2.09": [1472, 704],    "3.22": [1856, 576],    "6.83": [2624, 384],
    "0.1": [320, 3328],     "0.41": [704, 1728],    "0.94": [1024, 1088],   "2.18": [1536, 704],    "3.33": [1920, 576],    "7.0": [2688, 384],
    "0.11": [384, 3520],    "0.42": [704, 1664],    "1.06": [1088, 1024],   "2.27": [1600, 704],    "3.44": [1984, 576],    "8.0": [3072, 384],
    "0.12": [384, 3200],    "0.44": [704, 1600],    "1.12": [1152, 1024],   "2.5": [1600, 640],     "3.88": [1984, 512],
    "0.14": [384, 2688],    "0.46": [704, 1536],    "1.13": [1088, 960],    "2.6": [1664, 640],     "4.0": [2048, 512],
    "0.15": [448, 3008],    "0.48": [704, 1472],    "1.2": [1152, 960],     "2.7": [1728, 640],     "4.12": [2112, 512],
    "0.16": [448, 2816],    "0.5": [768, 1536],     "1.36": [1216, 896],    "2.8": [1792, 640],     "4.25": [2176, 512],
    "0.19": [448, 2304],    "0.52": [768, 1472],    "1.46": [1216, 832],    "3.11": [1792, 576],    "4.38": [2240, 512],
    "0.24": [512, 2112],    "0.55": [768, 1408],    "1.54": [1280, 832],    "3.22": [1856, 576],    "5.0": [2240, 448],
    "0.26": [512, 1984],    "0.59": [832, 1408],    "1.83": [1408, 768],    "3.33": [1920, 576],    "5.14": [2304, 448],
    "0.29": [576, 1984],    "0.62": [832, 1344],    "1.92": [1472, 768],    "3.44": [1984, 576],    "5.71": [2560, 448],
    "0.31": [576, 1856],    "0.65": [832, 1280],    "2.09": [1472, 704],    "3.88": [1984, 512],    "6.83": [2624, 384],
    "0.34": [640, 1856],    "0.68": [832, 1216],    "2.18": [1536, 704],    "4.0": [2048, 512],     "7.0": [2688, 384],
    "0.38": [640, 1664],    "0.74": [896, 1216],    "2.27": [1600, 704],    "4.12": [2112, 512],    "8.0": [3072, 384],
    "0.4": [640, 1600],     "0.83": [960, 1152],    "2.5": [1600, 640],     "4.25": [2176, 512],
    "0.41": [704, 1728],    "0.88": [960, 1088],    "2.6": [1664, 640],     "4.38": [2240, 512],
    "0.42": [704, 1664],    "0.89": [1024, 1152],   "2.7": [1728, 640],     "5.0": [2240, 448],
    "0.44": [704, 1600],    "0.94": [1024, 1088],   "2.8": [1792, 640],     "5.14": [2304, 448]
}
```

Stable Diffusion 1.5 / 2.0-base (512px) मॉडल्स के लिए, निम्न mapping काम करेगा:

```json
{
    "1.3": [832, 640], "1.0": [768, 768], "2.0": [1024, 512],
    "0.64": [576, 896], "0.77": [640, 832], "0.79": [704, 896],
    "0.53": [576, 1088], "1.18": [832, 704], "0.85": [704, 832],
    "0.56": [576, 1024], "0.92": [704, 768], "1.78": [1024, 576],
    "1.56": [896, 576], "0.67": [640, 960], "1.67": [960, 576],
    "0.5": [512, 1024], "1.09": [768, 704], "1.08": [832, 768],
    "0.44": [512, 1152], "0.71": [640, 896], "1.4": [896, 640],
    "0.39": [448, 1152], "2.25": [1152, 512], "2.57": [1152, 448],
    "0.4": [512, 1280], "3.5": [1344, 384], "2.12": [1088, 512],
    "0.3": [448, 1472], "2.71": [1216, 448], "8.25": [2112, 256],
    "0.29": [384, 1344], "2.86": [1280, 448], "6.2": [1984, 320],
    "0.6": [576, 960]
}
```
