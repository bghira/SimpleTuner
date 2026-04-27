# SimpleTuner 💹

> ℹ️ `report_to`, `push_to_hub`, या वेबहुक्स जिन्हें हाथ से कॉन्फ़िगर करना होता है, उनके opt-in फ़्लैग के अलावा किसी भी तीसरे पक्ष को कोई डेटा नहीं भेजा जाता।

**SimpleTuner** सरलता पर केंद्रित है, ताकि कोड आसानी से समझा जा सके। यह कोडबेस एक साझा अकादमिक अभ्यास है, और योगदान का स्वागत है।

यदि आप हमारे समुदाय में शामिल होना चाहते हैं, तो आप हमें Terminus Research Group के [Discord](https://discord.gg/JGkSwEbjRb) पर पा सकते हैं।
यदि आपके कोई प्रश्न हों, तो कृपया वहाँ बेझिझक संपर्क करें।

<img width="1944" height="1657" alt="image" src="https://github.com/user-attachments/assets/af3a24ec-7347-4ddf-8edf-99818a246de1" />


## अनुक्रमणिका

- [डिज़ाइन दर्शन](#design-philosophy)
- [ट्यूटोरियल](#tutorial)
- [विशेषताएँ](#features)
  - [मुख्य प्रशिक्षण विशेषताएँ](#core-training-features)
  - [मॉडल आर्किटेक्चर समर्थन](#model-architecture-support)
  - [उन्नत प्रशिक्षण तकनीकें](#advanced-training-techniques)
  - [मॉडल-विशिष्ट विशेषताएँ](#model-specific-features)
  - [क्विकस्टार्ट गाइड](#quickstart-guides)
- [हार्डवेयर आवश्यकताएँ](#hardware-requirements)
- [टूलकिट](#toolkit)
- [सेटअप](#setup)
- [समस्या समाधान](#troubleshooting)

## डिज़ाइन दर्शन {#design-philosophy}

- **सरलता**: अधिकांश उपयोग मामलों के लिए अच्छे डिफ़ॉल्ट सेटिंग्स रखने का लक्ष्य, ताकि कम टिंकरिंग करनी पड़े।
- **बहुमुखीपन**: छोटे डेटासेट से लेकर बड़े संग्रह तक छवियों की विस्तृत मात्रा संभालने के लिए डिज़ाइन।
- **अत्याधुनिक विशेषताएँ**: केवल वे सुविधाएँ शामिल करता है जो प्रभावी साबित हुई हैं, बिना परीक्षण विकल्पों को जोड़ने से बचते हुए।

## ट्यूटोरियल {#tutorial}

नया [वेब UI ट्यूटोरियल](/documentation/webui/TUTORIAL.md) या [क्लास कमांड‑लाइन ट्यूटोरियल](/documentation/TUTORIAL.md) शुरू करने से पहले कृपया इस README को पूरी तरह देखें, क्योंकि इसमें वह महत्वपूर्ण जानकारी है जो आपको पहले जाननी चाहिए।

यदि आप पूरी डॉक्यूमेंटेशन पढ़े बिना या किसी वेब इंटरफेस का उपयोग किए बिना मैन्युअल रूप से क्विक स्टार्ट करना चाहते हैं, तो [क्विक स्टार्ट](/documentation/QUICKSTART.md) गाइड का उपयोग करें।

मेमोरी‑सीमित सिस्टम्स के लिए, [DeepSpeed दस्तावेज़](/documentation/DEEPSPEED.md) देखें, जो बताता है कि 🤗Accelerate के जरिए Microsoft के DeepSpeed को optimiser state offload के लिए कैसे कॉन्फ़िगर करें। DTensor‑आधारित sharding और context parallelism के लिए, [FSDP2 गाइड](/documentation/FSDP2.md) पढ़ें, जो SimpleTuner के भीतर नया FullyShardedDataParallel v2 वर्कफ़्लो कवर करता है।

मल्टी‑नोड वितरित प्रशिक्षण के लिए, [यह गाइड](/documentation/DISTRIBUTED.md) INSTALL और Quickstart गाइड्स की कॉन्फ़िग्स को मल्टी‑नोड प्रशिक्षण के लिए अनुकूल बनाने में मदद करेगा, और अरबों सैंपल्स वाले इमेज डेटासेट के लिए ऑप्टिमाइज़ करने में सहायक होगा।

---

## विशेषताएँ {#features}

SimpleTuner कई diffusion मॉडल आर्किटेक्चर में एक‑समान फीचर उपलब्धता के साथ व्यापक प्रशिक्षण समर्थन देता है:

### मुख्य प्रशिक्षण विशेषताएँ {#core-training-features}

- **उपयोगकर्ता‑अनुकूल वेब UI** - एक सुव्यवस्थित डैशबोर्ड से अपना पूरा प्रशिक्षण जीवनचक्र प्रबंधित करें
- **मल्टी‑मोडल प्रशिक्षण** - **छवि, वीडियो, और ऑडियो** जनरेटिव मॉडल्स के लिए एकीकृत पाइपलाइन
- **मल्टी‑GPU प्रशिक्षण** - स्वचालित ऑप्टिमाइज़ेशन के साथ कई GPUs पर वितरित प्रशिक्षण
- **उन्नत कैशिंग** - तेज़ प्रशिक्षण के लिए छवि, वीडियो, ऑडियो, और कैप्शन एम्बेडिंग्स डिस्क पर कैश
- **CaptionFlow integration** - Web UI job queue से local GPUs पर [bghira/CaptionFlow](https://github.com/bghira/CaptionFlow) के साथ dataset captions generate करें; [CaptionFlow integration guide](/documentation/CAPTIONFLOW.hi.md) देखें
- **आस्पेक्ट बकेटिंग** - विविध छवि/वीडियो आकारों और आस्पेक्ट अनुपातों का समर्थन
- **कॉन्सेप्ट स्लाइडर्स** - LoRA/LyCORIS/full (LyCORIS `full` के जरिए) के लिए स्लाइडर‑फ्रेंडली टार्गेटिंग, पॉज़िटिव/नेगेटिव/न्यूट्रल सैम्पलिंग और प्रति‑प्रॉम्प्ट strength; [Slider LoRA गाइड](/documentation/SLIDER_LORA.md) देखें
- **मेमोरी ऑप्टिमाइज़ेशन** - अधिकांश मॉडल 24G GPU पर प्रशिक्षित हो सकते हैं, कई ऑप्टिमाइज़ेशन के साथ 16G पर भी
- **DeepSpeed & FSDP2 एकीकरण** - optim/grad/parameter sharding, context parallel attention, gradient checkpointing, और optimizer state offload के साथ छोटे GPUs पर बड़े मॉडल प्रशिक्षित करें
- **S3 प्रशिक्षण** - क्लाउड स्टोरेज (Cloudflare R2, Wasabi S3) से सीधे प्रशिक्षण
- **EMA समर्थन** - बेहतर स्थिरता और गुणवत्ता के लिए exponential moving average weights
- **कस्टम experiment trackers** - `simpletuner/custom-trackers` में `accelerate.GeneralTracker` रखें और `--report_to=custom-tracker --custom_tracker=<name>` उपयोग करें

### मल्टी‑यूज़र और एंटरप्राइज विशेषताएँ

SimpleTuner एक पूर्ण मल्टी‑यूज़र प्रशिक्षण प्लेटफ़ॉर्म के साथ एंटरप्राइज‑ग्रेड फीचर्स शामिल करता है — **मुफ़्त और ओपन सोर्स, हमेशा के लिए**।

- **वर्कर ऑर्केस्ट्रेशन** - वितरित GPU वर्कर्स रजिस्टर करें जो केंद्रीय पैनल से auto‑connect होते हैं और SSE के जरिए जॉब डिस्पैच प्राप्त करते हैं; ephemeral (क्लाउड‑लॉन्च्ड) और persistent (हमेशा‑ऑन) वर्कर्स का समर्थन; [Worker Orchestration Guide](/documentation/experimental/server/WORKERS.md) देखें
- **SSO एकीकरण** - LDAP/Active Directory या OIDC providers (Okta, Azure AD, Keycloak, Google) के साथ प्रमाणीकरण; [External Auth Guide](/documentation/experimental/server/EXTERNAL_AUTH.md) देखें
- **भूमिका‑आधारित एक्सेस कंट्रोल** - चार डिफ़ॉल्ट भूमिकाएँ (Viewer, Researcher, Lead, Admin) और 17+ granular permissions; glob patterns के साथ resource rules परिभाषित करके कॉन्फ़िग्स, हार्डवेयर, या providers को टीम‑वार सीमित करें
- **Organizations और Teams** - ceiling‑based quotas के साथ hierarchical multi‑tenant संरचना; org limits पूर्ण अधिकतम तय करते हैं, team limits org सीमा के भीतर काम करते हैं
- **Quotas और Spending Limits** - लागत सीमा (daily/monthly), job concurrency limits, और submission rate limits को org, team, या user स्तर पर लागू करें; कार्रवाइयाँ: block, warn, या approval आवश्यक
- **प्राथमिकता‑आधारित जॉब क्यू** - पाँच प्राथमिकता स्तर (Low → Critical), टीमों के बीच fair‑share scheduling, लंबे इंतज़ार वाली जॉब्स के लिए starvation रोकथाम, और admin priority overrides
- **Approval वर्कफ़्लो** - कॉन्फ़िगरेबल नियम लागत सीमा पार करने वाली जॉब्स, पहली‑बार उपयोगकर्ताओं, या विशिष्ट हार्डवेयर अनुरोधों पर approval ट्रिगर करते हैं; UI, API, या ईमेल रिप्लाई के जरिए approval दें
- **ईमेल नोटिफ़िकेशन** - जॉब स्थिति, approval अनुरोध, quota चेतावनी, और पूर्णता अलर्ट के लिए SMTP/IMAP एकीकरण
- **API Keys और Scoped Permissions** - CI/CD pipelines के लिए expiration और सीमित scope के साथ API keys बनाएं
- **Audit Logging** - compliance के लिए chain verification के साथ सभी उपयोगकर्ता कार्रवाइयों को ट्रैक करें; [Audit Guide](/documentation/experimental/server/AUDIT.md) देखें

डिप्लॉयमेंट विवरण के लिए, [Enterprise Guide](/documentation/experimental/server/ENTERPRISE.md) देखें।

### मॉडल आर्किटेक्चर समर्थन {#model-architecture-support}

| मॉडल | पैरामीटर | PEFT LoRA | Lycoris | फुल-रैंक | ControlNet | क्वांटाइज़ेशन | फ्लो मैचिंग | टेक्स्ट एन्कोडर |
|-------|------------|-----------|---------|-----------|------------|--------------|---------------|---------------|
| **Stable Diffusion XL** | 3.5B | ✓ | ✓ | ✓ | ✓ | int8/nf4 | ✗ | CLIP-L/G |
| **Stable Diffusion 3** | 2B-8B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L/G + T5-XXL |
| **Flux.1** | 12B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L + T5-XXL |
| **Flux.2** | 32B | ✓ | ✓ | ✓* | ✗ | int8/fp8/nf4 | ✓ | Mistral-3 Small |
| **ACE-Step** | 3.5B | ✓ | ✓ | ✓* | ✗ | int8 | ✓ | UMT5 |
| **HeartMuLa** | 3B | ✓ | ✓ | ✓* | ✗ | int8 | ✗ | कोई नहीं |
| **Chroma 1** | 8.9B | ✓ | ✓ | ✓* | ✗ | int8/fp8/nf4 | ✓ | T5-XXL |
| **Auraflow** | 6.8B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | UMT5-XXL |
| **PixArt Sigma** | 0.6B-0.9B | ✗ | ✓ | ✓ | ✓ | int8 | ✗ | T5-XXL |
| **Sana** | 0.6B-4.8B | ✗ | ✓ | ✓ | ✗ | int8 | ✓ | Gemma2-2B |
| **Lumina2** | 2B | ✓ | ✓ | ✓ | ✗ | int8 | ✓ | Gemma2 |
| **Kwai Kolors** | 5B | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ChatGLM-6B |
| **LTX Video** | 5B | ✓ | ✓ | ✓ | ✗ | int8/fp8 | ✓ | T5-XXL |
| **LTX Video 2** | 19B | ✓ | ✓ | ✓* | ✗ | int8/fp8 | ✓ | Gemma3 |
| **Wan Video** | 1.3B-14B | ✓ | ✓ | ✓* | ✗ | int8 | ✓ | UMT5 |
| **HiDream** | 17B (8.5B MoE) | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L + T5-XXL + Llama |
| **Cosmos2** | 2B-14B | ✗ | ✓ | ✓ | ✗ | int8 | ✓ | T5-XXL |
| **OmniGen** | 3.8B | ✓ | ✓ | ✓ | ✗ | int8/fp8 | ✓ | T5-XXL |
| **Qwen Image** | 20B | ✓ | ✓ | ✓* | ✗ | int8/nf4 (req.) | ✓ | T5-XXL |
| **SD 1.x/2.x (Legacy)** | 0.9B | ✓ | ✓ | ✓ | ✓ | int8/nf4 | ✗ | CLIP-L |

*✓ = समर्थित, ✗ = समर्थित नहीं, * = फुल‑रैंक प्रशिक्षण के लिए DeepSpeed आवश्यक*

### उन्नत प्रशिक्षण तकनीकें {#advanced-training-techniques}

- **TREAD** - Kontext प्रशिक्षण सहित ट्रांसफ़ॉर्मर मॉडल्स के लिए token‑wise dropout
- **Masked loss training** - segmentation/depth गाइडेंस के साथ बेहतर convergence
- **Prior regularization** - character consistency के लिए उन्नत प्रशिक्षण स्थिरता
- **Gradient checkpointing** - मेमोरी/स्पीड ऑप्टिमाइज़ेशन के लिए कॉन्फ़िगरेबल intervals
- **Loss functions** - scheduling सपोर्ट के साथ L2, Huber, Smooth L1
- **SNR weighting** - बेहतर प्रशिक्षण डायनेमिक्स के लिए Min‑SNR gamma weighting
- **Group offloading** - Diffusers v0.33+ module‑group CPU/disk staging, वैकल्पिक CUDA streams के साथ
- **Validation adapter sweeps** - validation के दौरान अस्थायी रूप से LoRA adapters (single या JSON presets) जोड़ें ताकि training loop को छुए बिना adapter‑only या comparison renders मापे जा सकें
- **External validation hooks** - built‑in validation pipeline या post‑upload steps को अपने स्क्रिप्ट्स से बदलें, ताकि आप किसी अन्य GPU पर checks चला सकें या artifacts को किसी भी cloud provider पर भेज सकें ([details](/documentation/OPTIONS.md#validation_method))
- **CREPA regularization** - video DiTs के लिए cross‑frame representation alignment ([guide](/documentation/experimental/VIDEO_CREPA.md))
- **LoRA I/O formats** - PEFT LoRAs को standard Diffusers layout या ComfyUI‑style `diffusion_model.*` keys (Flux/Flux2/Lumina2/Z-Image auto‑detect ComfyUI inputs) में load/save करें

### मॉडल‑विशिष्ट विशेषताएँ {#model-specific-features}

- **Flux Kontext** - Flux मॉडल्स के लिए edit conditioning और image‑to‑image प्रशिक्षण
- **PixArt two-stage** - PixArt Sigma के लिए eDiff प्रशिक्षण पाइपलाइन समर्थन
- **Flow matching models** - beta/uniform distributions के साथ उन्नत scheduling
- **HiDream MoE** - Mixture of Experts gate loss augmentation
- **T5 masked training** - Flux और संगत मॉडल्स के लिए बेहतर fine details
- **QKV fusion** - मेमोरी और स्पीड ऑप्टिमाइज़ेशन (Flux, Lumina2)
- **TREAD integration** - अधिकांश मॉडल्स के लिए selective token routing
- **Wan 2.x I2V** - high/low stage presets के साथ 2.1 time‑embedding fallback (Wan quickstart देखें)
- **Classifier-free guidance** - distilled मॉडल्स के लिए वैकल्पिक CFG reintroduction

### क्विकस्टार्ट गाइड {#quickstart-guides}

सभी समर्थित मॉडल्स के लिए विस्तृत क्विकस्टार्ट गाइड उपलब्ध हैं:

- **[TwinFlow Few-Step (RCGM) Guide](/documentation/distillation/TWINFLOW.md)** - few‑step/one‑step generation के लिए RCGM auxiliary loss सक्षम करें (flow मॉडल्स या diff2flow के जरिए diffusion)
- **[Flux.1 Guide](/documentation/quickstart/FLUX.md)** - Kontext editing support और QKV fusion शामिल
- **[Flux.2 Guide](/documentation/quickstart/FLUX2.md)** - **NEW!** Mistral‑3 टेक्स्ट encoder के साथ नवीनतम विशाल Flux मॉडल
- **[Z-Image Guide](/documentation/quickstart/ZIMAGE.md)** - Base/Turbo LoRA with assistant adapter + TREAD acceleration
- **[ACE-Step Guide](/documentation/quickstart/ACE_STEP.md)** - **NEW!** ऑडियो जनरेशन मॉडल प्रशिक्षण (text‑to‑music)
- **[HeartMuLa Guide](/documentation/quickstart/HEARTMULA.md)** - **NEW!** ऑटोरिग्रेसिव ऑडियो जनरेशन मॉडल प्रशिक्षण (text‑to‑audio)
- **[Chroma Guide](/documentation/quickstart/CHROMA.md)** - Lodestone का flow‑matching transformer, Chroma‑specific schedules के साथ
- **[Stable Diffusion 3 Guide](/documentation/quickstart/SD3.md)** - ControlNet के साथ full और LoRA प्रशिक्षण
- **[Stable Diffusion XL Guide](/documentation/quickstart/SDXL.md)** - पूर्ण SDXL प्रशिक्षण पाइपलाइन
- **[Auraflow Guide](/documentation/quickstart/AURAFLOW.md)** - flow‑matching मॉडल प्रशिक्षण
- **[PixArt Sigma Guide](/documentation/quickstart/SIGMA.md)** - दो‑stage समर्थन वाला DiT मॉडल
- **[Sana Guide](/documentation/quickstart/SANA.md)** - lightweight flow‑matching मॉडल
- **[Lumina2 Guide](/documentation/quickstart/LUMINA2.md)** - 2B पैरामीटर flow‑matching मॉडल
- **[Kwai Kolors Guide](/documentation/quickstart/KOLORS.md)** - ChatGLM encoder के साथ SDXL‑आधारित
- **[LongCat-Video Guide](/documentation/quickstart/LONGCAT_VIDEO.md)** - Qwen‑2.5‑VL के साथ flow‑matching text‑to‑video और image‑to‑video
- **[LongCat-Video Edit Guide](/documentation/quickstart/LONGCAT_VIDEO_EDIT.md)** - conditioning‑first flavour (image‑to‑video)
- **[LongCat-Image Guide](/documentation/quickstart/LONGCAT_IMAGE.md)** - Qwen‑2.5‑VL encoder के साथ 6B bilingual flow‑matching मॉडल
- **[LongCat-Image Edit Guide](/documentation/quickstart/LONGCAT_EDIT.md)** - reference latents की आवश्यकता वाला image editing flavour
- **[LTX Video Guide](/documentation/quickstart/LTXVIDEO.md)** - वीडियो diffusion प्रशिक्षण
- **[Hunyuan Video 1.5 Guide](/documentation/quickstart/HUNYUANVIDEO.md)** - SR stages के साथ 8.3B flow‑matching T2V/I2V
- **[Wan Video Guide](/documentation/quickstart/WAN.md)** - TREAD समर्थन के साथ video flow‑matching
- **[HiDream Guide](/documentation/quickstart/HIDREAM.md)** - उन्नत फीचर्स वाला MoE मॉडल
- **[Cosmos2 Guide](/documentation/quickstart/COSMOS2IMAGE.md)** - multi‑modal image generation
- **[OmniGen Guide](/documentation/quickstart/OMNIGEN.md)** - unified image generation मॉडल
- **[Qwen Image Guide](/documentation/quickstart/QWEN_IMAGE.md)** - 20B पैरामीटर large‑scale प्रशिक्षण
- **[Stable Cascade Stage C Guide](/quickstart/STABLE_CASCADE_C.md)** - combined prior+decoder validation के साथ prior LoRAs
- **[Kandinsky 5.0 Image Guide](/documentation/quickstart/KANDINSKY5_IMAGE.md)** - Qwen2.5‑VL + Flux VAE के साथ image generation
- **[Kandinsky 5.0 Video Guide](/documentation/quickstart/KANDINSKY5_VIDEO.md)** - HunyuanVideo VAE के साथ video generation

---

## हार्डवेयर आवश्यकताएँ {#hardware-requirements}

### सामान्य आवश्यकताएँ

- **NVIDIA**: RTX 3080+ अनुशंसित (H200 तक परीक्षण किया गया)
- **AMD**: 7900 XTX 24GB और MI300X सत्यापित (NVIDIA के मुकाबले अधिक मेमोरी उपयोग)
- **Apple**: LoRA प्रशिक्षण के लिए 24GB+ unified memory वाला M3 Max+

### मॉडल आकार के अनुसार मेमोरी दिशानिर्देश

- **बड़े मॉडल (12B+)**: फुल‑रैंक के लिए A100‑80G, LoRA/Lycoris के लिए 24G+
- **मध्यम मॉडल (2B-8B)**: LoRA के लिए 16G+, फुल‑रैंक प्रशिक्षण के लिए 40G+
- **छोटे मॉडल (<2B)**: अधिकांश प्रशिक्षण प्रकारों के लिए 12G+ पर्याप्त

**नोट**: क्वांटाइज़ेशन (int8/fp8/nf4) मेमोरी आवश्यकताओं को काफी कम करता है। मॉडल‑विशिष्ट आवश्यकताओं के लिए व्यक्तिगत [क्विकस्टार्ट गाइड](#quickstart-guides) देखें।

## सेटअप {#setup}

SimpleTuner को अधिकांश उपयोगकर्ताओं के लिए pip के जरिए इंस्टॉल किया जा सकता है:

```bash
# Base installation (CPU-only PyTorch)
pip install simpletuner

# CUDA users (NVIDIA GPUs)
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://download.pytorch.org/whl/cu130

# ROCm users (AMD GPUs)
pip install 'simpletuner[rocm]' --extra-index-url https://download.pytorch.org/whl/rocm7.1 --extra-index-url https://download.pytorch.org/whl/rocm7.1

# Apple Silicon users (M1/M2/M3/M4 Macs)
pip install 'simpletuner[apple]'
```

मैन्युअल इंस्टॉलेशन या डेवलपमेंट सेटअप के लिए, [इंस्टॉलेशन दस्तावेज़](/documentation/INSTALL.md) देखें।

## समस्या समाधान {#troubleshooting}

अधिक विस्तृत जानकारी के लिए `export SIMPLETUNER_LOG_LEVEL=DEBUG` को अपने environment (`config/config.env`) फ़ाइल में जोड़कर debug logs सक्षम करें।

प्रशिक्षण लूप के प्रदर्शन विश्लेषण के लिए, `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` सेट करने पर टाइमस्टैम्प मिलेंगे जो आपकी कॉन्फ़िगरेशन में किसी भी समस्या को हाइलाइट करते हैं।

उपलब्ध विकल्पों की व्यापक सूची के लिए, [इस दस्तावेज़](/documentation/OPTIONS.md) से परामर्श करें।
