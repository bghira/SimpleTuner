# TREAD प्रशिक्षण दस्तावेज़

> ⚠️ **प्रयोगात्मक फीचर**: SimpleTuner में TREAD समर्थन हाल ही में लागू हुआ है। यह कार्यात्मक है, लेकिन optimal कॉन्फ़िगरेशन अभी तलाशे जा रहे हैं और कुछ व्यवहार भविष्य की रिलीज़ में बदल सकते हैं।

## अवलोकन

TREAD (Token Routing for Efficient Architecture-agnostic Diffusion Training) एक प्रशिक्षण तेज़ करने की विधि है, जो transformer layers में tokens को बुद्धिमानी से route करके diffusion मॉडल प्रशिक्षण तेज़ करती है। कुछ layers में केवल सबसे महत्वपूर्ण tokens को प्रोसेस करके, TREAD कंप्यूट लागत को कम करते हुए मॉडल गुणवत्ता बनाए रख सकता है।

[Krause आदि (2025)](https://arxiv.org/abs/2501.04765) के शोध पर आधारित, TREAD निम्न तरीके से प्रशिक्षण गति बढ़ाता है:
- हर transformer layer में कौन से tokens प्रोसेस होंगे, इसका डायनामिक चयन
- skip connections के जरिए सभी tokens में gradient flow बनाए रखना
- importance‑आधारित routing निर्णयों का उपयोग

speedup सीधे `selection_ratio` के अनुपात में होता है — जितना 1.0 के करीब, उतने अधिक tokens drop होते हैं और प्रशिक्षण उतना तेज़ होता है।

## TREAD कैसे काम करता है

### मूल अवधारणा

प्रशिक्षण के दौरान, TREAD:
1. **Tokens route करता है** - चुने गए transformer layers के लिए, importance के आधार पर tokens का subset चुनता है
2. **Subset प्रोसेस करता है** - केवल चुने गए tokens expensive attention और MLP ऑपरेशन्स से गुजरते हैं
3. **पूरी sequence बहाल करता है** - प्रोसेसिंग के बाद full token sequence बहाल होता है और सभी tokens तक gradients पहुँचते हैं

### Token चयन

Tokens को L1‑norm (importance score) के आधार पर चुना जाता है, और exploration के लिए वैकल्पिक randomization भी हो सकता है:
- higher importance वाले tokens के बने रहने की संभावना अधिक होती है
- importance‑based और random selection का मिश्रण विशिष्ट patterns पर overfitting रोकता है
- force‑keep masks कुछ tokens (जैसे masked regions) को कभी drop न होने देते हैं

## कॉन्फ़िगरेशन

### बेसिक सेटअप

SimpleTuner में TREAD प्रशिक्षण सक्षम करने के लिए, अपनी कॉन्फ़िगरेशन में यह जोड़ें:

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 2,
        "end_layer_idx": 5
      }
    ]
  }
}
```

### Route कॉन्फ़िगरेशन

हर route एक window तय करता है जहाँ token routing सक्रिय होती है:
- `selection_ratio`: drop किए जाने वाले tokens का हिस्सा (0.5 = 50% tokens रखें)
- `start_layer_idx`: पहला layer जहाँ routing शुरू होती है (0‑indexed)
- `end_layer_idx`: आख़िरी layer जहाँ routing सक्रिय रहती है

Negative indices समर्थित हैं: `-1` आख़िरी layer को दर्शाता है।

### उन्नत उदाहरण

अलग‑अलग selection ratios के साथ कई routing windows:

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.3,
        "start_layer_idx": 1,
        "end_layer_idx": 3
      },
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 4,
        "end_layer_idx": 8
      },
      {
        "selection_ratio": 0.7,
        "start_layer_idx": -4,
        "end_layer_idx": -1
      }
    ]
  }
}
```

## संगतता

### समर्थित मॉडल
- **FLUX Dev/Kontext, Wan, AuraFlow, PixArt, और SD3** - फिलहाल केवल यही मॉडल families समर्थित हैं
- अन्य diffusion transformers के लिए भविष्य में समर्थन योजनाबद्ध है

### इनके साथ अच्छा काम करता है
- **Masked Loss Training** - mask/segmentation conditioning के साथ TREAD masked regions को स्वतः preserve करता है
- **Multi‑GPU Training** - distributed प्रशिक्षण setups के साथ संगत
- **Quantized Training** - int8/int4/NF4 quantization के साथ उपयोग किया जा सकता है

### सीमाएँ
- केवल training के दौरान सक्रिय (inference में नहीं)
- gradient computation आवश्यक (eval mode में काम नहीं करेगा)
- फिलहाल FLUX और Wan‑specific implementation; Lumina2 और अन्य architectures में उपलब्ध नहीं

## प्रदर्शन विचार

### गति लाभ
- training speedup `selection_ratio` के अनुपात में है (1.0 के करीब = अधिक tokens drop = तेज़ प्रशिक्षण)
- **सबसे बड़े speedups लंबे वीडियो इनपुट और उच्च resolutions पर मिलते हैं** क्योंकि attention की complexity O(n²) है
- आम तौर पर 20‑40% speedup, लेकिन परिणाम कॉन्फ़िगरेशन पर निर्भर करते हैं
- masked loss training के साथ speedup कम हो जाता है क्योंकि masked tokens drop नहीं हो सकते

### गुणवत्ता trade‑offs
- **अधिक token dropping से शुरुआती loss अधिक होता है** जब LoRA/LoKr training शुरू होती है
- loss आमतौर पर जल्दी ठीक हो जाता है और images जल्दी normalize हो जाती हैं, जब तक selection ratio बहुत ऊँचा न हो
  - यह नेटवर्क का intermediary layers में कम tokens के अनुसार adjust होना हो सकता है
- conservative ratios (0.1‑0.25) आमतौर पर गुणवत्ता बनाए रखते हैं
- aggressive ratios (>0.35) convergence को ज़रूर प्रभावित करेंगे

### LoRA‑विशिष्ट विचार
- प्रदर्शन डेटा‑dependent हो सकता है — optimal routing configs के लिए और exploration चाहिए
- शुरुआती loss spike LoRA/LoKr में full fine‑tuning की तुलना में अधिक दिखता है

### अनुशंसित सेटिंग्स

speed और quality का संतुलन:
```json
{
  "routes": [
    {"selection_ratio": 0.5, "start_layer_idx": 2, "end_layer_idx": -2}
  ]
}
```

अधिकतम गति (बड़े loss spike की अपेक्षा करें):
```json
{
  "routes": [
    {"selection_ratio": 0.7, "start_layer_idx": 1, "end_layer_idx": -1}
  ]
}
```

उच्च‑resolution प्रशिक्षण (1024px+):
```json
{
  "routes": [
    {"selection_ratio": 0.6, "start_layer_idx": 2, "end_layer_idx": -3}
  ]
}
```

## तकनीकी विवरण

### Router implementation

TREAD router (`TREADRouter` class) संभालता है:
- L1‑norm के जरिए token importance calculation
- efficient routing के लिए permutation generation
- gradient‑preserving token restoration

### Attention के साथ एकीकरण

TREAD routed sequence से मेल कराने के लिए rotary position embeddings (RoPE) बदलता है:
- text tokens अपनी मूल positions बनाए रखते हैं
- image tokens shuffled/sliced positions का उपयोग करते हैं
- routing के दौरान positional consistency सुनिश्चित होती है
- **Note**: FLUX के लिए RoPE implementation 100% सही नहीं हो सकता, लेकिन व्यवहार में functional लगता है

### Masked Loss संगतता

Masked loss प्रशिक्षण के दौरान:
- mask के भीतर tokens स्वतः force‑keep रहते हैं
- महत्वपूर्ण training signal drop होने से बचता है
- `conditioning_type` में ["mask", "segmentation"] होने पर सक्रिय
- **Note**: इससे speedup कम हो जाता है क्योंकि अधिक tokens प्रोसेस करने पड़ते हैं

## ज्ञात समस्याएँ और सीमाएँ

### Implementation स्थिति
- **प्रयोगात्मक फीचर** - TREAD समर्थन नया है और इसमें अज्ञात समस्याएँ हो सकती हैं
- **RoPE handling** - token routing के लिए rotary position embedding implementation पूरी तरह सही नहीं हो सकती
- **सीमित परीक्षण** - optimal routing configs का व्यापक परीक्षण नहीं हुआ है

### प्रशिक्षण व्यवहार
- **शुरुआती loss spike** - TREAD के साथ LoRA/LoKr training शुरू करने पर शुरुआती loss ऊँचा होता है लेकिन जल्दी ठीक होता है
- **LoRA प्रदर्शन** - कुछ कॉन्फ़िगरेशन में LoRA training थोड़ी धीमी हो सकती है
- **कॉन्फ़िगरेशन sensitivity** - प्रदर्शन routing कॉन्फ़िगरेशन विकल्पों पर बहुत निर्भर है

### ज्ञात बग (ठीक किए गए)
- Masked loss training पहले के संस्करणों में टूट गया था लेकिन सही model flavor checking (`kontext` guard) के साथ ठीक कर दिया गया है

## समस्या समाधान

### सामान्य समस्याएँ

**"TREAD training requires you to configure the routes"**
- सुनिश्चित करें कि `tread_config` में `routes` array शामिल है
- प्रत्येक route के लिए `selection_ratio`, `start_layer_idx`, और `end_layer_idx` होना चाहिए

**उम्मीद से धीमा प्रशिक्षण**
- जांचें कि routes सार्थक layer ranges को कवर करते हैं
- अधिक aggressive selection ratios पर विचार करें
- जांचें कि gradient checkpointing conflict तो नहीं कर रहा
- LoRA training के लिए कुछ slowdown अपेक्षित है — अलग routing configs आज़माएँ

**LoRA/LoKr के साथ उच्च शुरुआती loss**
- यह अपेक्षित व्यवहार है — नेटवर्क को कम tokens के अनुसार adapt होना होता है
- loss सामान्यतः कुछ सौ steps में ठीक हो जाता है
- यदि loss में सुधार नहीं हो रहा, `selection_ratio` घटाएँ (अधिक tokens रखें)

**गुणवत्ता में गिरावट**
- selection ratios घटाएँ (अधिक tokens रखें)
- शुरुआती layers (0‑2) या अंतिम layers में routing से बचें
- बढ़ी हुई efficiency के लिए पर्याप्त training data सुनिश्चित करें

## व्यावहारिक उदाहरण

### High‑Resolution प्रशिक्षण (1024px+)
उच्च resolutions पर अधिकतम लाभ के लिए:
```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.6, "start_layer_idx": 2, "end_layer_idx": -3}
    ]
  }
}
```

### LoRA फाइन‑ट्यूनिंग
शुरुआती loss spike को कम करने के लिए conservative config:
```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.4, "start_layer_idx": 3, "end_layer_idx": -4}
    ]
  }
}
```

### Masked Loss Training
Masks के साथ training करते समय, masked regions के tokens सुरक्षित रहते हैं:
```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.7, "start_layer_idx": 2, "end_layer_idx": -2}
    ]
  }
}
```
Note: forced token preservation के कारण वास्तविक speedup 0.7 से कम होगा।

## भविष्य का काम

SimpleTuner में TREAD समर्थन नया होने के कारण, आगे सुधार के कई क्षेत्र हैं:

- **कॉन्फ़िगरेशन optimization** - अलग use cases के लिए optimal routing configs खोजने हेतु अधिक परीक्षण
- **LoRA प्रदर्शन** - कुछ LoRA कॉन्फ़िगरेशन धीमे क्यों होते हैं, इस पर जांच
- **RoPE implementation** - बेहतर correctness के लिए rotary position embedding handling में refinement
- **Extended मॉडल समर्थन** - Flux के बाहर अन्य diffusion transformer architectures के लिए implementation
- **Automated कॉन्फ़िगरेशन** - मॉडल और dataset विशेषताओं के आधार पर optimal routing तय करने के लिए tools

TREAD समर्थन बेहतर बनाने के लिए समुदाय के योगदान और परीक्षण परिणामों का स्वागत है।

## संदर्भ

- [TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training](https://arxiv.org/abs/2501.04765)
- [SimpleTuner Flux Documentation](quickstart/FLUX.md#tread-training)

## Citation

```bibtex
@misc{krause2025treadtokenroutingefficient,
      title={TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training},
      author={Felix Krause and Timy Phan and Vincent Tao Hu and Björn Ommer},
      year={2025},
      eprint={2501.04765},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.04765},
}
```
