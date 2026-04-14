# Scheduled Sampling (Rollout)

## पृष्ठभूमि

मानक diffusion ट्रेनिंग "Teacher Forcing" पर निर्भर करती है। हम एक साफ इमेज लेते हैं, उसमें सटीक मात्रा में शोर जोड़ते हैं, और मॉडल से उस शोर (या velocity/original image) की भविष्यवाणी कराते हैं। मॉडल को दिया गया इनपुट हमेशा "परफ़ेक्ट" शोर वाला होता है—यह बिल्कुल सैद्धांतिक noise schedule पर होता है।

लेकिन inference (generation) के दौरान, मॉडल अपने ही आउटपुट्स पर निर्भर रहता है। यदि वह step $t$ पर एक छोटी गलती करता है, तो वह गलती step $t-1$ में चली जाती है। ये त्रुटियाँ जमा होती हैं, जिससे जेनरेशन वैध इमेज के manifold से भटक सकती है। ट्रेनिंग (परफ़ेक्ट इनपुट्स) और inference (अपूर्ण इनपुट्स) के बीच इस अंतर को **Exposure Bias** कहा जाता है।

**Scheduled Sampling** (यहाँ अक्सर "Rollout" कहा जाता है) इस समस्या को हल करने के लिए मॉडल को उसके अपने आउटपुट्स पर ट्रेन करता है।

## यह कैसे काम करता है

सिर्फ साफ इमेज में शोर जोड़ने के बजाय, ट्रेनिंग लूप कभी-कभी एक छोटा inference session चलाता है:

1.  **Target timestep** $t$ चुनें (वही step जिस पर हम ट्रेन करना चाहते हैं)।
2.  **Source timestep** $t+k$ चुनें (schedule में पीछे की ओर एक अधिक noisy step)।
3.  मॉडल के *वर्तमान* वेट्स का उपयोग करके $t+k$ से $t$ तक वास्तव में जेनरेट (denoise) करें।
4.  step $t$ पर इस self-generated, थोड़ा imperfect latent को ट्रेनिंग पास के लिए इनपुट के रूप में उपयोग करें।

ऐसा करने से मॉडल उन artifacts और errors वाले इनपुट्स देखता है जो वह स्वयं पैदा करता है। वह सीखता है, "अच्छा, मैंने यह गलती की; इसे ऐसे सुधारता हूँ," जिससे जेनरेशन फिर से वैध पथ की ओर लौट आती है।

## कॉन्फ़िगरेशन

यह फीचर प्रयोगात्मक है और कम्प्यूटेशनल ओवरहेड जोड़ता है, लेकिन यह prompt adherence और संरचनात्मक स्थिरता में काफी सुधार कर सकता है, खासकर छोटे datasets (Dreambooth) पर।

इसे सक्षम करने के लिए आपको non-zero `max_step_offset` कॉन्फ़िगर करना होगा।

### बेसिक सेटअप

अपने `config.json` में यह जोड़ें:

```json
{
  "scheduled_sampling_max_step_offset": 10,
  "scheduled_sampling_probability": 1.0,
  "scheduled_sampling_sampler": "unipc"
}
```

### विकल्प संदर्भ

#### `scheduled_sampling_max_step_offset` (Integer)
**डिफ़ॉल्ट:** `0` (निष्क्रिय)
रोलआउट की अधिकतम स्टेप संख्या। यदि इसे `10` सेट किया है, तो trainer हर सैंपल के लिए 0 से 10 के बीच एक यादृच्छिक रोलआउट लंबाई चुनेगा।
> 🟢 **सिफारिश:** छोटा शुरू करें (जैसे `5` से `10`)। छोटे रोलआउट भी मॉडल को त्रुटि सुधार सीखने में मदद करते हैं, बिना ट्रेनिंग को बहुत धीमा किए।

#### `scheduled_sampling_probability` (Float)
**डिफ़ॉल्ट:** `0.0`
किसी भी batch item के रोलआउट से गुजरने की संभावना (0.0 से 1.0)।
*   `1.0`: हर सैंपल रोलआउट होता है (सबसे अधिक compute)।
*   `0.5`: 50% सैंपल्स मानक ट्रेनिंग, 50% रोलआउट।

#### `scheduled_sampling_ramp_steps` (Integer)
**डिफ़ॉल्ट:** `0`
यदि सेट किया गया, तो probability `scheduled_sampling_prob_start` (डिफ़ॉल्ट 0.0) से `scheduled_sampling_prob_end` (डिफ़ॉल्ट 0.5) तक इतने global steps में linearly ramp होगी।
> 🟢 **टिप:** यह एक "warmup" की तरह काम करता है। इससे मॉडल पहले basic denoising सीखता है, फिर धीरे-धीरे अपनी गलतियों को ठीक करने वाला कठिन कार्य जोड़ता है।

#### `scheduled_sampling_sampler` (String)
**डिफ़ॉल्ट:** `unipc`
रोलआउट जेनरेशन स्टेप्स के लिए उपयोग किया गया solver।
*   **विकल्प:** `unipc` (अनुशंसित, तेज़ और सटीक), `euler`, `dpm`।
*   इन छोटे sampling bursts के लिए `unipc` आमतौर पर गति और सटीकता का सर्वश्रेष्ठ संतुलन है।

### Flow Matching + ReflexFlow

Flow-matching मॉडल्स (`--prediction_type flow_matching`) के लिए scheduled sampling अब ReflexFlow-style exposure bias mitigation सपोर्ट करता है:

*   `scheduled_sampling_reflexflow`: rollout के दौरान ReflexFlow enhancements सक्षम करें (flow-matching मॉडल्स के लिए scheduled sampling सक्रिय होने पर स्वतः सक्षम; opt out करने के लिए `--scheduled_sampling_reflexflow=false` दें)।
*   `scheduled_sampling_reflexflow_alpha`: exposure-bias आधारित loss weight का स्केल (frequency compensation)।
*   `scheduled_sampling_reflexflow_beta1`: directional anti-drift regularizer का स्केल (पेपर जैसा डिफ़ॉल्ट 10.0)।
*   `scheduled_sampling_reflexflow_beta2`: frequency-compensated loss का स्केल (डिफ़ॉल्ट 1.0)।

ये आपके पहले से गणना किए गए rollout predictions/latents को reuse करते हैं, अतिरिक्त gradient pass से बचते हैं, और biased rollouts को clean trajectory के साथ align रखते हैं जबकि denoising की शुरुआत में missing low-frequency components पर जोर देते हैं।

### परफॉर्मेंस प्रभाव

> ⚠️ **चेतावनी:** रोलआउट सक्षम करने पर ट्रेनिंग लूप *के अंदर* मॉडल को inference मोड में चलाना पड़ता है।
>
> यदि आप `max_step_offset=10` सेट करते हैं, तो मॉडल प्रति ट्रेनिंग स्टेप तक 10 अतिरिक्त forward passes चला सकता है। इससे आपका `it/s` (iterations per second) घटेगा। ट्रेनिंग गति और गुणवत्ता के बीच संतुलन बनाने के लिए `scheduled_sampling_probability` समायोजित करें।
