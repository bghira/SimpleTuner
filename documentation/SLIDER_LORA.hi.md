# Slider LoRA टार्गेटिंग

इस गाइड में हम SimpleTuner में slider‑style adapter ट्रेन करेंगे। हम Z-Image Turbo का उपयोग करेंगे क्योंकि यह तेज़ ट्रेन होता है, Apache 2.0 लाइसेंस के साथ आता है, और अपने आकार के हिसाब से शानदार परिणाम देता है — distilled weights के साथ भी।

पूर्ण compatibility matrix (LoRA, LyCORIS, full‑rank) के लिए [documentation/QUICKSTART.md](QUICKSTART.md) के Sliders कॉलम को देखें; यह गाइड सभी architectures पर लागू होती है।

Slider targeting standard LoRA, LyCORIS (जिसमें `full` शामिल है), और ControlNet के साथ काम करता है। यह toggle CLI और WebUI दोनों में उपलब्ध है; सब कुछ SimpleTuner में शामिल है, कोई अतिरिक्त install नहीं चाहिए।

## चरण 1 — बेस सेटअप फॉलो करें

- **CLI**: environment, install, hardware नोट्स, और starter `config.json` के लिए `documentation/quickstart/ZIMAGE.md` देखें।
- **WebUI**: trainer wizard चलाने के लिए `documentation/webui/TUTORIAL.md` उपयोग करें; सामान्य रूप से Z-Image Turbo चुनें।

इन गाइड्स का पालन आप dataset कॉन्फ़िगर करने तक कर सकते हैं क्योंकि sliders केवल यह बदलते हैं कि adapters कहाँ लगाए जाते हैं और डेटा कैसे sample होता है।

## चरण 2 — slider targets सक्षम करें

- CLI: `"slider_lora_target": true` जोड़ें (या `--slider_lora_target true` पास करें)।
- WebUI: Model → LoRA Config → Advanced → “Use slider LoRA targets” को चेक करें।

LyCORIS के लिए, `lora_type: "lycoris"` रखें और `lycoris_config.json` के लिए नीचे details सेक्शन में दिए presets उपयोग करें।

## चरण 3 — slider‑friendly datasets बनाएं

Concept sliders "opposites" के contrastive dataset से सीखते हैं। छोटे before/after pairs बनाएं (4–6 pairs शुरुआत के लिए पर्याप्त हैं, अधिक हों तो बेहतर):

- **Positive bucket**: “कॉनसेप्ट का अधिक” (जैसे, brighter eyes, stronger smile, extra sand)। `"slider_strength": 0.5` सेट करें (कोई भी positive मान)।
- **Negative bucket**: “कॉनसेप्ट का कम” (जैसे, dimmer eyes, neutral expression)। `"slider_strength": -0.5` सेट करें (कोई भी negative मान)।
- **Neutral bucket (वैकल्पिक)**: नियमित उदाहरण। `slider_strength` छोड़ दें या इसे `0` सेट करें।

Positive/negative फ़ोल्डर्स में फ़ाइल‑नामों का मेल रखना ज़रूरी नहीं है — बस सुनिश्चित करें कि दोनों buckets में samples की संख्या बराबर हो।

## चरण 4 — dataloader को अपने buckets की ओर इंगित करें

- Z-Image quickstart से वही dataloader JSON pattern उपयोग करें।
- हर backend entry में `slider_strength` जोड़ें। SimpleTuner:
  - बैचेस को **positive → negative → neutral** क्रम में rotate करेगा ताकि दोनों दिशाएँ fresh रहें।
  - प्रत्येक backend की probability का सम्मान करेगा, इसलिए आपके weighting knobs काम करते रहेंगे।

अतिरिक्त flags की ज़रूरत नहीं—सिर्फ़ `slider_strength` फ़ील्ड्स।

## चरण 5 — ट्रेन करें

सामान्य कमांड (`simpletuner train ...`) उपयोग करें या WebUI से शुरू करें। Flag on होते ही slider targeting स्वतः सक्रिय हो जाती है।

## चरण 6 — Validate (वैकल्पिक slider tweaks)

Prompt libraries per‑prompt adapter scales के साथ A/B checks कर सकती हैं:

```json
{
  "plain": "regular prompt",
  "slider_plus": { "prompt": "same prompt", "adapter_strength": 1.2 },
  "slider_minus": { "prompt": "same prompt", "adapter_strength": 0.5 }
}
```

यदि इसे छोड़ दिया जाए, तो validation आपका global strength उपयोग करेगा।

---

## संदर्भ और विवरण

<details>
<summary>ये targets क्यों? (technical)</summary>

SimpleTuner slider LoRAs को self‑attention, conv/proj, और time‑embedding layers तक route करता है ताकि Concept Sliders का “leave text alone” नियम अपनाया जा सके। ControlNet runs भी slider targeting का सम्मान करते हैं। Assistant adapters frozen रहते हैं।
</details>

<details>
<summary>डिफ़ॉल्ट slider target सूची (आर्किटेक्चर के अनुसार)</summary>

- General (SD1.x, SDXL, SD3, Lumina2, Wan, HiDream, LTXVideo, Qwen-Image, Cosmos, Stable Cascade, आदि):

  ```json
  [
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn1.to_qkv", "to_qkv",
    "proj_in", "proj_out",
    "conv_in", "conv_out",
    "time_embedding.linear_1", "time_embedding.linear_2"
  ]
  ```

- Flux / Flux2 / Chroma / AuraFlow (केवल visual stream):

  ```json
  ["to_q", "to_k", "to_v", "to_out.0", "to_qkv"]
  ```

  Flux2 variants में `attn.to_q`, `attn.to_k`, `attn.to_v`, `attn.to_out.0`, `attn.to_qkv_mlp_proj` शामिल हैं।

- Kandinsky 5 (image/video):

  ```json
  ["attn1.to_query", "attn1.to_key", "attn1.to_value", "conv_in", "conv_out", "time_embedding.linear_1", "time_embedding.linear_2"]
  ```

</details>

<details>
<summary>LyCORIS presets (LoKr उदाहरण)</summary>

अधिकांश मॉडल:

```json
{
  "algo": "lokr",
  "multiplier": 1.0,
  "linear_dim": 4,
  "linear_alpha": 1,
  "apply_preset": {
    "target_module": [
      "attn1.to_q",
      "attn1.to_k",
      "attn1.to_v",
      "attn1.to_out.0",
      "conv_in",
      "conv_out",
      "time_embedding.linear_1",
      "time_embedding.linear_2"
    ]
  }
}
```

Flux/Chroma/AuraFlow: targets को `"attn.to_q","attn.to_k","attn.to_v","attn.to_out.0","attn.to_qkv_mlp_proj"` में बदलें (`attn.` को तब हटाएँ जब checkpoints में वह मौजूद न हो)। टेक्स्ट/context को untouched रखने के लिए `add_*` projections से बचें।

Kandinsky 5: `attn1.to_query/key/value` के साथ `conv_*` और `time_embedding.linear_*` उपयोग करें।
</details>

<details>
<summary>Sampling कैसे काम करता है (technical)</summary>

`slider_strength` टैग वाले backends को sign के आधार पर समूहित किया जाता है और fixed cycle में sample किया जाता है: positive → negative → neutral। हर समूह के भीतर सामान्य backend probabilities लागू होती हैं। Exhausted backends हट जाते हैं और cycle बाकी बचे backends के साथ जारी रहता है।
</details>
