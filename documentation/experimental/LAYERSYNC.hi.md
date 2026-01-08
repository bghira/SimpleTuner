# LayerSync (SimpleTuner)

LayerSync transformer मॉडलों के लिए एक “खुद को सिखाओ” प्रकार का nudging है: एक लेयर (student) एक मजबूत लेयर (teacher) के साथ align होना सीखती है। यह हल्का, self-contained है, और अतिरिक्त मॉडल डाउनलोड की जरूरत नहीं होती।

## कब उपयोग करें

- आप ऐसे transformer families ट्रेन कर रहे हैं जो hidden states एक्सपोज़ करती हैं (जैसे Flux/Flux Kontext/Flux.2, PixArt Sigma, SD3/SDXL, Sana, Wan, Qwen Image/Edit, Hunyuan Video, LTXVideo, Kandinsky5 Video, Chroma, ACE-Step, HiDream, Cosmos/LongCat/Z-Image/Auraflow)।
- आप बाहरी teacher checkpoint शिप किए बिना built-in regularizer चाहते हैं।
- आप mid-training drift या unstable heads देख रहे हैं और किसी mid-layer को deeper teacher की ओर खींचना चाहते हैं।
- आपके पास थोड़ा VRAM headroom है ताकि वर्तमान step के लिए student/teacher activations रख सकें।

## क्विक सेटअप (WebUI)

1. **Training → Loss functions** खोलें।
2. **LayerSync** सक्षम करें।
3. **Student Block** को mid-layer पर और **Teacher Block** को उससे deeper पर सेट करें। 24-layer DiT-style मॉडल्स (Flux, PixArt, SD3) पर `8` → `16` से शुरू करें; छोटे स्टैक्स में teacher को student से कुछ blocks आगे रखें।
4. **Weight** को `0.2` पर रखें (LayerSync सक्षम होने पर डिफ़ॉल्ट)।
5. सामान्य रूप से ट्रेन करें; logs में `layersync_loss` और `layersync_similarity` शामिल होंगे।

## क्विक सेटअप (config JSON / CLI)

```json
{
  "layersync_enabled": true,
  "layersync_student_block": 8,
  "layersync_teacher_block": 16,
  "layersync_lambda": 0.2
}
```

## ट्यूनिंग नॉब्स

- `layersync_student_block` / `layersync_teacher_block`: 1-based-friendly indexing; पहले `idx-1` की कोशिश होती है, फिर `idx`।
- `layersync_lambda`: cosine loss को स्केल करता है; सक्षम होने पर > 0 होना चाहिए (डिफ़ॉल्ट `0.2`)।
- Teacher डिफ़ॉल्ट रूप से student block ही होता है, जिससे loss self-similarity बन जाता है।
- VRAM: दोनों लेयर्स की activations aux loss चलने तक रखी जाती हैं; यदि मेमोरी बचानी हो तो LayerSync (या CREPA) बंद करें।
- CREPA/TwinFlow के साथ ठीक काम करता है; ये एक ही hidden-state buffer साझा करते हैं।

<details>
<summary>कैसे काम करता है (प्रैक्टिशनर)</summary>

- Flattened student और teacher tokens के बीच negative cosine similarity गणना करता है; अधिक weight student को teacher के फीचर्स की ओर धकेलता है।
- Teacher tokens हमेशा detached होते हैं ताकि gradients पीछे न बहें।
- इमेज और वीडियो transformers के लिए 3D `(B, S, D)` और 4D `(B, T, P, D)` hidden states हैंडल करता है।
- Upstream option mapping:
  - `--encoder-depth` → `--layersync_student_block`
  - `--gt-encoder-depth` → `--layersync_teacher_block`
  - `--reg-weight` → `--layersync_lambda`
- डिफ़ॉल्ट्स: डिफ़ॉल्ट रूप से बंद; सक्षम और unset होने पर `layersync_lambda=0.2`।

</details>

<details>
<summary>तकनीकी (SimpleTuner internals)</summary>

- इम्प्लीमेंटेशन: `simpletuner/helpers/training/layersync.py`; `ModelFoundation._apply_layersync_regularizer` से कॉल होता है।
- Hidden-state capture: तब ट्रिगर होता है जब LayerSync या CREPA इसे मांगते हैं; transformers states को `layer_{idx}` के रूप में `_store_hidden_state` से रखते हैं।
- Layer resolution: पहले 1-based फिर 0-based indices आज़माता है; यदि मांगी गई लेयर नहीं मिले तो error देता है।
- Loss path: student/teacher tokens normalize करता है, mean cosine similarity निकालता है, `layersync_loss` और `layersync_similarity` लॉग करता है, और scaled loss को मुख्य objective में जोड़ता है।
- Interaction: CREPA के बाद चलता है ताकि दोनों एक ही buffer reuse कर सकें; फिर buffer साफ करता है।

</details>

## सामान्य pitfalls

- student block गायब → startup error; `layersync_student_block` स्पष्ट रूप से सेट करें।
- Weight ≤ 0 → startup error; यदि अनिश्चित हों तो डिफ़ॉल्ट `0.2` रखें।
- मॉडल से ज्यादा गहरे blocks माँगना → “LayerSync could not find layer” त्रुटि; indices घटाएँ।
- ऐसे मॉडल्स पर सक्षम करना जो transformer hidden states एक्सपोज़ नहीं करते (Kolors, Lumina2, Stable Cascade C, Kandinsky5 Image, OmniGen) फेल होगा; transformer-backed families तक सीमित रहें।
- VRAM spikes: block indices कम करें या CREPA/LayerSync बंद करें ताकि hidden-state buffer मुक्त हो।

LayerSync का उपयोग तब करें जब आप बिना बाहरी teachers जोड़े intermediate representations को हल्के से steer करने के लिए एक सस्ता, built-in regularizer चाहते हों।
