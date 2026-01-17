# CREPA (वीडियो रेगुलराइज़ेशन)

Cross-frame Representation Alignment (CREPA) वीडियो मॉडल्स के लिए एक हल्का regularizer है। यह हर फ्रेम के hidden states को frozen vision encoder के फीचर्स के साथ—मौजूदा फ्रेम **और उसके पड़ोसियों**—की ओर nudges करता है, जिससे temporal consistency बेहतर होती है और आपका मुख्य loss बदले बिना सुधार मिलता है।

## कब उपयोग करें

- आप जटिल motion, scene changes, या occlusions वाले वीडियो पर ट्रेन कर रहे हैं।
- आप वीडियो DiT (LoRA या full) fine-tune कर रहे हैं और फ्रेम्स के बीच flicker/identity drift देख रहे हैं।
- समर्थित मॉडल फैमिलीज़: `kandinsky5_video`, `ltxvideo`, `sanavideo`, और `wan` (अन्य फैमिलीज़ CREPA hooks एक्सपोज़ नहीं करतीं)।
- आपके पास अतिरिक्त VRAM है (CREPA सेटिंग्स के अनुसार ~1–2GB जोड़ता है) DINO encoder और VAE के लिए, जिन्हें ट्रेनिंग के दौरान latents को pixels में decode करने हेतु मेमोरी में रहना पड़ता है।

## क्विक सेटअप (WebUI)

1. **Training → Loss functions** खोलें।
2. **CREPA** सक्षम करें।
3. **CREPA Block Index** को encoder-side लेयर पर सेट करें। शुरुआत के लिए:
   - Kandinsky5 Video: `8`
   - LTXVideo / Wan: `8`
   - SanaVideo: `10`
4. **Weight** को `0.5` पर रखें।
5. **Adjacent Distance** को `1` और **Temporal Decay** को `1.0` पर रखें ताकि सेटअप मूल CREPA पेपर के करीब रहे।
6. vision encoder के लिए डिफ़ॉल्ट्स रखें (`dinov2_vitg14`, resolution `518`)। केवल तब बदलें जब आपको छोटा encoder चाहिए (जैसे VRAM बचाने हेतु `dinov2_vits14` + image size `224`)।
7. सामान्य रूप से ट्रेन करें। CREPA एक auxiliary loss जोड़ता है और `crepa_loss` / `crepa_similarity` लॉग करता है।

## क्विक सेटअप (config JSON / CLI)

अपने `config.json` या CLI args में यह जोड़ें:

```json
{
  "crepa_enabled": true,
  "crepa_block_index": 8,
  "crepa_lambda": 0.5,
  "crepa_adjacent_distance": 1,
  "crepa_adjacent_tau": 1.0,
  "crepa_encoder": "dinov2_vitg14",
  "crepa_encoder_image_size": 518
}
```

## ट्यूनिंग नॉब्स

- `crepa_spatial_align`: patch-level संरचना बनाए रखता है (डिफ़ॉल्ट)। मेमोरी कम हो तो `false` सेट करें।
- `crepa_normalize_by_frames`: clip लंबाइयों के साथ loss scale को स्थिर रखता है (डिफ़ॉल्ट)। लंबे clips का योगदान बढ़ाना हो तो बंद करें।
- `crepa_drop_vae_encoder`: यदि आप केवल latents **decode** करते हैं तो मेमोरी बचाता है (यदि आपको pixels encode करने हैं तो unsafe)।
- `crepa_adjacent_distance=0`: per-frame REPA* जैसा व्यवहार (कोई neighbour सहायता नहीं); distance decay के लिए `crepa_adjacent_tau` के साथ जोड़ें।
- `crepa_cumulative_neighbors=true` (केवल config): सिर्फ nearest neighbours की जगह सभी offsets `1..d` उपयोग करें।
- `crepa_use_backbone_features=true`: external encoder छोड़कर deeper transformer block के साथ align करें; teacher चुनने के लिए `crepa_teacher_block_index` सेट करें।
- Encoder size: VRAM कम हो तो `dinov2_vits14` + `224` पर जाएँ; सर्वोत्तम गुणवत्ता के लिए `dinov2_vitg14` + `518` रखें।

## Coefficient scheduling

CREPA ट्रेनिंग के दौरान coefficient (`crepa_lambda`) को warmup, decay, और similarity threshold आधारित automatic cutoff के साथ schedule करने का समर्थन करता है। यह विशेष रूप से text2video ट्रेनिंग के लिए उपयोगी है जहाँ CREPA बहुत अधिक या बहुत लंबे समय तक लागू करने पर horizontal/vertical stripes या washed-out feel पैदा कर सकता है।

### बेसिक scheduling

```json
{
  "crepa_enabled": true,
  "crepa_lambda": 0.5,
  "crepa_scheduler": "cosine",
  "crepa_warmup_steps": 100,
  "crepa_decay_steps": 5000,
  "crepa_lambda_end": 0.0
}
```

यह कॉन्फ़िगरेशन:
1. पहले 100 steps में CREPA weight को 0 से 0.5 तक ramp करता है
2. 5000 steps में cosine schedule से 0.5 से 0.0 तक decay करता है
3. Step 5100 के बाद, CREPA प्रभावी रूप से disable हो जाता है

### Scheduler types

- `constant`: कोई decay नहीं, weight `crepa_lambda` पर रहता है (डिफ़ॉल्ट)
- `linear`: `crepa_lambda` से `crepa_lambda_end` तक linear interpolation
- `cosine`: Smooth cosine annealing (ज़्यादातर मामलों के लिए अनुशंसित)
- `polynomial`: `crepa_power` के ज़रिए configurable power के साथ polynomial decay

### Step-based cutoff

किसी विशेष step के बाद hard cutoff के लिए:

```json
{
  "crepa_cutoff_step": 3000
}
```

Step 3000 के बाद CREPA पूरी तरह disable हो जाता है।

### Similarity-based cutoff

यह सबसे flexible approach है—जब similarity metric plateau हो जाता है, तो CREPA स्वचालित रूप से disable हो जाता है, जो दर्शाता है कि मॉडल ने पर्याप्त temporal alignment सीख लिया है:

```json
{
  "crepa_similarity_threshold": 0.9,
  "crepa_similarity_ema_decay": 0.99,
  "crepa_threshold_mode": "permanent"
}
```

- `crepa_similarity_threshold`: जब similarity का exponential moving average इस value तक पहुँचता है, CREPA cut off हो जाता है
- `crepa_similarity_ema_decay`: Smoothing factor (0.99 ≈ 100-step window)
- `crepa_threshold_mode`: `permanent` (बंद रहता है) या `recoverable` (similarity गिरे तो फिर से enable हो सकता है)

### अनुशंसित कॉन्फ़िगरेशन

**image2video (i2v) के लिए**:
```json
{
  "crepa_scheduler": "constant",
  "crepa_lambda": 0.5
}
```
मानक CREPA i2v के लिए अच्छा काम करता है क्योंकि reference frame consistency को anchor करता है।

**text2video (t2v) के लिए**:
```json
{
  "crepa_scheduler": "cosine",
  "crepa_lambda": 0.5,
  "crepa_warmup_steps": 100,
  "crepa_decay_steps": 0,
  "crepa_lambda_end": 0.1,
  "crepa_similarity_threshold": 0.85,
  "crepa_threshold_mode": "permanent"
}
```
ट्रेनिंग के दौरान CREPA को decay करता है और artifacts रोकने के लिए similarity saturate होने पर cut off करता है।

**solid backgrounds (t2v) के लिए**:
```json
{
  "crepa_cutoff_step": 2000
}
```
Early cutoff uniform backgrounds पर stripe artifacts को रोकता है।

<details>
<summary>कैसे काम करता है (प्रैक्टिशनर)</summary>

- चुने हुए DiT block से hidden states कैप्चर करता है, उन्हें LayerNorm+Linear head के जरिए प्रोजेक्ट करता है, और frozen vision features के साथ align करता है।
- डिफ़ॉल्ट रूप से DINOv2 से pixel frames encode करता है; backbone mode में एक deeper transformer block reuse होता है।
- हर फ्रेम को उसके neighbours के साथ exponential distance decay (`crepa_adjacent_tau`) के साथ align करता है; cumulative mode में वैकल्पिक रूप से `d` तक सभी offsets sum होते हैं।
- Spatial/temporal alignment tokens को resample करता है ताकि DiT patches और encoder patches cosine similarity से पहले align हों; loss patches और frames पर औसत लिया जाता है।

</details>

<details>
<summary>तकनीकी (SimpleTuner internals)</summary>

- इम्प्लीमेंटेशन: `simpletuner/helpers/training/crepa.py`; `ModelFoundation._init_crepa_regularizer` से रजिस्टर होता है और trainable मॉडल पर अटैच होता है (projector optimizer कवरेज के लिए मॉडल पर रहता है)।
- Hidden-state capture: वीडियो transformers `crepa_hidden_states` (और वैकल्पिक रूप से `crepa_frame_features`) stash करते हैं जब `crepa_enabled` true होता है; backbone mode shared hidden-state buffer से `layer_{idx}` भी खींच सकता है।
- Loss path: `crepa_use_backbone_features` ऑन न हो तो VAE से latents को pixels में decode करता है; projected hidden states और encoder features को normalize करता है, distance-weighted cosine similarity लागू करता है, `crepa_loss` / `crepa_similarity` लॉग करता है, और scaled loss जोड़ता है।
- Interaction: LayerSync से पहले चलता है ताकि दोनों hidden-state buffer reuse कर सकें; फिर buffer साफ करता है। वैध block index और transformer config से inferred hidden size की आवश्यकता होती है।

</details>

## सामान्य pitfalls

- unsupported families पर CREPA सक्षम करने से hidden states गायब होंगे; `kandinsky5_video`, `ltxvideo`, `sanavideo`, या `wan` तक सीमित रहें।
- **Block index बहुत ऊँचा** → “hidden states not returned”। index कम करें; यह transformer blocks पर zero-based है।
- **VRAM spikes** → `crepa_spatial_align=false` आज़माएँ, छोटा encoder (`dinov2_vits14` + `224`) या कम block index चुनें।
- **Backbone mode errors** → `crepa_block_index` (student) और `crepa_teacher_block_index` (teacher) दोनों को मौजूदा लेयर्स पर सेट करें।
- **Out of memory** → यदि RamTorch मदद नहीं कर रहा, तो एक बड़ा GPU ही समाधान हो सकता है—यदि H200 या B200 भी काम न करें, तो कृपया issue रिपोर्ट फाइल करें।
