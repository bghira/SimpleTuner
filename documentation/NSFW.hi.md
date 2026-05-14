# NSFW Classifier Checks

SimpleTuner में optional classifier checks हैं जो VAE cache preprocessing के दौरान samples reject कर सकते हैं। यह feature एक local filtering tool है। यह legal advice, compliance system, या यह guarantee नहीं है कि कोई dataset किसी खास use के लिए lawful या acceptable है।

## आपकी जिम्मेदारी

आप यह तय करने के लिए जिम्मेदार हैं कि आपका dataset, training run, model output, और publication या distribution plan आप पर लागू rules का पालन करता है या नहीं।

इन rules में local, regional, national, और platform-specific requirements शामिल हो सकते हैं। वे consent, age, likeness rights, privacy, publicity rights, obscenity rules, employment या institutional policy, और result किसी real person को depict या impersonate करता है या नहीं, इन बातों पर निर्भर कर सकते हैं। Laws समय के साथ बदलते हैं और jurisdiction के अनुसार अलग होते हैं।

SimpleTuner यह निर्णय आपके लिए नहीं करेगा। यह आपको नहीं बताएगा कि आपकी policy incomplete है, आपके thresholds law से match करते हैं या नहीं, या model output publish करने के लिए safe है या नहीं। यदि आप अनिश्चित हैं, तो अपने jurisdiction और use case के लिए qualified legal advice लें।

## Privacy

NSFW classifier checks SimpleTuner चलाने वाली machine पर locally चलते हैं।

- यह feature dataset samples को किसी third-party moderation API पर नहीं भेजता।
- Classifier results third parties को forward नहीं किए जाते।
- `--report_to` training telemetry option NSFW classifier results receive नहीं करता।
- Reports instance पर local VAE cache directory में `nsfw_classifier_report_rank*.json` के रूप में stored होती हैं।

एक network-facing behavior यह हो सकता है कि classifier weights local model cache में न होने पर Hugging Face से normal model loading हो। Model local उपलब्ध होने के बाद classification itself on-instance चलता है।

## Opt-in behavior

यह feature default रूप से disabled है। इसे enable करें:

```bash
--enable_nsfw_check=true
```

Checks केवल उन uncached samples पर लागू होते हैं जिन्हें VAE cache process करने वाला है। Existing VAE caches trusted माने जाते हैं, और `skip_file_discovery=vae` enforcement bypass करता है क्योंकि SimpleTuner मानता है कि आपने cache अपनी policy के तहत पहले ही तैयार कर लिया है।

Evaluation datasets scan नहीं होते।

## Supported classifiers

SimpleTuner `AutoImageProcessor` और `AutoModelForImageClassification` के माध्यम से standard Hugging Face Transformers image-classification models support करता है।

Default models हैं:

```text
Falconsai/nsfw_image_detection:threshold=0.5,AdamCodd/vit-base-nsfw-detector:threshold=0.5
```

आप अपनी CSV list दे सकते हैं:

```bash
--nsfw_check_models="org/model-a:threshold=0.5,org/model-b:threshold=0.7"
```

SimpleTuner इन classifiers के लिए `trust_remote_code` enable नहीं करता और इस feature के लिए `timm` dependency add नहीं करता। Custom code या non-Transformers backends मांगने वाले models इस scanner में supported नहीं हैं।

## Non-NSFW use

Option names में NSFW होने के बावजूद, यह mechanism sexual-content filtering तक सीमित नहीं है। यदि classifier recognizable labels और scores emit करता है जो SimpleTuner के expected unsafe/safe label hints से साफ़ map होते हैं, तो इसे अन्य binary या label-score checks के लिए भी use किया जा सकता है।

Examples में prohibited visual category, brand-sensitive content, या दूसरी locally defined dataset policy वाले samples reject करना शामिल हो सकता है। फिर भी classifier labels, thresholds, और vote settings आपकी policy से match करते हैं या नहीं, यह validate करना आपकी जिम्मेदारी है।

## Legal context

Adult sexual content हर जगह automatically illegal नहीं होता, और NSFW model training SimpleTuner में automatically disallowed नहीं है। इसका मतलब यह नहीं कि कोई specific dataset, output, या deployment lawful है।

High-risk areas में शामिल हैं:

- Minors या apparent minors से संबंधित content। United States में FBI Internet Crime Complaint Center कहता है कि generative AI और similar tools से बनाया गया child sexual abuse material illegal है।
- Non-consensual intimate imagery, sexual exploitation, harassment, blackmail, या बिना permission distribution।
- ऐसे outputs जो किसी real person को impersonate, recreate, या misleadingly depict करते हैं, खासकर sexual, fraudulent, या reputationally harmful purposes के लिए। FTC ने AI-enabled impersonation और deepfake fraud risks highlight किए हैं।
- Deepfake disclosure और transparency rules। उदाहरण के लिए, EU AI Act Article 50 कुछ AI-generated या manipulated image, audio, या video content जो deepfake हो, उसके लिए transparency obligations शामिल करता है।
- Contractual या platform rules, जिनमें dataset licenses, hosting provider policies, workplace rules, payment processor rules, और model distribution terms शामिल हैं।

Classifier को अपनी review process में एक control मानें, पूरी review process नहीं।

## Related options

- `--enable_nsfw_check`
- `--nsfw_check_models`
- `--nsfw_check_min_votes`
- `--nsfw_check_backend_types`
- `--nsfw_check_sample_types`
- `--delete_nsfw_images`
- `--nsfw_check_video_frame_count`
- `--nsfw_check_video_frame_selection`
- `--nsfw_check_video_min_flagged_frames`

VAE cache integration details के लिए [DATALOADER.hi.md#nsfw-classifier-checks-during-vae-caching](DATALOADER.hi.md#nsfw-classifier-checks-during-vae-caching) देखें।

## References

- [FBI IC3: Child Sexual Abuse Material Created by Generative AI and Similar Online Tools is Illegal](https://www.ic3.gov/PSA/2024/PSA240329)
- [FTC: Proposed protections to combat AI impersonation of individuals](https://www.ftc.gov/news-events/news/press-releases/2024/02/ftc-proposes-new-protections-combat-ai-impersonation-individuals)
- [EU AI Act Article 50: transparency obligations](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50)
