# SimpleTuner WebUI ट्यूटोरियल

## परिचय

यह ट्यूटोरियल आपको SimpleTuner Web इंटरफ़ेस शुरू करने में मदद करेगा।

## आवश्यकताओं की इंस्टॉलेशन

Ubuntu सिस्टम पर आवश्यक पैकेज इंस्टॉल करने से शुरू करें:

```bash
apt -y install python3.13-venv python3.13-dev
apt -y install libopenmpi-dev openmpi-bin cuda-toolkit-12-8 libaio-dev # यदि आप DeepSpeed उपयोग कर रहे हैं
apt -y install ffmpeg # यदि वीडियो मॉडल ट्रेन कर रहे हैं
```

## Workspace डायरेक्टरी बनाना

एक workspace में आपकी configurations, output models, validation images, और संभवतः datasets शामिल होते हैं।

Vast या समान providers पर आप `/workspace/simpletuner` डायरेक्टरी उपयोग कर सकते हैं:

```bash
mkdir -p /workspace/simpletuner
export SIMPLETUNER_WORKSPACE=/workspace/simpletuner
cd $SIMPLETUNER_WORKSPACE
```

यदि आप इसे अपने home डायरेक्टरी में बनाना चाहते हैं:
```bash
mkdir ~/simpletuner-workspace
export SIMPLETUNER_WORKSPACE=~/simpletuner-workspace
cd $SIMPLETUNER_WORKSPACE
```

## Workspace में SimpleTuner इंस्टॉल करना

Dependencies इंस्टॉल करने के लिए virtual environment बनाएँ:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
```

### CUDA-विशिष्ट dependencies

NVIDIA उपयोगकर्ताओं को सही dependencies के लिए CUDA extras इस्तेमाल करनी होंगी:

```bash
pip install -e 'simpletuner[cuda]'
# CUDA 13 / Blackwell users (NVIDIA B-series GPUs):
# pip install -e 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
# या, यदि आपने git से clone किया है:
# pip install -e '.[cuda]'
```

Apple और ROCm हार्डवेयर के उपयोगकर्ताओं के लिए अन्य extras हैं, [installation निर्देश](../INSTALL.md) देखें।

## सर्वर शुरू करना

SSL के साथ पोर्ट 8080 पर सर्वर शुरू करने के लिए:

```bash
# DeepSpeed के लिए CUDA_HOME को सही लोकेशन पर पॉइंट करना होगा
export CUDA_HOME=/usr/local/cuda-12.8
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH

simpletuner server --ssl --port 8080
```

अब अपने ब्राउज़र में https://localhost:8080 खोलें।

आपको SSH के जरिए पोर्ट फॉरवर्ड करना पड़ सकता है, उदाहरण के लिए:

```bash
ssh -L 8080:localhost:8080 user@remote-server
```

> **टिप:** यदि आपके पास पहले से कोई configuration environment (जैसे पहले के CLI उपयोग से) है, तो आप `--env` के साथ सर्वर शुरू कर सकते हैं ताकि सर्वर तैयार होते ही ट्रेनिंग अपने आप शुरू हो जाए:
>
> ```bash
> simpletuner server --ssl --port 8080 --env my-training-config
> ```
>
> यह सर्वर शुरू करने और फिर WebUI में “Start Training” क्लिक करने के बराबर है, लेकिन unattended startup की अनुमति देता है।

## पहली बार सेटअप: admin अकाउंट बनाना

पहली बार लॉन्च पर, SimpleTuner आपसे एक administrator अकाउंट बनाने को कहता है। WebUI खोलते ही आपको setup स्क्रीन दिखेगी जो पहला admin user बनाने के लिए कहती है।

अपना email, username, और एक सुरक्षित पासवर्ड दर्ज करें। इस अकाउंट में पूर्ण प्रशासनिक अधिकार होंगे।

### Users प्रबंधन

सेटअप के बाद, आप **Manage Users** पेज से users मैनेज कर सकते हैं (admin के रूप में लॉगिन होने पर sidebar से उपलब्ध):

- **Users tab**: user अकाउंट बनाना, संपादित करना, और हटाना। permission levels असाइन करना (viewer, researcher, lead, admin)।
- **Levels tab**: finer-grained access control के साथ कस्टम permission levels परिभाषित करना।
- **Auth Providers tab**: single sign-on के लिए external authentication (OIDC, LDAP) कॉन्फ़िगर करना।
- **Registration tab**: नए users के self-register करने को नियंत्रित करना (डिफ़ॉल्ट से disabled)।

### Automation के लिए API keys

Users अपनी प्रोफ़ाइल या admin पैनल से scripted access के लिए API keys बना सकते हैं। API keys `st_` prefix का उपयोग करते हैं और `X-API-Key` header के साथ उपयोग किए जा सकते हैं:

```bash
curl -s http://localhost:8080/api/training/status \
  -H 'X-API-Key: st_your_key_here'
```

> **नोट:** निजी/आंतरिक deployments के लिए public registration बंद रखें और users को admin पैनल से मैन्युअली बनाएं।

## WebUI का उपयोग

### Onboarding स्टेप्स

पेज लोड होने के बाद, आपके environment को सेटअप करने के लिए onboarding प्रश्न पूछे जाएंगे।

#### Configuration directory

विशेष configuration मान `configs_dir` एक फ़ोल्डर की ओर इंगित करता है जिसमें आपकी सभी SimpleTuner configurations होती हैं, जिन्हें subdirectories में व्यवस्थित करने की सलाह है - **Web UI यह आपके लिए कर देगा**:

```
configs/
├── an-environment-named-something
│   ├── config.json
│   ├── lycoris_config.json
│   └── multidatabackend-DataBackend-Name.json
```

<img width="429" height="640" alt="image" src="https://github.com/user-attachments/assets/4be22081-f13d-4aed-a87c-2313ddefc8a4" />

##### कमांड-लाइन उपयोग से माइग्रेशन

यदि आप पहले बिना WebUI के SimpleTuner उपयोग कर रहे थे, तो आप अपने मौजूदा config/ फ़ोल्डर को पॉइंट कर सकते हैं और आपके सभी environments अपने आप discover हो जाएंगे।

नए उपयोगकर्ताओं के लिए, configs और datasets का डिफ़ॉल्ट लोकेशन `~/.simpletuner/` होगा और सलाह है कि datasets को अधिक स्पेस वाली जगह पर ले जाएँ:

<img width="429" height="640" alt="image" src="https://github.com/user-attachments/assets/c5b3ab53-654e-4a9b-8e2d-7951f11619ef" />


#### (Multi-)GPU चयन और कॉन्फ़िगरेशन

डिफ़ॉल्ट paths कॉन्फ़िगर करने के बाद, आप multi-GPU कॉन्फ़िगरेशन वाले स्टेप पर पहुँचेंगे (NVIDIA सिस्टम पर चित्रित)

<img width="429" height="640" alt="image" src="https://github.com/user-attachments/assets/61d5a7bc-0a02-4a0a-8df0-207cce4b7bc1" />

यदि आपके पास कई GPUs हैं और आप सिर्फ दूसरे GPU का उपयोग करना चाहते हैं, तो आप यहाँ यह सेट कर सकते हैं।

> **Multi-GPU उपयोगकर्ताओं के लिए नोट:** कई GPUs के साथ ट्रेनिंग करने पर आपके डेटासेट आकार की आवश्यकताएँ अनुपातिक रूप से बढ़ती हैं। प्रभावी बैच साइज़ `train_batch_size × num_gpus × gradient_accumulation_steps` होता है। यदि आपका डेटासेट इससे छोटा है, तो आपको या तो अपने डेटासेट कॉन्फ़िगरेशन में `repeats` बढ़ाना होगा या Advanced सेटिंग्स में `--allow_dataset_oversubscription` विकल्प सक्षम करना होगा। अधिक विवरण के लिए नीचे [बैच साइज़ सेक्शन](#multi-gpu-batch-size-considerations) देखें।

#### अपना पहला training environment बनाना

यदि आपके `configs_dir` में कोई pre-existing configurations नहीं मिलीं, तो आपको **अपना पहला training environment** बनाने के लिए कहा जाएगा:

<img width="500" height="640" alt="image" src="https://github.com/user-attachments/assets/2110287a-16fd-4f87-893b-86d2a555a10f" />

**Bootstrap From Example** का उपयोग करके एक example config चुनकर शुरू करें, या यदि आप setup wizard उपयोग करना चाहते हैं तो एक वर्णनात्मक नाम दर्ज करके एक random environment बना लें।

### Training environments के बीच स्विच करना

यदि आपके पास pre-existing configuration environments थे, तो वे इस drop-down मेनू में दिखेंगे।

अन्यथा, onboarding के दौरान बनाया गया विकल्प पहले से चयनित और सक्रिय होगा।

<img width="448" height="225" alt="image" src="https://github.com/user-attachments/assets/66fef6a9-2040-47fd-b22d-918470677992" />

**Manage Configs** का उपयोग करके `Environment` टैब पर जाएँ जहाँ आपके environments, dataloader और अन्य configurations की सूची मिलती है।

### Configuration wizard

मैंने एक व्यापक setup wizard बनाने पर काफी मेहनत की है, जो आपको सबसे महत्वपूर्ण सेटिंग्स को आसानी से कॉन्फ़िगर करने में मदद करेगा।

<img width="470" height="358" alt="image" src="https://github.com/user-attachments/assets/e4bf1a4e-716c-4101-b753-e9e24bb42d8a" />

ऊपरी बाएँ navigation मेनू में Wizard बटन आपको selection dialogue पर ले जाएगा:

<img width="448" height="440" alt="image" src="https://github.com/user-attachments/assets/68324fa8-3ca9-45b1-b947-1e7738fd1d8c" />

फिर सभी built-in model variants प्रस्तुत किए जाते हैं। हर variant आवश्यक सेटिंग्स जैसे Attention Masking या extended token limits पहले से सक्षम करता है।

#### LoRA मॉडल विकल्प

यदि आप LoRA ट्रेन करना चाहते हैं, तो यहाँ आप model quantisation विकल्प सेट कर पाएँगे।

सामान्य तौर पर, यदि आप Stable Diffusion प्रकार का मॉडल ट्रेन नहीं कर रहे हैं, तो int8-quanto की सिफारिश की जाती है क्योंकि यह गुणवत्ता को नुकसान नहीं पहुँचाता और बड़े batch sizes की अनुमति देता है।

Cosmos2, Sana, और PixArt जैसे छोटे मॉडल्स को quantisation पसंद नहीं होता।

<img width="508" height="600" alt="image" src="https://github.com/user-attachments/assets/c2e721f2-b4da-4cd0-84fd-7ac81993e87c" />

#### Full-rank training

Full-rank training की सलाह नहीं दी जाती, क्योंकि सामान्यतः यह LoRA/LyCORIS की तुलना में समान डेटासेट के लिए बहुत अधिक समय और संसाधन लेता है।

हालाँकि, यदि आप full checkpoint ट्रेन करना चाहते हैं, तो आप यहाँ DeepSpeed ZeRO stages कॉन्फ़िगर कर सकते हैं, जो Auraflow, Flux, और बड़े मॉडल्स के लिए आवश्यक होंगे।

FSDP2 समर्थित है, लेकिन इस wizard में कॉन्फ़िगर नहीं किया जा सकता। यदि आप इसे उपयोग करना चाहते हैं तो DeepSpeed को disabled छोड़ दें और बाद में FSDP2 मैन्युअली कॉन्फ़िगर करें।

<img width="508" height="600" alt="image" src="https://github.com/user-attachments/assets/88438f1c-b0a2-4249-afd0-7878aa1abada" />


#### आप कितने समय तक ट्रेन करना चाहते हैं?

आपको तय करना होगा कि आप ट्रेनिंग समय को epochs में मापना चाहते हैं या steps में। अंततः दोनों लगभग बराबर होते हैं, लेकिन कुछ लोग किसी एक को पसंद करते हैं।

<img width="508" height="475" alt="image" src="https://github.com/user-attachments/assets/dcb54279-0ce7-4c66-a9ab-4dd26f87278c" />

#### Hugging Face Hub पर अपना मॉडल साझा करना

वैकल्पिक रूप से, आप अपने अंतिम *और* intermediate checkpoints को [Hugging Face Hub](https://hf.co) पर publish कर सकते हैं, लेकिन इसके लिए एक अकाउंट चाहिए होगा - आप wizard या Publishing टैब के जरिए hub में लॉगिन कर सकते हैं। किसी भी स्थिति में, आप बाद में इसे सक्षम/अक्षम कर सकते हैं।

यदि आप मॉडल publish करना चुनते हैं, तो अगर आप अपने मॉडल को सार्वजनिक नहीं करना चाहते तो `Private repo` चुनना न भूलें।

<img width="508" height="370" alt="image" src="https://github.com/user-attachments/assets/8d2d282b-e66f-48a8-a40e-4e4ecc2d280b" />

#### Checkpoint frequency

Training के दौरान आपका मॉडल समय-समय पर disk पर save होगा। अधिक checkpoints रखने के लिए अधिक disk space चाहिए।

Checkpoints बाद में training resume करने देते हैं, बिना सभी steps दोबारा चलाए। कुछ checkpoints रखने से आप अपने मॉडल के कई versions test कर सकते हैं और जो सबसे अच्छा काम करे उसे रख सकते हैं।

हर 10% पर checkpoint रखना recommended है, हालांकि यह इस बात पर निर्भर करता है कि आप कितने data पर train कर रहे हैं। छोटे dataset के साथ, overfitting से बचने की जांच के लिए आपको अक्सर checkpoint करना चाहिए।

बहुत बड़े datasets में लंबे checkpoint intervals बेहतर होते हैं ताकि उन्हें disk पर लिखने में समय बर्बाद न हो।

<img width="508" height="485" alt="image" src="https://github.com/user-attachments/assets/c7b1cd0b-a1b9-47ec-87f9-1ecac2e0841a" />

#### Model validations

यदि आप trainer को समय-समय पर इमेजेस जनरेट करवाना चाहते हैं, तो wizard के इस चरण पर एक single validation prompt कॉन्फ़िगर कर सकते हैं। wizard पूरा होने के बाद `Validations & Output` टैब में multiple prompt library कॉन्फ़िगर की जा सकती है।

क्या आप validation को अपने स्वयं के script या सेवा में आउटसोर्स करना चाहते हैं? wizard के बाद validation टैब में **Validation Method** को `external-script` पर सेट करें और `--validation_external_script` दें। आप placeholders जैसे `{local_checkpoint_path}`, `{global_step}`, `{tracker_run_name}`, `{tracker_project_name}`, `{model_family}`, `{huggingface_path}` और किसी भी `validation_*` config मान (जैसे `validation_num_inference_steps`, `validation_guidance`, `validation_noise_scheduler`) के साथ training context पास कर सकते हैं। training को ब्लॉक किए बिना fire-and-forget के लिए `--validation_external_background` सक्षम करें।

डिस्क पर checkpoint लिखते ही hook चाहिए? हर save के तुरंत बाद script चलाने के लिए `--post_checkpoint_script` उपयोग करें (uploads शुरू होने से पहले)। यह वही placeholders स्वीकार करता है, जहाँ `{remote_checkpoint_path}` खाली रहता है।

यदि आप SimpleTuner की built-in publishing providers (या Hugging Face Hub uploads) बनाए रखना चाहते हैं लेकिन remote URL के साथ अपनी automation ट्रिगर करना चाहते हैं, तो `--post_upload_script` उपयोग करें। यह हर upload पर placeholders `{remote_checkpoint_path}`, `{local_checkpoint_path}`, `{global_step}`, `{tracker_run_name}`, `{tracker_project_name}`, `{model_family}`, `{huggingface_path}` के साथ चलती है। SimpleTuner script output कैप्चर नहीं करता—अपने tracker अपडेट्स सीधे script से emit करें।

उदाहरण hook:

```bash
--post_upload_script='/opt/hooks/notify.sh {remote_checkpoint_path} {tracker_project_name} {tracker_run_name}'
```

जहाँ `notify.sh` URL को आपके tracker web API पर पोस्ट करता है। आप इसे Slack, कस्टम dashboards, या किसी अन्य इंटीग्रेशन के लिए अनुकूलित कर सकते हैं।

Working sample: `simpletuner/examples/external-validation/replicate_post_upload.py` `{remote_checkpoint_path}`, `{model_family}`, `{model_type}`, `{lora_type}`, और `{huggingface_path}` का उपयोग करके uploads के बाद Replicate inference ट्रिगर करना दिखाता है।

Another sample: `simpletuner/examples/external-validation/wavespeed_post_upload.py` WaveSpeed API कॉल करता है और परिणाम के लिए पोल करता है, वही placeholders उपयोग करते हुए।

Flux-focused sample: `simpletuner/examples/external-validation/fal_post_upload.py` fal.ai Flux LoRA endpoint कॉल करता है; इसे `FAL_KEY` चाहिए और केवल तब चलता है जब `model_family` में `flux` शामिल हो।

Local GPU sample: `simpletuner/examples/external-validation/use_second_gpu.py` किसी दूसरे GPU (डिफ़ॉल्ट `cuda:1`) पर Flux LoRA inference चलाता है और uploads के बिना भी उपयोग किया जा सकता है।

<img width="508" height="600" alt="image" src="https://github.com/user-attachments/assets/e699ba57-526b-4f60-9e8c-0ba410761c9f" />

#### Training statistics लॉग करना

यदि आप अपनी training statistics को किसी target API पर भेजना चाहते हैं, तो SimpleTuner कई विकल्प सपोर्ट करता है।

नोट: आपका कोई भी व्यक्तिगत डेटा, training logs, captions, या डेटा कभी भी SimpleTuner प्रोजेक्ट डेवलपर्स को नहीं भेजा जाता। आपके डेटा का नियंत्रण **आपके** हाथों में है।

<img width="508" height="438" alt="image" src="https://github.com/user-attachments/assets/0f8d15c5-456f-4637-af7e-c2f5f31cb968" />

#### Dataset कॉन्फ़िगरेशन

इस चरण पर, आप तय कर सकते हैं कि किसी मौजूदा dataset को रखना है या नया configuration बनाना है (बाकी को छुए बिना) Dataset Creation Wizard के माध्यम से, जो क्लिक करने पर दिखाई देगा।

<img width="508" height="290" alt="image" src="https://github.com/user-attachments/assets/b5a7f883-e180-4662-b84c-fff609c6b1df" />

##### Dataset Wizard

यदि आपने नया dataset बनाने का विकल्प चुना, तो आपको यह wizard दिखेगा, जो local या cloud dataset जोड़ने की प्रक्रिया बताएगा।

<img width="508" height="332" alt="image" src="https://github.com/user-attachments/assets/c523930b-563e-4b5d-b104-8e7ce4658b2c" />

<img width="508" height="508" alt="image" src="https://github.com/user-attachments/assets/c263f58e-fd85-437e-811a-967b94e309fd" />

लोकल dataset के लिए, आप **Browse directories** बटन का उपयोग करके dataset browser modal खोल सकते हैं।

<img width="396" height="576" alt="image" src="https://github.com/user-attachments/assets/14c51685-3559-4d16-be59-ed4b0959ca32" />

यदि आपने onboarding के दौरान datasets डायरेक्टरी सही सेट की है, तो आपको यहाँ अपनी चीज़ें दिखेंगी।

जिस डायरेक्टरी को जोड़ना है उस पर क्लिक करें, और फिर **Select Directory**।

<img width="454" height="356" alt="image" src="https://github.com/user-attachments/assets/1d482655-158a-4e3f-93b7-ef158396813c" />

इसके बाद आप resolution values और cropping कॉन्फ़िगर करने के लिए guided होंगे।

जब आप captions कॉन्फ़िगर करने वाले स्टेप पर पहुँचें, तो **ध्यान से तय करें** कि कौन सा विकल्प सही है।

यदि आप सिर्फ एक trigger word उपयोग करना चाहते हैं, तो वह **Instance Prompt** विकल्प है।

<img width="1146" height="896" alt="image" src="https://github.com/user-attachments/assets/6252bf9a-5e68-41c6-8a95-906993f2f546" />

##### वैकल्पिक: अपने ब्राउज़र से dataset अपलोड करें

यदि आपकी इमेजेस और captions अभी box पर नहीं हैं, तो dataset wizard में अब **Upload** बटन **Browse directories** के पास होता है। आप यह कर सकते हैं:

- अपनी configured datasets डायरेक्टरी के नीचे एक नया subfolder बनाएं, फिर individual files या ZIP अपलोड करें (images plus .txt/.jsonl/.csv metadata स्वीकार किए जाते हैं)।
- SimpleTuner को ZIP को उस फ़ोल्डर में extract करने दें (local backends के लिए आकार सीमित; बहुत बड़े archives अस्वीकार हो जाते हैं)।
- नई अपलोड की गई फ़ोल्डर को तुरंत browser में चुनकर UI छोड़े बिना wizard जारी रखें।

#### Learning rate, batch size और optimiser

Dataset wizard पूरा होने के बाद (या यदि आपने मौजूदा datasets रखने का विकल्प चुना), आपको optimiser/learning rate और batch size के लिए presets दिए जाएंगे।

ये केवल शुरुआती बिंदु हैं जो नए उपयोगकर्ताओं को अपनी शुरुआती ट्रेनिंग रन में बेहतर विकल्प चुनने में मदद करते हैं - अनुभवी उपयोगकर्ताओं के लिए **Manual configuration** चुनें।

**नोट**: यदि आप बाद में DeepSpeed उपयोग करने की योजना बना रहे हैं, तो यहाँ optimiser की पसंद ज्यादा मायने नहीं रखती।

##### Multi-GPU Batch Size विचार

कई GPUs के साथ ट्रेनिंग करते समय, आपका डेटासेट **effective batch size** को संभाल सके यह जरूरी है:

```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

यदि आपका डेटासेट इससे छोटा है, तो SimpleTuner एक error देगा और विशिष्ट मार्गदर्शन दिखाएगा। आप:
- batch size घटा सकते हैं
- अपने डेटासेट कॉन्फ़िगरेशन में `repeats` मान बढ़ा सकते हैं
- Advanced settings में **Allow Dataset Oversubscription** सक्षम कर सकते हैं ताकि repeats स्वतः समायोजित हो जाएँ

डेटासेट sizing पर अधिक जानकारी के लिए [DATALOADER.md](../DATALOADER.md#multi-gpu-training-and-dataset-sizing) देखें।

<img width="1118" height="1015" alt="image" src="https://github.com/user-attachments/assets/25d5650d-e77b-42fe-b749-06c0ec92b1e2" />

#### Memory optimisation presets

consumer हार्डवेयर पर सेटअप आसान बनाने के लिए, हर मॉडल के लिए light, balanced, या aggressive memory savings चुनने हेतु custom presets दिए गए हैं।

**Training** टैब के **Memory Optimisation** सेक्शन में आपको **Load Presets** बटन मिलेगा:

<img width="1048" height="940" alt="image" src="https://github.com/user-attachments/assets/804e84f6-7eb8-493e-95d2-a89d930bafa5" />

जिससे यह इंटरफ़ेस खुलता है:

<img width="1048" height="940" alt="image" src="https://github.com/user-attachments/assets/775aaee5-c3c0-4659-bbea-ebb39e3eb098" />


#### Review और save

यदि आप चुनी गई सभी values से खुश हैं, तो **Finish** पर क्लिक करें।

इसके बाद आपका नया environment सक्रिय रूप से selected होगा और training के लिए तैयार रहेगा!

अधिकांश मामलों में यही सभी सेटिंग्स होंगी जिनकी आपको जरूरत होती है। आप अतिरिक्त datasets जोड़ सकते हैं या अन्य सेटिंग्स बदल सकते हैं।

<img width="1096" height="1403" alt="image" src="https://github.com/user-attachments/assets/29fd0bb3-aab2-4455-9612-583ed949ce64" />

**Environment** पेज पर आपको नया कॉन्फ़िगर किया हुआ training job दिखेगा, और configuration को download या duplicate करने के बटन होंगे, यदि आप इसे template की तरह उपयोग करना चाहते हैं।

<img width="1881" height="874" alt="image" src="https://github.com/user-attachments/assets/33c0cafa-3fd8-40ee-b6fa-3704b6e698da" />

**नोट**: **Default** environment विशेष है, और इसे सामान्य training environment के रूप में उपयोग करने की सलाह नहीं है; इसकी settings को किसी भी environment में स्वचालित रूप से merge किया जा सकता है जो **Use environment defaults** विकल्प सक्षम करता है:

<img width="1521" height="991" alt="image" src="https://github.com/user-attachments/assets/9d18b0c1-608e-4ab2-be14-65b98907ec69" />
