# वितरित प्रशिक्षण (Multi-node)

इस दस्तावेज़ में SimpleTuner के साथ उपयोग के लिए 4‑way 8xH100 क्लस्टर कॉन्फ़िगर करने के नोट्स* शामिल हैं।

> *यह गाइड पूर्ण end‑to‑end इंस्टॉलेशन निर्देश नहीं देता। इसके बजाय, ये [INSTALL](INSTALL.md) दस्तावेज़ या किसी [quickstart गाइड](QUICKSTART.md) का पालन करते समय ध्यान रखने योग्य बातें हैं।

## Storage backend

Multi‑node प्रशिक्षण में डिफ़ॉल्ट रूप से `output_dir` के लिए nodes के बीच shared storage चाहिए।

VAE latents पहले से तैयार किए जाने पर हर rank को सभी datasets का `cache_dir_vae` पढ़ने की सुविधा चाहिए। डिफ़ॉल्ट पथ `output_dir` के अंदर होता है। कस्टम पथ के लिए NFS जैसे shared storage या `image_embeds` के ज़रिए कॉन्फ़िगर किए गए S3 जैसे shared cache backend का उपयोग करें। `vae_cache_ondemand` चालू होने पर हर rank प्रशिक्षण के दौरान अपने लिए ज़रूरी latents बनाता है।

पहले से तैयार text embeddings और conditioning image embeddings की caches चालू होने पर उन्हें भी हर rank के लिए पढ़ने योग्य storage चाहिए।


### Ubuntu NFS उदाहरण

शुरुआत के लिए एक बेसिक storage उदाहरण।

#### 'master' node पर जो checkpoints लिखेगा

**1. NFS Server Packages इंस्टॉल करें**

```bash
sudo apt update
sudo apt install nfs-kernel-server
```

**2. NFS Export कॉन्फ़िगर करें**

डायरेक्टरी शेयर करने के लिए NFS exports फ़ाइल एडिट करें:

```bash
sudo nano /etc/exports
```

फ़ाइल के अंत में यह लाइन जोड़ें (`slave_ip` को अपने slave मशीन के IP से बदलें):

```
~/simpletuner/output slave_ip(rw,sync,no_subtree_check)
```

*यदि कई slaves या पूरा subnet allow करना हो, तो उपयोग करें:*

```
~/simpletuner/output subnet_ip/24(rw,sync,no_subtree_check)
```

**3. Shared Directory Export करें**

```bash
sudo exportfs -a
```

**4. NFS Server रीस्टार्ट करें**

```bash
sudo systemctl restart nfs-kernel-server
```

**5. NFS Server स्थिति जाँचें**

```bash
sudo systemctl status nfs-kernel-server
```

---

#### slave nodes पर जो optimiser और अन्य states भेजते हैं

**1. NFS Client Packages इंस्टॉल करें**

```bash
sudo apt update
sudo apt install nfs-common
```

**2. Mount Point Directory बनाएँ**

सुनिश्चित करें कि डायरेक्टरी मौजूद है (यह आपके setup के अनुसार पहले से होनी चाहिए):

```bash
sudo mkdir -p ~/simpletuner/output
```

*Note:* यदि डायरेक्टरी में डेटा है, बैकअप लें, क्योंकि mounting से existing contents छिप जाएंगे।

**3. NFS Share माउंट करें**

master की shared डायरेक्टरी को slave की लोकल डायरेक्टरी पर माउंट करें (`master_ip` को master के IP से बदलें):

```bash
sudo mount master_ip:~/simpletuner/output ~/simpletuner/output
```

**4. Mount सत्यापित करें**

जाँचें कि mount सफल हुआ है:

```bash
mount | grep ~/simpletuner/output
```

**5. Write Access टेस्ट करें**

write permissions की पुष्टि के लिए test file बनाएँ:

```bash
touch ~/simpletuner/output/test_file_from_slave.txt
```

फिर master मशीन पर देखें कि फ़ाइल `~/simpletuner/output` में दिख रही है या नहीं।

**6. Mount को स्थायी बनाएं**

रीबूट पर mount बना रहे, इसके लिए `/etc/fstab` में जोड़ें:

```bash
sudo nano /etc/fstab
```

अंत में यह लाइन जोड़ें:

```
master_ip:~/simpletuner/output ~/simpletuner/output nfs defaults 0 0
```

---

#### **अतिरिक्त विचार:**

- **User Permissions:** सुनिश्चित करें कि `ubuntu` user का UID और GID दोनों मशीनों पर समान हो, ताकि file permissions consistent रहें। `id ubuntu` से UIDs जाँच सकते हैं।

- **Firewall Settings:** यदि firewall सक्षम है, तो NFS ट्रैफ़िक की अनुमति दें। master मशीन पर:

  ```bash
  sudo ufw allow from slave_ip to any port nfs
  ```

- **Clocks सिंक्रोनाइज़ करें:** वितरित सेटअप में clocks का synchronized होना अच्छा अभ्यास है। `ntp` या `systemd-timesyncd` उपयोग करें।

- **DeepSpeed Checkpoints टेस्ट करें:** एक छोटा DeepSpeed job चलाकर पुष्टि करें कि checkpoints master की डायरेक्टरी में सही लिखे जा रहे हैं।


## Dataloader कॉन्फ़िगरेशन

बहुत बड़े datasets को कुशलता से संभालना चुनौती हो सकता है। SimpleTuner datasets को हर node पर स्वतः shard करता है और preprocessing को क्लस्टर के हर उपलब्ध GPU में बाँटता है, जबकि throughput बनाए रखने के लिए asynchronous queues और threads का उपयोग करता है।

### Multi‑GPU प्रशिक्षण के लिए dataset sizing

जब कई GPUs या nodes पर प्रशिक्षण हो, तो dataset में **effective batch size** पूरा करने के लिए पर्याप्त samples होने चाहिए:

```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

**उदाहरण गणनाएँ:**

| कॉन्फ़िगरेशन | गणना | Effective Batch Size |
|--------------|-------------|---------------------|
| 1 node, 8 GPUs, batch_size=4, grad_accum=1 | 4 × 8 × 1 | 32 samples |
| 2 nodes, 16 GPUs, batch_size=8, grad_accum=2 | 8 × 16 × 2 | 256 samples |
| 4 nodes, 32 GPUs, batch_size=8, grad_accum=1 | 8 × 32 × 1 | 256 samples |

आपके dataset के हर aspect ratio bucket में कम से कम इतने samples होने चाहिए (`repeats` को ध्यान में रखते हुए), नहीं तो प्रशिक्षण एक विस्तृत error message के साथ फेल होगा।

#### छोटे datasets के लिए समाधान

यदि आपका dataset effective batch size से छोटा है:

1. **Batch size घटाएँ** - `train_batch_size` कम करें
2. **GPU count घटाएँ** - कम GPUs पर ट्रेन करें (लेकिन प्रशिक्षण धीमा होगा)
3. **Repeats बढ़ाएँ** - अपने [dataloader कॉन्फ़िगरेशन](DATALOADER.md#repeats) में `repeats` सेट करें
4. **Automatic oversubscription सक्षम करें** - repeats स्वतः समायोजित करने के लिए `--allow_dataset_oversubscription` उपयोग करें

`--allow_dataset_oversubscription` फ़्लैग ([OPTIONS.md](OPTIONS.md#--allow_dataset_oversubscription) में दस्तावेज़ित) आपके कॉन्फ़िगरेशन के लिए न्यूनतम आवश्यक repeats को स्वतः गणना और लागू करता है, जिससे यह prototyping या छोटे datasets के लिए आदर्श बनता है।

केवल ऑडियो वाले प्रशिक्षण में भी हर अवधि बकेट के नमूने डेटा-पैरेलल रैंक में बाँटे जाते हैं। एक ही रेप्लिका के कॉन्टेक्स्ट-पैरेलल रैंक अपना हिस्सा साझा करते हैं। यदि VAE तैयार करते समय प्रशिक्षण इंडेक्स से नमूने हटते हैं, तो सैंपलिंग से पहले सहेजे गए नमूने फिर से रैंक में बाँटे जाते हैं।

पहले से इंडेक्स किए गए Webshart Parquet कैप्चर में नेटिव सैंपलिंग के साथ हर रैंक तैयार सदस्यता व्यू खोलकर संदर्भों के बैच पढ़ता है। समर्थित इमेज क्रॉप की ज्यामिति डिस्क पर शार्ड के अनुसार कैश होती है। यह कैश ऐसे स्टोरेज पर रखें जिसे संबंधित रैंक खोल सकें। स्रोत ऑब्जेक्ट और मैनिफ़ेस्ट स्थिर रखें। चेकपॉइंट से जारी रखते समय डेटासेट के नियम, सीड, सैंपलिंग नीति, रैंक की संख्या, बैच साइज़ और ग्रेडिएंट अक्युमुलेशन की सेटिंग एक जैसी रखें। प्रीफ़ेच बैचों को बफ़र करता है। चेकपॉइंट उन बैचों को दर्ज करता है जिन्हें प्रशिक्षण ने इस्तेमाल किया।

`shard_local` के लिए योग्य, बिना इंडेक्स वाले Parquet इमेज डेटासेट में सभी डेटा-पैरेलल रैंक एक ही विंडो पढ़ते हैं। हर विंडो में अधिकतम चार आसन्न शार्ड होते हैं और हर रैंक को प्रत्येक ग्लोबल बैच का अपना हिस्सा मिलता है। विंडो के अंत में हर आस्पेक्ट बकेट के वे नमूने छोड़ दिए जाते हैं जिनसे ग्रेडिएंट अक्युमुलेशन का पूरा चक्र नहीं बनता। प्रशिक्षण फिर शुरू करने पर सहेजी गई विंडो खुलती है और आखिरी इस्तेमाल किए गए बैच के बाद से पढ़ाई जारी रहती है।

### Slow image scan / discovery

**discovery** backend वर्तमान में aspect bucket data collection को एक ही node तक सीमित करता है। बहुत बड़े datasets में यह **काफी** समय ले सकता है क्योंकि हर इमेज को storage से पढ़कर उसकी geometry निकालनी पड़ती है।

इस समस्या के लिए [parquet metadata_backend](DATALOADER.md#parquet-caption-strategy-json-lines-datasets) का उपयोग करें, जिससे आप डेटा को अपने तरीके से preprocess कर सकते हैं। जैसा कि linked डॉक्युमेंट सेक्शन में बताया गया है, parquet table में `filename`, `width`, `height`, और `caption` columns होते हैं ताकि डेटा को जल्दी और कुशलता से buckets में वर्गीकृत किया जा सके।


### Storage space

बहुत बड़े datasets, खासकर T5-XXL text encoder उपयोग करते समय, मूल data, image embeds, और text embeds के लिए अत्यधिक जगह लेते हैं।

#### Cloud storage

Cloudflare R2 जैसे providers का उपयोग करके बहुत बड़े datasets को कम storage फीस में रखा जा सकता है।

`multidatabackend.json` में `aws` type कॉन्फ़िगर करने का उदाहरण [dataloader configuration guide](DATALOADER.md#local-cache-with-cloud-dataset) में देखें।

- Image data लोकल या S3 पर रखा जा सकता है
  - यदि इमेजेस S3 में हैं, तो preprocessing speed नेटवर्क bandwidth के अनुसार घटती है
  - यदि इमेजेस लोकल हैं, तो **training** के दौरान NVMe throughput का लाभ नहीं मिलता
- Image embeds और text embeds को अलग‑अलग लोकल या cloud storage पर रखा जा सकता है
  - Embeds को cloud storage पर रखने से training rate पर बहुत कम असर पड़ता है, क्योंकि वे parallel में fetch होते हैं

Ideally, सभी इमेजेस और embeds किसी cloud storage bucket में हों। इससे preprocessing और training resume के दौरान समस्याओं का जोखिम कम होता है।

#### On‑demand VAE encoding

ऐसे बड़े datasets के लिए जहाँ cached VAE latents रखना storage constraints या धीमे shared storage के कारण व्यावहारिक नहीं है, आप `--vae_cache_disable` उपयोग कर सकते हैं। यह VAE cache को पूरी तरह disable करता है और training के दौरान VAE को images on‑the‑fly encode करने पर मजबूर करता है।

इससे GPU compute बढ़ता है, लेकिन cached latents के लिए storage और network I/O काफी कम हो जाता है।

#### Filesystem scan caches संरक्षित करना

यदि आपके datasets इतने बड़े हैं कि नई इमेजेस स्कैन करना bottleneck बन जाए, तो हर dataloader config entry में `preserve_data_backend_cache=true` जोड़कर backend को नई इमेजेस के लिए स्कैन होने से रोक सकते हैं।

**Note** कि आपको फिर `image_embeds` data backend type ([यहाँ अधिक जानकारी](DATALOADER.md#local-cache-with-cloud-dataset)) उपयोग करना चाहिए ताकि cache lists अलग रहें, खासकर यदि preprocessing job interrupted हो जाए। इससे **image list** startup पर दोबारा scan नहीं होगी।

#### Data compression

`config.json` में नीचे दिए गए मान जोड़कर data compression सक्षम करें:

```json
{
    ...
    "--compress_disk_cache": true,
    ...
}
```

यह inline gzip का उपयोग करके बड़े text और image embeds द्वारा ली गई redundant disk space को कम करेगा।

## 🤗 Accelerate के जरिए कॉन्फ़िगरेशन

जब `accelerate config` (`~/.cache/huggingface/accelerate/default_config.yaml`) का उपयोग करके SimpleTuner चलाते हैं, तो ये विकल्प `config/config.env` की सामग्री पर प्राथमिकता लेते हैं।

DeepSpeed के बिना Accelerate के लिए एक उदाहरण default_config:

```yaml
# this should be updated on EACH node.
machine_rank: 0
# Everything below here is the same on EACH node.
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
dynamo_config:
  dynamo_backend: NO
enable_cpu_affinity: false
main_process_ip: 10.0.0.100
main_process_port: 8080
main_training_function: main
mixed_precision: bf16
num_machines: 4
num_processes: 32
rdzv_backend: static
same_network: false
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### DeepSpeed

यह दस्तावेज़ [dedicated page](DEEPSPEED.md) जितना विवरण नहीं देता।

Multi‑node पर DeepSpeed optimize करते समय, सबसे कम संभव ZeRO level चुनना **आवश्यक** है।

उदाहरण के लिए, 80G NVIDIA GPU ZeRO level 1 offload के साथ Flux सफलतापूर्वक ट्रेन कर सकता है, जिससे overhead काफ़ी कम हो जाता है।

निम्न लाइनों को जोड़ें:

```yaml
# Update this from MULTI_GPU to DEEPSPEED
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 1
  gradient_clipping: 0.01
  zero3_init_flag: false
  zero_stage: 1
```

### torch compile ऑप्टिमाइज़ेशन

अधिक performance के लिए (compatibility issues की कीमत पर) आप torch compile सक्षम कर सकते हैं; हर node की yaml config में निम्न लाइनों को जोड़ें:

```yaml
dynamo_config:
  # Update this from NO to INDUCTOR
  dynamo_backend: INDUCTOR
  dynamo_mode: max-autotune
  dynamo_use_dynamic: false
  dynamo_use_fullgraph: false
```

## अपेक्षित प्रदर्शन

- 4x H100 SXM5 nodes local network से जुड़े
- 1TB memory प्रति node
- shared S3‑compatible data backend (Cloudflare R2) से training cache streaming, same region में
- प्रति accelerator batch size **8**, और **कोई** gradient accumulation steps नहीं
  - कुल effective batch size **256**
- 1024px resolution पर data bucketing सक्षम
- **Speed**: Flux.1‑dev (12B) को full‑rank पर ट्रेन करते समय 1024x1024 data के साथ 15 seconds per step

कम batch sizes, कम resolution, और torch compile सक्षम करने से speed **iterations per second** तक जा सकती है:

- resolution 512px तक घटाएँ और data bucketing disable करें (केवल square crops)
- DeepSpeed optimizer को AdamW से Lion fused optimiser पर बदलें
- torch compile को max‑autotune के साथ सक्षम करें
- **Speed**: 2 iterations per second

## GPU Health Monitoring

SimpleTuner में automatic GPU health monitoring शामिल है जो hardware failures को जल्दी detect करता है, जो distributed training में विशेष रूप से महत्वपूर्ण है जहाँ एक GPU की failure पूरे cluster में compute समय और पैसा बर्बाद कर सकती है।

### GPU Circuit Breaker

**GPU circuit breaker** हमेशा enabled रहता है और निम्न को monitor करता है:

- **ECC errors** - Uncorrectable memory errors detect करता है (A100/H100 GPUs के लिए महत्वपूर्ण)
- **Temperature** - Thermal shutdown threshold के पास पहुँचने पर alert
- **Throttling** - Thermal या power issues से hardware slowdown detect करता है
- **CUDA errors** - Training के दौरान runtime errors capture करता है

जब GPU fault detect होता है:

1. एक `gpu.fault` webhook emit होता है (यदि webhooks configured हैं)
2. Circuit open होता है ताकि faulty hardware पर आगे training रुके
3. Training cleanly exit होती है ताकि orchestrators instance terminate कर सकें

### Webhook configuration

GPU fault alerts प्राप्त करने के लिए अपने `config.json` में webhooks configure करें:

```json
{
  "--webhook_config": "config/webhooks.json"
}
```

Discord alerts के लिए उदाहरण `webhooks.json`:

```json
{
  "webhook_url": "https://discord.com/api/webhooks/...",
  "webhook_type": "discord"
}
```

### Multi-node considerations

Multi-node training में:

- हर node अपना GPU health monitor चलाता है
- किसी भी node पर GPU fault उस node से webhook trigger करता है
- Distributed communication failure के कारण training job सभी nodes पर fail होगा
- Orchestrators को cluster में किसी भी node से failures monitor करने चाहिए

विस्तृत webhook payload format और programmatic access के लिए [Resilience Infrastructure](experimental/cloud/RESILIENCE.md#gpu-circuit-breaker) देखें।

## Distributed training caveats

- हर node पर समान संख्या में accelerators उपलब्ध होना चाहिए
- केवल LoRA/LyCORIS को quantize किया जा सकता है, इसलिए full distributed model training के लिए DeepSpeed आवश्यक है
- यह बहुत उच्च‑लागत वाला ऑपरेशन है, और बड़े batch sizes आपको धीमा कर सकते हैं, जिससे GPU count बढ़ाने की ज़रूरत पड़ सकती है। बजटिंग का सावधानी से संतुलन रखना चाहिए।
- (DeepSpeed) ZeRO 3 के साथ training करते समय validations disable करनी पड़ सकती हैं
- (DeepSpeed) ZeRO level 3 के साथ saving करने पर मॉडल के sharded copies बनते हैं, जबकि levels 1 और 2 अपेक्षित रूप से काम करते हैं
- (DeepSpeed) DeepSpeed के CPU‑based optimisers का उपयोग आवश्यक हो जाता है क्योंकि यह optim states के sharding और offload को संभालता है।
