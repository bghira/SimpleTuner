# API ट्रेनिंग ट्यूटोरियल

## परिचय

यह गाइड SimpleTuner ट्रेनिंग जॉब्स को **पूरी तरह HTTP API के जरिए** चलाने की प्रक्रिया दिखाती है, जबकि सेटअप और डेटासेट प्रबंधन कमांड लाइन पर ही रहता है। यह अन्य ट्यूटोरियल्स की संरचना का अनुसरण करती है, लेकिन WebUI ऑनबोर्डिंग को छोड़ देती है। आप:

- यूनिफाइड सर्वर इंस्टॉल और स्टार्ट करेंगे
- OpenAPI स्कीमा ढूंढेंगे और डाउनलोड करेंगे
- REST कॉल्स से environment बनाएंगे और अपडेट करेंगे
- `/api/training` के जरिए ट्रेनिंग जॉब्स को वैलिडेट, लॉन्च और मॉनिटर करेंगे
- दो सिद्ध कॉन्फ़िगरेशन चुनेंगे: PixArt Sigma 900M का फुल फाइन-ट्यून और Flux Kontext LyCORIS LoRA रन

## पूर्वापेक्षाएँ

- Python 3.10–3.13, Git, और `pip`
- वर्चुअल एनवायरनमेंट में SimpleTuner इंस्टॉल (`pip install 'simpletuner[cuda]'` या आपके प्लेटफ़ॉर्म के अनुरूप वैरिएंट)
  - CUDA 13 / Blackwell users (NVIDIA B-series GPUs): `pip install 'simpletuner[cuda13]'`
- आवश्यक Hugging Face रिपॉजिटरीज़ तक पहुंच (`huggingface-cli login` पहले करें, अगर मॉडल gated हों)
- लोकल में staged डेटासेट्स जिनके साथ कैप्शन्स हों (PixArt के लिए caption टेक्स्ट फाइलें, Kontext के लिए paired edit/reference फोल्डर्स)
- `curl` और `jq` वाला शेल

## सर्वर शुरू करें

अपने SimpleTuner चेकआउट से (या जहां पैकेज इंस्टॉल हो):

```bash
simpletuner server --port 8001
```

API `http://localhost:8001` पर उपलब्ध है। सर्वर को चलता रहने दें और नीचे दिए गए कमांड्स दूसरे टर्मिनल में चलाएं।

> **टिप:** अगर आपके पास पहले से ट्रेन करने के लिए कॉन्फ़िगरेशन environment तैयार है, तो `--env` के साथ सर्वर स्टार्ट कर सकते हैं ताकि सर्वर पूरी तरह लोड होने पर ट्रेनिंग अपने आप शुरू हो जाए:
>
> ```bash
> simpletuner server --port 8001 --env my-training-config
> ```
>
> यह स्टार्टअप पर आपके कॉन्फ़िगरेशन को वैलिडेट करता है और सर्वर तैयार होते ही ट्रेनिंग लॉन्च कर देता है—अनअटेंडेड या स्क्रिप्टेड डिप्लॉयमेंट के लिए उपयोगी। `--env` विकल्प `simpletuner train --env` जैसा ही व्यवहार करता है।

### कॉन्फ़िगरेशन और डिप्लॉयमेंट

प्रोडक्शन उपयोग के लिए, आप bind address और port को कॉन्फ़िगर कर सकते हैं:

| विकल्प | Environment Variable | डिफ़ॉल्ट | विवरण |
|--------|----------------------|---------|-------|
| `--host` | `SIMPLETUNER_HOST` | `0.0.0.0` | सर्वर को किस address पर bind करना है (reverse proxy के पीछे `127.0.0.1` रखें) |
| `--port` | `SIMPLETUNER_PORT` | `8001` | सर्वर का port |

<details>
<summary><b>Production Deployment Options (TLS, Reverse Proxy, Systemd, Docker)</b></summary>

प्रोडक्शन डिप्लॉयमेंट के लिए TLS termination हेतु reverse proxy इस्तेमाल करने की सलाह दी जाती है।

#### Nginx Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name training.example.com;

    # TLS configuration (example using Let's Encrypt paths)
    ssl_certificate /etc/letsencrypt/live/training.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/training.example.com/privkey.pem;

    # WebSocket support for SSE streaming (Critical for real-time logs)
    location /api/training/stream {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        # SSE-specific settings
        proxy_buffering off;
        proxy_read_timeout 86400s;
    }

    # Main application
    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        # Large file uploads for datasets
        client_max_body_size 10G;
        proxy_request_buffering off;
    }
}
```

#### Caddy Configuration

```caddyfile
training.example.com {
    reverse_proxy 127.0.0.1:8001 {
        # SSE streaming support
        flush_interval -1
    }
    # Large file uploads
    request_body {
        max_size 10GB
    }
}
```

#### systemd Service

```ini
[Unit]
Description=SimpleTuner Training Server
After=network.target

[Service]
Type=simple
User=trainer
WorkingDirectory=/home/trainer/simpletuner-workspace
Environment="SIMPLETUNER_HOST=127.0.0.1"
Environment="SIMPLETUNER_PORT=8001"
ExecStart=/home/trainer/simpletuner-workspace/.venv/bin/simpletuner server
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

#### Docker Compose with Traefik

```yaml
version: '3.8'
services:
  simpletuner:
    image: ghcr.io/bghira/simpletuner:latest
    command: simpletuner server --host 0.0.0.0 --port 8001
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.simpletuner.rule=Host(`training.example.com`)"
      - "traefik.http.services.simpletuner.loadbalancer.server.port=8001"
```
</details>
## प्रमाणीकरण

SimpleTuner मल्टी-यूज़र ऑथेंटिकेशन सपोर्ट करता है। पहली बार लॉन्च पर आपको एक admin अकाउंट बनाना होगा।

### पहली बार सेटअप

देखें कि सेटअप की जरूरत है या नहीं:

```bash
curl -s http://localhost:8001/api/cloud/auth/setup/status | jq
```

अगर `needs_setup` `true` है, तो पहला admin बनाएं:

```bash
curl -s -X POST http://localhost:8001/api/cloud/auth/setup/first-admin \
  -H 'Content-Type: application/json' \
  -d '{
    "email": "admin@example.com",
    "username": "admin",
    "password": "your-secure-password"
  }'
```

### API keys

स्क्रिप्टेड एक्सेस के लिए, लॉगिन के बाद API key जनरेट करें:

```bash
# Login first (stores session cookie)
curl -s -X POST http://localhost:8001/api/cloud/auth/login \
  -H 'Content-Type: application/json' \
  -c cookies.txt \
  -d '{"username": "admin", "password": "your-secure-password"}'

# Create an API key
curl -s -X POST http://localhost:8001/api/cloud/auth/api-keys \
  -H 'Content-Type: application/json' \
  -b cookies.txt \
  -d '{"name": "automation-key"}' | jq
```

रिटर्न हुई key (जो `st_` से शुरू होती है) को आगे की रिक्वेस्ट्स में इस्तेमाल करें:

```bash
curl -s http://localhost:8001/api/training/status \
  -H 'X-API-Key: st_your_key_here'
```

### यूज़र प्रबंधन

Admins API से या WebUI की **Manage Users** पेज से अतिरिक्त यूज़र्स बना सकते हैं:

```bash
# Create a new user (requires admin session)
curl -s -X POST http://localhost:8001/api/users \
  -H 'Content-Type: application/json' \
  -b cookies.txt \
  -d '{
    "email": "researcher@example.com",
    "username": "researcher",
    "password": "their-password",
    "level_names": ["researcher"]
  }'
```

> **नोट:** डिफ़ॉल्ट रूप से public registration बंद रहता है। Admins **Manage Users → Registration** टैब में इसे चालू कर सकते हैं, लेकिन निजी डिप्लॉयमेंट्स के लिए इसे बंद रखना बेहतर है।

## API खोजें

FastAPI इंटरएक्टिव डॉक्यूमेंटेशन और OpenAPI स्कीमा उपलब्ध कराता है:

```bash
# FastAPI Swagger UI
python -m webbrowser http://localhost:8001/docs

# ReDoc view
python -m webbrowser http://localhost:8001/redoc

# Download the schema for local inspection
curl -o openapi.json http://localhost:8001/openapi.json
jq '.info' openapi.json
```

इस ट्यूटोरियल में इस्तेमाल किए गए हर endpoint का डॉक्यूमेंटेशन वहीं `configurations` और `training` टैग्स के तहत है।

## फास्ट पाथ: environments के बिना रन करें

अगर आप **config/environment मैनेजमेंट पूरी तरह छोड़ना चाहते हैं**, तो पूरी CLI-जैसी payload सीधे training endpoints को POST करके एक-बार वाला रन शुरू कर सकते हैं:

1. अपने डेटासेट का वर्णन करने वाली dataloader JSON बनाएं या reuse करें। ट्रेनर को `--data_backend_config` में रेफ़र किया गया path ही चाहिए।

   ```bash
   cat <<'JSON' > config/multidatabackend-once.json
   [
     {
       "id": "demo-images",
       "type": "local",
       "dataset_type": "image",
       "instance_data_dir": "/data/datasets/demo",
       "caption_strategy": "textfile",
       "resolution": 1024,
       "resolution_type": "pixel_area"
     },
     {
       "id": "demo-text-embeds",
       "type": "local",
       "dataset_type": "text_embeds",
       "default": true,
       "cache_dir": "/data/cache/text/demo"
     }
   ]
   JSON
   ```

2. इनलाइन कॉन्फ़िगरेशन को वैलिडेट करें। हर ज़रूरी CLI आर्ग्यूमेंट दें (`--model_family`, `--model_type`, `--pretrained_model_name_or_path`, `--output_dir`, `--data_backend_config`, और `--num_train_epochs` या `--max_train_steps` में से एक):

   ```bash
   curl -s -X POST http://localhost:8001/api/training/validate \
     -F __active_tab__=model \
     -F --model_family=pixart_sigma \
     -F --model_type=full \
     -F --model_flavour=900M-1024-v0.6 \
     -F --pretrained_model_name_or_path=terminusresearch/pixart-900m-1024-ft-v0.6 \
     -F --output_dir=/workspace/output/inline-demo \
     -F --data_backend_config=config/multidatabackend-once.json \
     -F --train_batch_size=1 \
     -F --learning_rate=0.0001 \
     -F --max_train_steps=200 \
     -F --num_train_epochs=0
   ```

   हरा “Configuration Valid” स्निपेट पुष्टि करता है कि ट्रेनर payload को स्वीकार करेगा।

3. **उसी** form fields के साथ ट्रेनिंग लॉन्च करें (आप `--seed` या `--validation_prompt` जैसे overrides जोड़ सकते हैं):

   ```bash
   curl -s -X POST http://localhost:8001/api/training/start \
     -F __active_tab__=model \
     -F --model_family=pixart_sigma \
     -F --model_type=full \
     -F --model_flavour=900M-1024-v0.6 \
     -F --pretrained_model_name_or_path=terminusresearch/pixart-900m-1024-ft-v0.6 \
     -F --output_dir=/workspace/output/inline-demo \
     -F --data_backend_config=config/multidatabackend-once.json \
     -F --train_batch_size=1 \
     -F --learning_rate=0.0001 \
     -F --max_train_steps=200 \
     -F --num_train_epochs=0 \
     -F --validation_prompt='test shot of <token>'
   ```

सर्वर सबमिटेड सेटिंग्स को defaults के साथ merge करता है, resolved config को सक्रिय फ़ाइल में लिखता है, और ट्रेनिंग शुरू कर देता है। आप यही तरीका किसी भी model family के लिए इस्तेमाल कर सकते हैं—बाकी सेक्शन तब के लिए हैं जब आपको re-usable environments चाहिए।
### ad-hoc रन मॉनिटर करें

आप वही status endpoints इस्तेमाल करके प्रगति ट्रैक कर सकते हैं, जो आगे गाइड में उपयोग होंगे:

- हाई-लेवल स्थिति, active job ID, और startup stage info के लिए `GET /api/training/status` पोल करें।
- incremental logs के लिए `GET /api/training/events?since_index=N` या `/api/training/events/stream` पर WebSocket से स्ट्रीम करें।

Push-स्टाइल अपडेट के लिए, अपने form fields के साथ webhook सेटिंग्स दें:

```bash
curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=model \
  -F --model_family=pixart_sigma \
  ... \
  -F --webhook_config='[{"webhook_type":"raw","callback_url":"https://example.com/simpletuner","log_level":"info","ssl_no_verify":false}]' \
  -F --webhook_reporting_interval=10
```

Payload को string के रूप में JSON serialise करना जरूरी है; सर्वर `callback_url` पर job lifecycle अपडेट्स भेजता है। समर्थित फ़ील्ड्स के लिए `documentation/OPTIONS.md` में `--webhook_config` का विवरण या `config/webhooks.json` टेम्पलेट देखें।

<details>
<summary><b>Reverse Proxies के लिए Webhook Configuration</b></summary>

HTTPS के साथ reverse proxy इस्तेमाल करते समय आपका webhook URL public endpoint होना चाहिए:

1.  **Public Server:** `https://training.example.com/webhook/callback`
2.  **Tunneling:** local dev के लिए ngrok या cloudflared इस्तेमाल करें।

**Real-time Logs (SSE) Troubleshooting:**
अगर `GET /api/training/events` काम करता है, लेकिन stream हैंग हो जाता है:
*   **Nginx:** `proxy_buffering off;` और `proxy_read_timeout` को बड़ा रखें (जैसे 86400s)।
*   **CloudFlare:** long-lived connections को terminate कर देता है; CloudFlare Tunnel का उपयोग करें या stream endpoint के लिए proxy bypass करें।
</details>

### मैन्युअल वैलिडेशन ट्रिगर करें

अगर आप शेड्यूल्ड वैलिडेशन इंटरवल्स **के बीच** evaluation run फ़ोर्स करना चाहते हैं, तो नया endpoint कॉल करें:

```bash
curl -s -X POST http://localhost:8001/api/training/validation/run
```

- सर्वर active `job_id` लौटाता है।
- ट्रेनर अगली gradient synchronization के तुरंत बाद validation run कतार में डालता है (यह current micro-batch को interrupt नहीं करता)।
- रन आपके configured validation prompts/settings को reuse करता है, इसलिए resulting images सामान्य event/log streams में दिखाई देती हैं।
- बिल्ट-इन pipeline की जगह external executable से validation offload करना चाहते हैं? config (या payload) में `--validation_method=external-script` सेट करें और `--validation_external_script` को अपनी स्क्रिप्ट पर पॉइंट करें। आप placeholders के साथ training context पास कर सकते हैं: `{local_checkpoint_path}`, `{global_step}`, `{tracker_run_name}`, `{tracker_project_name}`, `{model_family}`, `{huggingface_path}`, `{remote_checkpoint_path}` (validation के लिए खाली), साथ ही `validation_*` config values (जैसे `validation_num_inference_steps`, `validation_guidance`, `validation_noise_scheduler`)। अगर आप script को fire-and-forget रखना चाहते हैं, तो `--validation_external_background` enable करें ताकि training block न हो।
- हर बार जब checkpoint लोकली लिखा जाए (background uploads चल रहे हों तब भी) तुरंत automation चलाना चाहते हैं? `--post_checkpoint_script='/opt/hooks/run_eval.sh {local_checkpoint_path} {global_step}'` सेट करें। यह validation hooks वाले same placeholders उपयोग करता है; `{remote_checkpoint_path}` इस hook के लिए खाली resolve होता है।
- SimpleTuner के built-in uploads रखना चाहते हैं, लेकिन remote URL को अपनी toolchain को देना चाहते हैं? `--post_upload_script` कॉन्फ़िगर करें; यह प्रति publishing provider/Hugging Face Hub upload पर `{remote_checkpoint_path}` (यदि backend देता है) और वही context placeholders के साथ चलता है। SimpleTuner आपके script का परिणाम ingest नहीं करता, इसलिए अपने tracker में artifacts/metrics खुद लॉग करें।
  - उदाहरण: `--post_upload_script='/opt/hooks/notify.sh {remote_checkpoint_path} {tracker_project_name} {tracker_run_name}'` जहां `notify.sh` आपका tracker API कॉल करता है।
  - Working samples:
    - `simpletuner/examples/external-validation/replicate_post_upload.py` `{remote_checkpoint_path}`, `{model_family}`, `{model_type}`, `{lora_type}`, और `{huggingface_path}` के साथ Replicate inference ट्रिगर करता है।
    - `simpletuner/examples/external-validation/wavespeed_post_upload.py` WaveSpeed inference ट्रिगर करता है और completion तक poll करता है, वही placeholders इस्तेमाल करता है।
    - `simpletuner/examples/external-validation/fal_post_upload.py` fal.ai Flux LoRA inference ट्रिगर करता है (इसके लिए `FAL_KEY` और `model_family` में `flux` होना चाहिए)।
    - `simpletuner/examples/external-validation/use_second_gpu.py` बिना uploads के दूसरी GPU पर Flux LoRA inference चलाता है।

अगर कोई job active नहीं है तो endpoint HTTP 400 लौटाता है, इसलिए scripting retries से पहले `/api/training/status` चेक करें।

### मैन्युअल चेकपॉइंट ट्रिगर करें

वर्तमान मॉडल स्टेट को तुरंत persist करने के लिए (अगले शेड्यूल्ड checkpoint का इंतज़ार किए बिना), यह कॉल करें:

```bash
curl -s -X POST http://localhost:8001/api/training/checkpoint/run
```

- सर्वर active `job_id` लौटाता है।
- ट्रेनर अगली gradient synchronization के बाद शेड्यूल्ड checkpoints जैसी ही settings के साथ checkpoint सेव करता है (upload नियम, rolling retention, आदि)।
- Rolling cleanup और webhook notifications शेड्यूल्ड checkpoint जैसे ही व्यवहार करते हैं।

Validation की तरह, अगर कोई training job चल नहीं रहा है तो endpoint HTTP 400 लौटाता है।

### Validation previews स्ट्रीम करें

जिन models में Tiny AutoEncoder (या equivalent) hooks हैं, वे image/video sampling के दौरान **प्रति-स्टेप validation previews** emit कर सकते हैं। यह फीचर payload में CLI flags जोड़कर enable करें:

```bash
curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=validation \
  -F --validation_preview=true \
  -F --validation_preview_steps=4 \
  -F --validation_num_inference_steps=20 \
  …other fields…
```

- `--validation_preview` (डिफ़ॉल्ट `false`) preview decoder अनलॉक करता है।
- `--validation_preview_steps` तय करता है कि कितनी बार intermediate frames emit हों। ऊपर के उदाहरण में आपको steps 1,5,9,13,17,20 पर events मिलेंगे (पहला step हमेशा emit होता है, फिर हर चौथा step)।

हर preview एक `validation.image` event के रूप में प्रकाशित होता है (देखें `simpletuner/helpers/training/validation.py:899-929`)। आप इन्हें raw webhooks, `GET /api/training/events`, या `/api/training/events/stream` SSE stream से consume कर सकते हैं। एक typical payload:

```json
{
  "type": "validation.image",
  "title": "Validation (step 5/20): night bench",
  "body": "night bench shot of <token>",
  "data": {
    "step": 5,
    "timestep": 563.0,
    "resolution": [1024, 1024],
    "validation_type": "intermediary",
    "prompt": "night bench shot of <token>",
    "step_label": "5/20"
  },
  "images": [
    {"src": "data:image/png;base64,...", "mime_type": "image/png"}
  ]
}
```

Video-capable models `videos` array संलग्न करते हैं (GIF data URIs `mime_type: image/gif` के साथ)। चूंकि ये events near-real-time में stream होते हैं, आप इन्हें सीधे dashboards में दिखा सकते हैं या raw webhook backend के जरिए Slack/Discord पर भेज सकते हैं।
## सामान्य API वर्कफ़्लो

1. **Environment बनाएं** – `POST /api/configs/environments`
2. **Dataloader फ़ाइल भरें** – जनरेट हुई `config/<env>/multidatabackend.json` एडिट करें
3. **Training hyperparameters अपडेट करें** – `PUT /api/configs/<env>`
4. **Environment सक्रिय करें** – `POST /api/configs/<env>/activate`
5. **Training parameters वैलिडेट करें** – `POST /api/training/validate`
6. **Training लॉन्च करें** – `POST /api/training/start`
7. **जॉब मॉनिटर/स्टॉप करें** – `/api/training/status`, `/api/training/events`, `/api/training/stop`, `/api/training/cancel`

नीचे हर उदाहरण इसी फ्लो का अनुसरण करता है।

## वैकल्पिक: API के जरिए datasets upload करें (local backends)

अगर डेटासेट अभी उस मशीन पर नहीं है जहां SimpleTuner चल रहा है, तो आप dataloader जोड़ने से पहले HTTP के जरिए push कर सकते हैं। Upload endpoints कॉन्फ़िगर किए गए `datasets_dir` (WebUI onboarding के दौरान सेट) का सम्मान करते हैं और लोकल फाइल सिस्टम के लिए हैं:

1. **Datasets root के अंदर एक target folder बनाएं**:

   ```bash
   DATASETS_DIR=${DATASETS_DIR:-/workspace/simpletuner/datasets}
   curl -s -X POST http://localhost:8001/api/datasets/folders \
     -F parent_path="$DATASETS_DIR" \
     -F folder_name="pixart-upload"
   ```

2. **Files या ZIP upload करें** (images के साथ वैकल्पिक `.txt/.jsonl/.csv` metadata स्वीकार किए जाते हैं):

   ```bash
   # Upload a zip (automatically extracted on the server)
   curl -s -X POST http://localhost:8001/api/datasets/upload/zip \
     -F target_path="$DATASETS_DIR/pixart-upload" \
     -F file=@/path/to/dataset.zip

   # Or upload individual files
   curl -s -X POST http://localhost:8001/api/datasets/upload \
     -F target_path="$DATASETS_DIR/pixart-upload" \
     -F files[]=@image001.png \
     -F files[]=@image001.txt
   ```

> **Uploads Troubleshooting:** अगर reverse proxy के साथ बड़े uploads पर "Entity Too Large" error आती है, तो body size limit बढ़ाएं (जैसे Nginx में `client_max_body_size 10G;` या Caddy में `request_body { max_size 10GB }`)।

Upload के बाद, `multidatabackend.json` एंट्री को नए फोल्डर पर पॉइंट करें (उदाहरण के लिए, `"/data/datasets/pixart-upload"`)।

## उदाहरण: PixArt Sigma 900M फुल फाइन-ट्यून

### 1. REST के जरिए environment बनाएं

```bash
curl -s -X POST http://localhost:8001/api/configs/environments \
  -H 'Content-Type: application/json' \
  -d
```json
{
        "name": "pixart-api-demo",
        "model_family": "pixart_sigma",
        "model_flavour": "900M-1024-v0.6",
        "model_type": "full",
        "description": "PixArt 900M API-driven training"
      }
```

यह `config/pixart-api-demo/` और एक starter `multidatabackend.json` बनाता है।

### 2. डेटासेट वायर्ड करें

Dataloader फ़ाइल एडिट करें (paths को अपने actual dataset/cache locations से बदलें):

```bash
cat <<'JSON' > config/pixart-api-demo/multidatabackend.json
[
  {
    "id": "pixart-camera",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "/data/datasets/pseudo-camera-10k",
    "caption_strategy": "filename",
    "resolution": 1.0,
    "resolution_type": "area",
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "cache_dir_vae": "/data/cache/vae/pixart/pseudo-camera-10k",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square",
    "metadata_backend": "discovery"
  },
  {
    "id": "pixart-text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "/data/cache/text/pixart/pseudo-camera-10k",
    "write_batch_size": 128
  }
]
JSON
```

### 3. API के जरिए hyperparameters अपडेट करें

वर्तमान config लें, overrides merge करें, और परिणाम वापस push करें:

```bash
curl -s http://localhost:8001/api/configs/pixart-api-demo \
  | jq '.config + {
      "--output_dir": "/workspace/output/pixart900m",
      "--train_batch_size": 2,
      "--gradient_accumulation_steps": 2,
      "--learning_rate": 0.0001,
      "--optimizer": "adamw_bf16",
      "--lr_scheduler": "cosine",
      "--lr_warmup_steps": 500,
      "--max_train_steps": 1800,
      "--num_train_epochs": 0,
      "--validation_prompt": "a studio portrait of <token> wearing a leather jacket",
      "--validation_guidance": 3.8,
      "--validation_resolution": "1024x1024",
      "--validation_num_inference_steps": 28,
      "--cache_dir_vae": "/data/cache/vae/pixart",
      "--seed": 1337,
      "--resume_from_checkpoint": "latest",
      "--base_model_precision": "bf16",
      "--dataloader_prefetch": true,
      "--report_to": "none",
      "--checkpoints_total_limit": 4,
      "--validation_seed": 12345,
      "--data_backend_config": "pixart-api-demo/multidatabackend.json"
    }' > /tmp/pixart-config.json

jq '{
      "name": "pixart-api-demo",
      "description": "PixArt 900M full tune (API)",
      "tags": ["pixart", "api"],
      "config": .
    }' /tmp/pixart-config.json > /tmp/pixart-update.json

curl -s -X PUT http://localhost:8001/api/configs/pixart-api-demo \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/pixart-update.json
```

### 4. Environment सक्रिय करें

```bash
curl -s -X POST http://localhost:8001/api/configs/pixart-api-demo/activate
```

### 5. लॉन्च से पहले वैलिडेट करें

`validate` form-encoded data लेता है। कम से कम `num_train_epochs` या `max_train_steps` में से एक को 0 रखें:

```bash
curl -s -X POST http://localhost:8001/api/training/validate \
  -F __active_tab__=model \
  -F --num_train_epochs=0
```

सफलता ब्लॉक (`Configuration Valid`) का मतलब है कि ट्रेनर merged configuration स्वीकार करता है।

### 6. ट्रेनिंग शुरू करें

```bash
curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=model \
  -F --num_train_epochs=0
```

Response में job ID शामिल होता है। ट्रेनिंग step 3 में सेव किए गए parameters के साथ चलती है।

### 7. मॉनिटर और रोकें

```bash
# Query coarse status
curl -s http://localhost:8001/api/training/status | jq

# Stream incremental log events
curl -s 'http://localhost:8001/api/training/events?since_index=0' | jq

# Cancel or stop
curl -s -X POST http://localhost:8001/api/training/stop
curl -s -X POST http://localhost:8001/api/training/cancel -F job_id=<JOB_ID>
```

PixArt नोट्स:

- चुने गए `train_batch_size * gradient_accumulation_steps` के लिए डेटासेट पर्याप्त बड़ा रखें
- अगर आपको mirror चाहिए तो `HF_ENDPOINT` सेट करें, और `terminusresearch/pixart-900m-1024-ft-v0.6` डाउनलोड करने से पहले authenticate करें
- अपने prompts के अनुसार `--validation_guidance` को 3.6 और 4.4 के बीच ट्यून करें
## उदाहरण: Flux Kontext LyCORIS LoRA

Kontext अपनी pipeline का बड़ा हिस्सा Flux Dev के साथ शेयर करता है, लेकिन paired edit/reference images की जरूरत होती है।

### 1. Environment प्रोविज़न करें

```bash
curl -s -X POST http://localhost:8001/api/configs/environments \
  -H 'Content-Type: application/json' \
  -d
```json
{
        "name": "kontext-api-demo",
        "model_family": "flux",
        "model_flavour": "kontext",
        "model_type": "lora",
        "lora_type": "lycoris",
        "description": "Flux Kontext LoRA via API"
      }
```

### 2. Paired dataloader वर्णन करें

Kontext को edit/reference pairs और एक text-embed cache चाहिए:

```bash
cat <<'JSON' > config/kontext-api-demo/multidatabackend.json
[
  {
    "id": "kontext-edit",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "/data/datasets/kontext/edit",
    "conditioning_data": ["kontext-reference"],
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "caption_strategy": "textfile",
    "minimum_image_size": 768,
    "maximum_image_size": 1536,
    "target_downsample_size": 1024,
    "cache_dir_vae": "/data/cache/vae/kontext/edit",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square"
  },
  {
    "id": "kontext-reference",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/data/datasets/kontext/reference",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "/data/cache/vae/kontext/reference"
  },
  {
    "id": "kontext-text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "/data/cache/text/kontext"
  }
]
JSON
```

एडिट और reference फोल्डर में फाइलनाम मैच होना चाहिए; SimpleTuner नामों के आधार पर embeddings जोड़ता है।

### 3. Kontext-विशिष्ट hyperparameters लागू करें

```bash
curl -s http://localhost:8001/api/configs/kontext-api-demo \
  | jq '.config + {
      "--output_dir": "/workspace/output/kontext",
      "--train_batch_size": 1,
      "--gradient_accumulation_steps": 4,
      "--learning_rate": 0.00001,
      "--optimizer": "optimi-lion",
      "--lr_scheduler": "cosine",
      "--lr_warmup_steps": 200,
      "--max_train_steps": 12000,
      "--num_train_epochs": 0,
      "--validation_prompt": "a cinematic 1024px product photo of <token>",
      "--validation_guidance": 2.5,
      "--validation_resolution": "1024x1024",
      "--validation_num_inference_steps": 30,
      "--cache_dir_vae": "/data/cache/vae/kontext",
      "--seed": 777,
      "--resume_from_checkpoint": "latest",
      "--base_model_precision": "int8-quanto",
      "--dataloader_prefetch": true,
      "--report_to": "wandb",
      "--lora_rank": 16,
      "--lora_dropout": 0.05,
      "--conditioning_multidataset_sampling": "combined",
      "--clip_skip": 2,
      "--data_backend_config": "kontext-api-demo/multidatabackend.json"
    }' > /tmp/kontext-config.json

jq '{
      "name": "kontext-api-demo",
      "description": "Flux Kontext LyCORIS (API)",
      "tags": ["flux", "kontext", "api"],
      "config": .
    }' /tmp/kontext-config.json > /tmp/kontext-update.json

curl -s -X PUT http://localhost:8001/api/configs/kontext-api-demo \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/kontext-update.json
```

### 4. Activate, validate, और launch करें

```bash
curl -s -X POST http://localhost:8001/api/configs/kontext-api-demo/activate

curl -s -X POST http://localhost:8001/api/training/validate \
  -F __active_tab__=model \
  -F --num_train_epochs=0

curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=model \
  -F --num_train_epochs=0
```

Kontext टिप्स:

- `conditioning_type=reference_strict` crops को aligned रखता है; अगर आपके datasets के aspect ratio अलग हों तो `reference_loose` पर स्विच करें
- 1024 px पर 24 GB VRAM में रहने के लिए `int8-quanto` से quantise करें; full precision के लिए Hopper/Blackwell-class GPUs चाहिए
- multi-node runs के लिए सर्वर लॉन्च से पहले `--accelerate_config` या `CUDA_VISIBLE_DEVICES` सेट करें

## GPU-aware queuing के साथ local jobs सबमिट करें

Multi-GPU मशीन पर, आप queue API के जरिए GPU allocation awareness के साथ local training jobs सबमिट कर सकते हैं। अगर जरूरी GPUs उपलब्ध नहीं हैं तो jobs queued रहेंगे।

### GPU availability चेक करें

```bash
curl -s "http://localhost:8001/api/system/status?include_allocation=true" | jq '.gpu_allocation'
```

Response दिखाता है कि कौन से GPUs उपलब्ध हैं:

```json
{
  "allocated_gpus": [0, 1],
  "available_gpus": [2, 3],
  "running_local_jobs": 1,
  "devices": [
    {"index": 0, "name": "A100", "memory_gb": 40, "allocated": true, "job_id": "abc123"},
    {"index": 1, "name": "A100", "memory_gb": 40, "allocated": true, "job_id": "abc123"},
    {"index": 2, "name": "A100", "memory_gb": 40, "allocated": false, "job_id": null},
    {"index": 3, "name": "A100", "memory_gb": 40, "allocated": false, "job_id": null}
  ]
}
```

आप queue statistics (local GPU info सहित) भी पा सकते हैं:

```bash
curl -s http://localhost:8001/api/queue/stats | jq '.local'
```

### एक local job सबमिट करें

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "no_wait": false,
    "any_gpu": false
  }'
```

Options:

| विकल्प | डिफ़ॉल्ट | विवरण |
|--------|---------|-------------|
| `config_name` | required | चलाने के लिए training environment का नाम |
| `no_wait` | false | true होने पर GPUs उपलब्ध न हों तो तुरंत reject करें |
| `any_gpu` | false | true होने पर कॉन्फ़िगर किए device IDs की जगह कोई भी उपलब्ध GPU लें |

Response:

```json
{
  "success": true,
  "job_id": "abc123",
  "status": "running",
  "allocated_gpus": [0, 1],
  "queue_position": null
}
```

`status` फ़ील्ड परिणाम बताता है:

- `running` - job तुरंत शुरू हो गई और GPUs allocate हुए
- `queued` - job queued है, GPUs उपलब्ध होने पर शुरू होगी
- `rejected` - GPUs उपलब्ध नहीं थे और `no_wait` true था

### Local concurrency limits कॉन्फ़िगर करें

Admins queue concurrency endpoint से local jobs और GPUs की अधिकतम संख्या सीमित कर सकते हैं:

```bash
# Get current limits
curl -s http://localhost:8001/api/queue/stats | jq '{local_gpu_max_concurrent, local_job_max_concurrent}'

# Update limits (alongside cloud limits)
curl -s -X POST http://localhost:8001/api/queue/concurrency \
  -H 'Content-Type: application/json' \
  -d '{
    "local_gpu_max_concurrent": 6,
    "local_job_max_concurrent": 2
  }'
```

अनलिमिटेड GPU उपयोग के लिए `local_gpu_max_concurrent` को `null` पर सेट करें।

### CLI विकल्प

यही functionality CLI के जरिए भी उपलब्ध है:

```bash
# Submit with default queuing behavior
simpletuner jobs submit my-config

# Reject if GPUs unavailable
simpletuner jobs submit my-config --no-wait

# Use any available GPUs
simpletuner jobs submit my-config --any-gpu

# Preview what would happen (dry-run)
simpletuner jobs submit my-config --dry-run
```

## Remote workers को jobs dispatch करें

अगर आपके पास remote GPU मशीनें workers के रूप में रजिस्टर्ड हैं (देखें [Worker Orchestration](../experimental/server/WORKERS.md)), तो आप queue API से jobs उन्हें भेज सकते हैं।

### उपलब्ध workers चेक करें

```bash
curl -s http://localhost:8001/api/admin/workers | jq '.workers[] | {name, status, gpu_name, gpu_count}'
```

### किसी specific target पर सबमिट करें

```bash
# Prefer remote workers, fall back to local GPUs (default)
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "auto"
  }'

# Force dispatch to remote workers only
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "worker"
  }'

# Run only on orchestrator's local GPUs
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "local"
  }'
```

### Label के आधार पर workers चुनें

Workers में filtering के लिए labels हो सकते हैं (जैसे GPU type, location, team):

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "worker",
    "worker_labels": {"gpu_type": "a100*", "team": "nlp"}
  }'
```

Labels glob patterns सपोर्ट करते हैं (`*` किसी भी characters से match करता है)।

## काम के endpoints एक नज़र में

- `GET /api/configs/` – environments सूचीबद्ध करें (`?config_type=model` के साथ training configs)
- `GET /api/configs/examples` – shipped templates की सूची
- `POST /api/configs/{name}/dataloader` – defaults चाहिए तो dataloader फ़ाइल regenerate करें
- `GET /api/training/status` – high-level state, active `job_id`, और startup stage info
- `GET /api/training/events?since_index=N` – incremental trainer log stream
- `POST /api/training/checkpoints` – active job के output directory के checkpoints सूचीबद्ध करें
- `GET /api/system/status?include_allocation=true` – GPU allocation info सहित system metrics
- `GET /api/queue/stats` – local GPU allocation सहित queue stats
- `POST /api/queue/submit` – GPU-aware queuing के साथ local या worker job सबमिट करें
- `POST /api/queue/concurrency` – cloud और local concurrency limits अपडेट करें
- `GET /api/admin/workers` – रजिस्टर्ड workers और उनकी स्थिति सूचीबद्ध करें

## आगे क्या करें

- `documentation/OPTIONS.md` में options definitions देखें
- Automation के लिए इन REST calls को `jq`/`yq` या Python client के साथ मिलाएं
- Real-time dashboards के लिए `/api/training/events/stream` WebSockets को hook करें
- Exported configs (`GET /api/configs/<env>/export`) को version-control के लिए reuse करें
- Replicate के जरिए **cloud GPUs पर training चलाएं**—देखें [Cloud Training Tutorial](../experimental/cloud/TUTORIAL.md)

इन पैटर्न्स के साथ आप WebUI छुए बिना SimpleTuner training पूरी तरह स्क्रिप्ट कर सकते हैं, जबकि setup process के लिए भरोसेमंद CLI का सहारा बना रहता है।
