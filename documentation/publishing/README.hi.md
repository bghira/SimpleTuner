# Publishing Providers

SimpleTuner अब `--publishing_config` के जरिए training outputs को कई destinations पर publish कर सकता है। Hugging Face uploads अभी भी `--push_to_hub` से नियंत्रित रहते हैं; `publishing_config` अन्य providers के लिए additive है और मुख्य प्रोसेस पर validation पूरा होने के बाद चलता है।

## कॉन्फ़िग फ़ॉर्मेट
- inline JSON (`--publishing_config='[{"provider": "s3", ...}]'`), SDK के जरिए Python dict, या JSON फ़ाइल का पाथ स्वीकार करता है।
- Values एक list में normalise होती हैं, ठीक वैसे जैसे `--webhook_config` व्यवहार करता है।
- हर entry में `provider` key आवश्यक है। optional `base_path` remote destination के अंदर paths को prefix करता है। यदि आपका कॉन्फ़िग URI नहीं दे सकता, तो provider पहली बार query होने पर एक warning लॉग करेगा।

## डिफ़ॉल्ट artifact
Publishing रन की `output_dir` (फ़ोल्डर्स और फ़ाइलें) को डायरेक्टरी के baseline नाम के साथ अपलोड करता है। Metadata में वर्तमान job id और validation type शामिल होते हैं ताकि downstream consumers URI को रन से जोड़ सकें।

## Providers
जब आप कोई provider उपयोग करें, तो प्रोजेक्ट `.venv` में वैकल्पिक dependencies इंस्टॉल करें।

### S3-compatible और Backblaze B2 (S3 API)
- Provider: `s3` या `backblaze_b2`
- Dependency: `pip install boto3`
- उदाहरण:
```json
[
  {
    "provider": "s3",
    "bucket": "simpletuner-models",
    "region": "us-east-1",
    "access_key": "AKIA...",
    "secret_key": "SECRET",
    "base_path": "runs/2024",
    "endpoint_url": "https://s3.us-west-004.backblazeb2.com",
    "public_base_url": "https://cdn.example.com/models"
  }
]
```

⚠️ **सुरक्षा नोट**: क्रेडेंशियल्स को कभी भी version control में commit न करें। प्रोडक्शन deployments के लिए environment variable substitution या secrets manager का उपयोग करें।

### Azure Blob Storage
- Provider: `azure_blob` (alias `azure`)
- Dependency: `pip install azure-storage-blob`
- उदाहरण:
```json
[
  {
    "provider": "azure_blob",
    "connection_string": "DefaultEndpointsProtocol=....",
    "container": "simpletuner",
    "base_path": "models/latest"
  }
]
```

### Dropbox
- Provider: `dropbox`
- Dependency: `pip install dropbox`
- उदाहरण:
```json
[
  {
    "provider": "dropbox",
    "token": "sl.12345",
    "base_path": "/SimpleTuner/runs"
  }
]
```
बड़ी फ़ाइलें upload sessions में अपने आप stream होती हैं; अनुमति मिलने पर shared links बनाए जाते हैं, अन्यथा `dropbox://` path रिकॉर्ड किया जाता है।

## CLI उपयोग
```
simpletuner-train \
  --publishing_config=config/publishing.json \
  --push_to_hub=true \
  ...
```
यदि आप SimpleTuner को programmatically कॉल कर रहे हैं, तो `publishing_config` में list/dict पास करें और यह आपके लिए normalise कर दिया जाएगा।
