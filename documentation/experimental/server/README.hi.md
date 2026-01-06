# Server और Multi-User फीचर्स

यह डायरेक्टरी SimpleTuner के server-side फीचर्स का डॉक्यूमेंटेशन रखती है जो local और cloud प्रशिक्षण दोनों पर लागू होते हैं।

## सामग्री

- [Worker Orchestration](WORKERS.md) - Distributed worker registration, job dispatch, और GPU fleet प्रबंधन
- [Enterprise Guide](ENTERPRISE.md) - Multi-user deployment, SSO, approvals, quotas, और governance
- [External Authentication](EXTERNAL_AUTH.md) - OIDC और LDAP identity provider सेटअप
- [Audit Logging](AUDIT.md) - chain verification के साथ सुरक्षा event logging

## इन डॉक्यूमेंट्स का उपयोग कब करें

ये फीचर्स तब प्रासंगिक हैं जब:

- worker orchestration के साथ कई GPU मशीनों पर प्रशिक्षण वितरित करना हो
- SimpleTuner को कई उपयोगकर्ताओं के लिए साझा सेवा के रूप में चलाना हो
- corporate identity providers (Okta, Azure AD, Keycloak, LDAP) के साथ इंटीग्रेशन करना हो
- job submission के लिए approval workflows चाहिए हों
- compliance या सुरक्षा के लिए user actions ट्रैक करने हों
- टीम quotas और resource limits प्रबंधित करने हों

क्लाउड-स्पेसिफिक डॉक्यूमेंटेशन (Replicate, job queues, webhooks) के लिए [Cloud Training](../cloud/README.md) देखें।
