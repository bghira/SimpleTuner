# Audit Logging

SimpleTuner का audit logging सिस्टम सुरक्षा-प्रासंगिक घटनाओं का tamper-evident रिकॉर्ड देता है। सभी administrative actions, authentication events, और job operations को cryptographic chain verification के साथ लॉग किया जाता है।

## ओवरव्यू

Audit log में यह कैप्चर होता है:
- **Authentication events**: Login attempts (success/failure), logouts, session expirations
- **User management**: User creation, updates, deletions, permission changes
- **API key operations**: Key creation, revocation, usage
- **Credential management**: Provider credential changes
- **Job operations**: Submissions, cancellations, approvals

## Audit Logs तक पहुँचना

### Web UI

Admin पैनल में **Audit** टैब पर जाएँ और filtering विकल्पों के साथ audit entries ब्राउज़ करें।

### CLI

```bash
# List recent audit entries
simpletuner auth audit list

# Filter by event type
simpletuner auth audit list --event-type auth.login.failed

# Filter by user
simpletuner auth audit user 123

# View security events only
simpletuner auth audit security

# Get statistics
simpletuner auth audit stats

# Verify chain integrity
simpletuner auth audit verify
```

### API Endpoints

सभी endpoints के लिए `admin.audit` permission आवश्यक है।

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/audit` | filters के साथ audit entries सूचीबद्ध करें |
| GET | `/api/audit/stats` | audit statistics प्राप्त करें |
| GET | `/api/audit/types` | उपलब्ध event types सूचीबद्ध करें |
| GET | `/api/audit/verify` | chain integrity सत्यापित करें |
| GET | `/api/audit/user/{user_id}` | किसी user के लिए entries प्राप्त करें |
| GET | `/api/audit/security` | security-related events प्राप्त करें |

## Event Types

### Authentication Events

| Event | Description |
|-------|-------------|
| `auth.login.success` | सफल login |
| `auth.login.failed` | असफल login प्रयास |
| `auth.logout` | User ने logout किया |
| `auth.session.expired` | Session समाप्त |
| `auth.api_key.used` | API key का उपयोग हुआ |

### User Management Events

| Event | Description |
|-------|-------------|
| `user.created` | नया user बनाया गया |
| `user.updated` | User विवरण अपडेट हुए |
| `user.deleted` | User हटाया गया |
| `user.password.changed` | User ने अपना पासवर्ड बदला |
| `user.level.changed` | User level/role बदला |
| `user.permission.changed` | User permission बदला |

### API Key Events

| Event | Description |
|-------|-------------|
| `api_key.created` | नया API key बनाया गया |
| `api_key.revoked` | API key revoke किया गया |

### Credential Events

| Event | Description |
|-------|-------------|
| `credential.created` | Provider credential जोड़ा गया |
| `credential.deleted` | Provider credential हटाया गया |
| `credential.used` | Credential का उपयोग हुआ |

### Job Events

| Event | Description |
|-------|-------------|
| `job.submitted` | Job queue में सबमिट हुआ |
| `job.cancelled` | Job रद्द किया गया |
| `job.approved` | Job approval दिया गया |
| `job.rejected` | Job approval अस्वीकार किया गया |

## Query Parameters

Audit entries सूचीबद्ध करते समय आप निम्न filters उपयोग कर सकते हैं:

| Parameter | Type | Description |
|-----------|------|-------------|
| `event_type` | string | event type के आधार पर फ़िल्टर |
| `actor_id` | int | action करने वाले user के आधार पर फ़िल्टर |
| `target_type` | string | target resource type के आधार पर फ़िल्टर |
| `target_id` | string | target resource ID के आधार पर फ़िल्टर |
| `since` | ISO date | शुरूआती timestamp |
| `until` | ISO date | अंतिम timestamp |
| `limit` | int | अधिकतम entries (1-500, डिफ़ॉल्ट 50) |
| `offset` | int | pagination offset |

## Chain Integrity

हर audit entry में शामिल है:
- उसकी सामग्री का cryptographic hash
- पिछले entry के hash का reference
- monotonic clock से timestamp

यह एक hash chain बनाता है जिससे tampering पकड़ी जा सकती है। integrity जांचने के लिए verify endpoint या CLI कमांड उपयोग करें:

```bash
# Verify entire chain
simpletuner auth audit verify

# Verify specific range
simpletuner auth audit verify --start-id 100 --end-id 200
```

Verification checks:
1. हर entry का hash उसकी सामग्री से मेल खाता है
2. हर entry पिछले entry के hash को सही तरह reference करता है
3. sequence में कोई gap नहीं

## Retention

Audit logs SimpleTuner database में स्टोर होते हैं। अपने deployment में retention कॉन्फ़िगर करें:

```bash
# Environment variable for retention period (days)
SIMPLETUNER_AUDIT_RETENTION_DAYS=365
```

पुरानी entries को आपकी compliance आवश्यकताओं के अनुसार archive या purge किया जा सकता है।

## Security Considerations

- Audit logs append-only हैं; entries को API के जरिए बदला या हटाया नहीं जा सकता
- logs देखने के लिए `admin.audit` permission आवश्यक है
- असफल login प्रयास IP addresses के साथ लॉग होते हैं ताकि security monitoring हो सके
- प्रोडक्शन deployments के लिए audit logs को SIEM में forward करने पर विचार करें
