# SimpleTuner WebUI इम्प्लीमेंटेशन विवरण

## डिज़ाइन ओवरव्यू

SimpleTuner का API लचीलेपन को ध्यान में रखकर बनाया गया है।

`trainer` मोड में FastAPI को अन्य सेवाओं में जोड़ने के लिए एक single port खोला जाता है।
`unified` मोड में एक अतिरिक्त port WebUI के लिए खोला जाता है ताकि वह remote `trainer` प्रोसेस से callback events प्राप्त कर सके।

## वेब फ़्रेमवर्क

WebUI FastAPI का उपयोग करके बनाया और सर्व किया गया है;

- reactive components के लिए Alpine.js उपयोग होता है
- dynamic content loading और interactivity के लिए HTMX
- FastAPI के साथ Starlette और SSE-Starlette का उपयोग डेटा-केंद्रित API सर्व करने और real-time updates के लिए server-sent events (SSE) देने हेतु किया जाता है
- HTML templating के लिए Jinja2

Alpine को उसकी सरलता और आसान इंटीग्रेशन के कारण चुना गया—यह NodeJS को स्टैक से बाहर रखता है, जिससे डिप्लॉय और मेंटेन करना आसान होता है।

HTMX का सिंटैक्स सरल और हल्का है जो Alpine के साथ अच्छी तरह चलता है। यह dynamic content loading, form handling, और interactivity के लिए व्यापक क्षमताएँ देता है बिना पूरे frontend framework की जरूरत के।

मैंने Starlette और SSE-Starlette चुने क्योंकि मैं कोड डुप्लिकेशन को न्यूनतम रखना चाहता था; बहुत adhoc procedural कोड से अधिक declarative अप्रोच पर जाने के लिए काफी refactoring करना पड़ा।

### डेटा फ्लो

ऐतिहासिक रूप से FastAPI ऐप job clusters के अंदर एक “service worker” की तरह भी काम करता था: trainer boot होता, एक सीमित callback surface एक्सपोज़ करता, और remote orchestrators HTTP के माध्यम से status updates वापस भेजते थे। WebUI उसी callback bus का पुनः उपयोग करता है। unified मोड में हम trainer और interface दोनों को in-process चलाते हैं, जबकि trainer-only deployments अभी भी `/callbacks` में events push कर सकते हैं और एक अलग WebUI instance उन्हें SSE के जरिए consume कर सकता है। किसी नए queue या broker की जरूरत नहीं—हम उसी infrastructure पर निर्भर हैं जो headless deployments के साथ आता है।

## Backend आर्किटेक्चर

trainer UI अब core SDK पर आधारित है जो loose procedural helpers की जगह well-defined services एक्सपोज़ करता है। FastAPI अभी भी हर request टर्मिनेट करता है, लेकिन अधिकांश routes पतले delegators हैं जो service objects में कॉल करते हैं। इससे HTTP लेयर सरल रहती है और CLI, config wizard, और भविष्य के APIs के लिए reusability बढ़ती है।

### Route handlers

`simpletuner/simpletuner_sdk/server/routes/web.py` `/web/trainer` surface को वायर करता है। यहाँ केवल दो दिलचस्प endpoints हैं:

- `trainer_page` – outer chrome (navigation, config selector, tabs list) render करता है। यह `TabService` से metadata लेता है और सब कुछ `trainer_htmx.html` template में डाल देता है।
- `render_tab` – generic HTMX target। हर tab button इस endpoint को tab नाम के साथ hit करता है; route `TabService.render_tab` के जरिए matching layout resolve करता है और HTML chunk लौटाता है।

बाकी HTTP router set `simpletuner/simpletuner_sdk/server/routes/` के नीचे है और यही pattern फॉलो करता है: business logic service module में होता है, route params निकालता है, service को कॉल करता है, और परिणाम को JSON या HTML में बदलता है।

### TabService

`TabService` training form का central orchestrator है। यह परिभाषित करता है:

- हर tab के लिए static metadata (title, icon, template, optional context hook)
- `render_tab()` जो
  1. tab config (template, description) लेता है
  2. `FieldService` से tab/section का field bundle मांगता है
  3. कोई भी tab-specific context inject करता है (datasets list, GPU inventory, onboarding state)
  4. `form_tab.html`, `datasets_tab.html`, आदि का Jinja render लौटाता है

इस लॉजिक को class में रखने से हम HTMX, CLI wizard, और tests के लिए एक ही rendering reuse कर सकते हैं। अब template में global state तक पहुँच नहीं होती—सब कुछ context के जरिए स्पष्ट रूप से दिया जाता है।

### FieldService और FieldRegistry

`FieldService` registry entries को template-ready dictionaries में बदलता है। जिम्मेदारियाँ:

- platform/model context के आधार पर fields फ़िल्टर करना (जैसे MPS मशीनों पर CUDA-only knobs छिपाना)
- dependency rules (`FieldDependency`) का मूल्यांकन ताकि UI controls को disable या hide कर सके (उदाहरण के लिए Dynamo extras तब तक greyed out रहते हैं जब तक backend चुना न जाए)
- hints, dynamic choices, display formatting, और column classes के साथ fields को समृद्ध करना

यह fields के raw catalog के लिए `FieldRegistry` पर निर्भर करता है, जो `simpletuner/simpletuner_sdk/server/services/field_registry` के नीचे एक declarative listing है। हर `ConfigField` CLI flag नाम, validation rules, importance ordering, dependency metadata, और default UI copy का वर्णन करता है। यह व्यवस्था अन्य layers (CLI parser, API, documentation generator) को एक ही source of truth साझा करने देती है, जबकि वे अपने-अपने फॉर्मेट में प्रस्तुत करते हैं।

### State persistence और onboarding

WebUI `WebUIStateStore` के जरिए हल्के preferences स्टोर करता है। यह `$SIMPLETUNER_WEB_UI_CONFIG` (या XDG path) से defaults पढ़ता है और एक्सपोज़ करता है:

- theme, dataset root, output dir defaults
- हर feature के लिए onboarding checklist state
- cached Accelerate overrides (केवल whitelisted keys जैसे `--num_processes`, `--dynamo_backend`)

ये मान initial `/web/trainer` render के दौरान पेज में inject किए जाते हैं ताकि Alpine stores बिना अतिरिक्त round-trips के boot हो सकें।

### HTMX + Alpine interaction

हर settings panel बस HTML का एक chunk है जिसमें Alpine behavior के लिए `x-data` होता है। Tab buttons `/web/trainer/tabs/{tab}` पर HTMX GETs ट्रिगर करते हैं; सर्वर rendered form लौटाता है और Alpine मौजूदा component state बनाए रखता है। एक छोटा helper (`trainer-form.js`) saved value changes को replay करता है ताकि tabs बदलने पर यूज़र की in-progress edits न खोएँ।

Server updates (training status, GPU telemetry) SSE endpoints (`sse_manager.js`) के जरिए आते हैं और Alpine stores में डेटा डालते हैं जो toasts, progress bars, और system banners को चलाते हैं।

### File layout cheatsheet

- `templates/` – Jinja templates; `partials/form_field.html` व्यक्तिगत controls render करता है। `partials/form_field_htmx.html` HTMX-friendly variant है जो wizard को two-way binding देता है।
- `static/js/modules/` – Alpine component scripts (trainer form, hardware inventory, dataset browser)।
- `static/js/services/` – shared helpers (dependency evaluation, SSE manager, event bus)।
- `simpletuner/simpletuner_sdk/server/services/` – backend service layer (fields, tabs, configs, datasets, maintenance, events)।

यह सब मिलकर WebUI को server side पर stateless रखता है, और stateful bits (form data, toasts) ब्राउज़र में रहते हैं। backend सिर्फ pure data transforms तक सीमित रहता है, जिससे testing आसान होती है और trainer तथा web server एक ही process में चलने पर threading issues से बचा जा सकता है।
