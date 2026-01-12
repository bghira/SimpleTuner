# SimpleTuner UI थीम

## परिचय

SimpleTuner WebUI कस्टम थीम का समर्थन करता है जो रंग बदल सकते हैं, बैकग्राउंड इमेज जोड़ सकते हैं, और कस्टम साउंड शामिल कर सकते हैं। थीम pip पैकेज के माध्यम से इंस्टॉल किए जा सकते हैं या लोकल फोल्डर में रखे जा सकते हैं।

## बिल्ट-इन थीम

SimpleTuner में तीन बिल्ट-इन थीम शामिल हैं:

- **Dark** - बैंगनी एक्सेंट के साथ डिफ़ॉल्ट डार्क थीम
- **Tron** - एक प्रयोगात्मक नियॉन सियान थीम
- **Light** - Windows 98 प्रेरित बेज थीम

## थीम इंस्टॉल करना

### विधि 1: Pip पैकेज (अनुशंसित)

pip के साथ सीधे थीम पैकेज इंस्टॉल करें:

```bash
pip install simpletuner-theme-yourtheme
```

Pip-इंस्टॉल्ड थीम Python entry points के माध्यम से स्वचालित रूप से खोजे जाते हैं।

### विधि 2: लोकल थीम फोल्डर

थीम को `~/.simpletuner/themes/yourtheme/` में रखें:

```
~/.simpletuner/themes/yourtheme/
├── theme.json    # थीम मैनिफेस्ट (आवश्यक)
├── theme.css     # CSS ओवरराइड (आवश्यक)
└── assets/       # वैकल्पिक इमेज और साउंड
    ├── images/
    │   └── sidebar-bg.png
    └── sounds/
        └── success.wav
```

## थीम बदलना

1. WebUI में **UI सेटिंग्स** खोलें
2. **थीम** सेलेक्टर खोजें
3. नए इंस्टॉल किए गए थीम खोजने के लिए **रिफ्रेश** क्लिक करें
4. अपना वांछित थीम चुनें

## कस्टम थीम बनाना

### टेम्पलेट का उपयोग करना

कस्टम थीम बनाने का सबसे आसान तरीका आधिकारिक टेम्पलेट को fork करना है:

```bash
git clone https://github.com/bghira/simpletuner-theme-template
cd simpletuner-theme-template
```

### थीम संरचना

```
simpletuner-theme-template/
├── setup.py                 # entry_points के साथ पैकेज सेटअप
├── build_theme.py           # CSS जनरेट करने के लिए बिल्ड स्क्रिप्ट
└── src/
    └── simpletuner_theme_template/
        ├── __init__.py      # थीम क्लास
        ├── tokens.yaml      # डिज़ाइन टोकन (इसे एडिट करें!)
        ├── theme.css        # जनरेटेड CSS
        └── assets/          # वैकल्पिक थीम एसेट
            ├── images/
            └── sounds/
```

### डिज़ाइन टोकन एडिट करना

रंगों को कस्टमाइज़ करने के लिए `tokens.yaml` एडिट करें:

```yaml
colors:
  primary:
    purple: "#your-primary-color"
  accent:
    blue: "#your-accent-color"
  dark:
    bg: "#your-background-color"
  text:
    primary: "#your-text-color"
```

फिर CSS रीबिल्ड करें:

```bash
python build_theme.py
```

### कस्टम एसेट जोड़ना

#### इमेज

इमेज को `assets/images/` में रखें और अपनी थीम क्लास में डिक्लेयर करें:

```python
@classmethod
def get_assets(cls) -> Dict:
    return {
        "images": {
            "sidebar-bg": "assets/images/sidebar-bg.png",
            "logo": "assets/images/logo.svg",
        },
        "sounds": {},
    }
```

अपने CSS में इमेज रेफरेंस करें:

```css
.sidebar {
    background-image: url('/api/themes/mytheme/assets/images/sidebar-bg');
}
```

#### साउंड

साउंड को `assets/sounds/` में रखें और डिक्लेयर करें:

```python
@classmethod
def get_assets(cls) -> Dict:
    return {
        "images": {},
        "sounds": {
            "success": "assets/sounds/success.wav",
            "error": "assets/sounds/error.wav",
        },
    }
```

समर्थित फॉर्मेट:
- **इमेज**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.svg`, `.webp`, `.ico`
- **साउंड**: `.wav`, `.mp3`, `.ogg`, `.m4a`

### Entry Point रजिस्ट्रेशन

pip-इंस्टॉलेबल थीम के लिए, `setup.py` के माध्यम से रजिस्टर करें:

```python
setup(
    name="simpletuner-theme-mytheme",
    # ...
    entry_points={
        "simpletuner.themes": [
            "mytheme = my_theme_package:MyTheme",
        ],
    },
)
```

आपकी थीम क्लास में होना चाहिए:

```python
class MyTheme:
    id = "mytheme"
    name = "My Theme"
    description = "A custom theme"
    author = "Your Name"

    @classmethod
    def get_css_path(cls) -> Path:
        return Path(__file__).parent / "theme.css"

    @classmethod
    def get_assets(cls) -> Dict:
        return {"images": {}, "sounds": {}}
```

### लोकल थीम मैनिफेस्ट

लोकल थीम के लिए, `theme.json` बनाएं:

```json
{
    "id": "mytheme",
    "name": "My Theme",
    "description": "A custom theme",
    "author": "Your Name",
    "version": "1.0.0",
    "assets": {
        "images": {
            "sidebar-bg": "assets/images/sidebar-bg.png"
        },
        "sounds": {
            "success": "assets/sounds/success.wav"
        }
    }
}
```

## CSS वेरिएबल रेफरेंस

थीम CSS कस्टम प्रॉपर्टीज को ओवरराइड करते हैं। मुख्य वेरिएबल में शामिल हैं:

### रंग

```css
:root {
    /* प्राइमरी */
    --colors-primary-purple: #667eea;
    --colors-primary-gradient: linear-gradient(...);

    /* बैकग्राउंड */
    --colors-dark-bg: #0a0a0a;
    --colors-dark-sidebar: #0f0f0f;
    --colors-card-bg: rgba(255, 255, 255, 0.04);

    /* टेक्स्ट */
    --colors-text-primary: #ffffff;
    --colors-text-secondary: #94a3b8;

    /* सिमेंटिक */
    --colors-semantic-success: #22c55e;
    --colors-semantic-error: #ef4444;
}
```

पूरी सूची के लिए SimpleTuner सोर्स में `static/css/tokens.css` देखें।

## सुरक्षा

थीम एसेट सुरक्षा के लिए वैलिडेट किए जाते हैं:

- एसेट नाम अल्फान्यूमेरिक होने चाहिए (हाइफन/अंडरस्कोर के साथ)
- पाथ ट्रैवर्सल प्रयास (`../`) ब्लॉक किए जाते हैं
- केवल व्हाइटलिस्टेड फाइल एक्सटेंशन की अनुमति है
- एसेट थीम मैनिफेस्ट में डिक्लेयर होने चाहिए

## समस्या निवारण

### थीम दिखाई नहीं दे रहा

1. थीम सेलेक्टर में **रिफ्रेश** क्लिक करें
2. जांचें कि आपका थीम पैकेज इंस्टॉल है: `pip list | grep theme`
3. लोकल थीम के लिए, वेरिफाई करें कि `theme.json` वैलिड JSON है

### CSS लागू नहीं हो रहा

1. 404 एरर के लिए ब्राउज़र कंसोल चेक करें
2. वेरिफाई करें कि `theme.css` मौजूद है और वैलिड CSS है
3. सुनिश्चित करें कि CSS वेरिएबल `tokens.css` से सही नाम उपयोग करते हैं

### एसेट लोड नहीं हो रहे

1. वेरिफाई करें कि एसेट `get_assets()` या `theme.json` में डिक्लेयर हैं
2. जांचें कि फाइल एक्सटेंशन अनुमत सूची में हैं
3. सुनिश्चित करें कि एसेट फाइलें डिक्लेयर्ड पाथ पर मौजूद हैं
