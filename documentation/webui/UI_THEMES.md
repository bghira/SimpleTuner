# SimpleTuner UI Themes

## Introduction

SimpleTuner WebUI supports custom themes that can change colors, add background images, and include custom sounds. Themes can be installed via pip packages or placed in a local folder.

## Built-in Themes

SimpleTuner includes two built-in themes:

- **Dark** - The default dark theme with purple accents
- **Tron** - An experimental neon cyan theme

## Installing Themes

### Method 1: Pip Package (Recommended)

Install theme packages directly with pip:

```bash
pip install simpletuner-theme-yourtheme
```

Pip-installed themes are automatically discovered via Python entry points.

### Method 2: Local Theme Folder

Place themes in `~/.simpletuner/themes/yourtheme/`:

```
~/.simpletuner/themes/yourtheme/
├── theme.json    # Theme manifest (required)
├── theme.css     # CSS overrides (required)
└── assets/       # Optional images and sounds
    ├── images/
    │   └── sidebar-bg.png
    └── sounds/
        └── success.wav
```

## Switching Themes

1. Open **UI Settings** in the WebUI
2. Find the **Theme** selector
3. Click **Refresh** to discover newly installed themes
4. Select your desired theme

## Creating Custom Themes

### Using the Template

The easiest way to create a custom theme is to fork the official template:

```bash
git clone https://github.com/bghira/simpletuner-theme-template
cd simpletuner-theme-template
```

### Theme Structure

```
simpletuner-theme-template/
├── setup.py                 # Package setup with entry_points
├── build_theme.py           # Build script for generating CSS
└── src/
    └── simpletuner_theme_template/
        ├── __init__.py      # Theme class
        ├── tokens.yaml      # Design tokens (edit this!)
        ├── theme.css        # Generated CSS
        └── assets/          # Optional theme assets
            ├── images/
            └── sounds/
```

### Editing Design Tokens

Edit `tokens.yaml` to customize colors:

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

Then rebuild the CSS:

```bash
python build_theme.py
```

### Adding Custom Assets

#### Images

Place images in `assets/images/` and declare them in your theme class:

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

Reference images in your CSS:

```css
.sidebar {
    background-image: url('/api/themes/mytheme/assets/images/sidebar-bg');
}
```

#### Sounds

Place sounds in `assets/sounds/` and declare them:

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

Supported formats:
- **Images**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.svg`, `.webp`, `.ico`
- **Sounds**: `.wav`, `.mp3`, `.ogg`, `.m4a`

### Entry Point Registration

For pip-installable themes, register via `setup.py`:

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

Your theme class must have:

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

### Local Theme Manifest

For local themes, create `theme.json`:

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

## CSS Variable Reference

Themes override CSS custom properties. Key variables include:

### Colors

```css
:root {
    /* Primary */
    --colors-primary-purple: #667eea;
    --colors-primary-gradient: linear-gradient(...);

    /* Backgrounds */
    --colors-dark-bg: #0a0a0a;
    --colors-dark-sidebar: #0f0f0f;
    --colors-card-bg: rgba(255, 255, 255, 0.04);

    /* Text */
    --colors-text-primary: #ffffff;
    --colors-text-secondary: #94a3b8;

    /* Semantic */
    --colors-semantic-success: #22c55e;
    --colors-semantic-error: #ef4444;
}
```

See `static/css/tokens.css` in the SimpleTuner source for the complete list.

## Security

Theme assets are validated for security:

- Asset names must be alphanumeric (with hyphens/underscores)
- Path traversal attempts (`../`) are blocked
- Only whitelisted file extensions are allowed
- Assets must be declared in the theme manifest

## Troubleshooting

### Theme not appearing

1. Click **Refresh** in the theme selector
2. Check that your theme package is installed: `pip list | grep theme`
3. For local themes, verify `theme.json` is valid JSON

### CSS not applying

1. Check browser console for 404 errors
2. Verify `theme.css` exists and is valid CSS
3. Ensure CSS variables use the correct names from `tokens.css`

### Assets not loading

1. Verify assets are declared in `get_assets()` or `theme.json`
2. Check file extensions are in the allowed list
3. Ensure asset files exist at the declared paths
