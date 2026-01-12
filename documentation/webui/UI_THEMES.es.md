# Temas de UI de SimpleTuner

## Introducción

La WebUI de SimpleTuner soporta temas personalizados que pueden cambiar colores, agregar imágenes de fondo e incluir sonidos personalizados. Los temas se pueden instalar mediante paquetes pip o colocarse en una carpeta local.

## Temas Integrados

SimpleTuner incluye tres temas integrados:

- **Dark** - El tema oscuro predeterminado con acentos púrpura
- **Tron** - Un tema experimental en cian neón
- **Light** - Un tema beige inspirado en Windows 98

## Instalación de Temas

### Método 1: Paquete Pip (Recomendado)

Instala paquetes de temas directamente con pip:

```bash
pip install simpletuner-theme-tutema
```

Los temas instalados mediante pip se descubren automáticamente a través de entry points de Python.

### Método 2: Carpeta de Tema Local

Coloca los temas en `~/.simpletuner/themes/tutema/`:

```
~/.simpletuner/themes/tutema/
├── theme.json    # Manifiesto del tema (requerido)
├── theme.css     # Sobrescrituras CSS (requerido)
└── assets/       # Imágenes y sonidos opcionales
    ├── images/
    │   └── sidebar-bg.png
    └── sounds/
        └── success.wav
```

## Cambiar Temas

1. Abre **Configuración de UI** en la WebUI
2. Encuentra el selector de **Tema**
3. Haz clic en **Actualizar** para descubrir temas recién instalados
4. Selecciona el tema deseado

## Crear Temas Personalizados

### Usando la Plantilla

La forma más fácil de crear un tema personalizado es hacer fork de la plantilla oficial:

```bash
git clone https://github.com/bghira/simpletuner-theme-template
cd simpletuner-theme-template
```

### Estructura del Tema

```
simpletuner-theme-template/
├── setup.py                 # Configuración del paquete con entry_points
├── build_theme.py           # Script de construcción para generar CSS
└── src/
    └── simpletuner_theme_template/
        ├── __init__.py      # Clase del tema
        ├── tokens.yaml      # Tokens de diseño (¡edita esto!)
        ├── theme.css        # CSS generado
        └── assets/          # Assets opcionales del tema
            ├── images/
            └── sounds/
```

### Editar Tokens de Diseño

Edita `tokens.yaml` para personalizar colores:

```yaml
colors:
  primary:
    purple: "#tu-color-primario"
  accent:
    blue: "#tu-color-de-acento"
  dark:
    bg: "#tu-color-de-fondo"
  text:
    primary: "#tu-color-de-texto"
```

Luego reconstruye el CSS:

```bash
python build_theme.py
```

### Agregar Assets Personalizados

#### Imágenes

Coloca imágenes en `assets/images/` y decláralas en tu clase de tema:

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

Referencia imágenes en tu CSS:

```css
.sidebar {
    background-image: url('/api/themes/mitema/assets/images/sidebar-bg');
}
```

#### Sonidos

Coloca sonidos en `assets/sounds/` y decláralos:

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

Formatos soportados:
- **Imágenes**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.svg`, `.webp`, `.ico`
- **Sonidos**: `.wav`, `.mp3`, `.ogg`, `.m4a`

### Registro de Entry Point

Para temas instalables mediante pip, regístralos via `setup.py`:

```python
setup(
    name="simpletuner-theme-mitema",
    # ...
    entry_points={
        "simpletuner.themes": [
            "mitema = mi_paquete_tema:MiTema",
        ],
    },
)
```

Tu clase de tema debe tener:

```python
class MiTema:
    id = "mitema"
    name = "Mi Tema"
    description = "Un tema personalizado"
    author = "Tu Nombre"

    @classmethod
    def get_css_path(cls) -> Path:
        return Path(__file__).parent / "theme.css"

    @classmethod
    def get_assets(cls) -> Dict:
        return {"images": {}, "sounds": {}}
```

### Manifiesto de Tema Local

Para temas locales, crea `theme.json`:

```json
{
    "id": "mitema",
    "name": "Mi Tema",
    "description": "Un tema personalizado",
    "author": "Tu Nombre",
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

## Referencia de Variables CSS

Los temas sobrescriben propiedades personalizadas CSS. Las variables clave incluyen:

### Colores

```css
:root {
    /* Primario */
    --colors-primary-purple: #667eea;
    --colors-primary-gradient: linear-gradient(...);

    /* Fondos */
    --colors-dark-bg: #0a0a0a;
    --colors-dark-sidebar: #0f0f0f;
    --colors-card-bg: rgba(255, 255, 255, 0.04);

    /* Texto */
    --colors-text-primary: #ffffff;
    --colors-text-secondary: #94a3b8;

    /* Semántico */
    --colors-semantic-success: #22c55e;
    --colors-semantic-error: #ef4444;
}
```

Consulta `static/css/tokens.css` en el código fuente de SimpleTuner para la lista completa.

## Seguridad

Los assets de tema se validan por seguridad:

- Los nombres de assets deben ser alfanuméricos (con guiones/guiones bajos)
- Los intentos de path traversal (`../`) se bloquean
- Solo se permiten extensiones de archivo en la lista blanca
- Los assets deben declararse en el manifiesto del tema

## Solución de Problemas

### El tema no aparece

1. Haz clic en **Actualizar** en el selector de temas
2. Verifica que tu paquete de tema esté instalado: `pip list | grep theme`
3. Para temas locales, verifica que `theme.json` sea un JSON válido

### El CSS no se aplica

1. Revisa la consola del navegador por errores 404
2. Verifica que `theme.css` exista y sea CSS válido
3. Asegúrate de que las variables CSS usen los nombres correctos de `tokens.css`

### Los assets no cargan

1. Verifica que los assets estén declarados en `get_assets()` o `theme.json`
2. Verifica que las extensiones de archivo estén en la lista permitida
3. Asegúrate de que los archivos de assets existan en las rutas declaradas
