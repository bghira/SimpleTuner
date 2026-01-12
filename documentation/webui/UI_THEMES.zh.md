# SimpleTuner UI 主题

## 简介

SimpleTuner WebUI 支持自定义主题，可以更改颜色、添加背景图片和自定义声音。主题可以通过 pip 包安装或放置在本地文件夹中。

## 内置主题

SimpleTuner 包含两个内置主题：

- **Dark** - 默认深色主题，带有紫色强调色
- **Tron** - 实验性霓虹青色主题

## 安装主题

### 方法 1：Pip 包（推荐）

直接使用 pip 安装主题包：

```bash
pip install simpletuner-theme-yourtheme
```

通过 pip 安装的主题会通过 Python 入口点自动发现。

### 方法 2：本地主题文件夹

将主题放置在 `~/.simpletuner/themes/yourtheme/`：

```
~/.simpletuner/themes/yourtheme/
├── theme.json    # 主题清单（必需）
├── theme.css     # CSS 覆盖（必需）
└── assets/       # 可选的图片和声音
    ├── images/
    │   └── sidebar-bg.png
    └── sounds/
        └── success.wav
```

## 切换主题

1. 在 WebUI 中打开 **UI 设置**
2. 找到 **主题** 选择器
3. 点击 **刷新** 以发现新安装的主题
4. 选择您想要的主题

## 创建自定义主题

### 使用模板

创建自定义主题最简单的方法是 fork 官方模板：

```bash
git clone https://github.com/bghira/simpletuner-theme-template
cd simpletuner-theme-template
```

### 主题结构

```
simpletuner-theme-template/
├── setup.py                 # 带有 entry_points 的包设置
├── build_theme.py           # 用于生成 CSS 的构建脚本
└── src/
    └── simpletuner_theme_template/
        ├── __init__.py      # 主题类
        ├── tokens.yaml      # 设计令牌（编辑这个！）
        ├── theme.css        # 生成的 CSS
        └── assets/          # 可选的主题资源
            ├── images/
            └── sounds/
```

### 编辑设计令牌

编辑 `tokens.yaml` 以自定义颜色：

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

然后重新构建 CSS：

```bash
python build_theme.py
```

### 添加自定义资源

#### 图片

将图片放在 `assets/images/` 并在主题类中声明：

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

在 CSS 中引用图片：

```css
.sidebar {
    background-image: url('/api/themes/mytheme/assets/images/sidebar-bg');
}
```

#### 声音

将声音放在 `assets/sounds/` 并声明：

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

支持的格式：
- **图片**：`.png`、`.jpg`、`.jpeg`、`.gif`、`.svg`、`.webp`、`.ico`
- **声音**：`.wav`、`.mp3`、`.ogg`、`.m4a`

### 入口点注册

对于可通过 pip 安装的主题，通过 `setup.py` 注册：

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

您的主题类必须包含：

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

### 本地主题清单

对于本地主题，创建 `theme.json`：

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

## CSS 变量参考

主题覆盖 CSS 自定义属性。关键变量包括：

### 颜色

```css
:root {
    /* 主要 */
    --colors-primary-purple: #667eea;
    --colors-primary-gradient: linear-gradient(...);

    /* 背景 */
    --colors-dark-bg: #0a0a0a;
    --colors-dark-sidebar: #0f0f0f;
    --colors-card-bg: rgba(255, 255, 255, 0.04);

    /* 文本 */
    --colors-text-primary: #ffffff;
    --colors-text-secondary: #94a3b8;

    /* 语义 */
    --colors-semantic-success: #22c55e;
    --colors-semantic-error: #ef4444;
}
```

完整列表请参见 SimpleTuner 源代码中的 `static/css/tokens.css`。

## 安全性

主题资源经过安全验证：

- 资源名称必须是字母数字（可带连字符/下划线）
- 路径遍历尝试（`../`）会被阻止
- 只允许白名单中的文件扩展名
- 资源必须在主题清单中声明

## 故障排除

### 主题未显示

1. 在主题选择器中点击 **刷新**
2. 检查您的主题包是否已安装：`pip list | grep theme`
3. 对于本地主题，验证 `theme.json` 是有效的 JSON

### CSS 未应用

1. 检查浏览器控制台是否有 404 错误
2. 验证 `theme.css` 存在且是有效的 CSS
3. 确保 CSS 变量使用 `tokens.css` 中的正确名称

### 资源未加载

1. 验证资源已在 `get_assets()` 或 `theme.json` 中声明
2. 检查文件扩展名是否在允许列表中
3. 确保资源文件存在于声明的路径
