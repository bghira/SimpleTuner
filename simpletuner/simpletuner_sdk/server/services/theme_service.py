"""Theme discovery and management service.

Discovers themes from:
1. Built-in themes (dark, tron)
2. Pip-installed packages with 'simpletuner.themes' entry points
3. Local themes in ~/.simpletuner/themes/
"""

import json
import logging
import re
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

# Allowed file extensions for theme assets
ALLOWED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico"})
ALLOWED_SOUND_EXTENSIONS = frozenset({".wav", ".mp3", ".ogg", ".m4a"})
ALLOWED_ASSET_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_SOUND_EXTENSIONS

# Pattern for valid asset names (alphanumeric, hyphens, underscores only)
VALID_ASSET_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


@dataclass
class ThemeAssets:
    """Theme asset declarations with paths relative to theme directory."""

    images: Dict[str, str] = field(default_factory=dict)
    sounds: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses."""
        return {"images": dict(self.images), "sounds": dict(self.sounds)}


@dataclass
class ThemeMetadata:
    """Theme metadata for UI display and CSS loading."""

    id: str
    name: str
    description: str
    author: str
    css_path: Optional[Path]
    source: str  # "builtin", "pip", "local"
    theme_dir: Optional[Path] = None  # Directory containing theme files
    assets: ThemeAssets = field(default_factory=ThemeAssets)


class ThemeSource(Protocol):
    """Protocol for theme discovery sources."""

    def discover(self) -> Dict[str, ThemeMetadata]:
        """Discover themes from this source."""
        ...


class EntryPointThemeSource:
    """Discovers themes registered via pip entry_points."""

    GROUP = "simpletuner.themes"

    def discover(self) -> Dict[str, ThemeMetadata]:
        """Discover themes from installed packages."""
        themes = {}
        try:
            eps = entry_points(group=self.GROUP)
        except TypeError:
            # Python 3.9 compatibility
            eps = entry_points().get(self.GROUP, [])

        for ep in eps:
            try:
                theme_class = ep.load()
                css_path = None
                theme_dir = None

                if hasattr(theme_class, "get_css_path"):
                    css_path = theme_class.get_css_path()
                    if css_path and not isinstance(css_path, Path):
                        css_path = Path(css_path)
                    if css_path:
                        theme_dir = css_path.parent

                # Parse assets from theme class if available
                assets = ThemeAssets()
                if hasattr(theme_class, "get_assets"):
                    try:
                        assets_data = theme_class.get_assets()
                        if isinstance(assets_data, dict):
                            assets = ThemeAssets(
                                images=assets_data.get("images", {}),
                                sounds=assets_data.get("sounds", {}),
                            )
                    except Exception as e:
                        logger.warning(f"Failed to load assets for theme '{ep.name}': {e}")

                themes[ep.name] = ThemeMetadata(
                    id=ep.name,
                    name=getattr(theme_class, "name", ep.name.replace("_", " ").title()),
                    description=getattr(theme_class, "description", ""),
                    author=getattr(theme_class, "author", "Unknown"),
                    css_path=css_path,
                    source="pip",
                    theme_dir=theme_dir,
                    assets=assets,
                )
            except Exception as e:
                logger.warning(f"Failed to load theme '{ep.name}': {e}")

        return themes


class LocalFolderThemeSource:
    """Discovers themes in ~/.simpletuner/themes/*/theme.json."""

    def __init__(self, themes_dir: Optional[Path] = None):
        self.themes_dir = themes_dir or Path.home() / ".simpletuner" / "themes"

    def discover(self) -> Dict[str, ThemeMetadata]:
        """Discover themes from local folder."""
        themes = {}
        if not self.themes_dir.exists():
            return themes

        for theme_dir in self.themes_dir.iterdir():
            if not theme_dir.is_dir():
                continue
            manifest = theme_dir / "theme.json"
            css_file = theme_dir / "theme.css"
            if not manifest.exists():
                continue
            if not css_file.exists():
                logger.warning(f"Theme '{theme_dir.name}' has manifest but no theme.css")
                continue

            try:
                data = json.loads(manifest.read_text())
                theme_id = data.get("id", theme_dir.name)

                # Parse assets from manifest
                assets = ThemeAssets()
                if "assets" in data and isinstance(data["assets"], dict):
                    assets_data = data["assets"]
                    assets = ThemeAssets(
                        images=assets_data.get("images", {}),
                        sounds=assets_data.get("sounds", {}),
                    )

                themes[theme_id] = ThemeMetadata(
                    id=theme_id,
                    name=data.get("name", theme_id.replace("_", " ").title()),
                    description=data.get("description", ""),
                    author=data.get("author", "Unknown"),
                    css_path=css_file,
                    source="local",
                    theme_dir=theme_dir,
                    assets=assets,
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid theme.json in '{theme_dir.name}': {e}")
            except Exception as e:
                logger.warning(f"Failed to load local theme '{theme_dir.name}': {e}")

        return themes


class ThemeService:
    """Aggregates themes from all sources."""

    _instance: Optional["ThemeService"] = None

    BUILTIN_THEMES = {
        "dark": ThemeMetadata(
            id="dark",
            name="Dark",
            description="Classic SimpleTuner palette",
            author="SimpleTuner",
            css_path=None,
            source="builtin",
        ),
        "tron": ThemeMetadata(
            id="tron",
            name="Tron",
            description="Experimental neon styling",
            author="SimpleTuner",
            css_path=None,
            source="builtin",
        ),
    }

    def __init__(self):
        self._sources: List[ThemeSource] = [
            EntryPointThemeSource(),
            LocalFolderThemeSource(),
        ]
        self._cache: Optional[Dict[str, ThemeMetadata]] = None

    @classmethod
    def get_instance(cls) -> "ThemeService":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def discover_themes(self) -> Dict[str, ThemeMetadata]:
        """Discover all themes from all sources.

        Returns cached results if available. Call invalidate_cache() to refresh.
        """
        if self._cache is not None:
            return self._cache

        themes = dict(self.BUILTIN_THEMES)
        for source in self._sources:
            try:
                discovered = source.discover()
                themes.update(discovered)
            except Exception as e:
                logger.warning(f"Theme source {type(source).__name__} failed: {e}")

        self._cache = themes
        return themes

    def invalidate_cache(self) -> None:
        """Clear the theme cache, forcing rediscovery on next access."""
        self._cache = None

    def get_theme(self, theme_id: str) -> Optional[ThemeMetadata]:
        """Get a specific theme by ID."""
        return self.discover_themes().get(theme_id)

    def is_valid_theme(self, theme_id: str) -> bool:
        """Check if a theme ID is valid."""
        return theme_id in self.discover_themes()

    def list_for_ui(self) -> List[Dict]:
        """Return theme list formatted for UI settings dropdown."""
        themes = self.discover_themes()
        return [
            {
                "value": meta.id,
                "label": meta.name,
                "description": meta.description,
                "source": meta.source,
                "hasAssets": bool(meta.assets.images or meta.assets.sounds),
            }
            for meta in themes.values()
        ]

    def get_theme_css_path(self, theme_id: str) -> Optional[Path]:
        """Get the CSS file path for a theme."""
        theme = self.get_theme(theme_id)
        if theme is None:
            return None
        return theme.css_path

    def get_theme_manifest(self, theme_id: str) -> Optional[Dict]:
        """Get theme manifest including asset URLs for JavaScript consumption."""
        theme = self.get_theme(theme_id)
        if theme is None:
            return None

        # Build asset URLs
        asset_urls = {"images": {}, "sounds": {}}
        for asset_name in theme.assets.images:
            asset_urls["images"][asset_name] = f"/api/themes/{theme_id}/assets/images/{asset_name}"
        for asset_name in theme.assets.sounds:
            asset_urls["sounds"][asset_name] = f"/api/themes/{theme_id}/assets/sounds/{asset_name}"

        return {
            "id": theme.id,
            "name": theme.name,
            "description": theme.description,
            "author": theme.author,
            "source": theme.source,
            "assets": asset_urls,
        }

    def validate_asset_name(self, asset_name: str) -> bool:
        """Validate asset name for security.

        Asset names must be alphanumeric with hyphens and underscores only.
        No path separators, dots (except extension), or special characters.
        """
        if not asset_name:
            return False

        # Must match safe pattern
        if not VALID_ASSET_NAME_PATTERN.match(asset_name):
            return False

        return True

    def get_asset_path(self, theme_id: str, asset_type: str, asset_name: str) -> Optional[Path]:
        """Get the filesystem path for a theme asset with security validation.

        Args:
            theme_id: The theme identifier
            asset_type: Either "images" or "sounds"
            asset_name: The asset name (without extension)

        Returns:
            Resolved Path if valid and exists, None otherwise

        Security:
            - Validates asset_name against safe pattern
            - Ensures resolved path is within theme directory
            - Validates file extension against whitelist
            - Checks asset is declared in theme manifest
        """
        # Validate asset type
        if asset_type not in ("images", "sounds"):
            logger.warning(f"Invalid asset type requested: {asset_type}")
            return None

        # Validate asset name format
        if not self.validate_asset_name(asset_name):
            logger.warning(f"Invalid asset name format: {asset_name}")
            return None

        # Get theme
        theme = self.get_theme(theme_id)
        if theme is None:
            logger.warning(f"Theme not found: {theme_id}")
            return None

        # Theme must have a directory for assets
        if theme.theme_dir is None:
            logger.debug(f"Theme '{theme_id}' has no theme_dir (builtin theme)")
            return None

        # Get declared assets for this type
        declared_assets = theme.assets.images if asset_type == "images" else theme.assets.sounds
        if asset_name not in declared_assets:
            logger.warning(f"Asset '{asset_name}' not declared in theme '{theme_id}' manifest")
            return None

        # Get the relative path from manifest
        relative_path = declared_assets[asset_name]

        # Security: validate relative path has no traversal
        if ".." in relative_path or relative_path.startswith("/"):
            logger.warning(f"Path traversal attempt in theme '{theme_id}': {relative_path}")
            return None

        # Resolve the full path
        try:
            asset_path = (theme.theme_dir / relative_path).resolve()
        except (ValueError, OSError) as e:
            logger.warning(f"Failed to resolve asset path: {e}")
            return None

        # Security: ensure resolved path is within theme directory
        theme_dir_resolved = theme.theme_dir.resolve()
        try:
            asset_path.relative_to(theme_dir_resolved)
        except ValueError:
            logger.warning(f"Asset path escapes theme directory: {asset_path} not in {theme_dir_resolved}")
            return None

        # Validate file extension
        allowed_extensions = ALLOWED_IMAGE_EXTENSIONS if asset_type == "images" else ALLOWED_SOUND_EXTENSIONS
        if asset_path.suffix.lower() not in allowed_extensions:
            logger.warning(f"Invalid file extension for {asset_type}: {asset_path.suffix}")
            return None

        # Check file exists
        if not asset_path.exists() or not asset_path.is_file():
            logger.warning(f"Asset file not found: {asset_path}")
            return None

        return asset_path

    def list_theme_assets(self, theme_id: str) -> Optional[Dict]:
        """List all assets declared by a theme.

        Returns dict with 'images' and 'sounds' keys mapping asset names to URLs.
        """
        theme = self.get_theme(theme_id)
        if theme is None:
            return None

        return {
            "images": list(theme.assets.images.keys()),
            "sounds": list(theme.assets.sounds.keys()),
        }
