"""Theme discovery and management service.

Discovers themes from:
1. Built-in themes (dark, tron)
2. Pip-installed packages with 'simpletuner.themes' entry points
3. Local themes in ~/.simpletuner/themes/
"""

import json
import logging
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ThemeMetadata:
    """Theme metadata for UI display and CSS loading."""

    id: str
    name: str
    description: str
    author: str
    css_path: Optional[Path]
    source: str  # "builtin", "pip", "local"


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
                if hasattr(theme_class, "get_css_path"):
                    css_path = theme_class.get_css_path()
                    if css_path and not isinstance(css_path, Path):
                        css_path = Path(css_path)

                themes[ep.name] = ThemeMetadata(
                    id=ep.name,
                    name=getattr(theme_class, "name", ep.name.replace("_", " ").title()),
                    description=getattr(theme_class, "description", ""),
                    author=getattr(theme_class, "author", "Unknown"),
                    css_path=css_path,
                    source="pip",
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
                themes[theme_id] = ThemeMetadata(
                    id=theme_id,
                    name=data.get("name", theme_id.replace("_", " ").title()),
                    description=data.get("description", ""),
                    author=data.get("author", "Unknown"),
                    css_path=css_file,
                    source="local",
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
            }
            for meta in themes.values()
        ]

    def get_theme_css_path(self, theme_id: str) -> Optional[Path]:
        """Get the CSS file path for a theme."""
        theme = self.get_theme(theme_id)
        if theme is None:
            return None
        return theme.css_path
