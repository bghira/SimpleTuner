"""Cache busting utilities for static assets.

Provides both server-side Jinja functions and client-side JavaScript utilities
for cache-busting CSS and JS files.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

# Module-level timestamp generated at import time
# This changes on server restart, providing session-based cache busting
_SESSION_TIMESTAMP = str(int(time.time()))


def get_cache_bust_timestamp() -> str:
    """Get the session-based cache bust timestamp.

    This timestamp is generated once per server session (at module import).
    It ensures all assets are refreshed when the server restarts.
    """
    return _SESSION_TIMESTAMP


def cache_bust_url(url: str, use_hash: bool = False, static_dir: Optional[Path] = None) -> str:
    """Add cache busting query parameter to a URL.

    Args:
        url: The asset URL (e.g., '/static/css/base.css')
        use_hash: If True, use file hash instead of timestamp (slower but more precise)
        static_dir: Directory containing static files (required if use_hash=True)

    Returns:
        URL with cache bust parameter (e.g., '/static/css/base.css?v=1703692800')
    """
    if use_hash and static_dir:
        # Extract relative path from URL
        if url.startswith("/static/"):
            rel_path = url[8:]  # Remove '/static/'
            file_path = static_dir / rel_path
            if file_path.exists():
                # Use first 8 chars of MD5 hash
                file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()[:8]
                return f"{url}?v={file_hash}"

    # Default: use session timestamp
    return f"{url}?v={_SESSION_TIMESTAMP}"


def setup_cache_busting(templates: "Jinja2Templates", static_dir: Optional[Path] = None) -> None:
    """Add cache busting functions to Jinja2 environment.

    This adds the following globals:
    - cache_bust_url(url): Returns URL with cache bust parameter
    - cache_timestamp: The session timestamp string

    Args:
        templates: Jinja2Templates instance to configure
        static_dir: Optional path to static files directory (enables hash-based busting)
    """
    env = templates.env

    # Add the cache busting function
    env.globals["cache_bust_url"] = lambda url, use_hash=False: cache_bust_url(url, use_hash, static_dir)
    env.globals["cache_timestamp"] = _SESSION_TIMESTAMP


# Client-side JavaScript for dynamic cache busting
CACHE_BUST_SCRIPT = f"""
<script>
(function() {{
    var cacheTimestamp = '{_SESSION_TIMESTAMP}';

    /**
     * Apply cache busting to a stylesheet link element.
     * @param {{string}} elementId - The ID of the link element
     * @param {{string}} cssPath - The CSS file path (without query string)
     */
    window.cacheBustCSS = function(elementId, cssPath) {{
        var el = document.getElementById(elementId);
        if (el) {{
            el.href = cssPath + '?v=' + cacheTimestamp;
        }}
    }};

    /**
     * Apply cache busting to multiple stylesheets at once.
     * @param {{Array<[string, string]>}} items - Array of [elementId, cssPath] pairs
     */
    window.cacheBustMultipleCSS = function(items) {{
        items.forEach(function(item) {{
            window.cacheBustCSS(item[0], item[1]);
        }});
    }};

    // Make timestamp available globally for other uses
    window.CACHE_TIMESTAMP = cacheTimestamp;
}})();
</script>
"""


def get_cache_bust_script() -> str:
    """Get the cache busting JavaScript snippet.

    This can be included once in the base template's <head>.
    """
    return CACHE_BUST_SCRIPT
