"""Register NiceGUI pages by importing modules with @ui.page decorators."""
from __future__ import annotations

# Import order is not critical; each module registers its routes via decorators.
from ..theme import ensure_legacy_theme

ensure_legacy_theme()

from . import index  # noqa: F401,E402
from . import datasets  # noqa: F401,E402
from . import trainer_overview  # noqa: F401,E402
from . import training  # noqa: F401,E402
from . import settings  # noqa: F401,E402
