"""Discovery of bundled training configurations, excluding shared assets."""

from pathlib import Path

CONFIG_FILENAMES = ("config.json", "config.toml", "config.env")


def discover_examples(root: Path) -> dict[str, Path]:
    """Return example names relative to the catalog root and their config files."""
    examples = {}
    for filename in CONFIG_FILENAMES:
        for config in root.rglob(filename):
            relative = config.relative_to(root)
            if any(part.startswith((".", "_")) for part in relative.parts):
                continue
            examples.setdefault(relative.parent.as_posix(), config)
    return dict(sorted(examples.items()))
