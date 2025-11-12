import os
from urllib.parse import urlparse


def _is_uri(path: str) -> bool:
    if not isinstance(path, str):
        return False
    parsed = urlparse(path)
    return bool(parsed.scheme and parsed.netloc)


def normalize_data_path(path: str, root: str | None = None) -> str:
    """Return a stable identifier for a dataset path, relative to root when possible."""
    if not isinstance(path, str):
        return ""

    if _is_uri(path):
        return path

    candidate_path = path
    if root and not os.path.isabs(candidate_path) and not _is_uri(root):
        candidate_path = os.path.join(root, candidate_path)

    abs_candidate = os.path.abspath(os.path.normpath(candidate_path))
    if root and not _is_uri(root):
        abs_root = os.path.abspath(os.path.normpath(root))
        try:
            rel_path = os.path.relpath(abs_candidate, abs_root)
            return os.path.normcase(rel_path)
        except ValueError:
            pass

    return os.path.normcase(abs_candidate)
