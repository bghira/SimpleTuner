"""Security utilities for SimpleTuner WebUI.

This module provides security-related utilities including:
- Secure path validation
- Input sanitization
- Security helpers
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Optional, List, Union
from urllib.parse import unquote

logger = logging.getLogger(__name__)


def validate_safe_path(
    path: str,
    allowed_dirs: Optional[List[Union[str, Path]]] = None,
    allow_symlinks: bool = False
) -> Optional[Path]:
    """Validate that a path is safe and within allowed directories.

    This function performs comprehensive path validation to prevent
    directory traversal attacks and other path-based vulnerabilities.

    Args:
        path: The path to validate
        allowed_dirs: List of allowed base directories. If None, uses current directory
        allow_symlinks: Whether to allow symlinks (default: False)

    Returns:
        The resolved Path object if valid, None if invalid

    Security checks performed:
        - Null byte detection
        - Path traversal patterns (.., ..\\, etc.)
        - URL encoding tricks
        - Absolute path escape attempts
        - Symlink validation
        - Path normalization
    """
    if not path:
        logger.warning("Empty path provided")
        return None

    # Decode any URL encoding
    try:
        decoded_path = unquote(path, errors='strict')
    except Exception as e:
        logger.warning(f"Failed to decode path: {e}")
        return None

    # Check for null bytes
    if '\x00' in decoded_path:
        logger.warning("Null byte detected in path")
        return None

    # Check for various path traversal patterns
    traversal_patterns = [
        '..',  # Basic traversal
        '..\\',  # Windows traversal
        '../',  # Unix traversal
        '..%2F',  # URL encoded forward slash
        '..%5C',  # URL encoded backslash
        '..%252F',  # Double encoded forward slash
        '..%255C',  # Double encoded backslash
        '\\..\\',  # Windows UNC paths
        '/../',  # Unix absolute traversal
    ]

    for pattern in traversal_patterns:
        if pattern in decoded_path:
            logger.warning(f"Path traversal pattern detected: {pattern}")
            return None

    # Additional regex patterns for complex traversal attempts
    dangerous_patterns = [
        r'\.\.+[/\\]',  # Multiple dots
        r'[/\\]\.\.+',  # Dots after separator
        r'\.\.[/\\]?\.\.',  # Chained traversals
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, decoded_path):
            logger.warning(f"Dangerous path pattern detected: {pattern}")
            return None

    try:
        # Create Path object and resolve it
        path_obj = Path(decoded_path)

        # If the path is absolute, ensure it's within allowed directories
        if path_obj.is_absolute():
            logger.warning("Absolute paths are not allowed")
            return None

        # Resolve the path (follows symlinks and normalizes)
        resolved_path = path_obj.resolve()

        # Get allowed directories
        if allowed_dirs is None:
            allowed_dirs = [Path.cwd()]
        else:
            allowed_dirs = [Path(d).resolve() for d in allowed_dirs]

        # Check if resolved path is within any allowed directory
        is_allowed = False
        for allowed_dir in allowed_dirs:
            try:
                # Check if the path is relative to the allowed directory
                resolved_path.relative_to(allowed_dir)
                is_allowed = True
                break
            except ValueError:
                # Path is not relative to this directory
                continue

        if not is_allowed:
            logger.warning(f"Path '{resolved_path}' is not within allowed directories")
            return None

        # Check for symlinks if not allowed
        if not allow_symlinks and path_obj.exists() and path_obj.is_symlink():
            logger.warning(f"Symlink detected: {path_obj}")
            return None

        # Additional check: ensure the resolved path still exists or can be created
        # within the allowed directory (prevents race conditions)
        parent = resolved_path.parent
        if not parent.exists():
            logger.warning(f"Parent directory does not exist: {parent}")
            return None

        return resolved_path

    except Exception as e:
        logger.error(f"Error validating path: {e}")
        return None


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize a filename to be safe for filesystem use.

    Args:
        filename: The filename to sanitize
        max_length: Maximum length for the filename (default: 255)

    Returns:
        Sanitized filename
    """
    if not filename:
        return "unnamed"

    # Remove any directory separators
    filename = os.path.basename(filename)

    # Remove null bytes
    filename = filename.replace('\x00', '')

    # Replace problematic characters
    # Allow alphanumeric, dash, underscore, and dot
    filename = re.sub(r'[^\w\-_.]', '_', filename)

    # Remove leading dots (hidden files on Unix)
    filename = filename.lstrip('.')

    # Ensure it's not empty after sanitization
    if not filename:
        filename = "unnamed"

    # Truncate to max length
    if len(filename) > max_length:
        # Preserve extension if possible
        name, ext = os.path.splitext(filename)
        if ext:
            max_name_length = max_length - len(ext)
            filename = name[:max_name_length] + ext
        else:
            filename = filename[:max_length]

    return filename


def is_safe_url(url: str, allowed_hosts: Optional[List[str]] = None) -> bool:
    """Check if a URL is safe for redirection.

    Args:
        url: The URL to check
        allowed_hosts: List of allowed hostnames

    Returns:
        True if the URL is safe, False otherwise
    """
    from urllib.parse import urlparse

    if not url:
        return False

    try:
        parsed = urlparse(url)

        # Reject URLs with potentially dangerous schemes
        dangerous_schemes = ['javascript', 'data', 'vbscript', 'file']
        if parsed.scheme in dangerous_schemes:
            return False

        # If no allowed hosts specified, only allow relative URLs
        if allowed_hosts is None:
            return not parsed.netloc

        # Check if the host is in the allowed list
        return parsed.netloc in allowed_hosts

    except Exception:
        return False