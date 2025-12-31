"""Credential encryption utilities.

Uses Fernet symmetric encryption to protect provider API tokens at rest.
The encryption key is derived from a master secret that should be:
- Set via SIMPLETUNER_CREDENTIAL_KEY environment variable
- Or auto-generated and stored in ~/.simpletuner/credential.key
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded encryption components
_fernet: Optional["Fernet"] = None  # noqa: F821
_key_path: Optional[Path] = None


def _get_key_path() -> Path:
    """Get the path for storing the auto-generated key."""
    global _key_path
    if _key_path is None:
        _key_path = Path.home() / ".simpletuner" / "credential.key"
    return _key_path


def _derive_key(secret: str) -> bytes:
    """Derive a 32-byte key from an arbitrary secret string.

    Uses PBKDF2 with a fixed salt (the secret is the entropy source).
    """
    # Use a fixed application salt - the secret provides the entropy
    app_salt = b"simpletuner-credential-encryption-v1"
    key = hashlib.pbkdf2_hmac(
        "sha256",
        secret.encode("utf-8"),
        app_salt,
        iterations=100_000,
        dklen=32,
    )
    return base64.urlsafe_b64encode(key)


def _load_or_create_key() -> bytes:
    """Load or create the encryption key.

    Priority:
    1. SIMPLETUNER_CREDENTIAL_KEY environment variable
    2. Existing key file at ~/.simpletuner/credential.key
    3. Generate new key and save to file
    """
    # Check environment variable first
    env_key = os.environ.get("SIMPLETUNER_CREDENTIAL_KEY")
    if env_key:
        logger.debug("Using credential key from environment variable")
        return _derive_key(env_key)

    key_path = _get_key_path()

    # Try to load existing key
    if key_path.exists():
        try:
            key_data = key_path.read_text().strip()
            logger.debug("Loaded credential key from %s", key_path)
            return _derive_key(key_data)
        except Exception as exc:
            logger.warning("Failed to read credential key: %s", exc)

    # Generate new key
    logger.info("Generating new credential encryption key")
    new_secret = secrets.token_urlsafe(32)

    try:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(new_secret)
        key_path.chmod(0o600)  # Owner read/write only
        logger.info("Saved credential key to %s", key_path)
    except Exception as exc:
        logger.warning("Could not save credential key: %s", exc)

    return _derive_key(new_secret)


def _get_fernet() -> "Fernet":  # noqa: F821
    """Get or create the Fernet cipher instance."""
    global _fernet
    if _fernet is None:
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            raise ImportError(
                "The 'cryptography' package is required for credential encryption. "
                "Install it with: pip install cryptography"
            )

        key = _load_or_create_key()
        _fernet = Fernet(key)

    return _fernet


def encrypt_credential(plaintext: str) -> str:
    """Encrypt a credential string.

    Args:
        plaintext: The credential value to encrypt

    Returns:
        Base64-encoded encrypted value
    """
    fernet = _get_fernet()
    encrypted = fernet.encrypt(plaintext.encode("utf-8"))
    return base64.urlsafe_b64encode(encrypted).decode("ascii")


def decrypt_credential(ciphertext: str) -> str:
    """Decrypt a credential string.

    Args:
        ciphertext: Base64-encoded encrypted value

    Returns:
        Decrypted plaintext value

    Raises:
        ValueError: If decryption fails (wrong key, corrupted data)
    """
    fernet = _get_fernet()
    try:
        encrypted = base64.urlsafe_b64decode(ciphertext.encode("ascii"))
        decrypted = fernet.decrypt(encrypted)
        return decrypted.decode("utf-8")
    except Exception as exc:
        raise ValueError(f"Failed to decrypt credential: {exc}")


def reset_cipher() -> None:
    """Reset the cipher instance (for testing)."""
    global _fernet, _key_path
    _fernet = None
    _key_path = None
