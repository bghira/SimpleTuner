"""Password hashing utilities.

Provides secure password hashing using argon2 (preferred) or bcrypt (fallback).
Also handles API key hashing and generation.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import string
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import argon2, fall back to bcrypt, then hashlib
_hasher_type: Optional[str] = None


def _get_hasher_type() -> str:
    """Determine which password hasher to use."""
    global _hasher_type
    if _hasher_type is not None:
        return _hasher_type

    try:
        import argon2

        _hasher_type = "argon2"
        return _hasher_type
    except ImportError:
        pass

    try:
        import bcrypt

        _hasher_type = "bcrypt"
        return _hasher_type
    except ImportError:
        pass

    # Fallback to PBKDF2 via hashlib (always available)
    _hasher_type = "pbkdf2"
    logger.warning(
        "Neither argon2-cffi nor bcrypt installed. Using PBKDF2 for password hashing. "
        "For better security, install argon2-cffi: pip install argon2-cffi"
    )
    return _hasher_type


class PasswordHasher:
    """Secure password hashing with automatic algorithm selection.

    Usage:
        hasher = PasswordHasher()

        # Hash a password
        hashed = hasher.hash("mypassword")

        # Verify a password
        if hasher.verify(hashed, "mypassword"):
            print("Password correct!")

        # Check if rehashing is needed (algorithm upgrade)
        if hasher.needs_rehash(hashed):
            new_hash = hasher.hash("mypassword")
    """

    # PBKDF2 settings (fallback)
    PBKDF2_ITERATIONS = 600000  # OWASP recommendation for SHA256
    PBKDF2_SALT_LENGTH = 32
    PBKDF2_HASH_LENGTH = 32

    def __init__(self):
        self._hasher_type = _get_hasher_type()

        if self._hasher_type == "argon2":
            from argon2 import PasswordHasher as Argon2Hasher

            self._argon2 = Argon2Hasher()
        elif self._hasher_type == "bcrypt":
            import bcrypt

            self._bcrypt = bcrypt

    def hash(self, password: str) -> str:
        """Hash a password.

        Args:
            password: The plaintext password to hash.

        Returns:
            The hashed password string.
        """
        if self._hasher_type == "argon2":
            return self._argon2.hash(password)
        elif self._hasher_type == "bcrypt":
            return self._bcrypt.hashpw(password.encode("utf-8"), self._bcrypt.gensalt(rounds=12)).decode("utf-8")
        else:
            return self._hash_pbkdf2(password)

    def verify(self, hashed: str, password: str) -> bool:
        """Verify a password against a hash.

        Args:
            hashed: The stored hash.
            password: The plaintext password to verify.

        Returns:
            True if the password matches, False otherwise.
        """
        if not hashed or not password:
            return False

        try:
            # Detect hash type from format
            if hashed.startswith("$argon2"):
                if self._hasher_type != "argon2":
                    logger.error("Argon2 hash found but argon2-cffi not installed")
                    return False
                from argon2.exceptions import VerifyMismatchError

                try:
                    self._argon2.verify(hashed, password)
                    return True
                except VerifyMismatchError:
                    return False

            elif hashed.startswith("$2"):
                # bcrypt hash
                if self._hasher_type == "bcrypt":
                    return self._bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
                else:
                    import bcrypt

                    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

            elif hashed.startswith("pbkdf2:"):
                return self._verify_pbkdf2(hashed, password)

            else:
                logger.warning("Unknown hash format")
                return False

        except Exception as exc:
            logger.error("Error verifying password: %s", exc)
            return False

    def needs_rehash(self, hashed: str) -> bool:
        """Check if a hash should be upgraded to current algorithm.

        Args:
            hashed: The stored hash.

        Returns:
            True if the hash should be upgraded.
        """
        if not hashed:
            return True

        # If we have argon2 and hash isn't argon2, upgrade
        if self._hasher_type == "argon2" and not hashed.startswith("$argon2"):
            return True

        # If we have argon2, check if parameters need updating
        if self._hasher_type == "argon2" and hashed.startswith("$argon2"):
            try:
                return self._argon2.check_needs_rehash(hashed)
            except Exception:
                return False

        # If using bcrypt and hash is PBKDF2, upgrade
        if self._hasher_type == "bcrypt" and hashed.startswith("pbkdf2:"):
            return True

        return False

    def _hash_pbkdf2(self, password: str) -> str:
        """Hash using PBKDF2-SHA256 (fallback)."""
        salt = secrets.token_bytes(self.PBKDF2_SALT_LENGTH)
        hash_bytes = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self.PBKDF2_ITERATIONS,
            dklen=self.PBKDF2_HASH_LENGTH,
        )
        return f"pbkdf2:sha256:{self.PBKDF2_ITERATIONS}${salt.hex()}${hash_bytes.hex()}"

    def _verify_pbkdf2(self, hashed: str, password: str) -> bool:
        """Verify a PBKDF2 hash."""
        try:
            # Format: pbkdf2:sha256:iterations$salt$hash
            parts = hashed.split(":")
            if len(parts) != 3:
                return False

            algo_parts = parts[2].split("$")
            if len(algo_parts) != 3:
                return False

            iterations = int(algo_parts[0])
            salt = bytes.fromhex(algo_parts[1])
            stored_hash = bytes.fromhex(algo_parts[2])

            computed_hash = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt,
                iterations,
                dklen=len(stored_hash),
            )

            return hmac.compare_digest(computed_hash, stored_hash)
        except Exception as exc:
            logger.error("Error verifying PBKDF2 hash: %s", exc)
            return False


class APIKeyGenerator:
    """Generates and validates API keys.

    API keys have the format: st_{random_string}
    The prefix allows easy identification in logs and config.

    Usage:
        gen = APIKeyGenerator()

        # Generate a new key
        full_key, prefix, hash = gen.generate()
        # Store prefix and hash, return full_key to user (once only!)

        # Validate a key
        if gen.verify(stored_hash, user_provided_key):
            print("Valid key!")
    """

    PREFIX = "st_"
    KEY_LENGTH = 32  # Characters after prefix
    PREFIX_DISPLAY_LENGTH = 8  # Characters to show in UI (after "st_")

    def __init__(self):
        self._hasher = PasswordHasher()

    def generate(self) -> Tuple[str, str, str]:
        """Generate a new API key.

        Returns:
            Tuple of (full_key, prefix_for_display, hash_for_storage)

        Example:
            full_key, prefix, hash = gen.generate()
            # full_key: "st_a1b2c3d4e5f6..." (return to user once)
            # prefix: "st_a1b2c3d4" (for display in UI)
            # hash: "$argon2..." (store in database)
        """
        # Generate random key
        alphabet = string.ascii_letters + string.digits
        random_part = "".join(secrets.choice(alphabet) for _ in range(self.KEY_LENGTH))
        full_key = f"{self.PREFIX}{random_part}"

        # Create display prefix
        prefix = full_key[: len(self.PREFIX) + self.PREFIX_DISPLAY_LENGTH]

        # Hash the full key for storage
        key_hash = self._hash_key(full_key)

        return full_key, prefix, key_hash

    def _hash_key(self, key: str) -> str:
        """Hash an API key for storage.

        Uses SHA256 instead of password hashing since API keys are
        high-entropy and don't need the same protection as passwords.
        """
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def verify(self, stored_hash: str, provided_key: str) -> bool:
        """Verify an API key against its stored hash.

        Args:
            stored_hash: The stored hash from the database.
            provided_key: The key provided by the user.

        Returns:
            True if the key matches.
        """
        if not stored_hash or not provided_key:
            return False

        if not provided_key.startswith(self.PREFIX):
            return False

        computed_hash = self._hash_key(provided_key)
        return hmac.compare_digest(computed_hash, stored_hash)


# Singleton instances
_password_hasher: Optional[PasswordHasher] = None
_api_key_generator: Optional[APIKeyGenerator] = None


def get_password_hasher() -> PasswordHasher:
    """Get the global password hasher instance."""
    global _password_hasher
    if _password_hasher is None:
        _password_hasher = PasswordHasher()
    return _password_hasher


def get_api_key_generator() -> APIKeyGenerator:
    """Get the global API key generator instance."""
    global _api_key_generator
    if _api_key_generator is None:
        _api_key_generator = APIKeyGenerator()
    return _api_key_generator
