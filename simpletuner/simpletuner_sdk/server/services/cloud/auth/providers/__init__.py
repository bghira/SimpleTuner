"""External authentication providers for cloud training.

Supports OIDC (OpenID Connect) and LDAP authentication.
"""

from .base import AuthProviderBase, ExternalUser
from .ldap import LDAPProvider
from .manager import AuthProviderManager
from .oidc import OIDCProvider

__all__ = [
    # Base
    "AuthProviderBase",
    "ExternalUser",
    # Providers
    "OIDCProvider",
    "LDAPProvider",
    # Manager
    "AuthProviderManager",
]
