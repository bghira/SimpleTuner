"""LDAP/Active Directory authentication provider.

Supports authentication with LDAP-compliant directories including
Active Directory, OpenLDAP, FreeIPA, etc.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base import AuthProviderBase, ExternalUser, ProviderConfig

logger = logging.getLogger(__name__)


class LDAPProvider(AuthProviderBase):
    """LDAP authentication provider.

    Configuration options (in config.config):
        - server: LDAP server URL (e.g., ldap://ldap.example.com:389)
        - bind_dn: DN for initial bind (optional, for user search)
        - bind_password: Password for bind_dn
        - base_dn: Base DN for searches (e.g., dc=example,dc=com)
        - user_search_filter: Filter template (default: "(uid={username})")
        - user_search_base: Base for user search (default: base_dn)
        - group_search_filter: Filter for groups (default: "(member={user_dn})")
        - group_search_base: Base for group search
        - email_attribute: Attribute for email (default: "mail")
        - username_attribute: Attribute for username (default: "uid")
        - display_name_attribute: Attribute for display name (default: "cn")
        - group_name_attribute: Attribute for group name (default: "cn")
        - use_ssl: Use LDAPS (default: False)
        - start_tls: Use StartTLS (default: False)
        - verify_ssl: Verify SSL certificates (default: True)
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)

        # Accept both "server" and "server_url" for compatibility
        self._server = config.config.get("server") or config.config.get("server_url", "")
        self._bind_dn = config.config.get("bind_dn")
        self._bind_password = config.config.get("bind_password")
        self._base_dn = config.config.get("base_dn", "")
        # Accept both "user_search_filter" and "user_filter" for compatibility
        self._user_search_filter = config.config.get("user_search_filter") or config.config.get(
            "user_filter", "(uid={username})"
        )
        # Accept both "user_search_base" and "user_base_dn" for compatibility
        self._user_search_base = config.config.get("user_search_base") or config.config.get("user_base_dn") or self._base_dn
        self._group_search_filter = config.config.get("group_search_filter", "(member={user_dn})")
        self._group_search_base = config.config.get("group_search_base") or self._base_dn
        self._email_attr = config.config.get("email_attribute", "mail")
        self._username_attr = config.config.get("username_attribute", "uid")
        self._display_name_attr = config.config.get("display_name_attribute", "cn")
        self._group_name_attr = config.config.get("group_name_attribute", "cn")
        self._use_ssl = config.config.get("use_ssl", False)
        self._start_tls = config.config.get("start_tls", False)
        self._verify_ssl = config.config.get("verify_ssl", True)

    def _get_connection(self):
        """Create an LDAP connection."""
        try:
            import ldap3
        except ImportError:
            raise ImportError("ldap3 package required for LDAP authentication. Install with: pip install ldap3")

        server = ldap3.Server(
            self._server,
            use_ssl=self._use_ssl,
            get_info=ldap3.ALL,
        )

        return ldap3.Connection(
            server,
            user=self._bind_dn,
            password=self._bind_password,
            auto_bind=False,
        )

    async def authenticate(
        self,
        credentials: Dict[str, Any],
    ) -> Tuple[bool, Optional[ExternalUser], Optional[str]]:
        """Authenticate a user with username/password.

        Args:
            credentials: {"username": "...", "password": "..."}

        Returns:
            Tuple of (success, user, error)
        """
        username = credentials.get("username", "")
        password = credentials.get("password", "")

        if not username or not password:
            return False, None, "Username and password required"

        try:
            import ldap3

            # First, find the user
            user_dn, user_attrs = await self._find_user(username)

            if not user_dn:
                return False, None, "User not found"

            # Try to bind as the user
            server = ldap3.Server(
                self._server,
                use_ssl=self._use_ssl,
                get_info=ldap3.NONE,
            )

            conn = ldap3.Connection(
                server,
                user=user_dn,
                password=password,
                auto_bind=False,
            )

            try:
                if self._start_tls:
                    conn.start_tls()

                if not conn.bind():
                    return False, None, "Invalid credentials"

                # Get user groups (pass username for posixGroup support)
                groups = await self._get_user_groups(user_dn, username)

                # Create user object
                user = self._attrs_to_user(user_dn, user_attrs, groups)
                return True, user, None
            finally:
                try:
                    conn.unbind()
                except Exception:
                    pass

        except ImportError as exc:
            return False, None, str(exc)
        except Exception as exc:
            logger.error("LDAP auth failed: %s", exc, exc_info=True)
            return False, None, str(exc)

    async def _find_user(self, username: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Find a user in LDAP and return their DN and attributes."""
        try:
            import ldap3

            conn = self._get_connection()

            try:
                if self._start_tls:
                    conn.start_tls()

                if self._bind_dn:
                    if not conn.bind():
                        logger.error("LDAP bind failed: %s", conn.result)
                        return None, {}
                else:
                    # Anonymous bind
                    conn.bind()

                # Search for user
                search_filter = self._user_search_filter.format(username=ldap3.utils.conv.escape_filter_chars(username))

                conn.search(
                    search_base=self._user_search_base,
                    search_filter=search_filter,
                    search_scope=ldap3.SUBTREE,
                    attributes=[self._email_attr, self._username_attr, self._display_name_attr],
                )

                if not conn.entries:
                    return None, {}

                entry = conn.entries[0]
                user_dn = entry.entry_dn

                attrs = {}
                for attr in [self._email_attr, self._username_attr, self._display_name_attr]:
                    if hasattr(entry, attr):
                        val = getattr(entry, attr)
                        if val:
                            attrs[attr] = str(val)

                return user_dn, attrs
            finally:
                try:
                    conn.unbind()
                except Exception:
                    pass

        except Exception as exc:
            logger.error("User search failed: %s", exc)
            return None, {}

    async def _get_user_groups(self, user_dn: str, username: str = "") -> List[str]:
        """Get groups a user belongs to.

        Supports both groupOfNames (member={user_dn}) and posixGroup (memberUid={username}).
        """
        try:
            import ldap3

            conn = self._get_connection()

            try:
                if self._start_tls:
                    conn.start_tls()

                if self._bind_dn:
                    if not conn.bind():
                        return []
                else:
                    conn.bind()

                # Support both {user_dn} and {username} placeholders
                search_filter = self._group_search_filter.format(
                    user_dn=ldap3.utils.conv.escape_filter_chars(user_dn),
                    username=ldap3.utils.conv.escape_filter_chars(username),
                )

                conn.search(
                    search_base=self._group_search_base,
                    search_filter=search_filter,
                    search_scope=ldap3.SUBTREE,
                    attributes=[self._group_name_attr],
                )

                groups = []
                for entry in conn.entries:
                    if hasattr(entry, self._group_name_attr):
                        val = getattr(entry, self._group_name_attr)
                        if val:
                            groups.append(str(val))

                return groups
            finally:
                try:
                    conn.unbind()
                except Exception:
                    pass

        except Exception as exc:
            logger.warning("Group search failed: %s", exc)
            return []

    def _attrs_to_user(
        self,
        user_dn: str,
        attrs: Dict[str, Any],
        groups: List[str],
    ) -> ExternalUser:
        """Convert LDAP attributes to ExternalUser."""
        email = attrs.get(self._email_attr, "")
        username = attrs.get(self._username_attr, "")
        display_name = attrs.get(self._display_name_attr)

        # Use DN as external_id
        return ExternalUser(
            external_id=user_dn,
            email=email,
            username=username or email.split("@")[0],
            display_name=display_name,
            provider_type="ldap",
            provider_name=self.name,
            groups=groups,
            roles=[],
            raw_attributes=attrs,
            email_verified=bool(email),
        )

    async def validate_token(self, token: str) -> Tuple[bool, Optional[ExternalUser]]:
        """LDAP doesn't use tokens - always returns False."""
        # LDAP auth is session-based, not token-based
        return False, None

    async def get_auth_url(self, redirect_uri: str, state: str) -> str:
        """LDAP doesn't use OAuth - raises NotImplementedError."""
        raise NotImplementedError("LDAP does not support OAuth flow")

    async def refresh_token(self, refresh_token: str) -> Tuple[Optional[str], Optional[str]]:
        """LDAP doesn't use tokens."""
        return None, None

    async def test_connection(self) -> Tuple[bool, Optional[str]]:
        """Test connection to LDAP server."""
        try:
            conn = self._get_connection()

            try:
                if self._start_tls:
                    conn.start_tls()

                if self._bind_dn:
                    if not conn.bind():
                        return False, f"Bind failed: {conn.result}"
                else:
                    conn.bind()

                return True, None
            finally:
                try:
                    conn.unbind()
                except Exception:
                    pass

        except ImportError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, str(exc)
