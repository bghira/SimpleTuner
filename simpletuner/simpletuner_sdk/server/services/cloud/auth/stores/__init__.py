"""Auth stores subpackage.

Provides specialized stores for different auth concerns:

- UserCrudStore: Core user CRUD and authentication
- PermissionStore: Levels, permissions, overrides, resource rules
- SessionStore: Web UI session management
- OAuthStateStore: OAuth/OIDC CSRF protection
- APIKeyStore: API key management
- QuotaStore: Quota management at all scopes
- OrgStore: Organization CRUD
- TeamStore: Team CRUD and membership
- CredentialStore: Per-user provider credentials

Usage::

    from .stores import get_user_crud_store, get_session_store

    async def my_handler():
        user_store = get_user_crud_store()
        user = await user_store.get(user_id)
"""

from .api_key_store import APIKeyStore, get_api_key_store
from .base import BaseAuthStore, get_default_db_path
from .credential_store import CredentialStore, get_credential_store
from .oauth_state_store import OAuthStateStore, get_oauth_state_store
from .org_store import OrgStore, get_org_store
from .permission_store import PermissionStore, get_permission_store
from .quota_store import QuotaStore, get_quota_store
from .session_store import SessionStore, get_session_store
from .team_store import TeamStore, get_team_store
from .user_crud_store import UserCrudStore, get_user_crud_store

__all__ = [
    # Base
    "BaseAuthStore",
    "get_default_db_path",
    # User CRUD
    "UserCrudStore",
    "get_user_crud_store",
    # Permissions
    "PermissionStore",
    "get_permission_store",
    # Session
    "SessionStore",
    "get_session_store",
    # OAuth
    "OAuthStateStore",
    "get_oauth_state_store",
    # API Key
    "APIKeyStore",
    "get_api_key_store",
    # Quota
    "QuotaStore",
    "get_quota_store",
    # Org
    "OrgStore",
    "get_org_store",
    # Team
    "TeamStore",
    "get_team_store",
    # Credentials
    "CredentialStore",
    "get_credential_store",
]
