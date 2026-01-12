"""State Backend Adapters.

Adapters that wrap the StateBackendProtocol for specific use cases:
    - AsyncRateLimiter: Sliding window rate limiting
    - AsyncCircuitBreaker: Circuit breaker pattern
    - AsyncTTLCache: Generic TTL cache
    - OIDCStateStore: OAuth/OIDC flow state
"""

from .circuit_breaker import AsyncCircuitBreaker
from .oidc_state import OIDCStateStore
from .rate_limiter import AsyncRateLimiter
from .ttl_cache import AsyncTTLCache

__all__ = [
    "AsyncRateLimiter",
    "AsyncCircuitBreaker",
    "AsyncTTLCache",
    "OIDCStateStore",
]
