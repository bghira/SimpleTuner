"""State Backend Implementations.

Available backends:
    - SQLiteStateBackend: Default, single-node, uses aiosqlite
    - MemoryStateBackend: Testing only, in-memory dict
    - PostgreSQLStateBackend: Multi-node, requires asyncpg
    - MySQLStateBackend: Multi-node, requires aiomysql
    - RedisStateBackend: Distributed cache, requires redis
"""

from .memory import MemoryStateBackend
from .sqlite import SQLiteStateBackend

__all__ = [
    "SQLiteStateBackend",
    "MemoryStateBackend",
]

# Optional backends - imported on demand to avoid missing dependency errors
# from .postgresql import PostgreSQLStateBackend
# from .mysql import MySQLStateBackend
# from .redis import RedisStateBackend
