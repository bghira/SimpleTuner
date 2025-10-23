"""Server-Side SSE Connection Manager.

Manages Server-Sent Events connections with:
- Connection limits per client
- Proper cleanup on disconnect
- Heartbeat/keepalive support
- Message broadcasting
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Set

from fastapi import Request
from sse_starlette import EventSourceResponse

logger = logging.getLogger(__name__)


@dataclass
class SSEConnection:
    """Represents a single SSE connection."""

    connection_id: str
    client_ip: str
    user_agent: str
    created_at: datetime
    last_activity: datetime
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    active: bool = True


class SSEManager:
    """Manages SSE connections with limits and cleanup."""

    def __init__(
        self,
        max_connections_per_ip: int = 5,
        max_total_connections: int = 100,
        heartbeat_interval: int = 15,  # Send heartbeats more frequently to prevent timeouts
        cleanup_interval: int = 60,
        connection_timeout: int = 600,  # 10 minutes - longer timeout to allow for slow connections
    ):
        self.max_connections_per_ip = max_connections_per_ip
        self.max_total_connections = max_total_connections
        self.heartbeat_interval = heartbeat_interval
        self.cleanup_interval = cleanup_interval
        self.connection_timeout = connection_timeout

        self.connections: Dict[str, SSEConnection] = {}
        self.connections_by_ip: Dict[str, Set[str]] = defaultdict(set)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start background tasks."""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        if not self._heartbeat_task:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self):
        """Stop background tasks and close all connections."""
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for conn_id in list(self.connections.keys()):
            await self.remove_connection(conn_id)

    async def add_connection(self, request: Request) -> Optional[SSEConnection]:
        """Add a new SSE connection with limits."""
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        # Check total connection limit
        if len(self.connections) >= self.max_total_connections:
            logger.warning(f"Total connection limit reached ({self.max_total_connections})")
            return None

        # Check per-IP connection limit
        ip_connections = self.connections_by_ip.get(client_ip, set())
        if len(ip_connections) >= self.max_connections_per_ip:
            logger.warning(f"Per-IP connection limit reached for {client_ip} " f"({self.max_connections_per_ip})")
            return None

        # Create new connection
        conn_id = str(uuid.uuid4())
        connection = SSEConnection(
            connection_id=conn_id,
            client_ip=client_ip,
            user_agent=user_agent,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
        )

        self.connections[conn_id] = connection
        self.connections_by_ip[client_ip].add(conn_id)

        logger.info(f"Added SSE connection {conn_id} from {client_ip}")
        return connection

    async def remove_connection(self, connection_id: str):
        """Remove a connection and clean up."""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]
        connection.active = False

        # Remove from tracking
        del self.connections[connection_id]
        self.connections_by_ip[connection.client_ip].discard(connection_id)

        # Clean up empty IP entries
        if not self.connections_by_ip[connection.client_ip]:
            del self.connections_by_ip[connection.client_ip]

        logger.info(f"Removed SSE connection {connection_id}")

    async def send_to_connection(self, connection_id: str, data: Dict[str, Any], event_type: Optional[str] = None):
        """Send data to a specific connection."""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]
        if not connection.active:
            return

        connection.last_activity = datetime.utcnow()

        message = {"data": data, "event": event_type, "timestamp": time.time()}

        await connection.queue.put(message)

    async def broadcast(
        self, data: Dict[str, Any], event_type: Optional[str] = None, filter_func: Optional[callable] = None
    ):
        """Broadcast data to all connections (with optional filter)."""
        tasks = []

        for conn_id, connection in self.connections.items():
            if not connection.active:
                continue

            # Apply filter if provided
            if filter_func and not filter_func(connection):
                continue

            tasks.append(self.send_to_connection(conn_id, data, event_type))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _cleanup_loop(self):
        """Background task to clean up stale connections."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_stale_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_stale_connections(self):
        """Remove connections that have been inactive too long."""
        current_time = datetime.utcnow()
        stale_connections = []

        for conn_id, connection in self.connections.items():
            inactive_seconds = (current_time - connection.last_activity).total_seconds()

            if inactive_seconds > self.connection_timeout:
                stale_connections.append(conn_id)

        for conn_id in stale_connections:
            logger.info(f"Removing stale connection {conn_id}")
            await self.remove_connection(conn_id)

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to keep connections alive."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                await self.broadcast({"type": "heartbeat", "timestamp": time.time()}, event_type="heartbeat")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

    async def create_event_generator(self, connection: SSEConnection):
        """Create an async generator for SSE events."""
        try:
            while connection.active:
                # Wait for messages with timeout
                try:
                    message = await asyncio.wait_for(connection.queue.get(), timeout=1.0)

                    # Format SSE message
                    if message.get("event"):
                        yield {"event": message["event"], "data": message["data"]}
                    else:
                        yield {"data": message["data"]}

                except asyncio.TimeoutError:
                    # Continue loop to check if still active
                    continue

        except asyncio.CancelledError:
            logger.info(f"Event generator cancelled for {connection.connection_id}")
        finally:
            await self.remove_connection(connection.connection_id)

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about current connections."""
        return {
            "total_connections": len(self.connections),
            "connections_by_ip": {ip: len(conns) for ip, conns in self.connections_by_ip.items()},
            "max_connections_per_ip": self.max_connections_per_ip,
            "max_total_connections": self.max_total_connections,
        }


# Global SSE manager instance
_sse_manager: Optional[SSEManager] = None


def get_sse_manager() -> SSEManager:
    """Get the global SSE manager instance."""
    global _sse_manager
    if _sse_manager is None:
        _sse_manager = SSEManager()
    return _sse_manager


async def initialize_sse_manager():
    """Initialize the SSE manager."""
    manager = get_sse_manager()
    await manager.start()
    logger.info("SSE manager initialized")


async def shutdown_sse_manager():
    """Shutdown the SSE manager."""
    manager = get_sse_manager()
    await manager.stop()
    logger.info("SSE manager shut down")
