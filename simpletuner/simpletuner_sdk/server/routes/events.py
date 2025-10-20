"""
Event and callback routes for SimpleTuner server.
Handles webhook callbacks and event broadcasting.
"""

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

try:  # pragma: no cover - optional dependency
    from sse_starlette.sse import EventSourceResponse  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback to basic streaming

    class EventSourceResponse(StreamingResponse):
        """Minimal fallback when sse-starlette is unavailable."""

        def __init__(self, content, *args, **kwargs):
            async def _adapt():
                async for message in content:
                    event = message.get("event")
                    data = message.get("data")
                    payload_parts = []
                    if event:
                        payload_parts.append(f"event: {event}\n")
                    if data is not None:
                        payload_parts.append(f"data: {data}\n")
                    payload_parts.append("\n")
                    yield "".join(payload_parts).encode("utf-8")

            super().__init__(_adapt(), media_type="text/event-stream", *args, **kwargs)


from ..services.callback_presenter import CallbackPresenter
from ..services.callback_service import CallbackService, get_default_callback_service
from ..services.sse_manager import get_sse_manager

logger = logging.getLogger("EventRoutes")

router = APIRouter(prefix="")


def _truncate_long_strings(
    obj: Any,
    *,
    max_length: int = 256,
    preview_length: int = 64,
    suffix: str = "...[truncated]...",
    _seen: set[int] | None = None,
) -> Any:
    """Return a copy of *obj* with overly long strings shortened for logging."""
    if _seen is None:
        _seen = set()

    if isinstance(obj, str):
        return obj if len(obj) <= max_length else f"{obj[:preview_length]}{suffix}"

    if obj is None or isinstance(obj, (int, float, bool)):
        return obj

    obj_id = id(obj)
    if obj_id in _seen:
        return "<recursion>"
    _seen.add(obj_id)

    if isinstance(obj, Mapping):
        return {
            (
                _truncate_long_strings(key, max_length=max_length, preview_length=preview_length, suffix=suffix, _seen=_seen)
                if isinstance(key, str)
                else key
            ): _truncate_long_strings(
                value, max_length=max_length, preview_length=preview_length, suffix=suffix, _seen=_seen
            )
            for key, value in obj.items()
        }

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        items = [
            _truncate_long_strings(item, max_length=max_length, preview_length=preview_length, suffix=suffix, _seen=_seen)
            for item in obj
        ]
        return tuple(items) if isinstance(obj, tuple) else items

    if isinstance(obj, set):
        return [
            _truncate_long_strings(item, max_length=max_length, preview_length=preview_length, suffix=suffix, _seen=_seen)
            for item in obj
        ]

    if hasattr(obj, "__dict__") and isinstance(getattr(obj, "__dict__", None), dict):
        return _truncate_long_strings(
            vars(obj), max_length=max_length, preview_length=preview_length, suffix=suffix, _seen=_seen
        )

    return str(obj)


def _get_callback_service(request: Request) -> CallbackService:
    service = getattr(request.app.state, "callback_service", None)
    if isinstance(service, CallbackService):
        return service
    return get_default_callback_service()


@router.post("/callback")
async def handle_callback(request: Request):
    """
    Endpoint to receive incoming callbacks and store them as events.
    """
    data = await request.json()

    callback_service = _get_callback_service(request)

    event = callback_service.handle_incoming(data)

    safe_raw = _truncate_long_strings(data)
    logger.debug("Received callback: %s", safe_raw)

    if event:
        logger.debug("Normalised callback: %s", _truncate_long_strings(event.to_payload()))

    return {"message": "Callback received successfully"}


@router.get("/broadcast")
async def broadcast(request: Request, last_event_index: int = 0):
    """
    Endpoint for long polling, where the client requests events newer than the last received index.
    """
    callback_service = _get_callback_service(request)

    try:
        # Long polling with timeout
        timeout = 30  # seconds
        start_time = asyncio.get_event_loop().time()

        while True:
            # Check for new events
            events = callback_service.stream_since(last_event_index)
            if events:
                payloads = [event.to_payload() for event in events]
                latest_index = max((event.index or last_event_index for event in events), default=last_event_index)
                return JSONResponse(content={"events": payloads, "next_index": latest_index})

            # Check timeout
            if asyncio.get_event_loop().time() - start_time > timeout:
                # Return empty response on timeout
                return JSONResponse(content={"events": [], "next_index": last_event_index})

            # Wait before checking again
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        # Handle client disconnect
        raise HTTPException(status_code=499, detail="Client disconnected")


@router.get("/api/events")
async def events_stream(request: Request):
    """
    Server-Sent Events endpoint for real-time updates with connection management.
    """
    # Get SSE manager
    sse_manager = get_sse_manager()

    # Try to add connection with limits
    connection = await sse_manager.add_connection(request)
    if not connection:
        # Connection limit reached
        return JSONResponse(
            status_code=503,
            content={
                "error": "Connection limit reached",
                "message": "Too many concurrent connections. Please try again later.",
            },
        )

    async def event_generator():
        """Generate SSE events with proper connection management."""
        callback_service = _get_callback_service(request)
        last_index = -1  # Start at -1 so stream_since(-1) includes index 0

        try:
            # Send initial connection event
            await sse_manager.send_to_connection(
                connection.connection_id,
                {"type": "connected", "message": "Connected to SimpleTuner"},
                event_type="connection",
            )

            # Set up event monitoring task
            async def monitor_events():
                nonlocal last_index
                while connection.active:
                    try:
                        # Check for new events
                        events = callback_service.stream_since(last_index)

                        for event in events:
                            event_type, payload = CallbackPresenter.to_sse(event)
                            await sse_manager.send_to_connection(
                                connection.connection_id,
                                payload,
                                event_type=event_type,
                            )
                            if event.index is not None:
                                last_index = event.index

                        # Wait before checking for more events
                        await asyncio.sleep(1)

                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"Error in event monitor: {e}")
                        await asyncio.sleep(5)  # Wait longer on error

            # Start monitoring task inside try block to ensure cleanup
            monitor_task = None
            try:
                monitor_task = asyncio.create_task(monitor_events())

                # Use SSE manager's event generator
                async for message in sse_manager.create_event_generator(connection):
                    if message.get("event"):
                        yield {"event": message["event"], "data": json.dumps(message["data"])}
                    else:
                        yield {"data": json.dumps(message["data"])}

            finally:
                # Clean up monitoring task - ensure it's cancelled even if creation failed
                if monitor_task is not None:
                    monitor_task.cancel()
                    try:
                        await monitor_task
                    except asyncio.CancelledError:
                        pass

        except Exception as e:
            logger.error(f"Error in SSE stream: {e}")
        finally:
            # Ensure connection is removed
            await sse_manager.remove_connection(connection.connection_id)

    return EventSourceResponse(event_generator())


@router.get("/api/events/recent")
async def get_recent_events(request: Request):
    """Get recent training events for HTMX display."""
    from fastapi.responses import HTMLResponse

    callback_service = _get_callback_service(request)

    events = callback_service.get_recent(limit=10)

    if not events:
        return HTMLResponse(
            """
        <div class="text-muted text-center py-3">
            <i class="fas fa-info-circle"></i> No recent events
        </div>
        """
        )

    html = ""
    for event in events:
        html += CallbackPresenter.to_htmx_tile(event)

    return HTMLResponse(html)
