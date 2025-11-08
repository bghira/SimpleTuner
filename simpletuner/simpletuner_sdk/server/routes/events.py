"""
Event and callback routes for SimpleTuner server.
Handles webhook callbacks and event broadcasting.
"""

import asyncio
import base64
import binascii
import json
import logging
import mimetypes
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request, Response
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


from ..services.callback_events import CallbackEvent, EventType
from ..services.callback_presenter import CallbackPresenter
from ..services.callback_service import CallbackService, get_default_callback_service
from ..services.sse_manager import get_sse_manager

logger = logging.getLogger("EventRoutes")

router = APIRouter(prefix="")

BROADCAST_TIMEOUT_ENV_VAR = "SIMPLETUNER_BROADCAST_TIMEOUT_SECONDS"
DEFAULT_BROADCAST_TIMEOUT = 30.0


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


def _prepare_replay_validation_payload(event: CallbackEvent, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a payload copy that proxies heavy media through download endpoints."""

    cleaned: dict[str, Any] = dict(payload)
    event_index = event.index
    manifest: dict[str, Any] = {}

    if event_index is not None:
        image_proxies = _build_media_proxy_list(event_index, event.images, media_kind="image")
        if image_proxies:
            cleaned["images"] = image_proxies
            manifest["images"] = len(image_proxies)
        else:
            cleaned.pop("images", None)

        video_proxies = _build_media_proxy_list(event_index, event.videos, media_kind="video")
        if video_proxies:
            cleaned["videos"] = video_proxies
            manifest["videos"] = len(video_proxies)
        else:
            cleaned.pop("videos", None)
    else:
        cleaned.pop("images", None)
        cleaned.pop("videos", None)

    validation_payload = cleaned.get("validation")
    if isinstance(validation_payload, Mapping):
        validation_copy = dict(validation_payload)
        if validation_copy.get("assets"):
            validation_copy["assets"] = []
        if validation_copy.get("images"):
            validation_copy["images"] = []
        cleaned["validation"] = validation_copy

    data_payload = cleaned.get("data")
    if isinstance(data_payload, Mapping):
        data_copy = dict(data_payload)
        for key in ("assets", "images"):
            if data_copy.get(key):
                data_copy[key] = []
        for key in ("image", "preview"):
            if data_copy.get(key):
                data_copy[key] = None
        cleaned["data"] = data_copy

    cleaned["media_filtered"] = True
    cleaned["media_proxy_available"] = bool(event_index is not None)
    if manifest:
        cleaned["media_manifest"] = manifest
    return cleaned


def _build_media_proxy_list(event_index: int, media_list: Sequence[Any] | None, *, media_kind: str) -> list[dict[str, Any]]:
    proxies: list[dict[str, Any]] = []
    if media_list is None:
        return proxies
    for idx, source in enumerate(media_list):
        proxy = _build_media_proxy(event_index, media_kind, idx, source)
        if proxy:
            proxies.append(proxy)
    return proxies


def _build_media_proxy(event_index: int, media_kind: str, media_index: int, source: Any) -> dict[str, Any] | None:
    default_mime = "video/mp4" if media_kind == "video" else "image/png"
    referenced_src, mime_hint = _extract_media_reference(source)

    if referenced_src and referenced_src.startswith(("http://", "https://", "//")):
        return {"src": referenced_src, "mime_type": mime_hint or default_mime}

    proxy_url = f"/api/events/{event_index}/media/{media_kind}/{media_index}"
    return {"src": proxy_url, "mime_type": mime_hint or default_mime}


def _extract_media_reference(source: Any) -> tuple[str | None, str | None]:
    """Return a tuple of (media source string, mime hint)."""

    if isinstance(source, Mapping):
        mime_hint = source.get("mime_type") or source.get("mime") or source.get("content_type")
        for key in ("src", "url", "data", "base64", "image", "image_base64", "video", "video_base64"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), mime_hint
        return None, mime_hint

    if isinstance(source, str):
        value = source.strip()
        return (value or None), None

    return None, None


def _resolve_media_blob(source: Any, *, default_mime: str) -> tuple[bytes, str]:
    referenced_src, mime_hint = _extract_media_reference(source)
    if not referenced_src:
        raise ValueError("Media payload missing data")

    if referenced_src.startswith("data:"):
        try:
            header, encoded = referenced_src.split(",", 1)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError("Malformed data URI") from exc
        mime = _extract_data_uri_mime(header) or mime_hint or default_mime
        try:
            return base64.b64decode(encoded), mime
        except binascii.Error as exc:  # pragma: no cover - defensive guard
            raise ValueError("Invalid base64 content in data URI") from exc

    if referenced_src.startswith(("http://", "https://", "//")):
        raise ValueError("Remote media cannot be proxied")

    path_candidate = Path(referenced_src)
    if path_candidate.is_file():
        mime = mime_hint or mimetypes.guess_type(path_candidate.name)[0] or default_mime
        return path_candidate.read_bytes(), mime

    try:
        return base64.b64decode(referenced_src), mime_hint or default_mime
    except binascii.Error as exc:
        raise ValueError("Media payload is not base64 encoded") from exc


def _extract_data_uri_mime(header: str) -> str | None:
    if ":" not in header:
        return None
    _, meta = header.split(":", 1)
    if ";" in meta:
        mime, _ = meta.split(";", 1)
        return mime or None
    return meta or None


def _resolve_broadcast_timeout(timeout_param: float | None) -> float:
    """Return timeout value using query parameter or environment override."""
    if timeout_param is not None:
        return timeout_param

    env_value = os.environ.get(BROADCAST_TIMEOUT_ENV_VAR)
    if env_value is not None:
        try:
            parsed = float(env_value)
        except ValueError:
            logger.warning(
                "Invalid %s value '%s'; falling back to default %.1f",
                BROADCAST_TIMEOUT_ENV_VAR,
                env_value,
                DEFAULT_BROADCAST_TIMEOUT,
            )
        else:
            if parsed > 0:
                return parsed
            logger.warning(
                "%s must be positive; received '%s'. Using default %.1f",
                BROADCAST_TIMEOUT_ENV_VAR,
                env_value,
                DEFAULT_BROADCAST_TIMEOUT,
            )

    return DEFAULT_BROADCAST_TIMEOUT


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
async def broadcast(
    request: Request,
    last_event_index: int = 0,
    timeout: Annotated[
        float | None,
        Query(gt=0.0, description="Maximum time in seconds to wait for new events."),
    ] = None,
):
    """
    Endpoint for long polling, where the client requests events newer than the last received index.
    """
    callback_service = _get_callback_service(request)

    try:
        # Long polling with timeout
        timeout_limit = _resolve_broadcast_timeout(timeout)
        start_time = asyncio.get_event_loop().time()

        while True:
            # Check for new events
            events = callback_service.stream_since(last_event_index)
            if events:
                payloads = [event.to_payload() for event in events]
                latest_index = max((event.index or last_event_index for event in events), default=last_event_index)
                return JSONResponse(content={"events": payloads, "next_index": latest_index})

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout_limit:
                # Return empty response on timeout
                return JSONResponse(content={"events": [], "next_index": last_event_index})

            # Wait before checking again
            remaining = max(timeout_limit - elapsed, 0.0)
            await asyncio.sleep(min(1.0, remaining))

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
        replay_cutoff_index = callback_service.latest_index()

        # Support SSE reconnection by reading Last-Event-ID header
        # This prevents replaying all events when the connection drops
        last_event_id = request.headers.get("Last-Event-ID")
        if last_event_id:
            try:
                last_index = int(last_event_id)
                logger.debug(f"SSE reconnection detected, resuming from event {last_index}")
            except ValueError:
                logger.warning(f"Invalid Last-Event-ID header: {last_event_id}, starting from beginning")
                last_index = -1
        else:
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
                            # Skip events that are already directly broadcast via SSE
                            # to prevent duplicate delivery to clients

                            should_skip = False

                            # Skip training progress events (broadcast via _broadcast_training_progress)
                            if event.type == EventType.TRAINING_PROGRESS:
                                should_skip = True

                            # Skip lifecycle stage events (broadcast via _broadcast_startup_stage)
                            if event.stage is not None:
                                should_skip = True

                            is_replay_event = (
                                replay_cutoff_index is not None
                                and event.index is not None
                                and event.index <= replay_cutoff_index
                            )

                            if should_skip:
                                # Update last_index but don't send to client
                                if event.index is not None:
                                    last_index = event.index
                                continue

                            # Drop intermediary validation.image events when replaying to avoid large payloads
                            if is_replay_event and event.type == EventType.VALIDATION_IMAGE:
                                if event.index is not None:
                                    last_index = event.index
                                continue

                            # Send non-broadcast events via polling
                            event_type, payload = CallbackPresenter.to_sse(event)

                            # Mark replay validation events and replace their media with download proxies
                            if is_replay_event and event.type == EventType.VALIDATION:
                                payload = dict(payload)
                                payload["is_replay"] = True
                                payload = _prepare_replay_validation_payload(event, payload)

                            await sse_manager.send_to_connection(
                                connection.connection_id,
                                payload,
                                event_type=event_type,
                                event_id=str(event.index) if event.index is not None else None,
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
                    sse_msg = {"data": json.dumps(message["data"])}

                    # Add event type if present
                    if message.get("event"):
                        sse_msg["event"] = message["event"]

                    # Add event ID if present (for Last-Event-ID tracking on reconnect)
                    if message.get("id"):
                        sse_msg["id"] = str(message["id"])

                    yield sse_msg

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


@router.get("/api/events/{event_index}/media/{media_kind}/{media_index}")
async def get_event_media(event_index: int, media_kind: str, media_index: int, request: Request):
    """Stream stored validation media for replay payloads."""

    if media_kind not in {"image", "video"}:
        raise HTTPException(status_code=404, detail="Unsupported media type")

    callback_service = _get_callback_service(request)
    event = callback_service.get_event_by_index(event_index)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")

    media_list = list(event.images if media_kind == "image" else event.videos)
    if media_index < 0 or media_index >= len(media_list):
        raise HTTPException(status_code=404, detail="Media index out of range")

    default_mime = "video/mp4" if media_kind == "video" else "image/png"
    try:
        blob, mime_type = _resolve_media_blob(media_list[media_index], default_mime=default_mime)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    headers = {"Cache-Control": "public, max-age=604800, immutable"}
    return Response(content=blob, media_type=mime_type, headers=headers)


@router.get("/api/events/recent")
async def get_recent_events(request: Request):
    """Get recent training events for HTMX display."""
    from fastapi.responses import HTMLResponse

    callback_service = _get_callback_service(request)

    events = callback_service.get_recent(limit=10)

    # Filter out intermediary validation images - these are only for the streaming lightbox
    # Final validation images (from validation events) should still appear in the event list
    events = [event for event in events if event.type != EventType.VALIDATION_IMAGE]

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
