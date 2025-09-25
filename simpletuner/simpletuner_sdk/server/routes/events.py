"""
Event and callback routes for SimpleTuner server.
Handles webhook callbacks and event broadcasting.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("EventRoutes")

router = APIRouter(prefix="")


@router.post("/callback")
async def handle_callback(request: Request):
    """
    Endpoint to receive incoming callbacks and store them as events.
    """
    data = await request.json()

    # Get event store from app state (will be set in unified mode)
    event_store = getattr(request.app.state, "event_store", None)

    if not event_store:
        # Fallback to module-level storage for standalone mode
        from ..services.event_store import get_default_store

        event_store = get_default_store()

    # Store the event
    if data.get("message_type") == "configure_webhook":
        # New session starting, clear old events
        event_store.clear()

    event_store.add_event(data)

    logger.info(f"Received callback: {data.get('message_type', 'unknown')}")
    return {"message": "Callback received successfully"}


@router.get("/broadcast")
async def broadcast(request: Request, last_event_index: int = 0):
    """
    Endpoint for long polling, where the client requests events newer than the last received index.
    """
    # Get event store
    event_store = getattr(request.app.state, "event_store", None)
    if not event_store:
        from ..services.event_store import get_default_store

        event_store = get_default_store()

    try:
        # Long polling with timeout
        timeout = 30  # seconds
        start_time = asyncio.get_event_loop().time()

        while True:
            # Check for new events
            events = event_store.get_events_since(last_event_index)
            if events:
                next_index = last_event_index + len(events)
                return JSONResponse(content={"events": events, "next_index": next_index})

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
    Server-Sent Events endpoint for real-time updates.
    """

    async def event_generator():
        """Generate SSE events."""
        # Get event store
        event_store = getattr(request.app.state, "event_store", None)
        if not event_store:
            from ..services.event_store import get_default_store

            event_store = get_default_store()

        last_index = 0

        # Send initial connection event
        yield f"data: {json.dumps({'type': 'connected', 'message': 'Connected to SimpleTuner'})}\n\n"

        try:
            while True:
                # Check for new events
                events = event_store.get_events_since(last_index)

                for event in events:
                    # Transform event data for frontend
                    sse_event = {
                        "type": event.get("message_type", "notification"),
                        "data": event,
                        "timestamp": event.get("timestamp"),
                    }

                    # Map specific event types
                    if event.get("message_type") == "training_progress":
                        sse_event["type"] = "training_progress"
                        sse_event["progress"] = event.get("progress", {})
                    elif event.get("message_type") == "validation_complete":
                        sse_event["type"] = "validation_complete"
                    elif event.get("message_type") == "error":
                        sse_event["type"] = "error"
                        sse_event["message"] = event.get("message", "Unknown error")
                    else:
                        sse_event["type"] = "notification"
                        sse_event["message"] = event.get("message", str(event))
                        sse_event["level"] = event.get("level", "info")

                    yield f"data: {json.dumps(sse_event)}\n\n"

                last_index += len(events)

                # Wait before checking for more events
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            # Client disconnected
            logger.info("SSE client disconnected")
            return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # CORS headers are now handled by middleware - see security_middleware.py
        },
    )


@router.get("/api/events/recent")
async def get_recent_events(request: Request):
    """Get recent training events for HTMX display."""
    from fastapi.responses import HTMLResponse

    # Get events from event store
    event_store = getattr(request.app.state, "event_store", None)

    if not event_store:
        from ..services.event_store import get_default_store

        event_store = get_default_store()

    try:
        # Get last 10 events
        events = event_store.get_events_since(max(0, event_store.get_event_count() - 10))
    except Exception:
        # Event store might be empty or unavailable
        events = []

    if not events:
        return HTMLResponse(
            """
        <div class="text-muted text-center py-3">
            <i class="fas fa-info-circle"></i> No recent events
        </div>
        """
        )

    html = ""
    for event in reversed(events):  # Show newest first
        event_type = event.get("message_type", "info")
        timestamp = event.get("timestamp", "")
        message = event.get("message", str(event))

        # Determine icon and color based on event type
        if event_type in ["training_progress", "progress"]:
            icon = "fas fa-chart-line text-info"
        elif event_type in ["error", "training_error"]:
            icon = "fas fa-exclamation-circle text-danger"
        elif event_type in ["validation_complete", "checkpoint_saved"]:
            icon = "fas fa-check-circle text-success"
        else:
            icon = "fas fa-info-circle text-muted"

        html += f"""
        <div class="event-item border-bottom py-2">
            <div class="d-flex align-items-start">
                <i class="{icon} me-2 mt-1"></i>
                <div class="flex-grow-1">
                    <div class="event-message">{message}</div>
                    {f'<small class="text-muted">{timestamp}</small>' if timestamp else ''}
                </div>
            </div>
        </div>
        """

    return HTMLResponse(html)
