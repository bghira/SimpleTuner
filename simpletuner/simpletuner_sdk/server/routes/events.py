"""
Event and callback routes for SimpleTuner server.
Handles webhook callbacks and event broadcasting.
"""

import asyncio
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

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
