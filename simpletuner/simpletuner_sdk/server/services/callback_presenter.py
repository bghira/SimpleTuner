"""Presenters to transform callback events into SSE and HTMX payloads."""

from __future__ import annotations

import html
import re
from typing import Any, Mapping

from .callback_events import CallbackEvent, EventType


class CallbackPresenter:
    """Utilities that translate typed callback events into presentation payloads."""

    CATEGORY_ICONS: Mapping[str, str] = {
        EventType.TRAINING_PROGRESS.value: "fas fa-chart-line text-info",
        EventType.LIFECYCLE_STAGE.value: "fas fa-cogs text-primary",
        EventType.CHECKPOINT.value: "fas fa-save text-primary",
        EventType.VALIDATION.value: "fas fa-check-double text-success",
        EventType.TRAINING_STATUS.value: "fas fa-info-circle text-secondary",
        EventType.TRAINING_SUMMARY.value: "fas fa-flag-checkered text-success",
        EventType.ERROR.value: "fas fa-exclamation-triangle text-danger",
        EventType.NOTIFICATION.value: "fas fa-bell text-muted",
        EventType.DEBUG.value: "fas fa-bug text-muted",
        EventType.DEBUG_IMAGE.value: "fas fa-image text-warning",
        EventType.VALIDATION_IMAGE.value: "fas fa-images text-success",
    }

    # Map event types to SSE channel names consumed by the WebUI
    _EVENT_TYPE_TO_SSE = {
        EventType.LIFECYCLE_STAGE: "startup_progress",
        EventType.TRAINING_PROGRESS: "training_progress",
        EventType.TRAINING_STATUS: "training_status",
        EventType.TRAINING_SUMMARY: "training_status",
        EventType.NOTIFICATION: "notification",
        EventType.ERROR: "error",
        EventType.CHECKPOINT: "callback:checkpoint",
        EventType.VALIDATION: "callback:validation",
        EventType.METRIC: "callback:metric",
        EventType.DEBUG: "callback:debug",
        EventType.DEBUG_IMAGE: "callback:debug",  # Route debug images same as debug
        EventType.VALIDATION_IMAGE: "callback:validation",  # Route validation images same as validation
    }

    @classmethod
    def to_dict(cls, event: CallbackEvent) -> dict[str, Any]:
        """Return a normalized payload for downstream consumers."""
        payload = event.to_payload()
        payload.setdefault("timestamp_display", event.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        return payload

    @classmethod
    def to_sse(cls, event: CallbackEvent) -> tuple[str, dict[str, Any]]:
        """Return an SSE tuple of event type and payload."""
        payload = cls.to_dict(event)
        event_type = cls._EVENT_TYPE_TO_SSE.get(event.type, event.type.value)
        if callable(event_type):
            event_type = event_type(event)
        return event_type, payload

    @staticmethod
    def _severity_to_bootstrap_class(severity: str) -> str:
        """Map severity levels to Bootstrap text classes."""
        mapping = {
            "success": "success",
            "info": "info",
            "warning": "warning",
            "error": "danger",  # Bootstrap uses 'danger' not 'error'
            "critical": "danger",  # Map critical to danger as well
            "danger": "danger",
            "debug": "secondary",
        }
        return mapping.get(str(severity).lower(), "info")

    @classmethod
    def to_htmx_tile(cls, event: CallbackEvent) -> str:
        """Render a compact HTML snippet suitable for HTMX fragments."""
        payload = cls.to_dict(event)
        icon = cls.CATEGORY_ICONS.get(event.type.value, "fas fa-info-circle text-muted")
        headline = payload.get("title") or payload.get("message") or "Update"
        body = payload.get("message")
        timestamp = payload.get("timestamp_display")
        severity = html.escape(str(payload.get("severity", "info")))

        images_html = cls._render_images(payload.get("images") or (), headline)

        headline_html = html.escape(str(headline))
        body_html = html.escape(str(body)) if body else ""
        timestamp_html = html.escape(timestamp) if timestamp else ""

        timestamp_block = f'<small class="text-muted">{timestamp_html}</small>' if timestamp_html else ""
        body_block = f'<div class="event-body text-muted">{body_html}</div>' if body_html else ""

        return (
            '<div class="event-item border-bottom py-2">'
            '<div class="d-flex align-items-start">'
            f'<i class="{icon} me-2 mt-1"></i>'
            '<div class="flex-grow-1">'
            f'<div class="event-headline text-{cls._severity_to_bootstrap_class(severity)}">{headline_html}</div>'
            f"{body_block}"
            f"{images_html}"
            f"{timestamp_block}"
            "</div>"
            "</div>"
            "</div>"
        )

    @staticmethod
    def _render_images(images: Any, alt: str | None) -> str:
        if not images:
            return ""
        rendered: list[str] = []
        alt_text = html.escape(str(alt)) if alt else "Event image"

        for idx, image in enumerate(images):
            src = CallbackPresenter._normalise_image_src(image)
            if not src:
                continue
            rendered.append(
                (
                    '<img src="{src}" alt="{alt}" '
                    'class="event-image img-fluid rounded border mt-2 cursor-pointer" '
                    'loading="lazy" data-lightbox="event-images" data-lightbox-group="event-{group}" '
                    'data-lightbox-index="{index}" />'
                ).format(src=html.escape(src, quote=True), alt=alt_text, group=html.escape(str(alt) or "event"), index=idx)
            )

        if not rendered:
            return ""

        return '<div class="event-images d-flex flex-wrap gap-2">' + "".join(rendered) + "</div>"

    @staticmethod
    def _normalise_image_src(image: Any) -> str | None:
        if image is None:
            return None

        if isinstance(image, str):
            value = image.strip()
            if not value:
                return None
            if value.startswith("data:"):
                return value
            # Check for URLs first
            if value.startswith(("http://", "https://", "//")):
                return value
            # Only treat as base64 if it looks like base64 (alphanumeric + /+=)
            if re.match(r"^[A-Za-z0-9+/]+=*$", value):
                return f"data:image/png;base64,{value}"
            # Unknown format - return as-is rather than corrupting
            return value

        if isinstance(image, Mapping):
            data = (
                image.get("src")
                or image.get("url")
                or image.get("data")
                or image.get("base64")
                or image.get("image")
                or image.get("image_base64")
            )
            if not isinstance(data, str) or not data.strip():
                return None
            data = data.strip()

            # Already a data URI - return as-is
            if data.startswith("data:"):
                return data

            # Check if it's a URL - return untouched (don't wrap URLs as base64!)
            if data.startswith(("http://", "https://", "//")):
                return data

            # Otherwise, treat as base64 data and wrap it
            mime = image.get("mime_type") or image.get("mime") or "image/png"
            return f"data:{mime};base64,{data}"

        return None


__all__ = ["CallbackPresenter"]
