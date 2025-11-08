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
    # Use the enum values directly (with dots) for consistency
    _EVENT_TYPE_TO_SSE = {
        EventType.LIFECYCLE_STAGE: EventType.LIFECYCLE_STAGE.value,
        EventType.TRAINING_PROGRESS: EventType.TRAINING_PROGRESS.value,
        EventType.TRAINING_STATUS: EventType.TRAINING_STATUS.value,
        EventType.TRAINING_SUMMARY: EventType.TRAINING_STATUS.value,
        EventType.NOTIFICATION: EventType.NOTIFICATION.value,
        EventType.ERROR: EventType.ERROR.value,
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

        # Preserve the event type value in the payload for client-side filtering
        # This is especially important for distinguishing validation.image from validation
        if event.type == EventType.VALIDATION_IMAGE:
            payload.setdefault("type", EventType.VALIDATION_IMAGE.value)
        elif event.type == EventType.VALIDATION:
            # Regular validation events may have a type in their payload already
            # Only set it if not present
            if "type" not in payload:
                payload.setdefault("type", EventType.VALIDATION.value)

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
        videos_html = cls._render_videos(payload.get("videos") or (), headline)
        media_html = images_html + videos_html

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
            f"{media_html}"
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
    def _render_videos(videos: Any, alt: str | None) -> str:
        if not videos:
            return ""
        rendered: list[str] = []
        alt_text = html.escape(str(alt)) if alt else "Event video"

        for video in videos:
            src = CallbackPresenter._normalise_video_src(video)
            if not src:
                continue
            rendered.append(
                (
                    '<video src="{src}" class="event-video img-fluid rounded border mt-2" '
                    "controls muted playsinline loop></video>"
                ).format(src=html.escape(src, quote=True))
            )

        if not rendered:
            return ""

        return '<div class="event-videos d-flex flex-wrap gap-2">' + "".join(rendered) + "</div>"

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

    @staticmethod
    def _normalise_video_src(video: Any) -> str | None:
        if video is None:
            return None

        if isinstance(video, str):
            value = video.strip()
            if not value:
                return None
            if value.startswith("data:") or value.startswith(("http://", "https://", "//")):
                return value
            if re.match(r"^[A-Za-z0-9+/]+=*$", value):
                return f"data:video/mp4;base64,{value}"
            return value

        if isinstance(video, Mapping):
            data = (
                video.get("src")
                or video.get("url")
                or video.get("data")
                or video.get("base64")
                or video.get("video")
                or video.get("video_base64")
            )
            if not isinstance(data, str) or not data.strip():
                return None
            data = data.strip()
            if data.startswith("data:") or data.startswith(("http://", "https://", "//")):
                return data
            mime = video.get("mime_type") or video.get("mime") or "video/mp4"
            return f"data:{mime};base64,{data}"

        return None


__all__ = ["CallbackPresenter"]
