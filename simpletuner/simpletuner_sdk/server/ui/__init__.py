# agpl-3.0-or-later © 2024 bghira
"""NiceGUI integration entrypoint for the SimpleTuner server."""
from __future__ import annotations

from fastapi import FastAPI
from nicegui import app as nicegui_app


def mount_ui(app: FastAPI, prefix: str = "/web") -> None:
    """Mount the NiceGUI application under the provided URL prefix."""
    if getattr(app.state, "nicegui_mounted", False):
        return

    # Configure basic application metadata so NiceGUI can render page titles
    config = nicegui_app.config
    if not config.has_run_config:
        config.add_run_config(
            reload=False,
            title="SimpleTuner",
            viewport="width=device-width, initial-scale=1",
            favicon="/static/favicon.ico",
            dark=False,
            language="en-US",
            binding_refresh_interval=0.1,
            reconnect_timeout=3.0,
            message_history_length=1000,
            tailwind=False,
            prod_js=True,
            show_welcome_message=False,
        )
        config.endpoint_documentation = "none"

    # Import side effects: registers pages on the global NiceGUI app
    from . import pages  # noqa: F401

    app.mount(prefix, nicegui_app)
    app.state.nicegui_mounted = True
