"""
FastAPI application factory for SimpleTuner server with multiple modes.
"""

import logging
import os
from enum import Enum
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .utils.paths import get_simpletuner_root, get_template_directory, get_static_directory


logger = logging.getLogger("SimpleTunerServer")


class ServerMode(Enum):
    """Server operation modes."""

    TRAINER = "trainer"  # Training API only (port 8001)
    CALLBACK = "callback"  # Callback receiver only (port 8002)
    UNIFIED = "unified"  # Both APIs in single process


def create_app(
    mode: ServerMode = ServerMode.TRAINER,
    enable_cors: bool = True,
    static_dir: Optional[str] = None,
    template_dir: Optional[str] = None,
) -> FastAPI:
    """
    Create a FastAPI application with the specified mode.

    Args:
        mode: Server mode (trainer, callback, or unified)
        enable_cors: Whether to enable CORS
        static_dir: Path to static files directory
        template_dir: Path to templates directory

    Returns:
        Configured FastAPI application
    """

    # Create base app
    title = f"SimpleTuner {mode.value.capitalize()} Server"
    app = FastAPI(title=title)

    # Configure CORS if enabled
    if enable_cors:
        origins = [
            "http://localhost:8000",
            "http://localhost:8001",
            "http://localhost:8002",
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8001",
            "http://127.0.0.1:8002",
        ]

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Mount static files if directory exists
    if static_dir and os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    else:
        # Use absolute path to SimpleTuner's static directory
        static_path = get_static_directory()
        if static_path.exists():
            app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    # Set up template directory
    if template_dir:
        os.environ["TEMPLATE_DIR"] = template_dir
    else:
        # Use absolute path to SimpleTuner's templates directory
        template_path = get_template_directory()
        if template_path.exists():
            os.environ["TEMPLATE_DIR"] = str(template_path)

    # Add routes based on mode
    if mode in (ServerMode.TRAINER, ServerMode.UNIFIED):
        _add_trainer_routes(app)

    if mode in (ServerMode.CALLBACK, ServerMode.UNIFIED):
        _add_callback_routes(app)

    # Add health check
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "mode": mode.value}

    # Add root redirect for trainer mode
    if mode == ServerMode.TRAINER:
        from fastapi.responses import RedirectResponse

        @app.get("/")
        async def root():
            """Redirect to web interface"""
            return RedirectResponse(url="/web/trainer")

    return app


def _add_trainer_routes(app: FastAPI):
    """Add training-related routes to the app."""

    # Import and add existing routes
    from simpletuner.simpletuner_sdk.configuration import Configuration
    from simpletuner.simpletuner_sdk.interface import WebInterface
    from simpletuner.simpletuner_sdk.training_host import TrainingHost

    # Initialize web interface
    web_interface = WebInterface()
    app.include_router(web_interface.router)

    # Configuration controller
    config_controller = Configuration()
    app.include_router(config_controller.router)

    # Training host controller
    training_host = TrainingHost()
    app.include_router(training_host.router)

    # Add API routes
    from .routes.configs import router as configs_router
    from .routes.datasets import router as datasets_router
    from .routes.models import router as models_router
    from .routes.validation import router as validation_router
    from .routes.training import router as training_router
    from .routes.web import router as web_router
    from .routes.webui_state import router as webui_state_router
    from .routes.fields import router as fields_router

    app.include_router(models_router)
    app.include_router(datasets_router)
    app.include_router(configs_router)
    app.include_router(validation_router)
    app.include_router(training_router)
    app.include_router(web_router)
    app.include_router(webui_state_router)
    app.include_router(fields_router)

    logger.info("Added trainer routes")


def _add_callback_routes(app: FastAPI):
    """Add callback/event routes to the app."""

    from .routes.events import router as events_router

    app.include_router(events_router)

    logger.info("Added callback routes")


def create_unified_app() -> FastAPI:
    """
    Create a unified app with both trainer and callback functionality.
    This enables direct event passing without HTTP overhead.
    """

    app = create_app(mode=ServerMode.UNIFIED)

    # Set up shared event store for unified mode
    from .services.event_store import EventStore

    event_store = EventStore()

    # Store event store in app state for access by routes
    app.state.event_store = event_store
    app.state.mode = ServerMode.UNIFIED

    # Configure webhook handler to use direct event store in unified mode
    _configure_unified_webhooks(app)

    return app


def _configure_unified_webhooks(app: FastAPI):
    """Configure webhooks to use direct event store in unified mode."""

    # This will be implemented to intercept webhook calls
    # and write directly to event store when target is localhost
    pass
