"""
FastAPI application factory for SimpleTuner server with multiple modes.
"""

import asyncio
import json
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .utils.paths import get_simpletuner_root, get_static_directory, get_template_directory

logger = logging.getLogger("SimpleTunerServer")

# Track if we're shutting down to avoid duplicate cleanup
_shutting_down = False

# Record the first process that imported this module so the shutdown endpoint
# can signal parent reload/watchdog processes when running in dev mode.
os.environ.setdefault("SIMPLETUNER_SERVER_ROOT_PID", str(os.getpid()))


# These placeholders allow tests to monkeypatch heavy imports before the app factory runs.
WebInterface = None
ConfigurationClass = None
TrainingHostClass = None

# Backwards compatible aliases used by tests that patch module-level references.
Configuration = None
TrainingHost = None


class ServerMode(Enum):
    """Server operation modes."""

    TRAINER = "trainer"  # Training API only (port 8001)
    CALLBACK = "callback"  # Callback receiver only (port 8002)
    UNIFIED = "unified"  # Both APIs in single process


def cleanup_training_processes():
    """Terminate all active training processes."""
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True

    try:
        from simpletuner.simpletuner_sdk import process_keeper

        # Get all active processes
        processes = process_keeper.list_processes()
        if processes:
            logger.info(f"Terminating {len(processes)} active training process(es)...")
            for job_id in processes.keys():
                logger.info(f"Terminating training process: {job_id}")
                try:
                    # Use shorter timeout for faster shutdown
                    process_keeper.terminate_process(job_id)
                except Exception as e:
                    logger.error(f"Error terminating process {job_id}: {e}")
            logger.info("All training processes terminated.")
        else:
            logger.info("No active training processes to terminate.")
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {e}")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down immediately...")
    cleanup_training_processes()
    sys.exit(0)


async def _autostart_training(env: str) -> None:
    """Load configuration from the specified environment and start training."""
    from simpletuner.cli import _candidate_config_paths
    from simpletuner.helpers.configuration import env_file
    from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
    from simpletuner.simpletuner_sdk.server.services.training_service import start_training_job, validate_training_config

    store = ConfigStore()

    backend_override = os.environ.get(
        "SIMPLETUNER_CONFIG_BACKEND",
        os.environ.get("CONFIG_BACKEND", os.environ.get("CONFIG_TYPE")),
    )
    config_path_override = os.environ.get("CONFIG_PATH")

    # Respect the WebUI onboarding configs_dir if present, overriding the generic default
    if getattr(store, "config_dir", None):
        resolved_config_dir = str(store.config_dir)
        if os.environ.get("SIMPLETUNER_CONFIG_DIR") != resolved_config_dir:
            os.environ["SIMPLETUNER_CONFIG_DIR"] = resolved_config_dir

    candidate_paths = _candidate_config_paths(env, backend_override, config_path_override)
    config_path = next((path for path in candidate_paths if path.is_file()), None)

    if config_path is None:
        checked = "\n  - ".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(f"Configuration for environment '{env}' not found. Checked:\n  - {checked}")

    backend = (backend_override or "").lower() or None
    if backend not in {"json", "toml", "env"}:
        suffix = config_path.suffix.lower()
        if suffix == ".toml":
            backend = "toml"
        elif suffix == ".env":
            backend = "env"
        else:
            backend = "json"

    def _coerce_cli_keys(config: dict[str, object]) -> dict[str, object]:
        cli_config: dict[str, object] = {}
        for key, value in config.items():
            cli_key = key if str(key).startswith("--") else f"--{key}"
            cli_config[cli_key] = value
        return cli_config

    def _load_config_from_path(path: Path, backend_name: str) -> dict[str, object]:
        if backend_name == "json":
            with path.open("r", encoding="utf-8") as handle:
                raw_data = json.load(handle)

            if isinstance(raw_data, dict):
                if "_metadata" in raw_data and isinstance(raw_data.get("config"), dict):
                    config_section = raw_data.get("config", {})
                    config_dict = config_section if isinstance(config_section, dict) else {}
                    return _coerce_cli_keys(config_dict)

                return _coerce_cli_keys({k: v for k, v in raw_data.items() if k != "_metadata"})

            return {}

        if backend_name == "toml":
            try:
                import toml
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("TOML support is not available; install toml to use --env with toml configs.") from exc

            raw_data = toml.load(path)
            if isinstance(raw_data, dict):
                return _coerce_cli_keys(raw_data)
            return {}

        if backend_name == "env":
            # env configs are already loaded into the environment; map them to CLI args
            mapped: dict[str, object] = {}
            for env_var, arg_name in env_file.env_to_args_map.items():
                value = os.environ.get(env_var)
                if value is None:
                    continue
                cli_key = arg_name if arg_name.startswith("--") else f"--{arg_name}"
                mapped[cli_key] = value
            return mapped

        return {}

    config_data = _load_config_from_path(config_path, backend)
    complete_config = _coerce_cli_keys(config_data)

    # Validate configuration
    validation_result = validate_training_config(store, complete_config, config_data)
    if validation_result.errors:
        error_list = ", ".join(validation_result.errors)
        raise RuntimeError(f"Configuration validation failed: {error_list}")

    # Start the training job
    job_id = start_training_job(complete_config)
    logger.info("Auto-started training job: %s", job_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    logger.info("SimpleTuner server starting up...")

    # Configure third-party loggers after imports
    try:
        from simpletuner.helpers import log_format

        if hasattr(log_format, "configure_third_party_loggers"):
            log_format.configure_third_party_loggers()
    except Exception:
        pass  # Don't fail startup if this fails

    # Initialize SSE manager for heartbeats and connection management
    from simpletuner.simpletuner_sdk.server.services.sse_manager import initialize_sse_manager, shutdown_sse_manager

    await initialize_sse_manager()
    logger.info("SSE manager started")

    # Register signal handlers for immediate shutdown (only works in main thread)
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except ValueError:
        # signal.signal() only works in main thread - ignore in test environments
        logger.debug("Could not register signal handlers (not in main thread)")

    # Auto-start training if --env was specified
    autostart_env = os.environ.get("SIMPLETUNER_SERVER_AUTOSTART_ENV")
    if autostart_env:
        logger.info("Auto-starting training for environment: %s", autostart_env)
        try:
            await _autostart_training(autostart_env)
        except Exception as e:
            logger.error("Failed to auto-start training: %s", e)

    try:
        yield
    except asyncio.CancelledError:
        logger.debug("Application lifespan cancelled; continuing shutdown")
    finally:
        # Shutdown
        logger.info("SimpleTuner server shutting down...")
        await shutdown_sse_manager()
        cleanup_training_processes()


def create_app(
    mode: ServerMode = ServerMode.TRAINER,
    enable_cors: bool = True,
    static_dir: Optional[str] = None,
    template_dir: Optional[str] = None,
    ssl_no_verify: bool = False,
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
    # Force strict model imports to ensure models load properly
    # This must be set BEFORE any model loading happens
    os.environ.setdefault("SIMPLETUNER_STRICT_MODEL_IMPORTS", "1")

    # Create base app with lifespan handler
    title = f"SimpleTuner {mode.value.capitalize()} Server"
    app = FastAPI(title=title, lifespan=lifespan)

    # Store SSL configuration in app state
    app.state.ssl_no_verify = ssl_no_verify

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

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        p = get_static_directory() / "favicon.ico"
        if not p.is_file():
            raise HTTPException(status_code=404, detail="favicon not found")
        return FileResponse(
            p,
            media_type="image/x-icon",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

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

    global WebInterface, ConfigurationClass, TrainingHostClass

    global Configuration, TrainingHost

    if WebInterface is None:
        from simpletuner.simpletuner_sdk.interface import WebInterface as WebInterfaceClass

        WebInterface = WebInterfaceClass
    else:
        WebInterfaceClass = WebInterface

    if Configuration is not None and ConfigurationClass is not Configuration:
        ConfigurationClass = Configuration
    elif ConfigurationClass is None:
        from simpletuner.simpletuner_sdk.configuration import Configuration as _Configuration

        ConfigurationClass = _Configuration
    Configuration = ConfigurationClass

    if TrainingHost is not None and TrainingHostClass is not TrainingHost:
        TrainingHostClass = TrainingHost
    elif TrainingHostClass is None:
        from simpletuner.simpletuner_sdk.training_host import TrainingHost as _TrainingHost

        TrainingHostClass = _TrainingHost
    TrainingHost = TrainingHostClass

    def _include_router_if_present(router: object) -> None:
        if isinstance(router, APIRouter):
            app.include_router(router)

    # Initialize web interface
    web_interface = WebInterfaceClass()
    _include_router_if_present(getattr(web_interface, "router", None))

    # Configuration controller
    config_controller = ConfigurationClass()
    _include_router_if_present(getattr(config_controller, "router", None))

    # Training host controller
    training_host = TrainingHostClass()
    _include_router_if_present(getattr(training_host, "router", None))

    # Add API routes
    from .routes.caption_filters import router as caption_filters_router
    from .routes.checkpoints import router as checkpoints_router
    from .routes.configs import router as configs_router
    from .routes.datasets import router as datasets_router
    from .routes.fields import router as fields_router
    from .routes.git import router as git_router
    from .routes.hardware import router as hardware_router
    from .routes.lycoris import router as lycoris_router
    from .routes.models import router as models_router
    from .routes.prompt_libraries import router as prompt_libraries_router
    from .routes.publishing import router as publishing_router
    from .routes.system import router as system_router
    from .routes.training import router as training_router
    from .routes.validation import router as validation_router
    from .routes.version import router as version_router
    from .routes.web import router as web_router
    from .routes.webui_state import router as webui_state_router

    for router in (
        models_router,
        datasets_router,
        caption_filters_router,
        checkpoints_router,
        configs_router,
        lycoris_router,
        prompt_libraries_router,
        validation_router,
        training_router,
        web_router,
        webui_state_router,
        fields_router,
        publishing_router,
        hardware_router,
        git_router,
        system_router,
        version_router,
    ):
        _include_router_if_present(router)

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
    from .services.callback_service import CallbackService
    from .services.event_store import EventStore

    event_store = EventStore()
    callback_service = CallbackService(event_store)

    # Store event service in app state for access by routes
    app.state.event_store = event_store
    app.state.callback_service = callback_service
    app.state.mode = ServerMode.UNIFIED

    # Configure webhook handler to use direct event store in unified mode
    _configure_unified_webhooks(app)

    return app


def _configure_unified_webhooks(app: FastAPI):
    """Configure webhooks to use direct event store in unified mode."""

    # This will be implemented to intercept webhook calls
    # and write directly to event store when target is localhost
    pass
