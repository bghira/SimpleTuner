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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .utils.paths import get_avatars_directory, get_simpletuner_root, get_static_directory, get_template_directory

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


def _coerce_cli_keys(config: dict[str, object]) -> dict[str, object]:
    """Ensure all config keys have CLI-style '--' prefix.

    Args:
        config: Configuration dictionary with potentially unprefixed keys

    Returns:
        New dictionary with all keys prefixed with '--'
    """
    cli_config: dict[str, object] = {}
    for key, value in config.items():
        cli_key = key if str(key).startswith("--") else f"--{key}"
        cli_config[cli_key] = value
    return cli_config


def _load_json_config(path: Path) -> dict[str, object]:
    """Load and parse a JSON configuration file.

    Args:
        path: Path to the JSON config file

    Returns:
        Configuration dictionary with CLI-style keys
    """
    with path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    if not isinstance(raw_data, dict):
        return {}

    # Handle WebUI-style configs with _metadata wrapper
    if "_metadata" in raw_data and isinstance(raw_data.get("config"), dict):
        config_section = raw_data.get("config", {})
        config_dict = config_section if isinstance(config_section, dict) else {}
        return _coerce_cli_keys(config_dict)

    return _coerce_cli_keys({k: v for k, v in raw_data.items() if k != "_metadata"})


def _load_toml_config(path: Path) -> dict[str, object]:
    """Load and parse a TOML configuration file.

    Args:
        path: Path to the TOML config file

    Returns:
        Configuration dictionary with CLI-style keys

    Raises:
        RuntimeError: If toml package is not installed
    """
    try:
        import toml
    except ImportError as exc:
        raise RuntimeError("TOML support is not available; install toml to use --env with toml configs.") from exc

    raw_data = toml.load(path)
    if isinstance(raw_data, dict):
        return _coerce_cli_keys(raw_data)
    return {}


def _load_env_config() -> dict[str, object]:
    """Load configuration from environment variables.

    Returns:
        Configuration dictionary with CLI-style keys mapped from env vars
    """
    from simpletuner.helpers.configuration import env_file

    mapped: dict[str, object] = {}
    for env_var, arg_name in env_file.env_to_args_map.items():
        value = os.environ.get(env_var)
        if value is None:
            continue
        cli_key = arg_name if arg_name.startswith("--") else f"--{arg_name}"
        mapped[cli_key] = value
    return mapped


def _load_config_from_path(path: Path, backend: str) -> dict[str, object]:
    """Load configuration from a file based on backend type.

    Args:
        path: Path to the configuration file
        backend: Backend type ('json', 'toml', or 'env')

    Returns:
        Configuration dictionary with CLI-style keys
    """
    if backend == "json":
        return _load_json_config(path)
    if backend == "toml":
        return _load_toml_config(path)
    if backend == "env":
        return _load_env_config()
    return {}


def _resolve_config_path(env: str) -> tuple[Path, str]:
    """Resolve configuration file path and backend type.

    Args:
        env: Environment name to load configuration for

    Returns:
        Tuple of (config_path, backend_name)

    Raises:
        FileNotFoundError: If no configuration file is found
    """
    from simpletuner.cli import _candidate_config_paths
    from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore

    store = ConfigStore()

    backend_override = os.environ.get(
        "SIMPLETUNER_CONFIG_BACKEND",
        os.environ.get("CONFIG_BACKEND", os.environ.get("CONFIG_TYPE")),
    )
    config_path_override = os.environ.get("CONFIG_PATH")

    # Respect the WebUI onboarding configs_dir if present
    if getattr(store, "config_dir", None):
        resolved_config_dir = str(store.config_dir)
        if os.environ.get("SIMPLETUNER_CONFIG_DIR") != resolved_config_dir:
            os.environ["SIMPLETUNER_CONFIG_DIR"] = resolved_config_dir

    candidate_paths = _candidate_config_paths(env, backend_override, config_path_override)
    config_path = next((path for path in candidate_paths if path.is_file()), None)

    if config_path is None:
        checked = "\n  - ".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(f"Configuration for environment '{env}' not found. Checked:\n  - {checked}")

    # Detect backend from override or file extension
    backend = (backend_override or "").lower() or None
    if backend not in {"json", "toml", "env"}:
        suffix = config_path.suffix.lower()
        if suffix == ".toml":
            backend = "toml"
        elif suffix == ".env":
            backend = "env"
        else:
            backend = "json"

    return config_path, backend


async def _autostart_training(env: str) -> None:
    """Load configuration from the specified environment and start training.

    Args:
        env: Environment name to load configuration for

    Raises:
        FileNotFoundError: If configuration file is not found
        RuntimeError: If configuration validation fails
    """
    from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
    from simpletuner.simpletuner_sdk.server.services.training_service import start_training_job, validate_training_config

    config_path, backend = _resolve_config_path(env)
    config_data = _load_config_from_path(config_path, backend)
    complete_config = _coerce_cli_keys(config_data)

    store = ConfigStore()
    validation_result = validate_training_config(store, complete_config, config_data)
    if validation_result.errors:
        error_list = ", ".join(validation_result.errors)
        raise RuntimeError(f"Configuration validation failed: {error_list}")

    job_id = start_training_job(complete_config)
    logger.info("Auto-started training job: %s", job_id)


def _get_retention_config() -> tuple[int, int]:
    """Get retention configuration from environment variables.

    Returns:
        Tuple of (job_retention_days, audit_retention_days)

    Environment variables:
        SIMPLETUNER_JOB_RETENTION_DAYS: Days to retain completed jobs (default: 90)
        SIMPLETUNER_AUDIT_RETENTION_DAYS: Days to retain audit logs (default: 90)

    Enterprise compliance note: Set these to higher values (e.g., 365, 730, or 0 to disable)
    if your organization requires longer retention periods.
    """
    job_retention = int(os.environ.get("SIMPLETUNER_JOB_RETENTION_DAYS", "90"))
    audit_retention = int(os.environ.get("SIMPLETUNER_AUDIT_RETENTION_DAYS", "90"))
    return job_retention, audit_retention


async def _periodic_cleanup_task():
    """
    Background task that periodically cleans up old jobs and audit logs.

    Runs daily and removes:
    - Jobs older than SIMPLETUNER_JOB_RETENTION_DAYS (default 90)
    - Audit log entries older than SIMPLETUNER_AUDIT_RETENTION_DAYS (default 90)
    - Stale upload progress files older than 60 minutes

    Set retention days to 0 to disable cleanup for that category.
    """
    # Wait 1 hour before first cleanup (let server fully start)
    await asyncio.sleep(3600)

    job_retention_days, audit_retention_days = _get_retention_config()
    logger.info(
        "Cleanup task started: job_retention=%d days, audit_retention=%d days",
        job_retention_days,
        audit_retention_days,
    )

    while True:
        try:
            from simpletuner.simpletuner_sdk.server.services.cloud.container import get_job_store

            store = get_job_store()

            # Allow disabling cleanup by setting retention to 0
            jobs_removed = 0
            if job_retention_days > 0:
                jobs_removed = await store.cleanup_old_jobs(retention_days=job_retention_days)

            audit_removed = 0
            if audit_retention_days > 0:
                audit_removed = await store.cleanup_audit_log(max_age_days=audit_retention_days)

            upload_removed = store.cleanup_stale_upload_progress(max_age_minutes=60)

            if jobs_removed > 0 or audit_removed > 0 or upload_removed > 0:
                logger.info(
                    "Cleanup completed: %d jobs removed, %d audit entries removed, %d stale uploads removed",
                    jobs_removed,
                    audit_removed,
                    upload_removed,
                )
        except Exception as exc:
            logger.warning("Periodic cleanup failed: %s", exc)

        # Sleep for 24 hours before next cleanup
        await asyncio.sleep(86400)


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

    # Reconcile orphaned local jobs from previous run and process pending jobs
    try:
        from simpletuner.simpletuner_sdk.server.services.local_gpu_allocator import get_gpu_allocator

        allocator = get_gpu_allocator()

        # Check for orphaned jobs from a previous server run
        reconcile_stats = await allocator.reconcile_on_startup()
        if reconcile_stats["orphaned"] > 0 or reconcile_stats["no_pid"] > 0:
            logger.warning(
                "Reconciled orphaned local jobs: %d dead, %d adopted, %d no-pid",
                reconcile_stats["orphaned"],
                reconcile_stats["adopted"],
                reconcile_stats["no_pid"],
            )
        elif reconcile_stats["adopted"] > 0:
            logger.info(
                "Adopted %d running local jobs from previous server run",
                reconcile_stats["adopted"],
            )

        started = await allocator.process_pending_jobs()
        if started:
            logger.info("Started %d pending local jobs on startup: %s", len(started), started)
    except Exception as e:
        logger.warning("Failed to process pending jobs on startup: %s", e)

    # Auto-start training if --env was specified
    autostart_env = os.environ.get("SIMPLETUNER_SERVER_AUTOSTART_ENV")
    if autostart_env:
        logger.info("Auto-starting training for environment: %s", autostart_env)
        try:
            await _autostart_training(autostart_env)
        except Exception as e:
            logger.error("Failed to auto-start training: %s", e)

    # Start background cleanup task for cloud jobs
    cleanup_task = asyncio.create_task(_periodic_cleanup_task())

    # Start background job polling (auto-syncs job statuses without webhooks)
    try:
        from simpletuner.simpletuner_sdk.server.services.cloud.background_tasks import (
            start_background_tasks,
            stop_background_tasks,
        )

        await start_background_tasks()
        logger.info("Background task manager started")
    except Exception as e:
        logger.warning("Failed to start background tasks: %s", e)
        stop_background_tasks = None

    # Configure rate limiters from provider config
    try:
        from simpletuner.simpletuner_sdk.server.routes.cloud._shared import configure_rate_limits_from_provider

        await configure_rate_limits_from_provider()
        logger.debug("Rate limiters configured from provider config")
    except Exception as e:
        logger.debug("Failed to configure rate limiters: %s", e)

    try:
        yield
    except asyncio.CancelledError:
        logger.debug("Application lifespan cancelled; continuing shutdown")
    finally:
        # Shutdown
        logger.info("SimpleTuner server shutting down...")

        # Stop background tasks
        if stop_background_tasks:
            try:
                await stop_background_tasks()
            except Exception as e:
                logger.warning("Error stopping background tasks: %s", e)

        # Cancel cleanup task
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

        # Close state backend (flushes pending writes, closes connections)
        try:
            from simpletuner.simpletuner_sdk.server.services.cloud.container import close_state_backend

            await close_state_backend()
            logger.debug("State backend closed")
        except Exception as e:
            logger.debug("Error closing state backend: %s", e)

        # Close all AsyncSQLiteStore instances (aiosqlite connections)
        try:
            from simpletuner.simpletuner_sdk.server.services.cloud.storage.async_base import AsyncSQLiteStore

            await AsyncSQLiteStore.close_all_instances()
            logger.debug("AsyncSQLiteStore instances closed")
        except Exception as e:
            logger.debug("Error closing AsyncSQLiteStore instances: %s", e)

        # Shutdown thread_keeper executor
        try:
            from simpletuner.simpletuner_sdk import thread_keeper

            if hasattr(thread_keeper, "executor"):
                logger.debug("Shutting down thread_keeper executor")
                thread_keeper.executor.shutdown(wait=True, cancel_futures=True)
        except Exception as e:
            logger.debug("Error shutting down thread_keeper executor: %s", e)

        # Shutdown HuggingFace service executor
        try:
            from simpletuner.simpletuner_sdk.server.services.huggingface_service import HUGGINGFACE_SERVICE

            HUGGINGFACE_SERVICE.shutdown()
            logger.debug("HuggingFace service executor shut down")
        except Exception as e:
            logger.debug("Error shutting down HuggingFace service: %s", e)

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

    # Configure security middleware (CORS, rate limiting, security headers)
    if enable_cors:
        from .middleware import setup_security_middleware

        setup_security_middleware(app)

    # Mount avatars directory FIRST (more specific path must come before /static)
    # User-uploaded content stored in SimpleTuner home dir, not package static dir
    avatars_dir = get_avatars_directory()
    app.mount("/static/avatars", StaticFiles(directory=str(avatars_dir)), name="avatars")

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

    # Serve UI sounds from freedesktop sound themes
    SOUND_THEME_PATHS = [
        Path("/usr/share/sounds/ocean/stereo"),
        Path("/usr/share/sounds/freedesktop/stereo"),
        Path("/usr/share/sounds/gnome/default/alerts"),
    ]
    SOUND_FILE_MAP = {
        "completion-success.oga": "completion-success.oga",
        "dialog-error.oga": "dialog-error.oga",
        "dialog-warning.oga": "dialog-warning.oga",
        "dialog-information.oga": "dialog-information.oga",
    }

    @app.get("/api/sounds/{filename}", include_in_schema=False)
    def serve_sound(filename: str):
        # Handle retro hover sound easter egg (Norton SystemWorks 2006 tribute)
        if filename == "retro-hover.wav":
            retro_path = get_static_directory() / "sounds" / "retro-hover.wav"
            if retro_path.is_file():
                return FileResponse(
                    retro_path,
                    media_type="audio/wav",
                    headers={"Cache-Control": "public, max-age=86400"},
                )
            raise HTTPException(status_code=404, detail="Retro sound not available")

        if filename not in SOUND_FILE_MAP:
            raise HTTPException(status_code=404, detail="Sound not found")
        mapped_name = SOUND_FILE_MAP[filename]
        for theme_path in SOUND_THEME_PATHS:
            sound_file = theme_path / mapped_name
            if sound_file.is_file():
                return FileResponse(
                    sound_file,
                    media_type="audio/ogg",
                    headers={"Cache-Control": "public, max-age=86400"},
                )
        raise HTTPException(status_code=404, detail="Sound file not available on this system")

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
    from .routes import global_router
    from .routes.caption_filters import router as caption_filters_router
    from .routes.checkpoints import router as checkpoints_router
    from .routes.cloud import router as cloud_router
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
        global_router,
        models_router,
        datasets_router,
        caption_filters_router,
        checkpoints_router,
        cloud_router,
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

    # Register cloud exception handlers for automatic CloudError -> HTTP response conversion
    from .services.cloud.exceptions import register_exception_handlers

    register_exception_handlers(app)

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

    return app
