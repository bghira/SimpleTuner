"""Example FastAPI app configuration showing security improvements.

This file demonstrates how to integrate all the security and performance
improvements into a FastAPI application.
"""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Import middleware and security utilities
from .middleware.security_middleware import setup_security_middleware
from .routes import web, events, configs, datasets, validation, training, fields, models
from .routes.debug_router import router as debug_router

# Import caching services
from .services.cache_service import get_config_cache
from .services.field_registry_wrapper import lazy_field_registry


def create_app() -> FastAPI:
    """Create and configure the FastAPI application with security improvements."""

    # Create FastAPI app
    app = FastAPI(
        title="SimpleTuner WebUI",
        version="2.0.0",
        description="Enhanced Web UI for SimpleTuner with security improvements",
        docs_url="/api/docs" if os.getenv("DEBUG_MODE", "false").lower() == "true" else None,
        redoc_url="/api/redoc" if os.getenv("DEBUG_MODE", "false").lower() == "true" else None,
    )

    # Setup security middleware (CORS, headers, rate limiting)
    setup_security_middleware(app)

    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # Include routers
    app.include_router(web.router)
    app.include_router(events.router)
    app.include_router(configs.router)
    app.include_router(datasets.router)
    app.include_router(validation.router)
    app.include_router(training.router)
    app.include_router(fields.router)
    app.include_router(models.router)

    # Only include debug routes if DEBUG_MODE is enabled
    if os.getenv("DEBUG_MODE", "false").lower() == "true":
        app.include_router(debug_router)

    # Configure caching
    cache_ttl = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes default
    lazy_field_registry.set_cache_ttl(cache_ttl)

    # Startup event
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        # Warm up caches
        for tab in ["basic", "model", "training", "advanced", "validation"]:
            lazy_field_registry.get_fields_for_tab(tab)

        # Log security configuration
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Security middleware configured")
        logger.info(f"Debug mode: {os.getenv('DEBUG_MODE', 'false')}")
        logger.info(f"Production mode: {os.getenv('PRODUCTION', 'false')}")
        logger.info(f"Cache TTL: {cache_ttl} seconds")

    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        # Clear caches
        lazy_field_registry.clear_cache()
        get_config_cache().clear()

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": "2.0.0"}

    return app


# Example environment variables for security configuration:
#
# # Logging
# LOG_LEVEL=INFO
#
# # Debug mode (enables debug routes and API docs)
# DEBUG_MODE=false
#
# # Production mode (enforces stricter security)
# PRODUCTION=true
#
# # CORS configuration
# ALLOWED_ORIGINS=https://example.com,https://app.example.com
#
# # Rate limiting
# RATE_LIMIT_CALLS=100
# RATE_LIMIT_PERIOD=60
#
# # Cache configuration
# CACHE_TTL=300
#
# # Template directory
# TEMPLATE_DIR=/path/to/templates


if __name__ == "__main__":
    # Example of running with uvicorn
    import uvicorn

    app = create_app()

    # Run with security best practices
    uvicorn.run(
        app,
        host="127.0.0.1",  # Only bind to localhost in development
        port=8000,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        # In production, use a reverse proxy (nginx) and set:
        # proxy_headers=True,
        # forwarded_allow_ips="*",
    )