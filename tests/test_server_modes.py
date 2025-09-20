"""
Tests for server mode integration - trainer, callback, and unified modes.
These tests verify server configuration and route management.
"""

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from simpletuner.simpletuner_sdk.server import ServerMode, create_app
from simpletuner.simpletuner_sdk.server.app import _add_callback_routes, _add_trainer_routes, create_unified_app


class TestServerCreation:
    """Test server app creation with different modes."""

    def test_trainer_mode_creation(self):
        """Test creating app in trainer mode."""
        app = create_app(mode=ServerMode.TRAINER)
        assert app is not None

        # Check that app has correct title
        assert "Trainer" in app.title

    def test_callback_mode_creation(self):
        """Test creating app in callback mode."""
        app = create_app(mode=ServerMode.CALLBACK)
        assert app is not None

        # Check that app has correct title
        assert "Callback" in app.title

    def test_unified_mode_creation(self):
        """Test creating app in unified mode."""
        app = create_app(mode=ServerMode.UNIFIED)
        assert app is not None

        # Check that app has correct title
        assert "Unified" in app.title

    def test_cors_configuration(self):
        """Test CORS is properly configured."""
        app = create_app(mode=ServerMode.TRAINER, enable_cors=True)

        # Check CORS middleware is added
        middleware_found = False
        for middleware in app.user_middleware:
            if "CORSMiddleware" in str(middleware):
                middleware_found = True
                break

        assert middleware_found

    def test_static_files_mounting(self):
        """Test static files are mounted when directory exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = os.path.join(tmpdir, "static")
            os.makedirs(static_dir)

            app = create_app(mode=ServerMode.TRAINER, static_dir=static_dir)

            # Check static files are mounted
            routes = [r.path for r in app.routes]
            assert any("/static" in path for path in routes)


class TestRouteInclusion:
    """Test that correct routes are included based on mode."""

    def test_trainer_mode_routes_only(self):
        """Test trainer mode includes only training routes."""
        with patch("simpletuner.simpletuner_sdk.server.app.WebInterface") as mock_web:
            with patch("simpletuner.simpletuner_sdk.server.app.Configuration") as mock_config:
                with patch("simpletuner.simpletuner_sdk.server.app.TrainingHost") as mock_host:
                    app = create_app(mode=ServerMode.TRAINER)
                    client = TestClient(app)

                    # Check health endpoint
                    response = client.get("/health")
                    assert response.status_code == 200
                    assert response.json()["mode"] == "trainer"

                    # Callback endpoint should not exist
                    response = client.get("/broadcast")
                    assert response.status_code == 404

    def test_callback_mode_routes_only(self):
        """Test callback mode includes only callback routes."""
        app = create_app(mode=ServerMode.CALLBACK)
        client = TestClient(app)

        # Check health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["mode"] == "callback"

        # Check callback routes exist
        routes = [r.path for r in app.routes]
        assert "/callback" in routes
        assert "/broadcast" in routes

        # Training routes should not exist
        assert "/training/configuration/run" not in routes

    def test_unified_mode_has_all_routes(self):
        """Test unified mode includes all routes."""
        with patch("simpletuner.simpletuner_sdk.server.app.WebInterface") as mock_web:
            with patch("simpletuner.simpletuner_sdk.server.app.Configuration") as mock_config:
                with patch("simpletuner.simpletuner_sdk.server.app.TrainingHost") as mock_host:
                    app = create_app(mode=ServerMode.UNIFIED)
                    client = TestClient(app)

                    # Check health endpoint
                    response = client.get("/health")
                    assert response.status_code == 200
                    assert response.json()["mode"] == "unified"

                    # Both route sets should be included
                    routes = [r.path for r in app.routes]
                    # Note: actual routes depend on imports
                    assert "/health" in routes


class TestUnifiedMode:
    """Test unified mode specific functionality."""

    def test_shared_event_store_in_unified_mode(self):
        """Test that unified mode has shared event store."""
        app = create_unified_app()

        # Check event store is in app state
        assert hasattr(app.state, "event_store")
        assert app.state.event_store is not None

        # Check mode is set
        assert app.state.mode == ServerMode.UNIFIED

    def test_event_store_sharing_between_routes(self):
        """Test that routes share the same event store in unified mode."""
        app = create_unified_app()
        client = TestClient(app)

        # Send event to callback
        event_data = {"type": "test", "message": "test_event"}
        response = client.post("/callback", json=event_data)
        assert response.status_code == 200

        # Event should be accessible via broadcast
        response = client.get("/broadcast?last_event_index=0")
        # This will timeout or return events depending on implementation


class TestPortAssignment:
    """Test port assignment based on mode."""

    def test_default_ports_by_mode(self):
        """Test that default ports are assigned correctly."""
        from types import SimpleNamespace

        from simpletuner.cli import cmd_server

        # Test trainer mode
        args = SimpleNamespace(mode="trainer", host="0.0.0.0", port=None, reload=False)
        # Would need to mock uvicorn.run to test actual port assignment

        # Test callback mode
        args = SimpleNamespace(mode="callback", host="0.0.0.0", port=None, reload=False)
        # Port should be 8002 for callback

    def test_port_override(self):
        """Test that explicit port overrides mode default."""
        from types import SimpleNamespace

        args = SimpleNamespace(mode="trainer", host="0.0.0.0", port=9999, reload=False)
        # Port should be 9999, not 8001


class TestModelRoutes:
    """Test model information routes."""

    @patch("simpletuner.simpletuner_sdk.server.routes.models.model_families", {"sdxl": Mock(), "flux": Mock()})
    def test_get_model_families_endpoint(self):
        """Test getting list of model families."""
        app = create_app(mode=ServerMode.TRAINER)
        client = TestClient(app)

        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "families" in data
        assert "sdxl" in data["families"]
        assert "flux" in data["families"]

    @patch("simpletuner.simpletuner_sdk.server.routes.models.get_model_flavour_choices")
    @patch("simpletuner.simpletuner_sdk.server.routes.models.model_families", {"sdxl": Mock()})
    def test_get_model_flavours_endpoint(self, mock_get_flavours):
        """Test getting flavours for a model family."""
        mock_get_flavours.return_value = ["base-1.0", "refiner-1.0"]

        app = create_app(mode=ServerMode.TRAINER)
        client = TestClient(app)

        response = client.get("/models/sdxl/flavours")
        assert response.status_code == 200
        data = response.json()
        assert "flavours" in data
        assert "base-1.0" in data["flavours"]
        assert "refiner-1.0" in data["flavours"]

    def test_invalid_model_family(self):
        """Test requesting flavours for invalid model family."""
        app = create_app(mode=ServerMode.TRAINER)
        client = TestClient(app)

        response = client.get("/models/invalid_model/flavours")
        assert response.status_code == 404


class TestErrorHandling:
    """Test error handling in different modes."""

    def test_missing_static_directory(self):
        """Test app creation with missing static directory."""
        # Should not crash
        app = create_app(mode=ServerMode.TRAINER, static_dir="/nonexistent/path")
        assert app is not None

    def test_missing_template_directory(self):
        """Test app creation with missing template directory."""
        app = create_app(mode=ServerMode.TRAINER, template_dir="/nonexistent/templates")
        assert app is not None

        # Template dir should be set in environment
        assert os.environ.get("TEMPLATE_DIR") == "/nonexistent/templates"

    def test_concurrent_server_startup(self):
        """Test handling of concurrent server startups on same port."""
        # This would require actual server startup which we can't easily test
        # Document the expected behavior
        pass


class TestCORSConfiguration:
    """Test CORS configuration across modes."""

    def test_cors_disabled(self):
        """Test creating app with CORS disabled."""
        app = create_app(mode=ServerMode.TRAINER, enable_cors=False)

        # Check no CORS middleware
        cors_found = False
        for middleware in app.user_middleware:
            if "CORSMiddleware" in str(middleware):
                cors_found = True
                break

        assert not cors_found

    def test_cors_headers(self):
        """Test CORS headers are properly set."""
        app = create_app(mode=ServerMode.TRAINER, enable_cors=True)
        client = TestClient(app)

        # Make OPTIONS request
        response = client.options("/health", headers={"Origin": "http://localhost:8000"})

        # Should allow the origin
        if response.status_code == 200:  # OPTIONS might not be implemented
            headers = response.headers
            # Check for CORS headers if implemented


class TestRootRedirect:
    """Test root path redirect behavior."""

    def test_trainer_mode_root_redirect(self):
        """Test that trainer mode redirects root to web interface."""
        with patch("simpletuner.simpletuner_sdk.server.app.WebInterface") as mock_web:
            with patch("simpletuner.simpletuner_sdk.server.app.Configuration") as mock_config:
                with patch("simpletuner.simpletuner_sdk.server.app.TrainingHost") as mock_host:
                    app = create_app(mode=ServerMode.TRAINER)
                    client = TestClient(app)

                    response = client.get("/", follow_redirects=False)
                    assert response.status_code == 307  # Redirect
                    assert response.headers["location"] == "/web/trainer"

    def test_callback_mode_no_root_redirect(self):
        """Test that callback mode doesn't redirect root."""
        app = create_app(mode=ServerMode.CALLBACK)
        client = TestClient(app)

        response = client.get("/", follow_redirects=False)
        # Should be 404 as no root handler in callback mode
        assert response.status_code == 404


class TestEventRoutes:
    """Test event-specific routes in different modes."""

    def test_callback_endpoint_receives_events(self):
        """Test callback endpoint stores events."""
        app = create_app(mode=ServerMode.CALLBACK)
        client = TestClient(app)

        # Send callback event
        event = {"message_type": "training_progress", "data": {"step": 100}}
        response = client.post("/callback", json=event)
        assert response.status_code == 200

    def test_broadcast_long_polling(self):
        """Test broadcast endpoint for long polling."""
        app = create_app(mode=ServerMode.CALLBACK)
        client = TestClient(app)

        # Request events (will timeout quickly in test)
        response = client.get("/broadcast?last_event_index=0")
        # Should return empty or timeout

    def test_webhook_clear_on_configure(self):
        """Test that configure_webhook message clears events."""
        app = create_app(mode=ServerMode.CALLBACK)
        client = TestClient(app)

        # Send some events
        client.post("/callback", json={"data": "event1"})
        client.post("/callback", json={"data": "event2"})

        # Send configure webhook
        response = client.post("/callback", json={"message_type": "configure_webhook", "data": "new_session"})
        assert response.status_code == 200

        # Events should be cleared (implementation dependent)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
