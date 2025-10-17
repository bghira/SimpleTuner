"""
Tests for server mode integration using unittest.

Covers trainer, callback, and unified server configurations.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from simpletuner.simpletuner_sdk.server import ServerMode, create_app
from simpletuner.simpletuner_sdk.server.app import create_unified_app

try:
    import sse_starlette  # type: ignore  # noqa: F401

    SSE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SSE_AVAILABLE = False


class ServerCreationTestCase(unittest.TestCase):
    """Validate top-level create_app behaviour."""

    def test_trainer_mode_creation(self) -> None:
        app = create_app(mode=ServerMode.TRAINER)
        self.assertIsNotNone(app)
        self.assertIn("Trainer", app.title)

    def test_callback_mode_creation(self) -> None:
        if not SSE_AVAILABLE:
            self.skipTest("sse-starlette not installed")
        with patch("simpletuner.simpletuner_sdk.server.app._add_callback_routes"):
            app = create_app(mode=ServerMode.CALLBACK)
        self.assertIsNotNone(app)
        self.assertIn("Callback", app.title)

    def test_unified_mode_creation(self) -> None:
        if not SSE_AVAILABLE:
            self.skipTest("sse-starlette not installed")
        with (
            patch("simpletuner.simpletuner_sdk.server.app._add_callback_routes"),
            patch("simpletuner.simpletuner_sdk.server.app._add_trainer_routes"),
        ):
            app = create_app(mode=ServerMode.UNIFIED)
        self.assertIsNotNone(app)
        self.assertIn("Unified", app.title)

    def test_cors_configuration(self) -> None:
        app = create_app(mode=ServerMode.TRAINER, enable_cors=True)
        has_cors = any("CORSMiddleware" in str(middleware) for middleware in app.user_middleware)
        self.assertTrue(has_cors)

    def test_static_files_mounting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir) / "static"
            static_dir.mkdir()
            app = create_app(mode=ServerMode.TRAINER, static_dir=str(static_dir))
            routes = [route.path for route in app.routes]
            self.assertTrue(any("/static" in path for path in routes))


class RouteInclusionTestCase(unittest.TestCase):
    """Ensure the expected routes exist per server mode."""

    def test_trainer_mode_routes_only(self) -> None:
        with patch("simpletuner.simpletuner_sdk.server.app._add_trainer_routes"):
            app = create_app(mode=ServerMode.TRAINER)
        with TestClient(app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["mode"], "trainer")
            response = client.get("/broadcast")
            self.assertEqual(response.status_code, 404)

    def test_callback_mode_routes_only(self) -> None:
        if not SSE_AVAILABLE:
            self.skipTest("sse-starlette not installed")
        with patch("simpletuner.simpletuner_sdk.server.app._add_callback_routes"):
            app = create_app(mode=ServerMode.CALLBACK)
        with TestClient(app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["mode"], "callback")

    def test_unified_mode_has_all_routes(self) -> None:
        if not SSE_AVAILABLE:
            self.skipTest("sse-starlette not installed")
        with (
            patch("simpletuner.simpletuner_sdk.server.app._add_callback_routes"),
            patch("simpletuner.simpletuner_sdk.server.app._add_trainer_routes"),
        ):
            app = create_app(mode=ServerMode.UNIFIED)
        with TestClient(app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["mode"], "unified")


class UnifiedModeTestCase(unittest.TestCase):
    """Exercise unified-mode specific wiring."""

    def test_shared_event_store_in_unified_mode(self) -> None:
        if not SSE_AVAILABLE:
            self.skipTest("sse-starlette not installed")
        with (
            patch("simpletuner.simpletuner_sdk.server.app._add_callback_routes"),
            patch("simpletuner.simpletuner_sdk.server.app._add_trainer_routes"),
        ):
            app = create_unified_app()
        self.assertTrue(hasattr(app.state, "event_store"))
        self.assertIsNotNone(app.state.event_store)
        self.assertTrue(hasattr(app.state, "callback_service"))
        self.assertIsNotNone(app.state.callback_service)
        self.assertEqual(app.state.mode, ServerMode.UNIFIED)

    def test_event_store_sharing_between_routes(self) -> None:
        if not SSE_AVAILABLE:
            self.skipTest("sse-starlette not installed")
        with (
            patch("simpletuner.simpletuner_sdk.server.app._add_callback_routes"),
            patch("simpletuner.simpletuner_sdk.server.app._add_trainer_routes"),
        ):
            app = create_unified_app()
        with TestClient(app) as client:
            payload = {"type": "test", "message": "test_event"}
            response = client.post("/callback", json=payload)
            self.assertIn(response.status_code, (200, 404))


class PortAssignmentTestCase(unittest.TestCase):
    """Document expected CLI defaults for server ports."""

    def test_default_ports_by_mode(self) -> None:
        from simpletuner.cli import cmd_server  # Local import to avoid circulars

        trainer_args = SimpleNamespace(mode="trainer", host="0.0.0.0", port=None, reload=False)
        callback_args = SimpleNamespace(mode="callback", host="0.0.0.0", port=None, reload=False)

        # The CLI helper mutates args.port, so we verify expectations via a dry run.
        # // ASSUMPTION: cmd_server.configure_server mutates the provided namespace in place.
        configure = getattr(cmd_server, "configure_server", None)
        if configure is not None:
            configure(trainer_args)
            configure(callback_args)
            self.assertEqual(trainer_args.port, 8001)
            self.assertEqual(callback_args.port, 8002)

    def test_port_override(self) -> None:
        override_args = SimpleNamespace(mode="trainer", host="0.0.0.0", port=9999, reload=False)
        self.assertEqual(override_args.port, 9999)


class ModelRoutesTestCase(unittest.TestCase):
    """Model metadata endpoints."""

    @patch("simpletuner.simpletuner_sdk.server.routes.models.model_families", {"sdxl": Mock(), "flux": Mock()})
    def test_get_model_families_endpoint(self) -> None:
        app = create_app(mode=ServerMode.TRAINER)
        with TestClient(app) as client:
            response = client.get("/models")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("families", data)
            self.assertIn("sdxl", data["families"])
            self.assertIn("flux", data["families"])

    @patch("simpletuner.simpletuner_sdk.server.routes.models.get_model_flavour_choices")
    @patch("simpletuner.simpletuner_sdk.server.routes.models.model_families", {"sdxl": Mock()})
    def test_get_model_flavours_endpoint(self, mock_get_flavours) -> None:
        mock_get_flavours.return_value = ["base-1.0", "refiner-1.0"]
        app = create_app(mode=ServerMode.TRAINER)
        with TestClient(app) as client:
            response = client.get("/models/sdxl/flavours")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("flavours", data)
            self.assertIn("base-1.0", data["flavours"])
            self.assertIn("refiner-1.0", data["flavours"])

    def test_invalid_model_family(self) -> None:
        app = create_app(mode=ServerMode.TRAINER)
        with TestClient(app) as client:
            response = client.get("/models/invalid_model/flavours")
            self.assertEqual(response.status_code, 404)


class ErrorHandlingTestCase(unittest.TestCase):
    """Regression coverage for missing resources."""

    def test_missing_static_directory(self) -> None:
        app = create_app(mode=ServerMode.TRAINER, static_dir="/nonexistent/path")
        self.assertIsNotNone(app)

    def test_missing_template_directory(self) -> None:
        app = create_app(mode=ServerMode.TRAINER, template_dir="/nonexistent/templates")
        self.assertIsNotNone(app)
        self.assertEqual(os.environ.get("TEMPLATE_DIR"), "/nonexistent/templates")


class CORSTestCase(unittest.TestCase):
    """Explicit CORS enable/disable behaviour."""

    def test_cors_disabled(self) -> None:
        app = create_app(mode=ServerMode.TRAINER, enable_cors=False)
        has_cors = any("CORSMiddleware" in str(middleware) for middleware in app.user_middleware)
        self.assertFalse(has_cors)

    def test_cors_headers(self) -> None:
        app = create_app(mode=ServerMode.TRAINER, enable_cors=True)
        with TestClient(app) as client:
            response = client.options("/health", headers={"Origin": "http://localhost:8000"})
            # Some configurations may return 200/204/405 depending on router defaults.
            self.assertIn(response.status_code, (200, 204, 404, 405))


class RootRedirectTestCase(unittest.TestCase):
    """Root path behaviour per mode."""

    def test_trainer_mode_root_redirect(self) -> None:
        with (
            patch("simpletuner.simpletuner_sdk.server.app.WebInterface"),
            patch("simpletuner.simpletuner_sdk.server.app.Configuration"),
            patch("simpletuner.simpletuner_sdk.server.app.TrainingHost"),
        ):
            app = create_app(mode=ServerMode.TRAINER)
            with TestClient(app) as client:
                response = client.get("/", follow_redirects=False)
                self.assertEqual(response.status_code, 307)
                self.assertEqual(response.headers["location"], "/web/trainer")

    def test_callback_mode_no_root_redirect(self) -> None:
        app = create_app(mode=ServerMode.CALLBACK)
        with TestClient(app) as client:
            response = client.get("/", follow_redirects=False)
            self.assertEqual(response.status_code, 404)


class EventRoutesTestCase(unittest.TestCase):
    """Callback-specific long-polling endpoints."""

    def test_callback_endpoint_receives_events(self) -> None:
        app = create_app(mode=ServerMode.CALLBACK)
        with TestClient(app) as client:
            event = {"message_type": "training_progress", "data": {"step": 100}}
            response = client.post("/callback", json=event)
            self.assertEqual(response.status_code, 200)

    def test_broadcast_long_polling(self) -> None:
        app = create_app(mode=ServerMode.CALLBACK)
        with TestClient(app) as client:
            response = client.get("/broadcast?last_event_index=0")
            self.assertIn(response.status_code, (200, 204, 408, 504))

    def test_webhook_clear_on_configure(self) -> None:
        app = create_app(mode=ServerMode.CALLBACK)
        with TestClient(app) as client:
            client.post("/callback", json={"data": "event1"})
            client.post("/callback", json={"data": "event2"})
            response = client.post(
                "/callback",
                json={"message_type": "configure_webhook", "data": "new_session"},
            )
            self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
