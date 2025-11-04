"""
Integration test for server startup.

This test does NOT mock out route addition - it verifies the server
can actually start and models load properly.
"""

from __future__ import annotations

import os
import unittest

from fastapi.testclient import TestClient

from simpletuner.simpletuner_sdk.server import ServerMode, create_app


class ServerStartupIntegrationTest(unittest.TestCase):
    """Verify the server can actually start without mocking critical paths.

    IMPORTANT: These tests do NOT set SIMPLETUNER_STRICT_MODEL_IMPORTS.
    They verify that create_app() works out of the box, which ensures
    the server startup code in cli.py properly enables strict mode.
    """

    def test_trainer_mode_starts_with_models(self) -> None:
        """Verify trainer mode can start and models load."""
        app = create_app(mode=ServerMode.TRAINER)
        self.assertIsNotNone(app)

        with TestClient(app) as client:
            # Verify health endpoint
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)

            # Verify models endpoint has actual models
            response = client.get("/models")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("families", data)
            # Should have at least some model families loaded
            self.assertGreater(len(data["families"]), 0, "No model families loaded - check SIMPLETUNER_STRICT_MODEL_IMPORTS")

    def test_unified_mode_starts_with_models(self) -> None:
        """Verify unified mode can start and models load."""
        try:
            import sse_starlette  # noqa: F401
        except ImportError:
            self.skipTest("sse-starlette not installed")

        app = create_app(mode=ServerMode.UNIFIED)
        self.assertIsNotNone(app)

        with TestClient(app) as client:
            # Verify health endpoint
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["mode"], "unified")

            # Verify models endpoint has actual models
            response = client.get("/models")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertGreater(len(data["families"]), 0, "No model families loaded in unified mode")


if __name__ == "__main__":
    unittest.main()
