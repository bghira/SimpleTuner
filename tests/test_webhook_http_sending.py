#!/usr/bin/env python3
"""Test that webhooks are actually sent via HTTP."""

import json
import os
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event, Thread
from time import sleep
from unittest.mock import MagicMock, patch

import requests
from PIL import Image

from simpletuner.helpers.webhooks.handler import WebhookHandler


class TestWebhookHTTPServer(BaseHTTPRequestHandler):
    """Simple HTTP server to receive webhook requests."""

    received_requests = []

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode("utf-8"))
            self.received_requests.append(
                {
                    "path": self.path,
                    "data": data,
                    "headers": dict(self.headers),
                }
            )
        except Exception as e:
            print(f"Error parsing request: {e}")

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def log_message(self, format, *args):
        # Suppress server logs
        pass


class TestWebhookHTTPSending(unittest.TestCase):
    """Test that webhooks are sent correctly via HTTP."""

    @classmethod
    def setUpClass(cls):
        """Start test HTTP server."""
        cls.server = HTTPServer(("127.0.0.1", 8999), TestWebhookHTTPServer)
        cls.server_thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        sleep(0.1)  # Give server time to start

    @classmethod
    def tearDownClass(cls):
        """Stop test HTTP server."""
        cls.server.shutdown()

    def setUp(self):
        """Clear received requests before each test."""
        TestWebhookHTTPServer.received_requests.clear()

    def test_worker_progress_does_not_wait_for_http_and_coalesces_backlog(self):
        request_started = Event()
        release_request = Event()
        progress_returned = Event()
        backlog_delivered = Event()
        reports = []

        def post(*args, **kwargs):
            reports.append(kwargs["json"])
            if len(reports) == 1:
                request_started.set()
                if not release_request.wait(5):
                    raise AssertionError("The test did not release the first request")
                raise requests.ConnectionError("Connection reset")
            if len(reports) == 3:
                backlog_delivered.set()
            return MagicMock()

        def report_first_step():
            handler.send_raw({"type": "training.progress", "progress": {"current": 1}})
            progress_returned.set()

        env = {
            "SIMPLETUNER_WORKER_STATUS_URL": "https://orchestrator.example/status",
            "SIMPLETUNER_WORKER_TOKEN": "test-token",
        }
        with (
            patch.dict(os.environ, env),
            patch("requests.Session.post", side_effect=post),
            patch("simpletuner.helpers.webhooks.worker_telemetry._RETRY_INTERVAL", 0.01),
        ):
            handler = WebhookHandler(None, "worker")
            producer = Thread(target=report_first_step, daemon=True)
            producer.start()
            try:
                self.assertTrue(request_started.wait(3))
                self.assertTrue(progress_returned.wait(1), "Worker HTTP blocked the training producer")
                for step in range(2, 20):
                    handler.send_raw({"type": "training.progress", "progress": {"current": step}})
                handler.send_lifecycle_stage("validation", "Validation", "running")
                handler.send_lifecycle_stage("validation", "Validation", "completed")
                handler.send_raw({"type": "training.progress", "progress": {"current": 20}})
            finally:
                release_request.set()
                producer.join(timeout=5)
            try:
                self.assertTrue(backlog_delivered.wait(3))
            finally:
                handler.close_worker_telemetry()

        self.assertEqual(len(reports), 3)
        self.assertEqual(reports[0], {"progress": {"current": 1}})
        self.assertEqual(reports[1]["validation"]["status"], "completed")
        self.assertEqual(reports[2], {"progress": {"current": 20}})

    def test_worker_reports_only_training_progress_to_its_status_endpoint(self):
        env = {
            "SIMPLETUNER_WORKER_STATUS_URL": "http://127.0.0.1:8999/worker-status",
            "SIMPLETUNER_WORKER_TOKEN": "worker-test-token",
        }
        progress = {"current": 12, "metrics": {"loss": 0}}
        with patch.dict(os.environ, env):
            handler = WebhookHandler(
                None, "worker", webhook_config={"webhook_type": "raw", "callback_url": "http://127.0.0.1:8999/user-callback"}
            )
            handler.send_raw({"type": "training.status", "progress": progress})
            handler.send_raw({"type": "notification"})
            handler.close_worker_telemetry()
        reports = TestWebhookHTTPServer.received_requests
        self.assertCountEqual([r["path"] for r in reports], ["/worker-status", "/user-callback", "/user-callback"])
        worker_report = next(r for r in reports if r["path"] == "/worker-status")
        self.assertEqual(worker_report["headers"]["X-Worker-Token"], "worker-test-token")
        self.assertEqual(worker_report["data"], {"progress": progress})
        for report in reports:
            if report["path"] == "/user-callback":
                self.assertNotIn("X-Worker-Token", report["headers"])

    def test_worker_reports_validation_stage_and_sampling_counts_without_media(self):
        env = {"SIMPLETUNER_WORKER_STATUS_URL": "http://127.0.0.1:8999/status", "SIMPLETUNER_WORKER_TOKEN": "test-token"}
        with patch.dict(os.environ, env):
            handler = WebhookHandler(None, "worker")
            handler.send_lifecycle_stage("validation", "Running Validation", "running")
            handler.close_worker_telemetry()
            handler = WebhookHandler(None, "worker")
            handler.send_raw(
                {"type": "validation.image", "data": {"step": 4, "total_steps": 8, "prompt": "sample prompt"}},
                images=["preview"],
            )
            handler.close_worker_telemetry()
            handler = WebhookHandler(None, "worker")
            handler.send_lifecycle_stage("validation", "Running Validation", "completed")
            handler.close_worker_telemetry()
        start, sample, end = [r["data"]["validation"] for r in TestWebhookHTTPServer.received_requests]
        self.assertEqual((start["status"], end["status"]), ("running", "completed"))
        self.assertEqual(set(sample), {"progress", "status", "timestamp"})
        self.assertEqual(sample["progress"], {"current": 4, "total": 8})

    def test_worker_delivers_final_validation_images_without_configured_webhooks(self):
        status_url = "http://127.0.0.1:8999/api/workers/job/test-job/status"
        env = {"SIMPLETUNER_WORKER_STATUS_URL": status_url, "SIMPLETUNER_WORKER_TOKEN": "test-token"}
        image = Image.new("RGB", (8, 8))
        with patch.dict(os.environ, env):
            handler = WebhookHandler(None, "worker")
            for label in ("first", "second"):
                handler.send_raw({"message": label}, message_type="training.validation", images=[image])
            handler.close_worker_telemetry()

        reports = TestWebhookHTTPServer.received_requests
        self.assertEqual([row["data"]["message"] for row in reports], ["first", "second"])
        for row in reports:
            self.assertEqual(row["path"], "/api/workers/job/test-job/validation")
            self.assertTrue(row["data"]["images"][0].startswith("iVBORw0KGgo"))
            self.assertEqual(row["headers"]["X-Worker-Token"], "test-token")

    def test_worker_validation_uses_configured_callback_for_the_same_orchestrator(self):
        status_url = "http://127.0.0.1:8999/api/workers/job/test-job/status"
        env = {"SIMPLETUNER_WORKER_STATUS_URL": status_url, "SIMPLETUNER_WORKER_TOKEN": "test-token"}
        with patch.dict(os.environ, env):
            handler = WebhookHandler(
                None, "worker", webhook_config={"webhook_type": "raw", "callback_url": "http://127.0.0.1:8999/callback"}
            )
            handler.send_raw(
                {"message": "Validation"}, message_type="training.validation", images=[Image.new("RGB", (8, 8))]
            )
            handler.close_worker_telemetry()

        (report,) = TestWebhookHTTPServer.received_requests
        self.assertEqual(report["path"], "/callback")
        self.assertNotIn("X-Worker-Token", report["headers"])
        self.assertEqual(report["data"]["type"], "training.validation")

    def test_simple_message_sent(self):
        """Test that a simple text message is sent via HTTP."""
        webhook_config = [
            {
                "webhook_type": "raw",
                "callback_url": "http://127.0.0.1:8999/callback",
                "log_level": "info",
            }
        ]

        mock_accelerator = MagicMock()
        mock_accelerator.is_main_process = True

        handler = WebhookHandler(
            mock_accelerator,
            "test_project",
            webhook_config=webhook_config,
        )

        # Send a test message using send_raw (raw webhooks require send_raw)
        handler.send_raw(
            structured_data={"message": "Test webhook message"}, message_type="notification", message_level="info"
        )

        # Give the request time to complete
        sleep(0.2)

        # Check that request was received
        self.assertEqual(len(TestWebhookHTTPServer.received_requests), 1)
        request = TestWebhookHTTPServer.received_requests[0]
        self.assertEqual(request["path"], "/callback")
        self.assertEqual(request["data"]["message"], "Test webhook message")

    def test_structured_data_sent(self):
        """Test that structured data is sent correctly."""
        webhook_config = [
            {
                "webhook_type": "raw",
                "callback_url": "http://127.0.0.1:8999/callback",
                "log_level": "info",
            }
        ]

        mock_accelerator = MagicMock()
        mock_accelerator.is_main_process = True

        handler = WebhookHandler(
            mock_accelerator,
            "test_project",
            webhook_config=webhook_config,
        )

        # Send structured data
        handler.send_raw(
            structured_data={"status": "training_started", "epoch": 1},
            message_type="training.status",
            message_level="info",
        )

        sleep(0.2)

        self.assertEqual(len(TestWebhookHTTPServer.received_requests), 1)
        request = TestWebhookHTTPServer.received_requests[0]
        self.assertEqual(request["data"]["status"], "training_started")
        self.assertEqual(request["data"]["type"], "training.status")  # Now uses dot notation
        self.assertEqual(request["data"]["epoch"], 1)

    def test_error_message_sent(self):
        """Test that error messages are sent with correct severity."""
        webhook_config = [
            {
                "webhook_type": "raw",
                "callback_url": "http://127.0.0.1:8999/callback",
                "log_level": "error",  # Only send errors
            }
        ]

        mock_accelerator = MagicMock()
        mock_accelerator.is_main_process = True

        handler = WebhookHandler(
            mock_accelerator,
            "test_project",
            webhook_config=webhook_config,
        )

        # Send an info message (should be filtered out)
        handler.send_raw(structured_data={"message": "Info message"}, message_type="notification", message_level="info")
        sleep(0.1)

        # Send an error message (should be sent)
        handler.send_raw(structured_data={"message": "Error occurred"}, message_type="notification", message_level="error")
        sleep(0.2)

        # Only the error should have been sent
        self.assertEqual(len(TestWebhookHTTPServer.received_requests), 1)
        request = TestWebhookHTTPServer.received_requests[0]
        self.assertEqual(request["data"]["message"], "Error occurred")

    def test_localhost_webhooks_use_http_fallback(self):
        """Test that localhost webhooks fall back to HTTP when callback service unavailable."""
        # Use 0.0.0.0 which triggers localhost detection
        webhook_config = [
            {
                "webhook_type": "raw",
                "callback_url": "http://127.0.0.1:8999/callback",
                "log_level": "info",
            }
        ]

        mock_accelerator = MagicMock()
        mock_accelerator.is_main_process = True

        handler = WebhookHandler(
            mock_accelerator,
            "test_project",
            webhook_config=webhook_config,
        )

        # This should trigger localhost detection and HTTP fallback
        handler.send_raw(
            structured_data={"message": "Localhost webhook test"}, message_type="notification", message_level="info"
        )
        sleep(0.2)

        # Request should have been received via HTTP fallback
        self.assertEqual(len(TestWebhookHTTPServer.received_requests), 1)
        request = TestWebhookHTTPServer.received_requests[0]
        self.assertIn("Localhost webhook test", request["data"]["message"])


if __name__ == "__main__":
    unittest.main()
