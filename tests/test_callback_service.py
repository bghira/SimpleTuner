from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

try:
    from simpletuner.simpletuner_sdk.api_state import APIState
    from simpletuner.simpletuner_sdk.server.services.callback_events import EventType
    from simpletuner.simpletuner_sdk.server.services.callback_presenter import CallbackPresenter
    from simpletuner.simpletuner_sdk.server.services.callback_service import CallbackService
    from simpletuner.simpletuner_sdk.server.services.event_store import EventStore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    CallbackPresenter = None  # type: ignore[assignment]
    CallbackService = None  # type: ignore[assignment]
    EventStore = None  # type: ignore[assignment]
    APIState = None  # type: ignore[assignment]
    _SKIP_REASON = str(exc)
else:
    _SKIP_REASON = ""


@unittest.skipIf(CallbackService is None or EventStore is None, f"Dependencies unavailable: {_SKIP_REASON}")
class CallbackServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.store = EventStore(max_events=5)
        self.service = CallbackService(self.store)
        self._tmpdir = tempfile.TemporaryDirectory()
        if APIState is not None:
            APIState.state = {}
            APIState.state_file = os.path.join(self._tmpdir.name, "api_state.json")
            APIState._state_file_initialised = True  # type: ignore[attr-defined]

    def tearDown(self) -> None:
        if APIState is not None:
            APIState.state = {}
        self._tmpdir.cleanup()

    def test_handle_incoming_assigns_index_and_payload(self) -> None:
        event = self.service.handle_incoming(
            {
                "message_type": "_send_webhook_msg",
                "message": "Training started",
                "job_id": "job-1",
                "timestamp": 1,
            }
        )
        self.assertIsNotNone(event)
        self.assertEqual(event.index, 0)
        self.assertEqual(event.type, EventType.NOTIFICATION)
        recent = self.service.get_recent(limit=1)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0].index, 0)

    def test_configure_webhook_clears_history(self) -> None:
        self.service.handle_incoming({"type": "notification", "message": "hello"})
        self.service.handle_incoming({"message_type": "configure_webhook", "message": "reset"})
        events = self.service.get_recent(limit=5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type.value, EventType.NOTIFICATION.value)

    def test_terminal_status_suppresses_future_updates(self) -> None:
        running_payload = {
            "type": "training.status",
            "job_id": "restart-job",
            "data": {"status": "running"},
            "global_step": 5,
            "total_num_steps": 10,
        }
        event = self.service.handle_incoming(running_payload)
        self.assertIsNotNone(event)

        failed_payload = {
            "type": "training.status",
            "job_id": "restart-job",
            "data": {"status": "failed"},
        }
        failed_event = self.service.handle_incoming(failed_payload)
        self.assertIsNotNone(failed_event)
        self.assertIsNone(APIState.get_state("training_progress"))

        # Training status events for the same job that's already terminal should be suppressed
        # at ingestion time - handle_incoming returns None for suppressed events
        retry_status = {
            "type": "training.status",
            "job_id": "restart-job",
            "data": {"status": "running"},
        }
        retry_event = self.service.handle_incoming(retry_status)
        # The event should be suppressed (None) because the job is already in terminal state
        self.assertIsNone(retry_event, "Events for terminal jobs should be suppressed at ingestion")

        # Verify no events for this job with running status exist
        recent_events = self.service.get_recent(limit=10)
        retry_in_recent = any(
            e.job_id == "restart-job" and e.data and isinstance(e.data, dict) and e.data.get("status") == "running"
            for e in recent_events
            if e.data
        )
        self.assertFalse(retry_in_recent, "Terminal job status updates should not appear in event history")

    def test_latest_index_tracks_highest_recorded_event(self) -> None:
        self.assertIsNone(self.service.latest_index())

        first = self.service.handle_incoming({"type": "notification", "message": "first"})
        self.assertIsNotNone(first)
        self.assertEqual(self.service.latest_index(), first.index)

        second = self.service.handle_incoming({"type": "notification", "message": "second"})
        self.assertIsNotNone(second)
        self.assertEqual(self.service.latest_index(), second.index)

    @patch("simpletuner.simpletuner_sdk.server.services.callback_service.get_sse_manager")
    def test_progress_event_broadcasts_via_sse(self, mock_get_sse_manager: Mock) -> None:
        """Test that progress events trigger SSE broadcasts with correct payload."""
        # Setup mock SSE manager
        mock_sse_manager = Mock()
        mock_sse_manager.broadcast = AsyncMock()
        mock_get_sse_manager.return_value = mock_sse_manager

        # Handle a progress event
        progress_payload = {
            "type": "training.progress",
            "job_id": "test-job-123",
            "progress": {
                "current": 100,
                "total": 1000,
                "percent": 10.0,
            },
            "data": {
                "loss": 0.5,
                "learning_rate": 0.0001,
                "epoch": 1,
                "total_epochs": 10,
            },
        }

        event = self.service.handle_incoming(progress_payload)
        self.assertIsNotNone(event)

        # Verify SSE manager was called
        mock_get_sse_manager.assert_called()

        # Verify broadcast was actually called (not just that it exists)
        self.assertTrue(mock_sse_manager.broadcast.called, "SSE manager broadcast should have been called")
        self.assertEqual(mock_sse_manager.broadcast.call_count, 1, "Broadcast should be called exactly once")

        # Verify the broadcast payload contains expected fields
        call_args = mock_sse_manager.broadcast.call_args
        self.assertIsNotNone(call_args, "Broadcast should have been called with arguments")

        args, kwargs = call_args
        payload = args[0]

        # Verify payload structure
        self.assertEqual(payload["type"], "training.progress")
        self.assertEqual(payload["job_id"], "test-job-123")
        self.assertEqual(payload["step"], 100)
        self.assertEqual(payload["total_steps"], 1000)
        self.assertEqual(payload["epoch"], 1)
        self.assertEqual(payload["total_epochs"], 10)
        self.assertEqual(payload["loss"], 0.5)
        self.assertEqual(payload["learning_rate"], 0.0001)
        self.assertEqual(kwargs.get("event_type"), "training.progress")

    @patch("simpletuner.simpletuner_sdk.server.services.callback_service.get_sse_manager")
    def test_lifecycle_stage_broadcasts_via_sse(self, mock_get_sse_manager: Mock) -> None:
        """Test that lifecycle stage events trigger SSE broadcasts."""
        # Setup mock SSE manager
        mock_sse_manager = Mock()
        mock_sse_manager.broadcast = AsyncMock()
        mock_get_sse_manager.return_value = mock_sse_manager

        # Handle a lifecycle stage event
        stage_payload = {
            "type": "lifecycle.stage",
            "job_id": "test-job-456",
            "stage": {
                "key": "model_loading",
                "label": "Loading Model",
                "status": "running",
                "progress": {
                    "current": 5,
                    "total": 10,
                    "percent": 50,
                },
            },
        }

        event = self.service.handle_incoming(stage_payload)
        self.assertIsNotNone(event)

        # Verify SSE manager was called
        mock_get_sse_manager.assert_called()

        # Verify broadcast was actually called
        self.assertTrue(mock_sse_manager.broadcast.called, "SSE manager broadcast should have been called")
        self.assertEqual(mock_sse_manager.broadcast.call_count, 1, "Broadcast should be called exactly once")

        # Verify the broadcast payload
        call_args = mock_sse_manager.broadcast.call_args
        self.assertIsNotNone(call_args, "Broadcast should have been called with arguments")

        args, kwargs = call_args
        payload = args[0]

        # Verify lifecycle stage payload structure
        self.assertEqual(payload["type"], "lifecycle.stage")
        self.assertEqual(payload["job_id"], "test-job-456")
        self.assertIn("stage", payload)
        self.assertEqual(payload["stage"]["key"], "model_loading")
        self.assertEqual(kwargs.get("event_type"), "lifecycle.stage")

    @patch("simpletuner.simpletuner_sdk.server.services.callback_service.get_sse_manager")
    def test_error_status_broadcasts_progress_reset(self, mock_get_sse_manager: Mock) -> None:
        """Test that error status events trigger progress reset broadcasts."""
        # Setup mock SSE manager
        mock_sse_manager = Mock()
        mock_sse_manager.broadcast = AsyncMock()
        mock_get_sse_manager.return_value = mock_sse_manager

        # Handle an error status event
        error_payload = {
            "type": "training.status",
            "job_id": "test-job-789",
            "data": {"status": "failed"},
            "message": "Training failed due to error",
        }

        event = self.service.handle_incoming(error_payload)
        self.assertIsNotNone(event)

        # Verify SSE manager was called for progress reset
        mock_get_sse_manager.assert_called()

        # Verify broadcast was actually called
        self.assertTrue(mock_sse_manager.broadcast.called, "SSE manager broadcast should have been called")
        self.assertEqual(mock_sse_manager.broadcast.call_count, 1, "Broadcast should be called exactly once")

        # Verify the broadcast payload for progress reset
        call_args = mock_sse_manager.broadcast.call_args
        self.assertIsNotNone(call_args, "Broadcast should have been called with arguments")

        args, kwargs = call_args
        payload = args[0]

        # Verify progress reset payload structure
        self.assertEqual(payload["type"], "training_progress")
        self.assertEqual(payload["job_id"], "test-job-789")
        self.assertEqual(payload["status"], "failed")
        self.assertTrue(payload.get("reset"), "Reset flag should be True")
        self.assertEqual(payload["percent"], 0)
        self.assertEqual(payload["step"], 0)
        self.assertEqual(kwargs.get("event_type"), "training_progress")

        # Verify training_progress state was cleared
        self.assertIsNone(APIState.get_state("training_progress"))


@unittest.skipIf(
    CallbackPresenter is None or CallbackService is None or EventStore is None,
    f"Dependencies unavailable: {_SKIP_REASON}",
)
class CallbackPresenterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        if APIState is not None:
            APIState.state = {}
            APIState.state_file = os.path.join(self._tmpdir.name, "api_state.json")
            APIState._state_file_initialised = True  # type: ignore[attr-defined]

    def tearDown(self) -> None:
        if APIState is not None:
            APIState.state = {}
        self._tmpdir.cleanup()

    def test_to_sse_returns_expected_shape(self) -> None:
        service = CallbackService(EventStore())
        event = service.handle_incoming({"message_type": "warning", "message": "Heads up"})
        self.assertIsNotNone(event)
        event_type, payload = CallbackPresenter.to_sse(event)
        self.assertEqual(event_type, "notification")
        self.assertIn("title", payload)
        self.assertIn("severity", payload)

    def test_htmx_tile_renders_base64_images(self) -> None:
        service = CallbackService(EventStore())
        event = service.handle_incoming(
            {"message_type": "validation_log", "message": "Validation sample", "images": ["ZmFrZS1iYXNlNjQ="]}
        )
        self.assertIsNotNone(event)
        html_snippet = CallbackPresenter.to_htmx_tile(event)
        self.assertIn("<img", html_snippet)
        self.assertIn("data:image/png;base64,ZmFrZS1iYXNlNjQ=", html_snippet)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
