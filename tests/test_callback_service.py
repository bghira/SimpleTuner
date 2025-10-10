from __future__ import annotations

import os
import tempfile
import unittest

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
        self.service.handle_incoming({"message_type": "_send_webhook_msg", "message": "hello"})
        self.service.handle_incoming({"message_type": "configure_webhook", "message": "reset"})
        events = self.service.get_recent(limit=5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type.value, EventType.NOTIFICATION.value)

    def test_terminal_status_suppresses_future_updates(self) -> None:
        running_payload = {
            "message_type": "training_status",
            "job_id": "restart-job",
            "status": "running",
            "global_step": 5,
            "total_num_steps": 10,
        }
        event = self.service.handle_incoming(running_payload)
        self.assertIsNotNone(event)

        failed_payload = {
            "message_type": "training_status",
            "job_id": "restart-job",
            "status": "failed",
        }
        failed_event = self.service.handle_incoming(failed_payload)
        self.assertIsNotNone(failed_event)
        self.assertIsNone(APIState.get_state("training_progress"))

        # Training status events for the same job should now be suppressed.
        retry_status = {
            "message_type": "training_status",
            "job_id": "restart-job",
            "status": "running",
        }
        suppressed_status = self.service.handle_incoming(retry_status)
        self.assertIsNone(suppressed_status)

        # Progress payloads should also be ignored once the job is terminal.
        suppressed_progress = self.service.handle_incoming(
            {
                "message_type": "progress_update",
                "job_id": "restart-job",
                "message": {"progress_type": "resume", "progress": 50, "total_elements": 100},
            }
        )
        self.assertIsNone(suppressed_progress)


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
