from __future__ import annotations

import unittest

try:
    from simpletuner.simpletuner_sdk.server.services.callback_presenter import CallbackPresenter
    from simpletuner.simpletuner_sdk.server.services.callback_service import CallbackService
    from simpletuner.simpletuner_sdk.server.services.event_store import EventStore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    CallbackPresenter = None  # type: ignore[assignment]
    CallbackService = None  # type: ignore[assignment]
    EventStore = None  # type: ignore[assignment]
    _SKIP_REASON = str(exc)
else:
    _SKIP_REASON = ""


@unittest.skipIf(CallbackService is None or EventStore is None, f"Dependencies unavailable: {_SKIP_REASON}")
class CallbackServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.store = EventStore(max_events=5)
        self.service = CallbackService(self.store)

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
        self.assertEqual(event.category.value, "job")
        recent = self.service.get_recent(limit=1)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0].index, 0)

    def test_configure_webhook_clears_history(self) -> None:
        self.service.handle_incoming({"message_type": "_send_webhook_msg", "message": "hello"})
        self.service.handle_incoming({"message_type": "configure_webhook", "message": "reset"})
        events = self.service.get_recent(limit=5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].message_type, "configure_webhook")

    def test_progress_events_are_deduped(self) -> None:
        progress_payload = {
            "message_type": "progress_update",
            "job_id": "job-1",
            "message": {
                "progress_type": "init",
                "progress": 10,
                "total_elements": 100,
                "current_estimated_index": 10,
            },
        }
        first = self.service.handle_incoming(progress_payload)
        second = self.service.handle_incoming(progress_payload)
        self.assertIs(first, second)
        events = self.service.get_recent(limit=5)
        self.assertEqual(len(events), 1)


@unittest.skipIf(
    CallbackPresenter is None or CallbackService is None or EventStore is None,
    f"Dependencies unavailable: {_SKIP_REASON}",
)
class CallbackPresenterTestCase(unittest.TestCase):
    def test_to_sse_returns_expected_shape(self) -> None:
        service = CallbackService(EventStore())
        event = service.handle_incoming({"message_type": "warning", "message": "Heads up"})
        self.assertIsNotNone(event)
        event_type, payload = CallbackPresenter.to_sse(event)
        self.assertEqual(event_type, f"callback:{event.category.value}")
        self.assertIn("headline", payload)
        self.assertIn("severity", payload)

    def test_htmx_tile_renders_base64_images(self) -> None:
        service = CallbackService(EventStore())
        event = service.handle_incoming(
            {
                "message_type": "validation_log",
                "message": "Validation sample",
                "images": ["ZmFrZS1iYXNlNjQ="]
            }
        )
        self.assertIsNotNone(event)
        html_snippet = CallbackPresenter.to_htmx_tile(event)
        self.assertIn("<img", html_snippet)
        self.assertIn("data:image/png;base64,ZmFrZS1iYXNlNjQ=", html_snippet)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
