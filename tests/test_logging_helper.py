import os
import types
import unittest
from unittest.mock import MagicMock, patch

from simpletuner.helpers.logging import WebhookLogger, flush_webhook_queue, get_logger


class WebhookLoggerTests(unittest.TestCase):
    def setUp(self):
        self._level = os.environ.get("SIMPLETUNER_LOG_LEVEL")
        os.environ["SIMPLETUNER_LOG_LEVEL"] = "ERROR"
        # Ensure logging does not complain about missing handlers during tests.
        import logging

        self._root_handler = logging.NullHandler()
        logging.getLogger().addHandler(self._root_handler)
        flush_webhook_queue(timeout=1.0)

    def tearDown(self):
        import logging

        logging.getLogger().removeHandler(self._root_handler)
        flush_webhook_queue(timeout=1.0)
        if self._level is None:
            os.environ.pop("SIMPLETUNER_LOG_LEVEL", None)
        else:
            os.environ["SIMPLETUNER_LOG_LEVEL"] = self._level

    @patch("simpletuner.helpers.training.state_tracker.StateTracker.set_webhook_handler")
    @patch("simpletuner.helpers.training.state_tracker.StateTracker.get_args", return_value=None)
    @patch("simpletuner.helpers.training.state_tracker.StateTracker.get_webhook_handler")
    def test_existing_handler_receives_messages(
        self,
        mock_get_handler,
        _mock_get_args,
        _mock_set_handler,
    ):
        handler_mock = MagicMock()
        mock_get_handler.return_value = handler_mock

        logger = get_logger("tests.logging.existing")
        self.assertIsInstance(logger, WebhookLogger)

        logger.error("boom %s", "value")
        flush_webhook_queue(timeout=1.0)

        handler_mock.send.assert_called_once()
        send_kwargs = handler_mock.send.call_args.kwargs
        self.assertEqual(send_kwargs["message"], "[tests.logging.existing] boom value")
        self.assertEqual(send_kwargs["message_level"], "error")

        handler_mock.send_raw.assert_called_once()
        raw_kwargs = handler_mock.send_raw.call_args.kwargs
        self.assertEqual(raw_kwargs["message_level"], "error")
        self.assertEqual(raw_kwargs["message_type"], "log.message")
        payload = raw_kwargs["structured_data"]
        self.assertEqual(payload["message"], "boom value")
        self.assertEqual(payload["logger"], "tests.logging.existing")
        self.assertEqual(payload["severity"], "error")

    @patch("simpletuner.helpers.training.state_tracker.StateTracker")
    @patch("simpletuner.helpers.webhooks.handler.WebhookHandler")
    def test_handler_created_when_missing(self, webhook_handler_cls, state_tracker_cls):
        handler_instance = MagicMock()
        webhook_handler_cls.return_value = handler_instance

        mock_state_tracker = state_tracker_cls
        mock_state_tracker.get_webhook_handler.return_value = None
        mock_state_tracker.get_args.return_value = types.SimpleNamespace(
            webhook_config={"webhook_url": "https://example.test", "webhook_type": "raw"},
            tracker_project_name="proj",
            tracker_run_name="run",
        )

        logger = get_logger("tests.logging.create")
        logger.error("auto")
        flush_webhook_queue(timeout=1.0)

        webhook_handler_cls.assert_called_once()
        mock_state_tracker.set_webhook_handler.assert_called_once_with(handler_instance)
        handler_instance.send.assert_called_once()
        handler_instance.send_raw.assert_called_once()

    @patch(
        "simpletuner.helpers.logging._load_env_webhook_config",
        return_value=[{"webhook_type": "raw", "webhook_url": "https://fallback.test"}],
    )
    @patch("simpletuner.helpers.logging._extract_webhook_config", return_value=None)
    @patch("simpletuner.helpers.webhooks.handler.WebhookHandler")
    def test_env_config_used_when_args_missing(
        self,
        webhook_handler_cls,
        extract_config_mock,
        load_env_mock,
    ):
        from simpletuner.helpers.training import state_tracker as tracker_module

        handler_instance = MagicMock()
        webhook_handler_cls.return_value = handler_instance

        with (
            patch.object(tracker_module.StateTracker, "get_webhook_handler", return_value=None) as get_handler,
            patch.object(tracker_module.StateTracker, "set_webhook_handler") as set_handler,
            patch.object(tracker_module.StateTracker, "get_args", return_value=None),
            patch.object(tracker_module.StateTracker, "get_accelerator", return_value=None),
            patch.object(tracker_module.StateTracker, "get_job_id", return_value="job-123"),
        ):
            logger = get_logger("tests.logging.fallback")
            logger.error("fallback message")
            flush_webhook_queue(timeout=1.0)

        self.assertTrue(get_handler.called, "Expected get_webhook_handler to be invoked")
        self.assertTrue(extract_config_mock.called, "Expected extract helper to be invoked")
        self.assertTrue(load_env_mock.called, "Expected env webhook configuration loader to be called")
        self.assertTrue(set_handler.called, "set_webhook_handler should be invoked")
        webhook_handler_cls.assert_called_once()
        kwargs = webhook_handler_cls.call_args.kwargs
        self.assertIn("webhook_config", kwargs)
        config = kwargs["webhook_config"]
        self.assertTrue(config, "Expected fallback webhook config to be provided")
        handler_instance.send.assert_called_once()
        handler_instance.send_raw.assert_called_once()

    def test_fallback_returns_none_without_env_config(self):
        from simpletuner.helpers import logging as logging_module

        with patch("simpletuner.helpers.logging._load_env_webhook_config", return_value=None):
            self.assertIsNone(logging_module._fallback_webhook_config())


if __name__ == "__main__":
    unittest.main()
