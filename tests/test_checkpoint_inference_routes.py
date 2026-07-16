import unittest
from unittest.mock import patch

from fastapi import status

from simpletuner.simpletuner_sdk.server.services.checkpoint_inference_service import (
    CHECKPOINT_INFERENCE_SERVICE,
    CheckpointInferenceServiceError,
)
from tests.test_webui_api import _WebUIBaseTestCase


class CheckpointInferenceRoutesTestCase(_WebUIBaseTestCase, unittest.TestCase):
    def test_start_inference_forwards_validated_request(self) -> None:
        response_payload = {"session_id": "session-one", "status": "loading"}
        with patch.object(CHECKPOINT_INFERENCE_SERVICE, "start", return_value=response_payload) as start:
            response = self.client.post(
                "/api/checkpoints/inference/start",
                json={
                    "environment": "test",
                    "checkpoint_names": ["checkpoint-100"],
                    "use_configured_prompt": False,
                    "custom_prompts": ["a prompt"],
                    "filename_style": "prompt",
                    "keep_loaded": True,
                    "streaming_preview": True,
                    "settings": {
                        "seed": 12,
                        "num_inference_steps": 20,
                        "validation_resolution": "512,768x512",
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), response_payload)
        self.assertEqual(start.call_args.kwargs["checkpoint_names"], ["checkpoint-100"])
        self.assertTrue(start.call_args.kwargs["streaming_preview"])
        self.assertEqual(
            start.call_args.kwargs["settings"],
            {"seed": 12, "num_inference_steps": 20, "validation_resolution": "512,768x512"},
        )

    def test_empty_checkpoint_selection_is_rejected(self) -> None:
        response = self.client.post(
            "/api/checkpoints/inference/start",
            json={"environment": "test", "checkpoint_names": [], "custom_prompts": ["a prompt"]},
        )

        self.assertEqual(response.status_code, 422)

    def test_service_conflict_maps_to_http_conflict(self) -> None:
        error = CheckpointInferenceServiceError("accelerator busy", status.HTTP_409_CONFLICT)
        with patch.object(CHECKPOINT_INFERENCE_SERVICE, "prompt_sources", side_effect=error):
            response = self.client.get("/api/checkpoints/inference/prompt-sources?environment=test")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], "accelerator busy")

    def test_active_inference_session_returns_environment_session(self) -> None:
        response_payload = {"session_id": "session-one", "status": "loaded"}
        with patch.object(
            CHECKPOINT_INFERENCE_SERVICE,
            "active_environment_session",
            return_value=response_payload,
        ) as active_environment_session:
            response = self.client.get("/api/checkpoints/inference/active?environment=test")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"session": response_payload})
        active_environment_session.assert_called_once_with("test")

    def test_delete_history_forwards_selected_media_paths(self) -> None:
        response_payload = {"deleted_count": 2, "media_paths": ["one.png", "two.png"]}
        with patch.object(CHECKPOINT_INFERENCE_SERVICE, "delete_history", return_value=response_payload) as delete_history:
            response = self.client.request(
                "DELETE",
                "/api/checkpoints/inference/history",
                json={
                    "environment": "test",
                    "media_paths": [
                        "session-one/checkpoint-100/one.png",
                        "session-one/checkpoint-100/two.png",
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), response_payload)
        delete_history.assert_called_once_with(
            "test",
            [
                "session-one/checkpoint-100/one.png",
                "session-one/checkpoint-100/two.png",
            ],
        )


if __name__ == "__main__":
    unittest.main()
