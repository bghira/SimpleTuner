"""Tests for Replicate SimpleTuner hardware profile routing."""

import unittest
from unittest.mock import AsyncMock, patch


class ReplicateHardwareProfileTests(unittest.IsolatedAsyncioTestCase):
    def test_profile_aliases_normalize(self):
        from simpletuner.simpletuner_sdk.server.services.cloud.replicate_profiles import (
            get_replicate_hardware_profile,
            normalize_replicate_hardware_profile,
        )

        self.assertEqual(normalize_replicate_hardware_profile(None), "h100")
        self.assertEqual(normalize_replicate_hardware_profile("advanced-trainer-h100-x4"), "h100-x4")
        self.assertEqual(normalize_replicate_hardware_profile("1xL40S"), "l40s")
        self.assertEqual(
            get_replicate_hardware_profile("l40s-x8").model,
            "simpletuner/advanced-trainer-l40s-x8",
        )

    def test_invalid_profile_raises(self):
        from simpletuner.simpletuner_sdk.server.services.cloud.replicate_profiles import normalize_replicate_hardware_profile

        with self.assertRaises(ValueError):
            normalize_replicate_hardware_profile("a10g")

    def test_default_hardware_info_includes_h100_and_l40s_pricing(self):
        from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import DEFAULT_HARDWARE_INFO

        self.assertEqual(DEFAULT_HARDWARE_INFO["gpu-l40s"]["cost_per_second"], 0.000972222)
        self.assertEqual(DEFAULT_HARDWARE_INFO["gpu-h100"]["cost_per_second"], 0.001525)

    async def test_replicate_client_uses_profile_model_for_latest_version(self):
        from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import ReplicateCogClient

        client = ReplicateCogClient()
        client.list_model_versions = AsyncMock(
            return_value=[
                {
                    "full_version": "simpletuner/advanced-trainer-h100-x4:version123",
                }
            ]
        )

        version = await client.get_effective_version("simpletuner/advanced-trainer-h100-x4")

        self.assertEqual(version, "simpletuner/advanced-trainer-h100-x4:version123")
        client.list_model_versions.assert_awaited_once_with("simpletuner/advanced-trainer-h100-x4")

    async def test_replicate_client_sends_selected_profile_metadata(self):
        from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import ReplicateCogClient

        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "id": "pred-123",
                    "status": "starting",
                    "created_at": "2026-06-27T00:00:00Z",
                }

        class _HTTPClient:
            async def post(self, *args, **kwargs):
                return _Response()

        client = ReplicateCogClient()
        client.get_token_for_user = AsyncMock(return_value="r8-token")
        client.get_effective_version = AsyncMock(return_value="simpletuner/advanced-trainer-l40s-x2:version123")
        client._get_http_client = AsyncMock(return_value=_HTTPClient())

        with patch.object(client, "_get_headers", return_value={"Authorization": "Bearer r8-token"}):
            job = await client.run_job(config={}, dataloader=[], hardware_profile="l40s-x2")

        self.assertEqual(job.hardware_type, "2x L40S")
        self.assertEqual(job.metadata["hardware_profile"], "l40s-x2")
        self.assertEqual(job.metadata["model"], "simpletuner/advanced-trainer-l40s-x2")
        client.get_effective_version.assert_awaited_once_with("simpletuner/advanced-trainer-l40s-x2")


if __name__ == "__main__":
    unittest.main()
