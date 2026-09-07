"""Tracker identity survives each cloud job persistence path."""

import unittest
from unittest.mock import AsyncMock

from simpletuner.simpletuner_sdk.server.services.cloud.job_submission import JobSubmissionService, SubmissionContext


class CloudJobTrackerMetadataTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_upload_job_records_project_and_run(self):
        for prefix in ("", "--"):
            with self.subTest(prefix=prefix):
                store = AsyncMock()
                service = JobSubmissionService(store)
                context = SubmissionContext(
                    config={f"{prefix}tracker_project_name": "portraits"},
                    dataloader_config=[],
                    tracker_run_name="shared-run",
                )
                await service._create_upload_job(context, "upload-1", {})
                job = store.add_job.call_args.args[0]
                self.assertEqual(job.metadata["tracker_project_name"], "portraits")
                self.assertEqual(job.metadata["tracker_run_name"], "shared-run")

    async def test_provider_update_preserves_project_identity(self):
        from simpletuner.simpletuner_sdk.server.services.cloud.base import CloudJobInfo, CloudJobStatus

        store = AsyncMock()
        service = JobSubmissionService(store)
        context = SubmissionContext(
            config={"tracker_project_name": "portraits"}, dataloader_config=[], tracker_run_name="shared-run"
        )
        job = CloudJobInfo(job_id="provider-1", provider="replicate", status=CloudJobStatus.RUNNING, created_at="now")
        await service._update_upload_job_from_provider("upload-1", job, context, {}, None, None, None)
        metadata = store.update_job.call_args.args[1]["metadata"]
        self.assertEqual(metadata["tracker_project_name"], "portraits")
        self.assertEqual(metadata["tracker_run_name"], "shared-run")
        self.assertEqual(metadata["prediction_id"], "provider-1")
