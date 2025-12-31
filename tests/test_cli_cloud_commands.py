"""Tests for CLI cloud commands.

Tests the command-line interface for cloud training operations:
- Job submission, listing, cancellation
- Job logs and status retrieval
- Configuration display and modification
- Cost limit management
"""

import argparse
import json
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch


class MockArgs:
    """Mock argparse.Namespace for testing."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestCloudSubmitCommand(unittest.TestCase):
    """Test cloud jobs submit command."""

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    @patch("sys.stdout", new_callable=StringIO)
    def test_submit_success(self, mock_stdout, mock_api):
        """Test successful job submission."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_submit

        mock_api.return_value = {
            "success": True,
            "job_id": "job-123",
            "status": "pending",
        }

        args = MockArgs(
            config="my-config",
            provider="replicate",
            idempotency_key=None,
            dry_run=False,
        )

        result = cmd_cloud_submit(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        self.assertIn("submitted successfully", output.lower())
        self.assertIn("job-123", output)

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    @patch("sys.stdout", new_callable=StringIO)
    def test_submit_with_idempotency_key(self, mock_stdout, mock_api):
        """Test submission with idempotency key returns existing job."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_submit

        mock_api.return_value = {
            "success": True,
            "job_id": "existing-job",
            "status": "running",
            "idempotent_hit": True,
        }

        args = MockArgs(
            config="my-config",
            provider="replicate",
            idempotency_key="unique-key-123",
            dry_run=False,
        )

        result = cmd_cloud_submit(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        self.assertIn("idempotency", output.lower())

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    @patch("sys.stdout", new_callable=StringIO)
    def test_submit_failure(self, mock_stdout, mock_api):
        """Test submission failure returns error."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_submit

        mock_api.return_value = {
            "success": False,
            "error": "API token not configured",
        }

        args = MockArgs(
            config="my-config",
            provider="replicate",
            idempotency_key=None,
            dry_run=False,
        )

        result = cmd_cloud_submit(args)

        self.assertEqual(result, 1)
        output = mock_stdout.getvalue()
        self.assertIn("error", output.lower())

    @patch("sys.stdout", new_callable=StringIO)
    def test_submit_missing_config(self, mock_stdout):
        """Test submission without config returns error."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_submit

        args = MockArgs(
            config=None,
            provider="replicate",
            idempotency_key=None,
            dry_run=False,
        )

        result = cmd_cloud_submit(args)

        self.assertEqual(result, 1)
        output = mock_stdout.getvalue()
        self.assertIn("config", output.lower())


class TestCloudListCommand(unittest.TestCase):
    """Test cloud jobs list command."""

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    @patch("sys.stdout", new_callable=StringIO)
    def test_list_jobs(self, mock_stdout, mock_api):
        """Test listing jobs."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_list

        mock_api.return_value = {
            "jobs": [
                {
                    "job_id": "job-1",
                    "status": "completed",
                    "config_name": "test-config",
                    "provider": "replicate",
                    "cost_usd": 1.50,
                    "duration_seconds": 3600,
                },
                {
                    "job_id": "job-2",
                    "status": "running",
                    "config_name": "train-lora",
                    "provider": "replicate",
                    "cost_usd": 0.75,
                    "duration_seconds": 1800,
                },
            ]
        }

        args = MockArgs(
            limit=20,
            status=None,
            provider=None,
            sync=False,
            format="table",
        )

        result = cmd_cloud_list(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        self.assertIn("job-1", output)
        self.assertIn("job-2", output)
        self.assertIn("Total: 2 jobs", output)

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    @patch("sys.stdout", new_callable=StringIO)
    def test_list_empty(self, mock_stdout, mock_api):
        """Test listing with no jobs."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_list

        mock_api.return_value = {"jobs": []}

        args = MockArgs(
            limit=20,
            status=None,
            provider=None,
            sync=False,
            format="table",
        )

        result = cmd_cloud_list(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        self.assertIn("no jobs found", output.lower())

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    @patch("sys.stdout", new_callable=StringIO)
    def test_list_json_format(self, mock_stdout, mock_api):
        """Test listing jobs in JSON format."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_list

        jobs = [{"job_id": "job-1", "status": "completed"}]
        mock_api.return_value = {"jobs": jobs}

        args = MockArgs(
            limit=20,
            status=None,
            provider=None,
            sync=False,
            format="json",
        )

        result = cmd_cloud_list(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        parsed = json.loads(output)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["job_id"], "job-1")


class TestCloudCancelCommand(unittest.TestCase):
    """Test cloud jobs cancel command."""

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    @patch("sys.stdout", new_callable=StringIO)
    def test_cancel_success(self, mock_stdout, mock_api):
        """Test successful job cancellation."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_cancel

        mock_api.return_value = {"success": True}

        args = MockArgs(job_id="job-123")

        result = cmd_cloud_cancel(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        self.assertIn("cancelled successfully", output.lower())

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    @patch("sys.stdout", new_callable=StringIO)
    def test_cancel_failure(self, mock_stdout, mock_api):
        """Test cancel failure."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_cancel

        mock_api.return_value = {"success": False}

        args = MockArgs(job_id="job-123")

        result = cmd_cloud_cancel(args)

        self.assertEqual(result, 1)
        output = mock_stdout.getvalue()
        self.assertIn("failed to cancel", output.lower())

    @patch("sys.stdout", new_callable=StringIO)
    def test_cancel_missing_job_id(self, mock_stdout):
        """Test cancel without job ID."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_cancel

        args = MockArgs(job_id=None)

        result = cmd_cloud_cancel(args)

        self.assertEqual(result, 1)
        output = mock_stdout.getvalue()
        self.assertIn("job id is required", output.lower())


class TestCloudGetCommand(unittest.TestCase):
    """Test cloud jobs get command."""

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    @patch("sys.stdout", new_callable=StringIO)
    def test_get_job_details(self, mock_stdout, mock_api):
        """Test getting job details."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_get

        mock_api.return_value = {
            "job": {
                "job_id": "job-123",
                "status": "completed",
                "config_name": "test-config",
                "provider": "replicate",
                "job_type": "training",
                "created_at": "2024-01-01T00:00:00Z",
                "duration_seconds": 3600,
                "cost_usd": 2.50,
            }
        }

        args = MockArgs(job_id="job-123", format="table")

        result = cmd_cloud_get(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        self.assertIn("job-123", output)
        self.assertIn("completed", output.lower())
        self.assertIn("test-config", output)

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    @patch("sys.stdout", new_callable=StringIO)
    def test_get_job_json_format(self, mock_stdout, mock_api):
        """Test getting job details in JSON format."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_get

        job_data = {
            "job_id": "job-123",
            "status": "running",
        }
        mock_api.return_value = {"job": job_data}

        args = MockArgs(job_id="job-123", format="json")

        result = cmd_cloud_get(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        parsed = json.loads(output)
        self.assertEqual(parsed["job_id"], "job-123")

    @patch("sys.stdout", new_callable=StringIO)
    def test_get_missing_job_id(self, mock_stdout):
        """Test get without job ID."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_get

        args = MockArgs(job_id=None, format="table")

        result = cmd_cloud_get(args)

        self.assertEqual(result, 1)


class TestCloudLogsCommand(unittest.TestCase):
    """Test cloud jobs logs command."""

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    @patch("sys.stdout", new_callable=StringIO)
    def test_get_logs(self, mock_stdout, mock_api):
        """Test getting job logs."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_logs

        mock_api.return_value = {"logs": "Step 1/100: loss=0.5\nStep 2/100: loss=0.45\n"}

        args = MockArgs(job_id="job-123", follow=False)

        result = cmd_cloud_logs(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        self.assertIn("Step 1/100", output)

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    @patch("sys.stdout", new_callable=StringIO)
    def test_get_logs_empty(self, mock_stdout, mock_api):
        """Test getting logs when none available."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_logs

        mock_api.return_value = {"logs": ""}

        args = MockArgs(job_id="job-123", follow=False)

        result = cmd_cloud_logs(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        self.assertIn("no logs available", output.lower())

    @patch("sys.stdout", new_callable=StringIO)
    def test_logs_missing_job_id(self, mock_stdout):
        """Test logs without job ID."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_logs

        args = MockArgs(job_id=None, follow=False)

        result = cmd_cloud_logs(args)

        self.assertEqual(result, 1)


class TestCloudRetryCommand(unittest.TestCase):
    """Test cloud jobs retry command."""

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    @patch("sys.stdout", new_callable=StringIO)
    def test_retry_failed_job(self, mock_stdout, mock_api):
        """Test retrying a failed job."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_retry

        # First call gets the job, second submits
        mock_api.side_effect = [
            {
                "job": {
                    "job_id": "job-123",
                    "status": "failed",
                    "config_name": "test-config",
                    "provider": "replicate",
                }
            },
            {
                "success": True,
                "job_id": "job-456",
                "status": "pending",
            },
        ]

        args = MockArgs(job_id="job-123")

        result = cmd_cloud_retry(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        self.assertIn("resubmitted", output.lower())
        self.assertIn("job-456", output)

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    @patch("sys.stdout", new_callable=StringIO)
    def test_retry_running_job_fails(self, mock_stdout, mock_api):
        """Test retrying a running job is rejected."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_retry

        mock_api.return_value = {
            "job": {
                "job_id": "job-123",
                "status": "running",
                "config_name": "test-config",
            }
        }

        args = MockArgs(job_id="job-123")

        result = cmd_cloud_retry(args)

        self.assertEqual(result, 1)
        output = mock_stdout.getvalue()
        self.assertIn("can only retry", output.lower())


class TestCloudStatusCommand(unittest.TestCase):
    """Test cloud status command."""

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    @patch("sys.stdout", new_callable=StringIO)
    def test_status_healthy(self, mock_stdout, mock_api):
        """Test status when system is healthy."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_status

        mock_api.side_effect = [
            {
                "status": "healthy",
                "uptime_seconds": 86400,
                "components": [
                    {"name": "database", "status": "healthy", "message": "Connected"},
                    {"name": "api", "status": "healthy", "message": "Running", "latency_ms": 50},
                ],
            },
            {"operational": True},
        ]

        args = MockArgs(replicate=False, format="table")

        result = cmd_cloud_status(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        self.assertIn("healthy", output.lower())
        self.assertIn("database", output.lower())

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    @patch("sys.stdout", new_callable=StringIO)
    def test_status_json_format(self, mock_stdout, mock_api):
        """Test status in JSON format."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_status

        health = {"status": "healthy"}
        system = {"operational": True}
        mock_api.side_effect = [health, system]

        args = MockArgs(replicate=False, format="json")

        result = cmd_cloud_status(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        parsed = json.loads(output)
        self.assertIn("health", parsed)
        self.assertIn("system", parsed)


class TestCloudDeleteCommand(unittest.TestCase):
    """Test cloud jobs delete command."""

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    @patch("builtins.input", return_value="y")
    @patch("sys.stdout", new_callable=StringIO)
    def test_delete_with_confirmation(self, mock_stdout, mock_input, mock_api):
        """Test deleting a job with confirmation."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_delete

        mock_api.return_value = {"success": True}

        args = MockArgs(job_id="job-123", force=False)

        result = cmd_cloud_delete(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        self.assertIn("deleted successfully", output.lower())

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    @patch("sys.stdout", new_callable=StringIO)
    def test_delete_force(self, mock_stdout, mock_api):
        """Test force delete without confirmation."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_delete

        mock_api.return_value = {"success": True}

        args = MockArgs(job_id="job-123", force=True)

        result = cmd_cloud_delete(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        self.assertIn("deleted successfully", output.lower())

    @patch("builtins.input", return_value="n")
    @patch("sys.stdout", new_callable=StringIO)
    def test_delete_cancelled(self, mock_stdout, mock_input):
        """Test delete cancelled by user."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_delete

        args = MockArgs(job_id="job-123", force=False)

        result = cmd_cloud_delete(args)

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        self.assertIn("cancelled", output.lower())


class TestCloudCommandDispatcher(unittest.TestCase):
    """Test the cloud command dispatcher."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_no_action_shows_help(self, mock_stdout):
        """Test dispatcher with no action shows help."""
        from simpletuner.cli.cloud import cmd_cloud

        args = MockArgs(cloud_action=None)

        result = cmd_cloud(args)

        self.assertEqual(result, 1)
        output = mock_stdout.getvalue()
        self.assertIn("please specify a cloud command", output.lower())

    @patch("sys.stdout", new_callable=StringIO)
    def test_unknown_action_error(self, mock_stdout):
        """Test dispatcher with unknown action."""
        from simpletuner.cli.cloud import cmd_cloud

        args = MockArgs(cloud_action="unknown-action")

        result = cmd_cloud(args)

        self.assertEqual(result, 1)
        output = mock_stdout.getvalue()
        self.assertIn("unknown cloud command", output.lower())


if __name__ == "__main__":
    unittest.main()
