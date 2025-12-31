"""Tests for Cloud CLI commands.

Tests the CLI interface for cloud training operations including:
- Job management (submit, list, cancel, status)
- Configuration (set-token, config show/set)
- Quotas and cost limits
- Approval workflow
- Notifications

Uses mocking to avoid actual API calls.
"""

import argparse
import unittest
from unittest.mock import MagicMock, patch


class TestCloudAPIRequest(unittest.TestCase):
    """Tests for the cloud_api_request utility function."""

    @patch("urllib.request.urlopen")
    def test_get_request_success(self, mock_urlopen):
        """Test successful GET request."""
        from simpletuner.cli.cloud.api import cloud_api_request

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"jobs": []}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with patch.dict(
            "os.environ",
            {"SIMPLETUNER_HOST": "localhost", "SIMPLETUNER_PORT": "8001"},
        ):
            result = cloud_api_request("GET", "/api/cloud/jobs")

        self.assertEqual(result, {"jobs": []})
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_post_request_with_data(self, mock_urlopen):
        """Test POST request with JSON data."""
        from simpletuner.cli.cloud.api import cloud_api_request

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"job_id": "test-123"}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with patch.dict(
            "os.environ",
            {"SIMPLETUNER_HOST": "localhost", "SIMPLETUNER_PORT": "8001"},
        ):
            result = cloud_api_request("POST", "/api/cloud/jobs", data={"config": "test"})

        self.assertEqual(result, {"job_id": "test-123"})
        # Verify data was included in the request
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        self.assertEqual(request_obj.method, "POST")
        self.assertIn(b'"config"', request_obj.data)

    @patch("urllib.request.urlopen")
    def test_ssl_configuration(self, mock_urlopen):
        """Test SSL configuration from environment."""
        from simpletuner.cli.cloud.api import cloud_api_request

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "ok"}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with patch.dict(
            "os.environ",
            {
                "SIMPLETUNER_HOST": "localhost",
                "SIMPLETUNER_PORT": "8001",
                "SIMPLETUNER_SSL_ENABLED": "true",
            },
        ):
            cloud_api_request("GET", "/api/cloud/status")

        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        self.assertTrue(request_obj.full_url.startswith("https://"))


class TestCloudJobsCommands(unittest.TestCase):
    """Tests for job management CLI commands."""

    def _make_args(self, **kwargs):
        """Create a mock args namespace."""
        defaults = {
            "provider": "replicate",
            "format": "table",
            "limit": 20,
            "status": None,
            "sync": False,
            "dry_run": False,
            "config": None,
            "job_id": None,
            "force": False,
            "follow": False,
            "tail": 100,
            "idempotency_key": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    def test_list_jobs_success(self, mock_api):
        """Test listing jobs successfully."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_list

        # cloud_api_request now returns a dict directly
        mock_api.return_value = {
            "jobs": [
                {
                    "job_id": "job-1",
                    "status": "completed",
                    "config_name": "test",
                    "provider": "replicate",
                    "cost_usd": 5.0,
                    "duration_seconds": 3600,
                },
                {
                    "job_id": "job-2",
                    "status": "running",
                    "config_name": "train",
                    "provider": "replicate",
                    "cost_usd": 2.5,
                    "duration_seconds": 1800,
                },
            ]
        }

        args = self._make_args()
        result = cmd_cloud_list(args)

        self.assertEqual(result, 0)
        mock_api.assert_called_once()

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    def test_list_jobs_with_status_filter(self, mock_api):
        """Test listing jobs with status filter."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_list

        # cloud_api_request now returns a dict directly
        mock_api.return_value = {"jobs": []}

        args = self._make_args(status="completed")
        cmd_cloud_list(args)

        call_args = mock_api.call_args
        self.assertIn("status=completed", call_args[0][1])

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    def test_list_jobs_json_format(self, mock_api):
        """Test listing jobs in JSON format."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_list

        # cloud_api_request now returns a dict directly
        mock_api.return_value = {"jobs": []}

        args = self._make_args(format="json")
        result = cmd_cloud_list(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    def test_cancel_job_success(self, mock_api):
        """Test canceling a job successfully."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_cancel

        # cloud_api_request now returns a dict directly
        mock_api.return_value = {"success": True, "status": "cancelled"}

        args = self._make_args(job_id="job-123")
        result = cmd_cloud_cancel(args)

        self.assertEqual(result, 0)
        mock_api.assert_called_once()
        self.assertIn("job-123", mock_api.call_args[0][1])

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    def test_cancel_job_not_found(self, mock_api):
        """Test canceling a non-existent job."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_cancel

        # cloud_api_request now returns a dict directly
        # When job is not found, it returns success=False
        mock_api.return_value = {"success": False, "error": "Job not found"}

        args = self._make_args(job_id="nonexistent")
        result = cmd_cloud_cancel(args)

        self.assertEqual(result, 1)

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    def test_get_job_details(self, mock_api):
        """Test getting job details."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_get

        # cloud_api_request now returns a dict directly
        mock_api.return_value = {
            "job": {
                "job_id": "job-123",
                "status": "completed",
                "config_name": "test",
                "provider": "replicate",
                "cost_usd": 5.50,
                "job_type": "training",
                "created_at": "2024-01-01T00:00:00Z",
                "started_at": "2024-01-01T00:01:00Z",
                "finished_at": "2024-01-01T01:00:00Z",
                "duration_seconds": 3540,
            }
        }

        args = self._make_args(job_id="job-123")
        result = cmd_cloud_get(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    def test_status_command(self, mock_api):
        """Test cloud status command."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_status

        # cloud_api_request now returns a dict directly
        # cmd_cloud_status makes two API calls: /api/cloud/health and /api/cloud/replicate/status
        def mock_api_response(method, endpoint, **kwargs):
            if "health" in endpoint:
                return {
                    "status": "healthy",
                    "uptime_seconds": 3600,
                    "components": [{"name": "database", "status": "healthy", "message": "OK", "latency_ms": 5}],
                }
            else:
                return {
                    "operational": True,
                    "ongoing_incidents": [],
                    "in_progress_maintenances": [],
                    "scheduled_maintenances": [],
                }

        mock_api.side_effect = mock_api_response

        args = self._make_args()
        result = cmd_cloud_status(args)

        self.assertEqual(result, 0)


class TestCloudConfigCommands(unittest.TestCase):
    """Tests for configuration CLI commands."""

    def _make_args(self, **kwargs):
        """Create a mock args namespace."""
        defaults = {
            "provider": "replicate",
            "format": "table",
            "token": None,
            "key": None,
            "value": None,
            "config_action": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("simpletuner.cli.cloud.config.cloud_api_request")
    def test_config_show(self, mock_api):
        """Test showing provider configuration."""
        from simpletuner.cli.cloud.config import cmd_cloud_config_show

        # cloud_api_request now returns a dict directly
        mock_api.return_value = {
            "config": {
                "provider": "replicate",
                "configured": True,
                "cost_limit_enabled": False,
            }
        }

        args = self._make_args()
        result = cmd_cloud_config_show(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.config.cloud_api_request")
    @patch("getpass.getpass")
    def test_set_token_with_prompt(self, mock_getpass, mock_api):
        """Test setting token with password prompt."""
        from simpletuner.cli.cloud.config import cmd_cloud_config_set_token

        mock_getpass.return_value = "r8_test_token"
        mock_api.return_value = {"success": True, "file_path": "/tmp/token"}

        args = self._make_args(token=None)
        result = cmd_cloud_config_set_token(args)

        self.assertEqual(result, 0)
        mock_getpass.assert_called_once()

    @patch("simpletuner.cli.cloud.config.cloud_api_request")
    def test_set_token_with_argument(self, mock_api):
        """Test setting token from command line argument."""
        from simpletuner.cli.cloud.config import cmd_cloud_config_set_token

        # cloud_api_request now returns a dict directly
        mock_api.return_value = {"success": True, "file_path": "/tmp/token"}

        args = self._make_args(token="r8_direct_token")
        result = cmd_cloud_config_set_token(args)

        self.assertEqual(result, 0)
        # Verify token was sent in request
        call_kwargs = mock_api.call_args[1]
        self.assertEqual(call_kwargs["data"]["token"], "r8_direct_token")

    @patch("simpletuner.cli.cloud.config.cloud_api_request")
    def test_delete_token(self, mock_api):
        """Test deleting API token."""
        from simpletuner.cli.cloud.config import cmd_cloud_config_delete_token

        # cloud_api_request now returns a dict directly
        mock_api.return_value = {"success": True}

        args = self._make_args()
        result = cmd_cloud_config_delete_token(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.config.cloud_api_request")
    def test_config_set_key_value(self, mock_api):
        """Test setting a config key-value pair."""
        from simpletuner.cli.cloud.config import cmd_cloud_config_set

        # cloud_api_request now returns a dict directly
        mock_api.return_value = {"config": {"hardware": "a100"}}

        args = self._make_args(key="hardware", value="a100")
        result = cmd_cloud_config_set(args)

        self.assertEqual(result, 0)


class TestCloudQuotaCommands(unittest.TestCase):
    """Tests for quota management CLI commands."""

    def _make_args(self, **kwargs):
        """Create a mock args namespace."""
        defaults = {
            "provider": "replicate",
            "format": "table",
            "type": None,
            "limit": None,
            "action": "warn",
            "user_id": None,
            "team_id": None,
            "org_id": None,
            "quota_id": None,
            "force": False,
            "quota_action": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("simpletuner.cli.cloud.quota.cloud_api_request")
    def test_quota_list(self, mock_api):
        """Test listing quotas."""
        from simpletuner.cli.cloud.quota import cmd_cloud_quota_list

        # cloud_api_request now returns a dict directly
        mock_api.return_value = {
            "quotas": [
                {"id": 1, "quota_type": "COST_MONTHLY", "limit_value": 100},
            ]
        }

        args = self._make_args()
        result = cmd_cloud_quota_list(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.quota.cloud_api_request")
    def test_quota_create(self, mock_api):
        """Test creating a quota."""
        from simpletuner.cli.cloud.quota import cmd_cloud_quota_create

        # cloud_api_request now returns a dict directly
        # The function checks result.get("quota")
        mock_api.return_value = {"quota": {"id": 1, "quota_type": "COST_MONTHLY", "limit_value": 100}}

        args = self._make_args(type="COST_MONTHLY", limit=100, action="block")
        result = cmd_cloud_quota_create(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.quota.cloud_api_request")
    def test_quota_types(self, mock_api):
        """Test listing quota types."""
        from simpletuner.cli.cloud.quota import cmd_cloud_quota_types

        # cloud_api_request now returns a dict directly
        # The function expects types to be a list of dicts with name, description, unit
        mock_api.return_value = {
            "types": [
                {"name": "COST_MONTHLY", "description": "Monthly cost limit", "unit": "USD"},
                {"name": "COST_DAILY", "description": "Daily cost limit", "unit": "USD"},
                {"name": "CONCURRENT_JOBS", "description": "Concurrent job limit", "unit": "jobs"},
            ]
        }

        args = self._make_args()
        result = cmd_cloud_quota_types(args)

        self.assertEqual(result, 0)


class TestCloudApprovalCommands(unittest.TestCase):
    """Tests for approval workflow CLI commands."""

    def _make_args(self, **kwargs):
        """Create a mock args namespace."""
        defaults = {
            "format": "table",
            "status": None,
            "approval_id": None,
            "reason": None,
            "limit": 20,
            "approval_action": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("simpletuner.cli.cloud.approval.cloud_api_request")
    def test_approval_list(self, mock_api):
        """Test listing approvals."""
        from simpletuner.cli.cloud.approval import cmd_cloud_approval_list

        # cloud_api_request now returns a dict directly
        mock_api.return_value = {
            "approvals": [
                {"id": 1, "status": "pending", "job_id": "job-123"},
            ]
        }

        args = self._make_args()
        result = cmd_cloud_approval_list(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.approval.cloud_api_request")
    def test_approval_pending(self, mock_api):
        """Test listing pending approvals."""
        from simpletuner.cli.cloud.approval import cmd_cloud_approval_pending

        # cloud_api_request now returns a dict directly
        mock_api.return_value = {"approvals": [], "count": 0}

        args = self._make_args()
        result = cmd_cloud_approval_pending(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.approval.cloud_api_request")
    def test_approval_approve(self, mock_api):
        """Test approving a request."""
        from simpletuner.cli.cloud.approval import cmd_cloud_approval_approve

        # cloud_api_request now returns a dict directly
        # The function checks result.get("success")
        mock_api.return_value = {"success": True, "job_id": "job-123"}

        args = self._make_args(approval_id=1, reason="Approved for testing")
        result = cmd_cloud_approval_approve(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.approval.cloud_api_request")
    def test_approval_reject(self, mock_api):
        """Test rejecting a request."""
        from simpletuner.cli.cloud.approval import cmd_cloud_approval_reject

        # cloud_api_request now returns a dict directly
        # The function checks result.get("success")
        mock_api.return_value = {"success": True}

        args = self._make_args(approval_id=1, reason="Cost too high")
        result = cmd_cloud_approval_reject(args)

        self.assertEqual(result, 0)


class TestCloudCostLimitCommand(unittest.TestCase):
    """Tests for cost limit CLI commands."""

    def _make_args(self, **kwargs):
        """Create a mock args namespace."""
        defaults = {
            "provider": "replicate",
            "limit_action": "show",
            "amount": None,
            "period": "monthly",
            "action": "warn",
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("simpletuner.cli.cloud.cost_limit.cloud_api_request")
    def test_cost_limit_show(self, mock_api):
        """Test showing cost limit."""
        from simpletuner.cli.cloud.cost_limit import cmd_cloud_cost_limit

        # cloud_api_request now returns a dict directly
        mock_api.return_value = {
            "config": {
                "cost_limit_enabled": True,
                "cost_limit_amount": 50.0,
                "cost_limit_period": "monthly",
                "cost_limit_action": "warn",
            }
        }

        args = self._make_args(limit_action="show")
        result = cmd_cloud_cost_limit(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.cost_limit.cloud_api_request")
    def test_cost_limit_set(self, mock_api):
        """Test setting cost limit."""
        from simpletuner.cli.cloud.cost_limit import cmd_cloud_cost_limit

        # cloud_api_request now returns a dict directly
        mock_api.return_value = {"config": {"cost_limit_enabled": True, "cost_limit_amount": 100.0}}

        args = self._make_args(limit_action="set", amount=100, period="monthly")
        result = cmd_cloud_cost_limit(args)

        self.assertEqual(result, 0)


class TestCloudAuditCommand(unittest.TestCase):
    """Tests for audit log CLI commands."""

    def _make_args(self, **kwargs):
        """Create a mock args namespace."""
        defaults = {
            "audit_action": "list",
            "event_type": None,
            "actor_id": None,
            "target_type": None,
            "target_id": None,
            "since": None,
            "until": None,
            "limit": 50,
            "offset": 0,
            "format": "table",
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("simpletuner.cli.cloud.audit.cloud_api_request")
    def test_audit_list(self, mock_api):
        """Test listing audit entries."""
        from simpletuner.cli.cloud.audit import cmd_cloud_audit

        # cloud_api_request now returns a dict directly
        mock_api.return_value = {
            "entries": [
                {"id": 1, "event_type": "login", "actor_id": "user-1", "timestamp": "2024-01-01T00:00:00Z"},
            ],
            "total": 1,
        }

        args = self._make_args(audit_action="list")
        result = cmd_cloud_audit(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.audit.cloud_api_request")
    def test_audit_stats(self, mock_api):
        """Test getting audit statistics."""
        from simpletuner.cli.cloud.audit import cmd_cloud_audit

        # cloud_api_request now returns a dict directly
        mock_api.return_value = {
            "total_entries": 100,
            "last_24h": 10,
            "security_events_24h": 2,
            "first_entry": "2024-01-01T00:00:00Z",
            "last_entry": "2024-01-15T00:00:00Z",
            "by_event_type": {"login": 50, "logout": 30, "job_start": 20},
        }

        args = self._make_args(audit_action="stats")
        result = cmd_cloud_audit(args)

        self.assertEqual(result, 0)


class TestFormatHelpers(unittest.TestCase):
    """Tests for format helper functions."""

    def test_format_job_status_completed(self):
        """Test formatting completed status."""
        from simpletuner.cli.cloud.api import format_job_status

        result = format_job_status("completed")
        self.assertIn("completed", result.lower())

    def test_format_job_status_failed(self):
        """Test formatting failed status."""
        from simpletuner.cli.cloud.api import format_job_status

        result = format_job_status("failed")
        self.assertIn("failed", result.lower())

    def test_format_job_status_running(self):
        """Test formatting running status."""
        from simpletuner.cli.cloud.api import format_job_status

        result = format_job_status("running")
        self.assertIn("running", result.lower())

    def test_format_bool_true(self):
        """Test formatting boolean True."""
        from simpletuner.cli.cloud.api import format_bool

        result = format_bool(True)
        self.assertIn("yes", result.lower())

    def test_format_bool_false(self):
        """Test formatting boolean False."""
        from simpletuner.cli.cloud.api import format_bool

        result = format_bool(False)
        self.assertIn("no", result.lower())


if __name__ == "__main__":
    unittest.main()
