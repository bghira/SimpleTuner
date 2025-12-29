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

    @patch("simpletuner.cli.cloud.api.requests.request")
    def test_get_request_success(self, mock_request):
        """Test successful GET request."""
        from simpletuner.cli.cloud.api import cloud_api_request

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jobs": []}
        mock_request.return_value = mock_response

        with patch.dict(
            "os.environ",
            {"SIMPLETUNER_HOST": "localhost", "SIMPLETUNER_PORT": "8001"},
        ):
            result = cloud_api_request("GET", "/api/cloud/jobs")

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json(), {"jobs": []})
        mock_request.assert_called_once()

    @patch("simpletuner.cli.cloud.api.requests.request")
    def test_post_request_with_data(self, mock_request):
        """Test POST request with JSON data."""
        from simpletuner.cli.cloud.api import cloud_api_request

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"job_id": "test-123"}
        mock_request.return_value = mock_response

        with patch.dict(
            "os.environ",
            {"SIMPLETUNER_HOST": "localhost", "SIMPLETUNER_PORT": "8001"},
        ):
            result = cloud_api_request("POST", "/api/cloud/jobs", data={"config": "test"})

        self.assertEqual(result.status_code, 201)
        call_kwargs = mock_request.call_args[1]
        self.assertEqual(call_kwargs["json"], {"config": "test"})

    @patch("simpletuner.cli.cloud.api.requests.request")
    def test_ssl_configuration(self, mock_request):
        """Test SSL configuration from environment."""
        from simpletuner.cli.cloud.api import cloud_api_request

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        with patch.dict(
            "os.environ",
            {
                "SIMPLETUNER_HOST": "localhost",
                "SIMPLETUNER_PORT": "8001",
                "SIMPLETUNER_SSL_ENABLED": "true",
            },
        ):
            cloud_api_request("GET", "/api/cloud/status")

        call_args = mock_request.call_args
        self.assertTrue(call_args[1]["url"].startswith("https://"))


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

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "jobs": [
                {"id": "job-1", "status": "completed", "config_name": "test"},
                {"id": "job-2", "status": "running", "config_name": "train"},
            ]
        }
        mock_api.return_value = mock_response

        args = self._make_args()
        result = cmd_cloud_list(args)

        self.assertEqual(result, 0)
        mock_api.assert_called_once()

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    def test_list_jobs_with_status_filter(self, mock_api):
        """Test listing jobs with status filter."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_list

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"jobs": []}
        mock_api.return_value = mock_response

        args = self._make_args(status="completed")
        cmd_cloud_list(args)

        call_args = mock_api.call_args
        self.assertIn("status=completed", call_args[0][1])

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    def test_list_jobs_json_format(self, mock_api):
        """Test listing jobs in JSON format."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_list

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"jobs": []}
        mock_api.return_value = mock_response

        args = self._make_args(format="json")
        result = cmd_cloud_list(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    def test_cancel_job_success(self, mock_api):
        """Test canceling a job successfully."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_cancel

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"status": "cancelled"}
        mock_api.return_value = mock_response

        args = self._make_args(job_id="job-123")
        result = cmd_cloud_cancel(args)

        self.assertEqual(result, 0)
        mock_api.assert_called_once()
        self.assertIn("job-123", mock_api.call_args[0][1])

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    def test_cancel_job_not_found(self, mock_api):
        """Test canceling a non-existent job."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_cancel

        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Job not found"}
        mock_api.return_value = mock_response

        args = self._make_args(job_id="nonexistent")
        result = cmd_cloud_cancel(args)

        self.assertEqual(result, 1)

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    def test_get_job_details(self, mock_api):
        """Test getting job details."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_get

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "job-123",
            "status": "completed",
            "config_name": "test",
            "provider": "replicate",
            "cost_usd": 5.50,
        }
        mock_api.return_value = mock_response

        args = self._make_args(job_id="job-123")
        result = cmd_cloud_get(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.jobs.cloud_api_request")
    def test_status_command(self, mock_api):
        """Test cloud status command."""
        from simpletuner.cli.cloud.jobs import cmd_cloud_status

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "status": "healthy",
            "provider": "replicate",
            "configured": True,
        }
        mock_api.return_value = mock_response

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

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "provider": "replicate",
            "configured": True,
            "cost_limit_enabled": False,
        }
        mock_api.return_value = mock_response

        args = self._make_args()
        result = cmd_cloud_config_show(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.config.cloud_api_request")
    @patch("simpletuner.cli.cloud.config.getpass")
    def test_set_token_with_prompt(self, mock_getpass, mock_api):
        """Test setting token with password prompt."""
        from simpletuner.cli.cloud.config import cmd_cloud_config_set_token

        mock_getpass.getpass.return_value = "r8_test_token"
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"status": "saved"}
        mock_api.return_value = mock_response

        args = self._make_args(token=None)
        result = cmd_cloud_config_set_token(args)

        self.assertEqual(result, 0)
        mock_getpass.getpass.assert_called_once()

    @patch("simpletuner.cli.cloud.config.cloud_api_request")
    def test_set_token_with_argument(self, mock_api):
        """Test setting token from command line argument."""
        from simpletuner.cli.cloud.config import cmd_cloud_config_set_token

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"status": "saved"}
        mock_api.return_value = mock_response

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

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"status": "deleted"}
        mock_api.return_value = mock_response

        args = self._make_args()
        result = cmd_cloud_config_delete_token(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.config.cloud_api_request")
    def test_config_set_key_value(self, mock_api):
        """Test setting a config key-value pair."""
        from simpletuner.cli.cloud.config import cmd_cloud_config_set

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"status": "updated"}
        mock_api.return_value = mock_response

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

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "quotas": [
                {"id": 1, "quota_type": "COST_MONTHLY", "limit_value": 100},
            ]
        }
        mock_api.return_value = mock_response

        args = self._make_args()
        result = cmd_cloud_quota_list(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.quota.cloud_api_request")
    def test_quota_create(self, mock_api):
        """Test creating a quota."""
        from simpletuner.cli.cloud.quota import cmd_cloud_quota_create

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": 1, "quota_type": "COST_MONTHLY"}
        mock_api.return_value = mock_response

        args = self._make_args(type="COST_MONTHLY", limit=100, action="block")
        result = cmd_cloud_quota_create(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.quota.cloud_api_request")
    def test_quota_types(self, mock_api):
        """Test listing quota types."""
        from simpletuner.cli.cloud.quota import cmd_cloud_quota_types

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"types": ["COST_MONTHLY", "COST_DAILY", "CONCURRENT_JOBS"]}
        mock_api.return_value = mock_response

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

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "approvals": [
                {"id": 1, "status": "pending", "job_id": "job-123"},
            ]
        }
        mock_api.return_value = mock_response

        args = self._make_args()
        result = cmd_cloud_approval_list(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.approval.cloud_api_request")
    def test_approval_pending(self, mock_api):
        """Test listing pending approvals."""
        from simpletuner.cli.cloud.approval import cmd_cloud_approval_pending

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"approvals": [], "count": 0}
        mock_api.return_value = mock_response

        args = self._make_args()
        result = cmd_cloud_approval_pending(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.approval.cloud_api_request")
    def test_approval_approve(self, mock_api):
        """Test approving a request."""
        from simpletuner.cli.cloud.approval import cmd_cloud_approval_approve

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"status": "approved"}
        mock_api.return_value = mock_response

        args = self._make_args(approval_id=1, reason="Approved for testing")
        result = cmd_cloud_approval_approve(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.approval.cloud_api_request")
    def test_approval_reject(self, mock_api):
        """Test rejecting a request."""
        from simpletuner.cli.cloud.approval import cmd_cloud_approval_reject

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"status": "rejected"}
        mock_api.return_value = mock_response

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

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "cost_limit_enabled": True,
            "cost_limit_amount": 50.0,
            "cost_limit_period": "monthly",
        }
        mock_api.return_value = mock_response

        args = self._make_args(limit_action="show")
        result = cmd_cloud_cost_limit(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.cost_limit.cloud_api_request")
    def test_cost_limit_set(self, mock_api):
        """Test setting cost limit."""
        from simpletuner.cli.cloud.cost_limit import cmd_cloud_cost_limit

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"status": "updated"}
        mock_api.return_value = mock_response

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

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "entries": [
                {"id": 1, "event_type": "login", "actor_id": "user-1"},
            ],
            "total": 1,
        }
        mock_api.return_value = mock_response

        args = self._make_args(audit_action="list")
        result = cmd_cloud_audit(args)

        self.assertEqual(result, 0)

    @patch("simpletuner.cli.cloud.audit.cloud_api_request")
    def test_audit_stats(self, mock_api):
        """Test getting audit statistics."""
        from simpletuner.cli.cloud.audit import cmd_cloud_audit

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "total_entries": 100,
            "last_24h": 10,
            "security_events_24h": 2,
        }
        mock_api.return_value = mock_response

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
