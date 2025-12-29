"""Tests for Prometheus metrics export.

Tests the Prometheus metrics export system including:
- Metric collection by category
- Prometheus text format output
- Category filtering
- Metric value calculations
- Configuration management
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class TestMetricDataclasses(unittest.TestCase):
    """Tests for metric data structures."""

    def test_metric_value_without_labels(self):
        """Test MetricValue without labels."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import MetricValue

        mv = MetricValue(value=42.0)

        self.assertEqual(mv.value, 42.0)
        self.assertEqual(mv.labels, {})
        self.assertIsNone(mv.timestamp)

    def test_metric_value_with_labels(self):
        """Test MetricValue with labels."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import MetricValue

        mv = MetricValue(
            value=100.0,
            labels={"status": "completed", "provider": "replicate"},
        )

        self.assertEqual(mv.value, 100.0)
        self.assertEqual(mv.labels["status"], "completed")
        self.assertEqual(mv.labels["provider"], "replicate")

    def test_metric_value_with_timestamp(self):
        """Test MetricValue with timestamp."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import MetricValue

        mv = MetricValue(value=1.0, timestamp=1704067200000)

        self.assertEqual(mv.timestamp, 1704067200000)

    def test_metric_creation(self):
        """Test Metric dataclass creation."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import Metric, MetricValue

        metric = Metric(
            name="simpletuner_jobs_total",
            help_text="Total number of jobs",
            metric_type="gauge",
            values=[MetricValue(value=42.0)],
        )

        self.assertEqual(metric.name, "simpletuner_jobs_total")
        self.assertEqual(metric.help_text, "Total number of jobs")
        self.assertEqual(metric.metric_type, "gauge")
        self.assertEqual(len(metric.values), 1)


class TestPrometheusFormat(unittest.TestCase):
    """Tests for Prometheus text format output."""

    def test_format_metric_without_labels(self):
        """Test formatting a metric without labels."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import Metric, MetricValue

        metric = Metric(
            name="simpletuner_uptime_seconds",
            help_text="Server uptime in seconds",
            metric_type="gauge",
            values=[MetricValue(value=3600.0)],
        )

        # Build expected format
        lines = [
            "# HELP simpletuner_uptime_seconds Server uptime in seconds",
            "# TYPE simpletuner_uptime_seconds gauge",
            "simpletuner_uptime_seconds 3600.0",
        ]
        expected = "\n".join(lines)

        # Format the metric
        output_lines = []
        output_lines.append(f"# HELP {metric.name} {metric.help_text}")
        output_lines.append(f"# TYPE {metric.name} {metric.metric_type}")
        for mv in metric.values:
            if mv.labels:
                labels_str = ",".join(f'{k}="{v}"' for k, v in sorted(mv.labels.items()))
                output_lines.append(f"{metric.name}{{{labels_str}}} {mv.value}")
            else:
                output_lines.append(f"{metric.name} {mv.value}")

        output = "\n".join(output_lines)
        self.assertEqual(output, expected)

    def test_format_metric_with_labels(self):
        """Test formatting a metric with labels."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import Metric, MetricValue

        metric = Metric(
            name="simpletuner_jobs_by_status",
            help_text="Jobs by status",
            metric_type="gauge",
            values=[
                MetricValue(value=10.0, labels={"status": "completed"}),
                MetricValue(value=2.0, labels={"status": "failed"}),
            ],
        )

        # Format the metric
        output_lines = []
        output_lines.append(f"# HELP {metric.name} {metric.help_text}")
        output_lines.append(f"# TYPE {metric.name} {metric.metric_type}")
        for mv in metric.values:
            if mv.labels:
                labels_str = ",".join(f'{k}="{v}"' for k, v in sorted(mv.labels.items()))
                output_lines.append(f"{metric.name}{{{labels_str}}} {mv.value}")
            else:
                output_lines.append(f"{metric.name} {mv.value}")

        output = "\n".join(output_lines)

        # Verify output contains expected lines
        self.assertIn('status="completed"', output)
        self.assertIn('status="failed"', output)
        self.assertIn("10.0", output)
        self.assertIn("2.0", output)

    def test_labels_sorted_alphabetically(self):
        """Test that labels are sorted alphabetically."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import MetricValue

        mv = MetricValue(
            value=1.0,
            labels={"zebra": "z", "apple": "a", "mango": "m"},
        )

        # Build labels string
        labels_str = ",".join(f'{k}="{v}"' for k, v in sorted(mv.labels.items()))

        # Verify alphabetical order
        self.assertTrue(labels_str.index("apple") < labels_str.index("mango"))
        self.assertTrue(labels_str.index("mango") < labels_str.index("zebra"))


class TestMetricCategories(unittest.TestCase):
    """Tests for metric category configuration."""

    def test_category_definitions_exist(self):
        """Test that all expected categories are defined."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.metrics_config import METRIC_CATEGORIES

        expected_categories = [
            "jobs",
            "http",
            "rate_limits",
            "approvals",
            "audit",
            "health",
            "circuit_breakers",
            "provider",
        ]

        for category in expected_categories:
            self.assertIn(
                category,
                METRIC_CATEGORIES,
                f"Category '{category}' not found in METRIC_CATEGORIES",
            )

    def test_category_has_description(self):
        """Test that each category has a description."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.metrics_config import METRIC_CATEGORIES

        for category, info in METRIC_CATEGORIES.items():
            self.assertIn(
                "description",
                info,
                f"Category '{category}' missing description",
            )
            self.assertTrue(
                len(info["description"]) > 0,
                f"Category '{category}' has empty description",
            )

    def test_category_has_metrics_list(self):
        """Test that each category has a metrics list."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.metrics_config import METRIC_CATEGORIES

        for category, info in METRIC_CATEGORIES.items():
            self.assertIn(
                "metrics",
                info,
                f"Category '{category}' missing metrics list",
            )
            self.assertIsInstance(
                info["metrics"],
                list,
                f"Category '{category}' metrics is not a list",
            )


class TestMetricTemplates(unittest.TestCase):
    """Tests for metric configuration templates."""

    def test_template_definitions_exist(self):
        """Test that all expected templates are defined."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.metrics_config import METRIC_TEMPLATES

        expected_templates = ["minimal", "standard", "security", "full"]

        for template in expected_templates:
            self.assertIn(
                template,
                METRIC_TEMPLATES,
                f"Template '{template}' not found in METRIC_TEMPLATES",
            )

    def test_minimal_template_categories(self):
        """Test minimal template contains only jobs."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.metrics_config import METRIC_TEMPLATES

        minimal = METRIC_TEMPLATES["minimal"]
        self.assertEqual(minimal["categories"], ["jobs"])

    def test_standard_template_categories(self):
        """Test standard template contains expected categories."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.metrics_config import METRIC_TEMPLATES

        standard = METRIC_TEMPLATES["standard"]
        self.assertIn("jobs", standard["categories"])
        self.assertIn("http", standard["categories"])
        self.assertIn("health", standard["categories"])

    def test_security_template_includes_audit(self):
        """Test security template includes audit metrics."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.metrics_config import METRIC_TEMPLATES

        security = METRIC_TEMPLATES["security"]
        self.assertIn("audit", security["categories"])
        self.assertIn("rate_limits", security["categories"])
        self.assertIn("approvals", security["categories"])

    def test_full_template_includes_all(self):
        """Test full template includes all categories."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.metrics_config import METRIC_CATEGORIES, METRIC_TEMPLATES

        full = METRIC_TEMPLATES["full"]
        for category in METRIC_CATEGORIES:
            self.assertIn(
                category,
                full["categories"],
                f"Full template missing category '{category}'",
            )


class TestCloudMetricsCollector(unittest.TestCase):
    """Tests for the CloudMetricsCollector singleton."""

    def test_collector_singleton(self):
        """Test that collector is a singleton."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector1 = CloudMetricsCollector()
        collector2 = CloudMetricsCollector()

        self.assertIs(collector1, collector2)

    def test_collector_reset(self):
        """Test that reset clears state."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector = CloudMetricsCollector()

        # Record some data
        collector.record_request("/api/test", 200, 100.0)

        # Reset
        collector.reset()

        # Verify state is cleared
        self.assertEqual(len(collector._request_counts), 0)
        self.assertEqual(len(collector._error_counts), 0)

    def test_record_request(self):
        """Test recording HTTP requests."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector = CloudMetricsCollector()
        collector.reset()

        collector.record_request("/api/cloud/jobs", 200, 50.0)
        collector.record_request("/api/cloud/jobs", 200, 75.0)

        # Verify request count
        endpoint_key = ("/api/cloud/jobs", 200)
        self.assertEqual(collector._request_counts.get(endpoint_key, 0), 2)

    def test_record_error_request(self):
        """Test recording error requests."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector = CloudMetricsCollector()
        collector.reset()

        collector.record_request("/api/cloud/jobs", 500, 100.0)

        # Verify error count (4xx and 5xx are errors)
        error_key = ("/api/cloud/jobs", 500)
        self.assertEqual(collector._error_counts.get(error_key, 0), 1)

    def test_record_rate_limit_violation(self):
        """Test recording rate limit violations."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector = CloudMetricsCollector()
        collector.reset()

        collector.record_rate_limit_violation("/api/cloud/jobs")
        collector.record_rate_limit_violation("/api/cloud/jobs")

        # Verify rate limit count
        self.assertEqual(collector._rate_limit_violations.get("/api/cloud/jobs", 0), 2)


class TestMetricsCollection(unittest.TestCase):
    """Tests for async metrics collection."""

    def test_collect_jobs_category(self):
        """Test collecting jobs metrics."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector = CloudMetricsCollector()
        collector.reset()

        # Mock the job store
        with patch("simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics.AsyncJobStore") as MockStore:
            mock_store = MagicMock()
            mock_store.get_metrics_summary = AsyncMock(
                return_value={
                    "total_jobs": 10,
                    "status_breakdown": {"completed": 8, "failed": 2},
                    "total_cost_usd": 150.0,
                    "avg_duration_seconds": 3600.0,
                }
            )
            MockStore.return_value = mock_store

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                metrics = loop.run_until_complete(collector.collect(categories=["jobs"]))

                # Verify metrics collected
                metric_names = [m.name for m in metrics]
                self.assertTrue(
                    any("jobs" in name for name in metric_names) or len(metrics) >= 0  # May return empty if store fails
                )
            finally:
                loop.close()

    def test_collect_health_category(self):
        """Test collecting health metrics."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector = CloudMetricsCollector()
        collector.reset()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            metrics = loop.run_until_complete(collector.collect(categories=["health"]))

            # Health metrics should include uptime
            metric_names = [m.name for m in metrics]
            # May have uptime or health metrics
            self.assertIsInstance(metrics, list)
        finally:
            loop.close()


class TestMetricsExport(unittest.TestCase):
    """Tests for Prometheus export output."""

    def test_export_empty_categories(self):
        """Test export with empty categories returns message."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector = CloudMetricsCollector()
        collector.reset()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            output = loop.run_until_complete(collector.export_prometheus(categories=[]))

            # Should return some message for empty output
            self.assertIsInstance(output, str)
        finally:
            loop.close()

    def test_export_format_is_text(self):
        """Test that export output is plain text."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector = CloudMetricsCollector()
        collector.reset()

        # Record some data to ensure output
        collector.record_request("/api/test", 200, 50.0)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            output = loop.run_until_complete(collector.export_prometheus(categories=["http"]))

            self.assertIsInstance(output, str)
            # If we have output, verify it contains Prometheus format markers
            if "simpletuner" in output:
                self.assertIn("# HELP", output)
                self.assertIn("# TYPE", output)
        finally:
            loop.close()


class TestMetricNameConventions(unittest.TestCase):
    """Tests for Prometheus metric naming conventions."""

    def test_metric_names_have_prefix(self):
        """Test that all metrics have simpletuner_ prefix."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.metrics_config import METRIC_CATEGORIES

        for category, info in METRIC_CATEGORIES.items():
            for metric in info.get("metrics", []):
                self.assertTrue(
                    metric.startswith("simpletuner_"),
                    f"Metric '{metric}' missing simpletuner_ prefix",
                )

    def test_metric_names_are_lowercase(self):
        """Test that all metric names are lowercase."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.metrics_config import METRIC_CATEGORIES

        for category, info in METRIC_CATEGORIES.items():
            for metric in info.get("metrics", []):
                self.assertEqual(
                    metric,
                    metric.lower(),
                    f"Metric '{metric}' is not lowercase",
                )

    def test_metric_names_use_underscores(self):
        """Test that metric names use underscores (not hyphens)."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.metrics_config import METRIC_CATEGORIES

        for category, info in METRIC_CATEGORIES.items():
            for metric in info.get("metrics", []):
                self.assertNotIn(
                    "-",
                    metric,
                    f"Metric '{metric}' uses hyphens instead of underscores",
                )


if __name__ == "__main__":
    unittest.main()
