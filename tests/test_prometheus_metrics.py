"""Tests for Prometheus-compatible metrics export.

Tests the prometheus_metrics module:
- Metric formatting in Prometheus text format
- MetricValue label serialization
- CloudMetricsCollector request tracking
- Rate limit violation tracking
- Category-based metric collection
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class TestMetricValue(unittest.TestCase):
    """Test MetricValue dataclass."""

    def test_metric_value_with_labels(self):
        """Test MetricValue stores labels correctly."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import MetricValue

        mv = MetricValue(value=42.5, labels={"status": "running", "provider": "replicate"})

        self.assertEqual(mv.value, 42.5)
        self.assertEqual(mv.labels["status"], "running")
        self.assertEqual(mv.labels["provider"], "replicate")

    def test_metric_value_with_timestamp(self):
        """Test MetricValue with timestamp."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import MetricValue

        mv = MetricValue(value=100, timestamp_ms=1609459200000)

        self.assertEqual(mv.timestamp_ms, 1609459200000)


class TestMetric(unittest.TestCase):
    """Test Metric dataclass and Prometheus formatting."""

    def test_to_prometheus_simple_metric(self):
        """Test Prometheus format for simple metric without labels."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import Metric, MetricValue

        metric = Metric(
            name="test_counter",
            help_text="A test counter",
            metric_type="counter",
            values=[MetricValue(value=42)],
        )

        output = metric.to_prometheus()

        self.assertIn("# HELP test_counter A test counter", output)
        self.assertIn("# TYPE test_counter counter", output)
        self.assertIn("test_counter 42", output)

    def test_to_prometheus_with_labels(self):
        """Test Prometheus format with labels."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import Metric, MetricValue

        metric = Metric(
            name="http_requests_total",
            help_text="Total HTTP requests",
            metric_type="counter",
            values=[
                MetricValue(value=100, labels={"method": "GET", "status": "200"}),
                MetricValue(value=5, labels={"method": "POST", "status": "500"}),
            ],
        )

        output = metric.to_prometheus()

        self.assertIn('http_requests_total{method="GET",status="200"} 100', output)
        self.assertIn('http_requests_total{method="POST",status="500"} 5', output)

    def test_to_prometheus_with_timestamp(self):
        """Test Prometheus format includes timestamp when present."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import Metric, MetricValue

        metric = Metric(
            name="gauge_metric",
            help_text="A gauge",
            metric_type="gauge",
            values=[MetricValue(value=3.14, timestamp_ms=1609459200000)],
        )

        output = metric.to_prometheus()

        self.assertIn("gauge_metric 3.14 1609459200000", output)

    def test_to_prometheus_label_sorting(self):
        """Test labels are sorted alphabetically for consistency."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import Metric, MetricValue

        metric = Metric(
            name="test",
            help_text="Test",
            metric_type="gauge",
            values=[MetricValue(value=1, labels={"z": "last", "a": "first"})],
        )

        output = metric.to_prometheus()

        # Labels should be alphabetically sorted
        self.assertIn('test{a="first",z="last"} 1', output)


class TestCloudMetricsCollector(unittest.TestCase):
    """Test CloudMetricsCollector class."""

    def setUp(self):
        """Reset singleton for each test."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        # Reset singleton state
        CloudMetricsCollector._instance = None
        CloudMetricsCollector._request_counts = {}
        CloudMetricsCollector._error_counts = {}
        CloudMetricsCollector._latency_sums = {}
        CloudMetricsCollector._latency_counts = {}
        CloudMetricsCollector._rate_limit_violations = {}

    def test_singleton_pattern(self):
        """Test CloudMetricsCollector is a singleton."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector1 = CloudMetricsCollector()
        collector2 = CloudMetricsCollector()

        self.assertIs(collector1, collector2)

    def test_record_request(self):
        """Test recording HTTP request."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector = CloudMetricsCollector()
        collector.record_request("/api/jobs", "GET", 200, 45.5)
        collector.record_request("/api/jobs", "GET", 200, 55.5)

        self.assertEqual(collector._request_counts["GET_/api/jobs"], 2)
        self.assertAlmostEqual(collector._latency_sums["GET_/api/jobs"], 101.0)
        self.assertEqual(collector._latency_counts["GET_/api/jobs"], 2)

    def test_record_request_error(self):
        """Test recording HTTP error request."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector = CloudMetricsCollector()
        collector.record_request("/api/jobs", "POST", 500, 100.0)
        collector.record_request("/api/jobs", "POST", 404, 50.0)

        self.assertEqual(collector._error_counts["POST_/api/jobs_500"], 1)
        self.assertEqual(collector._error_counts["POST_/api/jobs_404"], 1)

    def test_record_rate_limit_violation(self):
        """Test recording rate limit violation."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector = CloudMetricsCollector()
        collector.record_rate_limit_violation("/api/jobs/submit")
        collector.record_rate_limit_violation("/api/jobs/submit")
        collector.record_rate_limit_violation("/api/jobs")

        self.assertEqual(collector._rate_limit_violations["/api/jobs/submit"], 2)
        self.assertEqual(collector._rate_limit_violations["/api/jobs"], 1)

    def test_collect_http_metrics(self):
        """Test collecting HTTP metrics."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector = CloudMetricsCollector()
        collector.record_request("/api/test", "GET", 200, 50.0)

        metrics = collector._collect_http_metrics()

        # Should have request count and latency metrics
        metric_names = [m.name for m in metrics]
        self.assertIn("simpletuner_http_requests_total", metric_names)

    def test_collect_rate_limit_metrics(self):
        """Test collecting rate limit metrics."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector = CloudMetricsCollector()
        collector.record_rate_limit_violation("/api/test")

        metrics = collector._collect_rate_limit_metrics()

        metric_names = [m.name for m in metrics]
        self.assertIn("simpletuner_rate_limit_violations_total", metric_names)


class TestMetricCollection(unittest.TestCase):
    """Test async metric collection."""

    def setUp(self):
        """Reset singleton for each test."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        CloudMetricsCollector._instance = None
        CloudMetricsCollector._request_counts = {}
        CloudMetricsCollector._error_counts = {}
        CloudMetricsCollector._latency_sums = {}
        CloudMetricsCollector._latency_counts = {}
        CloudMetricsCollector._rate_limit_violations = {}

    def test_collect_health_metrics(self):
        """Test collecting health metrics returns uptime."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector = CloudMetricsCollector()

        # Mock the container.get_job_store since it's imported inside the function
        with patch("simpletuner.simpletuner_sdk.server.services.cloud.container.get_job_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.list_jobs = AsyncMock(return_value=[])
            mock_get_store.return_value = mock_store

            metrics = asyncio.get_event_loop().run_until_complete(collector._collect_health_metrics())

        metric_names = [m.name for m in metrics]
        self.assertIn("simpletuner_uptime_seconds", metric_names)

    def test_collect_by_categories(self):
        """Test collecting specific categories."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector = CloudMetricsCollector()
        collector.record_request("/api/test", "GET", 200, 50.0)

        # Collect only http and rate_limits
        metrics = asyncio.get_event_loop().run_until_complete(collector.collect(categories=["http", "rate_limits"]))

        metric_names = [m.name for m in metrics]

        # Should have HTTP metrics
        self.assertIn("simpletuner_http_requests_total", metric_names)

        # Should NOT have health metrics (not in requested categories)
        self.assertNotIn("simpletuner_uptime_seconds", metric_names)

    def test_collect_invalid_category_ignored(self):
        """Test invalid categories are ignored."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector = CloudMetricsCollector()

        # Should not raise with invalid category
        metrics = asyncio.get_event_loop().run_until_complete(collector.collect(categories=["invalid_category"]))

        # Should return empty or only valid category metrics
        self.assertIsInstance(metrics, list)


class TestAllCategories(unittest.TestCase):
    """Test ALL_CATEGORIES constant."""

    def test_all_categories_defined(self):
        """Test all expected categories are defined."""
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import ALL_CATEGORIES

        expected = {
            "jobs",
            "http",
            "rate_limits",
            "approvals",
            "audit",
            "health",
            "circuit_breakers",
            "provider",
        }

        self.assertEqual(ALL_CATEGORIES, expected)


if __name__ == "__main__":
    unittest.main()
