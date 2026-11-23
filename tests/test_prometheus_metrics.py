"""Tests for Prometheus-compatible metrics export.

Tests the prometheus_metrics module:
- Metric formatting in Prometheus text format
- MetricValue label serialization
- CloudMetricsCollector request tracking
- Rate limit violation tracking
- Category-based metric collection
"""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from tests.unittest_support import APITestCase, create_router_client, run_async


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

    def test_to_prometheus_escapes_label_values(self):
        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import Metric, MetricValue

        metric = Metric("example", "An example", "gauge", [MetricValue(1, {"job_id": 'run"\\a\nb'})])
        self.assertIn(r'example{job_id="run\"\\a\nb"} 1', metric.to_prometheus())

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

            metrics = asyncio.run(collector._collect_health_metrics())

        metric_names = [m.name for m in metrics]
        self.assertIn("simpletuner_uptime_seconds", metric_names)

    def test_collect_by_categories(self):
        """Test collecting specific categories."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.prometheus_metrics import CloudMetricsCollector

        collector = CloudMetricsCollector()
        collector.record_request("/api/test", "GET", 200, 50.0)

        # Collect only http and rate_limits
        metrics = asyncio.run(collector.collect(categories=["http", "rate_limits"]))

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
        metrics = asyncio.run(collector.collect(categories=["invalid_category"]))

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
            "workers",
        }

        self.assertEqual(ALL_CATEGORIES, expected)


class WorkerPrometheusExportTests(APITestCase, unittest.TestCase):
    def test_scrape_and_preview_export_only_visible_worker_samples(self):
        from simpletuner.simpletuner_sdk.server.models.worker import Worker, WorkerStatus, WorkerType
        from simpletuner.simpletuner_sdk.server.routes.cloud import router as cloud_router
        from simpletuner.simpletuner_sdk.server.routes.metrics import router as metrics_router
        from simpletuner.simpletuner_sdk.server.routes.workers import router as workers_router
        from simpletuner.simpletuner_sdk.server.services.cloud.auth import User, get_current_user, get_optional_user
        from simpletuner.simpletuner_sdk.server.services.cloud.base import JobType, UnifiedJob
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository import JobRepository
        from simpletuner.simpletuner_sdk.server.services.worker_repository import WorkerRepository

        user = User(id=1, username="scraper", email="scraper@example.com")
        now = datetime.now(timezone.utc)

        with patch.dict(BaseSQLiteStore._instances, clear=True):
            workers = WorkerRepository(self.tmp_path / "workers.db")
            jobs = JobRepository(self.tmp_path / "jobs.db")
            with (
                patch("simpletuner.simpletuner_sdk.server.services.worker_repository._worker_repository", workers),
                patch("simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository._job_repository", jobs),
                patch.dict("os.environ", {"SIMPLETUNER_WEB_UI_CONFIG": str(self.tmp_path / "webui")}),
                create_router_client(metrics_router, workers_router, cloud_router) as client,
            ):
                client.app.dependency_overrides[get_current_user] = lambda: user
                client.app.dependency_overrides[get_optional_user] = lambda: user
                for owner in (1, 2):
                    worker_id, job_id, token = f"metrics-worker-{owner}", f"metrics-job-{owner}", f"token-{owner}"
                    run_async(
                        workers.create_worker(
                            Worker(
                                worker_id=worker_id,
                                name=f"Worker {owner}",
                                worker_type=WorkerType.PERSISTENT,
                                status=WorkerStatus.BUSY,
                                token_hash=WorkerRepository.hash_token(token),
                                user_id=owner,
                                current_job_id=job_id,
                            )
                        )
                    )
                    run_async(
                        jobs.add(
                            UnifiedJob(
                                job_id=job_id,
                                job_type=JobType.LOCAL,
                                provider="worker",
                                status="running",
                                config_name="shared",
                                created_at=now.isoformat(),
                                metadata={"worker_id": worker_id},
                                user_id=owner,
                            )
                        )
                    )
                    headers = {"X-Worker-Token": token}
                    client.post(
                        "/api/workers/heartbeat",
                        headers=headers,
                        json={
                            "worker_id": worker_id,
                            "current_job_id": job_id,
                            "status": "busy",
                            "system_metrics": {"system/cpu_percent": 21, "system/gpu/0/temperature_celsius": 55},
                        },
                    ).raise_for_status()
                    client.post(
                        f"/api/workers/job/{job_id}/status",
                        headers=headers,
                        json={"progress": {"current": 42, "metrics": {"loss": 0.2}}},
                    ).raise_for_status()

                settings = {"prometheus_enabled": True, "prometheus_categories": ["workers"]}
                client.put("/api/metrics/config", json=settings).raise_for_status()
                self.assertEqual(client.get("/api/metrics/config").json()["prometheus_categories"], ["workers"])

                scrape = client.get("/api/metrics/prometheus").raise_for_status().text
                self.assertIn(
                    'simpletuner_worker_gpu_temperature_celsius{gpu_index="0",worker_id="metrics-worker-1"} 55.0', scrape
                )
                self.assertIn(
                    'simpletuner_worker_training_loss{job_id="metrics-job-1",worker_id="metrics-worker-1"} 0.2', scrape
                )
                self.assertNotIn("metrics-worker-2", scrape)
                self.assertNotIn("metrics-job-2", scrape)

                preview = client.post("/api/metrics/config/preview", json=["workers"]).raise_for_status().json()
                self.assertEqual(preview["content"], scrape)
                self.assertEqual(client.get("/api/cloud/metrics/prometheus").raise_for_status().text, scrape)
                client.app.dependency_overrides[get_optional_user] = lambda: None
                self.assertNotIn("simpletuner_worker_", client.get("/api/cloud/metrics/prometheus").raise_for_status().text)
                self.assertNotIn(
                    "simpletuner_worker_",
                    client.post("/api/metrics/config/preview", json=["workers"]).raise_for_status().json()["content"],
                )

                worker = run_async(workers.get_worker("metrics-worker-1"))
                stale_sample = {**worker.system_sample, "sampled_at": (now - timedelta(minutes=10)).isoformat()}
                run_async(workers.update_worker(worker.worker_id, {"system_sample": stale_sample}))
                run_async(jobs.mark_completed("metrics-job-1"))
                scrape = client.get("/api/metrics/prometheus").raise_for_status().text
                self.assertIn("simpletuner_worker_system_sample_timestamp_seconds", scrape)
                self.assertNotIn("simpletuner_worker_gpu_temperature_celsius", scrape)
                self.assertNotIn("simpletuner_worker_training_loss", scrape)


if __name__ == "__main__":
    unittest.main()
