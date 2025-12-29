"""Tests for metrics configuration endpoints.

Tests cover:
- Metrics configuration get/update
- Category listing and validation
- Template listing and application
- Prometheus export preview
- Hint management (dismiss/show)
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from tests.unittest_support import APITestCase


class TestMetricsConfigRoutes(APITestCase, unittest.TestCase):
    """Tests for metrics configuration API routes."""

    def setUp(self):
        super().setUp()
        self._setup_api_environment()

    def tearDown(self):
        self._teardown_api_environment()
        super().tearDown()

    def _get_client(self):
        from simpletuner.simpletuner_sdk.server import ServerMode, create_app

        app = create_app(mode=ServerMode.UNIFIED)
        return TestClient(app)

    def test_get_metrics_config(self):
        """GET /api/cloud/metrics-config should return current config."""
        with self._get_client() as client:
            response = client.get("/api/cloud/metrics-config")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            # Verify response structure
            self.assertIn("prometheus_enabled", data)
            self.assertIn("prometheus_categories", data)
            self.assertIn("tensorboard_enabled", data)
            self.assertIn("endpoint_url", data)

            # Verify types
            self.assertIsInstance(data["prometheus_enabled"], bool)
            self.assertIsInstance(data["prometheus_categories"], list)
            self.assertIsInstance(data["tensorboard_enabled"], bool)
            self.assertEqual(data["endpoint_url"], "/api/metrics/prometheus")

    def test_update_metrics_config_enable(self):
        """PUT /api/cloud/metrics-config should enable prometheus."""
        with self._get_client() as client:
            response = client.put(
                "/api/cloud/metrics-config",
                json={"prometheus_enabled": True},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["prometheus_enabled"])

    def test_update_metrics_config_categories(self):
        """PUT /api/cloud/metrics-config should update categories."""
        with self._get_client() as client:
            # Update with valid categories
            response = client.put(
                "/api/cloud/metrics-config",
                json={"prometheus_categories": ["jobs", "http"]},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("jobs", data["prometheus_categories"])
            self.assertIn("http", data["prometheus_categories"])

    def test_update_metrics_config_filters_invalid_categories(self):
        """PUT should filter out invalid category names."""
        with self._get_client() as client:
            # Include both valid and invalid categories
            response = client.put(
                "/api/cloud/metrics-config",
                json={"prometheus_categories": ["jobs", "invalid_category", "http"]},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            # Valid categories should be kept
            self.assertIn("jobs", data["prometheus_categories"])
            self.assertIn("http", data["prometheus_categories"])
            # Invalid category should be filtered out
            self.assertNotIn("invalid_category", data["prometheus_categories"])


class TestMetricCategories(APITestCase, unittest.TestCase):
    """Tests for metric category listing."""

    def setUp(self):
        super().setUp()
        self._setup_api_environment()

    def tearDown(self):
        self._teardown_api_environment()
        super().tearDown()

    def _get_client(self):
        from simpletuner.simpletuner_sdk.server import ServerMode, create_app

        app = create_app(mode=ServerMode.UNIFIED)
        return TestClient(app)

    def test_list_categories(self):
        """GET /api/cloud/metrics-config/categories should list all categories."""
        with self._get_client() as client:
            response = client.get("/api/cloud/metrics-config/categories")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertIn("categories", data)
            categories = data["categories"]
            self.assertIsInstance(categories, list)
            self.assertGreater(len(categories), 0)

            # Verify structure of category info
            for cat in categories:
                self.assertIn("id", cat)
                self.assertIn("name", cat)
                self.assertIn("description", cat)
                self.assertIn("metrics", cat)
                self.assertIsInstance(cat["metrics"], list)

    def test_expected_categories_present(self):
        """Should include expected metric categories."""
        with self._get_client() as client:
            response = client.get("/api/cloud/metrics-config/categories")
            data = response.json()

            category_ids = [cat["id"] for cat in data["categories"]]

            # Verify expected categories are present
            expected = ["jobs", "http", "rate_limits", "approvals", "audit", "health"]
            for expected_cat in expected:
                self.assertIn(expected_cat, category_ids, f"Expected category '{expected_cat}' not found")


class TestMetricTemplates(APITestCase, unittest.TestCase):
    """Tests for metric preset templates."""

    def setUp(self):
        super().setUp()
        self._setup_api_environment()

    def tearDown(self):
        self._teardown_api_environment()
        super().tearDown()

    def _get_client(self):
        from simpletuner.simpletuner_sdk.server import ServerMode, create_app

        app = create_app(mode=ServerMode.UNIFIED)
        return TestClient(app)

    def test_list_templates(self):
        """GET /api/cloud/metrics-config/templates should list all templates."""
        with self._get_client() as client:
            response = client.get("/api/cloud/metrics-config/templates")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertIn("templates", data)
            templates = data["templates"]
            self.assertIsInstance(templates, list)
            self.assertGreater(len(templates), 0)

            # Verify structure of template info
            for tmpl in templates:
                self.assertIn("id", tmpl)
                self.assertIn("name", tmpl)
                self.assertIn("description", tmpl)
                self.assertIn("categories", tmpl)
                self.assertIsInstance(tmpl["categories"], list)

    def test_expected_templates_present(self):
        """Should include expected preset templates."""
        with self._get_client() as client:
            response = client.get("/api/cloud/metrics-config/templates")
            data = response.json()

            template_ids = [tmpl["id"] for tmpl in data["templates"]]

            # Verify expected templates are present
            expected = ["minimal", "standard", "security", "full"]
            for expected_tmpl in expected:
                self.assertIn(expected_tmpl, template_ids, f"Expected template '{expected_tmpl}' not found")

    def test_apply_template_minimal(self):
        """POST /api/cloud/metrics-config/apply-template/minimal should apply minimal template."""
        with self._get_client() as client:
            response = client.post("/api/cloud/metrics-config/apply-template/minimal")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertTrue(data["prometheus_enabled"])
            self.assertIn("jobs", data["prometheus_categories"])
            # Minimal should only include jobs
            self.assertEqual(len(data["prometheus_categories"]), 1)

    def test_apply_template_standard(self):
        """POST should apply standard template with jobs, http, health."""
        with self._get_client() as client:
            response = client.post("/api/cloud/metrics-config/apply-template/standard")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertTrue(data["prometheus_enabled"])
            self.assertIn("jobs", data["prometheus_categories"])
            self.assertIn("http", data["prometheus_categories"])
            self.assertIn("health", data["prometheus_categories"])

    def test_apply_template_full(self):
        """POST should apply full template with all categories."""
        with self._get_client() as client:
            # First get all available categories
            cat_response = client.get("/api/cloud/metrics-config/categories")
            all_categories = [c["id"] for c in cat_response.json()["categories"]]

            # Apply full template
            response = client.post("/api/cloud/metrics-config/apply-template/full")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            # Full template should include all categories
            for cat in all_categories:
                self.assertIn(cat, data["prometheus_categories"])

    def test_apply_template_not_found(self):
        """POST with invalid template ID should return 404."""
        with self._get_client() as client:
            response = client.post("/api/cloud/metrics-config/apply-template/nonexistent")
            self.assertEqual(response.status_code, 404)


class TestMetricsPreview(APITestCase, unittest.TestCase):
    """Tests for Prometheus export preview."""

    def setUp(self):
        super().setUp()
        self._setup_api_environment()

    def tearDown(self):
        self._teardown_api_environment()
        super().tearDown()

    def _get_client(self):
        from simpletuner.simpletuner_sdk.server import ServerMode, create_app

        app = create_app(mode=ServerMode.UNIFIED)
        return TestClient(app)

    def test_preview_export(self):
        """POST /api/cloud/metrics-config/preview should return export preview."""
        with self._get_client() as client:
            response = client.post(
                "/api/cloud/metrics-config/preview",
                params={"categories": ["jobs"]},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertIn("content", data)
            self.assertIn("metric_count", data)
            self.assertIn("categories_used", data)

            self.assertIsInstance(data["content"], str)
            self.assertIsInstance(data["metric_count"], int)
            self.assertIn("jobs", data["categories_used"])

    def test_preview_export_default_categories(self):
        """POST without categories should use configured categories."""
        with self._get_client() as client:
            # First set some categories
            client.put(
                "/api/cloud/metrics-config",
                json={"prometheus_categories": ["jobs", "http"]},
            )

            # Preview without specifying categories
            response = client.post("/api/cloud/metrics-config/preview")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            # Should use the configured categories
            self.assertIn("jobs", data["categories_used"])
            self.assertIn("http", data["categories_used"])


class TestMetricsHints(APITestCase, unittest.TestCase):
    """Tests for metrics hint management."""

    def setUp(self):
        super().setUp()
        self._setup_api_environment()

    def tearDown(self):
        self._teardown_api_environment()
        super().tearDown()

    def _get_client(self):
        from simpletuner.simpletuner_sdk.server import ServerMode, create_app

        app = create_app(mode=ServerMode.UNIFIED)
        return TestClient(app)

    def test_dismiss_hint(self):
        """POST /api/cloud/metrics-config/dismiss-hint should dismiss hint."""
        with self._get_client() as client:
            response = client.post("/api/cloud/metrics-config/dismiss-hint/hero")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertTrue(data["success"])
            self.assertEqual(data["hint"], "hero")

    def test_show_hint(self):
        """POST /api/cloud/metrics-config/show-hint should re-show hint."""
        with self._get_client() as client:
            # First dismiss
            client.post("/api/cloud/metrics-config/dismiss-hint/hero")

            # Then show again
            response = client.post("/api/cloud/metrics-config/show-hint/hero")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertTrue(data["success"])
            self.assertEqual(data["hint"], "hero")

    def test_dismiss_hint_idempotent(self):
        """Dismissing same hint twice should be idempotent."""
        with self._get_client() as client:
            # Dismiss twice
            response1 = client.post("/api/cloud/metrics-config/dismiss-hint/test-hint")
            response2 = client.post("/api/cloud/metrics-config/dismiss-hint/test-hint")

            self.assertEqual(response1.status_code, 200)
            self.assertEqual(response2.status_code, 200)
            self.assertTrue(response2.json()["success"])


class TestMetricCategoryDefinitions(unittest.TestCase):
    """Tests for metric category definitions."""

    def test_category_definitions_complete(self):
        """All categories should have required fields."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.metrics_config import METRIC_CATEGORIES

        for cat_id, cat_info in METRIC_CATEGORIES.items():
            self.assertIn("name", cat_info, f"Category {cat_id} missing 'name'")
            self.assertIn("description", cat_info, f"Category {cat_id} missing 'description'")
            self.assertIn("metrics", cat_info, f"Category {cat_id} missing 'metrics'")
            self.assertIsInstance(cat_info["metrics"], list)
            self.assertGreater(len(cat_info["metrics"]), 0, f"Category {cat_id} has no metrics defined")

    def test_template_definitions_complete(self):
        """All templates should have required fields and valid categories."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.metrics_config import METRIC_CATEGORIES, METRIC_TEMPLATES

        valid_categories = set(METRIC_CATEGORIES.keys())

        for tmpl_id, tmpl_info in METRIC_TEMPLATES.items():
            self.assertIn("name", tmpl_info, f"Template {tmpl_id} missing 'name'")
            self.assertIn("description", tmpl_info, f"Template {tmpl_id} missing 'description'")
            self.assertIn("categories", tmpl_info, f"Template {tmpl_id} missing 'categories'")

            # Verify all referenced categories exist
            for cat in tmpl_info["categories"]:
                self.assertIn(cat, valid_categories, f"Template {tmpl_id} references invalid category '{cat}'")

    def test_full_template_includes_all_categories(self):
        """Full template should include all defined categories."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.metrics_config import METRIC_CATEGORIES, METRIC_TEMPLATES

        full_template = METRIC_TEMPLATES["full"]
        all_categories = set(METRIC_CATEGORIES.keys())
        template_categories = set(full_template["categories"])

        self.assertEqual(all_categories, template_categories, "Full template should include all categories")


if __name__ == "__main__":
    unittest.main()
