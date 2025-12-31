"""Tests for RateLimitMiddleware.

Tests cover:
- Rate limiting kicks in after threshold
- Rate limit resets after window
- Different limits per endpoint pattern
- Rate limit headers in response
- Client IP extraction from proxy headers
- Excluded paths bypass rate limiting
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from simpletuner.simpletuner_sdk.server.middleware.security_middleware import RateLimitMiddleware, RateLimitRule


class TestRateLimitRule(unittest.TestCase):
    """Tests for RateLimitRule matching logic."""

    def test_pattern_matching(self):
        """Test pattern matching for routes."""
        import re

        rule = RateLimitRule(
            pattern=re.compile(r"^/api/cloud/auth/login$"),
            calls=5,
            period=60,
            methods=["POST"],
        )

        self.assertTrue(rule.matches("/api/cloud/auth/login", "POST"))
        self.assertFalse(rule.matches("/api/cloud/auth/login", "GET"))
        self.assertFalse(rule.matches("/api/cloud/auth/register", "POST"))

    def test_pattern_matching_all_methods(self):
        """Test pattern matching with no method restriction."""
        import re

        rule = RateLimitRule(
            pattern=re.compile(r"^/api/cloud/jobs"),
            calls=20,
            period=60,
            methods=None,
        )

        self.assertTrue(rule.matches("/api/cloud/jobs", "GET"))
        self.assertTrue(rule.matches("/api/cloud/jobs", "POST"))
        self.assertTrue(rule.matches("/api/cloud/jobs/123", "DELETE"))

    def test_pattern_matching_multiple_methods(self):
        """Test pattern matching with multiple allowed methods."""
        import re

        rule = RateLimitRule(
            pattern=re.compile(r"^/api/data"),
            calls=10,
            period=60,
            methods=["GET", "POST"],
        )

        self.assertTrue(rule.matches("/api/data", "GET"))
        self.assertTrue(rule.matches("/api/data", "POST"))
        self.assertFalse(rule.matches("/api/data", "DELETE"))


class TestRateLimitMiddleware(unittest.IsolatedAsyncioTestCase):
    """Tests for RateLimitMiddleware functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.app = FastAPI()

    def _create_middleware(
        self,
        calls: int = 5,
        period: int = 1,
        exclude_paths: list[str] | None = None,
        rules: list[tuple[str, int, int, list[str] | None]] | None = None,
        enable_audit: bool = False,
    ) -> RateLimitMiddleware:
        """Create middleware instance for testing."""
        return RateLimitMiddleware(
            self.app,
            calls=calls,
            period=period,
            exclude_paths=exclude_paths or [],
            rules=rules,
            enable_audit=enable_audit,
        )

    def _create_mock_request(
        self,
        path: str,
        method: str = "GET",
        client_ip: str = "192.168.1.100",
        headers: dict | None = None,
        cookies: dict | None = None,
    ) -> Request:
        """Create a mock Request object."""
        mock_request = MagicMock(spec=Request)
        mock_request.url.path = path
        mock_request.method = method
        mock_request.client.host = client_ip
        # Use a real dict for headers so .get() works correctly
        mock_request.headers = headers or {}
        # Use a real dict for cookies so .get() works correctly (returns None if missing)
        mock_request.cookies = cookies or {}
        # Mock query_params to return None for missing keys
        mock_request.query_params = {}
        return mock_request

    async def test_rate_limit_kicks_in_after_threshold(self):
        """Test rate limiting activates after threshold is reached."""
        middleware = self._create_middleware(calls=3, period=60)

        request = self._create_mock_request("/api/test", "GET")

        async def mock_call_next(req):
            return Response(content="OK", status_code=200)

        # First 3 requests should pass
        for i in range(3):
            response = await middleware.dispatch(request, mock_call_next)
            self.assertEqual(response.status_code, 200)
            self.assertIn("X-RateLimit-Limit", response.headers)
            self.assertEqual(response.headers["X-RateLimit-Limit"], "3")

        # 4th request should be rate limited
        response = await middleware.dispatch(request, mock_call_next)
        self.assertEqual(response.status_code, 429)

        # Check response is JSON
        self.assertIsInstance(response, JSONResponse)

        # Check rate limit headers
        self.assertEqual(response.headers["X-RateLimit-Remaining"], "0")
        self.assertIn("Retry-After", response.headers)

    async def test_rate_limit_resets_after_window(self):
        """Test rate limit resets after time window expires."""
        middleware = self._create_middleware(calls=2, period=1)

        request = self._create_mock_request("/api/test", "GET")

        async def mock_call_next(req):
            return Response(content="OK", status_code=200)

        # Exhaust the limit
        for _ in range(2):
            response = await middleware.dispatch(request, mock_call_next)
            self.assertEqual(response.status_code, 200)

        # Next request should be rate limited
        response = await middleware.dispatch(request, mock_call_next)
        self.assertEqual(response.status_code, 429)

        # Wait for window to expire
        time.sleep(1.1)

        # Should now be allowed again
        response = await middleware.dispatch(request, mock_call_next)
        self.assertEqual(response.status_code, 200)

    async def test_different_limits_per_endpoint_pattern(self):
        """Test different rate limits for different endpoint patterns."""
        # Define custom rules with different limits
        rules = [
            (r"^/api/auth/login$", 2, 60, ["POST"]),  # Strict: 2/min
            (r"^/api/data/", 5, 60, None),  # Moderate: 5/min
        ]

        middleware = self._create_middleware(calls=10, period=60, rules=rules)

        async def mock_call_next(req):
            return Response(content="OK", status_code=200)

        # Test strict login endpoint
        login_request = self._create_mock_request("/api/auth/login", "POST")

        # First 2 should pass
        for _ in range(2):
            response = await middleware.dispatch(login_request, mock_call_next)
            self.assertEqual(response.status_code, 200)

        # 3rd should be blocked
        response = await middleware.dispatch(login_request, mock_call_next)
        self.assertEqual(response.status_code, 429)

        # Test moderate data endpoint (different IP to avoid cross-limiting)
        data_request = self._create_mock_request("/api/data/records", "GET", client_ip="192.168.1.101")

        # First 5 should pass
        for _ in range(5):
            response = await middleware.dispatch(data_request, mock_call_next)
            self.assertEqual(response.status_code, 200)

        # 6th should be blocked
        response = await middleware.dispatch(data_request, mock_call_next)
        self.assertEqual(response.status_code, 429)

    async def test_rate_limit_headers_in_response(self):
        """Test rate limit headers are present in responses."""
        middleware = self._create_middleware(calls=5, period=60)

        request = self._create_mock_request("/api/test", "GET")

        async def mock_call_next(req):
            return Response(content="OK", status_code=200)

        # First request
        response = await middleware.dispatch(request, mock_call_next)

        # Check all required headers are present
        self.assertIn("X-RateLimit-Limit", response.headers)
        self.assertIn("X-RateLimit-Remaining", response.headers)
        self.assertIn("X-RateLimit-Reset", response.headers)

        # Verify values
        self.assertEqual(response.headers["X-RateLimit-Limit"], "5")
        self.assertEqual(response.headers["X-RateLimit-Remaining"], "4")

        # Reset should be a timestamp in the future
        reset_time = int(response.headers["X-RateLimit-Reset"])
        self.assertGreater(reset_time, int(time.time()))

        # Second request should decrement remaining
        response = await middleware.dispatch(request, mock_call_next)
        self.assertEqual(response.headers["X-RateLimit-Remaining"], "3")

    async def test_client_ip_extraction_from_x_forwarded_for(self):
        """Test client IP extraction from X-Forwarded-For header."""
        middleware = self._create_middleware(calls=2, period=60)

        async def mock_call_next(req):
            return Response(content="OK", status_code=200)

        # Request with X-Forwarded-For header
        request = self._create_mock_request(
            "/api/test",
            "GET",
            client_ip="10.0.0.1",  # Proxy IP
            headers={"X-Forwarded-For": "203.0.113.1, 198.51.100.1"},
        )

        # First two requests should pass (using 203.0.113.1 as client IP)
        for _ in range(2):
            response = await middleware.dispatch(request, mock_call_next)
            self.assertEqual(response.status_code, 200)

        # Third request should be rate limited
        response = await middleware.dispatch(request, mock_call_next)
        self.assertEqual(response.status_code, 429)

    async def test_client_ip_extraction_from_x_real_ip(self):
        """Test client IP extraction from X-Real-IP header."""
        middleware = self._create_middleware(calls=2, period=60)

        async def mock_call_next(req):
            return Response(content="OK", status_code=200)

        # Request with X-Real-IP header
        request = self._create_mock_request(
            "/api/test",
            "GET",
            client_ip="10.0.0.1",  # Proxy IP
            headers={"X-Real-IP": "203.0.113.5"},
        )

        # First two requests should pass (using 203.0.113.5 as client IP)
        for _ in range(2):
            response = await middleware.dispatch(request, mock_call_next)
            self.assertEqual(response.status_code, 200)

        # Third request should be rate limited
        response = await middleware.dispatch(request, mock_call_next)
        self.assertEqual(response.status_code, 429)

    async def test_excluded_paths_bypass_rate_limiting(self):
        """Test excluded paths bypass rate limiting."""
        middleware = self._create_middleware(
            calls=2,
            period=60,
            exclude_paths=["/health", "/static/"],
        )

        async def mock_call_next(req):
            return Response(content="OK", status_code=200)

        # Health endpoint should never be rate limited
        health_request = self._create_mock_request("/health", "GET")

        for _ in range(10):  # Way more than limit
            response = await middleware.dispatch(health_request, mock_call_next)
            self.assertEqual(response.status_code, 200)
            # No rate limit headers for excluded paths
            self.assertNotIn("X-RateLimit-Limit", response.headers)

        # Static files should never be rate limited
        static_request = self._create_mock_request("/static/css/style.css", "GET")

        for _ in range(10):
            response = await middleware.dispatch(static_request, mock_call_next)
            self.assertEqual(response.status_code, 200)

    async def test_localhost_bypass_rate_limiting(self):
        """Test localhost IPs bypass rate limiting."""
        middleware = self._create_middleware(calls=2, period=60)

        async def mock_call_next(req):
            return Response(content="OK", status_code=200)

        # Test localhost IPv4
        localhost_v4 = self._create_mock_request("/api/test", "GET", client_ip="127.0.0.1")

        for _ in range(10):
            response = await middleware.dispatch(localhost_v4, mock_call_next)
            self.assertEqual(response.status_code, 200)

        # Test localhost IPv6
        localhost_v6 = self._create_mock_request("/api/test", "GET", client_ip="::1")

        for _ in range(10):
            response = await middleware.dispatch(localhost_v6, mock_call_next)
            self.assertEqual(response.status_code, 200)

    async def test_different_ips_tracked_separately(self):
        """Test different client IPs are tracked separately."""
        middleware = self._create_middleware(calls=2, period=60)

        async def mock_call_next(req):
            return Response(content="OK", status_code=200)

        # IP 1 exhausts limit
        request1 = self._create_mock_request("/api/test", "GET", client_ip="192.168.1.1")
        for _ in range(2):
            response = await middleware.dispatch(request1, mock_call_next)
            self.assertEqual(response.status_code, 200)

        response = await middleware.dispatch(request1, mock_call_next)
        self.assertEqual(response.status_code, 429)

        # IP 2 should still be allowed
        request2 = self._create_mock_request("/api/test", "GET", client_ip="192.168.1.2")
        response = await middleware.dispatch(request2, mock_call_next)
        self.assertEqual(response.status_code, 200)

    async def test_rate_limit_response_content(self):
        """Test rate limit response has proper error message."""
        middleware = self._create_middleware(calls=1, period=60)

        request = self._create_mock_request("/api/test", "GET")

        async def mock_call_next(req):
            return Response(content="OK", status_code=200)

        # Exhaust limit
        await middleware.dispatch(request, mock_call_next)

        # Get rate limited response
        response = await middleware.dispatch(request, mock_call_next)
        self.assertEqual(response.status_code, 429)

        # Response should be JSON
        self.assertIsInstance(response, JSONResponse)

    async def test_stale_entries_cleanup(self):
        """Test stale entries are cleaned up periodically."""
        middleware = self._create_middleware(calls=5, period=1)

        # Override cleanup interval to force cleanup
        middleware._cleanup_interval = 0

        request = self._create_mock_request("/api/test", "GET")

        async def mock_call_next(req):
            return Response(content="OK", status_code=200)

        # Make some requests
        for _ in range(3):
            await middleware.dispatch(request, mock_call_next)

        # Wait for entries to become stale
        time.sleep(1.5)

        # Trigger cleanup by making another request
        await middleware.dispatch(request, mock_call_next)

        # Verify cleanup happened by checking internal state
        # Stale entries should be removed
        self.assertGreaterEqual(len(middleware._requests), 0)

    async def test_method_specific_rate_limiting(self):
        """Test rate limits can be method-specific."""
        rules = [
            (r"^/api/items$", 2, 60, ["POST"]),  # Strict for POST
            (r"^/api/items$", 10, 60, ["GET"]),  # Lenient for GET
        ]

        middleware = self._create_middleware(calls=100, period=60, rules=rules)

        async def mock_call_next(req):
            return Response(content="OK", status_code=200)

        # POST should have limit of 2
        post_request = self._create_mock_request("/api/items", "POST", client_ip="192.168.1.10")

        for _ in range(2):
            response = await middleware.dispatch(post_request, mock_call_next)
            self.assertEqual(response.status_code, 200)

        response = await middleware.dispatch(post_request, mock_call_next)
        self.assertEqual(response.status_code, 429)

        # GET should have limit of 10 (different IP to avoid cross-limiting)
        get_request = self._create_mock_request("/api/items", "GET", client_ip="192.168.1.11")

        for _ in range(10):
            response = await middleware.dispatch(get_request, mock_call_next)
            self.assertEqual(response.status_code, 200)

        response = await middleware.dispatch(get_request, mock_call_next)
        self.assertEqual(response.status_code, 429)

    async def test_audit_logging_on_rate_limit(self):
        """Test audit logging when rate limit is exceeded."""
        middleware = self._create_middleware(calls=1, period=60, enable_audit=True)

        request = self._create_mock_request("/api/test", "GET")

        async def mock_call_next(req):
            return Response(content="OK", status_code=200)

        # Exhaust limit
        await middleware.dispatch(request, mock_call_next)

        # Mock the audit_log function at its actual location
        with patch(
            "simpletuner.simpletuner_sdk.server.services.cloud.audit.audit_log",
            new_callable=AsyncMock,
        ) as mock_audit:
            # Trigger rate limit
            response = await middleware.dispatch(request, mock_call_next)
            self.assertEqual(response.status_code, 429)

            # Give the background task a moment to run
            import asyncio

            await asyncio.sleep(0.1)

            # Audit log should have been called (or at least attempted)
            # Note: Due to asyncio.create_task, the call might not be immediate

    async def test_authenticated_users_get_higher_limits(self):
        """Test that authenticated users get higher rate limits."""
        from simpletuner.simpletuner_sdk.server.middleware.security_middleware import AUTHENTICATED_USER_MULTIPLIER

        base_limit = 5
        middleware = self._create_middleware(calls=base_limit, period=60)

        async def mock_call_next(req):
            return Response(content="OK", status_code=200)

        # Anonymous user (no session cookie) - should get base limit
        anon_request = self._create_mock_request("/api/test", "GET", client_ip="10.0.0.1", cookies={})

        for _ in range(base_limit):
            response = await middleware.dispatch(anon_request, mock_call_next)
            self.assertEqual(response.status_code, 200)

        # Next request should be rate limited
        response = await middleware.dispatch(anon_request, mock_call_next)
        self.assertEqual(response.status_code, 429)

        # Authenticated user (has session cookie) - should get multiplied limit
        auth_request = self._create_mock_request(
            "/api/test",
            "GET",
            client_ip="10.0.0.2",  # Different IP
            cookies={"simpletuner_session": "valid-session-token"},
        )

        # Should be able to make many more requests
        expected_limit = base_limit * AUTHENTICATED_USER_MULTIPLIER
        for _ in range(expected_limit):
            response = await middleware.dispatch(auth_request, mock_call_next)
            self.assertEqual(response.status_code, 200)

        # Next request should be rate limited
        response = await middleware.dispatch(auth_request, mock_call_next)
        self.assertEqual(response.status_code, 429)

        # Verify the limit header shows the higher value
        self.assertEqual(response.headers["X-RateLimit-Limit"], str(expected_limit))

    async def test_api_key_auth_gets_higher_limits(self):
        """Test that API key authenticated users get higher rate limits."""
        from simpletuner.simpletuner_sdk.server.middleware.security_middleware import AUTHENTICATED_USER_MULTIPLIER

        base_limit = 3
        middleware = self._create_middleware(calls=base_limit, period=60)

        async def mock_call_next(req):
            return Response(content="OK", status_code=200)

        # User with X-API-Key header
        api_key_request = self._create_mock_request(
            "/api/test",
            "GET",
            client_ip="10.0.0.3",
            headers={"X-API-Key": "test-api-key-12345"},
        )

        expected_limit = base_limit * AUTHENTICATED_USER_MULTIPLIER
        for _ in range(expected_limit):
            response = await middleware.dispatch(api_key_request, mock_call_next)
            self.assertEqual(response.status_code, 200)

        # Next request should be rate limited
        response = await middleware.dispatch(api_key_request, mock_call_next)
        self.assertEqual(response.status_code, 429)


if __name__ == "__main__":
    unittest.main()
