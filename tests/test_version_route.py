"""Tests for the /api/version endpoint."""

from __future__ import annotations

import unittest

import simpletuner
from simpletuner.simpletuner_sdk.server import ServerMode

from tests.unittest_support import APITestCase


class VersionRouteTestCase(APITestCase, unittest.TestCase):
    """Validate that /api/version exposes expected metadata."""

    def test_version_endpoint_returns_metadata(self) -> None:
        with self.client_session(ServerMode.TRAINER) as client:
            response = client.get("/api/version")

        self.assertEqual(response.status_code, 200)
        payload = response.json()

        expected_version = getattr(simpletuner, "__version__", None)
        self.assertEqual(payload.get("version"), expected_version)

        expected_major = None
        if isinstance(expected_version, str) and expected_version:
            try:
                expected_major = int(expected_version.split(".")[0])
            except ValueError:
                expected_major = None

        if expected_major is not None:
            self.assertEqual(payload.get("major"), expected_major)
        else:
            self.assertIn("major", payload)

        self.assertIn("git_install", payload)
        self.assertIn("git_commit", payload)
        self.assertIn("git_dirty", payload)

        if payload.get("git_install"):
            git_commit = payload.get("git_commit")
            self.assertIsInstance(git_commit, str)
            self.assertTrue(git_commit)
