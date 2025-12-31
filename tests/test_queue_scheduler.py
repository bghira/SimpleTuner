"""Tests for queue scheduler background task functionality.

Tests cover:
- SchedulingPolicy decisions
- Queue entry models
- Priority handling
- Concurrency limits (global, per-user, per-team)
- Fair-share scheduling
- Starvation prevention
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from simpletuner.simpletuner_sdk.server.services.cloud.queue.models import (
    LEVEL_PRIORITY_MAP,
    QueueEntry,
    QueuePriority,
    QueueStatus,
)
from simpletuner.simpletuner_sdk.server.services.cloud.queue.scheduler import (
    SchedulingConfig,
    SchedulingDecision,
    SchedulingPolicy,
)


class TestQueueStatus(unittest.TestCase):
    """Tests for QueueStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        self.assertEqual(QueueStatus.PENDING.value, "pending")
        self.assertEqual(QueueStatus.READY.value, "ready")
        self.assertEqual(QueueStatus.RUNNING.value, "running")
        self.assertEqual(QueueStatus.COMPLETED.value, "completed")
        self.assertEqual(QueueStatus.FAILED.value, "failed")
        self.assertEqual(QueueStatus.CANCELLED.value, "cancelled")
        self.assertEqual(QueueStatus.BLOCKED.value, "blocked")

    def test_status_from_string(self):
        """Test status can be created from string."""
        status = QueueStatus("pending")
        self.assertEqual(status, QueueStatus.PENDING)

    def test_status_string_conversion(self):
        """Test status converts to string."""
        self.assertEqual(str(QueueStatus.RUNNING), "QueueStatus.RUNNING")
        self.assertEqual(QueueStatus.RUNNING.value, "running")


class TestQueuePriority(unittest.TestCase):
    """Tests for QueuePriority enum."""

    def test_priority_values(self):
        """Test priority values are correctly ordered."""
        self.assertEqual(QueuePriority.LOW.value, 0)
        self.assertEqual(QueuePriority.NORMAL.value, 10)
        self.assertEqual(QueuePriority.HIGH.value, 20)
        self.assertEqual(QueuePriority.URGENT.value, 30)
        self.assertEqual(QueuePriority.CRITICAL.value, 40)

    def test_priority_ordering(self):
        """Test priorities are ordered correctly."""
        self.assertLess(QueuePriority.LOW.value, QueuePriority.NORMAL.value)
        self.assertLess(QueuePriority.NORMAL.value, QueuePriority.HIGH.value)
        self.assertLess(QueuePriority.HIGH.value, QueuePriority.URGENT.value)
        self.assertLess(QueuePriority.URGENT.value, QueuePriority.CRITICAL.value)

    def test_priority_from_int(self):
        """Test priority can be created from int."""
        priority = QueuePriority(10)
        self.assertEqual(priority, QueuePriority.NORMAL)

    def test_level_priority_mapping(self):
        """Test user level to priority mapping."""
        self.assertEqual(LEVEL_PRIORITY_MAP["admin"], QueuePriority.URGENT)
        self.assertEqual(LEVEL_PRIORITY_MAP["lead"], QueuePriority.HIGH)
        self.assertEqual(LEVEL_PRIORITY_MAP["researcher"], QueuePriority.NORMAL)
        self.assertEqual(LEVEL_PRIORITY_MAP["viewer"], QueuePriority.LOW)


class TestQueueEntry(unittest.TestCase):
    """Tests for QueueEntry model."""

    def _create_entry(self, **kwargs) -> QueueEntry:
        """Create a test queue entry."""
        defaults = {
            "id": 1,
            "job_id": "job-123",
            "user_id": 1,
            "priority": QueuePriority.NORMAL,
            "status": QueueStatus.PENDING,
        }
        defaults.update(kwargs)
        return QueueEntry(**defaults)

    def test_create_entry(self):
        """Test creating a queue entry."""
        entry = self._create_entry()

        self.assertEqual(entry.id, 1)
        self.assertEqual(entry.job_id, "job-123")
        self.assertEqual(entry.user_id, 1)
        self.assertEqual(entry.priority, QueuePriority.NORMAL)
        self.assertEqual(entry.status, QueueStatus.PENDING)

    def test_effective_priority_default(self):
        """Test effective priority uses priority when no override."""
        entry = self._create_entry(priority=QueuePriority.HIGH)

        self.assertEqual(entry.effective_priority, QueuePriority.HIGH.value)
        self.assertEqual(entry.effective_priority, 20)

    def test_effective_priority_with_override(self):
        """Test effective priority uses override when set."""
        entry = self._create_entry(
            priority=QueuePriority.NORMAL,
            priority_override=25,
        )

        self.assertEqual(entry.effective_priority, 25)

    def test_entry_to_dict(self):
        """Test entry serialization to dict."""
        entry = self._create_entry(
            priority=QueuePriority.HIGH,
            estimated_cost=5.50,
        )

        d = entry.to_dict()

        self.assertEqual(d["id"], 1)
        self.assertEqual(d["job_id"], "job-123")
        self.assertEqual(d["priority"], 20)
        self.assertEqual(d["priority_name"], "high")
        self.assertEqual(d["effective_priority"], 20)
        self.assertEqual(d["estimated_cost"], 5.50)

    def test_entry_from_dict(self):
        """Test entry deserialization from dict."""
        data = {
            "id": 5,
            "job_id": "job-456",
            "user_id": 2,
            "priority": 20,  # HIGH
            "status": "running",
            "estimated_cost": 10.0,
        }

        entry = QueueEntry.from_dict(data)

        self.assertEqual(entry.id, 5)
        self.assertEqual(entry.job_id, "job-456")
        self.assertEqual(entry.priority, QueuePriority.HIGH)
        self.assertEqual(entry.status, QueueStatus.RUNNING)

    def test_entry_from_dict_string_priority(self):
        """Test entry deserialization with string priority."""
        data = {
            "job_id": "job-789",
            "priority": "HIGH",
        }

        entry = QueueEntry.from_dict(data)

        self.assertEqual(entry.priority, QueuePriority.HIGH)

    def test_entry_metadata(self):
        """Test entry with metadata."""
        entry = self._create_entry(metadata={"config": "sdxl-train", "steps": 1000})

        self.assertEqual(entry.metadata["config"], "sdxl-train")
        self.assertEqual(entry.metadata["steps"], 1000)

    def test_entry_retry_handling(self):
        """Test entry retry tracking."""
        entry = self._create_entry(
            attempt=2,
            max_attempts=3,
            error_message="Connection timeout",
        )

        self.assertEqual(entry.attempt, 2)
        self.assertEqual(entry.max_attempts, 3)
        self.assertEqual(entry.error_message, "Connection timeout")


class TestSchedulingConfig(unittest.TestCase):
    """Tests for SchedulingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SchedulingConfig()

        self.assertEqual(config.max_concurrent, 5)
        self.assertEqual(config.user_max_concurrent, 2)
        self.assertEqual(config.team_max_concurrent, 10)
        self.assertFalse(config.enable_fair_share)
        self.assertEqual(config.starvation_threshold_minutes, 60)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SchedulingConfig(
            max_concurrent=10,
            user_max_concurrent=4,
            team_max_concurrent=20,
            enable_fair_share=True,
            starvation_threshold_minutes=30,
        )

        self.assertEqual(config.max_concurrent, 10)
        self.assertEqual(config.user_max_concurrent, 4)
        self.assertTrue(config.enable_fair_share)


class TestSchedulingDecision(unittest.TestCase):
    """Tests for SchedulingDecision."""

    def test_decision_with_entry(self):
        """Test decision with selected entry."""
        entry = QueueEntry(id=1, job_id="job-123", user_id=1)
        decision = SchedulingDecision(
            entry=entry,
            reason="Selected for execution",
        )

        self.assertEqual(decision.entry, entry)
        self.assertEqual(decision.reason, "Selected for execution")
        self.assertEqual(decision.blocked_users, [])

    def test_decision_no_entry(self):
        """Test decision with no entry selected."""
        decision = SchedulingDecision(
            entry=None,
            reason="Global limit reached",
            blocked_users=[1, 2],
        )

        self.assertIsNone(decision.entry)
        self.assertIn("limit", decision.reason)
        self.assertEqual(decision.blocked_users, [1, 2])


class TestSchedulingPolicyBasic(unittest.TestCase):
    """Basic tests for SchedulingPolicy."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SchedulingConfig(
            max_concurrent=5,
            user_max_concurrent=2,
        )
        self.policy = SchedulingPolicy(self.config)

    def _create_entry(
        self,
        id: int,
        job_id: str,
        user_id: int,
        priority: QueuePriority = QueuePriority.NORMAL,
    ) -> QueueEntry:
        """Create a test queue entry."""
        return QueueEntry(
            id=id,
            job_id=job_id,
            user_id=user_id,
            priority=priority,
            status=QueueStatus.PENDING,
        )

    def test_select_from_empty_queue(self):
        """Test selecting from empty queue returns None."""
        decision = self.policy.select_next(
            running_count=0,
            user_running={},
            pending_entries=[],
        )

        self.assertIsNone(decision.entry)
        self.assertIn("No pending", decision.reason)

    def test_select_first_pending(self):
        """Test selecting first pending job."""
        entries = [
            self._create_entry(1, "job-1", user_id=1),
            self._create_entry(2, "job-2", user_id=2),
        ]

        decision = self.policy.select_next(
            running_count=0,
            user_running={},
            pending_entries=entries,
        )

        self.assertEqual(decision.entry.job_id, "job-1")

    def test_global_limit_blocks_all(self):
        """Test global concurrency limit blocks scheduling."""
        entries = [self._create_entry(1, "job-1", user_id=1)]

        decision = self.policy.select_next(
            running_count=5,  # At limit
            user_running={},
            pending_entries=entries,
        )

        self.assertIsNone(decision.entry)
        self.assertIn("Global limit", decision.reason)


class TestSchedulingPolicyUserLimits(unittest.TestCase):
    """Tests for per-user concurrency limits."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SchedulingConfig(
            max_concurrent=10,
            user_max_concurrent=2,
        )
        self.policy = SchedulingPolicy(self.config)

    def _create_entry(
        self,
        id: int,
        job_id: str,
        user_id: int,
        priority: QueuePriority = QueuePriority.NORMAL,
    ) -> QueueEntry:
        """Create a test queue entry."""
        return QueueEntry(
            id=id,
            job_id=job_id,
            user_id=user_id,
            priority=priority,
            status=QueueStatus.PENDING,
        )

    def test_user_at_limit_skipped(self):
        """Test user at concurrency limit is skipped."""
        entries = [
            self._create_entry(1, "job-1", user_id=1),  # User 1 at limit
            self._create_entry(2, "job-2", user_id=2),  # User 2 has room
        ]

        decision = self.policy.select_next(
            running_count=2,
            user_running={1: 2},  # User 1 at limit (2 running)
            pending_entries=entries,
        )

        # Should skip user 1, select user 2's job
        self.assertEqual(decision.entry.job_id, "job-2")
        self.assertIn(1, decision.blocked_users)

    def test_all_users_at_limit(self):
        """Test when all users are at their limits."""
        entries = [
            self._create_entry(1, "job-1", user_id=1),
            self._create_entry(2, "job-2", user_id=2),
        ]

        decision = self.policy.select_next(
            running_count=4,
            user_running={1: 2, 2: 2},  # Both at limit
            pending_entries=entries,
        )

        self.assertIsNone(decision.entry)
        self.assertIn("per-user", decision.reason.lower())
        self.assertEqual(len(decision.blocked_users), 2)

    def test_user_under_limit_accepted(self):
        """Test user under limit can run jobs."""
        entries = [
            self._create_entry(1, "job-1", user_id=1),
        ]

        decision = self.policy.select_next(
            running_count=1,
            user_running={1: 1},  # User 1 has 1 running (limit is 2)
            pending_entries=entries,
        )

        self.assertEqual(decision.entry.job_id, "job-1")


class TestSchedulingPolicyTeamLimits(unittest.TestCase):
    """Tests for per-team concurrency limits (fair-share)."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SchedulingConfig(
            max_concurrent=20,
            user_max_concurrent=5,
            team_max_concurrent=3,
            enable_fair_share=True,
        )
        self.policy = SchedulingPolicy(self.config)

    def _create_entry(
        self,
        id: int,
        job_id: str,
        user_id: int,
        team_id: Optional[str] = None,
        priority: QueuePriority = QueuePriority.NORMAL,
    ) -> QueueEntry:
        """Create a test queue entry."""
        return QueueEntry(
            id=id,
            job_id=job_id,
            user_id=user_id,
            team_id=team_id,
            priority=priority,
            status=QueueStatus.PENDING,
        )

    def test_team_at_limit_skipped(self):
        """Test team at concurrency limit is skipped."""
        entries = [
            self._create_entry(1, "job-1", user_id=1, team_id="team-a"),
            self._create_entry(2, "job-2", user_id=2, team_id="team-b"),
        ]

        decision = self.policy.select_next(
            running_count=5,
            user_running={},
            pending_entries=entries,
            team_running={"team-a": 3},  # Team A at limit
        )

        # Should skip team-a, select team-b's job
        self.assertEqual(decision.entry.job_id, "job-2")

    def test_fair_share_disabled(self):
        """Test team limits not enforced when fair-share disabled."""
        config = SchedulingConfig(
            max_concurrent=20,
            user_max_concurrent=5,
            team_max_concurrent=3,
            enable_fair_share=False,  # Disabled
        )
        policy = SchedulingPolicy(config)

        entries = [
            self._create_entry(1, "job-1", user_id=1, team_id="team-a"),
        ]

        decision = policy.select_next(
            running_count=5,
            user_running={},
            pending_entries=entries,
            team_running={"team-a": 10},  # Would be over limit if enabled
        )

        # With fair-share disabled, team limit doesn't apply
        self.assertEqual(decision.entry.job_id, "job-1")


class TestSchedulingPolicyPriority(unittest.TestCase):
    """Tests for priority-based scheduling."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SchedulingConfig(
            max_concurrent=10,
            user_max_concurrent=5,
        )
        self.policy = SchedulingPolicy(self.config)

    def _create_entry(
        self,
        id: int,
        job_id: str,
        user_id: int,
        priority: QueuePriority = QueuePriority.NORMAL,
        priority_override: Optional[int] = None,
    ) -> QueueEntry:
        """Create a test queue entry."""
        return QueueEntry(
            id=id,
            job_id=job_id,
            user_id=user_id,
            priority=priority,
            priority_override=priority_override,
            status=QueueStatus.PENDING,
        )

    def test_higher_priority_first(self):
        """Test higher priority jobs are selected first.

        Note: The policy expects pending_entries to be pre-sorted by priority.
        """
        # Pre-sorted by priority (highest first)
        entries = [
            self._create_entry(2, "job-2", user_id=2, priority=QueuePriority.HIGH),
            self._create_entry(1, "job-1", user_id=1, priority=QueuePriority.NORMAL),
        ]

        decision = self.policy.select_next(
            running_count=0,
            user_running={},
            pending_entries=entries,
        )

        # High priority job should be first
        self.assertEqual(decision.entry.job_id, "job-2")

    def test_priority_override_respected(self):
        """Test priority override is respected."""
        # Normal priority but with high override
        entry = self._create_entry(
            1,
            "job-1",
            user_id=1,
            priority=QueuePriority.NORMAL,
            priority_override=25,
        )

        # The effective priority should use the override
        self.assertEqual(entry.effective_priority, 25)


class TestSchedulingPolicyCanAccept(unittest.TestCase):
    """Tests for can_accept_job method."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SchedulingConfig(
            max_concurrent=5,
            user_max_concurrent=2,
        )
        self.policy = SchedulingPolicy(self.config)

    def test_can_accept_under_limits(self):
        """Test job can be accepted when under all limits."""
        can_accept, reason = self.policy.can_accept_job(
            user_id=1,
            running_count=3,
            user_running={1: 1},
        )

        self.assertTrue(can_accept)
        self.assertIn("can be scheduled", reason.lower())

    def test_cannot_accept_global_limit(self):
        """Test job rejected when global limit reached."""
        can_accept, reason = self.policy.can_accept_job(
            user_id=1,
            running_count=5,
            user_running={},
        )

        self.assertFalse(can_accept)
        self.assertIn("Global limit", reason)

    def test_cannot_accept_user_limit(self):
        """Test job rejected when user limit reached."""
        can_accept, reason = self.policy.can_accept_job(
            user_id=1,
            running_count=3,
            user_running={1: 2},  # User 1 at limit
        )

        self.assertFalse(can_accept)
        self.assertIn("User limit", reason)

    def test_can_accept_no_user(self):
        """Test job can be accepted with no user (system job)."""
        can_accept, reason = self.policy.can_accept_job(
            user_id=None,  # System job
            running_count=3,
            user_running={},
        )

        self.assertTrue(can_accept)


class TestSchedulingPolicyEdgeCases(unittest.TestCase):
    """Edge case tests for scheduling policy."""

    def test_empty_user_running_dict(self):
        """Test handling of empty user_running dict."""
        config = SchedulingConfig(max_concurrent=5, user_max_concurrent=2)
        policy = SchedulingPolicy(config)

        entry = QueueEntry(id=1, job_id="job-1", user_id=1)

        decision = policy.select_next(
            running_count=0,
            user_running={},  # Empty dict
            pending_entries=[entry],
        )

        self.assertEqual(decision.entry.job_id, "job-1")

    def test_null_user_id(self):
        """Test handling of null user_id (system jobs)."""
        config = SchedulingConfig(max_concurrent=5, user_max_concurrent=2)
        policy = SchedulingPolicy(config)

        entry = QueueEntry(id=1, job_id="job-1", user_id=None)  # System job

        decision = policy.select_next(
            running_count=0,
            user_running={},
            pending_entries=[entry],
        )

        # System jobs should be allowed
        self.assertEqual(decision.entry.job_id, "job-1")

    def test_multiple_entries_same_user_blocked(self):
        """Test multiple entries from blocked user are all skipped."""
        config = SchedulingConfig(max_concurrent=10, user_max_concurrent=2)
        policy = SchedulingPolicy(config)

        entries = [
            QueueEntry(id=1, job_id="job-1", user_id=1),
            QueueEntry(id=2, job_id="job-2", user_id=1),
            QueueEntry(id=3, job_id="job-3", user_id=1),
            QueueEntry(id=4, job_id="job-4", user_id=2),  # Different user
        ]

        decision = policy.select_next(
            running_count=2,
            user_running={1: 2},  # User 1 at limit
            pending_entries=entries,
        )

        # Should skip all of user 1's jobs, select user 2's
        self.assertEqual(decision.entry.job_id, "job-4")


class TestQueueEntryApproval(unittest.TestCase):
    """Tests for queue entry approval handling."""

    def test_entry_requires_approval(self):
        """Test entry with approval required."""
        entry = QueueEntry(
            id=1,
            job_id="job-expensive",
            user_id=1,
            requires_approval=True,
            approval_id=None,  # Not yet approved
            estimated_cost=100.0,
        )

        self.assertTrue(entry.requires_approval)
        self.assertIsNone(entry.approval_id)

    def test_entry_approved(self):
        """Test entry that has been approved."""
        entry = QueueEntry(
            id=1,
            job_id="job-approved",
            user_id=1,
            requires_approval=True,
            approval_id=42,  # Linked to approval
        )

        self.assertTrue(entry.requires_approval)
        self.assertEqual(entry.approval_id, 42)


class TestQueueEntryTimestamps(unittest.TestCase):
    """Tests for queue entry timestamp handling."""

    def test_entry_timestamps(self):
        """Test entry timestamp fields."""
        now = datetime.now(timezone.utc).isoformat()

        entry = QueueEntry(
            id=1,
            job_id="job-1",
            user_id=1,
            queued_at=now,
            started_at=None,
            completed_at=None,
        )

        self.assertEqual(entry.queued_at, now)
        self.assertIsNone(entry.started_at)
        self.assertIsNone(entry.completed_at)

    def test_entry_lifecycle_timestamps(self):
        """Test entry with complete lifecycle timestamps."""
        queued = "2024-01-15T10:00:00Z"
        started = "2024-01-15T10:05:00Z"
        completed = "2024-01-15T11:30:00Z"

        entry = QueueEntry(
            id=1,
            job_id="job-1",
            user_id=1,
            queued_at=queued,
            started_at=started,
            completed_at=completed,
            status=QueueStatus.COMPLETED,
        )

        d = entry.to_dict()
        self.assertEqual(d["queued_at"], queued)
        self.assertEqual(d["started_at"], started)
        self.assertEqual(d["completed_at"], completed)


if __name__ == "__main__":
    unittest.main()
