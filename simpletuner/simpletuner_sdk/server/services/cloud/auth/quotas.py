"""Quota management for cloud training.

Provides per-user and per-level quotas for cost, concurrency, and rate limiting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QuotaType(str, Enum):
    """Types of quotas that can be applied."""

    COST_MONTHLY = "cost_monthly"  # Max USD spend per month
    COST_DAILY = "cost_daily"  # Max USD spend per day
    CONCURRENT_JOBS = "concurrent_jobs"  # Max running jobs at once
    JOBS_PER_DAY = "jobs_per_day"  # Max job submissions per day
    JOBS_PER_HOUR = "jobs_per_hour"  # Max job submissions per hour
    LOCAL_GPUS = "local_gpus"  # Max GPUs for local training (org ceiling)


class QuotaAction(str, Enum):
    """Action to take when quota is exceeded."""

    BLOCK = "block"  # Block the action
    WARN = "warn"  # Allow but warn
    REQUIRE_APPROVAL = "require_approval"  # Require admin approval


class QuotaScope(str, Enum):
    """Scope at which a quota is defined."""

    GLOBAL = "global"  # System-wide default
    ORG = "org"  # Organization ceiling
    TEAM = "team"  # Team ceiling
    LEVEL = "level"  # Role-based default
    USER = "user"  # User-specific limit


@dataclass
class Quota:
    """A quota definition.

    Quotas can be scoped at different levels, with a ceiling model:
    - Organization quotas define absolute ceilings for all members
    - Team quotas define ceilings within the org (bounded by org)
    - User/Level quotas are bounded by team and org ceilings
    - Global quotas apply when no org/team membership exists

    The effective limit is: min(org, team, user/level, global)
    """

    id: int
    quota_type: QuotaType
    limit_value: float
    action: QuotaAction = QuotaAction.BLOCK

    # Scope identifiers (exactly one should be set, or none for global)
    user_id: Optional[int] = None  # User-specific quota
    level_id: Optional[int] = None  # Role-based quota
    team_id: Optional[int] = None  # Team ceiling
    org_id: Optional[int] = None  # Organization ceiling

    @property
    def scope(self) -> QuotaScope:
        """Determine the scope of this quota."""
        if self.user_id is not None:
            return QuotaScope.USER
        if self.level_id is not None:
            return QuotaScope.LEVEL
        if self.team_id is not None:
            return QuotaScope.TEAM
        if self.org_id is not None:
            return QuotaScope.ORG
        return QuotaScope.GLOBAL

    @property
    def is_global(self) -> bool:
        """Check if this is a global default quota."""
        return self.scope == QuotaScope.GLOBAL

    @property
    def is_ceiling(self) -> bool:
        """Check if this quota acts as a ceiling (org/team level)."""
        return self.scope in (QuotaScope.ORG, QuotaScope.TEAM)

    @property
    def priority(self) -> int:
        """Priority for quota display (higher = more specific).

        Note: For actual limit resolution, use ceiling logic instead.
        This is kept for backward compatibility in UI sorting.
        """
        if self.user_id is not None:
            return 4
        if self.level_id is not None:
            return 3
        if self.team_id is not None:
            return 2
        if self.org_id is not None:
            return 1
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "quota_type": self.quota_type.value,
            "limit_value": self.limit_value,
            "action": self.action.value,
            "scope": self.scope.value,
            "user_id": self.user_id,
            "level_id": self.level_id,
            "team_id": self.team_id,
            "org_id": self.org_id,
        }


@dataclass
class QuotaStatus:
    """Current status of a quota for a user.

    When ceilings are in effect, both the effective limit and the ceiling
    source are tracked for transparency.
    """

    quota_type: QuotaType
    limit_value: float
    current_value: float
    action: QuotaAction
    is_exceeded: bool
    is_warning: bool  # 80% or more used
    percent_used: float
    message: Optional[str] = None
    source: str = "global"  # "user", "level", "team", "org", or "global"

    # Ceiling tracking
    ceiling_source: Optional[str] = None  # Which level imposed the ceiling
    ceiling_value: Optional[float] = None  # The ceiling limit value
    is_ceiling_bound: bool = False  # True if limit is bounded by a ceiling

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        result = {
            "quota_type": self.quota_type.value,
            "limit": self.limit_value,
            "current": self.current_value,
            "action": self.action.value,
            "is_exceeded": self.is_exceeded,
            "is_warning": self.is_warning,
            "percent_used": round(self.percent_used, 1),
            "message": self.message,
            "source": self.source,
        }

        # Include ceiling info if limit is bounded
        if self.is_ceiling_bound:
            result["ceiling"] = {
                "source": self.ceiling_source,
                "value": self.ceiling_value,
                "is_bound": True,
            }

        return result


@dataclass
class EffectiveQuota:
    """Resolved quota with ceiling information.

    Represents the final computed limit after applying ceiling logic.
    """

    quota_type: QuotaType
    limit_value: float
    action: QuotaAction
    source: str  # "org", "team", "level", "user", or "global"

    # If ceiling was applied
    is_ceiling_bound: bool = False
    ceiling_source: Optional[str] = None
    ceiling_value: Optional[float] = None

    # Original request (before ceiling)
    original_source: Optional[str] = None
    original_value: Optional[float] = None


class QuotaChecker:
    """Checks and enforces quotas for users.

    Uses a ceiling model for quota resolution:
    - Org limits are absolute ceilings for all members
    - Team limits are ceilings within the org
    - User/Level limits are bounded by team and org ceilings
    - Effective limit = min(org, team, user/level, global)
    """

    WARNING_THRESHOLD = 80.0  # Percent at which to show warning

    def __init__(self, job_store, user_store):
        """Initialize with stores for data access.

        Args:
            job_store: JobStore instance for job data
            user_store: UserStore instance for quota data
        """
        self._job_store = job_store
        self._user_store = user_store

    async def get_effective_quotas(self, user_id: int) -> Dict[QuotaType, EffectiveQuota]:
        """Get the effective quotas for a user using ceiling model.

        Resolves quotas with ceiling logic:
        - Org and Team quotas act as ceilings (upper bounds)
        - User/Level quotas are bounded by these ceilings
        - Effective limit = min(org, team, user/level, global)

        Returns:
            Dict mapping quota type to EffectiveQuota with ceiling info
        """
        user = await self._user_store.get_user(user_id, include_permissions=True)
        if not user:
            return {}

        # Gather all quotas from different scopes
        all_quotas = await self._gather_all_quotas(user)

        # Resolve each quota type using ceiling logic
        effective: Dict[QuotaType, EffectiveQuota] = {}

        for quota_type in QuotaType:
            resolved = self._resolve_quota_with_ceiling(quota_type, all_quotas)
            if resolved:
                effective[quota_type] = resolved

        return effective

    async def _gather_all_quotas(self, user) -> Dict[str, List[Quota]]:
        """Gather quotas from all relevant scopes for a user.

        Returns:
            Dict with keys: "org", "team", "level", "user", "global"
        """
        result = {
            "org": [],
            "team": [],
            "level": [],
            "user": [],
            "global": [],
        }

        # Get user-specific quotas
        user_quotas = await self._user_store.get_user_quotas(user.id)
        result["user"] = list(user_quotas)

        # Get level quotas
        level_ids = [lvl.id for lvl in user.levels]
        if level_ids:
            level_quotas = await self._user_store.get_level_quotas(level_ids)
            result["level"] = list(level_quotas)

        # Get team quotas (if user has teams)
        if user.teams:
            team_ids = [t.id for t in user.teams]
            team_quotas = await self._user_store.get_team_quotas(team_ids)
            result["team"] = list(team_quotas)

        # Get org quotas (if user has org)
        if user.org_id:
            org_quotas = await self._user_store.get_org_quotas(user.org_id)
            result["org"] = list(org_quotas)

        # Get global quotas
        global_quotas = await self._user_store.get_global_quotas()
        result["global"] = list(global_quotas)

        return result

    def _resolve_quota_with_ceiling(
        self,
        quota_type: QuotaType,
        all_quotas: Dict[str, List[Quota]],
    ) -> Optional[EffectiveQuota]:
        """Resolve a single quota type using ceiling logic.

        Ceiling logic:
        1. Find the base limit (user > level > global)
        2. Apply ceilings from team and org (if lower than base)
        3. Track which ceiling was applied for transparency
        """
        # Get quotas of this type from each scope
        org_quota = self._find_quota_by_type(all_quotas["org"], quota_type)
        team_quota = self._find_quota_by_type(all_quotas["team"], quota_type)
        level_quota = self._find_quota_by_type(all_quotas["level"], quota_type)
        user_quota = self._find_quota_by_type(all_quotas["user"], quota_type)
        global_quota = self._find_quota_by_type(all_quotas["global"], quota_type)

        # Determine base limit (user > level > global)
        base_quota = user_quota or level_quota or global_quota
        if base_quota is None:
            # Check if we have a ceiling without a base
            if org_quota:
                return EffectiveQuota(
                    quota_type=quota_type,
                    limit_value=org_quota.limit_value,
                    action=org_quota.action,
                    source="org",
                )
            if team_quota:
                return EffectiveQuota(
                    quota_type=quota_type,
                    limit_value=team_quota.limit_value,
                    action=team_quota.action,
                    source="team",
                )
            return None  # No quota configured

        # Determine source of base limit
        if user_quota:
            base_source = "user"
        elif level_quota:
            base_source = "level"
        else:
            base_source = "global"

        base_value = base_quota.limit_value
        base_action = base_quota.action

        # Apply ceiling logic: effective = min(base, team_ceiling, org_ceiling)
        effective_value = base_value
        ceiling_source = None
        ceiling_value = None

        # Apply team ceiling if lower
        if team_quota and team_quota.limit_value < effective_value:
            effective_value = team_quota.limit_value
            ceiling_source = "team"
            ceiling_value = team_quota.limit_value

        # Apply org ceiling if lower (org always wins as ultimate ceiling)
        if org_quota and org_quota.limit_value < effective_value:
            effective_value = org_quota.limit_value
            ceiling_source = "org"
            ceiling_value = org_quota.limit_value

        # Determine if ceiling was applied
        is_ceiling_bound = ceiling_source is not None

        return EffectiveQuota(
            quota_type=quota_type,
            limit_value=effective_value,
            action=base_action,
            source=ceiling_source if is_ceiling_bound else base_source,
            is_ceiling_bound=is_ceiling_bound,
            ceiling_source=ceiling_source,
            ceiling_value=ceiling_value,
            original_source=base_source if is_ceiling_bound else None,
            original_value=base_value if is_ceiling_bound else None,
        )

    def _find_quota_by_type(
        self,
        quotas: List[Quota],
        quota_type: QuotaType,
    ) -> Optional[Quota]:
        """Find a quota of specific type from a list.

        For teams (user may have multiple), use the most permissive (highest limit).
        """
        matching = [q for q in quotas if q.quota_type == quota_type]
        if not matching:
            return None

        # For multiple quotas (e.g., from multiple teams), use highest limit
        return max(matching, key=lambda q: q.limit_value)

    async def check_quota(
        self,
        user_id: int,
        quota_type: QuotaType,
    ) -> QuotaStatus:
        """Check a specific quota for a user.

        Args:
            user_id: The user to check
            quota_type: The quota type to check

        Returns:
            QuotaStatus with current usage and limit info
        """
        quotas = await self.get_effective_quotas(user_id)
        quota = quotas.get(quota_type)

        if not quota:
            # No quota set, return unlimited
            return QuotaStatus(
                quota_type=quota_type,
                limit_value=float("inf"),
                current_value=0,
                action=QuotaAction.BLOCK,
                is_exceeded=False,
                is_warning=False,
                percent_used=0,
                message="No quota configured",
                source="none",
            )

        # Get current usage based on quota type
        current_value = await self._get_current_usage(user_id, quota_type)

        percent_used = (current_value / quota.limit_value * 100) if quota.limit_value > 0 else 0
        is_exceeded = current_value >= quota.limit_value
        is_warning = percent_used >= self.WARNING_THRESHOLD and not is_exceeded

        message = None
        if is_exceeded:
            msg_suffix = ""
            if quota.is_ceiling_bound:
                msg_suffix = f" (limited by {quota.ceiling_source} ceiling)"
            message = f"{quota_type.value} quota exceeded: {current_value:.2f} / {quota.limit_value:.2f}{msg_suffix}"
        elif is_warning:
            message = f"Approaching {quota_type.value} limit: {current_value:.2f} / {quota.limit_value:.2f}"

        return QuotaStatus(
            quota_type=quota_type,
            limit_value=quota.limit_value,
            current_value=current_value,
            action=quota.action,
            is_exceeded=is_exceeded,
            is_warning=is_warning,
            percent_used=percent_used,
            message=message,
            source=quota.source,
            # Ceiling tracking
            ceiling_source=quota.ceiling_source,
            ceiling_value=quota.ceiling_value,
            is_ceiling_bound=quota.is_ceiling_bound,
        )

    async def check_all_quotas(self, user_id: int) -> List[QuotaStatus]:
        """Check all quotas for a user.

        Returns list of QuotaStatus for each configured quota.
        Uses batched data loading to avoid N+1 queries.
        """
        quotas = await self.get_effective_quotas(user_id)
        if not quotas:
            return []

        # Batch load all usage data at once
        usage_data = await self._get_all_usage_data(user_id)
        statuses = []

        for quota_type, quota in quotas.items():
            current_value = self._extract_usage(quota_type, usage_data)
            status = self._build_quota_status_from_effective(quota, current_value)
            statuses.append(status)

        return statuses

    async def can_submit_job(self, user_id: int, estimated_cost: float = 0) -> Tuple[bool, List[QuotaStatus]]:
        """Check if a user can submit a new job.

        Args:
            user_id: The user attempting to submit
            estimated_cost: Estimated cost of the job (if known)

        Returns:
            Tuple of (can_submit, list of blocking/warning quota statuses)

        Uses batched data loading to avoid N+1 queries.
        """
        quotas = await self.get_effective_quotas(user_id)
        if not quotas:
            return True, []

        # Batch load all usage data at once (1 query instead of 5)
        usage_data = await self._get_all_usage_data(user_id)

        blocking_statuses = []
        warning_statuses = []

        # Check each relevant quota type
        quota_types_to_check = [
            QuotaType.CONCURRENT_JOBS,
            QuotaType.JOBS_PER_HOUR,
            QuotaType.JOBS_PER_DAY,
            QuotaType.COST_DAILY,
            QuotaType.COST_MONTHLY,
        ]

        for quota_type in quota_types_to_check:
            quota = quotas.get(quota_type)
            if not quota:
                continue

            current_value = self._extract_usage(quota_type, usage_data)
            status = self._build_quota_status_from_effective(quota, current_value)

            # For cost quotas, check if estimated cost would exceed
            if quota_type in (QuotaType.COST_DAILY, QuotaType.COST_MONTHLY) and estimated_cost > 0:
                would_exceed = (current_value + estimated_cost) >= quota.limit_value
                if would_exceed and not status.is_exceeded:
                    status.is_warning = True
                    status.message = f"Job would exceed {quota_type.value} limit"

            if status.is_exceeded:
                if status.action == QuotaAction.BLOCK:
                    blocking_statuses.append(status)
                else:
                    warning_statuses.append(status)
            elif status.is_warning:
                warning_statuses.append(status)

        can_submit = len(blocking_statuses) == 0
        return can_submit, blocking_statuses + warning_statuses

    async def _get_all_usage_data(self, user_id: int) -> Dict[str, Any]:
        """Batch load all usage data for quota checks.

        Returns a dict with pre-computed usage metrics.
        This replaces multiple individual queries with a single job list fetch.
        """
        from ..base import CloudJobStatus

        now = datetime.now(timezone.utc)
        hour_cutoff = (now - timedelta(hours=1)).isoformat()
        day_cutoff = (now - timedelta(days=1)).isoformat()
        month_cutoff = (now - timedelta(days=30)).isoformat()

        active_statuses = {
            CloudJobStatus.PENDING.value,
            CloudJobStatus.UPLOADING.value,
            CloudJobStatus.QUEUED.value,
            CloudJobStatus.RUNNING.value,
        }

        # Single query to fetch user's recent jobs
        jobs = await self._job_store.list_jobs(limit=1000, user_id=user_id)

        # Compute all metrics from the fetched jobs
        concurrent_count = 0
        hourly_count = 0
        daily_count = 0
        daily_cost = 0.0
        monthly_cost = 0.0

        for job in jobs:
            if job.status in active_statuses:
                concurrent_count += 1

            created = job.created_at
            if created >= hour_cutoff:
                hourly_count += 1
            if created >= day_cutoff:
                daily_count += 1
                daily_cost += job.cost_usd or 0
            if created >= month_cutoff:
                monthly_cost += job.cost_usd or 0

        return {
            "concurrent_jobs": concurrent_count,
            "jobs_per_hour": hourly_count,
            "jobs_per_day": daily_count,
            "cost_daily": daily_cost,
            "cost_monthly": monthly_cost,
        }

    def _extract_usage(self, quota_type: QuotaType, usage_data: Dict[str, Any]) -> float:
        """Extract usage value for a quota type from pre-computed data."""
        mapping = {
            QuotaType.CONCURRENT_JOBS: "concurrent_jobs",
            QuotaType.JOBS_PER_HOUR: "jobs_per_hour",
            QuotaType.JOBS_PER_DAY: "jobs_per_day",
            QuotaType.COST_DAILY: "cost_daily",
            QuotaType.COST_MONTHLY: "cost_monthly",
        }
        key = mapping.get(quota_type)
        return usage_data.get(key, 0) if key else 0

    def _build_quota_status_from_effective(
        self,
        quota: EffectiveQuota,
        current_value: float,
    ) -> QuotaStatus:
        """Build a QuotaStatus from an EffectiveQuota and current usage."""
        percent_used = (current_value / quota.limit_value * 100) if quota.limit_value > 0 else 0
        is_exceeded = current_value >= quota.limit_value
        is_warning = percent_used >= self.WARNING_THRESHOLD and not is_exceeded

        message = None
        if is_exceeded:
            msg_suffix = ""
            if quota.is_ceiling_bound:
                msg_suffix = f" (limited by {quota.ceiling_source} ceiling)"
            message = f"{quota.quota_type.value} quota exceeded: {current_value:.2f} / {quota.limit_value:.2f}{msg_suffix}"
        elif is_warning:
            message = f"Approaching {quota.quota_type.value} limit: {current_value:.2f} / {quota.limit_value:.2f}"

        return QuotaStatus(
            quota_type=quota.quota_type,
            limit_value=quota.limit_value,
            current_value=current_value,
            action=quota.action,
            is_exceeded=is_exceeded,
            is_warning=is_warning,
            percent_used=percent_used,
            message=message,
            source=quota.source,
            # Ceiling tracking
            ceiling_source=quota.ceiling_source,
            ceiling_value=quota.ceiling_value,
            is_ceiling_bound=quota.is_ceiling_bound,
        )

    async def _get_current_usage(self, user_id: int, quota_type: QuotaType) -> float:
        """Get current usage for a quota type.

        Note: For batch operations, prefer _get_all_usage_data() to avoid N+1 queries.
        This method is kept for single quota checks.
        """
        usage_data = await self._get_all_usage_data(user_id)
        return self._extract_usage(quota_type, usage_data)


# Default quotas for new installations
DEFAULT_QUOTAS = [
    # Global defaults (applied to all users without specific quotas)
    Quota(id=1, quota_type=QuotaType.CONCURRENT_JOBS, limit_value=5, action=QuotaAction.BLOCK),
    Quota(id=2, quota_type=QuotaType.JOBS_PER_DAY, limit_value=20, action=QuotaAction.WARN),
    Quota(id=3, quota_type=QuotaType.COST_MONTHLY, limit_value=500, action=QuotaAction.WARN),
]
