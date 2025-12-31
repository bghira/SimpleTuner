"""Rules engine for evaluating approval requirements."""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .approval_store import ApprovalStore
from .models import ApprovalRequest, ApprovalRule, ApprovalStatus, RuleCondition

logger = logging.getLogger(__name__)


@dataclass
class EvaluationContext:
    """Context for rule evaluation.

    Contains all information needed to evaluate approval rules.
    """

    user_id: int
    user_levels: List[str]  # Names of user's levels
    provider: str = "replicate"
    config_name: Optional[str] = None
    estimated_cost: float = 0.0
    hardware_type: Optional[str] = None

    # Historical data (populated by engine)
    user_job_count_today: int = 0
    user_total_job_count: int = 0


@dataclass
class EvaluationResult:
    """Result of rule evaluation."""

    requires_approval: bool
    matching_rule: Optional[ApprovalRule] = None
    reason: str = ""
    all_evaluated_rules: List[Tuple[ApprovalRule, bool]] = None  # (rule, matched)

    def __post_init__(self):
        if self.all_evaluated_rules is None:
            self.all_evaluated_rules = []


class ApprovalRulesEngine:
    """Evaluates approval rules for job submissions.

    Processes rules in priority order and determines if approval is needed.
    """

    def __init__(
        self,
        approval_store: Optional[ApprovalStore] = None,
        job_store=None,
    ):
        """Initialize the rules engine.

        Args:
            approval_store: ApprovalStore instance (uses singleton if None)
            job_store: JobStore instance for historical data
        """
        self._approval_store = approval_store or ApprovalStore()
        self._job_store = job_store

    async def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        """Evaluate all rules against the given context.

        Args:
            context: The evaluation context with job/user details

        Returns:
            EvaluationResult indicating if approval is needed and why
        """
        # Enrich context with historical data if job_store available
        if self._job_store:
            await self._enrich_context(context)

        # Get active rules, ordered by priority
        rules = await self._approval_store.list_rules(active_only=True)

        evaluated = []
        for rule in rules:
            matched = self._evaluate_rule(rule, context)
            evaluated.append((rule, matched))

            if matched:
                reason = self._format_reason(rule, context)
                logger.info(
                    "Approval required for user %d: rule '%s' matched (%s)",
                    context.user_id,
                    rule.name,
                    reason,
                )
                return EvaluationResult(
                    requires_approval=True,
                    matching_rule=rule,
                    reason=reason,
                    all_evaluated_rules=evaluated,
                )

        # No rules matched
        return EvaluationResult(
            requires_approval=False,
            reason="No approval rules matched",
            all_evaluated_rules=evaluated,
        )

    def _evaluate_rule(self, rule: ApprovalRule, context: EvaluationContext) -> bool:
        """Evaluate a single rule against the context.

        Returns True if the rule matches and approval is required.
        """
        # Check provider restriction
        if rule.applies_to_provider and rule.applies_to_provider != context.provider:
            return False

        # Check level restriction
        if rule.applies_to_level and rule.applies_to_level not in context.user_levels:
            return False

        # Check exemptions
        for exempt_level in rule.exempt_levels:
            if exempt_level in context.user_levels:
                return False  # User is exempt

        # Evaluate the condition
        return self._check_condition(rule.condition, rule.threshold, context)

    def _check_condition(
        self,
        condition: RuleCondition,
        threshold: str,
        context: EvaluationContext,
    ) -> bool:
        """Check if a specific condition is met."""
        if condition == RuleCondition.COST_EXCEEDS:
            try:
                threshold_value = float(threshold)
                return context.estimated_cost > threshold_value
            except ValueError:
                logger.warning("Invalid cost threshold: %s", threshold)
                return False

        elif condition == RuleCondition.HARDWARE_TYPE:
            # Threshold is the hardware type to match (or pattern)
            if not context.hardware_type:
                return False
            return fnmatch.fnmatch(context.hardware_type, threshold)

        elif condition == RuleCondition.PROVIDER:
            return context.provider == threshold

        elif condition == RuleCondition.USER_LEVEL:
            # Threshold is the minimum level; trigger if user is below
            # Lower priority = lower level
            # This requires level priorities, which we simplify here
            return threshold.lower() not in [lvl.lower() for lvl in context.user_levels]

        elif condition == RuleCondition.DAILY_JOBS_EXCEED:
            try:
                threshold_value = int(threshold)
                return context.user_job_count_today > threshold_value
            except ValueError:
                return False

        elif condition == RuleCondition.FIRST_JOB:
            return context.user_total_job_count == 0

        elif condition == RuleCondition.CONFIG_PATTERN:
            if not context.config_name:
                return False
            # Threshold is a pattern (glob or regex)
            if threshold.startswith("^") or threshold.endswith("$"):
                # Treat as regex
                try:
                    return bool(re.match(threshold, context.config_name))
                except re.error:
                    return False
            else:
                # Treat as glob
                return fnmatch.fnmatch(context.config_name, threshold)

        return False

    def _format_reason(self, rule: ApprovalRule, context: EvaluationContext) -> str:
        """Format a human-readable reason for requiring approval."""
        condition = rule.condition

        if condition == RuleCondition.COST_EXCEEDS:
            return f"Estimated cost (${context.estimated_cost:.2f}) exceeds threshold (${rule.threshold})"

        elif condition == RuleCondition.HARDWARE_TYPE:
            return f"Hardware type '{context.hardware_type}' requires approval"

        elif condition == RuleCondition.PROVIDER:
            return f"Provider '{context.provider}' requires approval"

        elif condition == RuleCondition.USER_LEVEL:
            return f"User level requires approval for this operation"

        elif condition == RuleCondition.DAILY_JOBS_EXCEED:
            return f"Daily job limit exceeded ({context.user_job_count_today} jobs today)"

        elif condition == RuleCondition.FIRST_JOB:
            return "First job from this user requires approval"

        elif condition == RuleCondition.CONFIG_PATTERN:
            return f"Config '{context.config_name}' matches approval-required pattern"

        return f"Rule '{rule.name}' triggered"

    async def _enrich_context(self, context: EvaluationContext) -> None:
        """Add historical data to the context."""
        from datetime import datetime, timedelta, timezone

        try:
            # Count today's jobs
            now = datetime.now(timezone.utc)
            today_start = now - timedelta(days=1)

            jobs = await self._job_store.list_jobs(limit=1000, user_id=context.user_id)

            context.user_total_job_count = len(jobs)
            context.user_job_count_today = sum(1 for j in jobs if j.created_at and j.created_at >= today_start.isoformat())
        except Exception as exc:
            logger.warning("Failed to enrich context: %s", exc)

    async def create_approval_request(
        self,
        job_id: str,
        context: EvaluationContext,
        rule: ApprovalRule,
        reason: str,
    ) -> ApprovalRequest:
        """Create an approval request for a job.

        Args:
            job_id: The job requiring approval
            context: The evaluation context
            rule: The rule that triggered approval
            reason: Human-readable reason

        Returns:
            The created approval request
        """
        request = await self._approval_store.create_request(
            job_id=job_id,
            user_id=context.user_id,
            rule_id=rule.id,
            reason=reason,
            provider=context.provider,
            config_name=context.config_name,
            estimated_cost=context.estimated_cost,
            hardware_type=context.hardware_type,
        )

        # Send notification for approval required
        await self._notify_approval_required(request, context, rule)

        return request

    async def _notify_approval_required(
        self,
        request: ApprovalRequest,
        context: EvaluationContext,
        rule: ApprovalRule,
    ) -> None:
        """Send notification that approval is required."""
        try:
            from ..notification import get_notifier

            notifier = get_notifier()

            # Get approvers (users with required level or higher)
            approvers = await self._get_approvers_for_level(rule.required_approver_level)

            await notifier.notify_approval_required(
                approval_request_id=request.id,
                job_id=request.job_id,
                approvers=approvers,
                reason=request.reason,
                estimated_cost=context.estimated_cost,
                config_name=context.config_name,
            )
        except Exception as exc:
            logger.debug("Failed to send approval notification: %s", exc)

    async def _get_level_hierarchy(self) -> Dict[str, int]:
        """Get level name to priority mapping from the database.

        Returns a dict mapping lowercase level names to their priority values.
        Falls back to minimal defaults if database is unavailable.
        """
        try:
            from ..auth import UserStore

            user_store = UserStore()
            levels = await user_store.list_levels()
            return {level.name.lower(): level.priority for level in levels}
        except Exception as exc:
            logger.debug("Failed to get level hierarchy from database: %s", exc)
            # Minimal fallback - just admin at highest priority
            return {"admin": 100}

    async def _get_approvers_for_level(self, required_level: str) -> List[str]:
        """Get list of approvers who can approve at the given level."""
        try:
            from ..auth import UserStore

            user_store = UserStore()
            users = await user_store.list_users()

            level_hierarchy = await self._get_level_hierarchy()
            required_priority = level_hierarchy.get(required_level.lower(), 50)

            approvers = []
            for user in users:
                if not user.is_active:
                    continue
                for level in user.level_names or []:
                    level_priority = level_hierarchy.get(level.lower(), 0)
                    if level_priority >= required_priority and user.email:
                        approvers.append(user.email)
                        break

            return approvers
        except Exception as exc:
            logger.debug("Failed to get approvers: %s", exc)
            return []

    async def can_approve(self, request_id: int, approver_user_levels: List[str]) -> bool:
        """Check if a user with given levels can approve a request.

        Args:
            request_id: The approval request ID
            approver_user_levels: The potential approver's level names

        Returns:
            True if the user can approve this request
        """
        request = await self._approval_store.get_request(request_id)
        if not request:
            return False

        rule = await self._approval_store.get_rule(request.rule_id)
        if not rule:
            return True  # Rule deleted, allow any approver

        required_level = rule.required_approver_level.lower()

        # Check if approver has required level or higher
        level_hierarchy = await self._get_level_hierarchy()

        required_priority = level_hierarchy.get(required_level, 0)

        for level in approver_user_levels:
            level_priority = level_hierarchy.get(level.lower(), 0)
            if level_priority >= required_priority:
                return True

        return False

    async def approve(
        self,
        request_id: int,
        approver_id: int,
        approver_levels: List[str],
        notes: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Approve a pending request.

        Args:
            request_id: The request to approve
            approver_id: User ID of the approver
            approver_levels: The approver's level names
            notes: Optional approval notes

        Returns:
            Tuple of (success, error_message)
        """
        # Check if user can approve
        can_approve = await self.can_approve(request_id, approver_levels)
        if not can_approve:
            return False, "Insufficient privileges to approve this request"

        # Get request before approval for notification
        request = await self._approval_store.get_request(request_id)

        success = await self._approval_store.approve_request(
            request_id=request_id,
            reviewed_by=approver_id,
            notes=notes,
        )

        if not success:
            return False, "Request not found or already processed"

        # Send approval notification
        if request:
            await self._notify_approval_decision(request, approver_id, "approved", notes)

        return True, None

    async def reject(
        self,
        request_id: int,
        approver_id: int,
        approver_levels: List[str],
        reason: str,
    ) -> Tuple[bool, Optional[str]]:
        """Reject a pending request.

        Args:
            request_id: The request to reject
            approver_id: User ID of the rejector
            approver_levels: The approver's level names
            reason: Rejection reason

        Returns:
            Tuple of (success, error_message)
        """
        # Check if user can approve/reject
        can_approve = await self.can_approve(request_id, approver_levels)
        if not can_approve:
            return False, "Insufficient privileges to reject this request"

        # Get request before rejection for notification
        request = await self._approval_store.get_request(request_id)

        success = await self._approval_store.reject_request(
            request_id=request_id,
            reviewed_by=approver_id,
            reason=reason,
        )

        if not success:
            return False, "Request not found or already processed"

        # Send rejection notification
        if request:
            await self._notify_approval_decision(request, approver_id, "rejected", reason)

        return True, None

    async def _notify_approval_decision(
        self,
        request: ApprovalRequest,
        approver_id: int,
        decision: str,
        notes: Optional[str] = None,
    ) -> None:
        """Send notification about approval decision."""
        try:
            from ..notification import NotificationEventType, get_notifier

            notifier = get_notifier()

            event_type = (
                NotificationEventType.APPROVAL_GRANTED if decision == "approved" else NotificationEventType.APPROVAL_REJECTED
            )

            await notifier.notify(
                event_type,
                {
                    "request_id": request.id,
                    "job_id": request.job_id,
                    "user_id": request.user_id,
                    "approver_id": approver_id,
                    "decision": decision,
                    "notes": notes,
                    "message": f"Job approval {decision}: {notes or 'No additional notes'}",
                    "severity": "info" if decision == "approved" else "warning",
                },
            )
        except Exception as exc:
            logger.debug("Failed to send approval decision notification: %s", exc)
