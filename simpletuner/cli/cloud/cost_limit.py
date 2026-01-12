"""
Cloud cost limit commands.

Handles cost-limit show, set, and disable commands.
"""

from .api import cloud_api_request


def cmd_cloud_cost_limit(args) -> int:
    """Manage cost limits for cloud providers."""
    limit_action = getattr(args, "limit_action", None)
    provider = getattr(args, "provider", "replicate")

    if limit_action == "show":
        result = cloud_api_request("GET", f"/api/cloud/providers/{provider}/config")
        config = result.get("config", {})

        print(f"Cost Limit Settings for {provider}:")
        print("=" * 50)
        print(f"  Enabled: {config.get('cost_limit_enabled', False)}")
        print(f"  Amount:  ${config.get('cost_limit_amount', 0):.2f}")
        print(f"  Period:  {config.get('cost_limit_period', 'monthly')}")
        print(f"  Action:  {config.get('cost_limit_action', 'warn')}")
        return 0

    elif limit_action == "set":
        amount = getattr(args, "amount", None)
        period = getattr(args, "period", "monthly")
        action = getattr(args, "action", "warn")

        if amount is None:
            print("Error: Amount is required.")
            return 1

        update_data = {
            "cost_limit_enabled": True,
            "cost_limit_amount": float(amount),
            "cost_limit_period": period,
            "cost_limit_action": action,
        }

        result = cloud_api_request(
            "PUT",
            f"/api/cloud/providers/{provider}/config",
            data=update_data,
        )

        if result.get("config") is not None:
            print(f"Cost limit set: ${amount} per {period} ({action})")
            return 0
        else:
            print("Error: Failed to set cost limit")
            return 1

    elif limit_action == "disable":
        result = cloud_api_request(
            "PUT",
            f"/api/cloud/providers/{provider}/config",
            data={"cost_limit_enabled": False},
        )

        if result.get("config") is not None:
            print("Cost limit disabled.")
            return 0
        else:
            print("Error: Failed to disable cost limit")
            return 1

    else:
        print("Error: Unknown cost-limit action. Use 'simpletuner cloud cost-limit --help'.")
        return 1
