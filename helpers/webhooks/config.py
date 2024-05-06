from json import load

supported_webhooks = ["discord"]


def check_discord_webhook_config(config: dict) -> bool:
    if "webhook_type" not in config or config["webhook_type"] != "discord":
        return
    if "webhook_url" not in config:
        raise ValueError("Discord webhook config is missing 'webhook_url' value.")
    return True


class WebhookConfig:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.values = self.load_config()
        if (
            "webhook_type" not in self.values
            or self.values["webhook_type"] not in supported_webhooks
        ):
            raise ValueError(
                f"Invalid webhook type specified in config. Supported values: {supported_webhooks}"
            )
        check_discord_webhook_config(self.values)

    def load_config(self):
        with open(self.config_path, "r") as f:
            return load(f)

    def get_config(self):
        return self.values

    def __getattr__(self, name):
        return self.values.get(name, None)
