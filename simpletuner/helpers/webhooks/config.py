from json import load

supported_webhooks = ["discord", "raw"]


def check_discord_webhook_config(config: dict) -> bool:
    if "webhook_type" not in config or config["webhook_type"] != "discord":
        return False
    if "webhook_url" not in config:
        raise ValueError("Discord webhook config is missing 'webhook_url' value.")
    return True


def check_raw_webhook_config(config: dict) -> bool:
    if config.get("webhook_type") != "raw":
        return False
    missing_fields = []
    required_fields = ["callback_url"]
    for config_field in required_fields:
        if not config.get(config_field):
            missing_fields.append(config_field)
    if missing_fields:
        raise ValueError(f"Missing fields on webhook config: {missing_fields}")
    return True


class WebhookConfig:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.values = self.load_config()
        if "webhook_type" not in self.values or self.values["webhook_type"] not in supported_webhooks:
            raise ValueError(f"Invalid webhook type specified in config. Supported values: {supported_webhooks}")
        if check_discord_webhook_config(self.values):
            self.webhook_type = "discord"
        elif check_raw_webhook_config(self.values):
            self.webhook_type = "raw"

    def load_config(self):
        with open(self.config_path, "r") as f:
            return load(f)

    def get_config(self):
        return self.values

    def __getattr__(self, name):
        return self.values.get(name, None)
