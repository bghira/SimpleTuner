from ipaddress import ip_address
from json import load
from socket import IPPROTO_TCP, gaierror, getaddrinfo
from urllib.parse import urlparse

supported_webhooks = ["discord", "raw"]


def validate_webhook_url(url: str, field_name: str):
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https":
        raise ValueError(f"Webhook config '{field_name}' must use https.")
    if not parsed_url.hostname:
        raise ValueError(f"Webhook config '{field_name}' must include a valid hostname.")
    try:
        resolved_addresses = {
            address_info[4][0]
            for address_info in getaddrinfo(parsed_url.hostname, parsed_url.port or 443, proto=IPPROTO_TCP)
        }
    except gaierror as e:
        raise ValueError(f"Unable to resolve webhook hostname for '{field_name}': {e}")
    for resolved_address in resolved_addresses:
        parsed_ip = ip_address(resolved_address)
        if (
            parsed_ip.is_private
            or parsed_ip.is_loopback
            or parsed_ip.is_link_local
            or parsed_ip.is_multicast
            or parsed_ip.is_reserved
            or parsed_ip.is_unspecified
        ):
            raise ValueError(f"Webhook config '{field_name}' resolves to a disallowed address.")


def check_discord_webhook_config(config: dict) -> bool:
    if "webhook_type" not in config or config["webhook_type"] != "discord":
        return False
    if "webhook_url" not in config:
        raise ValueError("Discord webhook config is missing 'webhook_url' value.")
    validate_webhook_url(config["webhook_url"], "webhook_url")
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
    validate_webhook_url(config["callback_url"], "callback_url")
    return True


class WebhookConfig:
    def __init__(self, config: dict):
        self.config = config
        if "webhook_type" not in self.config or self.config["webhook_type"] not in supported_webhooks:
            raise ValueError(f"Invalid webhook type specified in config. Supported values: {supported_webhooks}")
        if check_discord_webhook_config(self.config):
            self.webhook_type = "discord"
        elif check_raw_webhook_config(self.config):
            self.webhook_type = "raw"
        self.webhook_url = self.config.get("webhook_url") or self.config.get("callback_url")

    def load_config(self):
        with open(self.config_path, "r") as f:
            return load(f)

    def get_config(self):
        return self.config

    def __getattr__(self, name):
        return self.config.get(name, None)
