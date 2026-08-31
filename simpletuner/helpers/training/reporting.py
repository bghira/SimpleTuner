from __future__ import annotations

from typing import Any

REPORT_TO_CHOICES = (
    "simpletuner",
    "tensorboard",
    "wandb",
    "swanlab",
    "comet_ml",
    "custom-tracker",
    "none",
)


def report_to_tokens(report_to: Any) -> tuple[str, ...]:
    if isinstance(report_to, str):
        raw_values = report_to.split(",")
    elif isinstance(report_to, (list, tuple, set)):
        raw_values = []
        for value in report_to:
            raw_values.extend(str(value).split(","))
    else:
        return ()

    tokens: list[str] = []
    for value in raw_values:
        token = value.strip().lower()
        if token and token not in tokens:
            tokens.append(token)
    return tuple(tokens)


def normalize_report_to(report_to: Any) -> str | list[str] | None:
    tokens = report_to_tokens(report_to)
    if not tokens:
        return None

    invalid = [token for token in tokens if token not in REPORT_TO_CHOICES]
    if invalid:
        valid = ", ".join(REPORT_TO_CHOICES)
        raise ValueError(f"Unsupported --report_to value {invalid[0]!r}. Choose from: {valid}.")

    if "none" in tokens and len(tokens) > 1:
        raise ValueError("--report_to=none cannot be combined with other trackers.")

    if len(tokens) == 1:
        return tokens[0]
    return list(tokens)


def report_to_contains(report_to: Any, tracker_name: str) -> bool:
    return tracker_name.strip().lower() in report_to_tokens(report_to)


def report_to_is_disabled(report_to: Any) -> bool:
    tokens = report_to_tokens(report_to)
    return not tokens or tokens == ("none",)
