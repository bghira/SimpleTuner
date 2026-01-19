# Copyright 2025 SimpleTuner contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class HeartMuLaGenConfig:
    text_bos_id: int = 128000
    text_eos_id: int = 128001
    audio_eos_id: int = 8193
    empty_id: int = 0

    @classmethod
    def from_file(cls, path: str) -> "HeartMuLaGenConfig":
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        return cls(**data)
