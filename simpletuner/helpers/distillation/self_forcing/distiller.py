from __future__ import annotations

from typing import Any, Dict, Optional

from simpletuner.helpers.distillation.common import DistillationBase
from simpletuner.helpers.distillation.registry import DistillationRegistry

from .generator import SelfForcingODEGenerator


class SelfForcingDistillation(DistillationBase):
    """Lightweight distiller that demonstrates self-forcing ODE pair generation."""

    def __init__(
        self,
        teacher_model,
        student_model=None,
        *,
        noise_scheduler=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        default_config = {
            "distillation_type": "self_forcing",
            "ode_generator": {},
        }
        if config:
            default_config.update(config)

        super().__init__(teacher_model, student_model, default_config)
        self.noise_scheduler = noise_scheduler
        self._ode_generator = SelfForcingODEGenerator(config=default_config.get("ode_generator", {}))

    def requires_distillation_cache(self) -> bool:
        return True

    def get_required_distillation_cache_type(self) -> Optional[str]:
        return self.config.get("distillation_type", "self_forcing")

    def get_ode_generator_provider(self):
        return self._ode_generator


DistillationRegistry.register(
    "self_forcing",
    SelfForcingDistillation,
    requires_distillation_cache=True,
    distillation_type="self_forcing",
)
