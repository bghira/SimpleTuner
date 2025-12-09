# Copyright 2025 The Hunyuan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Local thin wrapper around the upstream diffusers HunyuanVideo 1.5 transformer.

SimpleTuner needs a couple of helper methods (e.g. the training code checks
for ``set_router`` / ``set_gradient_checkpointing_interval`` on every
transformer).  The official diffusers implementation does not expose those
APIs, so we provide a lightweight subclass that forwards everything to the
original module while keeping the public surface compatible with the rest of
the codebase.
"""

from diffusers.models import HunyuanVideo15Transformer3DModel as DiffusersHunyuanVideo15Transformer3DModel


class HunyuanVideo15Transformer3DModel(DiffusersHunyuanVideo15Transformer3DModel):
    """
    Wrapper that preserves SimpleTuner's training hooks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tread_router = None
        self._tread_routes = None
        self.gradient_checkpointing_interval = None

    # ------------------------------------------------------------------ #
    # Optional hooks used by SimpleTuner's training loop
    # ------------------------------------------------------------------ #
    def set_router(self, router, routes=None):
        """
        TREAD routing is not yet supported for the diffusers HunyuanVideo 1.5
        backbone.  Keep the method so the training wizard can probe for the
        capability, but raise a clear error if somebody attempts to enable it.
        """
        self._tread_router = router
        self._tread_routes = routes or []
        if router is not None and self._tread_routes:
            raise NotImplementedError(
                "TREAD routing is not currently supported for the diffusers HunyuanVideo 1.5 transformer."
            )

    def set_gradient_checkpointing_interval(self, interval: int):
        """
        Store the requested checkpoint interval.  The upstream transformer does
        not expose per-layer interval control, but keeping the attribute avoids
        attribute errors elsewhere in the training stack.
        """
        self.gradient_checkpointing_interval = interval
