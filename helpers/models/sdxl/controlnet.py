# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from diffusers import ControlNetModel as OriginalControlNetModel
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ControlNetModel(OriginalControlNetModel, PeftAdapterMixin):
    _supports_gradient_checkpointing = True
