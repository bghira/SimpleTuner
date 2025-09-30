"""Test transformers package initialization."""

import logging

# Suppress PyTorch warnings before torch imports in transformer tests
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
