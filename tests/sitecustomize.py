"""Site customization for tests - loaded automatically by Python before any imports."""

import logging

# Suppress PyTorch distributed warnings that appear on macOS/Windows
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
