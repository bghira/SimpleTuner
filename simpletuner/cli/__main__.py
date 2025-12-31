"""
Enable running CLI module directly: python -m simpletuner.cli
"""

import os
import sys

# Skip torch import for fast CLI startup (must be set before any simpletuner imports)
os.environ.setdefault("SIMPLETUNER_SKIP_TORCH", "1")

from . import main

if __name__ == "__main__":
    sys.exit(main())
