"""
Enable running CLI module directly: python -m simpletuner.cli
"""

import sys

from simpletuner.cli import main

if __name__ == "__main__":
    sys.exit(main())
