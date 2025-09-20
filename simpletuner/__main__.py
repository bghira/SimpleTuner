"""
Enable running SimpleTuner as a module: python -m simpletuner
"""

import sys

from simpletuner.cli import main

if __name__ == "__main__":
    sys.exit(main())
