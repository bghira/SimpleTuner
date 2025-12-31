#!/usr/bin/env python
"""
SimpleTuner CLI entry point.

This standalone script sets environment variables before importing the simpletuner
package to enable fast startup for simple API commands.
"""
import os
import sys

# Skip torch import for fast CLI startup (must be set before any simpletuner imports)
os.environ.setdefault("SIMPLETUNER_SKIP_TORCH", "1")


def main() -> int:
    """Entry point that wraps simpletuner.cli.main."""
    from simpletuner.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
