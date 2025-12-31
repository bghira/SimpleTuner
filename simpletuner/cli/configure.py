"""
Configure command for SimpleTuner CLI.

Provides an interactive configuration wizard.
"""

import sys
from pathlib import Path


def cmd_configure(args) -> int:
    """Handle configure command."""
    output_file = getattr(args, "output_file", "config.json")

    try:
        from simpletuner.configure import main as configure_main

        original_argv = sys.argv.copy()

        if Path(output_file).exists():
            sys.argv = ["configure.py", output_file]
            print(f"Loading existing configuration from: {output_file}")
        else:
            sys.argv = ["configure.py"]
            print(f"Creating new configuration. Will save to: {output_file}")

        try:
            configure_main()
            return 0
        except KeyboardInterrupt:
            print("\nConfiguration cancelled by user.")
            return 130
        except Exception as e:
            print(f"Error running configuration wizard: {e}")
            return 1
        finally:
            sys.argv = original_argv

    except ImportError as e:
        print(f"Error importing configuration module: {e}")
        return 1
