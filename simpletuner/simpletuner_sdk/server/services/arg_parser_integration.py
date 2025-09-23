"""Integration between cmd_args.py and field registry for help text extraction."""

from typing import Dict, Optional, Any
import argparse
from simpletuner.helpers.configuration.cmd_args import get_argument_parser


class ArgParserIntegration:
    """Extract help text and metadata from cmd_args.py argument parser."""

    def __init__(self):
        """Initialize with the argument parser."""
        self._parser = get_argument_parser()
        self._arg_help_cache = None

    def get_all_arguments(self) -> Dict[str, Dict[str, Any]]:
        """Extract all arguments with their metadata from the parser.

        Returns:
            Dictionary mapping argument names to their metadata including help text
        """
        if self._arg_help_cache is not None:
            return self._arg_help_cache

        self._arg_help_cache = {}

        for action in self._parser._actions:
            # Skip help action
            if isinstance(action, argparse._HelpAction):
                continue

            # Get the primary argument name (prefer long form)
            arg_names = action.option_strings
            if not arg_names:
                continue

            # Use the long form (--argument) as the key
            primary_name = None
            for name in arg_names:
                if name.startswith("--"):
                    primary_name = name
                    break

            if not primary_name:
                primary_name = arg_names[0]

            # Extract metadata
            self._arg_help_cache[primary_name] = {
                "names": arg_names,
                "help": action.help or "",
                "type": action.type.__name__ if action.type else "str",
                "default": action.default,
                "choices": action.choices,
                "required": action.required if hasattr(action, 'required') else False,
                "dest": action.dest,
                "nargs": action.nargs,
                "const": action.const,
                "metavar": action.metavar
            }

            # Special handling for store_true/store_false actions
            if isinstance(action, argparse._StoreTrueAction):
                self._arg_help_cache[primary_name]["type"] = "bool"
                self._arg_help_cache[primary_name]["default"] = False
            elif isinstance(action, argparse._StoreFalseAction):
                self._arg_help_cache[primary_name]["type"] = "bool"
                self._arg_help_cache[primary_name]["default"] = True

        return self._arg_help_cache

    def get_argument_help(self, arg_name: str) -> Optional[str]:
        """Get help text for a specific argument.

        Args:
            arg_name: Argument name (with or without -- prefix)

        Returns:
            Help text or None if not found
        """
        # Ensure we have the cache
        args = self.get_all_arguments()

        # Normalize argument name
        if not arg_name.startswith("-"):
            arg_name = f"--{arg_name}"

        arg_info = args.get(arg_name)
        if arg_info:
            return arg_info["help"]

        # Try without dashes
        arg_name_no_dash = arg_name.lstrip("-")
        for key, info in args.items():
            if info["dest"] == arg_name_no_dash:
                return info["help"]

        return None

    def get_argument_metadata(self, arg_name: str) -> Optional[Dict[str, Any]]:
        """Get full metadata for a specific argument.

        Args:
            arg_name: Argument name (with or without -- prefix)

        Returns:
            Metadata dictionary or None if not found
        """
        # Ensure we have the cache
        args = self.get_all_arguments()

        # Normalize argument name
        if not arg_name.startswith("-"):
            arg_name = f"--{arg_name}"

        return args.get(arg_name)

    def format_help_for_ui(self, help_text: str) -> str:
        """Format help text for UI display.

        Args:
            help_text: Raw help text from argparse

        Returns:
            Formatted help text suitable for UI tooltips
        """
        if not help_text:
            return ""

        # Clean up formatting
        help_text = help_text.strip()

        # Replace multiple spaces with single space
        import re
        help_text = re.sub(r'\s+', ' ', help_text)

        # Remove internal references to args.*
        help_text = re.sub(r'`?args\.(\w+)`?', r'\1', help_text)

        # Format lists nicely
        help_text = help_text.replace(". ", ".\n")

        return help_text


# Global instance
arg_parser_integration = ArgParserIntegration()