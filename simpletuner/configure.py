#!/usr/bin/env python3
import curses
import json
import os
import sys
import textwrap
import traceback
from typing import Any, Dict, List, Optional, Tuple

import huggingface_hub
import torch

from simpletuner.helpers.training import lycoris_defaults, quantised_precision_levels
from simpletuner.helpers.training.optimizer_param import optimizer_choices

# Constants
bf16_only_optims = [key for key, value in optimizer_choices.items() if value.get("precision", "any") == "bf16"]
any_precision_optims = [key for key, value in optimizer_choices.items() if value.get("precision", "any") == "any"]

model_classes = {
    "full": [
        "flux",
        "sdxl",
        "pixart_sigma",
        "kolors",
        "sd3",
        "sd1x",
        "sd2x",
        "ltxvideo",
        "wan",
        "sana",
        "deepfloyd",
        "omnigen",
        "hidream",
        "auraflow",
        "lumina2",
        "cosmos2image",
        "qwen_image",
        "chroma",
    ],
    "lora": [
        "flux",
        "sdxl",
        "kolors",
        "sd3",
        "sd1x",
        "sd2x",
        "ltxvideo",
        "wan",
        "deepfloyd",
        "auraflow",
        "hidream",
        "lumina2",
        "qwen_image",
        "chroma",
    ],
    "controlnet": [
        "sdxl",
        "sd1x",
        "sd2x",
        "hidream",
        "auraflow",
        "flux",
        "pixart_sigma",
        "sd3",
        "kolors",
        "chroma",
    ],
}

default_models = {
    "flux": "black-forest-labs/FLUX.1-dev",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "pixart_sigma": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    "kolors": "kwai-kolors/kolors-diffusers",
    "terminus": "ptx0/terminus-xl-velocity-v2",
    "sd3": "stabilityai/stable-diffusion-3.5-large",
    "sd2x": "stabilityai/stable-diffusion-2-1-base",
    "sd1x": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sana": "terminusresearch/sana-1.6b-1024px",
    "ltxvideo": "Lightricks/LTX-Video",
    "wan": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "hidream": "HiDream-ai/HiDream-I1-Full",
    "auraflow": "terminusresearch/auraflow-v0.3",
    "deepfloyd": "DeepFloyd/DeepFloyd-IF-I-XL-v1.0",
    "omnigen": "Shitao/OmniGen-v1-diffusers",
}

default_cfg = {
    "flux": 3.0,
    "sdxl": 4.2,
    "pixart_sigma": 3.4,
    "kolors": 5.0,
    "terminus": 8.0,
    "sd3": 5.0,
    "ltxvideo": 4.0,
    "hidream": 2.5,
    "wan": 4.0,
    "sana": 3.8,
    "omnigen": 3.2,
    "deepfloyd": 6.0,
    "sd2x": 7.0,
    "sd1x": 6.0,
}
model_labels = {
    "flux": "FLUX",
    "pixart_sigma": "PixArt Sigma",
    "kolors": "Kwai Kolors",
    "terminus": "Terminus",
    "sdxl": "Stable Diffusion XL",
    "sd3": "Stable Diffusion 3",
    "sd2x": "Stable Diffusion 2",
    "sd1x": "Stable Diffusion",
    "ltxvideo": "LTX Video",
    "wan": "WanX",
    "hidream": "HiDream I1",
    "sana": "Sana",
}

lora_ranks = [1, 16, 64, 128, 256]
learning_rates_by_rank = {
    1: "3e-4",
    16: "1e-4",
    64: "8e-5",
    128: "6e-5",
    256: "5.09e-5",
}


class ConfigState:
    """Holds the configuration state across navigation"""

    def __init__(self):
        self.env_contents = {
            "resume_from_checkpoint": "latest",
            "data_backend_config": "config/multidatabackend.json",
            "aspect_bucket_rounding": 2,
            "seed": 42,
            "minimum_image_size": 0,
            "disable_benchmark": False,
        }
        self.extra_args = []
        self.lycoris_config = None
        self.model_type = None
        self.use_lora = False
        self.use_lycoris = False
        self.lora_rank = 64
        self.whoami = None
        self.dataset_config = []
        self.completed_steps = set()
        self.loaded_config_path = None

    def load_from_file(self, config_path: str):
        """Load configuration from JSON file"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded_config = json.load(f)

            # Update env_contents with loaded values
            self.env_contents.update(loaded_config)

            # Extract model type
            if "model_type" in loaded_config:
                self.model_type = loaded_config["model_type"]

            # Check if using LoRA
            if "lora_type" in loaded_config:
                self.use_lora = True
                self.use_lycoris = loaded_config["lora_type"] == "lycoris"

            # Get LoRA rank if present
            if "lora_rank" in loaded_config:
                self.lora_rank = loaded_config["lora_rank"]

            # Load LyCORIS config if specified
            if "lycoris_config" in loaded_config and os.path.exists(loaded_config["lycoris_config"]):
                with open(loaded_config["lycoris_config"], "r", encoding="utf-8") as f:
                    self.lycoris_config = json.load(f)

            # Load dataset config if specified
            backend_config = loaded_config.get("data_backend_config", "config/multidatabackend.json")
            if os.path.exists(backend_config):
                with open(backend_config, "r", encoding="utf-8") as f:
                    self.dataset_config = json.load(f)

            self.loaded_config_path = config_path

            # Mark steps as completed based on what's configured
            if "output_dir" in loaded_config:
                self.completed_steps.add(0)  # Basic setup
            if "model_type" in loaded_config:
                self.completed_steps.add(1)  # Model type
            if "max_train_steps" in loaded_config or "num_train_epochs" in loaded_config:
                self.completed_steps.add(2)  # Training config
            if "model_family" in loaded_config:
                self.completed_steps.add(5)  # Model selection
            if "train_batch_size" in loaded_config:
                self.completed_steps.add(6)  # Training params
            if "optimizer" in loaded_config:
                self.completed_steps.add(7)  # Optimization
            if "validation_prompt" in loaded_config:
                self.completed_steps.add(11)  # Validation

            return True

        except Exception as e:
            return False


class MenuNavigator:
    """Helper class for menu navigation"""

    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.h, self.w = stdscr.getmaxyx()

    def show_menu(
        self,
        title: str,
        items: List[Tuple[str, Any]],
        current_values: Dict[str, str] = None,
        selected: int = 0,
    ) -> int:
        """Generic menu display with current values"""
        while True:
            self.stdscr.clear()

            # Title
            self.stdscr.addstr(1, 2, title, curses.A_BOLD)
            self.stdscr.addstr(2, 2, "─" * (self.w - 4))

            # Instructions
            self.stdscr.addstr(3, 2, "↑/↓: Navigate  Enter: Select  ←/Backspace: Back  q: Quit")
            self.stdscr.addstr(4, 2, "─" * (self.w - 4))

            # Menu items
            start_y = 6
            for idx, (item_name, _) in enumerate(items):
                if start_y + idx >= self.h - 2:
                    break

                # Highlight current selection
                attr = curses.A_REVERSE if idx == selected else curses.A_NORMAL

                # Build display text
                display_text = f"{idx + 1}. {item_name}"

                # Add current value if available
                if current_values and item_name in current_values:
                    value_text = current_values[item_name]
                    # Truncate if too long
                    max_value_len = self.w - len(display_text) - 10
                    if hasattr(value_text, "len") and len(value_text) > max_value_len:
                        value_text = "..." + value_text[-(max_value_len - 3) :]
                    display_text += f" [{value_text}]"

                # Ensure text fits
                if len(display_text) > self.w - 4:
                    display_text = display_text[: self.w - 7] + "..."

                self.stdscr.addstr(start_y + idx, 4, display_text, attr)

            self.stdscr.refresh()

            key = self.stdscr.getch()
            if key == ord("q"):
                return -1  # Quit
            elif key in [curses.KEY_LEFT, curses.KEY_BACKSPACE, 127, 8]:
                return -2  # Back
            elif key == curses.KEY_UP and selected > 0:
                selected -= 1
            elif key == curses.KEY_DOWN and selected < len(items) - 1:
                selected += 1
            elif key in [curses.KEY_ENTER, ord("\n"), ord("\r")]:
                return selected
            # Number key shortcuts
            elif ord("1") <= key <= ord("9"):
                num = key - ord("1")
                if num < len(items):
                    return num


class SimpleTunerNCurses:
    def __init__(self):
        self.state = ConfigState()
        self.current_step = 0
        self.menu_items = [
            ("Basic Setup", self.basic_setup),
            ("Model Type & LoRA/LyCORIS", self.model_type_setup),
            ("Training Configuration", self.training_config),
            ("Loss Configuration", self.loss_configuration),
            ("Hugging Face Hub", self.hub_setup),
            ("Model Selection", self.model_selection),
            ("Training Parameters", self.training_params),
            ("Optimization Settings", self.optimization_settings),
            ("VAE Configuration", self.vae_configuration),
            ("Flow Matching Configuration", self.flow_matching_config),
            ("Validation Settings", self.validation_settings),
            ("Advanced Options", self.advanced_options),
            ("ControlNet Configuration", self.controlnet_config),
            ("Model-Specific Options", self.model_specific_options),
            ("Dataset Configuration", self.dataset_config),
            ("Review & Save", self.review_and_save),
        ]

        # Try to load default config if it exists
        if os.path.exists("config/config.json"):
            self.state.load_from_file("config/config.json")

    def run(self):
        """Main entry point with error handling"""
        try:
            curses.wrapper(self._main_loop)
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

    def _main_loop(self, stdscr):
        """Main curses loop"""
        curses.curs_set(0)  # Hide cursor
        stdscr.clear()

        # Show startup screen
        self.show_startup_screen(stdscr)

        while True:
            try:
                action = self.show_main_menu(stdscr)
                if action == "quit":
                    break
                elif action is not None:
                    self.current_step = action
                    try:
                        self.menu_items[action][1](stdscr)
                        self.state.completed_steps.add(action)
                    except Exception as e:
                        import traceback

                        self.show_error(
                            stdscr,
                            f"Error in {self.menu_items[action][0]}: {str(e)}\n{traceback.format_exc()}",
                        )
            except KeyboardInterrupt:
                if self.confirm_quit(stdscr):
                    break

    def show_startup_screen(self, stdscr):
        """Show startup screen with loaded config info"""
        h, w = stdscr.getmaxyx()
        stdscr.clear()

        # ASCII art title (simple)
        title_lines = [
            "╔═══════════════════════════════════════╗",
            "║      SimpleTuner Configuration        ║",
            "║           ncurses Edition             ║",
            "╚═══════════════════════════════════════╝",
        ]

        start_y = (h - len(title_lines) - 6) // 2
        for idx, line in enumerate(title_lines):
            x = (w - len(line)) // 2
            stdscr.addstr(start_y + idx, x, line, curses.A_BOLD)

        info_y = start_y + len(title_lines) + 2

        if self.state.loaded_config_path:
            info = f"Loaded configuration: {self.state.loaded_config_path}"
            stdscr.addstr(info_y, (w - len(info)) // 2, info, curses.A_DIM)
            status = "Ready to modify existing configuration"
        else:
            status = "No configuration loaded - starting fresh"

        stdscr.addstr(info_y + 1, (w - len(status)) // 2, status)

        prompt = "Press any key to continue..."
        stdscr.addstr(h - 2, (w - len(prompt)) // 2, prompt, curses.A_DIM)

        stdscr.refresh()
        stdscr.getch()

    def show_main_menu(self, stdscr) -> Optional[int]:
        """Display the main menu/TOC"""
        stdscr.clear()
        h, w = stdscr.getmaxyx()

        # Title
        title = "SimpleTuner Configuration"
        stdscr.addstr(1, (w - len(title)) // 2, title, curses.A_BOLD)

        # Show loaded config info
        if self.state.loaded_config_path:
            info = f"Loaded: {self.state.loaded_config_path}"
            if len(info) > w - 4:
                info = "..." + info[-(w - 7) :]
            stdscr.addstr(2, 2, info, curses.A_DIM)

        stdscr.addstr(
            3,
            2,
            "↑/↓: Navigate  Enter: Select  'l': Load config  'q': Quit",
        )

        # Menu items
        start_y = 5
        selected = self.current_step
        max_visible = h - start_y - 2
        scroll_offset = 0

        if selected >= max_visible:
            scroll_offset = selected - max_visible + 1

        while True:
            visible_items = self.menu_items[scroll_offset : scroll_offset + max_visible]

            for idx, (item_name, _) in enumerate(visible_items):
                actual_idx = idx + scroll_offset
                y = start_y + idx

                # Highlight current selection
                attr = curses.A_REVERSE if actual_idx == selected else curses.A_NORMAL

                # Mark completed steps
                prefix = "[✓] " if actual_idx in self.state.completed_steps else "[ ] "
                text = f"{prefix}{actual_idx + 1}. {item_name}"

                # Ensure text fits
                if len(text) > w - 4:
                    text = text[: w - 7] + "..."

                stdscr.addstr(y, 2, text, attr)

            # Show scroll indicators
            if scroll_offset > 0:
                stdscr.addstr(start_y - 1, w - 10, "▲ More", curses.A_DIM)
            if scroll_offset + max_visible < len(self.menu_items):
                stdscr.addstr(h - 2, w - 10, "▼ More", curses.A_DIM)

            stdscr.refresh()

            key = stdscr.getch()
            if key == ord("q"):
                if self.confirm_quit(stdscr):
                    return "quit"
            elif key == ord("l"):
                if self.load_config_dialog(stdscr):
                    # Refresh the menu to show updated state
                    return None
            elif key == curses.KEY_UP and selected > 0:
                selected -= 1
                if selected < scroll_offset:
                    scroll_offset = selected
            elif key == curses.KEY_DOWN and selected < len(self.menu_items) - 1:
                selected += 1
                if selected >= scroll_offset + max_visible:
                    scroll_offset = selected - max_visible + 1
            elif key in [curses.KEY_ENTER, ord("\n"), ord("\r")]:
                return selected

    def show_error(self, stdscr, error_msg: str):
        """Display an error message and wait for acknowledgment"""
        h, w = stdscr.getmaxyx()

        # Create error window
        error_lines = textwrap.wrap(error_msg, w - 10)
        error_h = len(error_lines) + 4
        error_w = min(80, w - 4)

        error_win = curses.newwin(error_h, error_w, (h - error_h) // 2, (w - error_w) // 2)
        error_win.box()
        error_win.addstr(0, 2, " Error ", curses.A_BOLD | curses.color_pair(1))

        for idx, line in enumerate(error_lines):
            error_win.addstr(idx + 1, 2, line)

        error_win.addstr(error_h - 2, 2, "Press any key to continue...")
        error_win.refresh()
        error_win.getch()

    def get_input(
        self,
        stdscr,
        prompt: str,
        default: str = "",
        validation_fn=None,
        multiline=False,
    ) -> str:
        """Generic input function with validation"""
        h, w = stdscr.getmaxyx()

        # Clear and display prompt
        stdscr.clear()
        wrapped_prompt = textwrap.wrap(prompt, w - 4)
        for idx, line in enumerate(wrapped_prompt):
            stdscr.addstr(2 + idx, 2, line)

        if default:
            stdscr.addstr(2 + len(wrapped_prompt) + 1, 2, f"Default: {default}")

        input_y = 2 + len(wrapped_prompt) + 3
        stdscr.addstr(input_y, 2, "> ")

        curses.echo()
        curses.curs_set(1)

        try:
            if multiline:
                lines = []
                stdscr.addstr(input_y + 1, 2, "(Press Ctrl+D when done)")
                # Note: multiline input is complex in curses, simplified here
                user_input = stdscr.getstr(input_y, 4).decode("utf-8")
            else:
                user_input = stdscr.getstr(input_y, 4, w - 6).decode("utf-8")

            if not user_input and default:
                user_input = default

            if validation_fn and not validation_fn(user_input):
                raise ValueError("Invalid input")

            return user_input

        finally:
            curses.noecho()
            curses.curs_set(0)

    def show_options(self, stdscr, prompt: str, options: List[str], default: int = 0) -> int:
        """Show a list of options and return the selected index"""
        stdscr.clear()
        h, w = stdscr.getmaxyx()

        # Display prompt
        wrapped_prompt = textwrap.wrap(prompt, w - 4)
        for idx, line in enumerate(wrapped_prompt):
            stdscr.addstr(2 + idx, 2, line)

        start_y = 2 + len(wrapped_prompt) + 2
        selected = default

        while True:
            for idx, option in enumerate(options):
                if start_y + idx >= h - 2:
                    break

                attr = curses.A_REVERSE if idx == selected else curses.A_NORMAL
                text = f"{idx + 1}. {option}"
                if len(text) > w - 4:
                    text = text[: w - 7] + "..."
                stdscr.addstr(start_y + idx, 4, text, attr)

            stdscr.refresh()

            key = stdscr.getch()
            if key == curses.KEY_UP and selected > 0:
                selected -= 1
            elif key == curses.KEY_DOWN and selected < len(options) - 1:
                selected += 1
            elif key in [curses.KEY_ENTER, ord("\n"), ord("\r")]:
                return selected
            elif key == 27:  # ESC
                return -1

    def confirm_quit(self, stdscr) -> bool:
        """Confirm quit dialog"""
        return (
            self.show_options(
                stdscr,
                "Are you sure you want to quit? Unsaved changes will be lost.",
                ["No, continue", "Yes, quit"],
                0,
            )
            == 1
        )

    def find_config_files(self, base_path: str = "config") -> List[str]:
        """Find all config.json files in the config directory and subdirectories"""
        config_files = []

        if os.path.exists(base_path):
            for root, dirs, files in os.walk(base_path):
                if "config.json" in files:
                    config_path = os.path.join(root, "config.json")
                    config_files.append(config_path)

        return sorted(config_files)

    def load_config_dialog(self, stdscr) -> bool:
        """Show dialog to load a configuration file"""
        # Find available config files
        config_files = self.find_config_files()

        if not config_files:
            self.show_error(stdscr, "No config.json files found in config/ directory")
            return False

        # Add options for manual entry and new config
        options = ["Create new configuration", "Enter path manually"] + config_files

        selected = self.show_options(
            stdscr,
            "Select a configuration to load:",
            options,
            2 if len(config_files) > 0 else 0,
        )

        if selected == -1:  # ESC pressed
            return False

        if selected == 0:  # New configuration
            self.state = ConfigState()  # Reset to defaults
            self.state.loaded_config_path = None
            return True

        elif selected == 1:  # Manual entry
            config_path = self.get_input(stdscr, "Enter path to config.json:", "config/config.json")
            if os.path.exists(config_path):
                if self.state.load_from_file(config_path):
                    self.show_message(stdscr, f"Successfully loaded: {config_path}")
                    return True
                else:
                    self.show_error(stdscr, f"Failed to load: {config_path}")
                    return False
            else:
                self.show_error(stdscr, f"File not found: {config_path}")
                return False

        else:  # Selected a file from the list
            config_path = config_files[selected - 2]
            if self.state.load_from_file(config_path):
                self.show_message(stdscr, f"Successfully loaded: {config_path}")
                return True
            else:
                self.show_error(stdscr, f"Failed to load: {config_path}")
                return False

    def show_message(self, stdscr, message: str):
        """Display an informational message"""
        h, w = stdscr.getmaxyx()

        # Create message window
        msg_lines = textwrap.wrap(message, w - 10)
        msg_h = len(msg_lines) + 4
        msg_w = min(80, w - 4)

        msg_win = curses.newwin(msg_h, msg_w, (h - msg_h) // 2, (w - msg_w) // 2)
        msg_win.box()
        msg_win.addstr(0, 2, " Info ", curses.A_BOLD)

        for idx, line in enumerate(msg_lines):
            msg_win.addstr(idx + 1, 2, line)

        msg_win.addstr(msg_h - 2, 2, "Press any key to continue...")
        msg_win.refresh()
        msg_win.getch()

    def basic_setup(self, stdscr):
        """Step 1: Basic Setup - Now with menu"""
        nav = MenuNavigator(stdscr)

        while True:
            # Get current values
            current_values = {
                "Output Directory": self.state.env_contents.get("output_dir", "output/models"),
                "Resume from Checkpoint": self.state.env_contents.get("resume_from_checkpoint", "latest"),
                "Aspect Bucket Rounding": str(self.state.env_contents.get("aspect_bucket_rounding", 2)),
                "Seed": str(self.state.env_contents.get("seed", 42)),
                "Minimum Image Size": str(self.state.env_contents.get("minimum_image_size", 0)),
            }

            menu_items = [
                ("Output Directory", self._configure_output_dir),
                ("Resume from Checkpoint", self._configure_resume),
                ("Aspect Bucket Rounding", self._configure_aspect_rounding),
                ("Seed", self._configure_seed),
                ("Minimum Image Size", self._configure_min_image_size),
            ]

            selected = nav.show_menu("Basic Setup", menu_items, current_values)

            if selected == -1:  # Quit
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif selected == -2:  # Back
                return
            elif selected >= 0:
                # Execute the configuration function
                menu_items[selected][1](stdscr)

    def _configure_output_dir(self, stdscr):
        """Configure output directory"""
        current = self.state.env_contents.get("output_dir", "output/models")
        output_dir = self.get_input(
            stdscr,
            f"Enter the directory where you want to store your outputs\n(Current: {current}):",
            current,
        )

        if not os.path.exists(output_dir):
            if (
                self.show_options(
                    stdscr,
                    f"Directory {output_dir} doesn't exist. Create it?",
                    ["Yes", "No, choose another"],
                    0,
                )
                == 0
            ):
                os.makedirs(output_dir, exist_ok=True)
            else:
                return self._configure_output_dir(stdscr)

        self.state.env_contents["output_dir"] = output_dir

    def _configure_resume(self, stdscr):
        """Configure resume from checkpoint"""
        options = ["latest", "none", "Custom path"]
        current = self.state.env_contents.get("resume_from_checkpoint", "latest")

        # Find current option index
        default_idx = 0
        if current == "none":
            default_idx = 1
        elif current not in ["latest", "none"]:
            default_idx = 2

        selected = self.show_options(
            stdscr,
            f"Resume from checkpoint configuration\n(Current: {current})",
            options,
            default_idx,
        )

        if selected == 0:
            self.state.env_contents["resume_from_checkpoint"] = "latest"
        elif selected == 1:
            self.state.env_contents["resume_from_checkpoint"] = "none"
        elif selected == 2:
            custom_path = self.get_input(
                stdscr,
                "Enter checkpoint path:",
                current if current not in ["latest", "none"] else "",
            )
            self.state.env_contents["resume_from_checkpoint"] = custom_path

    def _configure_aspect_rounding(self, stdscr):
        """Configure aspect bucket rounding"""
        current = str(self.state.env_contents.get("aspect_bucket_rounding", 2))
        value = self.get_input(stdscr, "Set aspect bucket rounding (1-8, higher = more precise):", current)
        try:
            self.state.env_contents["aspect_bucket_rounding"] = int(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_seed(self, stdscr):
        """Configure training seed"""
        current = str(self.state.env_contents.get("seed", 42))
        value = self.get_input(stdscr, "Set training seed (for reproducibility):", current)
        try:
            self.state.env_contents["seed"] = int(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_min_image_size(self, stdscr):
        """Configure minimum image size"""
        current = str(self.state.env_contents.get("minimum_image_size", 0))
        value = self.get_input(stdscr, "Set minimum image size (0 to disable filtering):", current)
        try:
            self.state.env_contents["minimum_image_size"] = int(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def model_type_setup(self, stdscr):
        """Step 2: Model Type & LoRA/LyCORIS Setup - Now with menu"""
        nav = MenuNavigator(stdscr)

        while True:
            # Build current values and menu based on model type
            current_type = self.state.env_contents.get("model_type", "lora")

            current_values = {
                "Model Type": current_type.upper(),
            }

            menu_items = [
                ("Model Type", self._configure_model_type),
            ]

            if current_type == "lora":
                # Add LoRA-specific options
                lora_type = self.state.env_contents.get("lora_type", "standard")
                current_values["LoRA Type"] = "LyCORIS" if lora_type == "lycoris" else "Standard"
                menu_items.append(("LoRA Type", self._configure_lora_type))

                if lora_type == "lycoris":
                    current_values["LyCORIS Algorithm"] = "Configured" if self.state.lycoris_config else "Not configured"
                    menu_items.append(("LyCORIS Algorithm", self.configure_lycoris))
                else:
                    # Standard LoRA options
                    use_dora = self.state.env_contents.get("use_dora", "false") == "true"
                    current_values["DoRA"] = "Enabled" if use_dora else "Disabled"
                    current_values["LoRA Rank"] = str(self.state.env_contents.get("lora_rank", 64))
                    current_values["LoRA Alpha"] = str(self.state.env_contents.get("lora_alpha", 64))
                    current_values["LoRA Dropout"] = str(self.state.env_contents.get("lora_dropout", 0.0))
                    menu_items.extend(
                        [
                            ("DoRA", self._configure_dora),
                            ("LoRA Rank", self._configure_lora_rank),
                            ("LoRA Alpha", self._configure_lora_alpha),
                            ("LoRA Dropout", self._configure_lora_dropout),
                        ]
                    )
            else:
                # Full fine-tuning options
                use_ema = self.state.env_contents.get("use_ema", "false") == "true"
                current_values["EMA"] = "Enabled" if use_ema else "Disabled"
                menu_items.append(("EMA", self._configure_ema))

            selected = nav.show_menu("Model Type & LoRA/LyCORIS Configuration", menu_items, current_values)

            if selected == -1:  # Quit
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif selected == -2:  # Back
                return
            elif selected >= 0:
                # Execute the configuration function
                menu_items[selected][1](stdscr)

    def _configure_model_type(self, stdscr):
        """Configure model type (LoRA vs Full)"""
        current_type = self.state.env_contents.get("model_type", "lora")

        model_type_idx = self.show_options(
            stdscr,
            f"What type of model are you training?\n(Current: {current_type})",
            ["LoRA", "Full"],
            0 if current_type == "lora" else 1,
        )

        if model_type_idx >= 0:
            self.state.model_type = "lora" if model_type_idx == 0 else "full"
            self.state.env_contents["model_type"] = self.state.model_type

            if self.state.model_type == "lora":
                self.state.use_lora = True
                # Ensure we have a default lora_type
                if "lora_type" not in self.state.env_contents:
                    self.state.env_contents["lora_type"] = "standard"

    def _configure_lora_type(self, stdscr):
        """Configure LoRA type (Standard vs LyCORIS)"""
        current_lora_type = self.state.env_contents.get("lora_type", "standard")

        use_lycoris_idx = self.show_options(
            stdscr,
            f"Would you like to train a LyCORIS model?\n(Current: {'LyCORIS' if current_lora_type == 'lycoris' else 'Standard'})",
            ["No (Standard LoRA)", "Yes (LyCORIS)"],
            1 if current_lora_type == "lycoris" else 0,
        )

        if use_lycoris_idx == 1:
            self.state.use_lycoris = True
            self.state.env_contents["lora_type"] = "lycoris"
        else:
            self.state.use_lycoris = False
            self.state.env_contents["lora_type"] = "standard"

    def _configure_dora(self, stdscr):
        """Configure DoRA"""
        current_dora = self.state.env_contents.get("use_dora", "false") == "true"

        use_dora_idx = self.show_options(
            stdscr,
            f"Would you like to train a DoRA model?\n(Current: {'Yes' if current_dora else 'No'})",
            ["No", "Yes"],
            1 if current_dora else 0,
        )

        if use_dora_idx == 1:
            self.state.env_contents["use_dora"] = "true"
        elif "use_dora" in self.state.env_contents:
            del self.state.env_contents["use_dora"]

    def _configure_lora_rank(self, stdscr):
        """Configure LoRA rank"""
        current_rank = self.state.env_contents.get("lora_rank", 64)
        rank_options = [str(r) for r in lora_ranks]

        # Find current rank index
        default_rank_idx = 2  # Default to 64
        if current_rank in lora_ranks:
            default_rank_idx = lora_ranks.index(current_rank)

        rank_idx = self.show_options(
            stdscr,
            f"Set the LoRA rank:\n(Current: {current_rank})",
            rank_options,
            default_rank_idx,
        )

        if rank_idx >= 0:
            self.state.lora_rank = lora_ranks[rank_idx]
            self.state.env_contents["lora_rank"] = self.state.lora_rank

    def _configure_lora_alpha(self, stdscr):
        """Configure LoRA alpha"""
        current = str(self.state.env_contents.get("lora_alpha", self.state.lora_rank))
        value = self.get_input(
            stdscr,
            f"Set the LoRA alpha (typically same as rank):\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["lora_alpha"] = int(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_lora_dropout(self, stdscr):
        """Configure LoRA dropout"""
        current = str(self.state.env_contents.get("lora_dropout", 0.0))
        value = self.get_input(
            stdscr,
            f"Set the LoRA dropout (0.0 to disable):\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["lora_dropout"] = float(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_ema(self, stdscr):
        """Configure EMA for full fine-tuning"""
        current_ema = self.state.env_contents.get("use_ema", "false") == "true"

        use_ema_idx = self.show_options(
            stdscr,
            f"Would you like to use EMA for training?\n(Current: {'Yes' if current_ema else 'No'})",
            ["No", "Yes"],
            1 if current_ema else 0,
        )

        if use_ema_idx == 1:
            self.state.env_contents["use_ema"] = "true"
        elif "use_ema" in self.state.env_contents:
            del self.state.env_contents["use_ema"]

    def configure_lycoris(self, stdscr):
        """Configure LyCORIS settings - Now with menu navigation"""
        nav = MenuNavigator(stdscr)

        # Initialize with defaults or loaded config
        if not self.state.lycoris_config:
            self.state.lycoris_config = {"algo": "lora"}

        while True:
            # Build current values
            config = self.state.lycoris_config
            current_values = {
                "Algorithm": config.get("algo", "lora").upper(),
                "Multiplier": str(config.get("multiplier", 1.0)),
                "Linear Dimension": str(config.get("linear_dim", 1000000)),
                "Linear Alpha": str(config.get("linear_alpha", 1)),
            }

            menu_items = [
                ("Algorithm", self._configure_lycoris_algorithm),
                ("Multiplier", self._configure_lycoris_multiplier),
                ("Linear Dimension", self._configure_lycoris_linear_dim),
                ("Linear Alpha", self._configure_lycoris_linear_alpha),
            ]

            # Add algorithm-specific options
            algo = config.get("algo", "lora")
            if algo == "lokr":
                current_values["Factor"] = str(config.get("factor", 16))
                menu_items.append(("Factor", self._configure_lycoris_factor))
            elif algo == "dylora":
                current_values["Block Size"] = str(config.get("block_size", 0))
                menu_items.append(("Block Size", self._configure_lycoris_block_size))
            elif algo in ["diag-oft", "boft"]:
                current_values["Constraint"] = "Yes" if config.get("constraint", False) else "No"
                current_values["Rescaled"] = "Yes" if config.get("rescaled", False) else "No"
                menu_items.append(("Constraint", self._configure_lycoris_constraint))
                menu_items.append(("Rescaled", self._configure_lycoris_rescaled))

            menu_items.append(("Save Configuration", self._save_lycoris_config))

            selected = nav.show_menu("LyCORIS Configuration", menu_items, current_values)

            if selected == -1:  # Quit
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif selected == -2:  # Back
                return
            elif selected >= 0:
                # Execute the configuration function
                if selected == len(menu_items) - 1:  # Save Configuration
                    menu_items[selected][1](stdscr)
                    return
                else:
                    menu_items[selected][1](stdscr)

    def _configure_lycoris_algorithm(self, stdscr):
        """Configure LyCORIS algorithm"""
        algorithms = [
            (
                "LoRA",
                "lora",
                "Efficient, balanced fine-tuning. Good for general tasks.",
            ),
            (
                "LoHa",
                "loha",
                "Advanced, strong dampening. Ideal for multi-concept fine-tuning.",
            ),
            (
                "LoKr",
                "lokr",
                "Kronecker product-based. Use for complex transformations.",
            ),
            ("Full", "full", "Traditional full model tuning."),
            ("IA³", "ia3", "Efficient, tiny files, best for styles."),
            ("DyLoRA", "dylora", "Dynamic updates, efficient with large dims."),
            ("Diag-OFT", "diag-oft", "Fast convergence with orthogonal fine-tuning."),
            ("BOFT", "boft", "Advanced version of Diag-OFT with more flexibility."),
            ("GLoRA", "glora", "Generalized LoRA."),
        ]

        # Show detailed algorithm descriptions
        stdscr.clear()
        stdscr.addstr(1, 2, "LyCORIS Algorithm Selection", curses.A_BOLD)
        stdscr.addstr(3, 2, "Choose an algorithm:")

        # Display each algorithm with its description
        y = 5
        for idx, (name, key, desc) in enumerate(algorithms):
            if y + 1 < stdscr.getmaxyx()[0] - 2:
                stdscr.addstr(y, 2, f"{idx + 1}. {name} - {desc}")
                y += 1

        stdscr.addstr(y + 1, 2, "Press number to select...")
        stdscr.refresh()

        # Get algorithm choice
        while True:
            key = stdscr.getch()
            if ord("1") <= key <= ord("9"):
                algo_idx = key - ord("1")
                if algo_idx < len(algorithms):
                    break

        algo = algorithms[algo_idx][1]

        # Apply algorithm-specific defaults
        if algo in lycoris_defaults:
            self.state.lycoris_config.update(lycoris_defaults[algo].copy())
        else:
            self.state.lycoris_config = {"algo": algo}

    def _configure_lycoris_multiplier(self, stdscr):
        """Configure LyCORIS multiplier"""
        current = self.state.lycoris_config.get("multiplier", 1.0)
        value = self.get_input(
            stdscr,
            f"Set the effect multiplier. Adjust for stronger or subtler effects.\n(Current: {current})",
            str(current),
        )
        try:
            self.state.lycoris_config["multiplier"] = float(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_lycoris_linear_dim(self, stdscr):
        """Configure LyCORIS linear dimension"""
        current = self.state.lycoris_config.get("linear_dim", 1000000)
        value = self.get_input(
            stdscr,
            f"Set the linear dimension. Higher values mean more capacity but use more resources.\n(Current: {current})",
            str(current),
        )
        try:
            self.state.lycoris_config["linear_dim"] = int(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_lycoris_linear_alpha(self, stdscr):
        """Configure LyCORIS linear alpha"""
        current = self.state.lycoris_config.get("linear_alpha", 1)
        value = self.get_input(
            stdscr,
            f"Set the alpha scaling factor. Controls the impact on the model.\n(Current: {current})",
            str(current),
        )
        try:
            self.state.lycoris_config["linear_alpha"] = int(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_lycoris_factor(self, stdscr):
        """Configure LyCORIS factor (for LoKr)"""
        current = self.state.lycoris_config.get("factor", 16)
        value = self.get_input(
            stdscr,
            f"Set the factor for compression/expansion.\n(Current: {current})",
            str(current),
        )
        try:
            self.state.lycoris_config["factor"] = int(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_lycoris_block_size(self, stdscr):
        """Configure LyCORIS block size (for DyLoRA)"""
        current = self.state.lycoris_config.get("block_size", 0)
        value = self.get_input(
            stdscr,
            f"Set block size for DyLoRA (rows/columns updated per step).\n(Current: {current})",
            str(current),
        )
        try:
            self.state.lycoris_config["block_size"] = int(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_lycoris_constraint(self, stdscr):
        """Configure LyCORIS constraint (for OFT variants)"""
        current = self.state.lycoris_config.get("constraint", False)
        idx = self.show_options(
            stdscr,
            f"Enforce constraints (e.g., orthogonality)?\n(Current: {'Yes' if current else 'No'})",
            ["No", "Yes"],
            1 if current else 0,
        )
        if idx >= 0:
            self.state.lycoris_config["constraint"] = idx == 1

    def _configure_lycoris_rescaled(self, stdscr):
        """Configure LyCORIS rescaled (for OFT variants)"""
        current = self.state.lycoris_config.get("rescaled", False)
        idx = self.show_options(
            stdscr,
            f"Rescale transformations? Adjusts model impact.\n(Current: {'Yes' if current else 'No'})",
            ["No", "Yes"],
            1 if current else 0,
        )
        if idx >= 0:
            self.state.lycoris_config["rescaled"] = idx == 1

    def _save_lycoris_config(self, stdscr):
        """Save LyCORIS configuration"""
        os.makedirs("config", exist_ok=True)
        with open("config/lycoris_config.json", "w", encoding="utf-8") as f:
            json.dump(self.state.lycoris_config, f, indent=4)

        self.state.env_contents["lycoris_config"] = "config/lycoris_config.json"
        self.show_message(stdscr, "LyCORIS configuration saved successfully!")

    def training_config(self, stdscr):
        """Step 3: Training Configuration - Now with menu"""
        nav = MenuNavigator(stdscr)

        while True:
            # Get current values
            current_values = {}

            if "max_train_steps" in self.state.env_contents and self.state.env_contents["max_train_steps"] > 0:
                current_values["Training Duration"] = f"{self.state.env_contents['max_train_steps']} steps"
            elif "num_train_epochs" in self.state.env_contents and self.state.env_contents["num_train_epochs"] > 0:
                current_values["Training Duration"] = f"{self.state.env_contents['num_train_epochs']} epochs"
            else:
                current_values["Training Duration"] = "Not configured"

            current_values["Checkpointing Interval"] = f"{self.state.env_contents.get('checkpointing_steps', 500)} steps"
            current_values["Checkpoints to Keep"] = str(self.state.env_contents.get("checkpoints_total_limit", 5))

            # Scheduler configuration
            current_values["Training Scheduler"] = self.state.env_contents.get(
                "training_scheduler_timestep_spacing", "trailing"
            )
            current_values["Inference Scheduler"] = self.state.env_contents.get(
                "inference_scheduler_timestep_spacing", "trailing"
            )

            # Timestep bias
            current_values["Timestep Bias"] = self.state.env_contents.get("timestep_bias_strategy", "none")

            menu_items = [
                ("Training Duration", self._configure_training_duration),
                ("Checkpointing Interval", self._configure_checkpoint_interval),
                ("Checkpoints to Keep", self._configure_checkpoint_limit),
                ("Training Scheduler Spacing", self._configure_training_scheduler),
                ("Inference Scheduler Spacing", self._configure_inference_scheduler),
                ("Timestep Bias Strategy", self._configure_timestep_bias),
            ]

            # Add timestep bias options if enabled
            if self.state.env_contents.get("timestep_bias_strategy", "none") != "none":
                current_values["Bias Multiplier"] = str(self.state.env_contents.get("timestep_bias_multiplier", 1.0))
                menu_items.append(("Bias Multiplier", self._configure_bias_multiplier))

                if self.state.env_contents.get("timestep_bias_strategy") == "range":
                    current_values["Bias Begin"] = str(self.state.env_contents.get("timestep_bias_begin", 0))
                    current_values["Bias End"] = str(self.state.env_contents.get("timestep_bias_end", 1000))
                    menu_items.extend(
                        [
                            ("Bias Begin", self._configure_bias_begin),
                            ("Bias End", self._configure_bias_end),
                        ]
                    )

            selected = nav.show_menu("Training Configuration", menu_items, current_values)

            if selected == -1:  # Quit
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif selected == -2:  # Back
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_training_duration(self, stdscr):
        """Configure training duration (steps vs epochs)"""
        count_type_idx = self.show_options(
            stdscr,
            "Should we schedule the end of training by epochs or steps?",
            ["Steps", "Epochs"],
            0,
        )

        if count_type_idx == 0:
            max_steps = self.get_input(stdscr, "Set the maximum number of steps:", "10000")
            try:
                self.state.env_contents["max_train_steps"] = int(max_steps)
                self.state.env_contents["num_train_epochs"] = 0
            except ValueError:
                self.state.env_contents["max_train_steps"] = 10000
                self.state.env_contents["num_train_epochs"] = 0
        else:
            max_epochs = self.get_input(stdscr, "Set the maximum number of epochs:", "100")
            try:
                self.state.env_contents["num_train_epochs"] = int(max_epochs)
                self.state.env_contents["max_train_steps"] = 0
            except ValueError:
                self.state.env_contents["num_train_epochs"] = 100
                self.state.env_contents["max_train_steps"] = 0

    def _configure_checkpoint_interval(self, stdscr):
        """Configure checkpointing interval"""
        default_interval = 500
        if self.state.env_contents.get("max_train_steps", 0) > 0:
            if self.state.env_contents["max_train_steps"] < default_interval:
                default_interval = self.state.env_contents["max_train_steps"] // 10

        checkpoint_interval = self.get_input(stdscr, "Set the checkpointing interval (in steps):", str(default_interval))

        try:
            self.state.env_contents["checkpointing_steps"] = int(checkpoint_interval)
        except ValueError:
            self.state.env_contents["checkpointing_steps"] = default_interval

    def _configure_checkpoint_limit(self, stdscr):
        """Configure checkpoint limit"""
        checkpoint_limit = self.get_input(stdscr, "How many checkpoints do you want to keep?", "5")

        try:
            self.state.env_contents["checkpoints_total_limit"] = int(checkpoint_limit)
        except ValueError:
            self.state.env_contents["checkpoints_total_limit"] = 5

    def _configure_training_scheduler(self, stdscr):
        """Configure training scheduler timestep spacing"""
        options = ["leading", "linspace", "trailing"]
        current = self.state.env_contents.get("training_scheduler_timestep_spacing", "trailing")

        default_idx = 2  # trailing
        if current in options:
            default_idx = options.index(current)

        selected = self.show_options(
            stdscr,
            f"Training scheduler timestep spacing (SDXL Only)\n(Current: {current})",
            options,
            default_idx,
        )

        if selected >= 0:
            self.state.env_contents["training_scheduler_timestep_spacing"] = options[selected]

    def _configure_inference_scheduler(self, stdscr):
        """Configure inference scheduler timestep spacing"""
        options = ["leading", "linspace", "trailing"]
        current = self.state.env_contents.get("inference_scheduler_timestep_spacing", "trailing")

        default_idx = 2  # trailing
        if current in options:
            default_idx = options.index(current)

        selected = self.show_options(
            stdscr,
            f"Inference scheduler timestep spacing (SDXL Only)\n(Current: {current})",
            options,
            default_idx,
        )

        if selected >= 0:
            self.state.env_contents["inference_scheduler_timestep_spacing"] = options[selected]

    def _configure_timestep_bias(self, stdscr):
        """Configure timestep bias strategy"""
        options = ["none", "earlier", "later", "range"]
        current = self.state.env_contents.get("timestep_bias_strategy", "none")

        default_idx = 0
        if current in options:
            default_idx = options.index(current)

        selected = self.show_options(
            stdscr,
            f"Timestep bias strategy\n(Current: {current})\n\n"
            "earlier: Focus on earlier timesteps (high noise)\n"
            "later: Focus on later timesteps (low noise)\n"
            "range: Focus on a specific range\n"
            "none: No bias",
            options,
            default_idx,
        )

        if selected >= 0:
            self.state.env_contents["timestep_bias_strategy"] = options[selected]

    def _configure_bias_multiplier(self, stdscr):
        """Configure timestep bias multiplier"""
        current = str(self.state.env_contents.get("timestep_bias_multiplier", 1.0))
        value = self.get_input(
            stdscr,
            f"Timestep bias multiplier (higher = stronger bias)\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["timestep_bias_multiplier"] = float(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_bias_begin(self, stdscr):
        """Configure timestep bias begin"""
        current = str(self.state.env_contents.get("timestep_bias_begin", 0))
        value = self.get_input(
            stdscr,
            f"Beginning timestep for range bias\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["timestep_bias_begin"] = int(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_bias_end(self, stdscr):
        """Configure timestep bias end"""
        current = str(self.state.env_contents.get("timestep_bias_end", 1000))
        value = self.get_input(
            stdscr,
            f"Ending timestep for range bias\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["timestep_bias_end"] = int(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def loss_configuration(self, stdscr):
        """Step 4: Loss Configuration"""
        nav = MenuNavigator(stdscr)

        while True:
            current_values = {
                "Loss Type": self.state.env_contents.get("loss_type", "l2"),
                "SNR Gamma": str(self.state.env_contents.get("snr_gamma", 5.0)),
                "Noise Offset": str(self.state.env_contents.get("noise_offset", 0.0)),
                "Noise Offset Probability": str(self.state.env_contents.get("noise_offset_probability", 0.25)),
                "Input Perturbation": str(self.state.env_contents.get("input_perturbation", 0.0)),
                "Masked Loss Probability": str(self.state.env_contents.get("masked_loss_probability", 0.0)),
            }

            menu_items = [
                ("Loss Type", self._configure_loss_type),
                ("SNR Gamma", self._configure_snr_gamma),
                ("Noise Offset", self._configure_noise_offset),
                ("Noise Offset Probability", self._configure_noise_offset_prob),
                ("Input Perturbation", self._configure_input_perturbation),
                ("Masked Loss Probability", self._configure_masked_loss_prob),
            ]

            # Add Huber-specific options if Huber loss is selected
            if self.state.env_contents.get("loss_type") == "huber":
                current_values["Huber Schedule"] = self.state.env_contents.get("huber_schedule", "snr")
                current_values["Huber C"] = str(self.state.env_contents.get("huber_c", 0.1))
                menu_items.extend(
                    [
                        ("Huber Schedule", self._configure_huber_schedule),
                        ("Huber C", self._configure_huber_c),
                    ]
                )

            # Add soft min SNR option if SNR gamma is set
            if self.state.env_contents.get("snr_gamma", 0) > 0:
                current_values["Use Soft Min SNR"] = (
                    "Yes" if self.state.env_contents.get("use_soft_min_snr", False) else "No"
                )
                menu_items.append(("Use Soft Min SNR", self._configure_soft_min_snr))

            # Add input perturbation steps if perturbation is enabled
            if self.state.env_contents.get("input_perturbation", 0) > 0:
                current_values["Perturbation Steps"] = str(self.state.env_contents.get("input_perturbation_steps", 0))
                menu_items.append(("Perturbation Steps", self._configure_perturbation_steps))

            selected = nav.show_menu("Loss Configuration", menu_items, current_values)

            if selected == -1:  # Quit
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif selected == -2:  # Back
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_loss_type(self, stdscr):
        """Configure loss type"""
        options = ["l2", "huber", "smooth_l1"]
        current = self.state.env_contents.get("loss_type", "l2")

        default_idx = 0
        if current in options:
            default_idx = options.index(current)

        selected = self.show_options(
            stdscr,
            f"Select loss function\n(Current: {current})\n\n"
            "l2: Standard loss, good for most cases\n"
            "huber: Less sensitive to outliers\n"
            "smooth_l1: Combination of L1 and L2",
            options,
            default_idx,
        )

        if selected >= 0:
            self.state.env_contents["loss_type"] = options[selected]

    def _configure_snr_gamma(self, stdscr):
        """Configure SNR gamma"""
        current = str(self.state.env_contents.get("snr_gamma", 5.0))
        value = self.get_input(
            stdscr,
            f"SNR weighting gamma (0 to disable, 5.0 recommended)\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["snr_gamma"] = float(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_noise_offset(self, stdscr):
        """Configure noise offset"""
        current = str(self.state.env_contents.get("noise_offset", 0.0))
        value = self.get_input(
            stdscr,
            f"Noise offset scale (0 to disable, 0.1 typical)\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["noise_offset"] = float(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_noise_offset_prob(self, stdscr):
        """Configure noise offset probability"""
        current = str(self.state.env_contents.get("noise_offset_probability", 0.25))
        value = self.get_input(
            stdscr,
            f"Noise offset probability (0.0-1.0)\n(Current: {current})",
            current,
        )
        try:
            prob = float(value)
            if 0.0 <= prob <= 1.0:
                self.state.env_contents["noise_offset_probability"] = prob
            else:
                self.show_error(stdscr, "Value must be between 0.0 and 1.0")
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_input_perturbation(self, stdscr):
        """Configure input perturbation"""
        current = str(self.state.env_contents.get("input_perturbation", 0.0))
        value = self.get_input(
            stdscr,
            f"Input perturbation (0 to disable, 0.1 suggested)\n" "Helps training converge faster\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["input_perturbation"] = float(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_masked_loss_prob(self, stdscr):
        """Configure masked loss probability"""
        current = str(self.state.env_contents.get("masked_loss_probability", 0.0))
        value = self.get_input(
            stdscr,
            f"Masked loss probability (0.0-1.0)\n(Current: {current})",
            current,
        )
        try:
            prob = float(value)
            if 0.0 <= prob <= 1.0:
                self.state.env_contents["masked_loss_probability"] = prob
            else:
                self.show_error(stdscr, "Value must be between 0.0 and 1.0")
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_huber_schedule(self, stdscr):
        """Configure Huber loss schedule"""
        options = ["snr", "exponential", "constant"]
        current = self.state.env_contents.get("huber_schedule", "snr")

        default_idx = 0
        if current in options:
            default_idx = options.index(current)

        selected = self.show_options(
            stdscr,
            f"Huber loss schedule\n(Current: {current})",
            options,
            default_idx,
        )

        if selected >= 0:
            self.state.env_contents["huber_schedule"] = options[selected]

    def _configure_huber_c(self, stdscr):
        """Configure Huber C value"""
        current = str(self.state.env_contents.get("huber_c", 0.1))
        value = self.get_input(
            stdscr,
            f"Huber C threshold (L2 to L1 transition)\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["huber_c"] = float(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_soft_min_snr(self, stdscr):
        """Configure soft min SNR"""
        current = self.state.env_contents.get("use_soft_min_snr", False)

        selected = self.show_options(
            stdscr,
            f"Use soft min SNR calculation?\n(Current: {'Yes' if current else 'No'})",
            ["No", "Yes"],
            1 if current else 0,
        )

        if selected == 1:
            self.state.env_contents["use_soft_min_snr"] = True
            # Also need sigma_data
            sigma_data = self.get_input(
                stdscr,
                "Enter sigma_data value for soft min SNR:",
                "1.0",
            )
            try:
                self.state.env_contents["soft_min_snr_sigma_data"] = float(sigma_data)
            except ValueError:
                self.state.env_contents["soft_min_snr_sigma_data"] = 1.0
        elif selected == 0:
            if "use_soft_min_snr" in self.state.env_contents:
                del self.state.env_contents["use_soft_min_snr"]
            if "soft_min_snr_sigma_data" in self.state.env_contents:
                del self.state.env_contents["soft_min_snr_sigma_data"]

    def _configure_perturbation_steps(self, stdscr):
        """Configure input perturbation steps"""
        current = str(self.state.env_contents.get("input_perturbation_steps", 0))
        value = self.get_input(
            stdscr,
            f"Apply perturbation for first N steps (0 for all)\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["input_perturbation_steps"] = int(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def hub_setup(self, stdscr):
        """Step 5: Hugging Face Hub Setup - Now with menu"""
        nav = MenuNavigator(stdscr)

        # Check login status
        stdscr.clear()
        stdscr.addstr(2, 2, "Checking Hugging Face Hub login...")
        stdscr.refresh()

        try:
            self.state.whoami = huggingface_hub.whoami()
        except:
            self.state.whoami = None

        if not self.state.whoami:
            login_idx = self.show_options(
                stdscr,
                "You are not logged into Hugging Face Hub. Would you like to login?",
                ["Yes", "No"],
                0,
            )

            if login_idx == 0:
                stdscr.clear()
                stdscr.addstr(2, 2, "Please login to Hugging Face Hub in your terminal...")
                stdscr.addstr(4, 2, "Press any key when done...")
                stdscr.refresh()
                stdscr.getch()

                try:
                    huggingface_hub.login()
                    self.state.whoami = huggingface_hub.whoami()
                except:
                    self.show_error(stdscr, "Failed to login to Hugging Face Hub")
                    return

        if not self.state.whoami:
            return

        while True:
            # Build current values
            current_values = {
                "Logged in as": self.state.whoami["name"],
                "Push to Hub": ("Yes" if self.state.env_contents.get("push_to_hub", "false") == "true" else "No"),
            }

            menu_items = [
                ("Push to Hub", self._configure_push_to_hub),
            ]

            if self.state.env_contents.get("push_to_hub", "false") == "true":
                current_values["Model ID"] = self.state.env_contents.get(
                    "hub_model_id", f"simpletuner-{self.state.model_type}"
                )
                current_values["Push Checkpoints"] = (
                    "Yes" if self.state.env_contents.get("push_checkpoints_to_hub", "false") == "true" else "No"
                )
                current_values["Safe for Work"] = (
                    "Yes" if self.state.env_contents.get("model_card_safe_for_work", "false") == "true" else "No"
                )

                menu_items.extend(
                    [
                        ("Model ID", self._configure_model_id),
                        ("Push Checkpoints", self._configure_push_checkpoints),
                        ("Safe for Work", self._configure_sfw),
                    ]
                )

            selected = nav.show_menu("Hugging Face Hub Configuration", menu_items, current_values)

            if selected == -1:  # Quit
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif selected == -2:  # Back
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_push_to_hub(self, stdscr):
        """Configure push to hub"""
        current_push = self.state.env_contents.get("push_to_hub", "false") == "true"

        push_idx = self.show_options(
            stdscr,
            f"Do you want to push your model to Hugging Face Hub when completed?\n(Current: {'Yes' if current_push else 'No'})",
            ["Yes", "No"],
            0 if current_push else 1,
        )

        if push_idx == 0:
            self.state.env_contents["push_to_hub"] = "true"
        else:
            # Remove push settings if not pushing
            for key in [
                "push_to_hub",
                "hub_model_id",
                "push_checkpoints_to_hub",
                "model_card_safe_for_work",
            ]:
                if key in self.state.env_contents:
                    del self.state.env_contents[key]

    def _configure_model_id(self, stdscr):
        """Configure model ID"""
        current_model_id = self.state.env_contents.get("hub_model_id", f"simpletuner-{self.state.model_type}")

        model_id = self.get_input(
            stdscr,
            f"Model name (will be accessible as https://huggingface.co/{self.state.whoami['name']}/...):\n(Current: {current_model_id})",
            current_model_id,
        )

        self.state.env_contents["hub_model_id"] = model_id

    def _configure_push_checkpoints(self, stdscr):
        """Configure push checkpoints"""
        current_push_ckpt = self.state.env_contents.get("push_checkpoints_to_hub", "false") == "true"

        push_checkpoints_idx = self.show_options(
            stdscr,
            f"Push intermediary checkpoints to Hub?\n(Current: {'Yes' if current_push_ckpt else 'No'})",
            ["Yes", "No"],
            0 if current_push_ckpt else 1,
        )

        if push_checkpoints_idx == 0:
            self.state.env_contents["push_checkpoints_to_hub"] = "true"
        elif "push_checkpoints_to_hub" in self.state.env_contents:
            del self.state.env_contents["push_checkpoints_to_hub"]

    def _configure_sfw(self, stdscr):
        """Configure SFW setting"""
        current_sfw = self.state.env_contents.get("model_card_safe_for_work", "false") == "true"

        safe_idx = self.show_options(
            stdscr,
            f"Is your model safe-for-work?\n(Current: {'Yes' if current_sfw else 'No'})",
            ["No", "Yes"],
            1 if current_sfw else 0,
        )

        if safe_idx == 1:
            self.state.env_contents["model_card_safe_for_work"] = "true"
        elif "model_card_safe_for_work" in self.state.env_contents:
            del self.state.env_contents["model_card_safe_for_work"]

    def model_selection(self, stdscr):
        """Step 6: Model Selection - Now with menu"""
        nav = MenuNavigator(stdscr)

        while True:
            # Get current values
            current_values = {
                "Model Family": self.state.env_contents.get("model_family", "Not selected"),
                "Model Name": self.state.env_contents.get("pretrained_model_name_or_path", "Not selected"),
            }

            menu_items = [
                ("Model Family", self._configure_model_family),
                ("Model Name", self._configure_model_name),
            ]

            # Add Flux-specific options if applicable
            if (
                self.state.env_contents.get("model_family") == "flux"
                and self.state.model_type == "lora"
                and not self.state.use_lycoris
            ):
                current_values["Flux LoRA Target"] = self.state.env_contents.get("flux_lora_target", "all")
                menu_items.append(("Flux LoRA Target", self._configure_flux_target))

            # Add prediction type if applicable
            model_families_with_prediction = ["sdxl", "sd2x", "sd1x"]
            if self.state.env_contents.get("model_family") in model_families_with_prediction:
                current_values["Prediction Type"] = self.state.env_contents.get("prediction_type", "epsilon")
                menu_items.append(("Prediction Type", self._configure_prediction_type))

            selected = nav.show_menu("Model Selection", menu_items, current_values)

            if selected == -1:  # Quit
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif selected == -2:  # Back
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_model_family(self, stdscr):
        """Configure model family"""
        model_type = self.state.model_type or "lora"
        available_models = model_classes["full"]

        current_family = self.state.env_contents.get("model_family", "")
        default_idx = 0
        if current_family in available_models:
            default_idx = available_models.index(current_family)

        model_idx = self.show_options(
            stdscr,
            "Which model family are you training?",
            available_models,
            default_idx,
        )

        if model_idx >= 0:
            model_class = available_models[model_idx]
            self.state.env_contents["model_family"] = model_class

    def _configure_model_name(self, stdscr):
        """Configure model name from HF Hub"""
        model_class = self.state.env_contents.get("model_family", "")
        if not model_class:
            self.show_error(stdscr, "Please select a model family first")
            return

        default_model = default_models.get(model_class, "")
        current_model = self.state.env_contents.get("pretrained_model_name_or_path", default_model)

        while True:
            model_name = self.get_input(stdscr, "Enter the model name from Hugging Face Hub:", current_model)

            stdscr.clear()
            stdscr.addstr(2, 2, f"Checking model: {model_name}...")
            stdscr.refresh()

            try:
                model_info = huggingface_hub.model_info(model_name)
                if hasattr(model_info, "id"):
                    break
            except:
                self.show_error(stdscr, f"Could not load model: {model_name}")
                continue

        self.state.env_contents["pretrained_model_name_or_path"] = model_name

    def _configure_flux_target(self, stdscr):
        """Configure Flux LoRA target layers"""
        flux_targets = [
            "mmdit",
            "context",
            "all",
            "all+ffs",
            "ai-toolkit",
            "tiny",
            "nano",
        ]

        current_target = self.state.env_contents.get("flux_lora_target", "all")
        default_idx = 2  # Default to "all"
        if current_target in flux_targets:
            default_idx = flux_targets.index(current_target)

        target_idx = self.show_options(stdscr, "Set Flux target layers:", flux_targets, default_idx)

        if target_idx >= 0:
            self.state.env_contents["flux_lora_target"] = flux_targets[target_idx]

    def _configure_prediction_type(self, stdscr):
        """Configure prediction type"""
        options = ["epsilon", "v_prediction", "sample"]
        current = self.state.env_contents.get("prediction_type", "epsilon")

        default_idx = 0
        if current in options:
            default_idx = options.index(current)

        selected = self.show_options(
            stdscr,
            f"Prediction type (for SDXL derivatives)\n(Current: {current})",
            options,
            default_idx,
        )

        if selected >= 0:
            self.state.env_contents["prediction_type"] = options[selected]

    def training_params(self, stdscr):
        """Step 7: Training Parameters - Now with menu"""
        nav = MenuNavigator(stdscr)

        while True:
            # Get current values
            current_values = {
                "Batch Size": str(self.state.env_contents.get("train_batch_size", 1)),
                "Gradient Checkpointing": (
                    "Enabled" if self.state.env_contents.get("gradient_checkpointing", "true") == "true" else "Disabled"
                ),
                "Caption Dropout": str(self.state.env_contents.get("caption_dropout_probability", 0.1)),
                "Resolution Type": self.state.env_contents.get("resolution_type", "pixel_area"),
                "Resolution": str(self.state.env_contents.get("resolution", "1024")),
            }

            menu_items = [
                ("Batch Size", self._configure_batch_size),
                ("Gradient Checkpointing", self._configure_gradient_checkpointing),
                ("Caption Dropout", self._configure_caption_dropout),
                ("Resolution Type", self._configure_resolution_type),
                ("Resolution", self._configure_resolution),
            ]

            # Add gradient checkpointing interval if applicable
            if self.state.env_contents.get("model_family") in [
                "sdxl",
                "flux",
                "sd3",
                "sana",
                "chroma",
            ]:
                gc_interval = self.state.env_contents.get("gradient_checkpointing_interval", 0)
                current_values["GC Interval"] = str(gc_interval) if gc_interval > 0 else "Disabled"
                menu_items.insert(2, ("GC Interval", self._configure_gc_interval))

            # Add gradient accumulation
            current_values["Gradient Accumulation"] = str(self.state.env_contents.get("gradient_accumulation_steps", 1))
            menu_items.append(("Gradient Accumulation", self._configure_gradient_accumulation))

            selected = nav.show_menu("Training Parameters", menu_items, current_values)

            if selected == -1:  # Quit
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif selected == -2:  # Back
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_batch_size(self, stdscr):
        """Configure batch size"""
        current = str(self.state.env_contents.get("train_batch_size", 1))
        batch_size = self.get_input(
            stdscr,
            "Set the training batch size.\n" "Larger values will require larger datasets, more VRAM, and slow things down.",
            current,
        )

        try:
            self.state.env_contents["train_batch_size"] = int(batch_size)
        except ValueError:
            self.state.env_contents["train_batch_size"] = 1

    def _configure_gradient_checkpointing(self, stdscr):
        """Configure gradient checkpointing"""
        current = self.state.env_contents.get("gradient_checkpointing", "true") == "true"

        gc_idx = self.show_options(
            stdscr,
            f"Enable gradient checkpointing? (Saves VRAM, slower training)\n(Current: {'Enabled' if current else 'Disabled'})",
            ["Enable", "Disable"],
            0 if current else 1,
        )

        if gc_idx == 0:
            self.state.env_contents["gradient_checkpointing"] = "true"
        else:
            self.state.env_contents["gradient_checkpointing"] = "false"

    def _configure_gc_interval(self, stdscr):
        """Configure gradient checkpointing interval"""
        current = str(self.state.env_contents.get("gradient_checkpointing_interval", 0))

        gc_interval = self.get_input(
            stdscr,
            "Would you like to configure a gradient checkpointing interval?\n"
            "A value larger than 1 will increase VRAM usage but speed up training\n"
            "by skipping checkpoint creation every Nth layer.\n"
            "A zero will disable this feature.",
            current,
        )

        try:
            interval = int(gc_interval)
            if interval > 1:
                self.state.env_contents["gradient_checkpointing_interval"] = interval
            elif "gradient_checkpointing_interval" in self.state.env_contents:
                del self.state.env_contents["gradient_checkpointing_interval"]
        except ValueError:
            self.show_message(
                stdscr,
                "Could not parse gradient checkpointing interval. Not enabling.",
            )

    def _configure_caption_dropout(self, stdscr):
        """Configure caption dropout"""
        default_dropout = "0.05" if any([self.state.use_lora, self.state.use_lycoris]) else "0.1"
        current = str(self.state.env_contents.get("caption_dropout_probability", default_dropout))

        caption_dropout = self.get_input(
            stdscr,
            "Set the caption dropout rate, or use 0.0 to disable it.\n"
            "Dropout might be a good idea to disable for Flux training,\n"
            "but experimentation is warranted.",
            current,
        )

        try:
            self.state.env_contents["caption_dropout_probability"] = float(caption_dropout)
        except ValueError:
            self.state.env_contents["caption_dropout_probability"] = float(default_dropout)

    def _configure_resolution_type(self, stdscr):
        """Configure resolution type"""
        res_type_options = [
            "'pixel' - will size images with the shorter edge",
            "'area' - will measure in megapixels, great for aspect-bucketing",
            "'pixel_area' - combination that lets you set area using pixels",
        ]

        current_type = self.state.env_contents.get("resolution_type", "pixel_area")
        res_types = ["pixel", "area", "pixel_area"]
        default_idx = 2
        if current_type in res_types:
            default_idx = res_types.index(current_type)

        res_idx = self.show_options(
            stdscr,
            "How do you want to measure dataset resolutions?",
            res_type_options,
            default_idx,
        )

        if res_idx >= 0:
            self.state.env_contents["resolution_type"] = res_types[res_idx]

    def _configure_resolution(self, stdscr):
        """Configure resolution"""
        res_type = self.state.env_contents.get("resolution_type", "pixel_area")

        if res_type in ["pixel", "pixel_area"]:
            default_res = "1024"
            resolution_unit = "pixel"
        else:
            default_res = "1.0"
            resolution_unit = "megapixel"

        current = str(self.state.env_contents.get("resolution", default_res))

        resolution = self.get_input(
            stdscr,
            f"What would you like the default resolution of your datasets to be?\n"
            f"The default for {res_type} is {default_res} {resolution_unit}s.",
            current,
        )

        self.state.env_contents["resolution"] = resolution

    def _configure_gradient_accumulation(self, stdscr):
        """Configure gradient accumulation steps"""
        current = str(self.state.env_contents.get("gradient_accumulation_steps", 1))
        value = self.get_input(
            stdscr,
            f"Gradient accumulation steps (1 = disabled)\n" "Simulates larger batch sizes\n(Current: {current})",
            current,
        )
        try:
            steps = int(value)
            if steps > 1:
                self.state.env_contents["gradient_accumulation_steps"] = steps
            elif "gradient_accumulation_steps" in self.state.env_contents:
                del self.state.env_contents["gradient_accumulation_steps"]
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def optimization_settings(self, stdscr):
        """Step 8: Optimization Settings - Now with menu"""
        nav = MenuNavigator(stdscr)

        while True:
            # Get current values
            current_values = {
                "Mixed Precision": self.state.env_contents.get("mixed_precision", "bf16"),
                "Optimizer": self.state.env_contents.get("optimizer", "adamw_bf16"),
                "Learning Rate": self.state.env_contents.get("learning_rate", "1e-4"),
                "LR Scheduler": self.state.env_contents.get("lr_scheduler", "polynomial"),
                "Warmup Steps": str(self.state.env_contents.get("lr_warmup_steps", 0)),
                "Gradient Precision": self.state.env_contents.get("gradient_precision", "unmodified"),
                "Max Grad Norm": str(self.state.env_contents.get("max_grad_norm", 1.0)),
            }

            menu_items = [
                ("Mixed Precision", self._configure_mixed_precision),
                ("Optimizer", self._configure_optimizer),
                ("Learning Rate", self._configure_learning_rate),
                ("LR Scheduler", self._configure_lr_scheduler),
                ("Warmup Steps", self._configure_warmup_steps),
                ("Gradient Precision", self._configure_gradient_precision),
                ("Max Grad Norm", self._configure_max_grad_norm),
            ]

            # Add TF32 option if CUDA available
            if torch.cuda.is_available():
                tf32_disabled = "disable_tf32" in self.state.env_contents
                current_values["TF32"] = "Disabled" if tf32_disabled else "Enabled"
                menu_items.insert(0, ("TF32", self._configure_tf32))

            # Add quantization option
            if "base_model_precision" in self.state.env_contents:
                current_values["Quantization"] = self.state.env_contents["base_model_precision"]
            else:
                current_values["Quantization"] = "Disabled"
            menu_items.append(("Quantization", self._configure_quantization))

            # Add text encoder precision options if quantization is enabled
            if self.state.env_contents.get("base_model_precision", "no_change") != "no_change":
                for i in range(1, 5):
                    key = f"text_encoder_{i}_precision"
                    if key in self.state.env_contents:
                        current_values[f"Text Encoder {i}"] = self.state.env_contents[key]
                    menu_items.append(
                        (
                            f"Text Encoder {i} Precision",
                            lambda s, idx=i: self._configure_text_encoder_precision(s, idx),
                        )
                    )

            selected = nav.show_menu("Optimization Settings", menu_items, current_values)

            if selected == -1:  # Quit
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif selected == -2:  # Back
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_tf32(self, stdscr):
        """Configure TF32"""
        current = "disable_tf32" in self.state.env_contents

        tf32_idx = self.show_options(
            stdscr,
            f"Would you like to enable TF32 mode?\n(Current: {'Disabled' if current else 'Enabled'})",
            ["Yes", "No"],
            1 if current else 0,
        )

        if tf32_idx == 1:
            self.state.env_contents["disable_tf32"] = "true"
        elif "disable_tf32" in self.state.env_contents:
            del self.state.env_contents["disable_tf32"]

    def _configure_mixed_precision(self, stdscr):
        """Configure mixed precision"""
        current = self.state.env_contents.get("mixed_precision", "bf16")
        precision_options = ["bf16", "fp8", "no (fp32)"]
        precision_values = ["bf16", "fp8", "no"]

        default_idx = 0
        if current in precision_values:
            default_idx = precision_values.index(current)

        mixed_precision_idx = self.show_options(
            stdscr,
            f"Set mixed precision mode:\n(Current: {current})",
            precision_options,
            default_idx,
        )

        if mixed_precision_idx >= 0:
            self.state.env_contents["mixed_precision"] = precision_values[mixed_precision_idx]

    def _configure_optimizer(self, stdscr):
        """Configure optimizer"""
        # Get compatible optimizers based on precision
        if self.state.env_contents.get("mixed_precision") == "bf16":
            compatible_optims = bf16_only_optims + any_precision_optims
        else:
            compatible_optims = any_precision_optims

        current_optimizer = self.state.env_contents.get("optimizer", compatible_optims[0])
        default_optim_idx = 0
        if current_optimizer in compatible_optims:
            default_optim_idx = compatible_optims.index(current_optimizer)

        optim_idx = self.show_options(
            stdscr,
            f"Choose an optimizer:\n(Current: {current_optimizer})",
            compatible_optims,
            default_optim_idx,
        )

        if optim_idx >= 0:
            self.state.env_contents["optimizer"] = compatible_optims[optim_idx]

    def _configure_learning_rate(self, stdscr):
        """Configure learning rate"""
        # Dynamic defaults based on configuration
        default_lr = "1e-6"

        if self.state.model_type == "lora" and hasattr(self.state, "lora_rank"):
            if self.state.lora_rank in learning_rates_by_rank:
                default_lr = learning_rates_by_rank[self.state.lora_rank]
            else:
                default_lr = "1e-4"
        elif self.state.env_contents.get("optimizer") == "prodigy":
            default_lr = "1.0"

        current_lr = self.state.env_contents.get("learning_rate", default_lr)

        lr = self.get_input(
            stdscr,
            f"Set the learning rate:\n" f"(Current: {current_lr}, Suggested for your config: {default_lr})",
            current_lr,
        )

        self.state.env_contents["learning_rate"] = lr

    def _configure_lr_scheduler(self, stdscr):
        """Configure learning rate scheduler"""
        lr_schedulers = [
            "polynomial",
            "constant",
            "cosine",
            "cosine_with_restarts",
            "linear",
            "sine",
        ]
        current_scheduler = self.state.env_contents.get("lr_scheduler", "polynomial")
        default_sched_idx = 0
        if current_scheduler in lr_schedulers:
            default_sched_idx = lr_schedulers.index(current_scheduler)

        lr_idx = self.show_options(
            stdscr,
            f"Set learning rate scheduler:\n(Current: {current_scheduler})",
            lr_schedulers,
            default_sched_idx,
        )

        if lr_idx >= 0:
            lr_scheduler = lr_schedulers[lr_idx]
            self.state.env_contents["lr_scheduler"] = lr_scheduler

            # Handle polynomial scheduler extra args
            if lr_scheduler == "polynomial":
                if "lr_end=1e-8" not in self.state.extra_args:
                    self.state.extra_args.append("lr_end=1e-8")
            else:
                # Remove lr_end if switching away from polynomial
                self.state.extra_args = [arg for arg in self.state.extra_args if not arg.startswith("lr_end")]

    def _configure_warmup_steps(self, stdscr):
        """Configure warmup steps"""
        # Dynamic default
        default_warmup = "100"
        if self.state.env_contents.get("max_train_steps", 0) > 0:
            calculated_warmup = max(100, int(self.state.env_contents["max_train_steps"]) // 10)
            default_warmup = str(calculated_warmup)

        current_warmup = self.state.env_contents.get("lr_warmup_steps", default_warmup)

        warmup = self.get_input(
            stdscr,
            f"Set the number of warmup steps before the learning rate reaches its peak.\n"
            f"This is set to 10 percent of the total runtime by default, or 100 steps, whichever is higher.\n"
            f"(Current: {current_warmup})",
            current_warmup,
        )

        try:
            self.state.env_contents["lr_warmup_steps"] = int(warmup)
        except ValueError:
            self.state.env_contents["lr_warmup_steps"] = int(default_warmup)

    def _configure_gradient_precision(self, stdscr):
        """Configure gradient precision"""
        options = ["unmodified", "fp32"]
        current = self.state.env_contents.get("gradient_precision", "unmodified")

        default_idx = 0
        if current in options:
            default_idx = options.index(current)

        selected = self.show_options(
            stdscr,
            f"Gradient precision\n(Current: {current})\n\n" "fp32 is slower but more accurate for gradient accumulation",
            options,
            default_idx,
        )

        if selected >= 0:
            self.state.env_contents["gradient_precision"] = options[selected]

    def _configure_max_grad_norm(self, stdscr):
        """Configure max gradient norm"""
        current = str(self.state.env_contents.get("max_grad_norm", 1.0))
        value = self.get_input(
            stdscr,
            f"Max gradient norm (0 to disable clipping)\n(Current: {current})",
            current,
        )
        try:
            norm = float(value)
            if norm > 0:
                self.state.env_contents["max_grad_norm"] = norm
            elif "max_grad_norm" in self.state.env_contents:
                del self.state.env_contents["max_grad_norm"]
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_quantization(self, stdscr):
        """Configure quantization"""
        quant_warning = ""
        if self.state.use_lora:
            quant_warning = "\n\nNOTE: Currently, a bug prevents multi-GPU training with LoRA quantization."

        current_quantization = "base_model_precision" in self.state.env_contents

        quant_idx = self.show_options(
            stdscr,
            f"Would you like to enable model quantization?{quant_warning}\n"
            f"(Current: {'Enabled' if current_quantization else 'Disabled'})",
            ["Yes", "No"],
            0 if not current_quantization else 1,
        )

        if quant_idx == 0:
            # Handle DoRA disabling
            if self.state.env_contents.get("use_dora") == "true":
                stdscr.clear()
                stdscr.addstr(2, 2, "Note: DoRA will be disabled for quantisation.", curses.A_BOLD)
                stdscr.addstr(4, 2, "Press any key to continue...")
                stdscr.refresh()
                stdscr.getch()
                del self.state.env_contents["use_dora"]

            # Get quantization type
            current_quant_type = self.state.env_contents.get("base_model_precision", "int8-quanto")
            quant_types = list(quantised_precision_levels)
            default_quant_idx = 0

            if current_quant_type in quant_types:
                default_quant_idx = quant_types.index(current_quant_type)

            quant_type_idx = self.show_options(
                stdscr,
                f"Choose quantization type:\n(Current: {current_quant_type})",
                quant_types,
                default_quant_idx,
            )

            if quant_type_idx >= 0:
                self.state.env_contents["base_model_precision"] = quant_types[quant_type_idx]
        else:
            # Remove quantization if disabled
            if "base_model_precision" in self.state.env_contents:
                del self.state.env_contents["base_model_precision"]

    def _configure_text_encoder_precision(self, stdscr, encoder_num):
        """Configure text encoder precision"""
        key = f"text_encoder_{encoder_num}_precision"
        current = self.state.env_contents.get(key, "no_change")

        quant_types = ["no_change"] + list(quantised_precision_levels)
        default_idx = 0
        if current in quant_types:
            default_idx = quant_types.index(current)

        selected = self.show_options(
            stdscr,
            f"Text Encoder {encoder_num} precision\n(Current: {current})",
            quant_types,
            default_idx,
        )

        if selected >= 0:
            if quant_types[selected] == "no_change":
                if key in self.state.env_contents:
                    del self.state.env_contents[key]
            else:
                self.state.env_contents[key] = quant_types[selected]

    def vae_configuration(self, stdscr):
        """Step 9: VAE Configuration"""
        nav = MenuNavigator(stdscr)

        while True:
            current_values = {
                "VAE Dtype": self.state.env_contents.get("vae_dtype", "default"),
                "VAE Batch Size": str(self.state.env_contents.get("vae_batch_size", 4)),
                "VAE Tiling": ("Enabled" if self.state.env_contents.get("vae_enable_tiling", False) else "Disabled"),
                "VAE Slicing": ("Enabled" if self.state.env_contents.get("vae_enable_slicing", False) else "Disabled"),
                "Keep VAE Loaded": ("Yes" if self.state.env_contents.get("keep_vae_loaded", False) else "No"),
                "Cache Scan Behaviour": self.state.env_contents.get("vae_cache_scan_behaviour", "recreate"),
            }

            menu_items = [
                ("VAE Dtype", self._configure_vae_dtype),
                ("VAE Batch Size", self._configure_vae_batch_size),
                ("VAE Tiling", self._configure_vae_tiling),
                ("VAE Slicing", self._configure_vae_slicing),
                ("Keep VAE Loaded", self._configure_keep_vae_loaded),
                ("Cache Scan Behaviour", self._configure_cache_scan_behaviour),
            ]

            selected = nav.show_menu("VAE Configuration", menu_items, current_values)

            if selected == -1:  # Quit
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif selected == -2:  # Back
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_vae_dtype(self, stdscr):
        """Configure VAE dtype"""
        options = ["default", "fp16", "fp32", "bf16"]
        current = self.state.env_contents.get("vae_dtype", "default")

        default_idx = 0
        if current in options:
            default_idx = options.index(current)

        selected = self.show_options(
            stdscr,
            f"VAE dtype\n(Current: {current})\n\n" "bf16 is default for SDXL due to NaN issues\n" "fp16 is not recommended",
            options,
            default_idx,
        )

        if selected >= 0:
            self.state.env_contents["vae_dtype"] = options[selected]

    def _configure_vae_batch_size(self, stdscr):
        """Configure VAE batch size"""
        current = str(self.state.env_contents.get("vae_batch_size", 4))
        value = self.get_input(
            stdscr,
            f"VAE batch size for pre-caching\n" "Lower values help with VRAM issues\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["vae_batch_size"] = int(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_vae_tiling(self, stdscr):
        """Configure VAE tiling"""
        current = self.state.env_contents.get("vae_enable_tiling", False)

        selected = self.show_options(
            stdscr,
            f"Enable VAE tiling? (For very large images)\n(Current: {'Yes' if current else 'No'})",
            ["No", "Yes"],
            1 if current else 0,
        )

        if selected == 1:
            self.state.env_contents["vae_enable_tiling"] = True
        elif selected == 0 and "vae_enable_tiling" in self.state.env_contents:
            del self.state.env_contents["vae_enable_tiling"]

    def _configure_vae_slicing(self, stdscr):
        """Configure VAE slicing"""
        current = self.state.env_contents.get("vae_enable_slicing", False)

        selected = self.show_options(
            stdscr,
            f"Enable VAE slicing? (For video models)\n(Current: {'Yes' if current else 'No'})",
            ["No", "Yes"],
            1 if current else 0,
        )

        if selected == 1:
            self.state.env_contents["vae_enable_slicing"] = True
        elif selected == 0 and "vae_enable_slicing" in self.state.env_contents:
            del self.state.env_contents["vae_enable_slicing"]

    def _configure_keep_vae_loaded(self, stdscr):
        """Configure keep VAE loaded"""
        current = self.state.env_contents.get("keep_vae_loaded", False)

        selected = self.show_options(
            stdscr,
            f"Keep VAE loaded in memory?\n" "Reduces disk churn but uses VRAM\n(Current: {'Yes' if current else 'No'})",
            ["No", "Yes"],
            1 if current else 0,
        )

        if selected == 1:
            self.state.env_contents["keep_vae_loaded"] = True
        elif selected == 0 and "keep_vae_loaded" in self.state.env_contents:
            del self.state.env_contents["keep_vae_loaded"]

    def _configure_cache_scan_behaviour(self, stdscr):
        """Configure cache scan behaviour"""
        options = ["recreate", "sync"]
        current = self.state.env_contents.get("vae_cache_scan_behaviour", "recreate")

        default_idx = 0
        if current in options:
            default_idx = options.index(current)

        selected = self.show_options(
            stdscr,
            f"Cache scan behaviour\n(Current: {current})\n\n"
            "recreate: Delete and rebuild inconsistent entries\n"
            "sync: Update bucket metadata to match reality",
            options,
            default_idx,
        )

        if selected >= 0:
            self.state.env_contents["vae_cache_scan_behaviour"] = options[selected]

    def flow_matching_config(self, stdscr):
        """Step 10: Flow Matching Configuration (for Flux/SD3)"""
        # Check if applicable model
        if self.state.env_contents.get("model_family") not in ["flux", "sd3", "sana"]:
            self.show_message(
                stdscr,
                "Flow matching configuration is only applicable for Flux, SD3, and Sana models.",
            )
            return

        nav = MenuNavigator(stdscr)

        while True:
            current_values = {
                "Sigmoid Scale": str(self.state.env_contents.get("flow_sigmoid_scale", 1.0)),
                "Schedule Type": self._get_flow_schedule_type(),
                "Schedule Shift": str(self.state.env_contents.get("flow_schedule_shift", 3.0)),
                "Auto Shift": ("Enabled" if self.state.env_contents.get("flow_schedule_auto_shift", False) else "Disabled"),
            }

            menu_items = [
                ("Sigmoid Scale", self._configure_sigmoid_scale),
                ("Schedule Type", self._configure_flow_schedule),
                ("Schedule Shift", self._configure_schedule_shift),
                ("Auto Shift", self._configure_auto_shift),
            ]

            # Add Flux-specific options
            if self.state.env_contents.get("model_family") == "flux":
                current_values["Flux Fast Schedule"] = (
                    "Enabled" if self.state.env_contents.get("flux_fast_schedule", False) else "Disabled"
                )
                current_values["Guidance Mode"] = self.state.env_contents.get("flux_guidance_mode", "constant")
                current_values["Attention Masking"] = (
                    "Enabled" if self.state.env_contents.get("flux_attention_masked_training", False) else "Disabled"
                )

                menu_items.extend(
                    [
                        ("Flux Fast Schedule", self._configure_flux_fast_schedule),
                        ("Guidance Mode", self._configure_flux_guidance_mode),
                        ("Attention Masking", self._configure_flux_attention_masking),
                    ]
                )

                # Add guidance value options based on mode
                if self.state.env_contents.get("flux_guidance_mode") == "constant":
                    current_values["Guidance Value"] = str(self.state.env_contents.get("flux_guidance_value", 1.0))
                    menu_items.append(("Guidance Value", self._configure_flux_guidance_value))
                else:
                    current_values["Guidance Min"] = str(self.state.env_contents.get("flux_guidance_min", 1.0))
                    current_values["Guidance Max"] = str(self.state.env_contents.get("flux_guidance_max", 4.0))
                    menu_items.extend(
                        [
                            ("Guidance Min", self._configure_flux_guidance_min),
                            ("Guidance Max", self._configure_flux_guidance_max),
                        ]
                    )

            # Add beta schedule options if using beta
            if self.state.env_contents.get("flow_use_beta_schedule", False):
                current_values["Beta Alpha"] = str(self.state.env_contents.get("flow_beta_schedule_alpha", 2.0))
                current_values["Beta Beta"] = str(self.state.env_contents.get("flow_beta_schedule_beta", 2.0))
                menu_items.extend(
                    [
                        ("Beta Alpha", self._configure_beta_alpha),
                        ("Beta Beta", self._configure_beta_beta),
                    ]
                )

            selected = nav.show_menu("Flow Matching Configuration", menu_items, current_values)

            if selected == -1:  # Quit
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif selected == -2:  # Back
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _get_flow_schedule_type(self):
        """Get current flow schedule type"""
        if self.state.env_contents.get("flow_use_uniform_schedule", False):
            return "uniform"
        elif self.state.env_contents.get("flow_use_beta_schedule", False):
            return "beta"
        else:
            return "sigmoid"

    def _configure_sigmoid_scale(self, stdscr):
        """Configure sigmoid scale"""
        current = str(self.state.env_contents.get("flow_sigmoid_scale", 1.0))
        value = self.get_input(
            stdscr,
            f"Sigmoid scale factor for flow-matching\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["flow_sigmoid_scale"] = float(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_flow_schedule(self, stdscr):
        """Configure flow schedule type"""
        options = ["sigmoid", "uniform", "beta"]
        current_type = self._get_flow_schedule_type()

        default_idx = 0
        if current_type in options:
            default_idx = options.index(current_type)

        selected = self.show_options(
            stdscr,
            f"Flow-matching noise schedule\n(Current: {current_type})\n\n"
            "sigmoid: Default schedule\n"
            "uniform: May cause bias toward dark images\n"
            "beta: Approximates sigmoid with customizable shape",
            options,
            default_idx,
        )

        if selected >= 0:
            # Reset all schedule flags
            for key in ["flow_use_uniform_schedule", "flow_use_beta_schedule"]:
                if key in self.state.env_contents:
                    del self.state.env_contents[key]

            if options[selected] == "uniform":
                self.state.env_contents["flow_use_uniform_schedule"] = True
            elif options[selected] == "beta":
                self.state.env_contents["flow_use_beta_schedule"] = True

    def _configure_schedule_shift(self, stdscr):
        """Configure schedule shift"""
        current = str(self.state.env_contents.get("flow_schedule_shift", 3.0))
        value = self.get_input(
            stdscr,
            f"Schedule shift (0-4.0)\n"
            "Higher values focus on large compositional features\n"
            "Lower values focus on fine details\n"
            "Sana and SD3 were trained with 3.0\n(Current: {current})",
            current,
        )
        try:
            shift = float(value)
            if shift > 0:
                self.state.env_contents["flow_schedule_shift"] = shift
            elif "flow_schedule_shift" in self.state.env_contents:
                del self.state.env_contents["flow_schedule_shift"]
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_auto_shift(self, stdscr):
        """Configure auto shift"""
        current = self.state.env_contents.get("flow_schedule_auto_shift", False)

        selected = self.show_options(
            stdscr,
            f"Auto-shift based on resolution?\n"
            "Shift value grows exponentially with pixel count\n"
            "May need lower learning rate\n(Current: {'Yes' if current else 'No'})",
            ["No", "Yes"],
            1 if current else 0,
        )

        if selected == 1:
            self.state.env_contents["flow_schedule_auto_shift"] = True
        elif selected == 0 and "flow_schedule_auto_shift" in self.state.env_contents:
            del self.state.env_contents["flow_schedule_auto_shift"]

    def _configure_flux_fast_schedule(self, stdscr):
        """Configure Flux fast schedule"""
        current = self.state.env_contents.get("flux_fast_schedule", False)

        selected = self.show_options(
            stdscr,
            f"Use Flux.1S fast schedule? (Experimental)\n(Current: {'Yes' if current else 'No'})",
            ["No", "Yes"],
            1 if current else 0,
        )

        if selected == 1:
            self.state.env_contents["flux_fast_schedule"] = True
        elif selected == 0 and "flux_fast_schedule" in self.state.env_contents:
            del self.state.env_contents["flux_fast_schedule"]

    def _configure_flux_guidance_mode(self, stdscr):
        """Configure Flux guidance mode"""
        options = ["constant", "random-range"]
        current = self.state.env_contents.get("flux_guidance_mode", "constant")

        default_idx = 0
        if current in options:
            default_idx = options.index(current)

        selected = self.show_options(
            stdscr,
            f"Flux guidance mode\n(Current: {current})\n\n"
            "constant: Single value for all samples\n"
            "random-range: Random value per sample",
            options,
            default_idx,
        )

        if selected >= 0:
            self.state.env_contents["flux_guidance_mode"] = options[selected]

    def _configure_flux_guidance_value(self, stdscr):
        """Configure Flux guidance value"""
        current = str(self.state.env_contents.get("flux_guidance_value", 1.0))
        value = self.get_input(
            stdscr,
            f"Flux guidance value\n" "1.0 preserves CFG distillation for Dev model\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["flux_guidance_value"] = float(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_flux_guidance_min(self, stdscr):
        """Configure Flux guidance min"""
        current = str(self.state.env_contents.get("flux_guidance_min", 1.0))
        value = self.get_input(
            stdscr,
            f"Minimum guidance value for random range\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["flux_guidance_min"] = float(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_flux_guidance_max(self, stdscr):
        """Configure Flux guidance max"""
        current = str(self.state.env_contents.get("flux_guidance_max", 4.0))
        value = self.get_input(
            stdscr,
            f"Maximum guidance value for random range\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["flux_guidance_max"] = float(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_flux_attention_masking(self, stdscr):
        """Configure Flux attention masking"""
        current = self.state.env_contents.get("flux_attention_masked_training", False)

        selected = self.show_options(
            stdscr,
            f"Use attention masking? (Can be destructive)\n(Current: {'Yes' if current else 'No'})",
            ["No", "Yes"],
            1 if current else 0,
        )
        if selected == 1:
            self.state.env_contents["flux_attention_masked_training"] = True
        elif selected == 0 and "flux_attention_masked_training" in self.state.env_contents:
            del self.state.env_contents["flux_attention_masked_training"]

    def _configure_beta_alpha(self, stdscr):
        """Configure beta schedule alpha"""
        current = str(self.state.env_contents.get("flow_beta_schedule_alpha", 2.0))
        value = self.get_input(
            stdscr,
            f"Beta schedule alpha value\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["flow_beta_schedule_alpha"] = float(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_beta_beta(self, stdscr):
        """Configure beta schedule beta"""
        current = str(self.state.env_contents.get("flow_beta_schedule_beta", 2.0))
        value = self.get_input(
            stdscr,
            f"Beta schedule beta value\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["flow_beta_schedule_beta"] = float(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def validation_settings(self, stdscr):
        """Step 11: Validation Settings - Now with menu"""
        nav = MenuNavigator(stdscr)

        while True:
            # Get current values
            current_values = {
                "Validation Seed": self.state.env_contents.get("validation_seed", "42"),
                "Validation Steps": self.state.env_contents.get(
                    "validation_steps",
                    str(self.state.env_contents.get("checkpointing_steps", 500)),
                ),
                "Validation Resolution": self.state.env_contents.get("validation_resolution", "1024x1024"),
                "Guidance Scale": self.state.env_contents.get("validation_guidance", "3.0"),
                "Guidance Rescale": self.state.env_contents.get("validation_guidance_rescale", "0.0"),
                "Inference Steps": self.state.env_contents.get("validation_num_inference_steps", "20"),
                "Validation Prompt": self.state.env_contents.get("validation_prompt", "A photo-realistic image of a cat")[
                    :40
                ]
                + "...",
                "Evaluation Type": self.state.env_contents.get("evaluation_type", "none"),
            }

            menu_items = [
                ("Validation Seed", self._configure_val_seed),
                ("Validation Steps", self._configure_val_steps),
                ("Validation Resolution", self._configure_val_resolution),
                ("Guidance Scale", self._configure_val_guidance),
                ("Guidance Rescale", self._configure_val_rescale),
                ("Inference Steps", self._configure_val_inference_steps),
                ("Validation Prompt", self._configure_val_prompt),
                ("Evaluation Type", self._configure_evaluation_type),
            ]

            # Add noise scheduler option
            current_values["Noise Scheduler"] = self.state.env_contents.get("validation_noise_scheduler", "default")
            menu_items.append(("Noise Scheduler", self._configure_val_noise_scheduler))

            # Add torch compile option
            current_values["Torch Compile"] = (
                "Enabled" if self.state.env_contents.get("validation_torch_compile", False) else "Disabled"
            )
            menu_items.append(("Torch Compile", self._configure_val_torch_compile))

            selected = nav.show_menu("Validation Settings", menu_items, current_values)

            if selected == -1:  # Quit
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif selected == -2:  # Back
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_val_seed(self, stdscr):
        """Configure validation seed"""
        current = self.state.env_contents.get("validation_seed", "42")
        val_seed = self.get_input(stdscr, "Set the seed for validation:", current)
        self.state.env_contents["validation_seed"] = val_seed

    def _configure_val_steps(self, stdscr):
        """Configure validation steps"""
        default_val_steps = str(self.state.env_contents.get("checkpointing_steps", 500))
        current = self.state.env_contents.get("validation_steps", default_val_steps)
        val_steps = self.get_input(stdscr, "How many steps between validation outputs?", current)
        self.state.env_contents["validation_steps"] = val_steps

    def _configure_val_resolution(self, stdscr):
        """Configure validation resolution"""
        current = self.state.env_contents.get("validation_resolution", "1024x1024")
        val_res = self.get_input(
            stdscr,
            "Set validation resolution (e.g., 1024x1024 or comma-separated list):",
            current,
        )
        # Clean up resolution
        val_res = ",".join([x.strip() for x in val_res.split(",")])
        self.state.env_contents["validation_resolution"] = val_res

    def _configure_val_guidance(self, stdscr):
        """Configure validation guidance"""
        model_family = self.state.env_contents.get("model_family", "flux")
        default_cfg_val = str(default_cfg.get(model_family, 3.0))
        current = self.state.env_contents.get("validation_guidance", default_cfg_val)

        val_guidance = self.get_input(stdscr, "Set guidance scale for validation:", current)
        self.state.env_contents["validation_guidance"] = val_guidance

    def _configure_val_rescale(self, stdscr):
        """Configure validation guidance rescale"""
        current = self.state.env_contents.get("validation_guidance_rescale", "0.0")
        val_rescale = self.get_input(
            stdscr,
            "Set guidance rescale (dynamic thresholding, 0.0 to disable):",
            current,
        )
        self.state.env_contents["validation_guidance_rescale"] = val_rescale

    def _configure_val_inference_steps(self, stdscr):
        """Configure validation inference steps"""
        current = self.state.env_contents.get("validation_num_inference_steps", "20")
        val_inf_steps = self.get_input(stdscr, "Set number of inference steps for validation:", current)
        self.state.env_contents["validation_num_inference_steps"] = val_inf_steps

    def _configure_val_prompt(self, stdscr):
        """Configure validation prompt"""
        current = self.state.env_contents.get("validation_prompt", "A photo-realistic image of a cat")
        val_prompt = self.get_input(stdscr, "Set the validation prompt:", current)
        self.state.env_contents["validation_prompt"] = val_prompt

    def _configure_evaluation_type(self, stdscr):
        """Configure evaluation type"""
        options = ["none", "clip"]
        current = self.state.env_contents.get("evaluation_type", "none")

        default_idx = 0
        if current in options:
            default_idx = options.index(current)

        selected = self.show_options(
            stdscr,
            f"Enable CLIP evaluation?\n(Current: {current})\n\n" "CLIP scores measure prompt adherence",
            options,
            default_idx,
        )

        if selected >= 0:
            self.state.env_contents["evaluation_type"] = options[selected]

    def _configure_val_noise_scheduler(self, stdscr):
        """Configure validation noise scheduler"""
        options = ["default", "ddim", "ddpm", "euler", "euler-a", "unipc", "dpm++"]
        current = self.state.env_contents.get("validation_noise_scheduler", "default")

        # Find default index
        default_idx = 0
        for idx, opt in enumerate(options):
            if opt == current or (current is None and opt == "default"):
                default_idx = idx
                break

        selected = self.show_options(
            stdscr,
            f"Validation noise scheduler\n(Current: {current})",
            options,
            default_idx,
        )

        if selected >= 0:
            if options[selected] == "default":
                if "validation_noise_scheduler" in self.state.env_contents:
                    del self.state.env_contents["validation_noise_scheduler"]
            else:
                self.state.env_contents["validation_noise_scheduler"] = options[selected]

    def _configure_val_torch_compile(self, stdscr):
        """Configure validation torch compile"""
        current = self.state.env_contents.get("validation_torch_compile", False)

        selected = self.show_options(
            stdscr,
            f"Enable torch.compile() for validation? (Can speed up inference)\n(Current: {'Yes' if current else 'No'})",
            ["No", "Yes"],
            1 if current else 0,
        )

        if selected == 1:
            self.state.env_contents["validation_torch_compile"] = True
        elif selected == 0 and "validation_torch_compile" in self.state.env_contents:
            del self.state.env_contents["validation_torch_compile"]

    def advanced_options(self, stdscr):
        """Step 12: Advanced Options - Now with menu"""
        nav = MenuNavigator(stdscr)

        while True:
            # Get current values
            current_values = {
                "Tracking": self.state.env_contents.get("report_to", "none"),
                "SageAttention": self.state.env_contents.get("attention_mechanism", "diffusers"),
                "Disk Cache Compression": ("Enabled" if "compress_disk_cache" in self.state.extra_args else "Disabled"),
                "Torch Compile": (
                    "Enabled" if self.state.env_contents.get("validation_torch_compile", "false") == "true" else "Disabled"
                ),
                "Prompt Library": ("Configured" if "user_prompt_library" in self.state.env_contents else "Not configured"),
                "Rescale Betas Zero SNR": (
                    "Enabled" if self.state.env_contents.get("rescale_betas_zero_snr", False) else "Disabled"
                ),
            }

            menu_items = [
                ("Tracking (W&B/TensorBoard)", self._configure_tracking),
                ("SageAttention", self._configure_sageattention),
                ("Disk Cache Compression", self._configure_disk_compression),
                ("Torch Compile", self._configure_torch_compile),
                ("Prompt Library", self._configure_prompt_library),
                ("Rescale Betas Zero SNR", self._configure_rescale_betas),
                ("Freeze Encoder Settings", self._configure_freeze_encoder),
                ("Distillation Settings", self._configure_distillation),
            ]

            selected = nav.show_menu("Advanced Options", menu_items, current_values)

            if selected == -1:  # Quit
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif selected == -2:  # Back
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_tracking(self, stdscr):
        """Configure experiment tracking"""
        # Sub-menu for tracking configuration
        nav = MenuNavigator(stdscr)

        while True:
            current_report_to = self.state.env_contents.get("report_to", "none")

            current_values = {
                "Report To": current_report_to,
            }

            menu_items = [
                ("Report To", self._configure_report_to),
            ]

            if current_report_to != "none":
                current_values["Project Name"] = self.state.env_contents.get(
                    "tracker_project_name", f"{self.state.model_type}-training"
                )
                current_values["Run Name"] = self.state.env_contents.get(
                    "tracker_run_name", f"simpletuner-{self.state.model_type}"
                )

                menu_items.extend(
                    [
                        ("Project Name", self._configure_project_name),
                        ("Run Name", self._configure_run_name),
                    ]
                )

            selected = nav.show_menu("Tracking Configuration", menu_items, current_values)

            if selected == -1:  # Quit
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif selected == -2:  # Back
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_report_to(self, stdscr):
        """Configure where to report tracking"""
        options = ["None", "Weights & Biases", "TensorBoard", "Both"]

        current = self.state.env_contents.get("report_to", "none")
        default_idx = 0
        if current == "wandb":
            default_idx = 1
        elif current == "tensorboard":
            default_idx = 2
        elif "wandb" in current and "tensorboard" in current:
            default_idx = 3

        selected = self.show_options(stdscr, "Select tracking services:", options, default_idx)

        if selected == 0:
            self.state.env_contents["report_to"] = "none"
        elif selected == 1:
            self.state.env_contents["report_to"] = "wandb"
        elif selected == 2:
            self.state.env_contents["report_to"] = "tensorboard"
        elif selected == 3:
            self.state.env_contents["report_to"] = "wandb,tensorboard"

    def _configure_project_name(self, stdscr):
        """Configure tracking project name"""
        current = self.state.env_contents.get("tracker_project_name", f"{self.state.model_type}-training")
        project_name = self.get_input(
            stdscr,
            "Enter the name of your Weights & Biases project:",
            current,
        )
        self.state.env_contents["tracker_project_name"] = project_name

    def _configure_run_name(self, stdscr):
        """Configure tracking run name"""
        current = self.state.env_contents.get("tracker_run_name", f"simpletuner-{self.state.model_type}")
        run_name = self.get_input(
            stdscr,
            "Enter the name of your Weights & Biases runs.\n"
            "This can use shell commands, which can be used to dynamically set the run name.",
            current,
        )
        self.state.env_contents["tracker_run_name"] = run_name

    def _configure_sageattention(self, stdscr):
        """Configure SageAttention"""
        current_mechanism = self.state.env_contents.get("attention_mechanism", "diffusers")

        sage_idx = self.show_options(
            stdscr,
            f"Would you like to use SageAttention for image validation generation?\n(Current: {current_mechanism})",
            ["No", "Yes"],
            1 if current_mechanism == "sageattention" else 0,
        )

        if sage_idx == 1:
            self.state.env_contents["attention_mechanism"] = "sageattention"

            # Configure usage scope
            current_usage = self.state.env_contents.get("sageattention_usage", "inference")

            # Show detailed warning for training usage
            stdscr.clear()
            h, w = stdscr.getmaxyx()

            warning_text = (
                "Would you like to use SageAttention to cover the forward and backward pass during training?\n\n"
                "WARNING: This has the undesirable consequence of leaving the attention layers untrained, "
                "as SageAttention lacks the capability to fully track gradients through quantisation.\n\n"
                "If you are not training the attention layers for some reason, this may not matter and "
                "you can safely enable this. For all other use-cases, reconsideration and caution are warranted."
            )

            # Wrap text for display
            wrapped_lines = []
            for paragraph in warning_text.split("\n"):
                if paragraph:
                    wrapped_lines.extend(textwrap.wrap(paragraph, w - 4))
                else:
                    wrapped_lines.append("")

            # Display warning
            y = 2
            for line in wrapped_lines:
                if y < h - 4:
                    stdscr.addstr(y, 2, line)
                    y += 1

            sage_training_idx = self.show_options(stdscr, "", ["No (Inference only)", "Yes (Training + Inference)"], 0)

            if sage_training_idx == 1:
                self.state.env_contents["sageattention_usage"] = "both"
            else:
                self.state.env_contents["sageattention_usage"] = "inference"
        else:
            self.state.env_contents["attention_mechanism"] = "diffusers"
            if "sageattention_usage" in self.state.env_contents:
                del self.state.env_contents["sageattention_usage"]

    def _configure_disk_compression(self, stdscr):
        """Configure disk cache compression"""
        current = "compress_disk_cache" in self.state.extra_args

        compress_idx = self.show_options(
            stdscr,
            f"Would you like to compress the disk cache?\n(Current: {'Yes' if current else 'No'})",
            ["Yes", "No"],
            0 if current else 1,
        )

        if compress_idx == 0:
            if "compress_disk_cache" not in self.state.extra_args:
                self.state.extra_args.append("compress_disk_cache")
        else:
            self.state.extra_args = [arg for arg in self.state.extra_args if arg != "compress_disk_cache"]

    def _configure_torch_compile(self, stdscr):
        """Configure torch compile"""
        current = self.state.env_contents.get("validation_torch_compile", "false") == "true"

        compile_idx = self.show_options(
            stdscr,
            f"Would you like to use torch compile during validations?\n(Current: {'Yes' if current else 'No'})",
            ["No", "Yes"],
            1 if current else 0,
        )

        self.state.env_contents["validation_torch_compile"] = "true" if compile_idx == 1 else "false"

    def _configure_prompt_library(self, stdscr):
        """Configure prompt library generation"""
        # Check if library exists
        should_generate_by_default = 1  # Default to "No"
        if not os.path.exists("config/user_prompt_library.json"):
            should_generate_by_default = 0  # Default to "Yes" if no library exists

        prompt_lib_idx = self.show_options(
            stdscr,
            "Would you like to generate a very rudimentary subject-centric prompt library for your dataset?\n"
            "This will download a small 1B Llama 3.2 model.\n"
            "If a user prompt library exists, it will be overwritten.",
            ["Yes", "No"],
            should_generate_by_default,
        )

        if prompt_lib_idx == 0:
            trigger = self.get_input(
                stdscr,
                "Enter a trigger word (or a few words) that you would like Llama 3.2 1B to expand:",
                "Character Name",
            )

            num_prompts = self.get_input(stdscr, "How many prompts would you like to generate?", "8")

            try:
                num_prompts_int = int(num_prompts)
            except ValueError:
                num_prompts_int = 8

            try:
                from simpletuner.helpers.prompt_expander import PromptExpander

                stdscr.clear()
                stdscr.addstr(2, 2, "Initializing Llama 3.2 1B model...")
                stdscr.addstr(4, 2, "This may take a moment on first run...")
                stdscr.refresh()

                PromptExpander.initialize_model()

                stdscr.addstr(6, 2, "Generating prompts...")
                stdscr.refresh()

                user_prompt_library = PromptExpander.generate_prompts(trigger_phrase=trigger, num_prompts=num_prompts_int)

                # Save the prompt library
                with open("config/user_prompt_library.json", "w", encoding="utf-8") as f:
                    json.dump(user_prompt_library, f, indent=4)

                self.state.env_contents["user_prompt_library"] = "config/user_prompt_library.json"

                self.show_message(stdscr, "Prompt library generated successfully!")

            except Exception as e:
                self.show_error(stdscr, f"(warning) Failed to generate prompt library: {str(e)}")

    def _configure_rescale_betas(self, stdscr):
        """Configure rescale betas zero SNR"""
        current = self.state.env_contents.get("rescale_betas_zero_snr", False)

        selected = self.show_options(
            stdscr,
            f"Rescale betas to zero terminal SNR?\n"
            "Recommended for v_prediction training\n(Current: {'Yes' if current else 'No'})",
            ["No", "Yes"],
            1 if current else 0,
        )

        if selected == 1:
            self.state.env_contents["rescale_betas_zero_snr"] = True
        elif selected == 0 and "rescale_betas_zero_snr" in self.state.env_contents:
            del self.state.env_contents["rescale_betas_zero_snr"]

    def _configure_freeze_encoder(self, stdscr):
        """Configure freeze encoder settings"""
        nav = MenuNavigator(stdscr)

        while True:
            current_values = {
                "Freeze Strategy": self.state.env_contents.get("freeze_encoder_strategy", "none"),
                "Layer Freeze Strategy": self.state.env_contents.get("layer_freeze_strategy", "none"),
            }

            menu_items = [
                ("Freeze Strategy", self._configure_freeze_strategy),
                ("Layer Freeze Strategy", self._configure_layer_freeze_strategy),
            ]

            # Add strategy-specific options
            strategy = self.state.env_contents.get("freeze_encoder_strategy", "none")
            if strategy in ["before", "after", "between"]:
                if strategy in ["before", "between"]:
                    current_values["Freeze Before"] = str(self.state.env_contents.get("freeze_encoder_before", 17))
                    menu_items.append(("Freeze Before", self._configure_freeze_before))
                if strategy in ["after", "between"]:
                    current_values["Freeze After"] = str(self.state.env_contents.get("freeze_encoder_after", 17))
                    menu_items.append(("Freeze After", self._configure_freeze_after))

            selected = nav.show_menu("Freeze Encoder Settings", menu_items, current_values)

            if selected == -1:  # Quit
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif selected == -2:  # Back
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_freeze_strategy(self, stdscr):
        """Configure freeze encoder strategy"""
        options = ["none", "before", "after", "between"]
        current = self.state.env_contents.get("freeze_encoder_strategy", "none")

        default_idx = 0
        if current in options:
            default_idx = options.index(current)

        selected = self.show_options(
            stdscr,
            f"Text encoder freeze strategy\n(Current: {current})\n\n"
            "before: Freeze layers before specified layer\n"
            "after: Freeze layers after specified layer\n"
            "between: Freeze layers between two specified layers",
            options,
            default_idx,
        )

        if selected >= 0:
            self.state.env_contents["freeze_encoder_strategy"] = options[selected]

    def _configure_layer_freeze_strategy(self, stdscr):
        """Configure layer freeze strategy"""
        options = ["none", "bitfit"]
        current = self.state.env_contents.get("layer_freeze_strategy", "none")

        default_idx = 0
        if current in options:
            default_idx = options.index(current)

        selected = self.show_options(
            stdscr,
            f"Layer freeze strategy\n(Current: {current})\n\n"
            "none: Normal training\n"
            "bitfit: Freeze weights, train only bias",
            options,
            default_idx,
        )

        if selected >= 0:
            self.state.env_contents["layer_freeze_strategy"] = options[selected]

    def _configure_freeze_before(self, stdscr):
        """Configure freeze before layer"""
        current = str(self.state.env_contents.get("freeze_encoder_before", 17))
        value = self.get_input(
            stdscr,
            f"Freeze layers before this layer number\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["freeze_encoder_before"] = int(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_freeze_after(self, stdscr):
        """Configure freeze after layer"""
        current = str(self.state.env_contents.get("freeze_encoder_after", 17))
        value = self.get_input(
            stdscr,
            f"Freeze layers after this layer number\n(Current: {current})",
            current,
        )
        try:
            self.state.env_contents["freeze_encoder_after"] = int(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_distillation(self, stdscr):
        """Configure distillation settings"""
        current_method = self.state.env_contents.get("distillation_method")

        use_distillation = self.show_options(
            stdscr,
            f"Enable distillation training?\n(Current: {'Yes - ' + current_method if current_method else 'No'})",
            ["No", "Yes"],
            1 if current_method else 0,
        )

        if use_distillation == 1:
            # Choose distillation method
            methods = ["lcm", "dcm"]
            default_idx = 0
            if current_method in methods:
                default_idx = methods.index(current_method)

            method_idx = self.show_options(
                stdscr,
                "Select distillation method:",
                ["LCM (Latent Consistency)", "DCM (Direct Consistency)"],
                default_idx,
            )

            if method_idx >= 0:
                self.state.env_contents["distillation_method"] = methods[method_idx]

                # TODO: Add distillation_config options if needed
        else:
            if "distillation_method" in self.state.env_contents:
                del self.state.env_contents["distillation_method"]
            if "distillation_config" in self.state.env_contents:
                del self.state.env_contents["distillation_config"]

    def controlnet_config(self, stdscr):
        """Step 13: ControlNet Configuration"""
        current_control = "control" in self.state.env_contents or "controlnet" in self.state.env_contents

        use_control = self.show_options(
            stdscr,
            f"Enable control/ControlNet training?\n(Current: {'Yes' if current_control else 'No'})",
            ["No", "Yes"],
            1 if current_control else 0,
        )

        if use_control == 0:
            # Remove control settings
            for key in [
                "control",
                "controlnet",
                "controlnet_model_name_or_path",
                "controlnet_custom_config",
                "conditioning_multidataset_sampling",
            ]:
                if key in self.state.env_contents:
                    del self.state.env_contents[key]
            return

        # Choose control type
        control_type = self.show_options(
            stdscr,
            "Select control type:",
            ["Channel-wise control", "ControlNet"],
            1 if "controlnet" in self.state.env_contents else 0,
        )

        if control_type == 0:
            self.state.env_contents["control"] = True
            if "controlnet" in self.state.env_contents:
                del self.state.env_contents["controlnet"]
        else:
            self.state.env_contents["controlnet"] = True
            if "control" in self.state.env_contents:
                del self.state.env_contents["control"]

            # Configure ControlNet model path
            current_path = self.state.env_contents.get("controlnet_model_name_or_path", "")
            model_path = self.get_input(
                stdscr,
                "Enter ControlNet model path (optional):",
                current_path,
            )
            if model_path:
                self.state.env_contents["controlnet_model_name_or_path"] = model_path

        # Configure conditioning dataset sampling
        current_sampling = self.state.env_contents.get("conditioning_multidataset_sampling", "random")
        sampling_idx = self.show_options(
            stdscr,
            f"Conditioning dataset sampling\n(Current: {current_sampling})",
            ["Random (default)", "Combined (higher VRAM)"],
            0 if current_sampling == "random" else 1,
        )

        if sampling_idx == 1:
            self.state.env_contents["conditioning_multidataset_sampling"] = "combined"
        else:
            self.state.env_contents["conditioning_multidataset_sampling"] = "random"

    def model_specific_options(self, stdscr):
        """Step 14: Model-Specific Options"""
        model_family = self.state.env_contents.get("model_family")

        if not model_family:
            self.show_message(stdscr, "Please select a model family first.")
            return

        nav = MenuNavigator(stdscr)

        while True:
            current_values = {}
            menu_items = []

            # LTX Video options
            if model_family == "ltxvideo":
                current_values["Train Mode"] = self.state.env_contents.get("ltx_train_mode", "i2v")
                current_values["I2V Probability"] = str(self.state.env_contents.get("ltx_i2v_prob", 0.1))
                current_values["Protect First Frame"] = (
                    "Yes" if self.state.env_contents.get("ltx_protect_first_frame", False) else "No"
                )

                menu_items.extend(
                    [
                        ("Train Mode", self._configure_ltx_train_mode),
                        ("I2V Probability", self._configure_ltx_i2v_prob),
                        ("Protect First Frame", self._configure_ltx_protect_first),
                    ]
                )

                if not self.state.env_contents.get("ltx_protect_first_frame", False):
                    current_values["Partial Noise Fraction"] = str(
                        self.state.env_contents.get("ltx_partial_noise_fraction", 0.05)
                    )
                    menu_items.append(("Partial Noise Fraction", self._configure_ltx_partial_noise))

            # SD3 options
            elif model_family == "sd3":
                current_values["CLIP Uncond Behaviour"] = self.state.env_contents.get(
                    "sd3_clip_uncond_behaviour", "empty_string"
                )
                current_values["T5 Uncond Behaviour"] = self.state.env_contents.get("sd3_t5_uncond_behaviour", "follow_clip")

                menu_items.extend(
                    [
                        ("CLIP Uncond Behaviour", self._configure_sd3_clip_uncond),
                        ("T5 Uncond Behaviour", self._configure_sd3_t5_uncond),
                    ]
                )

            # T5 padding (for models with T5)
            if model_family in ["flux", "sd3", "pixart_sigma"]:
                current_values["T5 Padding"] = self.state.env_contents.get("t5_padding", "unmodified")
                menu_items.append(("T5 Padding", self._configure_t5_padding))

            # Sana options
            elif model_family == "sana":
                current_values["Complex Human Instruction"] = (
                    "Enabled" if self.state.env_contents.get("sana_complex_human_instruction", True) else "Disabled"
                )
                menu_items.append(("Complex Human Instruction", self._configure_sana_instruction))

            if not menu_items:
                self.show_message(stdscr, f"No specific options available for {model_family}.")
                return

            selected = nav.show_menu(f"{model_family.upper()} Specific Options", menu_items, current_values)

            if selected == -1:  # Quit
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif selected == -2:  # Back
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_ltx_train_mode(self, stdscr):
        """Configure LTX train mode"""
        options = ["t2v", "i2v"]
        current = self.state.env_contents.get("ltx_train_mode", "i2v")

        default_idx = 1 if current == "i2v" else 0

        selected = self.show_options(
            stdscr,
            f"LTX training mode\n(Current: {current})\n\n" "t2v: Text-to-video\n" "i2v: Image-to-video",
            options,
            default_idx,
        )

        if selected >= 0:
            self.state.env_contents["ltx_train_mode"] = options[selected]

    def _configure_ltx_i2v_prob(self, stdscr):
        """Configure LTX i2v probability"""
        current = str(self.state.env_contents.get("ltx_i2v_prob", 0.1))
        value = self.get_input(
            stdscr,
            f"I2V probability (0.0-1.0)\n" "Chance of applying i2v style training\n(Current: {current})",
            current,
        )
        try:
            prob = float(value)
            if 0.0 <= prob <= 1.0:
                self.state.env_contents["ltx_i2v_prob"] = prob
            else:
                self.show_error(stdscr, "Value must be between 0.0 and 1.0")
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_ltx_protect_first(self, stdscr):
        """Configure LTX protect first frame"""
        current = self.state.env_contents.get("ltx_protect_first_frame", False)

        selected = self.show_options(
            stdscr,
            f"Fully protect first frame in i2v?\n(Current: {'Yes' if current else 'No'})",
            ["No", "Yes"],
            1 if current else 0,
        )

        if selected == 1:
            self.state.env_contents["ltx_protect_first_frame"] = True
        elif selected == 0 and "ltx_protect_first_frame" in self.state.env_contents:
            del self.state.env_contents["ltx_protect_first_frame"]

    def _configure_ltx_partial_noise(self, stdscr):
        """Configure LTX partial noise fraction"""
        current = str(self.state.env_contents.get("ltx_partial_noise_fraction", 0.05))
        value = self.get_input(
            stdscr,
            f"Maximum noise fraction for first frame (0.0-1.0)\n" "0.05 = 5% noise, 95% original\n(Current: {current})",
            current,
        )
        try:
            frac = float(value)
            if 0.0 <= frac <= 1.0:
                self.state.env_contents["ltx_partial_noise_fraction"] = frac
            else:
                self.show_error(stdscr, "Value must be between 0.0 and 1.0")
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_sd3_clip_uncond(self, stdscr):
        """Configure SD3 CLIP uncond behaviour"""
        options = ["empty_string", "zero"]
        current = self.state.env_contents.get("sd3_clip_uncond_behaviour", "empty_string")

        default_idx = 0 if current == "empty_string" else 1

        selected = self.show_options(
            stdscr,
            f"SD3 CLIP unconditional behaviour\n(Current: {current})",
            options,
            default_idx,
        )

        if selected >= 0:
            self.state.env_contents["sd3_clip_uncond_behaviour"] = options[selected]

    def _configure_sd3_t5_uncond(self, stdscr):
        """Configure SD3 T5 uncond behaviour"""
        options = ["follow_clip", "empty_string", "zero"]
        current = self.state.env_contents.get("sd3_t5_uncond_behaviour", "follow_clip")

        default_idx = 0
        if current in options:
            default_idx = options.index(current)

        selected = self.show_options(
            stdscr,
            f"SD3 T5 unconditional behaviour\n(Current: {current})",
            ["Follow CLIP setting", "Empty string", "Zero"],
            default_idx,
        )

        if selected >= 0:
            if selected == 0:
                if "sd3_t5_uncond_behaviour" in self.state.env_contents:
                    del self.state.env_contents["sd3_t5_uncond_behaviour"]
            else:
                self.state.env_contents["sd3_t5_uncond_behaviour"] = options[selected]

    def _configure_t5_padding(self, stdscr):
        """Configure T5 padding"""
        options = ["unmodified", "zero"]
        current = self.state.env_contents.get("t5_padding", "unmodified")

        default_idx = 0 if current == "unmodified" else 1

        selected = self.show_options(
            stdscr,
            f"T5 padding behaviour\n(Current: {current})",
            options,
            default_idx,
        )

        if selected >= 0:
            self.state.env_contents["t5_padding"] = options[selected]

    def _configure_sana_instruction(self, stdscr):
        """Configure Sana complex human instruction"""
        current = self.state.env_contents.get("sana_complex_human_instruction", True)

        selected = self.show_options(
            stdscr,
            f"Attach complex human instruction to prompts?\n"
            "Required for Gemma model\n(Current: {'Yes' if current else 'No'})",
            ["No", "Yes"],
            1 if current else 0,
        )

        if selected == 0:
            self.state.env_contents["sana_complex_human_instruction"] = False
        elif selected == 1 and "sana_complex_human_instruction" in self.state.env_contents:
            del self.state.env_contents["sana_complex_human_instruction"]

    def dataset_config(self, stdscr):
        """Step 15: Dataset Configuration - Comprehensive version"""
        config_idx = self.show_options(stdscr, "Would you like to configure your dataloader?", ["Yes", "No"], 0)

        if config_idx == 1:
            return

        nav = MenuNavigator(stdscr)

        # Initialize dataset manager if not exists
        if not hasattr(self, "_datasets"):
            self._datasets = []
            self._current_dataset_idx = 0

        while True:
            # Build menu
            current_values = {
                "Datasets Configured": str(len(self._datasets)),
            }

            menu_items = [
                ("Add New Dataset", self._add_dataset),
                ("Edit Existing Dataset", self._edit_dataset),
                ("Remove Dataset", self._remove_dataset),
                ("Configure Text Embeds Dataset", self._configure_text_embeds),
                ("Configure Image Embeds Dataset", self._configure_image_embeds),
                ("Review All Datasets", self._review_datasets),
                ("Apply Configuration", self._apply_all_datasets),
            ]

            selected = nav.show_menu("Dataset Configuration Manager", menu_items, current_values)

            if selected == -1:  # Quit
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif selected == -2:  # Back
                return
            elif selected >= 0:
                if selected == len(menu_items) - 1:  # Apply Configuration
                    menu_items[selected][1](stdscr)
                    return
                else:
                    menu_items[selected][1](stdscr)

    def _add_dataset(self, stdscr):
        """Add a new dataset configuration"""
        # Choose dataset type
        dataset_types = ["image", "video", "conditioning"]
        type_idx = self.show_options(
            stdscr,
            "Select dataset type:",
            ["Image Dataset", "Video Dataset", "Conditioning Dataset"],
            0,
        )

        if type_idx < 0:
            return

        dataset_type = dataset_types[type_idx]

        # Choose backend type
        backend_types = ["local", "aws", "csv", "huggingface"]
        backend_idx = self.show_options(
            stdscr,
            "Select storage backend:",
            ["Local Filesystem", "AWS S3", "CSV URL List", "Hugging Face Dataset"],
            0,
        )

        if backend_idx < 0:
            return

        backend_type = backend_types[backend_idx]

        # Create dataset config
        dataset = {
            "dataset_type": dataset_type,
            "type": backend_type,
            "disabled": False,
        }

        # Configure based on type
        if dataset_type == "image":
            self._configure_image_dataset(stdscr, dataset)
        elif dataset_type == "video":
            self._configure_video_dataset(stdscr, dataset)
        elif dataset_type == "conditioning":
            self._configure_conditioning_dataset(stdscr, dataset)

        self._datasets.append(dataset)
        self.show_message(stdscr, f"Dataset '{dataset['id']}' added successfully!")

    def _configure_image_dataset(self, stdscr, dataset):
        """Configure an image dataset with all options"""
        nav = MenuNavigator(stdscr)

        # Initialize with defaults
        dataset.update(
            {
                "id": f"dataset-{len(self._datasets)}",
                "resolution": 1024,
                "resolution_type": "pixel_area",
                "caption_strategy": "textfile",
                "crop": False,
                "repeats": 0,
                "minimum_image_size": 0,
                "metadata_backend": "discovery",
            }
        )

        while True:
            # Build current values
            current_values = {
                "Dataset ID": dataset["id"],
                "Resolution": f"{dataset['resolution']} ({dataset['resolution_type']})",
                "Caption Strategy": dataset["caption_strategy"],
                "Crop": "Yes" if dataset.get("crop", False) else "No",
                "Repeats": str(dataset.get("repeats", 0)),
            }

            # Add backend-specific values
            if dataset["type"] == "local":
                current_values["Data Directory"] = dataset.get("instance_data_dir", "Not set")
                current_values["VAE Cache"] = dataset.get("cache_dir_vae", "Not set")
            elif dataset["type"] == "aws":
                current_values["S3 Bucket"] = dataset.get("aws_bucket_name", "Not set")
                current_values["S3 Prefix"] = dataset.get("aws_data_prefix", "")

            menu_items = [
                (
                    "Basic Settings",
                    lambda s: self._configure_dataset_basics(s, dataset),
                ),
                (
                    "Resolution Settings",
                    lambda s: self._configure_resolution_settings(s, dataset),
                ),
                (
                    "Caption Settings",
                    lambda s: self._configure_caption_settings(s, dataset),
                ),
                ("Crop Settings", lambda s: self._configure_crop_settings(s, dataset)),
                (
                    "Cache Settings",
                    lambda s: self._configure_cache_settings(s, dataset),
                ),
                (
                    "Advanced Settings",
                    lambda s: self._configure_advanced_settings(s, dataset),
                ),
            ]

            # Add backend-specific options
            if dataset["type"] == "local":
                menu_items.insert(
                    1,
                    (
                        "Local Path Settings",
                        lambda s: self._configure_local_paths(s, dataset),
                    ),
                )
            elif dataset["type"] == "aws":
                menu_items.insert(
                    1,
                    (
                        "AWS Settings",
                        lambda s: self._configure_aws_settings(s, dataset),
                    ),
                )
            elif dataset["type"] == "csv":
                menu_items.insert(
                    1,
                    (
                        "CSV Settings",
                        lambda s: self._configure_csv_settings(s, dataset),
                    ),
                )
            elif dataset["type"] == "huggingface":
                menu_items.insert(
                    1,
                    (
                        "HuggingFace Settings",
                        lambda s: self._configure_hf_settings(s, dataset),
                    ),
                )

            # Add conditioning options if applicable
            if "conditioning" not in dataset and dataset["dataset_type"] == "image":
                menu_items.append(
                    (
                        "Auto-generate Conditioning",
                        lambda s: self._configure_auto_conditioning(s, dataset),
                    )
                )

            selected = nav.show_menu(f"Configure Image Dataset: {dataset['id']}", menu_items, current_values)

            if selected == -1:  # Quit
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif selected == -2:  # Back
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_dataset_basics(self, stdscr, dataset):
        """Configure basic dataset settings"""
        nav = MenuNavigator(stdscr)

        while True:
            current_values = {
                "Dataset ID": dataset["id"],
                "Disabled": "Yes" if dataset.get("disabled", False) else "No",
                "Probability": str(dataset.get("probability", 1.0)),
                "Repeats": str(dataset.get("repeats", 0)),
                "Regularization Data": ("Yes" if dataset.get("is_regularisation_data", False) else "No"),
            }

            menu_items = [
                ("Dataset ID", lambda s: self._set_dataset_id(s, dataset)),
                ("Enable/Disable", lambda s: self._toggle_dataset(s, dataset)),
                ("Sampling Probability", lambda s: self._set_probability(s, dataset)),
                ("Dataset Repeats", lambda s: self._set_repeats(s, dataset)),
                (
                    "Regularization Data",
                    lambda s: self._toggle_regularization(s, dataset),
                ),
            ]

            selected = nav.show_menu("Basic Dataset Settings", menu_items, current_values)

            if selected == -1 or selected == -2:
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_resolution_settings(self, stdscr, dataset):
        """Configure resolution and image size settings"""
        nav = MenuNavigator(stdscr)

        while True:
            current_values = {
                "Resolution": str(dataset.get("resolution", 1024)),
                "Resolution Type": dataset.get("resolution_type", "pixel_area"),
                "Min Image Size": str(dataset.get("minimum_image_size", 0)),
                "Max Image Size": (
                    str(dataset.get("maximum_image_size", 0)) if dataset.get("maximum_image_size") else "Not set"
                ),
                "Target Downsample": (
                    str(dataset.get("target_downsample_size", 0)) if dataset.get("target_downsample_size") else "Not set"
                ),
                "Min Aspect Ratio": str(dataset.get("minimum_aspect_ratio", 0.5)),
                "Max Aspect Ratio": str(dataset.get("maximum_aspect_ratio", 3.0)),
            }

            menu_items = [
                ("Resolution", lambda s: self._set_resolution(s, dataset)),
                ("Resolution Type", lambda s: self._set_resolution_type(s, dataset)),
                ("Minimum Image Size", lambda s: self._set_min_image_size(s, dataset)),
                ("Maximum Image Size", lambda s: self._set_max_image_size(s, dataset)),
                (
                    "Target Downsample Size",
                    lambda s: self._set_target_downsample(s, dataset),
                ),
                ("Aspect Ratio Limits", lambda s: self._set_aspect_limits(s, dataset)),
            ]

            selected = nav.show_menu("Resolution Settings", menu_items, current_values)

            if selected == -1 or selected == -2:
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_caption_settings(self, stdscr, dataset):
        """Configure caption-related settings"""
        nav = MenuNavigator(stdscr)

        while True:
            current_values = {
                "Caption Strategy": dataset.get("caption_strategy", "textfile"),
                "Instance Prompt": dataset.get("instance_prompt", "Not set"),
                "Prepend Instance": ("Yes" if dataset.get("prepend_instance_prompt", False) else "No"),
                "Only Instance": ("Yes" if dataset.get("only_instance_prompt", False) else "No"),
            }

            # Add parquet settings if applicable
            if dataset.get("caption_strategy") == "parquet":
                if "parquet" in dataset:
                    current_values["Parquet File"] = dataset["parquet"].get("path", "Not set")
                else:
                    current_values["Parquet File"] = "Not configured"

            menu_items = [
                ("Caption Strategy", lambda s: self._set_caption_strategy(s, dataset)),
                ("Instance Prompt", lambda s: self._set_instance_prompt(s, dataset)),
                (
                    "Prepend Instance Prompt",
                    lambda s: self._toggle_prepend_instance(s, dataset),
                ),
                (
                    "Only Instance Prompt",
                    lambda s: self._toggle_only_instance(s, dataset),
                ),
            ]

            # Add strategy-specific options
            if dataset.get("caption_strategy") == "parquet":
                menu_items.append(("Parquet Settings", lambda s: self._configure_parquet(s, dataset)))

            # Add caption filter list option for text datasets
            menu_items.append(("Caption Filter List", lambda s: self._set_caption_filter(s, dataset)))

            selected = nav.show_menu("Caption Settings", menu_items, current_values)

            if selected == -1 or selected == -2:
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_crop_settings(self, stdscr, dataset):
        """Configure cropping and aspect settings"""
        nav = MenuNavigator(stdscr)

        while True:
            current_values = {
                "Crop Enabled": "Yes" if dataset.get("crop", False) else "No",
                "Crop Style": dataset.get("crop_style", "center"),
                "Crop Aspect": dataset.get("crop_aspect", "square"),
                "Aspect Buckets": (
                    str(len(dataset.get("crop_aspect_buckets", []))) + " buckets"
                    if "crop_aspect_buckets" in dataset
                    else "Not set"
                ),
                "Aspect Rounding": str(dataset.get("aspect_bucket_rounding", 2)),
            }

            menu_items = [
                ("Enable Cropping", lambda s: self._toggle_crop(s, dataset)),
                ("Crop Style", lambda s: self._set_crop_style(s, dataset)),
                ("Crop Aspect", lambda s: self._set_crop_aspect(s, dataset)),
                ("Aspect Buckets", lambda s: self._set_aspect_buckets(s, dataset)),
                ("Aspect Rounding", lambda s: self._set_aspect_rounding(s, dataset)),
            ]

            selected = nav.show_menu("Crop Settings", menu_items, current_values)

            if selected == -1 or selected == -2:
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_cache_settings(self, stdscr, dataset):
        """Configure cache-related settings"""
        nav = MenuNavigator(stdscr)

        while True:
            current_values = {
                "VAE Cache Dir": dataset.get("cache_dir_vae", "Not set"),
                "Clear VAE Each Epoch": ("Yes" if dataset.get("vae_cache_clear_each_epoch", False) else "No"),
                "Hash Filenames": ("Yes" if dataset.get("hash_filenames", False) else "No"),
                "Skip Discovery": dataset.get("skip_file_discovery", "None"),
                "Preserve Cache": ("Yes" if dataset.get("preserve_data_backend_cache", False) else "No"),
                "Text Embeds": dataset.get("text_embeds", "default"),
                "Image Embeds": dataset.get("image_embeds", "Not set"),
            }

            menu_items = [
                ("VAE Cache Directory", lambda s: self._set_vae_cache_dir(s, dataset)),
                (
                    "Clear VAE Cache Each Epoch",
                    lambda s: self._toggle_vae_clear(s, dataset),
                ),
                ("Hash Filenames", lambda s: self._toggle_hash_filenames(s, dataset)),
                ("Skip File Discovery", lambda s: self._set_skip_discovery(s, dataset)),
                (
                    "Preserve Backend Cache",
                    lambda s: self._toggle_preserve_cache(s, dataset),
                ),
                (
                    "Text Embeds Dataset",
                    lambda s: self._set_text_embeds_ref(s, dataset),
                ),
                (
                    "Image Embeds Dataset",
                    lambda s: self._set_image_embeds_ref(s, dataset),
                ),
            ]

            selected = nav.show_menu("Cache Settings", menu_items, current_values)

            if selected == -1 or selected == -2:
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_text_embeds(self, stdscr):
        """Configure a text embeds dataset"""
        dataset = {
            "dataset_type": "text_embeds",
            "type": "local",
            "default": False,
            "write_batch_size": 128,
        }

        # Get ID
        dataset["id"] = self.get_input(
            stdscr,
            "Enter text embeds dataset ID:",
            f"text-embeds-{len(self._datasets)}",
        )

        # Set as default?
        default_idx = self.show_options(stdscr, "Set as default text embeds dataset?", ["No", "Yes"], 0)
        dataset["default"] = default_idx == 1

        # Storage backend
        backend_idx = self.show_options(stdscr, "Storage backend:", ["Local", "AWS S3"], 0)

        if backend_idx == 0:
            dataset["type"] = "local"
            dataset["cache_dir"] = self.get_input(
                stdscr,
                "Enter cache directory for text embeds:",
                f"cache/text/{self.state.env_contents.get('model_family', 'model')}",
            )
        else:
            dataset["type"] = "aws"
            self._configure_aws_settings(stdscr, dataset)

        # Write batch size
        batch_size = self.get_input(stdscr, "Write batch size:", str(dataset["write_batch_size"]))
        try:
            dataset["write_batch_size"] = int(batch_size)
        except ValueError:
            pass

        self._datasets.append(dataset)
        self.show_message(stdscr, f"Text embeds dataset '{dataset['id']}' added!")

    def _configure_image_embeds(self, stdscr):
        """Configure an image embeds dataset"""
        dataset = {
            "dataset_type": "image_embeds",
            "type": "local",
        }

        # Get ID
        dataset["id"] = self.get_input(
            stdscr,
            "Enter image embeds dataset ID:",
            f"image-embeds-{len(self._datasets)}",
        )

        # Storage backend
        backend_idx = self.show_options(stdscr, "Storage backend:", ["Local", "AWS S3"], 0)

        if backend_idx == 0:
            dataset["type"] = "local"
        else:
            dataset["type"] = "aws"
            self._configure_aws_settings(stdscr, dataset)

        self._datasets.append(dataset)
        self.show_message(stdscr, f"Image embeds dataset '{dataset['id']}' added!")

    def _apply_all_datasets(self, stdscr):
        """Apply all dataset configurations"""
        if not self._datasets:
            self.show_error(stdscr, "No datasets configured!")
            return

        # Validate configuration
        has_text_embeds = any(d.get("dataset_type") == "text_embeds" and d.get("default") for d in self._datasets)

        if not has_text_embeds:
            # Auto-create default text embeds
            self._datasets.append(
                {
                    "id": "text-embeds",
                    "dataset_type": "text_embeds",
                    "default": True,
                    "type": "local",
                    "cache_dir": f"cache/text/{self.state.env_contents.get('model_family', 'model')}",
                    "write_batch_size": 128,
                }
            )

        # Save configuration
        self.state.dataset_config = self._datasets
        self.state.env_contents["data_backend_config"] = self.state.env_contents.get(
            "data_backend_config", "config/multidatabackend.json"
        )

        self.show_message(stdscr, f"Applied {len(self._datasets)} dataset configurations!")

    # Helper methods for all the configuration options...
    def _set_dataset_id(self, stdscr, dataset):
        """Set dataset ID"""
        dataset["id"] = self.get_input(
            stdscr,
            "Enter dataset ID (unique identifier):",
            dataset.get("id", "my-dataset"),
        )

    def _toggle_dataset(self, stdscr, dataset):
        """Toggle dataset enabled/disabled"""
        current = dataset.get("disabled", False)
        idx = self.show_options(
            stdscr,
            f"Dataset currently: {'Disabled' if current else 'Enabled'}",
            ["Enable", "Disable"],
            1 if current else 0,
        )
        if idx >= 0:
            dataset["disabled"] = idx == 1

    def _set_probability(self, stdscr, dataset):
        """Set sampling probability"""
        current = str(dataset.get("probability", 1.0))
        value = self.get_input(stdscr, "Sampling probability (0.0-1.0):", current)
        try:
            prob = float(value)
            if 0.0 <= prob <= 1.0:
                dataset["probability"] = prob
        except ValueError:
            pass

    def _configure_parquet(self, stdscr, dataset):
        """Configure parquet settings"""
        if "parquet" not in dataset:
            dataset["parquet"] = {}

        nav = MenuNavigator(stdscr)
        parquet = dataset["parquet"]

        while True:
            current_values = {
                "Path": parquet.get("path", "Not set"),
                "Filename Column": parquet.get("filename_column", "id"),
                "Caption Column": parquet.get("caption_column", "caption"),
                "Width Column": parquet.get("width_column", "Not set"),
                "Height Column": parquet.get("height_column", "Not set"),
                "ID Has Extension": ("Yes" if parquet.get("identifier_includes_extension", False) else "No"),
            }

            menu_items = [
                ("Parquet File Path", lambda s: self._set_parquet_path(s, parquet)),
                (
                    "Filename Column",
                    lambda s: self._set_parquet_column(s, parquet, "filename_column", "Filename column name:"),
                ),
                (
                    "Caption Column",
                    lambda s: self._set_parquet_column(s, parquet, "caption_column", "Caption column name:"),
                ),
                (
                    "Width Column",
                    lambda s: self._set_parquet_column(s, parquet, "width_column", "Width column name (optional):"),
                ),
                (
                    "Height Column",
                    lambda s: self._set_parquet_column(s, parquet, "height_column", "Height column name (optional):"),
                ),
                (
                    "Fallback Caption",
                    lambda s: self._set_parquet_column(
                        s,
                        parquet,
                        "fallback_caption_column",
                        "Fallback caption column (optional):",
                    ),
                ),
                (
                    "ID Includes Extension",
                    lambda s: self._toggle_parquet_extension(s, parquet),
                ),
            ]

            selected = nav.show_menu("Parquet Configuration", menu_items, current_values)

            if selected == -1 or selected == -2:
                dataset["metadata_backend"] = "parquet"
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_dataset_id(self, stdscr):
        """Configure dataset ID"""
        current = self._dataset_values["id"]
        dataset_id = self.get_input(
            stdscr,
            "Enter the name of your dataset.\n"
            "This will be used to generate the cache directory.\n"
            "It should be simple, and not contain spaces or special characters.",
            current,
        )
        self._dataset_values["id"] = dataset_id

    def _configure_dataset_path(self, stdscr):
        """Configure dataset path"""
        current = self._dataset_values["path"]
        dataset_path = self.get_input(
            stdscr,
            "Enter the path to your dataset.\n"
            "This should be a directory containing images and text files for their caption.\n"
            "For reliability, use an absolute (full) path, beginning with a '/'",
            current,
        )
        self._dataset_values["path"] = dataset_path

    def _configure_caption_strategy(self, stdscr):
        """Configure caption strategy"""
        stdscr.clear()
        stdscr.addstr(1, 2, "Caption Strategy Selection", curses.A_BOLD)
        stdscr.addstr(3, 2, "How should the dataloader handle captions?")
        stdscr.addstr(5, 2, "-> 'filename' will use the names of your image files as the caption")
        stdscr.addstr(
            6,
            2,
            "-> 'textfile' requires a image.txt file to go next to your image.png file",
        )
        stdscr.addstr(7, 2, "-> 'instanceprompt' will just use one trigger phrase for all images")
        stdscr.refresh()

        caption_strategies = ["filename", "textfile", "instanceprompt"]
        current = self._dataset_values["caption_strategy"]
        default_idx = 1
        if current in caption_strategies:
            default_idx = caption_strategies.index(current)

        caption_idx = self.show_options(stdscr, "", caption_strategies, default_idx)

        if caption_idx >= 0:
            self._dataset_values["caption_strategy"] = caption_strategies[caption_idx]

    def _configure_instance_prompt(self, stdscr):
        """Configure instance prompt"""
        current = self._dataset_values.get("instance_prompt", "Character Name")
        instance_prompt = self.get_input(
            stdscr,
            "Enter the instance_prompt you want to use for all images in this dataset:",
            current,
        )
        self._dataset_values["instance_prompt"] = instance_prompt

    def _configure_dataset_repeats(self, stdscr):
        """Configure dataset repeats"""
        current = str(self._dataset_values["repeats"])
        repeats = self.get_input(
            stdscr,
            "How many times do you want to repeat each image in the dataset?\n"
            "A value of zero means the dataset will only be seen once;\n"
            "a value of one will cause the dataset to be sampled twice.",
            current,
        )

        try:
            self._dataset_values["repeats"] = int(repeats)
        except ValueError:
            self._dataset_values["repeats"] = 10

    def _configure_resolutions(self, stdscr):
        """Configure resolutions"""
        model_family = self.state.env_contents.get("model_family", "flux")
        multi_resolution_capable = ["flux"]

        if model_family in multi_resolution_capable:
            default_res = "256,512,768,1024,1440"
            multi_res_text = (
                "Multiple resolutions may be provided, but this is only recommended for Flux.\n"
                "A comma-separated list of values or a single item can be given."
            )
        else:
            default_res = "1024"
            multi_res_text = (
                "A comma-separated list of values or a single item can be given to train on multiple base resolutions.\n"
                "WARNING: Multi-resolution training is not recommended for this model type."
            )

        current = ",".join(map(str, self._dataset_values["resolutions"]))

        resolutions_str = self.get_input(
            stdscr,
            f"Which resolutions do you want to train?\n{multi_res_text}",
            current,
        )

        try:
            if "," in resolutions_str:
                resolutions = [int(r.strip()) for r in resolutions_str.split(",")]

                # Show warning for non-Flux models
                if model_family not in multi_resolution_capable:
                    stdscr.clear()
                    stdscr.addstr(2, 2, "WARNING", curses.A_BOLD | curses.color_pair(1))
                    stdscr.addstr(
                        4,
                        2,
                        "Most models do not play well with multi-resolution training,",
                    )
                    stdscr.addstr(5, 2, "resulting in degraded outputs and broken hearts.")
                    stdscr.addstr(6, 2, "Proceed with caution.")
                    stdscr.addstr(8, 2, "Press any key to continue...")
                    stdscr.refresh()
                    stdscr.getch()
            else:
                resolutions = [int(resolutions_str)]

            self._dataset_values["resolutions"] = resolutions
        except ValueError:
            self.show_error(stdscr, "Invalid resolution value. Keeping current settings.")

    def _configure_cache_dir(self, stdscr):
        """Configure cache directory"""
        current = self._dataset_values["cache_dir"]
        cache_dir = self.get_input(
            stdscr,
            "Where will your VAE and text encoder caches be written to?\n"
            "Subdirectories will be created inside for you automatically.",
            current,
        )
        self._dataset_values["cache_dir"] = cache_dir

    def _configure_large_images(self, stdscr):
        """Configure large images handling"""
        current = self._dataset_values["has_large_images"]

        large_img_idx = self.show_options(
            stdscr,
            f"Do you have very-large images in the dataset (eg. much larger than 1024x1024)?\n(Current: {'Yes' if current else 'No'})",
            ["No", "Yes"],
            1 if current else 0,
        )

        self._dataset_values["has_large_images"] = large_img_idx == 1

    def _apply_dataset_config(self, stdscr):
        """Apply the dataset configuration"""
        self._build_dataset_config(
            self._dataset_values["id"],
            self._dataset_values["path"],
            self._dataset_values["caption_strategy"],
            self._dataset_values.get("instance_prompt"),
            self._dataset_values["repeats"],
            self._dataset_values["resolutions"],
            self._dataset_values["cache_dir"],
            self._dataset_values["has_large_images"],
        )

        # Show configuration summary
        stdscr.clear()
        stdscr.addstr(1, 2, "Dataset Configuration Summary", curses.A_BOLD)
        stdscr.addstr(3, 2, f"Dataset ID: {self._dataset_values['id']}")
        stdscr.addstr(4, 2, f"Dataset Path: {self._dataset_values['path']}")
        stdscr.addstr(5, 2, f"Caption Strategy: {self._dataset_values['caption_strategy']}")
        stdscr.addstr(6, 2, f"Repeats: {self._dataset_values['repeats']}")
        stdscr.addstr(
            7,
            2,
            f"Resolutions: {', '.join(map(str, self._dataset_values['resolutions']))}",
        )
        stdscr.addstr(8, 2, f"Cache Directory: {self._dataset_values['cache_dir']}")

        if self._dataset_values.get("instance_prompt"):
            stdscr.addstr(9, 2, f"Instance Prompt: {self._dataset_values['instance_prompt']}")

        stdscr.addstr(
            11,
            2,
            "Dataset entries will be created for both cropped and uncropped versions.",
        )
        stdscr.addstr(13, 2, "Configuration applied successfully!")
        stdscr.addstr(15, 2, "Press any key to continue...")
        stdscr.refresh()
        stdscr.getch()

    def _build_dataset_config(
        self,
        dataset_id,
        dataset_path,
        caption_strategy,
        instance_prompt,
        dataset_repeats,
        resolutions,
        cache_dir,
        has_large_images,
    ):
        """Helper to build dataset configuration"""
        resolution_configs = {
            64: {"resolution": 64, "minimum_image_size": 48},
            96: {"resolution": 96, "minimum_image_size": 64},
            128: {"resolution": 128, "minimum_image_size": 96},
            256: {"resolution": 256, "minimum_image_size": 128},
            512: {"resolution": 512, "minimum_image_size": 256},
            768: {"resolution": 768, "minimum_image_size": 512},
            1024: {"resolution": 1024, "minimum_image_size": 768},
            1440: {"resolution": 1440, "minimum_image_size": 1024},
            2048: {"resolution": 2048, "minimum_image_size": 1440},
        }

        default_dataset = {
            "id": "PLACEHOLDER",
            "type": "local",
            "instance_data_dir": None,
            "crop": False,
            "resolution_type": "pixel_area",
            "metadata_backend": "discovery",
            "caption_strategy": "filename",
            "cache_dir_vae": "vae",
        }

        default_cropped = default_dataset.copy()
        default_cropped.update(
            {
                "id": "PLACEHOLDER-crop",
                "crop": True,
                "crop_aspect": "square",
                "crop_style": "center",
                "vae_cache_clear_each_epoch": False,
                "cache_dir_vae": "vae-crop",
            }
        )

        datasets = [
            {
                "id": "text-embed-cache",
                "dataset_type": "text_embeds",
                "default": True,
                "type": "local",
                "cache_dir": os.path.abspath(os.path.join(cache_dir, self.state.env_contents["model_family"], "text")),
                "write_batch_size": 128,
            }
        ]

        def create_dataset(resolution, template):
            dataset = template.copy()
            dataset.update(resolution_configs.get(resolution, {"resolution": resolution}))
            dataset["id"] = (
                f"{dataset_id}-{resolution}" if "crop" not in dataset["id"] else f"{dataset_id}-crop-{resolution}"
            )
            dataset["instance_data_dir"] = os.path.abspath(dataset_path)
            dataset["repeats"] = dataset_repeats
            dataset["cache_dir_vae"] = os.path.abspath(
                os.path.join(
                    cache_dir,
                    self.state.env_contents["model_family"],
                    dataset["cache_dir_vae"],
                    str(resolution),
                )
            )
            dataset["caption_strategy"] = caption_strategy

            if instance_prompt:
                dataset["instance_prompt"] = instance_prompt

            if has_large_images:
                dataset["maximum_image_size"] = resolution
                dataset["target_downsample_size"] = resolution

            return dataset

        for resolution in resolutions:
            datasets.append(create_dataset(resolution, default_dataset))
            datasets.append(create_dataset(resolution, default_cropped))

        self.state.dataset_config = datasets

    def review_and_save(self, stdscr):
        """Step 16: Review and Save Configuration"""
        stdscr.clear()
        h, w = stdscr.getmaxyx()

        # Display configuration summary
        stdscr.addstr(1, 2, "Configuration Summary", curses.A_BOLD)
        # Print the JSON configuration
        config_str = json.dumps(self.state.env_contents, indent=4)
        lines = config_str.splitlines()
        for i, line in enumerate(lines):
            if i + 3 < h:
                stdscr.addstr(i + 3, 2, line)

        stdscr.addstr(h - 3, 2, "Press 's' to save, 'b' to go back, 'q' to quit without saving")
        stdscr.refresh()

        while True:
            key = stdscr.getch()
            if key == ord("s"):
                self._save_configuration(stdscr)
                return
            elif key == ord("b"):
                return
            elif key == ord("q"):
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt

    def _save_configuration(self, stdscr):
        """Save the configuration files"""
        try:
            # Determine save path
            save_path = "config/config.json"

            if self.state.loaded_config_path:
                save_options = [
                    f"Save to original location ({self.state.loaded_config_path})",
                    "Save to new location",
                    "Save to default (config/config.json)",
                ]

                save_choice = self.show_options(stdscr, "Where would you like to save?", save_options, 0)

                if save_choice == 0:
                    save_path = self.state.loaded_config_path
                elif save_choice == 1:
                    save_path = self.get_input(
                        stdscr,
                        "Enter save path for config.json:",
                        "config/my-preset/config.json",
                    )
                    save_dir = os.path.dirname(save_path)
                    if save_dir and not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                elif save_choice == 2:
                    save_path = "config/config.json"
                else:
                    return  # Cancelled

            # Ensure config directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save main config
            with open(save_path, "w") as f:
                json.dump(self.state.env_contents, f, indent=4)

            # Save dataset config if configured
            if self.state.dataset_config:
                backend_path = self.state.env_contents.get("data_backend_config", "config/multidatabackend.json")
                backend_dir = os.path.dirname(backend_path)
                if backend_dir and not os.path.exists(backend_dir):
                    os.makedirs(backend_dir, exist_ok=True)

                with open(backend_path, "w") as f:
                    json.dump(self.state.dataset_config, f, indent=4)

            stdscr.clear()
            stdscr.addstr(2, 2, "Configuration saved successfully!", curses.A_BOLD)
            stdscr.addstr(4, 2, "Files created:")
            stdscr.addstr(5, 4, f"- {save_path}")

            if self.state.dataset_config:
                stdscr.addstr(
                    6,
                    4,
                    f"- {self.state.env_contents.get('data_backend_config', 'config/multidatabackend.json')}",
                )

            if self.state.lycoris_config:
                stdscr.addstr(
                    7,
                    4,
                    f"- {self.state.env_contents.get('lycoris_config', 'config/lycoris_config.json')}",
                )

            stdscr.addstr(9, 2, "Press any key to continue...")
            stdscr.refresh()
            stdscr.getch()

            # Update loaded path
            self.state.loaded_config_path = save_path

        except Exception as e:
            self.show_error(stdscr, f"Failed to save configuration: {str(e)}")

    def _configure_video_dataset(self, stdscr, dataset):
        """Configure a video dataset"""
        # Use most of the image dataset configuration
        self._configure_image_dataset(stdscr, dataset)

        # Add video-specific configuration
        nav = MenuNavigator(stdscr)

        if "video" not in dataset:
            dataset["video"] = {}

        video_config = dataset["video"]

        while True:
            current_values = {
                "Num Frames": str(video_config.get("num_frames", 125)),
                "Min Frames": str(video_config.get("min_frames", 125)),
                "Max Frames": (str(video_config.get("max_frames", 0)) if video_config.get("max_frames") else "Not set"),
                "Is I2V": "Yes" if video_config.get("is_i2v", True) else "No",
            }

            menu_items = [
                (
                    "Number of Frames",
                    lambda s: self._set_video_frames(s, video_config, "num_frames", "Number of frames to train on:"),
                ),
                (
                    "Minimum Frames",
                    lambda s: self._set_video_frames(s, video_config, "min_frames", "Minimum video length (frames):"),
                ),
                (
                    "Maximum Frames",
                    lambda s: self._set_video_frames(
                        s,
                        video_config,
                        "max_frames",
                        "Maximum video length (frames, 0=unlimited):",
                    ),
                ),
                ("I2V Training", lambda s: self._toggle_i2v(s, video_config)),
            ]

            selected = nav.show_menu("Video-Specific Settings", menu_items, current_values)

            if selected == -1 or selected == -2:
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_conditioning_dataset(self, stdscr, dataset):
        """Configure a conditioning dataset"""
        # Set conditioning type
        cond_type_idx = self.show_options(stdscr, "Select conditioning type:", ["ControlNet", "Mask"], 0)

        if cond_type_idx < 0:
            return

        dataset["conditioning_type"] = "controlnet" if cond_type_idx == 0 else "mask"

        # Use base image dataset configuration
        self._configure_image_dataset(stdscr, dataset)

    def _configure_auto_conditioning(self, stdscr, dataset):
        """Configure automatic conditioning generation"""
        if "conditioning" not in dataset:
            dataset["conditioning"] = []

        nav = MenuNavigator(stdscr)

        while True:
            current_values = {"Conditioning Types": f"{len(dataset['conditioning'])} configured"}

            menu_items = [
                (
                    "Add Conditioning Type",
                    lambda s: self._add_conditioning_type(s, dataset),
                ),
                (
                    "Remove Conditioning Type",
                    lambda s: self._remove_conditioning_type(s, dataset),
                ),
                (
                    "Review Conditioning",
                    lambda s: self._review_conditioning(s, dataset),
                ),
            ]

            selected = nav.show_menu("Auto-Generate Conditioning", menu_items, current_values)

            if selected == -1 or selected == -2:
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _add_conditioning_type(self, stdscr, dataset):
        """Add a conditioning type"""
        cond_types = [
            ("superresolution", "Generate low-quality versions for super-resolution"),
            ("jpeg_artifacts", "Add JPEG compression artifacts"),
            ("depth_midas", "Generate depth maps"),
            ("random_masks", "Create random masks for inpainting"),
            ("canny", "Generate Canny edge maps"),
        ]

        # Show conditioning types
        type_idx = self.show_options(
            stdscr,
            "Select conditioning type to generate:",
            [f"{t[0]} - {t[1]}" for t in cond_types],
            0,
        )

        if type_idx < 0:
            return

        cond_type = cond_types[type_idx][0]
        cond_config = {"type": cond_type}

        # Configure type-specific parameters
        if cond_type == "superresolution":
            self._configure_superresolution_params(stdscr, cond_config)
        elif cond_type == "jpeg_artifacts":
            self._configure_jpeg_params(stdscr, cond_config)
        elif cond_type == "depth_midas":
            self._configure_depth_params(stdscr, cond_config)
        elif cond_type == "random_masks":
            self._configure_mask_params(stdscr, cond_config)
        elif cond_type == "canny":
            self._configure_canny_params(stdscr, cond_config)

        # Configure captions
        caption_idx = self.show_options(
            stdscr,
            "Caption strategy for generated conditioning:",
            ["Use source captions", "No captions", "Single caption", "Random captions"],
            0,
        )

        if caption_idx == 1:
            cond_config["captions"] = False
        elif caption_idx == 2:
            caption = self.get_input(
                stdscr,
                "Enter caption for all conditioning images:",
                "conditioning image",
            )
            cond_config["captions"] = caption
        elif caption_idx == 3:
            num_captions = self.get_input(stdscr, "How many random captions?", "5")
            captions = []
            for i in range(int(num_captions)):
                caption = self.get_input(stdscr, f"Enter caption {i+1}:", f"conditioning variant {i+1}")
                captions.append(caption)
            cond_config["captions"] = captions

        dataset["conditioning"].append(cond_config)

    def _configure_local_paths(self, stdscr, dataset):
        """Configure local filesystem paths"""
        dataset["instance_data_dir"] = self.get_input(
            stdscr,
            "Enter path to image directory:",
            dataset.get("instance_data_dir", "/path/to/images"),
        )

        dataset["cache_dir_vae"] = self.get_input(
            stdscr,
            "Enter VAE cache directory:",
            dataset.get("cache_dir_vae", "cache/vae"),
        )

    def _configure_aws_settings(self, stdscr, dataset):
        """Configure AWS S3 settings"""
        nav = MenuNavigator(stdscr)

        while True:
            current_values = {
                "Bucket": dataset.get("aws_bucket_name", "Not set"),
                "Prefix": dataset.get("aws_data_prefix", ""),
                "Region": dataset.get("aws_region_name", "us-east-1"),
                "Endpoint": dataset.get("aws_endpoint_url", "Default"),
            }

            menu_items = [
                ("S3 Bucket Name", lambda s: self._set_aws_bucket(s, dataset)),
                ("S3 Prefix", lambda s: self._set_aws_prefix(s, dataset)),
                ("AWS Region", lambda s: self._set_aws_region(s, dataset)),
                ("Custom Endpoint", lambda s: self._set_aws_endpoint(s, dataset)),
                ("Access Credentials", lambda s: self._set_aws_credentials(s, dataset)),
            ]

            selected = nav.show_menu("AWS S3 Configuration", menu_items, current_values)

            if selected == -1 or selected == -2:
                return
            elif selected >= 0:
                menu_items[selected][1](stdscr)

    def _configure_csv_settings(self, stdscr, dataset):
        """Configure CSV dataset settings"""
        dataset["csv_file"] = self.get_input(stdscr, "Enter path to CSV file:", dataset.get("csv_file", "dataset.csv"))

        dataset["csv_caption_column"] = self.get_input(
            stdscr,
            "Caption column name in CSV:",
            dataset.get("csv_caption_column", "caption"),
        )

        dataset["csv_cache_dir"] = self.get_input(
            stdscr,
            "Cache directory for downloaded images:",
            dataset.get("csv_cache_dir", "cache/csv"),
        )

        # Force caption strategy
        dataset["caption_strategy"] = "csv"

    def _configure_hf_settings(self, stdscr, dataset):
        """Configure Hugging Face dataset settings"""
        dataset["dataset_name"] = self.get_input(
            stdscr,
            "Hugging Face dataset name (e.g., 'username/dataset-name'):",
            dataset.get("dataset_name", ""),
        )

        dataset["image_column"] = self.get_input(stdscr, "Image column name:", dataset.get("image_column", "image"))

        dataset["caption_column"] = self.get_input(stdscr, "Caption column name:", dataset.get("caption_column", "caption"))

        # Optional subset
        subset = self.get_input(
            stdscr,
            "Dataset subset/config (leave empty for default):",
            dataset.get("subset", ""),
        )
        if subset:
            dataset["subset"] = subset

        # Force strategies
        dataset["caption_strategy"] = "huggingface"
        dataset["metadata_backend"] = "huggingface"

    # Resolution and sizing helpers
    def _set_resolution(self, stdscr, dataset):
        """Set dataset resolution"""
        res_type = dataset.get("resolution_type", "pixel_area")

        if res_type == "area":
            prompt = "Resolution in megapixels (e.g., 1.0 for ~1024x1024):"
            default = "1.0"
        elif res_type == "pixel_area":
            prompt = "Resolution in pixels (e.g., 1024 for ~1024x1024):"
            default = "1024"
        else:
            prompt = "Resolution for shorter edge in pixels:"
            default = "1024"

        value = self.get_input(stdscr, prompt, str(dataset.get("resolution", default)))

        try:
            if res_type == "area":
                dataset["resolution"] = float(value)
            else:
                dataset["resolution"] = int(value)
        except ValueError:
            pass

    def _set_resolution_type(self, stdscr, dataset):
        """Set resolution measurement type"""
        types = ["pixel_area", "area", "pixel"]
        descriptions = [
            "pixel_area - Total pixel count (1024 = ~1024x1024)",
            "area - Megapixels (1.0 = ~1024x1024)",
            "pixel - Shorter edge length",
        ]

        current = dataset.get("resolution_type", "pixel_area")
        default_idx = types.index(current) if current in types else 0

        idx = self.show_options(stdscr, "Resolution measurement type:", descriptions, default_idx)

        if idx >= 0:
            dataset["resolution_type"] = types[idx]

    # Caption strategy helpers
    def _set_caption_strategy(self, stdscr, dataset):
        """Set caption strategy"""
        strategies = ["textfile", "filename", "instanceprompt", "parquet"]

        if dataset.get("type") == "csv":
            strategies = ["csv"]
        elif dataset.get("type") == "huggingface":
            strategies = ["huggingface"]

        descriptions = {
            "textfile": "Text files next to images",
            "filename": "Use cleaned filenames",
            "instanceprompt": "Single prompt for all",
            "parquet": "From parquet/JSONL file",
            "csv": "From CSV file",
            "huggingface": "From HF dataset",
        }

        current = dataset.get("caption_strategy", "textfile")
        options = [f"{s} - {descriptions.get(s, s)}" for s in strategies]
        default_idx = strategies.index(current) if current in strategies else 0

        idx = self.show_options(stdscr, "Caption strategy:", options, default_idx)

        if idx >= 0:
            dataset["caption_strategy"] = strategies[idx]

    # Crop configuration helpers
    def _set_crop_style(self, stdscr, dataset):
        """Set crop style"""
        if not dataset.get("crop", False):
            self.show_message(stdscr, "Enable cropping first!")
            return

        styles = ["center", "random", "corner", "face"]
        current = dataset.get("crop_style", "center")
        default_idx = styles.index(current) if current in styles else 0

        idx = self.show_options(stdscr, "Crop style:", styles, default_idx)

        if idx >= 0:
            dataset["crop_style"] = styles[idx]

    def _set_crop_aspect(self, stdscr, dataset):
        """Set crop aspect"""
        if not dataset.get("crop", False):
            self.show_message(stdscr, "Enable cropping first!")
            return

        aspects = ["square", "preserve", "closest", "random"]
        descriptions = [
            "square - Always 1:1",
            "preserve - Keep original aspect",
            "closest - Match nearest bucket",
            "random - Random from buckets",
        ]

        current = dataset.get("crop_aspect", "square")
        default_idx = aspects.index(current) if current in aspects else 0

        idx = self.show_options(stdscr, "Crop aspect:", descriptions, default_idx)

        if idx >= 0:
            dataset["crop_aspect"] = aspects[idx]

    def _set_aspect_buckets(self, stdscr, dataset):
        """Set custom aspect ratio buckets"""
        current = dataset.get("crop_aspect_buckets", [])
        current_str = ", ".join(map(str, current)) if current else "0.5, 0.75, 1.0, 1.33, 1.5, 2.0"

        value = self.get_input(stdscr, "Aspect ratio buckets (comma-separated):", current_str)

        try:
            buckets = [float(x.strip()) for x in value.split(",")]
            dataset["crop_aspect_buckets"] = sorted(buckets)
        except ValueError:
            self.show_error(stdscr, "Invalid bucket values")

    # AWS helpers
    def _set_aws_bucket(self, stdscr, dataset):
        """Set AWS bucket name"""
        dataset["aws_bucket_name"] = self.get_input(stdscr, "S3 bucket name:", dataset.get("aws_bucket_name", "my-bucket"))

    def _set_aws_credentials(self, stdscr, dataset):
        """Set AWS credentials"""
        use_env = self.show_options(stdscr, "AWS credentials source:", ["Environment/IAM", "Specify here"], 0)

        if use_env == 0:
            # Clear any stored credentials
            for key in ["aws_access_key_id", "aws_secret_access_key"]:
                if key in dataset:
                    del dataset[key]
        else:
            dataset["aws_access_key_id"] = self.get_input(stdscr, "AWS Access Key ID:", dataset.get("aws_access_key_id", ""))
            dataset["aws_secret_access_key"] = self.get_input(
                stdscr,
                "AWS Secret Access Key:",
                dataset.get("aws_secret_access_key", ""),
            )

    # Review and edit helpers
    def _review_datasets(self, stdscr):
        """Review all configured datasets"""
        if not self._datasets:
            self.show_message(stdscr, "No datasets configured yet!")
            return

        stdscr.clear()
        h, w = stdscr.getmaxyx()

        stdscr.addstr(1, 2, "Configured Datasets", curses.A_BOLD)

        y = 3
        for i, dataset in enumerate(self._datasets):
            if y + 3 >= h - 2:
                stdscr.addstr(h - 2, 2, "Press any key for more...")
                stdscr.getch()
                stdscr.clear()
                stdscr.addstr(1, 2, "Configured Datasets (continued)", curses.A_BOLD)
                y = 3

            stdscr.addstr(
                y,
                2,
                f"[{i+1}] {dataset.get('id', 'unnamed')} ({dataset.get('dataset_type', 'image')})",
            )
            stdscr.addstr(
                y + 1,
                4,
                f"Type: {dataset.get('type', 'local')}, " f"Resolution: {dataset.get('resolution', 'N/A')}",
            )
            if dataset.get("disabled", False):
                stdscr.addstr(y + 1, w - 20, "[DISABLED]", curses.A_DIM)
            y += 3

        stdscr.addstr(h - 2, 2, "Press any key to continue...")
        stdscr.getch()

    def _edit_dataset(self, stdscr):
        """Edit an existing dataset"""
        if not self._datasets:
            self.show_message(stdscr, "No datasets to edit!")
            return

        # Show dataset list
        options = []
        for i, ds in enumerate(self._datasets):
            options.append(f"{ds.get('id', 'unnamed')} ({ds.get('dataset_type', 'image')}, {ds.get('type', 'local')})")

        idx = self.show_options(stdscr, "Select dataset to edit:", options, 0)

        if idx >= 0:
            dataset = self._datasets[idx]

            if dataset.get("dataset_type") == "image":
                self._configure_image_dataset(stdscr, dataset)
            elif dataset.get("dataset_type") == "video":
                self._configure_video_dataset(stdscr, dataset)
            elif dataset.get("dataset_type") == "conditioning":
                self._configure_conditioning_dataset(stdscr, dataset)
            elif dataset.get("dataset_type") == "text_embeds":
                # Inline edit for text embeds
                self._configure_text_embeds(stdscr)
                self._datasets.pop()  # Remove the newly added one
            elif dataset.get("dataset_type") == "image_embeds":
                # Inline edit for image embeds
                self._configure_image_embeds(stdscr)
                self._datasets.pop()  # Remove the newly added one

    def _remove_dataset(self, stdscr):
        """Remove a dataset"""
        if not self._datasets:
            self.show_message(stdscr, "No datasets to remove!")
            return

        # Show dataset list
        options = []
        for i, ds in enumerate(self._datasets):
            options.append(f"{ds.get('id', 'unnamed')} ({ds.get('dataset_type', 'image')})")

        idx = self.show_options(stdscr, "Select dataset to remove:", options, 0)

        if idx >= 0:
            removed = self._datasets.pop(idx)
            self.show_message(stdscr, f"Removed dataset: {removed.get('id', 'unnamed')}")


def main():
    """Main entry point"""
    import sys

    # Check for command line arguments
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)

    configurator = SimpleTunerNCurses()

    # Load specified config or show startup message
    if config_path:
        if not configurator.state.load_from_file(config_path):
            print(f"Error: Failed to load config: {config_path}")
            sys.exit(1)
        print(f"Loaded configuration from: {config_path}")
    elif configurator.state.loaded_config_path:
        print(f"Loaded existing configuration from: {configurator.state.loaded_config_path}")
    else:
        print("No existing configuration found. Starting fresh setup.")

    configurator.run()


if __name__ == "__main__":
    main()
