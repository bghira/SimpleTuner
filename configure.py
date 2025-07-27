#!/usr/bin/env python3
import os
import sys
import json
import curses
import textwrap
from typing import Dict, Any, List, Tuple, Optional
import traceback
import huggingface_hub
import torch
from helpers.training import quantised_precision_levels, lycoris_defaults
from helpers.training.optimizer_param import optimizer_choices

# Constants
bf16_only_optims = [
    key
    for key, value in optimizer_choices.items()
    if value.get("precision", "any") == "bf16"
]
any_precision_optims = [
    key
    for key, value in optimizer_choices.items()
    if value.get("precision", "any") == "any"
]

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
            if "lycoris_config" in loaded_config and os.path.exists(
                loaded_config["lycoris_config"]
            ):
                with open(loaded_config["lycoris_config"], "r", encoding="utf-8") as f:
                    self.lycoris_config = json.load(f)

            # Load dataset config if specified
            backend_config = loaded_config.get(
                "data_backend_config", "config/multidatabackend.json"
            )
            if os.path.exists(backend_config):
                with open(backend_config, "r", encoding="utf-8") as f:
                    self.dataset_config = json.load(f)

            self.loaded_config_path = config_path

            # Mark steps as completed based on what's configured
            if "output_dir" in loaded_config:
                self.completed_steps.add(0)  # Basic setup
            if "model_type" in loaded_config:
                self.completed_steps.add(1)  # Model type
            if (
                "max_train_steps" in loaded_config
                or "num_train_epochs" in loaded_config
            ):
                self.completed_steps.add(2)  # Training config
            if "model_family" in loaded_config:
                self.completed_steps.add(4)  # Model selection
            if "train_batch_size" in loaded_config:
                self.completed_steps.add(5)  # Training params
            if "optimizer" in loaded_config:
                self.completed_steps.add(6)  # Optimization
            if "validation_prompt" in loaded_config:
                self.completed_steps.add(7)  # Validation

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
            self.stdscr.addstr(
                3, 2, "↑/↓: Navigate  Enter: Select  ←/Backspace: Back  q: Quit"
            )
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
                    if len(value_text) > max_value_len:
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
            ("Hugging Face Hub", self.hub_setup),
            ("Model Selection", self.model_selection),
            ("Training Parameters", self.training_params),
            ("Optimization Settings", self.optimization_settings),
            ("Validation Settings", self.validation_settings),
            ("Advanced Options", self.advanced_options),
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
                        self.show_error(
                            stdscr, f"Error in {self.menu_items[action][0]}: {str(e)}"
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

        while True:
            for idx, (item_name, _) in enumerate(self.menu_items):
                y = start_y + idx
                if y >= h - 2:
                    break

                # Highlight current selection
                attr = curses.A_REVERSE if idx == selected else curses.A_NORMAL

                # Mark completed steps
                prefix = "[✓] " if idx in self.state.completed_steps else "[ ] "
                text = f"{prefix}{idx + 1}. {item_name}"

                # Ensure text fits
                if len(text) > w - 4:
                    text = text[: w - 7] + "..."

                stdscr.addstr(y, 2, text, attr)

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
            elif key == curses.KEY_DOWN and selected < len(self.menu_items) - 1:
                selected += 1
            elif key in [curses.KEY_ENTER, ord("\n"), ord("\r")]:
                return selected

    def show_error(self, stdscr, error_msg: str):
        """Display an error message and wait for acknowledgment"""
        h, w = stdscr.getmaxyx()

        # Create error window
        error_lines = textwrap.wrap(error_msg, w - 10)
        error_h = len(error_lines) + 4
        error_w = min(80, w - 4)

        error_win = curses.newwin(
            error_h, error_w, (h - error_h) // 2, (w - error_w) // 2
        )
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

    def show_options(
        self, stdscr, prompt: str, options: List[str], default: int = 0
    ) -> int:
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
            config_path = self.get_input(
                stdscr, "Enter path to config.json:", "config/config.json"
            )
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
                "Output Directory": self.state.env_contents.get(
                    "output_dir", "output/models"
                ),
                "Resume from Checkpoint": self.state.env_contents.get(
                    "resume_from_checkpoint", "latest"
                ),
                "Aspect Bucket Rounding": str(
                    self.state.env_contents.get("aspect_bucket_rounding", 2)
                ),
                "Seed": str(self.state.env_contents.get("seed", 42)),
                "Minimum Image Size": str(
                    self.state.env_contents.get("minimum_image_size", 0)
                ),
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
        value = self.get_input(
            stdscr, "Set aspect bucket rounding (1-8, higher = more precise):", current
        )
        try:
            self.state.env_contents["aspect_bucket_rounding"] = int(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_seed(self, stdscr):
        """Configure training seed"""
        current = str(self.state.env_contents.get("seed", 42))
        value = self.get_input(
            stdscr, "Set training seed (for reproducibility):", current
        )
        try:
            self.state.env_contents["seed"] = int(value)
        except ValueError:
            self.show_error(stdscr, "Invalid value. Keeping current setting.")

    def _configure_min_image_size(self, stdscr):
        """Configure minimum image size"""
        current = str(self.state.env_contents.get("minimum_image_size", 0))
        value = self.get_input(
            stdscr, "Set minimum image size (0 to disable filtering):", current
        )
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
                current_values["LoRA Type"] = (
                    "LyCORIS" if lora_type == "lycoris" else "Standard"
                )
                menu_items.append(("LoRA Type", self._configure_lora_type))

                if lora_type == "lycoris":
                    current_values["LyCORIS Algorithm"] = (
                        "Configured" if self.state.lycoris_config else "Not configured"
                    )
                    menu_items.append(("LyCORIS Algorithm", self.configure_lycoris))
                else:
                    # Standard LoRA options
                    use_dora = (
                        self.state.env_contents.get("use_dora", "false") == "true"
                    )
                    current_values["DoRA"] = "Enabled" if use_dora else "Disabled"
                    current_values["LoRA Rank"] = str(
                        self.state.env_contents.get("lora_rank", 64)
                    )
                    menu_items.append(("DoRA", self._configure_dora))
                    menu_items.append(("LoRA Rank", self._configure_lora_rank))
            else:
                # Full fine-tuning options
                use_ema = self.state.env_contents.get("use_ema", "false") == "true"
                current_values["EMA"] = "Enabled" if use_ema else "Disabled"
                menu_items.append(("EMA", self._configure_ema))

            selected = nav.show_menu(
                "Model Type & LoRA/LyCORIS Configuration", menu_items, current_values
            )

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
                current_values["Constraint"] = (
                    "Yes" if config.get("constraint", False) else "No"
                )
                current_values["Rescaled"] = (
                    "Yes" if config.get("rescaled", False) else "No"
                )
                menu_items.append(("Constraint", self._configure_lycoris_constraint))
                menu_items.append(("Rescaled", self._configure_lycoris_rescaled))

            menu_items.append(("Save Configuration", self._save_lycoris_config))

            selected = nav.show_menu(
                "LyCORIS Configuration", menu_items, current_values
            )

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

            if (
                "max_train_steps" in self.state.env_contents
                and self.state.env_contents["max_train_steps"] > 0
            ):
                current_values["Training Duration"] = (
                    f"{self.state.env_contents['--max_train_steps']} steps"
                )
            elif (
                "num_train_epochs" in self.state.env_contents
                and self.state.env_contents["num_train_epochs"] > 0
            ):
                current_values["Training Duration"] = (
                    f"{self.state.env_contents['--num_train_epochs']} epochs"
                )
            else:
                current_values["Training Duration"] = "Not configured"

            current_values["Checkpointing Interval"] = (
                f"{self.state.env_contents.get('--checkpointing_steps', 500)} steps"
            )
            current_values["Checkpoints to Keep"] = str(
                self.state.env_contents.get("checkpoints_total_limit", 5)
            )

            menu_items = [
                ("Training Duration", self._configure_training_duration),
                ("Checkpointing Interval", self._configure_checkpoint_interval),
                ("Checkpoints to Keep", self._configure_checkpoint_limit),
            ]

            selected = nav.show_menu(
                "Training Configuration", menu_items, current_values
            )

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
            max_steps = self.get_input(
                stdscr, "Set the maximum number of steps:", "10000"
            )
            try:
                self.state.env_contents["max_train_steps"] = int(max_steps)
                self.state.env_contents["num_train_epochs"] = 0
            except ValueError:
                self.state.env_contents["max_train_steps"] = 10000
                self.state.env_contents["num_train_epochs"] = 0
        else:
            max_epochs = self.get_input(
                stdscr, "Set the maximum number of epochs:", "100"
            )
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

        checkpoint_interval = self.get_input(
            stdscr, "Set the checkpointing interval (in steps):", str(default_interval)
        )

        try:
            self.state.env_contents["checkpointing_steps"] = int(checkpoint_interval)
        except ValueError:
            self.state.env_contents["checkpointing_steps"] = default_interval

    def _configure_checkpoint_limit(self, stdscr):
        """Configure checkpoint limit"""
        checkpoint_limit = self.get_input(
            stdscr, "How many checkpoints do you want to keep?", "5"
        )

        try:
            self.state.env_contents["checkpoints_total_limit"] = int(checkpoint_limit)
        except ValueError:
            self.state.env_contents["checkpoints_total_limit"] = 5

    def hub_setup(self, stdscr):
        """Step 4: Hugging Face Hub Setup - Now with menu"""
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
                stdscr.addstr(
                    2, 2, "Please login to Hugging Face Hub in your terminal..."
                )
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
                "Push to Hub": (
                    "Yes"
                    if self.state.env_contents.get("push_to_hub", "false") == "true"
                    else "No"
                ),
            }

            menu_items = [
                ("Push to Hub", self._configure_push_to_hub),
            ]

            if self.state.env_contents.get("push_to_hub", "false") == "true":
                current_values["Model ID"] = self.state.env_contents.get(
                    "hub_model_id", f"simpletuner-{self.state.model_type}"
                )
                current_values["Push Checkpoints"] = (
                    "Yes"
                    if self.state.env_contents.get("push_checkpoints_to_hub", "false")
                    == "true"
                    else "No"
                )
                current_values["Safe for Work"] = (
                    "Yes"
                    if self.state.env_contents.get("model_card_safe_for_work", "false")
                    == "true"
                    else "No"
                )

                menu_items.extend(
                    [
                        ("Model ID", self._configure_model_id),
                        ("Push Checkpoints", self._configure_push_checkpoints),
                        ("Safe for Work", self._configure_sfw),
                    ]
                )

            selected = nav.show_menu(
                "Hugging Face Hub Configuration", menu_items, current_values
            )

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
        current_model_id = self.state.env_contents.get(
            "hub_model_id", f"simpletuner-{self.state.model_type}"
        )

        model_id = self.get_input(
            stdscr,
            f"Model name (will be accessible as https://huggingface.co/{self.state.whoami['name']}/...):\n(Current: {current_model_id})",
            current_model_id,
        )

        self.state.env_contents["hub_model_id"] = model_id

    def _configure_push_checkpoints(self, stdscr):
        """Configure push checkpoints"""
        current_push_ckpt = (
            self.state.env_contents.get("push_checkpoints_to_hub", "false") == "true"
        )

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
        current_sfw = (
            self.state.env_contents.get("model_card_safe_for_work", "false") == "true"
        )

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
        """Step 5: Model Selection - Now with menu"""
        nav = MenuNavigator(stdscr)

        while True:
            # Get current values
            current_values = {
                "Model Family": self.state.env_contents.get(
                    "model_family", "Not selected"
                ),
                "Model Name": self.state.env_contents.get(
                    "pretrained_model_name_or_path", "Not selected"
                ),
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
                current_values["Flux LoRA Target"] = self.state.env_contents.get(
                    "flux_lora_target", "all"
                )
                menu_items.append(("Flux LoRA Target", self._configure_flux_target))

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
        current_model = self.state.env_contents.get(
            "pretrained_model_name_or_path", default_model
        )

        while True:
            model_name = self.get_input(
                stdscr, "Enter the model name from Hugging Face Hub:", current_model
            )

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

        target_idx = self.show_options(
            stdscr, "Set Flux target layers:", flux_targets, default_idx
        )

        if target_idx >= 0:
            self.state.env_contents["flux_lora_target"] = flux_targets[target_idx]

    def training_params(self, stdscr):
        """Step 6: Training Parameters - Now with menu"""
        nav = MenuNavigator(stdscr)

        while True:
            # Get current values
            current_values = {
                "Batch Size": str(self.state.env_contents.get("train_batch_size", 1)),
                "Gradient Checkpointing": (
                    "Enabled"
                    if self.state.env_contents.get("gradient_checkpointing", "true")
                    == "true"
                    else "Disabled"
                ),
                "Caption Dropout": str(
                    self.state.env_contents.get("caption_dropout_probability", 0.1)
                ),
                "Resolution Type": self.state.env_contents.get(
                    "resolution_type", "pixel_area"
                ),
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
            ]:
                gc_interval = self.state.env_contents.get(
                    "gradient_checkpointing_interval", 0
                )
                current_values["GC Interval"] = (
                    str(gc_interval) if gc_interval > 0 else "Disabled"
                )
                menu_items.insert(2, ("GC Interval", self._configure_gc_interval))

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
            "Set the training batch size.\n"
            "Larger values will require larger datasets, more VRAM, and slow things down.",
            current,
        )

        try:
            self.state.env_contents["train_batch_size"] = int(batch_size)
        except ValueError:
            self.state.env_contents["train_batch_size"] = 1

    def _configure_gradient_checkpointing(self, stdscr):
        """Configure gradient checkpointing"""
        current = (
            self.state.env_contents.get("gradient_checkpointing", "true") == "true"
        )

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
        default_dropout = (
            "0.05" if any([self.state.use_lora, self.state.use_lycoris]) else "0.1"
        )
        current = str(
            self.state.env_contents.get("caption_dropout_probability", default_dropout)
        )

        caption_dropout = self.get_input(
            stdscr,
            "Set the caption dropout rate, or use 0.0 to disable it.\n"
            "Dropout might be a good idea to disable for Flux training,\n"
            "but experimentation is warranted.",
            current,
        )

        try:
            self.state.env_contents["caption_dropout_probability"] = float(
                caption_dropout
            )
        except ValueError:
            self.state.env_contents["caption_dropout_probability"] = float(
                default_dropout
            )

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

    def optimization_settings(self, stdscr):
        """Step 7: Optimization Settings - Now with menu"""
        nav = MenuNavigator(stdscr)

        while True:
            # Get current values
            current_values = {
                "Mixed Precision": self.state.env_contents.get(
                    "mixed_precision", "bf16"
                ),
                "Optimizer": self.state.env_contents.get("optimizer", "adamw_bf16"),
                "Learning Rate": self.state.env_contents.get("learning_rate", "1e-4"),
                "LR Scheduler": self.state.env_contents.get(
                    "lr_scheduler", "polynomial"
                ),
                "Warmup Steps": str(
                    self.state.env_contents.get("lr_warmup_steps", 100)
                ),
            }

            menu_items = [
                ("Mixed Precision", self._configure_mixed_precision),
                ("Optimizer", self._configure_optimizer),
                ("Learning Rate", self._configure_learning_rate),
                ("LR Scheduler", self._configure_lr_scheduler),
                ("Warmup Steps", self._configure_warmup_steps),
            ]

            # Add TF32 option if CUDA available
            if torch.cuda.is_available():
                tf32_disabled = "disable_tf32" in self.state.env_contents
                current_values["TF32"] = "Disabled" if tf32_disabled else "Enabled"
                menu_items.insert(0, ("TF32", self._configure_tf32))

            # Add quantization option
            if "base_model_precision" in self.state.env_contents:
                current_values["Quantization"] = self.state.env_contents[
                    "base_model_precision"
                ]
            else:
                current_values["Quantization"] = "Disabled"
            menu_items.append(("Quantization", self._configure_quantization))

            selected = nav.show_menu(
                "Optimization Settings", menu_items, current_values
            )

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
            self.state.env_contents["mixed_precision"] = precision_values[
                mixed_precision_idx
            ]

    def _configure_optimizer(self, stdscr):
        """Configure optimizer"""
        # Get compatible optimizers based on precision
        if self.state.env_contents.get("mixed_precision") == "bf16":
            compatible_optims = bf16_only_optims + any_precision_optims
        else:
            compatible_optims = any_precision_optims

        current_optimizer = self.state.env_contents.get(
            "optimizer", compatible_optims[0]
        )
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
            f"Set the learning rate:\n"
            f"(Current: {current_lr}, Suggested for your config: {default_lr})",
            current_lr,
        )

        self.state.env_contents["learning_rate"] = lr

    def _configure_lr_scheduler(self, stdscr):
        """Configure learning rate scheduler"""
        lr_schedulers = ["polynomial", "constant"]
        current_scheduler = self.state.env_contents.get("lr_scheduler", "polynomial")
        default_sched_idx = 0 if current_scheduler == "polynomial" else 1

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
                self.state.extra_args = [
                    arg for arg in self.state.extra_args if not arg.startswith("lr_end")
                ]

    def _configure_warmup_steps(self, stdscr):
        """Configure warmup steps"""
        # Dynamic default
        default_warmup = "100"
        if self.state.env_contents.get("max_train_steps", 0) > 0:
            calculated_warmup = max(
                100, int(self.state.env_contents["max_train_steps"]) // 10
            )
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
                stdscr.addstr(
                    2, 2, "Note: DoRA will be disabled for quantisation.", curses.A_BOLD
                )
                stdscr.addstr(4, 2, "Press any key to continue...")
                stdscr.refresh()
                stdscr.getch()
                del self.state.env_contents["use_dora"]

            # Get quantization type
            current_quant_type = self.state.env_contents.get(
                "base_model_precision", "int8-quanto"
            )
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
                self.state.env_contents["base_model_precision"] = quant_types[
                    quant_type_idx
                ]
        else:
            # Remove quantization if disabled
            if "base_model_precision" in self.state.env_contents:
                del self.state.env_contents["base_model_precision"]

    def validation_settings(self, stdscr):
        """Step 8: Validation Settings - Now with menu"""
        nav = MenuNavigator(stdscr)

        while True:
            # Get current values
            current_values = {
                "Validation Seed": self.state.env_contents.get("validation_seed", "42"),
                "Validation Steps": self.state.env_contents.get(
                    "validation_steps",
                    str(self.state.env_contents.get("checkpointing_steps", 500)),
                ),
                "Validation Resolution": self.state.env_contents.get(
                    "validation_resolution", "1024x1024"
                ),
                "Guidance Scale": self.state.env_contents.get(
                    "validation_guidance", "3.0"
                ),
                "Guidance Rescale": self.state.env_contents.get(
                    "validation_guidance_rescale", "0.0"
                ),
                "Inference Steps": self.state.env_contents.get(
                    "validation_num_inference_steps", "20"
                ),
                "Validation Prompt": self.state.env_contents.get(
                    "validation_prompt", "A photo-realistic image of a cat"
                )[:40]
                + "...",
            }

            menu_items = [
                ("Validation Seed", self._configure_val_seed),
                ("Validation Steps", self._configure_val_steps),
                ("Validation Resolution", self._configure_val_resolution),
                ("Guidance Scale", self._configure_val_guidance),
                ("Guidance Rescale", self._configure_val_rescale),
                ("Inference Steps", self._configure_val_inference_steps),
                ("Validation Prompt", self._configure_val_prompt),
            ]

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
        val_steps = self.get_input(
            stdscr, "How many steps between validation outputs?", current
        )
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

        val_guidance = self.get_input(
            stdscr, "Set guidance scale for validation:", current
        )
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
        val_inf_steps = self.get_input(
            stdscr, "Set number of inference steps for validation:", current
        )
        self.state.env_contents["validation_num_inference_steps"] = val_inf_steps

    def _configure_val_prompt(self, stdscr):
        """Configure validation prompt"""
        current = self.state.env_contents.get(
            "validation_prompt", "A photo-realistic image of a cat"
        )
        val_prompt = self.get_input(stdscr, "Set the validation prompt:", current)
        self.state.env_contents["validation_prompt"] = val_prompt

    def advanced_options(self, stdscr):
        """Step 9: Advanced Options - Now with menu"""
        nav = MenuNavigator(stdscr)

        while True:
            # Get current values
            current_values = {
                "Tracking": self.state.env_contents.get("report_to", "none"),
                "SageAttention": self.state.env_contents.get(
                    "attention_mechanism", "diffusers"
                ),
                "Disk Cache Compression": (
                    "Enabled"
                    if "compress_disk_cache" in self.state.extra_args
                    else "Disabled"
                ),
                "Torch Compile": (
                    "Enabled"
                    if self.state.env_contents.get("validation_torch_compile", "false")
                    == "true"
                    else "Disabled"
                ),
                "Prompt Library": (
                    "Configured"
                    if "user_prompt_library" in self.state.env_contents
                    else "Not configured"
                ),
            }

            menu_items = [
                ("Tracking (W&B/TensorBoard)", self._configure_tracking),
                ("SageAttention", self._configure_sageattention),
                ("Disk Cache Compression", self._configure_disk_compression),
                ("Torch Compile", self._configure_torch_compile),
                ("Prompt Library", self._configure_prompt_library),
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

            selected = nav.show_menu(
                "Tracking Configuration", menu_items, current_values
            )

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

        selected = self.show_options(
            stdscr, "Select tracking services:", options, default_idx
        )

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
        current = self.state.env_contents.get(
            "tracker_project_name", f"{self.state.model_type}-training"
        )
        project_name = self.get_input(
            stdscr,
            "Enter the name of your Weights & Biases project:",
            current,
        )
        self.state.env_contents["tracker_project_name"] = project_name

    def _configure_run_name(self, stdscr):
        """Configure tracking run name"""
        current = self.state.env_contents.get(
            "tracker_run_name", f"simpletuner-{self.state.model_type}"
        )
        run_name = self.get_input(
            stdscr,
            "Enter the name of your Weights & Biases runs.\n"
            "This can use shell commands, which can be used to dynamically set the run name.",
            current,
        )
        self.state.env_contents["tracker_run_name"] = run_name

    def _configure_sageattention(self, stdscr):
        """Configure SageAttention"""
        current_mechanism = self.state.env_contents.get(
            "attention_mechanism", "diffusers"
        )

        sage_idx = self.show_options(
            stdscr,
            f"Would you like to use SageAttention for image validation generation?\n(Current: {current_mechanism})",
            ["No", "Yes"],
            1 if current_mechanism == "sageattention" else 0,
        )

        if sage_idx == 1:
            self.state.env_contents["attention_mechanism"] = "sageattention"

            # Configure usage scope
            current_usage = self.state.env_contents.get(
                "sageattention_usage", "inference"
            )

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

            sage_training_idx = self.show_options(
                stdscr, "", ["No (Inference only)", "Yes (Training + Inference)"], 0
            )

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
            self.state.extra_args = [
                arg for arg in self.state.extra_args if arg != "compress_disk_cache"
            ]

    def _configure_torch_compile(self, stdscr):
        """Configure torch compile"""
        current = (
            self.state.env_contents.get("validation_torch_compile", "false") == "true"
        )

        compile_idx = self.show_options(
            stdscr,
            f"Would you like to use torch compile during validations?\n(Current: {'Yes' if current else 'No'})",
            ["No", "Yes"],
            1 if current else 0,
        )

        self.state.env_contents["validation_torch_compile"] = (
            "true" if compile_idx == 1 else "false"
        )

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

            num_prompts = self.get_input(
                stdscr, "How many prompts would you like to generate?", "8"
            )

            try:
                num_prompts_int = int(num_prompts)
            except ValueError:
                num_prompts_int = 8

            try:
                from helpers.prompt_expander import PromptExpander

                stdscr.clear()
                stdscr.addstr(2, 2, "Initializing Llama 3.2 1B model...")
                stdscr.addstr(4, 2, "This may take a moment on first run...")
                stdscr.refresh()

                PromptExpander.initialize_model()

                stdscr.addstr(6, 2, "Generating prompts...")
                stdscr.refresh()

                user_prompt_library = PromptExpander.generate_prompts(
                    trigger_phrase=trigger, num_prompts=num_prompts_int
                )

                # Save the prompt library
                with open(
                    "config/user_prompt_library.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(user_prompt_library, f, indent=4)

                self.state.env_contents["user_prompt_library"] = (
                    "config/user_prompt_library.json"
                )

                self.show_message(stdscr, "Prompt library generated successfully!")

            except Exception as e:
                self.show_error(
                    stdscr, f"(warning) Failed to generate prompt library: {str(e)}"
                )

    def dataset_config(self, stdscr):
        """Step 10: Dataset Configuration - Now with menu"""
        config_idx = self.show_options(
            stdscr, "Would you like to configure your dataloader?", ["Yes", "No"], 0
        )

        if config_idx == 1:
            return

        nav = MenuNavigator(stdscr)

        # Initialize dataset values if not configured
        if not hasattr(self, "_dataset_values"):
            self._dataset_values = {
                "id": "my-dataset",
                "path": "/datasets/my-dataset",
                "caption_strategy": "textfile",
                "instance_prompt": None,
                "repeats": 10,
                "resolutions": [1024],
                "cache_dir": "cache/",
                "has_large_images": False,
            }

        while True:
            # Build current values display
            current_values = {
                "Dataset ID": self._dataset_values["id"],
                "Dataset Path": self._dataset_values["path"],
                "Caption Strategy": self._dataset_values["caption_strategy"],
                "Dataset Repeats": str(self._dataset_values["repeats"]),
                "Resolutions": ", ".join(map(str, self._dataset_values["resolutions"])),
                "Cache Directory": self._dataset_values["cache_dir"],
                "Large Images": (
                    "Yes" if self._dataset_values["has_large_images"] else "No"
                ),
            }

            if self._dataset_values["caption_strategy"] == "instanceprompt":
                current_values["Instance Prompt"] = self._dataset_values.get(
                    "instance_prompt", "Not set"
                )

            menu_items = [
                ("Dataset ID", self._configure_dataset_id),
                ("Dataset Path", self._configure_dataset_path),
                ("Caption Strategy", self._configure_caption_strategy),
                ("Dataset Repeats", self._configure_dataset_repeats),
                ("Resolutions", self._configure_resolutions),
                ("Cache Directory", self._configure_cache_dir),
                ("Large Images", self._configure_large_images),
                ("Apply Configuration", self._apply_dataset_config),
            ]

            if self._dataset_values["caption_strategy"] == "instanceprompt":
                # Insert instance prompt before Apply
                menu_items.insert(
                    -1, ("Instance Prompt", self._configure_instance_prompt)
                )

            selected = nav.show_menu(
                "Dataset Configuration", menu_items, current_values
            )

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
        stdscr.addstr(
            5, 2, "-> 'filename' will use the names of your image files as the caption"
        )
        stdscr.addstr(
            6,
            2,
            "-> 'textfile' requires a image.txt file to go next to your image.png file",
        )
        stdscr.addstr(
            7, 2, "-> 'instanceprompt' will just use one trigger phrase for all images"
        )
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
                    stdscr.addstr(
                        5, 2, "resulting in degraded outputs and broken hearts."
                    )
                    stdscr.addstr(6, 2, "Proceed with caution.")
                    stdscr.addstr(8, 2, "Press any key to continue...")
                    stdscr.refresh()
                    stdscr.getch()
            else:
                resolutions = [int(resolutions_str)]

            self._dataset_values["resolutions"] = resolutions
        except ValueError:
            self.show_error(
                stdscr, "Invalid resolution value. Keeping current settings."
            )

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
        stdscr.addstr(
            5, 2, f"Caption Strategy: {self._dataset_values['caption_strategy']}"
        )
        stdscr.addstr(6, 2, f"Repeats: {self._dataset_values['repeats']}")
        stdscr.addstr(
            7,
            2,
            f"Resolutions: {', '.join(map(str, self._dataset_values['resolutions']))}",
        )
        stdscr.addstr(8, 2, f"Cache Directory: {self._dataset_values['cache_dir']}")

        if self._dataset_values.get("instance_prompt"):
            stdscr.addstr(
                9, 2, f"Instance Prompt: {self._dataset_values['instance_prompt']}"
            )

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
                "cache_dir": os.path.abspath(
                    os.path.join(
                        cache_dir, self.state.env_contents["model_family"], "text"
                    )
                ),
                "write_batch_size": 128,
            }
        ]

        def create_dataset(resolution, template):
            dataset = template.copy()
            dataset.update(
                resolution_configs.get(resolution, {"resolution": resolution})
            )
            dataset["id"] = (
                f"{dataset_id}-{resolution}"
                if "crop" not in dataset["id"]
                else f"{dataset_id}-crop-{resolution}"
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
        """Step 11: Review and Save Configuration"""
        stdscr.clear()
        h, w = stdscr.getmaxyx()

        # Display configuration summary
        stdscr.addstr(1, 2, "Configuration Summary", curses.A_BOLD)

        y = 3
        config_items = []

        # Prepare summary
        config_items.append(f"Model Type: {self.state.model_type}")
        config_items.append(
            f"Output Directory: {self.state.env_contents.get('--output_dir', 'Not set')}"
        )
        config_items.append(
            f"Model Family: {self.state.env_contents.get('--model_family', 'Not set')}"
        )
        config_items.append(
            f"Base Model: {self.state.env_contents.get('--pretrained_model_name_or_path', 'Not set')}"
        )

        if self.state.use_lora:
            config_items.append(
                f"LoRA Type: {self.state.env_contents.get('--lora_type', 'standard')}"
            )
            if not self.state.use_lycoris:
                config_items.append(
                    f"LoRA Rank: {self.state.env_contents.get('--lora_rank', 'Not set')}"
                )

        # Display items
        for item in config_items:
            if y < h - 4:
                stdscr.addstr(y, 4, item[: w - 6])
                y += 1

        stdscr.addstr(
            h - 3, 2, "Press 's' to save, 'b' to go back, 'q' to quit without saving"
        )
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

                save_choice = self.show_options(
                    stdscr, "Where would you like to save?", save_options, 0
                )

                if save_choice == 0:
                    save_path = self.state.loaded_config_path
                elif save_choice == 1:
                    save_path = self.get_input(
                        stdscr,
                        "Enter save path for config.json:",
                        "config/my-preset/config.json",
                    )
                    # Create directory if needed
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
                backend_path = self.state.env_contents.get(
                    "data_backend_config", "config/multidatabackend.json"
                )
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
                    f"- {self.state.env_contents.get('--data_backend_config', 'config/multidatabackend.json')}",
                )

            if self.state.lycoris_config:
                stdscr.addstr(
                    7,
                    4,
                    f"- {self.state.env_contents.get('--lycoris_config', 'config/lycoris_config.json')}",
                )

            stdscr.addstr(9, 2, "Press any key to continue...")
            stdscr.refresh()
            stdscr.getch()

            # Update loaded path
            self.state.loaded_config_path = save_path

        except Exception as e:
            self.show_error(stdscr, f"Failed to save configuration: {str(e)}")


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
        print(
            f"Loaded existing configuration from: {configurator.state.loaded_config_path}"
        )
    else:
        print("No existing configuration found. Starting fresh setup.")

    print("\nStarting SimpleTuner configuration tool...")
    print("Press any key to continue...")
    input()

    configurator.run()


if __name__ == "__main__":
    main()
