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
    key for key, value in optimizer_choices.items()
    if value.get("precision", "any") == "bf16"
]
any_precision_optims = [
    key for key, value in optimizer_choices.items()
    if value.get("precision", "any") == "any"
]

model_classes = {
    "full": ["flux", "sdxl", "pixart_sigma", "kolors", "sd3", "sd1x", "sd2x",
             "ltxvideo", "wan", "sana", "deepfloyd", "omnigen", "hidream",
             "auraflow", "lumina2", "cosmos2image"],
    "lora": ["flux", "sdxl", "kolors", "sd3", "sd1x", "sd2x", "ltxvideo",
             "wan", "deepfloyd", "auraflow", "hidream", "lumina2"],
    "controlnet": ["sdxl", "sd1x", "sd2x", "hidream", "auraflow", "flux",
                   "pixart_sigma", "sd3", "kolors"],
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
    "flux": 3.0, "sdxl": 4.2, "pixart_sigma": 3.4, "kolors": 5.0,
    "terminus": 8.0, "sd3": 5.0, "ltxvideo": 4.0, "hidream": 2.5,
    "wan": 4.0, "sana": 3.8, "omnigen": 3.2, "deepfloyd": 6.0,
    "sd2x": 7.0, "sd1x": 6.0,
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
    1: "3e-4", 16: "1e-4", 64: "8e-5", 128: "6e-5", 256: "5.09e-5",
}

class ConfigState:
    """Holds the configuration state across navigation"""
    def __init__(self):
        self.env_contents = {
            "--resume_from_checkpoint": "latest",
            "--data_backend_config": "config/multidatabackend.json",
            "--aspect_bucket_rounding": 2,
            "--seed": 42,
            "--minimum_image_size": 0,
            "--disable_benchmark": False,
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
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                
            # Update env_contents with loaded values
            self.env_contents.update(loaded_config)
            
            # Extract model type
            if "--model_type" in loaded_config:
                self.model_type = loaded_config["--model_type"]
                
            # Check if using LoRA
            if "--lora_type" in loaded_config:
                self.use_lora = True
                self.use_lycoris = loaded_config["--lora_type"] == "lycoris"
                
            # Get LoRA rank if present
            if "--lora_rank" in loaded_config:
                self.lora_rank = loaded_config["--lora_rank"]
                
            # Load LyCORIS config if specified
            if "--lycoris_config" in loaded_config and os.path.exists(loaded_config["--lycoris_config"]):
                with open(loaded_config["--lycoris_config"], 'r', encoding='utf-8') as f:
                    self.lycoris_config = json.load(f)
                    
            # Load dataset config if specified
            backend_config = loaded_config.get("--data_backend_config", "config/multidatabackend.json")
            if os.path.exists(backend_config):
                with open(backend_config, 'r', encoding='utf-8') as f:
                    self.dataset_config = json.load(f)
                    
            self.loaded_config_path = config_path
            
            # Mark steps as completed based on what's configured
            if "--output_dir" in loaded_config:
                self.completed_steps.add(0)  # Basic setup
            if "--model_type" in loaded_config:
                self.completed_steps.add(1)  # Model type
            if "--max_train_steps" in loaded_config or "--num_train_epochs" in loaded_config:
                self.completed_steps.add(2)  # Training config
            if "--model_family" in loaded_config:
                self.completed_steps.add(4)  # Model selection
            if "--train_batch_size" in loaded_config:
                self.completed_steps.add(5)  # Training params
            if "--optimizer" in loaded_config:
                self.completed_steps.add(6)  # Optimization
            if "--validation_prompt" in loaded_config:
                self.completed_steps.add(7)  # Validation
                
            return True
            
        except Exception as e:
            return False

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
                        self.show_error(stdscr, f"Error in {self.menu_items[action][0]}: {str(e)}")
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
                info = "..." + info[-(w-7):]
            stdscr.addstr(2, 2, info, curses.A_DIM)
            
        stdscr.addstr(3, 2, "Use arrow keys to navigate, Enter to select, 'q' to quit, 'l' to load config")
        
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
                    text = text[:w-7] + "..."
                    
                stdscr.addstr(y, 2, text, attr)
            
            stdscr.refresh()
            
            key = stdscr.getch()
            if key == ord('q'):
                if self.confirm_quit(stdscr):
                    return "quit"
            elif key == ord('l'):
                if self.load_config_dialog(stdscr):
                    # Refresh the menu to show updated state
                    return None
            elif key == curses.KEY_UP and selected > 0:
                selected -= 1
            elif key == curses.KEY_DOWN and selected < len(self.menu_items) - 1:
                selected += 1
            elif key in [curses.KEY_ENTER, ord('\n'), ord('\r')]:
                return selected

    def show_error(self, stdscr, error_msg: str):
        """Display an error message and wait for acknowledgment"""
        h, w = stdscr.getmaxyx()
        
        # Create error window
        error_lines = textwrap.wrap(error_msg, w - 10)
        error_h = len(error_lines) + 4
        error_w = min(80, w - 4)
        
        error_win = curses.newwin(error_h, error_w, 
                                 (h - error_h) // 2, 
                                 (w - error_w) // 2)
        error_win.box()
        error_win.addstr(0, 2, " Error ", curses.A_BOLD | curses.color_pair(1))
        
        for idx, line in enumerate(error_lines):
            error_win.addstr(idx + 1, 2, line)
            
        error_win.addstr(error_h - 2, 2, "Press any key to continue...")
        error_win.refresh()
        error_win.getch()

    def get_input(self, stdscr, prompt: str, default: str = "", 
                  validation_fn=None, multiline=False) -> str:
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
                user_input = stdscr.getstr(input_y, 4).decode('utf-8')
            else:
                user_input = stdscr.getstr(input_y, 4, w - 6).decode('utf-8')
                
            if not user_input and default:
                user_input = default
                
            if validation_fn and not validation_fn(user_input):
                raise ValueError("Invalid input")
                
            return user_input
            
        finally:
            curses.noecho()
            curses.curs_set(0)

    def show_options(self, stdscr, prompt: str, options: List[str], 
                     default: int = 0) -> int:
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
                    text = text[:w-7] + "..."
                stdscr.addstr(start_y + idx, 4, text, attr)
                
            stdscr.refresh()
            
            key = stdscr.getch()
            if key == curses.KEY_UP and selected > 0:
                selected -= 1
            elif key == curses.KEY_DOWN and selected < len(options) - 1:
                selected += 1
            elif key in [curses.KEY_ENTER, ord('\n'), ord('\r')]:
                return selected
            elif key == 27:  # ESC
                return -1

    def confirm_quit(self, stdscr) -> bool:
        """Confirm quit dialog"""
        return self.show_options(stdscr, 
                               "Are you sure you want to quit? Unsaved changes will be lost.",
                               ["No, continue", "Yes, quit"], 0) == 1

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
        
        selected = self.show_options(stdscr,
                                   "Select a configuration to load:",
                                   options, 
                                   2 if len(config_files) > 0 else 0)
        
        if selected == -1:  # ESC pressed
            return False
            
        if selected == 0:  # New configuration
            self.state = ConfigState()  # Reset to defaults
            self.state.loaded_config_path = None
            return True
            
        elif selected == 1:  # Manual entry
            config_path = self.get_input(stdscr,
                                       "Enter path to config.json:",
                                       "config/config.json")
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
        
        msg_win = curses.newwin(msg_h, msg_w, 
                               (h - msg_h) // 2, 
                               (w - msg_w) // 2)
        msg_win.box()
        msg_win.addstr(0, 2, " Info ", curses.A_BOLD)
        
        for idx, line in enumerate(msg_lines):
            msg_win.addstr(idx + 1, 2, line)
            
        msg_win.addstr(msg_h - 2, 2, "Press any key to continue...")
        msg_win.refresh()
        msg_win.getch()

    def basic_setup(self, stdscr):
        """Step 1: Basic Setup"""
        # Show current value if exists
        current_output_dir = self.state.env_contents.get("--output_dir", "output/models")
        
        output_dir = self.get_input(stdscr, 
                                   f"Enter the directory where you want to store your outputs\n(Current: {current_output_dir}):",
                                   current_output_dir)
        
        if not os.path.exists(output_dir):
            if self.show_options(stdscr, 
                               f"Directory {output_dir} doesn't exist. Create it?",
                               ["Yes", "No, choose another"], 0) == 0:
                os.makedirs(output_dir, exist_ok=True)
            else:
                return self.basic_setup(stdscr)
                
        self.state.env_contents["--output_dir"] = output_dir

    def model_type_setup(self, stdscr):
        """Step 2: Model Type & LoRA/LyCORIS Setup"""
        # Check current model type
        current_type = self.state.env_contents.get("--model_type", "lora")
        
        model_type_idx = self.show_options(stdscr,
                                         f"What type of model are you training?\n(Current: {current_type})",
                                         ["LoRA", "Full"], 
                                         0 if current_type == "lora" else 1)
        
        self.state.model_type = "lora" if model_type_idx == 0 else "full"
        self.state.env_contents["--model_type"] = self.state.model_type
        
        if self.state.model_type == "lora":
            self.state.use_lora = True
            
            # Check if already configured
            current_lora_type = self.state.env_contents.get("--lora_type", "standard")
            default_lycoris = 0 if current_lora_type == "lycoris" else 1
            
            use_lycoris_idx = self.show_options(stdscr,
                                              f"Would you like to train a LyCORIS model?\n(Current: {current_lora_type})",
                                              ["Yes", "No"], default_lycoris)
            
            if use_lycoris_idx == 0:
                self.state.use_lycoris = True
                self.state.env_contents["--lora_type"] = "lycoris"
                self.configure_lycoris(stdscr)
            else:
                self.state.env_contents["--lora_type"] = "standard"
                
                # Check for DoRA
                current_dora = self.state.env_contents.get("--use_dora", "false") == "true"
                use_dora_idx = self.show_options(stdscr,
                                               f"Would you like to train a DoRA model?\n(Current: {'Yes' if current_dora else 'No'})",
                                               ["No", "Yes"], 1 if current_dora else 0)
                
                if use_dora_idx == 1:
                    self.state.env_contents["--use_dora"] = "true"
                elif "--use_dora" in self.state.env_contents:
                    del self.state.env_contents["--use_dora"]
                    
                # LoRA rank selection
                current_rank = self.state.env_contents.get("--lora_rank", 64)
                rank_options = [str(r) for r in lora_ranks]
                
                # Find current rank index
                default_rank_idx = 2  # Default to 64
                if current_rank in lora_ranks:
                    default_rank_idx = lora_ranks.index(current_rank)
                    
                rank_idx = self.show_options(stdscr,
                                           f"Set the LoRA rank:\n(Current: {current_rank})",
                                           rank_options, default_rank_idx)
                
                self.state.lora_rank = lora_ranks[rank_idx]
                self.state.env_contents["--lora_rank"] = self.state.lora_rank
        else:
            # Full fine-tuning
            current_ema = self.state.env_contents.get("--use_ema", "false") == "true"
            use_ema_idx = self.show_options(stdscr,
                                          f"Would you like to use EMA for training?\n(Current: {'Yes' if current_ema else 'No'})",
                                          ["No", "Yes"], 1 if current_ema else 0)
            
            if use_ema_idx == 1:
                self.state.env_contents["--use_ema"] = "true"
            elif "--use_ema" in self.state.env_contents:
                del self.state.env_contents["--use_ema"]

    def configure_lycoris(self, stdscr):
        """Configure LyCORIS settings"""
        algorithms = [
            ("LoRA", "lora", "Efficient, balanced fine-tuning"),
            ("LoHa", "loha", "Advanced, strong dampening"),
            ("LoKr", "lokr", "Kronecker product-based"),
            ("Full", "full", "Traditional full model tuning"),
            ("IA³", "ia3", "Efficient, tiny files, best for styles"),
            ("DyLoRA", "dylora", "Dynamic updates"),
            ("Diag-OFT", "diag-oft", "Fast convergence"),
            ("BOFT", "boft", "Advanced OFT"),
            ("GLoRA", "glora", "Generalized LoRA"),
        ]
        
        algo_names = [f"{name} - {desc}" for name, _, desc in algorithms]
        algo_idx = self.show_options(stdscr,
                                   "Select a LyCORIS algorithm:",
                                   algo_names, 2)  # Default to LoKr
        
        algo = algorithms[algo_idx][1]
        default_config = lycoris_defaults.get(algo, {}).copy()
        
        # Get multiplier
        multiplier = self.get_input(stdscr,
                                  "Set the effect multiplier (adjust for stronger or subtler effects):",
                                  str(default_config.get('multiplier', 1.0)))
        
        try:
            default_config['multiplier'] = float(multiplier)
        except ValueError:
            default_config['multiplier'] = 1.0
            
        # Continue with other LyCORIS parameters...
        # (Similar pattern for other parameters)
        
        self.state.lycoris_config = default_config
        
        # Save LyCORIS config
        with open("config/lycoris_config.json", "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
            
        self.state.env_contents["--lycoris_config"] = "config/lycoris_config.json"

    def training_config(self, stdscr):
        """Step 3: Training Configuration"""
        count_type_idx = self.show_options(stdscr,
                                         "Should we schedule the end of training by epochs or steps?",
                                         ["Steps", "Epochs"], 0)
        
        if count_type_idx == 0:
            max_steps = self.get_input(stdscr,
                                     "Set the maximum number of steps:",
                                     "10000")
            try:
                self.state.env_contents["--max_train_steps"] = int(max_steps)
                self.state.env_contents["--num_train_epochs"] = 0
            except ValueError:
                self.state.env_contents["--max_train_steps"] = 10000
                self.state.env_contents["--num_train_epochs"] = 0
        else:
            max_epochs = self.get_input(stdscr,
                                      "Set the maximum number of epochs:",
                                      "100")
            try:
                self.state.env_contents["--num_train_epochs"] = int(max_epochs)
                self.state.env_contents["--max_train_steps"] = 0
            except ValueError:
                self.state.env_contents["--num_train_epochs"] = 100
                self.state.env_contents["--max_train_steps"] = 0
                
        # Checkpointing
        default_interval = 500
        if self.state.env_contents.get("--max_train_steps", 0) > 0:
            if self.state.env_contents["--max_train_steps"] < default_interval:
                default_interval = self.state.env_contents["--max_train_steps"] // 10
                
        checkpoint_interval = self.get_input(stdscr,
                                           "Set the checkpointing interval (in steps):",
                                           str(default_interval))
        
        try:
            self.state.env_contents["--checkpointing_steps"] = int(checkpoint_interval)
        except ValueError:
            self.state.env_contents["--checkpointing_steps"] = default_interval
            
        checkpoint_limit = self.get_input(stdscr,
                                        "How many checkpoints do you want to keep?",
                                        "5")
        
        try:
            self.state.env_contents["--checkpoints_total_limit"] = int(checkpoint_limit)
        except ValueError:
            self.state.env_contents["--checkpoints_total_limit"] = 5

    def hub_setup(self, stdscr):
        """Step 4: Hugging Face Hub Setup"""
        stdscr.clear()
        stdscr.addstr(2, 2, "Checking Hugging Face Hub login...")
        stdscr.refresh()
        
        try:
            self.state.whoami = huggingface_hub.whoami()
        except:
            self.state.whoami = None
            
        if not self.state.whoami:
            login_idx = self.show_options(stdscr,
                                        "You are not logged into Hugging Face Hub. Would you like to login?",
                                        ["Yes", "No"], 0)
            
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
                    
        if self.state.whoami:
            stdscr.clear()
            stdscr.addstr(2, 2, f"Connected as: {self.state.whoami['name']}")
            stdscr.refresh()
            
            # Check current push settings
            current_push = self.state.env_contents.get("--push_to_hub", "false") == "true"
            
            push_idx = self.show_options(stdscr,
                                       f"Do you want to push your model to Hugging Face Hub when completed?\n(Current: {'Yes' if current_push else 'No'})",
                                       ["Yes", "No"], 0 if current_push else 1)
            
            if push_idx == 0:
                self.state.env_contents["--push_to_hub"] = "true"
                
                # Get current model ID
                current_model_id = self.state.env_contents.get("--hub_model_id", f"simpletuner-{self.state.model_type}")
                
                model_id = self.get_input(stdscr,
                                        f"Model name (will be accessible as https://huggingface.co/{self.state.whoami['name']}/...):\n(Current: {current_model_id})",
                                        current_model_id)
                
                self.state.env_contents["--hub_model_id"] = model_id
                
                # Check current checkpoint push setting
                current_push_ckpt = self.state.env_contents.get("--push_checkpoints_to_hub", "false") == "true"
                
                push_checkpoints_idx = self.show_options(stdscr,
                                                       f"Push intermediary checkpoints to Hub?\n(Current: {'Yes' if current_push_ckpt else 'No'})",
                                                       ["Yes", "No"], 0 if current_push_ckpt else 1)
                
                if push_checkpoints_idx == 0:
                    self.state.env_contents["--push_checkpoints_to_hub"] = "true"
                elif "--push_checkpoints_to_hub" in self.state.env_contents:
                    del self.state.env_contents["--push_checkpoints_to_hub"]
                    
                # Check SFW setting
                current_sfw = self.state.env_contents.get("--model_card_safe_for_work", "false") == "true"
                
                safe_idx = self.show_options(stdscr,
                                           f"Is your model safe-for-work?\n(Current: {'Yes' if current_sfw else 'No'})",
                                           ["No", "Yes"], 1 if current_sfw else 0)
                
                if safe_idx == 1:
                    self.state.env_contents["--model_card_safe_for_work"] = "true"
                elif "--model_card_safe_for_work" in self.state.env_contents:
                    del self.state.env_contents["--model_card_safe_for_work"]
            else:
                # Remove push settings if not pushing
                for key in ["--push_to_hub", "--hub_model_id", "--push_checkpoints_to_hub", "--model_card_safe_for_work"]:
                    if key in self.state.env_contents:
                        del self.state.env_contents[key]

    def model_selection(self, stdscr):
        """Step 5: Model Selection"""
        model_type = self.state.model_type or "lora"
        available_models = model_classes[model_type]
        
        model_idx = self.show_options(stdscr,
                                    "Which model family are you training?",
                                    available_models, 0)
        
        if model_idx == -1:
            return
            
        model_class = available_models[model_idx]
        self.state.env_contents["--model_family"] = model_class
        
        # Model name from HF Hub
        default_model = default_models.get(model_class, "")
        
        while True:
            model_name = self.get_input(stdscr,
                                      "Enter the model name from Hugging Face Hub:",
                                      default_model)
            
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
                
        self.state.env_contents["--model_type"] = model_type
        self.state.env_contents["--pretrained_model_name_or_path"] = model_name
        
        # Flux-specific options
        if model_class == "flux" and model_type == "lora" and not self.state.use_lycoris:
            flux_targets = ["mmdit", "context", "all", "all+ffs", "ai-toolkit", "tiny", "nano"]
            target_idx = self.show_options(stdscr,
                                         "Set Flux target layers:",
                                         flux_targets, 2)  # Default to "all"
            
            if target_idx != -1:
                self.state.env_contents["--flux_lora_target"] = flux_targets[target_idx]

    def training_params(self, stdscr):
        """Step 6: Training Parameters"""
        # Batch size
        batch_size = self.get_input(stdscr,
                                  "Set the training batch size (larger values need more VRAM):",
                                  "1")
        
        try:
            self.state.env_contents["--train_batch_size"] = int(batch_size)
        except ValueError:
            self.state.env_contents["--train_batch_size"] = 1
            
        # Gradient checkpointing
        self.state.env_contents["--gradient_checkpointing"] = "true"
        
        if self.state.env_contents.get("--model_family") in ["sdxl", "flux", "sd3", "sana"]:
            gc_interval = self.get_input(stdscr,
                                       "Gradient checkpointing interval (0 to disable, >1 speeds up training):",
                                       "0")
            
            try:
                interval = int(gc_interval)
                if interval > 1:
                    self.state.env_contents["--gradient_checkpointing_interval"] = interval
            except ValueError:
                pass
                
        # Caption dropout
        caption_dropout = self.get_input(stdscr,
                                       "Set caption dropout rate (0.0 to disable):",
                                       "0.05" if self.state.use_lora else "0.1")
        
        try:
            self.state.env_contents["--caption_dropout_probability"] = float(caption_dropout)
        except ValueError:
            self.state.env_contents["--caption_dropout_probability"] = 0.05
            
        # Resolution settings
        res_types = ["pixel", "area", "pixel_area"]
        res_idx = self.show_options(stdscr,
                                  "How to measure dataset resolutions?",
                                  ["Pixel (shorter edge)", "Area (megapixels)", "Pixel Area (combination)"],
                                  2)
        
        if res_idx != -1:
            self.state.env_contents["--resolution_type"] = res_types[res_idx]
            
            if res_types[res_idx] in ["pixel", "pixel_area"]:
                default_res = "1024"
            else:
                default_res = "1.0"
                
            resolution = self.get_input(stdscr,
                                      f"Default resolution ({res_types[res_idx]}):",
                                      default_res)
            
            self.state.env_contents["--resolution"] = resolution

    def optimization_settings(self, stdscr):
        """Step 7: Optimization Settings"""
        # TF32
        if torch.cuda.is_available():
            tf32_idx = self.show_options(stdscr,
                                       "Enable TF32 mode?",
                                       ["Yes", "No"], 0)
            
            if tf32_idx == 1:
                self.state.env_contents["--disable_tf32"] = "true"
                
        # Mixed precision
        mixed_precision_idx = self.show_options(stdscr,
                                              "Set mixed precision mode:",
                                              ["bf16", "fp8", "no (fp32)"], 0)
        
        mixed_precision_map = ["bf16", "fp8", "no"]
        self.state.env_contents["--mixed_precision"] = mixed_precision_map[mixed_precision_idx]
        
        # Optimizer selection
        if self.state.env_contents["--mixed_precision"] == "bf16":
            compatible_optims = bf16_only_optims + any_precision_optims
        else:
            compatible_optims = any_precision_optims
            
        optim_idx = self.show_options(stdscr,
                                    "Choose an optimizer:",
                                    compatible_optims, 0)
        
        if optim_idx != -1:
            self.state.env_contents["--optimizer"] = compatible_optims[optim_idx]
            
        # Learning rate scheduler
        lr_schedulers = ["polynomial", "constant"]
        lr_idx = self.show_options(stdscr,
                                 "Set learning rate scheduler:",
                                 lr_schedulers, 0)
        
        if lr_idx != -1:
            lr_scheduler = lr_schedulers[lr_idx]
            self.state.env_contents["--lr_scheduler"] = lr_scheduler
            
            if lr_scheduler == "polynomial":
                self.state.extra_args.append("--lr_end=1e-8")
                
        # Learning rate
        default_lr = "1e-6"
        if self.state.model_type == "lora" and hasattr(self.state, 'lora_rank'):
            default_lr = learning_rates_by_rank.get(self.state.lora_rank, "1e-4")
        elif self.state.env_contents.get("--optimizer") == "prodigy":
            default_lr = "1.0"
            
        lr = self.get_input(stdscr,
                          "Set the learning rate:",
                          default_lr)
        
        self.state.env_contents["--learning_rate"] = lr
        
        # Warmup steps
        default_warmup = "100"
        if self.state.env_contents.get("--max_train_steps", 0) > 0:
            default_warmup = str(min(100, self.state.env_contents["--max_train_steps"] // 10))
            
        warmup = self.get_input(stdscr,
                              "Set warmup steps:",
                              default_warmup)
        
        try:
            self.state.env_contents["--lr_warmup_steps"] = int(warmup)
        except ValueError:
            self.state.env_contents["--lr_warmup_steps"] = 100
            
        # Quantization
        if self.state.use_lora:
            warning = "NOTE: Currently, a bug prevents multi-GPU training with LoRA quantization"
        else:
            warning = ""
            
        quant_idx = self.show_options(stdscr,
                                    f"Enable model quantization? {warning}",
                                    ["Yes", "No"], 0)
        
        if quant_idx == 0:
            if self.state.env_contents.get("--use_dora") == "true":
                del self.state.env_contents["--use_dora"]
                
            quant_types = list(quantised_precision_levels)
            quant_type_idx = self.show_options(stdscr,
                                             "Choose quantization type:",
                                             quant_types, 0)
            
            if quant_type_idx != -1:
                self.state.env_contents["--base_model_precision"] = quant_types[quant_type_idx]

    def validation_settings(self, stdscr):
        """Step 8: Validation Settings"""
        # Validation seed
        val_seed = self.get_input(stdscr,
                                "Set the seed for validation:",
                                "42")
        
        self.state.env_contents["--validation_seed"] = val_seed
        
        # Validation steps
        default_val_steps = str(self.state.env_contents.get("--checkpointing_steps", 500))
        val_steps = self.get_input(stdscr,
                                 "How many steps between validation outputs?",
                                 default_val_steps)
        
        self.state.env_contents["--validation_steps"] = val_steps
        
        # Validation resolution
        val_res = self.get_input(stdscr,
                               "Set validation resolution (e.g., 1024x1024 or comma-separated list):",
                               "1024x1024")
        
        # Clean up resolution
        val_res = ",".join([x.strip() for x in val_res.split(",")])
        self.state.env_contents["--validation_resolution"] = val_res
        
        # Validation guidance
        model_family = self.state.env_contents.get("--model_family", "flux")
        default_cfg_val = str(default_cfg.get(model_family, 3.0))
        
        val_guidance = self.get_input(stdscr,
                                    "Set guidance scale for validation:",
                                    default_cfg_val)
        
        self.state.env_contents["--validation_guidance"] = val_guidance
        
        # Guidance rescale
        val_rescale = self.get_input(stdscr,
                                   "Set guidance rescale (dynamic thresholding, 0.0 to disable):",
                                   "0.0")
        
        self.state.env_contents["--validation_guidance_rescale"] = val_rescale
        
        # Inference steps
        val_inf_steps = self.get_input(stdscr,
                                     "Set number of inference steps for validation:",
                                     "20")
        
        self.state.env_contents["--validation_num_inference_steps"] = val_inf_steps
        
        # Validation prompt
        val_prompt = self.get_input(stdscr,
                                  "Set the validation prompt:",
                                  "A photo-realistic image of a cat")
        
        self.state.env_contents["--validation_prompt"] = val_prompt

    def advanced_options(self, stdscr):
        """Step 9: Advanced Options"""
        # Tracking
        wandb_idx = self.show_options(stdscr,
                                    "Report to Weights & Biases?",
                                    ["Yes", "No"], 0)
        
        tensorboard_idx = self.show_options(stdscr,
                                          "Report to TensorBoard?",
                                          ["No", "Yes"], 0)
        
        report_to = "none"
        if wandb_idx == 0 or tensorboard_idx == 1:
            project_name = self.get_input(stdscr,
                                        "Enter tracker project name:",
                                        f"{self.state.model_type}-training")
            
            self.state.env_contents["--tracker_project_name"] = project_name
            
            run_name = self.get_input(stdscr,
                                    "Enter tracker run name:",
                                    f"simpletuner-{self.state.model_type}")
            
            self.state.env_contents["--tracker_run_name"] = run_name
            
            if wandb_idx == 0:
                report_to = "wandb"
            if tensorboard_idx == 1:
                report_to = "tensorboard" if report_to == "none" else f"{report_to},tensorboard"
                
        self.state.env_contents["--report_to"] = report_to
        
        # SageAttention
        self.state.env_contents["--attention_mechanism"] = "diffusers"
        
        sage_idx = self.show_options(stdscr,
                                   "Use SageAttention for validation?",
                                   ["No", "Yes"], 0)
        
        if sage_idx == 1:
            self.state.env_contents["--attention_mechanism"] = "sageattention"
            self.state.env_contents["--sageattention_usage"] = "inference"
            
            sage_training_idx = self.show_options(stdscr,
                                                "Use SageAttention for training? (WARNING: May leave attention layers untrained)",
                                                ["No", "Yes"], 0)
            
            if sage_training_idx == 1:
                self.state.env_contents["--sageattention_usage"] = "both"
                
        # Disk cache compression
        compress_idx = self.show_options(stdscr,
                                       "Compress disk cache?",
                                       ["Yes", "No"], 0)
        
        if compress_idx == 0:
            self.state.extra_args.append("--compress_disk_cache")
            
        # Torch compile
        compile_idx = self.show_options(stdscr,
                                      "Use torch compile during validations?",
                                      ["No", "Yes"], 0)
        
        self.state.env_contents["--validation_torch_compile"] = "true" if compile_idx == 1 else "false"
        
        # Prompt library generation
        prompt_lib_idx = self.show_options(stdscr,
                                         "Generate a prompt library? (requires Llama 3.2 1B download)",
                                         ["Yes", "No"], 0)
        
        if prompt_lib_idx == 0:
            trigger = self.get_input(stdscr,
                                   "Enter trigger word(s) for prompt expansion:",
                                   "Character Name")
            
            num_prompts = self.get_input(stdscr,
                                       "How many prompts to generate?",
                                       "8")
            
            try:
                from helpers.prompt_expander import PromptExpander
                
                stdscr.clear()
                stdscr.addstr(2, 2, "Initializing model and generating prompts...")
                stdscr.refresh()
                
                PromptExpander.initialize_model()
                user_prompt_library = PromptExpander.generate_prompts(
                    trigger_phrase=trigger,
                    num_prompts=int(num_prompts)
                )
                
                with open("config/user_prompt_library.json", "w", encoding="utf-8") as f:
                    json.dump(user_prompt_library, f, indent=4)
                    
                self.state.env_contents["--user_prompt_library"] = "config/user_prompt_library.json"
                
            except Exception as e:
                self.show_error(stdscr, f"Failed to generate prompt library: {str(e)}")

    def dataset_config(self, stdscr):
        """Step 10: Dataset Configuration"""
        config_idx = self.show_options(stdscr,
                                     "Configure dataloader?",
                                     ["Yes", "No"], 0)
        
        if config_idx == 1:
            return
            
        # Dataset basics
        dataset_id = self.get_input(stdscr,
                                  "Dataset name (simple, no spaces):",
                                  "my-dataset")
        
        dataset_path = self.get_input(stdscr,
                                    "Dataset path (absolute path recommended):",
                                    "/datasets/my-dataset")
        
        # Caption strategy
        caption_strategies = [
            ("filename", "Use image filenames as captions"),
            ("textfile", "Use .txt files next to images"),
            ("instanceprompt", "Use one trigger phrase for all"),
        ]
        
        caption_idx = self.show_options(stdscr,
                                      "How should captions be handled?",
                                      [f"{name} - {desc}" for name, desc in caption_strategies],
                                      1)
        
        caption_strategy = caption_strategies[caption_idx][0]
        instance_prompt = None
        
        if caption_strategy == "instanceprompt":
            instance_prompt = self.get_input(stdscr,
                                           "Enter instance prompt for all images:",
                                           "Character Name")
            
        # Dataset repeats
        repeats = self.get_input(stdscr,
                               "Dataset repeats (0 = once, 1 = twice, etc.):",
                               "10")
        
        try:
            dataset_repeats = int(repeats)
        except ValueError:
            dataset_repeats = 10
            
        # Resolutions
        default_res = "1024"
        if self.state.env_contents.get("--model_family") == "flux":
            default_res = "256,512,768,1024,1440"
            
        resolutions_str = self.get_input(stdscr,
                                       "Training resolutions (comma-separated for multiple):",
                                       default_res)
        
        try:
            if "," in resolutions_str:
                resolutions = [int(r.strip()) for r in resolutions_str.split(",")]
            else:
                resolutions = [int(resolutions_str)]
        except ValueError:
            resolutions = [1024]
            
        # Cache directory
        cache_dir = self.get_input(stdscr,
                                 "Cache directory:",
                                 "cache/")
        
        # Large images
        large_img_idx = self.show_options(stdscr,
                                        "Do you have very large images (much larger than 1024x1024)?",
                                        ["No", "Yes"], 0)
        
        has_large_images = large_img_idx == 1
        
        # Build dataset configuration
        self._build_dataset_config(dataset_id, dataset_path, caption_strategy,
                                 instance_prompt, dataset_repeats, resolutions,
                                 cache_dir, has_large_images)

    def _build_dataset_config(self, dataset_id, dataset_path, caption_strategy,
                            instance_prompt, dataset_repeats, resolutions,
                            cache_dir, has_large_images):
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
        default_cropped.update({
            "id": "PLACEHOLDER-crop",
            "crop": True,
            "crop_aspect": "square",
            "crop_style": "center",
            "vae_cache_clear_each_epoch": False,
            "cache_dir_vae": "vae-crop",
        })
        
        datasets = [{
            "id": "text-embed-cache",
            "dataset_type": "text_embeds",
            "default": True,
            "type": "local",
            "cache_dir": os.path.abspath(
                os.path.join(cache_dir, self.state.env_contents["--model_family"], "text")
            ),
            "write_batch_size": 128,
        }]
        
        def create_dataset(resolution, template):
            dataset = template.copy()
            dataset.update(resolution_configs.get(resolution, {"resolution": resolution}))
            dataset["id"] = f"{dataset_id}-{resolution}" if "crop" not in dataset["id"] else f"{dataset_id}-crop-{resolution}"
            dataset["instance_data_dir"] = os.path.abspath(dataset_path)
            dataset["repeats"] = dataset_repeats
            dataset["cache_dir_vae"] = os.path.abspath(
                os.path.join(cache_dir, self.state.env_contents["--model_family"],
                           dataset["cache_dir_vae"], str(resolution))
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
        config_items.append(f"Output Directory: {self.state.env_contents.get('--output_dir', 'Not set')}")
        config_items.append(f"Model Family: {self.state.env_contents.get('--model_family', 'Not set')}")
        config_items.append(f"Base Model: {self.state.env_contents.get('--pretrained_model_name_or_path', 'Not set')}")
        
        if self.state.use_lora:
            config_items.append(f"LoRA Type: {self.state.env_contents.get('--lora_type', 'standard')}")
            if not self.state.use_lycoris:
                config_items.append(f"LoRA Rank: {self.state.env_contents.get('--lora_rank', 'Not set')}")
                
        # Display items
        for item in config_items:
            if y < h - 4:
                stdscr.addstr(y, 4, item[:w-6])
                y += 1
                
        stdscr.addstr(h - 3, 2, "Press 's' to save, 'b' to go back, 'q' to quit without saving")
        stdscr.refresh()
        
        while True:
            key = stdscr.getch()
            if key == ord('s'):
                self._save_configuration(stdscr)
                return
            elif key == ord('b'):
                return
            elif key == ord('q'):
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
                    "Save to default (config/config.json)"
                ]
                
                save_choice = self.show_options(stdscr,
                                              "Where would you like to save?",
                                              save_options, 0)
                
                if save_choice == 0:
                    save_path = self.state.loaded_config_path
                elif save_choice == 1:
                    save_path = self.get_input(stdscr,
                                             "Enter save path for config.json:",
                                             "config/my-preset/config.json")
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
                backend_path = self.state.env_contents.get("--data_backend_config", "config/multidatabackend.json")
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
                stdscr.addstr(6, 4, f"- {self.state.env_contents.get('--data_backend_config', 'config/multidatabackend.json')}")
                
            if self.state.lycoris_config:
                stdscr.addstr(7, 4, f"- {self.state.env_contents.get('--lycoris_config', 'config/lycoris_config.json')}")
                
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
        print(f"Loaded existing configuration from: {configurator.state.loaded_config_path}")
    else:
        print("No existing configuration found. Starting fresh setup.")
    
    print("\nStarting SimpleTuner configuration tool...")
    print("Press any key to continue...")
    input()
    
    configurator.run()


if __name__ == "__main__":
    main()