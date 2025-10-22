#!/usr/bin/env python3
"""SimpleTuner configuration wizard driven by FieldRegistry metadata."""

from __future__ import annotations

import curses
import json
import os
import sys
import textwrap
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import simpletuner.helpers.models  # noqa: F401  # Ensure model registry population
from simpletuner.helpers.models.registry import ModelRegistry

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

    from simpletuner.simpletuner_sdk.server.services.field_service import FieldService
    from simpletuner.simpletuner_sdk.server.services.tab_service import TabService


@dataclass
class TabEntry:
    """Metadata for a configuration tab."""

    name: str
    title: str
    description: str
    id: str


def _build_model_class_map() -> Dict[str, List[str]]:
    """Construct capability-aware model family listings used by legacy consumers."""

    families: List[tuple[str, str]] = []
    lora_supported = set()
    control_supported = set()

    for family, model_cls in ModelRegistry.model_families().items():
        if not getattr(model_cls, "ENABLED_IN_WIZARD", True):
            continue

        display_name = getattr(model_cls, "NAME", family)
        families.append((family, display_name.lower()))

        try:
            if hasattr(model_cls, "supports_lora") and model_cls.supports_lora():
                lora_supported.add(family)
        except Exception:
            pass

        try:
            if hasattr(model_cls, "supports_controlnet") and model_cls.supports_controlnet():
                control_supported.add(family)
        except Exception:
            pass

    families.sort(key=lambda item: item[1])
    ordered = [family for family, _ in families]

    return {
        "full": ordered,
        "lora": [family for family in ordered if family in lora_supported],
        "controlnet": [family for family in ordered if family in control_supported],
    }


model_classes = _build_model_class_map()

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
    """Holds configuration values and interacts with the FieldRegistry."""

    def __init__(self, field_service: "FieldService"):
        self.field_service = field_service
        self.registry = field_service.field_registry
        self.field_defs = self._load_field_definitions()
        self.aliases = {name: self._compute_aliases(field) for name, field in self.field_defs.items()}
        self.values = self._initialize_defaults()
        self.loaded_config_path: Optional[str] = None
        self.webui_defaults: Dict[str, Any] = {}
        self.unknown_values: Dict[str, Any] = {}

    def _load_field_definitions(self) -> Dict[str, Any]:
        definitions: Dict[str, Any] = {}
        try:
            fields_iterable = self.registry.get_all_fields()
        except AttributeError:
            fields_iterable = getattr(self.registry, "_fields", {}).values()

        for field in fields_iterable or []:
            if not field:
                continue
            definitions[field.name] = field
        return definitions

    def _initialize_defaults(self) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {}
        for name, field in self.field_defs.items():
            if getattr(field, "webui_only", False):
                continue
            defaults[name] = field.default_value
        return defaults

    def _compute_aliases(self, field: Any) -> List[str]:
        aliases = set()
        aliases.add(field.name)
        arg_name = getattr(field, "arg_name", "")
        if arg_name:
            aliases.add(arg_name)
            cleaned = arg_name.lstrip("-")
            if cleaned:
                aliases.add(f"--{cleaned}")
        aliases.add(f"--{field.name}")
        for alias in getattr(field, "aliases", []) or []:
            if alias:
                aliases.add(alias)
        return list(aliases)

    def reset_to_defaults(self) -> None:
        """Reset configuration values to FieldRegistry defaults."""

        self.values = self._initialize_defaults()
        self.loaded_config_path = None
        self.unknown_values = {}

    def get_value(self, field_name: str) -> Any:
        """Return the current value for a field."""

        if field_name in self.values:
            return self.values[field_name]
        field = self.field_defs.get(field_name)
        if field and not getattr(field, "webui_only", False):
            return field.default_value
        return None

    def set_value(self, field_name: str, value: Any) -> None:
        """Persist a value for a field."""

        if field_name not in self.field_defs:
            self.values[field_name] = value
            return

        field = self.field_defs[field_name]
        if getattr(field, "webui_only", False):
            return

        self.values[field_name] = value
        for alias in self.aliases.get(field_name, []):
            self.unknown_values.pop(alias, None)

    def as_config_data(self) -> Dict[str, Any]:
        """Return configuration data, including aliases, for FieldService context."""

        data: Dict[str, Any] = dict(self.unknown_values)
        for name, field in self.field_defs.items():
            if getattr(field, "webui_only", False):
                continue

            value = self.values.get(name, field.default_value)
            data[name] = value
            data[f"--{name}"] = value

            arg_name = getattr(field, "arg_name", "")
            if arg_name:
                data[arg_name] = value
                cleaned = arg_name.lstrip("-")
                if cleaned:
                    data.setdefault(f"--{cleaned}", value)

        return data

    def to_serializable(self) -> Dict[str, Any]:
        """Return a serializable dictionary suitable for writing to disk."""

        data = dict(self.unknown_values)
        for name, field in self.field_defs.items():
            if getattr(field, "webui_only", False):
                continue
            value = self.values.get(name, field.default_value)
            if value is None:
                continue
            data[name] = value
        return data

    def load_from_file(self, config_path: str) -> bool:
        """Load configuration from a JSON file."""

        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return False

        if not isinstance(payload, dict):
            return False

        self.apply_config(payload)
        self.loaded_config_path = config_path
        return True

    def apply_config(self, payload: Dict[str, Any]) -> None:
        """Apply configuration values from a dictionary."""

        self.values = self._initialize_defaults()
        recognized: set[str] = set()

        for name, field in self.field_defs.items():
            if getattr(field, "webui_only", False):
                continue

            for alias in self.aliases.get(name, [name]):
                if alias in payload:
                    self.values[name] = payload[alias]
                    recognized.add(alias)

        self.unknown_values = {key: value for key, value in payload.items() if key not in recognized}


class MenuNavigator:
    """Helper class for menu navigation."""

    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.h, self.w = stdscr.getmaxyx()

    def show_menu(
        self,
        title: str,
        items: List[tuple[str, Any]],
        current_values: Optional[Dict[str, str]] = None,
        selected: int = 0,
    ) -> int:
        """Display a scrollable menu."""

        while True:
            self.stdscr.clear()

            self.stdscr.addstr(1, 2, title, curses.A_BOLD)
            self.stdscr.addstr(2, 2, "─" * (self.w - 4))
            self.stdscr.addstr(3, 2, "↑/↓: Navigate  Enter: Select  ←/Backspace: Back  q: Quit")
            self.stdscr.addstr(4, 2, "─" * (self.w - 4))

            start_y = 6
            for idx, (item_name, _) in enumerate(items):
                if start_y + idx >= self.h - 2:
                    break

                attr = curses.A_REVERSE if idx == selected else curses.A_NORMAL
                display_text = f"{idx + 1}. {item_name}"

                if current_values and item_name in current_values:
                    value_text = current_values[item_name]
                    max_value_len = self.w - len(display_text) - 10
                    if max_value_len > 0 and len(value_text) > max_value_len:
                        value_text = "..." + value_text[-(max_value_len - 3) :]
                    display_text += f" [{value_text}]"

                if len(display_text) > self.w - 4:
                    display_text = display_text[: self.w - 7] + "..."

                self.stdscr.addstr(start_y + idx, 4, display_text, attr)

            self.stdscr.refresh()

            key = self.stdscr.getch()
            if key == ord("q"):
                return -1
            if key in [curses.KEY_LEFT, curses.KEY_BACKSPACE, 127, 8]:
                return -2
            if key == curses.KEY_UP and selected > 0:
                selected -= 1
            elif key == curses.KEY_DOWN and selected < len(items) - 1:
                selected += 1
            elif key in [curses.KEY_ENTER, ord("\n"), ord("\r")]:
                return selected
            elif ord("1") <= key <= ord("9"):
                num = key - ord("1")
                if num < len(items):
                    return num


class SimpleTunerNCurses:
    """Interactive curses interface powered by FieldRegistry metadata."""

    def __init__(self):
        from fastapi.templating import Jinja2Templates

        from simpletuner.simpletuner_sdk.server.services.field_service import FieldService
        from simpletuner.simpletuner_sdk.server.services.tab_service import TabService

        templates_dir = Path(__file__).resolve().parent / "templates"
        self.templates = Jinja2Templates(directory=str(templates_dir))

        try:
            self.tab_service = TabService(self.templates)
        except Exception as exc:  # pragma: no cover - initialization guard
            raise RuntimeError(f"Failed to initialise TabService: {exc}") from exc

        try:
            self.field_service = FieldService()
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Missing optional dependency '{exc.name}'. Install project extras to use the configurator."
            ) from exc

        self.state = ConfigState(self.field_service)
        self.tab_entries = self._build_tab_entries()
        self.tab_lookup = {entry.name: entry for entry in self.tab_entries}
        self.menu_items = [
            (entry.title, partial(self.edit_tab, tab_name=entry.name), entry.description or "") for entry in self.tab_entries
        ]
        self.menu_items.append(("Review & Save", self.review_and_save, "Review the configuration and write it to disk"))
        self._menu_index = 0

        default_config = Path("config/config.json")
        if default_config.exists():
            self.state.load_from_file(str(default_config))

    def _build_tab_entries(self) -> List[TabEntry]:
        entries: List[TabEntry] = []
        raw_tabs = self.tab_service.get_all_tabs()
        skip_tabs = {"checkpoints", "ui_settings"}

        for tab in raw_tabs:
            name = tab.get("name")
            if not name or name in skip_tabs:
                continue

            try:
                fields = self.field_service.field_registry.get_fields_for_tab(name)
            except Exception:
                fields = []

            fields = [field for field in fields if not getattr(field, "webui_only", False)]
            if not fields:
                continue

            entries.append(
                TabEntry(
                    name=name,
                    title=tab.get("title") or name.replace("_", " ").title(),
                    description=tab.get("description") or "",
                    id=tab.get("id") or name,
                )
            )

        return entries

    def run(self) -> None:
        """Launch the curses interface."""

        try:
            curses.wrapper(self._main_loop)
        except Exception as exc:
            print(f"Error: {exc}")
            traceback.print_exc()

    def _main_loop(self, stdscr) -> None:
        try:
            curses.curs_set(0)
        except curses.error:
            pass

        self.show_startup_screen(stdscr)

        while True:
            try:
                handler = self.show_main_menu(stdscr)
                if handler is None:
                    continue
                if handler == "quit":
                    break
                handler(stdscr)
            except KeyboardInterrupt:
                if self.confirm_quit(stdscr):
                    break

    def show_startup_screen(self, stdscr) -> None:
        """Display initial splash screen."""

        h, w = stdscr.getmaxyx()
        stdscr.clear()

        title_lines = [
            "╔═══════════════════════════════════════╗",
            "║      SimpleTuner Configuration        ║",
            "║       Registry-backed ncurses UI      ║",
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

    def show_main_menu(self, stdscr):
        """Render the main menu and return the selected handler."""

        if not self.menu_items:
            self.show_error(stdscr, "No tabs available from FieldRegistry.")
            return "quit"

        h, w = stdscr.getmaxyx()
        selected = min(self._menu_index, len(self.menu_items) - 1)
        max_visible = max(1, h - 7)
        scroll_offset = max(0, selected - max_visible + 1)

        while True:
            stdscr.clear()
            title = "SimpleTuner Configuration"
            stdscr.addstr(1, (w - len(title)) // 2, title, curses.A_BOLD)

            if self.state.loaded_config_path:
                info = f"Loaded: {self.state.loaded_config_path}"
                if len(info) > w - 4:
                    info = "..." + info[-(w - 7) :]
                stdscr.addstr(2, 2, info, curses.A_DIM)

            stdscr.addstr(3, 2, "↑/↓: Navigate  Enter: Select  'l': Load config  'q': Quit")

            visible_items = self.menu_items[scroll_offset : scroll_offset + max_visible]
            for idx, (item_name, _, _) in enumerate(visible_items):
                actual_idx = idx + scroll_offset
                y = 5 + idx
                attr = curses.A_REVERSE if actual_idx == selected else curses.A_NORMAL
                text = f"{actual_idx + 1}. {item_name}"
                if len(text) > w - 4:
                    text = text[: w - 7] + "..."
                stdscr.addstr(y, 2, text, attr)

            description = self.menu_items[selected][2]
            if description:
                desc_lines = textwrap.wrap(description, w - 4)
                if desc_lines:
                    stdscr.addstr(h - 2, 2, desc_lines[0][: w - 4], curses.A_DIM)

            if scroll_offset > 0:
                stdscr.addstr(4, w - 10, "▲ More", curses.A_DIM)
            if scroll_offset + max_visible < len(self.menu_items):
                stdscr.addstr(h - 3, w - 10, "▼ More", curses.A_DIM)

            stdscr.refresh()

            key = stdscr.getch()
            if key == ord("q"):
                if self.confirm_quit(stdscr):
                    return "quit"
            elif key == ord("l"):
                if self.load_config_dialog(stdscr):
                    selected = min(self._menu_index, len(self.menu_items) - 1)
                    scroll_offset = max(0, selected - max_visible + 1)
            elif key == curses.KEY_UP and selected > 0:
                selected -= 1
                if selected < scroll_offset:
                    scroll_offset = selected
            elif key == curses.KEY_DOWN and selected < len(self.menu_items) - 1:
                selected += 1
                if selected >= scroll_offset + max_visible:
                    scroll_offset = selected - max_visible + 1
            elif key in [curses.KEY_ENTER, ord("\n"), ord("\r")]:
                self._menu_index = selected
                return self.menu_items[selected][1]

    def show_error(self, stdscr, error_msg: str) -> None:
        """Display an error message."""

        h, w = stdscr.getmaxyx()
        error_lines = textwrap.wrap(error_msg, w - 10)
        error_h = len(error_lines) + 4
        error_w = min(80, w - 4)

        error_win = curses.newwin(error_h, error_w, (h - error_h) // 2, (w - error_w) // 2)
        error_win.box()
        error_win.addstr(0, 2, " Error ", curses.A_BOLD)

        for idx, line in enumerate(error_lines):
            error_win.addstr(idx + 1, 2, line)

        error_win.addstr(error_h - 2, 2, "Press any key to continue...")
        error_win.refresh()
        error_win.getch()

    def show_message(self, stdscr, message: str) -> None:
        """Display an informational message."""

        h, w = stdscr.getmaxyx()
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

    def get_input(
        self,
        stdscr,
        prompt: str,
        default: str = "",
        validation_fn=None,
        multiline: bool = False,
    ) -> str:
        """Prompt for user input."""

        h, w = stdscr.getmaxyx()
        stdscr.clear()
        wrapped_prompt = textwrap.wrap(prompt, w - 4)
        for idx, line in enumerate(wrapped_prompt):
            stdscr.addstr(2 + idx, 2, line)

        if default:
            stdscr.addstr(2 + len(wrapped_prompt) + 1, 2, f"Default: {default}")

        input_y = 2 + len(wrapped_prompt) + 3
        stdscr.addstr(input_y, 2, "> ")

        curses.echo()
        try:
            if multiline:
                user_input = stdscr.getstr(input_y, 4, w - 6).decode("utf-8")
            else:
                user_input = stdscr.getstr(input_y, 4, w - 6).decode("utf-8")
        finally:
            curses.noecho()

        if not user_input and default:
            user_input = default

        if validation_fn and not validation_fn(user_input):
            raise ValueError("Invalid input")

        return user_input

    def show_options(self, stdscr, prompt: str, options: List[str], default: int = 0) -> int:
        """Present a list of options and return the selected index."""

        stdscr.clear()
        h, w = stdscr.getmaxyx()

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
            elif key == 27:
                return -1

    def confirm_quit(self, stdscr) -> bool:
        """Confirm quitting the wizard."""

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
        """Return a list of discovered config.json files."""

        config_files: List[str] = []
        if os.path.exists(base_path):
            for root, _dirs, files in os.walk(base_path):
                if "config.json" in files:
                    config_files.append(os.path.join(root, "config.json"))
        return sorted(config_files)

    def load_config_dialog(self, stdscr) -> bool:
        """Allow the user to load or create a configuration file."""

        config_files = self.find_config_files()
        options = ["Create new configuration", "Enter path manually"] + config_files

        selected = self.show_options(
            stdscr,
            "Select a configuration to load:",
            options,
            2 if len(config_files) > 0 else 0,
        )

        if selected == -1:
            return False

        if selected == 0:
            self.state.reset_to_defaults()
            self.show_message(stdscr, "Started a new configuration.")
            return True
        if selected == 1:
            config_path = self.get_input(stdscr, "Enter path to config.json:", "config/config.json")
            if os.path.exists(config_path) and self.state.load_from_file(config_path):
                self.show_message(stdscr, f"Successfully loaded: {config_path}")
                return True
            self.show_error(stdscr, f"Failed to load: {config_path}")
            return False

        config_path = options[selected]
        if self.state.load_from_file(config_path):
            self.show_message(stdscr, f"Successfully loaded: {config_path}")
            return True

        self.show_error(stdscr, f"Failed to load: {config_path}")
        return False

    def _get_tab_structure(
        self,
        tab_name: str,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        config_data = self.state.as_config_data()
        tab_values = self.field_service.prepare_tab_field_values(tab_name, config_data, self.state.webui_defaults)
        fields, sections = self.field_service.build_template_tab(tab_name, tab_values, raw_config=config_data)
        return fields, sections, tab_values

    def edit_tab(self, stdscr, tab_name: str) -> None:
        """Display sections for the selected tab."""

        nav = MenuNavigator(stdscr)

        while True:
            fields, sections, _ = self._get_tab_structure(tab_name)
            section_fields = {
                section["id"]: [field for field in fields if field.get("section_id") == section["id"]]
                for section in sections
            }
            sections_with_fields = [section for section in sections if section_fields.get(section["id"])]

            if not sections_with_fields:
                self.show_message(stdscr, "No configurable fields in this tab with the current context.")
                return

            menu_items: List[tuple[str, Any]] = []
            current_values: Dict[str, str] = {}

            for section in sections_with_fields:
                title = section.get("title") or section["id"].replace("_", " ").title()
                count = len(section_fields[section["id"]])
                current_values[title] = f"{count} field{'s' if count != 1 else ''}"
                menu_items.append((title, partial(self.edit_section, tab_name=tab_name, section_id=section["id"])))

            choice = nav.show_menu(self.tab_lookup[tab_name].title, menu_items, current_values)

            if choice == -1:
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif choice == -2:
                return
            else:
                menu_items[choice][1](stdscr)

    def edit_section(self, stdscr, tab_name: str, section_id: str) -> None:
        """Display fields for a given section."""

        nav = MenuNavigator(stdscr)
        section_title = section_id.replace("_", " ").title()

        while True:
            fields, sections, _ = self._get_tab_structure(tab_name)
            section_fields = [field for field in fields if field.get("section_id") == section_id]

            if not section_fields:
                self.show_message(stdscr, "No fields available for this section with the current context.")
                return

            title_lookup = next((section.get("title") for section in sections if section["id"] == section_id), None)
            display_title = title_lookup or section_title

            menu_items: List[tuple[str, Any]] = []
            current_values: Dict[str, str] = {}

            for field in section_fields:
                label = field.get("label") or field["name"]
                if field.get("subsection"):
                    label = f"{label} [{field['subsection'].replace('_', ' ').title()}]"
                menu_items.append((label, partial(self.edit_field, tab_name=tab_name, field_name=field["name"])))
                current_values[label] = self._format_field_value(field)

            choice = nav.show_menu(display_title, menu_items, current_values)

            if choice == -1:
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif choice == -2:
                return
            else:
                menu_items[choice][1](stdscr)

    def edit_field(self, stdscr, tab_name: str, field_name: str) -> None:
        """Prompt the user to edit a specific field."""

        fields, _sections, _ = self._get_tab_structure(tab_name)
        field_dict = next((field for field in fields if field["name"] == field_name), None)
        if not field_dict:
            self.show_error(stdscr, f"Field '{field_name}' is not available in the current context.")
            return

        if field_dict.get("disabled"):
            description = field_dict.get("description") or "This field is disabled based on current selections."
            self.show_message(stdscr, description)
            return

        field_def = self.state.field_defs.get(field_name)
        field_type = field_dict.get("type", "text")
        current_value = field_dict.get("value", self.state.get_value(field_name))

        prompt_parts = [field_dict.get("label", field_name)]
        if field_dict.get("description"):
            prompt_parts.append("")
            prompt_parts.append(field_dict["description"])
        prompt = "\n".join(prompt_parts)

        if field_type == "checkbox":
            new_value = self._prompt_checkbox(stdscr, prompt, current_value)
        elif field_type == "select":
            new_value = self._prompt_select(stdscr, prompt, field_dict, current_value)
        elif field_type == "multi_select":
            new_value = self._prompt_multi_select(stdscr, prompt, field_dict, current_value)
        elif field_type == "number":
            new_value = self._prompt_number(stdscr, prompt, field_def, current_value)
        elif field_type == "textarea":
            new_value = self._prompt_text(stdscr, prompt, current_value, multiline=True)
        else:
            new_value = self._prompt_text(stdscr, prompt, current_value)

        if new_value is None:
            return

        if self._validate_and_set_field(stdscr, field_name, new_value):
            self.show_message(stdscr, "Field updated.")

    def _prompt_checkbox(self, stdscr, prompt: str, current_value: Any) -> Optional[bool]:
        options = ["Enabled", "Disabled"]
        default_idx = 0 if self._coerce_bool(current_value) else 1
        selected = self.show_options(stdscr, prompt, options, default_idx)
        if selected == -1:
            return None
        return selected == 0

    def _prompt_select(
        self,
        stdscr,
        prompt: str,
        field_dict: Dict[str, Any],
        current_value: Any,
    ) -> Optional[Any]:
        options = field_dict.get("options") or []
        if not options:
            text_default = "" if current_value in (None, "") else str(current_value)
            return self.get_input(stdscr, f"{prompt}\n\nEnter value:", text_default)

        display_options: List[str] = []
        values: List[Any] = []
        default_idx = 0

        for idx, option in enumerate(options):
            value = option.get("value")
            label = option.get("label") or str(value)
            display_options.append(label)
            values.append(value)
            if self._values_equal(value, current_value):
                default_idx = idx

        display_options.append("Enter custom value…")
        selected = self.show_options(stdscr, prompt, display_options, default_idx)
        if selected == -1:
            return None
        if selected == len(display_options) - 1:
            text_default = "" if current_value in (None, "") else str(current_value)
            custom_value = self.get_input(stdscr, "Enter custom value:", text_default)
            return custom_value if custom_value != "" else None
        return values[selected]

    def _prompt_multi_select(
        self,
        stdscr,
        prompt: str,
        field_dict: Dict[str, Any],
        current_value: Any,
    ) -> Optional[List[str]]:
        if isinstance(current_value, (list, tuple)):
            default_text = ", ".join(str(item) for item in current_value)
        elif current_value:
            default_text = str(current_value)
        else:
            default_text = ""

        options = field_dict.get("options") or []
        option_labels = ", ".join(str(option.get("value")) for option in options)
        extended_prompt = prompt
        if option_labels:
            extended_prompt = f"{prompt}\n\nAvailable options: {option_labels}"
        response = self.get_input(stdscr, extended_prompt, default_text)
        if response.strip() == "":
            return []
        return [item.strip() for item in response.split(",") if item.strip()]

    def _prompt_number(
        self,
        stdscr,
        prompt: str,
        field_def: Optional[Any],
        current_value: Any,
    ) -> Optional[Any]:
        default_value = field_def.default_value if field_def else None
        default_str = "" if current_value in (None, "") else str(current_value)
        response = self.get_input(stdscr, prompt, default_str)
        if response.strip() == "":
            return default_value

        try:
            if field_def and isinstance(field_def.default_value, float):
                return float(response)
            if "." in response.strip():
                return float(response)
            return int(response)
        except ValueError:
            self.show_error(stdscr, "Please enter a valid numeric value.")
            return None

    def _prompt_text(
        self,
        stdscr,
        prompt: str,
        current_value: Any,
        multiline: bool = False,
    ) -> Optional[str]:
        default_str = "" if current_value in (None, "None") else str(current_value)
        response = self.get_input(stdscr, prompt, default_str, multiline=multiline)
        return response

    def _validate_and_set_field(self, stdscr, field_name: str, value: Any) -> bool:
        context = self.state.as_config_data()
        context[field_name] = value
        context[f"--{field_name}"] = value

        field_def = self.state.field_defs.get(field_name)
        if field_def:
            arg_name = getattr(field_def, "arg_name", "")
            if arg_name:
                context[arg_name] = value

        try:
            errors = self.field_service.field_registry.validate_field_value(field_name, value, context)
        except Exception:
            errors = []

        if errors:
            self.show_error(stdscr, "\n".join(errors))
            return False

        self.state.set_value(field_name, value)
        return True

    def _format_field_value(self, field_dict: Dict[str, Any]) -> str:
        value = field_dict.get("value")
        field_type = field_dict.get("type", "text")

        if field_type == "checkbox":
            return "Enabled" if self._coerce_bool(value) else "Disabled"
        if field_type == "number":
            return str(value) if value not in (None, "") else "Not set"
        if field_type == "select":
            options = field_dict.get("options") or []
            label = next(
                (opt.get("label") for opt in options if self._values_equal(opt.get("value"), value)),
                None,
            )
            if label:
                return label
        if field_type == "multi_select":
            if not value:
                return "Not set"
            if isinstance(value, (list, tuple, set)):
                return ", ".join(str(item) for item in value)
        if value in (None, "", "Not configured"):
            return "Not set"
        return str(value)

    def review_and_save(self, stdscr) -> None:
        """Show current configuration and optionally write it to disk."""

        config_data = self.state.to_serializable()
        self._display_json(stdscr, config_data)
        choice = self.show_options(
            stdscr,
            "Choose an action:",
            ["Save to file", "Back"],
            0,
        )
        if choice == 0:
            self._save_config(stdscr, config_data)

    def _display_json(self, stdscr, data: Dict[str, Any]) -> None:
        """Render configuration data with simple scrolling."""

        text = json.dumps(data, indent=2, sort_keys=True)
        lines = text.splitlines()
        start = 0

        while True:
            stdscr.clear()
            h, w = stdscr.getmaxyx()
            title = "Review Configuration"
            stdscr.addstr(1, (w - len(title)) // 2, title, curses.A_BOLD)

            max_lines = h - 5
            y = 3
            displayed = 0
            idx = start

            while displayed < max_lines and idx < len(lines):
                wrapped = textwrap.wrap(lines[idx], w - 4) or [""]
                for chunk in wrapped:
                    if displayed >= max_lines:
                        break
                    stdscr.addstr(y + displayed, 2, chunk)
                    displayed += 1
                idx += 1

            stdscr.addstr(h - 2, 2, "↑/↓ scroll  Enter to continue", curses.A_DIM)
            stdscr.refresh()

            key = stdscr.getch()
            if key in (curses.KEY_ENTER, ord("\n"), ord("\r"), 27, ord("q")):
                return
            if key == curses.KEY_UP and start > 0:
                start -= 1
            elif key == curses.KEY_DOWN and start < max(0, len(lines) - 1):
                start += 1

    def _save_config(self, stdscr, config_data: Dict[str, Any]) -> None:
        """Persist configuration to disk."""

        default_path = self.state.loaded_config_path or "config/config.json"
        path = self.get_input(stdscr, "Enter file path to save config:", default_path)
        if not path:
            self.show_error(stdscr, "Path cannot be empty.")
            return

        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                json.dump(config_data, handle, indent=4, sort_keys=True)
            self.state.loaded_config_path = str(target)
            self.show_message(stdscr, f"Configuration saved to {target}")
        except Exception as exc:
            self.show_error(stdscr, f"Failed to save configuration: {exc}")

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "on"}
        return False

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        if left == right:
            return True
        if isinstance(left, str) and isinstance(right, str):
            return left.strip() == right.strip()
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return float(left) == float(right)
        return False


def main() -> None:
    """CLI entry point."""

    config_path: Optional[str] = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    try:
        configurator = SimpleTunerNCurses()
    except Exception as exc:
        print(f"Unable to start configurator: {exc}")
        traceback.print_exc()
        sys.exit(1)

    if config_path:
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        if configurator.state.load_from_file(config_path):
            print(f"Loaded configuration from: {config_path}")
        else:
            print(f"Error: Failed to load config: {config_path}")
            sys.exit(1)
    elif configurator.state.loaded_config_path:
        print(f"Loaded existing configuration from: {configurator.state.loaded_config_path}")
    else:
        print("No existing configuration found. Starting fresh setup.")

    configurator.run()


if __name__ == "__main__":
    main()
