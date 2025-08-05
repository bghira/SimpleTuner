#!/usr/bin/env python3
"""
SimpleTuner End-to-End Test Runner
A comprehensive testing framework with ncurses UI for running all example configurations
"""

import argparse
import curses
import json
import os
import subprocess
import sys
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import signal
import re
from dataclasses import dataclass
from enum import Enum


class TestStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    log_file: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class SimpleTunerTestRunner:
    def __init__(self, resumable: bool = False, parallel: int = 1):
        self.resumable = resumable
        self.parallel = parallel
        self.tests: Dict[str, TestResult] = {}
        self.log_queue = queue.Queue()
        self.current_test = None
        self.debug_log_path = Path("debug.log")
        self.state_file = Path(".test_runner_state.json")
        self.should_exit = False
        self.skip_current_test = False  # New flag for skipping individual tests
        self.tests_lock = threading.Lock()  # Thread safety
        self.test_processes = {}  # Track running processes

        # UI state
        self.selected_test = 0
        self.log_scroll = 0
        self.debug_log_scroll = 0
        self.test_viewport_start = 0  # Viewport tracking for test list

        # UI toggles
        self.show_test_list = False  # Hidden by default
        self.show_debug_log = False  # Hidden by default
        self.show_help = True  # Show help initially

        # Ensure debug log exists
        self.debug_log_path.touch(exist_ok=True)

    def discover_examples(self) -> List[str]:
        """Discover all example configurations"""
        examples_dir = Path("config/examples")
        if not examples_dir.exists():
            return []

        examples = []
        for item in examples_dir.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                examples.append(item.name)

        return sorted(examples)

    def load_state(self):
        """Load previous test state for resumable runs"""
        if self.resumable and self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    with self.tests_lock:
                        for name, data in state.items():
                            self.tests[name] = TestResult(
                                name=name,
                                status=TestStatus(data["status"]),
                                start_time=(
                                    datetime.fromisoformat(data["start_time"])
                                    if data.get("start_time")
                                    else None
                                ),
                                end_time=(
                                    datetime.fromisoformat(data["end_time"])
                                    if data.get("end_time")
                                    else None
                                ),
                                error=data.get("error"),
                                log_file=data.get("log_file"),
                            )
            except Exception as e:
                self.log_queue.put(f"Failed to load state: {e}")

    def save_state(self):
        """Save current test state"""
        with self.tests_lock:
            state = {}
            for name, result in self.tests.items():
                state[name] = {
                    "status": result.status.value,
                    "start_time": (
                        result.start_time.isoformat() if result.start_time else None
                    ),
                    "end_time": (
                        result.end_time.isoformat() if result.end_time else None
                    ),
                    "error": result.error,
                    "log_file": result.log_file,
                }

        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.log_queue.put(f"Failed to save state: {e}")

    def setup_environment(self, example_name: str) -> Dict[str, str]:
        """Set up environment for a specific example"""
        env = os.environ.copy()
        env["ENV"] = f"examples/{example_name}"

        # Create output directory for this test
        output_dir = Path(f"test_outputs/{example_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        return env

    def run_test(self, example_name: str, force_rerun: bool = False) -> TestResult:
        """Run a single test"""
        with self.tests_lock:
            if example_name not in self.tests:
                self.tests[example_name] = TestResult(
                    name=example_name, status=TestStatus.PENDING
                )
            result = self.tests[example_name]

        # Skip if resumable and already completed (unless force_rerun)
        if (
            not force_rerun
            and self.resumable
            and result.status in [TestStatus.SUCCESS, TestStatus.SKIPPED]
        ):
            self.log_queue.put(f"Skipping {example_name} (already completed)")
            return result

        with self.tests_lock:
            result.status = TestStatus.RUNNING
            result.start_time = datetime.now()
            result.log_file = f"test_outputs/{example_name}/training.log"
            result.error = None  # Clear any previous error

        self.log_queue.put(f"Starting test: {example_name}")

        try:
            # Set up environment
            env = self.setup_environment(example_name)

            # Run the training script
            cmd = ["./train.sh"]

            # Log the command being run
            self.log_queue.put(
                f"Running command: {' '.join(cmd)} with ENV={example_name}"
            )

            # Create log file
            Path(result.log_file).parent.mkdir(parents=True, exist_ok=True)

            with open(result.log_file, "w") as log_file:
                log_file.write(
                    f"=== Test started at {datetime.now().isoformat()} ===\n"
                )
                log_file.flush()

                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=0,  # Unbuffered for immediate output
                    universal_newlines=True,
                    preexec_fn=(
                        os.setsid if sys.platform != "win32" else None
                    ),  # Create new process group
                )

                # Track this process
                self.test_processes[example_name] = process

                # Monitor process
                while process.poll() is None:
                    # Check if we should skip this specific test
                    if self.skip_current_test and self.current_test == example_name:
                        self.log_queue.put(f"Skipping test: {example_name}")
                        try:
                            if sys.platform != "win32":
                                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            else:
                                process.terminate()
                        except:
                            process.terminate()
                        with self.tests_lock:
                            result.status = TestStatus.SKIPPED
                            result.error = "Test skipped by user"
                        self.skip_current_test = False  # Reset flag
                        break

                    # Check if we should exit entirely
                    if self.should_exit:
                        try:
                            if sys.platform != "win32":
                                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            else:
                                process.terminate()
                        except:
                            process.terminate()
                        with self.tests_lock:
                            result.status = TestStatus.SKIPPED
                            result.error = "Test runner interrupted"
                        break

                    # Force flush to ensure we can read the log
                    log_file.flush()
                    time.sleep(0.1)

                # Clean up process tracking
                if example_name in self.test_processes:
                    del self.test_processes[example_name]

                # Set final status if not already set
                if result.status == TestStatus.RUNNING:
                    if process.returncode == 0:
                        with self.tests_lock:
                            result.status = TestStatus.SUCCESS
                    else:
                        with self.tests_lock:
                            result.status = TestStatus.FAILED
                            result.error = (
                                f"Process exited with code {process.returncode}"
                            )

        except Exception as e:
            with self.tests_lock:
                result.status = TestStatus.FAILED
                result.error = str(e)
            self.log_queue.put(f"Test {example_name} failed: {e}")

        with self.tests_lock:
            result.end_time = datetime.now()
        self.save_state()

        return result

    def run_all_tests(self, examples: List[str]):
        """Run all tests in sequence"""
        # Ensure all tests are initialized
        with self.tests_lock:
            for example in examples:
                if example not in self.tests:
                    self.tests[example] = TestResult(
                        name=example, status=TestStatus.PENDING
                    )

        for example in examples:
            if self.should_exit:
                break

            self.current_test = example
            self.run_test(example)
            self.current_test = None

    def draw_header(self, stdscr, width: int):
        """Draw the header"""
        header = "SimpleTuner End-to-End Test Runner"
        try:
            # Clear header area only
            stdscr.move(0, 0)
            stdscr.clrtoeol()
            stdscr.move(1, 0)
            stdscr.clrtoeol()

            stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(0, (width - len(header)) // 2, header)
            stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
        except curses.error:
            pass

        # Status line
        with self.tests_lock:
            tests_list = list(self.tests.values())
            current = self.current_test or "None"
            if 0 <= self.selected_test < len(tests_list):
                viewing = tests_list[self.selected_test].name
            else:
                viewing = "None"

            status = f"Current: {current} | Viewing: {viewing} | "
            status += (
                f"✓:{len([t for t in tests_list if t.status == TestStatus.SUCCESS])} "
            )
            status += (
                f"✗:{len([t for t in tests_list if t.status == TestStatus.FAILED])} "
            )
            status += (
                f"●:{sum(1 for t in tests_list if t.status == TestStatus.RUNNING)} "
            )
            status += (
                f"○:{sum(1 for t in tests_list if t.status == TestStatus.PENDING)}"
            )

        try:
            stdscr.addstr(1, 2, status[: width - 4])
        except curses.error:
            pass

    def draw_test_overlay(self, stdscr, height: int, width: int):
        """Draw the test list as an overlay"""
        # Calculate overlay dimensions
        overlay_height = min(height - 8, len(self.tests) + 4)
        overlay_width = min(width - 10, 60)
        start_y = (height - overlay_height) // 2
        start_x = (width - overlay_width) // 2

        # Create overlay window
        overlay = curses.newwin(overlay_height, overlay_width, start_y, start_x)
        overlay.box()

        # Title
        title = " Test Status [t: toggle] "
        try:
            overlay.attron(curses.color_pair(2) | curses.A_BOLD)
            overlay.addstr(0, 2, title)
            overlay.attroff(curses.color_pair(2) | curses.A_BOLD)
        except curses.error:
            pass

        # Test list
        with self.tests_lock:
            visible_tests = list(self.tests.items())

        # Ensure selected_test is within bounds
        self.selected_test = max(0, min(self.selected_test, len(visible_tests) - 1))

        # Calculate viewport
        viewport_height = overlay_height - 3

        # Ensure selected test is in view
        if self.selected_test < self.test_viewport_start:
            self.test_viewport_start = self.selected_test
        elif self.selected_test >= self.test_viewport_start + viewport_height:
            self.test_viewport_start = self.selected_test - viewport_height + 1

        # Clamp viewport start
        self.test_viewport_start = max(
            0,
            min(self.test_viewport_start, max(0, len(visible_tests) - viewport_height)),
        )

        # Draw visible tests
        for i in range(viewport_height):
            test_idx = self.test_viewport_start + i
            if test_idx >= len(visible_tests):
                break

            y = i + 1
            name, result = visible_tests[test_idx]

            # Status indicator
            status_char = {
                TestStatus.PENDING: "○",
                TestStatus.RUNNING: "●",
                TestStatus.SUCCESS: "✓",
                TestStatus.FAILED: "✗",
                TestStatus.SKIPPED: "⊘",
            }.get(result.status, "?")

            # Colors
            color = {
                TestStatus.PENDING: 0,
                TestStatus.RUNNING: 3,
                TestStatus.SUCCESS: 4,
                TestStatus.FAILED: 5,
                TestStatus.SKIPPED: 6,
            }.get(result.status, 0)

            # Highlight selected
            is_selected = test_idx == self.selected_test
            if is_selected:
                overlay.attron(curses.A_REVERSE)

            if color:
                overlay.attron(curses.color_pair(color))

            # Format line
            duration = f"{result.duration:.1f}s" if result.duration else "---"
            max_name_len = overlay_width - 15
            truncated_name = name[:max_name_len]
            line = f" {status_char} {truncated_name:<{max_name_len}} {duration:>8}"

            try:
                overlay.addstr(y, 1, line[: overlay_width - 2])
            except curses.error:
                pass

            if color:
                overlay.attroff(curses.color_pair(color))

            if is_selected:
                overlay.attroff(curses.A_REVERSE)

        overlay.refresh()

    def draw_main_log(self, win, height: int, width: int):
        """Draw the main training log"""
        win.clear()
        win.box()

        # Get selected test
        selected = None
        with self.tests_lock:
            tests_list = list(self.tests.values())
            if tests_list and 0 <= self.selected_test < len(tests_list):
                selected = tests_list[self.selected_test]

        # Title
        title = f" Training Log: {selected.name if selected else 'None'} "
        try:
            win.attron(curses.color_pair(2) | curses.A_BOLD)
            win.addstr(0, 2, title[: width - 4])
            win.attroff(curses.color_pair(2) | curses.A_BOLD)
        except curses.error:
            pass

        # Log content
        if selected and selected.log_file and Path(selected.log_file).exists():
            try:
                with open(
                    selected.log_file, "r", encoding="utf-8", errors="ignore"
                ) as f:
                    # Seek to end first to get total size
                    f.seek(0, 2)
                    file_size = f.tell()
                    f.seek(0)

                    lines = f.readlines()

                # Show last N lines
                visible_lines = height - 3
                start_line = max(0, len(lines) - visible_lines - self.log_scroll)
                end_line = start_line + visible_lines

                for i, line in enumerate(lines[start_line:end_line]):
                    if i + 1 >= height - 1:
                        break
                    try:
                        # Color code certain patterns
                        color = 0
                        if "error" in line.lower() or "failed" in line.lower():
                            color = 5
                        elif "warning" in line.lower():
                            color = 3
                        elif "success" in line.lower() or "complete" in line.lower():
                            color = 4

                        clean_line = line.rstrip()[: width - 3]

                        if color:
                            win.attron(curses.color_pair(color))
                        win.addstr(i + 1, 1, clean_line)
                        if color:
                            win.attroff(curses.color_pair(color))
                    except curses.error:
                        pass

                # Scroll indicators
                if self.log_scroll < len(lines) - visible_lines:
                    try:
                        win.addstr(1, width - 3, "↑")
                    except curses.error:
                        pass
                if self.log_scroll > 0:
                    try:
                        win.addstr(height - 2, width - 3, "↓")
                    except curses.error:
                        pass

            except Exception as e:
                try:
                    win.addstr(1, 1, f"Error reading log: {e}"[: width - 3])
                except curses.error:
                    pass
        else:
            try:
                msg = "No log available - select a test with ↑↓"
                win.addstr(height // 2, max(1, (width - len(msg)) // 2), msg)
            except curses.error:
                pass

        win.refresh()

    def draw_debug_log(self, win, height: int, width: int):
        """Draw the debug.log panel"""
        win.clear()
        win.box()

        # Title
        title = " Debug Log [d: toggle] "
        try:
            win.attron(curses.color_pair(2) | curses.A_BOLD)
            win.addstr(0, 2, title[: width - 4])
            win.attroff(curses.color_pair(2) | curses.A_BOLD)
        except curses.error:
            pass

        # Debug log content
        if self.debug_log_path.exists():
            try:
                with open(
                    self.debug_log_path, "r", encoding="utf-8", errors="ignore"
                ) as f:
                    lines = f.readlines()

                # Show last N lines
                visible_lines = height - 3
                start_line = max(0, len(lines) - visible_lines - self.debug_log_scroll)
                end_line = start_line + visible_lines

                for i, line in enumerate(lines[start_line:end_line]):
                    if i + 1 >= height - 1:
                        break

                    # Color code by log level
                    color = 0
                    if "ERROR" in line:
                        color = 5
                    elif "WARNING" in line:
                        color = 3
                    elif "INFO" in line or "Starting test:" in line:
                        color = 4
                    elif "===" in line:
                        color = 2

                    try:
                        if color:
                            win.attron(curses.color_pair(color))
                        win.addstr(i + 1, 1, line.rstrip()[: width - 3])
                        if color:
                            win.attroff(curses.color_pair(color))
                    except curses.error:
                        pass

                # Scroll indicators
                if self.debug_log_scroll < len(lines) - visible_lines:
                    try:
                        win.addstr(1, width - 3, "↑")
                    except curses.error:
                        pass
                if self.debug_log_scroll > 0:
                    try:
                        win.addstr(height - 2, width - 3, "↓")
                    except curses.error:
                        pass

            except Exception as e:
                try:
                    win.addstr(1, 1, f"Error reading debug log: {e}"[: width - 3])
                except curses.error:
                    pass
        else:
            try:
                msg = "Debug log not available"
                win.addstr(height // 2, max(1, (width - len(msg)) // 2), msg)
            except curses.error:
                pass

        win.refresh()

    def draw_footer(self, stdscr, y: int, width: int):
        """Draw the footer with help"""
        help_text = (
            "q:Quit | t:Tests | d:Debug | ↑↓:Navigate | r:Rerun | s:Skip | h:Help"
        )
        try:
            stdscr.hline(y, 0, curses.ACS_HLINE, width)
            stdscr.addstr(y + 1, 2, help_text[: width - 4])
        except curses.error:
            pass

    def draw_help_overlay(self, stdscr, height: int, width: int):
        """Draw help overlay"""
        help_lines = [
            "SimpleTuner Test Runner Help",
            "",
            "Navigation:",
            "  ↑/↓    - Select test",
            "  PgUp/Dn- Scroll logs",
            "",
            "Controls:",
            "  t      - Toggle test list overlay",
            "  d      - Toggle debug log panel",
            "  h      - Toggle this help",
            "  r      - Rerun selected test",
            "  s      - Skip current test",
            "  q      - Quit",
            "",
            "Status Icons:",
            "  ○ Pending  ● Running  ✓ Success",
            "  ✗ Failed   ⊘ Skipped",
            "",
            "Press any key to continue...",
        ]

        # Calculate overlay dimensions
        overlay_height = len(help_lines) + 2
        overlay_width = max(len(line) for line in help_lines) + 4
        start_y = (height - overlay_height) // 2
        start_x = (width - overlay_width) // 2

        # Create overlay window
        overlay = curses.newwin(overlay_height, overlay_width, start_y, start_x)
        overlay.box()

        # Draw help text
        for i, line in enumerate(help_lines):
            try:
                if i == 0:
                    overlay.attron(curses.A_BOLD)
                overlay.addstr(i + 1, 2, line)
                if i == 0:
                    overlay.attroff(curses.A_BOLD)
            except curses.error:
                pass

        overlay.refresh()
        overlay.getch()  # Wait for any key

    def run_ui(self, stdscr):
        """Main UI loop"""
        # Initialize colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Header
        curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Panel titles
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Running
        curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Success
        curses.init_pair(5, curses.COLOR_RED, curses.COLOR_BLACK)  # Failed
        curses.init_pair(6, curses.COLOR_MAGENTA, curses.COLOR_BLACK)  # Skipped

        # Set up UI
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.nodelay(True)
        stdscr.clear()

        # Discover examples
        examples = self.discover_examples()
        if not examples:
            stdscr.addstr(10, 10, "No examples found in config/examples/")
            stdscr.addstr(11, 10, "Press any key to exit...")
            stdscr.refresh()
            stdscr.nodelay(False)
            stdscr.getch()
            return

        # Initialize tests BEFORE loading state
        with self.tests_lock:
            for example in examples:
                if example not in self.tests:
                    self.tests[example] = TestResult(
                        name=example, status=TestStatus.PENDING
                    )

        # Load state if resumable (this might override some test statuses)
        if self.resumable:
            self.load_state()

        # Ensure selected_test is valid
        self.selected_test = 0

        # Start debug log
        with open(self.debug_log_path, "a") as f:
            f.write(
                f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Test Runner Started ===\n"
            )
            f.write(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Found {len(examples)} examples: {', '.join(examples)}\n"
            )

        # Start test runner thread
        test_thread = threading.Thread(target=self.run_all_tests, args=(examples,))
        test_thread.daemon = True
        test_thread.start()

        # Show help on first run
        if self.show_help:
            self.draw_help_overlay(stdscr, *stdscr.getmaxyx())
            self.show_help = False

        # Initialize windows
        main_win = None
        debug_win = None
        last_height = 0
        last_width = 0
        force_redraw = True

        # UI loop
        while True:
            height, width = stdscr.getmaxyx()

            # Ensure minimum dimensions
            if height < 10 or width < 30:
                stdscr.clear()
                try:
                    stdscr.addstr(0, 0, "Terminal too small! (min 30x10)")
                except curses.error:
                    pass
                stdscr.refresh()
                key = stdscr.getch()
                if key == ord("q"):
                    self.should_exit = True
                    break
                time.sleep(0.1)
                continue

            # Check if we need to recreate windows (terminal resized)
            if height != last_height or width != last_width or force_redraw:
                stdscr.clear()
                stdscr.refresh()

                # Calculate window dimensions
                header_height = 2
                footer_height = 2

                if self.show_debug_log:
                    # Split horizontally
                    main_height = (height - header_height - footer_height) // 2
                    debug_height = height - header_height - footer_height - main_height

                    main_win = curses.newwin(main_height, width, header_height, 0)
                    debug_win = curses.newwin(
                        debug_height, width, header_height + main_height, 0
                    )
                else:
                    # Full screen for main log
                    main_height = height - header_height - footer_height
                    main_win = curses.newwin(main_height, width, header_height, 0)
                    debug_win = None

                last_height = height
                last_width = width
                force_redraw = False

            # Draw header
            self.draw_header(stdscr, width)

            # Draw main log
            if main_win:
                self.draw_main_log(main_win, main_win.getmaxyx()[0], width)

            # Draw debug log if enabled
            if self.show_debug_log and debug_win:
                self.draw_debug_log(debug_win, debug_win.getmaxyx()[0], width)

            # Draw footer
            self.draw_footer(stdscr, height - 2, width)

            # Draw test overlay if enabled
            if self.show_test_list:
                self.draw_test_overlay(stdscr, height, width)

            # Refresh
            stdscr.refresh()

            # Handle input
            key = stdscr.getch()

            if key != -1:  # Key was pressed
                if key == ord("q"):
                    self.should_exit = True
                    break
                elif key == ord("t"):
                    self.show_test_list = not self.show_test_list
                elif key == ord("d"):
                    self.show_debug_log = not self.show_debug_log
                    force_redraw = True
                elif key == ord("h"):
                    self.draw_help_overlay(stdscr, height, width)
                elif key == curses.KEY_UP:
                    with self.tests_lock:
                        self.selected_test = max(0, self.selected_test - 1)
                    self.log_scroll = 0  # Reset scroll when changing tests
                elif key == curses.KEY_DOWN:
                    with self.tests_lock:
                        self.selected_test = min(
                            len(self.tests) - 1, self.selected_test + 1
                        )
                    self.log_scroll = 0  # Reset scroll when changing tests
                elif key == curses.KEY_PPAGE:  # Page Up
                    if self.show_debug_log and not self.show_test_list:
                        self.debug_log_scroll = min(self.debug_log_scroll + 10, 1000)
                    else:
                        self.log_scroll = min(self.log_scroll + 10, 1000)
                elif key == curses.KEY_NPAGE:  # Page Down
                    if self.show_debug_log and not self.show_test_list:
                        self.debug_log_scroll = max(0, self.debug_log_scroll - 10)
                    else:
                        self.log_scroll = max(0, self.log_scroll - 10)
                elif key == ord("r"):  # Rerun selected test
                    with self.tests_lock:
                        tests_list = list(self.tests.values())
                        if tests_list and 0 <= self.selected_test < len(tests_list):
                            selected = tests_list[self.selected_test]
                            test_name = selected.name

                            # Check if this test is already running
                            if selected.status == TestStatus.RUNNING:
                                self.log_queue.put(
                                    f"Test {test_name} is already running"
                                )
                            else:
                                # Reset status and start new thread for rerun
                                selected.status = TestStatus.PENDING
                                threading.Thread(
                                    target=self.run_test,
                                    args=(test_name, True),  # force_rerun=True
                                ).start()
                                self.log_queue.put(f"Rerunning test: {test_name}")

                elif key == ord("s"):  # Skip current test
                    if self.current_test:
                        self.skip_current_test = True
                        self.log_queue.put(f"Skip requested for: {self.current_test}")
                elif key == curses.KEY_RESIZE:
                    force_redraw = True

            # Process log messages
            try:
                while True:
                    msg = self.log_queue.get_nowait()
                    with open(self.debug_log_path, "a") as f:
                        f.write(
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
                        )
                        f.flush()
            except queue.Empty:
                pass

            time.sleep(0.05)

        # Wait for test thread to finish
        self.should_exit = True
        test_thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description="SimpleTuner End-to-End Test Runner")
    parser.add_argument(
        "--resumable", action="store_true", help="Resume from previous run"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel tests (not implemented yet)",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean all test outputs before starting"
    )

    args = parser.parse_args()

    # Clean if requested
    if args.clean:
        import shutil

        if Path("test_outputs").exists():
            shutil.rmtree("test_outputs")
        if Path(".test_runner_state.json").exists():
            os.remove(".test_runner_state.json")
        if Path("debug.log").exists():
            os.remove("debug.log")
        if Path("cache").exists():
            shutil.rmtree("cache")
        if Path("output/examples").exists():
            shutil.rmtree("output/examples")

    # Ensure virtual environment is set up BEFORE starting UI
    if not Path(".venv").exists():
        print("Setting up virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        subprocess.run([".venv/bin/pip", "install", "poetry"], check=True)
        subprocess.run([".venv/bin/poetry", "install"], check=True)
        print("Virtual environment ready.")

    # Run the test runner
    runner = SimpleTunerTestRunner(resumable=args.resumable, parallel=args.parallel)

    try:
        curses.wrapper(runner.run_ui)
    except KeyboardInterrupt:
        print("\nTest runner interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for name, result in runner.tests.items():
        status_icon = {
            TestStatus.SUCCESS: "✓",
            TestStatus.FAILED: "✗",
            TestStatus.SKIPPED: "⊘",
            TestStatus.PENDING: "○",
            TestStatus.RUNNING: "●",
        }.get(result.status, "?")

        duration = f"{result.duration:.1f}s" if result.duration else "---"
        print(f"{status_icon} {name:<40} {duration:>10} {result.status.value}")
        if result.error:
            print(f"  └─ Error: {result.error}")

    print("=" * 80)

    # Exit with appropriate code
    failed = sum(1 for t in runner.tests.values() if t.status == TestStatus.FAILED)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
