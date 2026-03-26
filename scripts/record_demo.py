#!/usr/bin/env python3
"""Record an asciinema demo of a Claude Code session using Transolver-3 skills.

Drives a real Claude Code agent via claude_agent_sdk, captures all events
(text, tool use, tool results), and renders them into an asciinema v2 .cast
file with syntax-highlighted, terminal-styled output.

Usage:
    # Record with default scenario (data inspection + training + analysis)
    python scripts/record_demo.py

    # Record a specific scenario
    python scripts/record_demo.py --scenario data

    # Custom prompt
    python scripts/record_demo.py --prompt "Inspect the mesh at /Volumes/ml/transolver3/data/drivaer_001.npz"

    # List available scenarios
    python scripts/record_demo.py --list-scenarios

Requirements:
    pip install claude-agent-sdk>=0.1.39

The generated .cast file can be played with:
    asciinema play demo.cast
    # or embedded in a web page via <asciinema-player>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── ANSI color helpers ──────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"

# Databricks-inspired palette
ORANGE = "\033[38;5;208m"      # Databricks orange
BLUE = "\033[38;5;39m"         # Bright blue
GREEN = "\033[38;5;78m"        # Success green
RED = "\033[38;5;196m"         # Error red
YELLOW = "\033[38;5;220m"      # Warning yellow
CYAN = "\033[38;5;81m"         # Info cyan
GRAY = "\033[38;5;245m"        # Dim gray
WHITE = "\033[38;5;255m"       # Bright white
MAGENTA = "\033[38;5;176m"     # Tool names

BG_DARK = "\033[48;5;235m"     # Dark background for code blocks
BG_TOOL = "\033[48;5;236m"     # Slightly lighter for tool blocks


# ── Asciinema cast writer ───────────────────────────────────────────────────

@dataclass
class CastWriter:
    """Writes asciinema v2 .cast files with simulated typing effects."""

    width: int = 120
    height: int = 40
    title: str = "Transolver-3 on Databricks — Claude Code Demo"
    output_path: str = "demo.cast"

    _events: list[tuple[float, str, str]] = field(default_factory=list)
    _start_time: float = 0.0
    _current_time: float = 0.0

    def start(self):
        self._start_time = time.monotonic()
        self._current_time = 0.0
        self._events = []

    def _ts(self) -> float:
        return self._current_time

    def advance(self, seconds: float):
        """Advance the virtual clock."""
        self._current_time += seconds

    def write_output(self, text: str):
        """Write terminal output at current timestamp."""
        if text:
            self._events.append((self._ts(), "o", text))

    def write_typing(self, text: str, char_delay: float = 0.04):
        """Simulate typing character by character."""
        for ch in text:
            self._events.append((self._ts(), "o", ch))
            self.advance(char_delay)

    def write_line(self, text: str):
        """Write a full line with newline."""
        self.write_output(text + "\r\n")

    def write_blank(self):
        self.write_output("\r\n")

    def save(self):
        """Write the .cast file."""
        header = {
            "version": 2,
            "width": self.width,
            "height": self.height,
            "timestamp": int(time.time()),
            "title": self.title,
            "env": {"SHELL": "/bin/zsh", "TERM": "xterm-256color"},
            "theme": {
                "fg": "#d4d4d4",
                "bg": "#1e1e1e",
                "palette": "#1e1e1e:#f44747:#608b4e:#dcdcaa:#569cd6:#c586c0:#4ec9b0:#d4d4d4:#808080:#f44747:#608b4e:#dcdcaa:#569cd6:#c586c0:#4ec9b0:#ffffff",
            },
        }

        with open(self.output_path, "w") as f:
            f.write(json.dumps(header) + "\n")
            for ts, code, data in self._events:
                f.write(json.dumps([round(ts, 6), code, data]) + "\n")

        print(f"\nRecording saved to {self.output_path}")
        print(f"  Events: {len(self._events)}")
        print(f"  Duration: {self._current_time:.1f}s")
        print(f"\nPlay with: asciinema play {self.output_path}")


# ── Terminal rendering helpers ──────────────────────────────────────────────

def render_prompt(cast: CastWriter):
    """Render a terminal prompt."""
    cast.write_output(f"{BLUE}{BOLD}~/Transolver{RESET} {ORANGE}${RESET} ")


def render_user_prompt(cast: CastWriter, text: str):
    """Render the user typing a prompt to Claude."""
    cast.write_output(f"\r\n{ORANGE}{BOLD}╭─{RESET} {BOLD}Claude Code{RESET} {DIM}(Transolver-3){RESET}\r\n")
    cast.write_output(f"{ORANGE}{BOLD}│{RESET} ")
    cast.write_typing(text, char_delay=0.03)
    cast.write_output(f"\r\n{ORANGE}{BOLD}╰─▶{RESET}\r\n")
    cast.advance(0.3)


def render_assistant_text(cast: CastWriter, text: str, typing_speed: float = 0.008):
    """Render assistant text output with fast typing effect."""
    cast.write_blank()
    lines = text.split("\n")
    for i, line in enumerate(lines):
        # Render markdown-style formatting
        if line.startswith("## "):
            cast.write_output(f"  {BOLD}{CYAN}")
            cast.write_typing(line[3:], char_delay=typing_speed)
            cast.write_output(RESET)
        elif line.startswith("### "):
            cast.write_output(f"  {BOLD}{WHITE}")
            cast.write_typing(line[4:], char_delay=typing_speed)
            cast.write_output(RESET)
        elif line.startswith("- "):
            cast.write_output(f"  {GREEN}•{RESET} ")
            cast.write_typing(line[2:], char_delay=typing_speed)
        elif line.startswith("```"):
            cast.write_output(f"  {DIM}{'─' * 60}{RESET}")
        elif line.startswith("|"):
            cast.write_output(f"  {DIM}{line}{RESET}")
            cast.advance(typing_speed * 5)
        else:
            cast.write_output("  ")
            cast.write_typing(line, char_delay=typing_speed)
        cast.write_output("\r\n")
    cast.advance(0.2)


def render_tool_use(cast: CastWriter, tool_name: str, tool_input: dict):
    """Render a tool invocation block."""
    cast.write_blank()
    cast.write_output(f"  {MAGENTA}{BOLD}⚡ {tool_name}{RESET}")

    # Show key params based on tool type
    if tool_name == "Read":
        fp = tool_input.get("file_path", "")
        cast.write_output(f" {DIM}{fp}{RESET}")
    elif tool_name == "Bash":
        cmd = tool_input.get("command", "")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        cast.write_output(f" {DIM}{cmd}{RESET}")
    elif tool_name == "Write":
        fp = tool_input.get("file_path", "")
        cast.write_output(f" {DIM}{fp}{RESET}")
    elif tool_name == "Edit":
        fp = tool_input.get("file_path", "")
        cast.write_output(f" {DIM}{fp}{RESET}")
    elif tool_name == "Grep":
        pattern = tool_input.get("pattern", "")
        cast.write_output(f" {DIM}pattern={pattern}{RESET}")
    elif tool_name == "Glob":
        pattern = tool_input.get("pattern", "")
        cast.write_output(f" {DIM}{pattern}{RESET}")

    cast.write_output("\r\n")
    cast.advance(0.3)


def render_tool_result(cast: CastWriter, content: str, is_error: bool = False, max_lines: int = 12):
    """Render tool result output (truncated)."""
    if not content:
        return

    color = RED if is_error else GRAY
    lines = content.split("\n")
    shown = lines[:max_lines]

    for line in shown:
        # Truncate long lines
        if len(line) > 110:
            line = line[:107] + "..."
        cast.write_output(f"  {color}│{RESET} {DIM}{line}{RESET}\r\n")
        cast.advance(0.02)

    if len(lines) > max_lines:
        cast.write_output(f"  {color}│{RESET} {DIM}... ({len(lines) - max_lines} more lines){RESET}\r\n")

    cast.advance(0.2)


def render_thinking(cast: CastWriter, seconds: float = 1.5):
    """Render a thinking spinner."""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    steps = int(seconds / 0.1)
    for i in range(steps):
        frame = frames[i % len(frames)]
        cast.write_output(f"\r  {YELLOW}{frame}{RESET} {DIM}Thinking...{RESET}")
        cast.advance(0.1)
    cast.write_output(f"\r  {GREEN}✓{RESET} {DIM}Done{RESET}        \r\n")
    cast.advance(0.2)


def render_separator(cast: CastWriter):
    cast.write_blank()
    cast.write_output(f"  {DIM}{'─' * 70}{RESET}\r\n")
    cast.write_blank()
    cast.advance(0.3)


# ── Scenarios ───────────────────────────────────────────────────────────────

SCENARIOS = {
    "data": {
        "title": "Inspect and validate mesh data",
        "prompt": (
            "I have a mesh file at /Volumes/ml/transolver3/data/drivaer_001.npz. "
            "Can you inspect it, validate the data quality, and tell me what "
            "normalization I should use? Also estimate the GPU memory I'll need."
        ),
    },
    "train": {
        "title": "Train a model with the right config",
        "prompt": (
            "I want to train a Transolver-3 model on a 500K point mesh for "
            "automotive CFD (pressure + velocity). What config preset should I use? "
            "Set up the training notebook cells with MLflow tracking and mixed precision."
        ),
    },
    "analyze": {
        "title": "Analyze results and check for drift",
        "prompt": (
            "I just ran inference with my trained Transolver-3 model. The predictions "
            "are in a tensor called `predictions` with shape (1, 500000, 4) — channels "
            "are pressure, vx, vy, vz. Check if the predictions are physically plausible, "
            "compute per-channel error stats against `targets`, and check for drift."
        ),
    },
    "deploy": {
        "title": "Deploy model to Databricks serving",
        "prompt": (
            "My Transolver-3 model is trained and validated. Walk me through registering "
            "it in Unity Catalog, deploying a serving endpoint, and setting up monitoring "
            "with inference tables and drift detection."
        ),
    },
    "full": {
        "title": "End-to-end workflow demo",
        "prompt": (
            "I'm new to Transolver-3 on Databricks. Walk me through the complete workflow: "
            "1) inspect my mesh data at /Volumes/ml/transolver3/data/drivaer_001.npz, "
            "2) pick the right model config for my mesh size, "
            "3) show me how to train with MLflow tracking, "
            "4) explain how to analyze results and check for drift. "
            "Keep it practical with notebook-ready code."
        ),
    },
}


# ── Agent execution + recording ─────────────────────────────────────────────

async def run_and_record(
    prompt: str,
    cast: CastWriter,
    cwd: str | None = None,
    timeout_seconds: int = 600,
    model: str | None = None,
) -> None:
    """Run a Claude Code agent and record the session to asciinema."""
    try:
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
        from claude_agent_sdk.types import (
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )
    except ImportError:
        print("ERROR: claude-agent-sdk not installed.")
        print("Install with: pip install 'claude-agent-sdk>=0.1.39'")
        return

    # Load skills as system prompt
    skills_dir = Path(cwd or os.getcwd()) / "skills"
    skill_parts = []
    for skill_file in sorted(skills_dir.glob("transolver-*.md")):
        skill_parts.append(skill_file.read_text())
    system_prompt = "\n\n---\n\n".join(skill_parts) if skill_parts else ""

    # Build env (inherit Anthropic/Databricks vars)
    env = {}
    for key, value in os.environ.items():
        if key.startswith(("ANTHROPIC_", "CLAUDE_CODE_", "DATABRICKS_")):
            env[key] = value
    env.pop("CLAUDECODE", None)
    if model:
        env["ANTHROPIC_MODEL"] = model

    options = ClaudeAgentOptions(
        cwd=cwd or os.getcwd(),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="bypassPermissions",
        system_prompt=system_prompt,
        setting_sources=[],
        env=env,
    )

    # Start recording
    cast.start()

    # Render intro
    cast.write_output(f"{BOLD}{ORANGE}")
    cast.write_output("  ╔══════════════════════════════════════════════════════════════╗\r\n")
    cast.write_output("  ║         Transolver-3 on Databricks — Claude Code Demo       ║\r\n")
    cast.write_output("  ╚══════════════════════════════════════════════════════════════╝\r\n")
    cast.write_output(RESET)
    cast.advance(1.0)

    # Render the user prompt
    render_user_prompt(cast, prompt)
    render_thinking(cast, seconds=2.0)

    # Run the agent
    start_time = time.monotonic()

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for msg in client.receive_response():
                elapsed = time.monotonic() - start_time
                if elapsed > timeout_seconds:
                    render_assistant_text(cast, f"\n{RED}[Timeout after {timeout_seconds}s]{RESET}")
                    break

                if isinstance(msg, AssistantMessage):
                    for block in getattr(msg, "content", []):
                        if isinstance(block, TextBlock):
                            render_assistant_text(cast, block.text)
                        elif isinstance(block, ToolUseBlock):
                            render_tool_use(cast, block.name, block.input if isinstance(block.input, dict) else {})
                        elif isinstance(block, ToolResultBlock):
                            content = getattr(block, "content", "")
                            is_error = getattr(block, "is_error", False)
                            render_tool_result(cast, str(content), is_error)

                elif isinstance(msg, UserMessage):
                    for block in getattr(msg, "content", []):
                        if isinstance(block, ToolResultBlock):
                            content = getattr(block, "content", "")
                            is_error = getattr(block, "is_error", False)
                            render_tool_result(cast, str(content), is_error)

                elif isinstance(msg, ResultMessage):
                    duration = getattr(msg, "duration_ms", None)
                    cost = getattr(msg, "cost", None)
                    render_separator(cast)
                    parts = []
                    if duration:
                        parts.append(f"Duration: {duration/1000:.1f}s")
                    if cost:
                        parts.append(f"Cost: ${cost:.4f}")
                    if parts:
                        cast.write_output(f"  {DIM}{' | '.join(parts)}{RESET}\r\n")
                    cast.advance(0.5)

    except Exception as e:
        render_assistant_text(cast, f"\n{RED}Error: {e}{RESET}")

    # Outro
    cast.advance(1.0)
    cast.write_blank()
    cast.write_output(f"  {GREEN}{BOLD}Demo complete.{RESET} {DIM}See skills/ for full documentation.{RESET}\r\n")
    cast.advance(2.0)

    cast.save()


def _run_in_fresh_loop(coro):
    """Run async coroutine in a fresh event loop on a dedicated thread."""
    import concurrent.futures

    result_holder: dict[str, Any] = {}

    def _thread_target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_holder["value"] = loop.run_until_complete(coro)
        except Exception as e:
            result_holder["error"] = e
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(_thread_target)
        future.result(timeout=700)
    except concurrent.futures.TimeoutError:
        pool.shutdown(wait=False)
        raise
    else:
        pool.shutdown(wait=True)

    if "error" in result_holder:
        raise result_holder["error"]
    return result_holder.get("value")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Record an asciinema demo of Claude Code with Transolver-3 skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/record_demo.py --scenario full
              python scripts/record_demo.py --scenario data --output data-demo.cast
              python scripts/record_demo.py --prompt "Show me how to check drift"
              python scripts/record_demo.py --list-scenarios
        """),
    )
    parser.add_argument(
        "--scenario", "-s",
        choices=list(SCENARIOS.keys()),
        default="full",
        help="Predefined demo scenario (default: full)",
    )
    parser.add_argument(
        "--prompt", "-p",
        help="Custom prompt (overrides --scenario)",
    )
    parser.add_argument(
        "--output", "-o",
        default="demo.cast",
        help="Output .cast file path (default: demo.cast)",
    )
    parser.add_argument(
        "--width", type=int, default=120,
        help="Terminal width (default: 120)",
    )
    parser.add_argument(
        "--height", type=int, default=40,
        help="Terminal height (default: 40)",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Agent timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--model", "-m",
        help="Override model (e.g., claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--list-scenarios", action="store_true",
        help="List available scenarios and exit",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory (default: repo root)",
    )

    args = parser.parse_args()

    if args.list_scenarios:
        print("Available scenarios:\n")
        for name, info in SCENARIOS.items():
            print(f"  {name:12s}  {info['title']}")
            prompt_preview = info["prompt"][:80] + "..." if len(info["prompt"]) > 80 else info["prompt"]
            print(f"  {' ':12s}  {prompt_preview}\n")
        return

    prompt = args.prompt or SCENARIOS[args.scenario]["prompt"]
    title = f"Transolver-3 — {SCENARIOS.get(args.scenario, {}).get('title', 'Custom Demo')}"

    # Resolve cwd to repo root
    cwd = args.cwd
    if cwd is None:
        cwd = str(Path(__file__).resolve().parent.parent)

    cast = CastWriter(
        width=args.width,
        height=args.height,
        title=title,
        output_path=args.output,
    )

    print(f"Recording demo: {title}")
    print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model or os.environ.get('ANTHROPIC_MODEL', 'default')}")
    print()

    _run_in_fresh_loop(
        run_and_record(
            prompt=prompt,
            cast=cast,
            cwd=cwd,
            timeout_seconds=args.timeout,
            model=args.model,
        )
    )


if __name__ == "__main__":
    main()
