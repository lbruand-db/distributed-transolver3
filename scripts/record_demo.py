#!/usr/bin/env python3
"""Record an asciinema demo of a real Claude Code CLI session.

Spawns the actual `claude` CLI with --output-format stream-json to get
real-time streaming events, then renders them into an asciinema v2 .cast
file that looks like an actual interactive Claude Code session — with
typing effects, tool blocks, spinners, and Claude's real output.

Usage:
    python scripts/record_demo.py                     # full scenario
    python scripts/record_demo.py --scenario data      # specific scenario
    python scripts/record_demo.py --prompt "..."       # custom prompt
    python scripts/record_demo.py --list-scenarios     # list options

The generated .cast file can be played with:
    asciinema play demo.cast
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path


# ── ANSI styling (matches Claude Code's visual style) ──────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
WHITE = "\033[37m"
GRAY = "\033[90m"
B_CYAN = "\033[38;5;81m"
B_GREEN = "\033[38;5;78m"
B_ORANGE = "\033[38;5;208m"
B_MAGENTA = "\033[38;5;176m"


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
            "automotive CFD (pressure + velocity). What config preset should "
            "I use? Set up the training notebook cells with MLflow tracking "
            "and mixed precision."
        ),
    },
    "analyze": {
        "title": "Analyze results and check for drift",
        "prompt": (
            "I just ran inference with my trained Transolver-3 model. The "
            "predictions are in a tensor called `predictions` with shape "
            "(1, 500000, 4) — channels are pressure, vx, vy, vz. Check if "
            "the predictions are physically plausible, compute per-channel "
            "error stats against `targets`, and check for drift."
        ),
    },
    "deploy": {
        "title": "Deploy model to Databricks serving",
        "prompt": (
            "My Transolver-3 model is trained and validated. Walk me through "
            "registering it in Unity Catalog, deploying a serving endpoint, "
            "and setting up monitoring with inference tables and drift "
            "detection."
        ),
    },
    "full": {
        "title": "End-to-end workflow demo",
        "prompt": (
            "I'm new to Transolver-3 on Databricks. Walk me through the "
            "complete workflow: "
            "1) inspect my mesh data at "
            "/Volumes/ml/transolver3/data/drivaer_001.npz, "
            "2) pick the right model config for my mesh size, "
            "3) show me how to train with MLflow tracking, "
            "4) explain how to analyze results and check for drift. "
            "Keep it practical with notebook-ready code."
        ),
    },
}


# ── Cast file writer ────────────────────────────────────────────────────────


class CastRecorder:
    """Accumulates terminal output events and writes an asciinema .cast."""

    def __init__(self, width: int, height: int, title: str):
        self.width = width
        self.height = height
        self.title = title
        self.events: list[tuple[float, str]] = []
        self.t = 0.0  # virtual clock

    def emit(self, text: str, advance: float = 0.0):
        """Write output text at the current virtual timestamp."""
        if text:
            self.events.append((self.t, text))
        self.t += advance

    def type_text(self, text: str, cps: float = 60.0):
        """Emit text character-by-character at `cps` chars per second."""
        delay = 1.0 / cps
        for ch in text:
            self.events.append((self.t, ch))
            self.t += delay

    def newline(self):
        self.emit("\r\n")

    def pause(self, seconds: float):
        self.t += seconds

    def save(self, path: str):
        header = {
            "version": 2,
            "width": self.width,
            "height": self.height,
            "timestamp": int(time.time()),
            "duration": round(self.t, 3),
            "title": self.title,
            "idle_time_limit": 3.0,
            "env": {"SHELL": "/bin/zsh", "TERM": "xterm-256color"},
            "theme": {
                "fg": "#d4d4d4",
                "bg": "#1e1e1e",
                "palette": (
                    "#1e1e1e:#f44747:#608b4e:#dcdcaa"
                    ":#569cd6:#c586c0:#4ec9b0:#d4d4d4"
                    ":#808080:#f44747:#608b4e:#dcdcaa"
                    ":#569cd6:#c586c0:#4ec9b0:#ffffff"
                ),
            },
        }
        with open(path, "w") as f:
            f.write(json.dumps(header) + "\n")
            for ts, text in self.events:
                f.write(json.dumps([round(ts, 6), "o", text]) + "\n")

        size_kb = os.path.getsize(path) / 1024
        print(f"\n{'=' * 50}")
        print(f"Recording saved to {path}")
        print(f"  Events:   {len(self.events)}")
        print(f"  Duration: {self.t:.1f}s")
        print(f"  Size:     {size_kb:.1f} KB")
        print("\nPlay with:")
        print(f"  asciinema play {path}")


# ── Rendering helpers (Claude Code style) ───────────────────────────────────


def render_prompt_line(rec: CastRecorder, cwd: str):
    """Render a Claude Code-style prompt."""
    short_cwd = cwd.replace(os.path.expanduser("~"), "~")
    rec.emit(f"{B_CYAN}{BOLD}{short_cwd} >{RESET} ", advance=0.05)


def render_user_input(rec: CastRecorder, prompt: str):
    """Simulate the user typing their prompt."""
    rec.type_text(prompt, cps=40)
    rec.newline()
    rec.pause(0.3)


def render_thinking_spinner(rec: CastRecorder, duration: float = 2.0):
    """Render a Claude-style thinking spinner."""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    steps = int(duration / 0.12)
    for i in range(steps):
        frame = frames[i % len(frames)]
        rec.emit(f"\r{YELLOW}{frame} Thinking…{RESET}  ")
        rec.pause(0.12)
    rec.emit(f"\r{' ' * 30}\r")  # clear spinner line
    rec.pause(0.1)


def render_tool_block(rec: CastRecorder, tool_name: str, tool_input: dict):
    """Render a Claude Code tool use block."""
    rec.newline()
    rec.emit(f"  {B_MAGENTA}{BOLD}⏵ {tool_name}{RESET}")

    # Show the most relevant parameter
    detail = ""
    if tool_name == "Read" and "file_path" in tool_input:
        detail = tool_input["file_path"]
    elif tool_name == "Bash" and "command" in tool_input:
        cmd = tool_input["command"]
        detail = cmd[:70] + "…" if len(cmd) > 70 else cmd
    elif tool_name == "Write" and "file_path" in tool_input:
        detail = tool_input["file_path"]
    elif tool_name == "Edit" and "file_path" in tool_input:
        detail = tool_input["file_path"]
    elif tool_name == "Grep" and "pattern" in tool_input:
        detail = f"/{tool_input['pattern']}/"
    elif tool_name == "Glob" and "pattern" in tool_input:
        detail = tool_input["pattern"]

    if detail:
        rec.emit(f" {DIM}{detail}{RESET}")
    rec.newline()
    rec.pause(0.2)


def render_tool_result_block(rec: CastRecorder, content: str, is_error: bool = False, max_lines: int = 8):
    """Render a truncated tool result."""
    if not content:
        return
    color = RED if is_error else GRAY
    lines = content.split("\n")
    shown = lines[:max_lines]
    for line in shown:
        if len(line) > 100:
            line = line[:97] + "…"
        rec.emit(f"  {color}│ {line}{RESET}")
        rec.newline()
        rec.pause(0.01)
    if len(lines) > max_lines:
        rec.emit(f"  {color}│ … ({len(lines) - max_lines} more lines){RESET}")
        rec.newline()
    rec.pause(0.1)


def render_assistant_text(rec: CastRecorder, text: str, cps: float = 120.0):
    """Render assistant markdown text with streaming typing effect."""
    delay = 1.0 / cps
    for ch in text:
        rec.events.append((rec.t, ch))
        # Slow down at newlines for readability
        if ch == "\n":
            rec.events[-1] = (rec.t, "\r\n")
            rec.t += delay * 3
        else:
            rec.t += delay
    rec.pause(0.15)


def render_result_footer(
    rec: CastRecorder,
    duration_ms: int | None = None,
    cost: float | None = None,
    session_id: str | None = None,
):
    """Render the session summary footer."""
    rec.newline()
    rec.emit(f"  {DIM}{'─' * 60}{RESET}")
    rec.newline()
    parts = []
    if duration_ms:
        parts.append(f"Duration: {duration_ms / 1000:.1f}s")
    if cost:
        parts.append(f"Cost: ${cost:.4f}")
    if session_id:
        parts.append(f"Session: {session_id[:8]}…")
    if parts:
        rec.emit(f"  {DIM}{' · '.join(parts)}{RESET}")
        rec.newline()
    rec.pause(1.0)


# ── Main: run claude and record ─────────────────────────────────────────────


def record_session(
    prompt: str,
    output_path: str,
    cwd: str,
    width: int = 120,
    height: int = 40,
    timeout: int = 600,
    model: str | None = None,
    title: str = "Transolver-3 — Claude Code Demo",
):
    """Run `claude -p --output-format stream-json` and record to .cast."""
    claude_bin = shutil.which("claude")
    if not claude_bin:
        print("ERROR: `claude` CLI not found in PATH.")
        sys.exit(1)

    cmd = [
        claude_bin,
        "-p",
        "--output-format",
        "stream-json",
        "--dangerously-skip-permissions",
        "--verbose",
    ]
    if model:
        cmd.extend(["--model", model])

    env = os.environ.copy()
    env["TERM"] = "xterm-256color"
    env["COLUMNS"] = str(width)
    env["LINES"] = str(height)
    env["NO_COLOR"] = ""  # don't suppress colors in json content

    rec = CastRecorder(width=width, height=height, title=title)

    # ── Intro banner ──
    rec.emit(f"\r\n  {B_ORANGE}{BOLD}Transolver-3 on Databricks{RESET}")
    rec.emit(f" {DIM}— Claude Code Demo{RESET}\r\n")
    rec.emit(f"  {DIM}{'─' * 50}{RESET}\r\n\r\n")
    rec.pause(0.8)

    # ── Show the user "typing" the prompt ──
    render_prompt_line(rec, cwd)
    render_user_input(rec, prompt)
    render_thinking_spinner(rec, duration=2.0)

    # ── Run claude ──
    print("Running: claude -p --output-format stream-json ...")
    print(f"CWD: {cwd}")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
    )

    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(prompt.encode("utf-8"))
    proc.stdin.close()

    # Read streaming JSON events line by line
    accumulated_text = []

    try:
        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            if etype == "assistant":
                # An assistant turn with content blocks
                for block in event.get("message", {}).get("content", []):
                    btype = block.get("type", "")
                    if btype == "text":
                        text = block.get("text", "")
                        render_assistant_text(rec, text)
                        accumulated_text.append(text)
                        # Live preview
                        sys.stdout.write(text)
                        sys.stdout.flush()
                    elif btype == "tool_use":
                        tool_name = block.get("name", "?")
                        tool_input = block.get("input", {})
                        render_tool_block(rec, tool_name, tool_input)
                        print(f"  [tool] {tool_name}", flush=True)

            elif etype == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    render_assistant_text(rec, text)
                    accumulated_text.append(text)
                    sys.stdout.write(text)
                    sys.stdout.flush()

            elif etype == "tool_use":
                tool_name = event.get("name", event.get("tool", "?"))
                tool_input = event.get("input", {})
                render_tool_block(rec, tool_name, tool_input)
                print(f"  [tool] {tool_name}", flush=True)

            elif etype == "tool_result":
                content = event.get("content", "")
                is_error = event.get("is_error", False)
                if isinstance(content, list):
                    # Content can be a list of text blocks
                    content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
                render_tool_result_block(rec, str(content), is_error=is_error)

            elif etype == "result":
                result_data = event.get("result", event)
                if isinstance(result_data, str):
                    # result is the final text, not a dict
                    result_data = event
                duration_ms = result_data.get("duration_ms") if isinstance(result_data, dict) else None
                cost_usd = None
                session_id = None
                if isinstance(result_data, dict):
                    cost_usd = result_data.get("cost_usd", result_data.get("cost"))
                    session_id = result_data.get("session_id")
                render_result_footer(
                    rec,
                    duration_ms=duration_ms,
                    cost=cost_usd,
                    session_id=session_id,
                )

            elif etype == "error":
                err_msg = event.get("error", {})
                if isinstance(err_msg, dict):
                    err_msg = err_msg.get("message", str(err_msg))
                rec.emit(f"\r\n  {RED}{BOLD}Error: {err_msg}{RESET}\r\n")
                rec.pause(0.5)

    except KeyboardInterrupt:
        print("\n\nInterrupted — saving partial recording.")
        proc.send_signal(signal.SIGTERM)
    finally:
        proc.stdout.close()
        proc.stderr.close()
        proc.wait(timeout=10)

    # ── Outro ──
    rec.pause(0.5)
    rec.newline()
    rec.emit(f"  {B_GREEN}{BOLD}Demo complete.{RESET} {DIM}See skills/ for full documentation.{RESET}")
    rec.newline()
    rec.pause(2.0)

    rec.save(output_path)


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=("Record an asciinema demo of a real Claude Code session with Transolver-3 skills"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/record_demo.py --scenario full
              python scripts/record_demo.py --scenario data -o data-demo.cast
              python scripts/record_demo.py --prompt "How to check drift?"
              python scripts/record_demo.py --list-scenarios
        """),
    )
    parser.add_argument(
        "--scenario",
        "-s",
        choices=list(SCENARIOS.keys()),
        default="full",
        help="Predefined demo scenario (default: full)",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        help="Custom prompt (overrides --scenario)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="demo.cast",
        help="Output .cast file path (default: demo.cast)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=120,
        help="Terminal width (default: 120)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=40,
        help="Terminal height (default: 40)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Override model (e.g., claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
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
            preview = info["prompt"][:70]
            if len(info["prompt"]) > 70:
                preview += "..."
            print(f"  {' ':12s}  {preview}\n")
        return

    prompt = args.prompt or SCENARIOS[args.scenario]["prompt"]
    scenario_info = SCENARIOS.get(args.scenario, {})
    title_suffix = scenario_info.get("title", "Custom Demo")
    title = f"Transolver-3 — {title_suffix}"

    cwd = args.cwd
    if cwd is None:
        cwd = str(Path(__file__).resolve().parent.parent)

    print(f"Recording: {title}")
    preview = prompt[:90] + "..." if len(prompt) > 90 else prompt
    print(f"Prompt: {preview}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model or 'default'}")
    print(f"{'=' * 50}\n")

    record_session(
        prompt=prompt,
        output_path=args.output,
        cwd=cwd,
        width=args.width,
        height=args.height,
        timeout=args.timeout,
        model=args.model,
        title=title,
    )


if __name__ == "__main__":
    main()
