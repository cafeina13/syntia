"""The help list must stay complete, readable, and under Discord's limit."""

import ast
import re
from pathlib import Path

import help_text

BOT_FILE = Path(__file__).parent.parent / "bot.py"


def _command_words_in_bot():
    # Read bot.py's `match command:` block and collect every word it handles,
    # e.g. "play", "add", "enqueue", ... (the catch-all `case _` has no word).
    tree = ast.parse(BOT_FILE.read_text(encoding="utf-8"))
    words = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.MatchValue) and isinstance(node.value, ast.Constant):
            words.add(node.value.value)
    return words


def test_help_fits_in_one_discord_message():
    assert len(help_text.help_message()) <= 2000


def test_every_bot_command_is_in_the_help():
    # Catches the classic mistake: adding a command to bot.py but not the help.
    words = _command_words_in_bot()
    assert words, "found no commands in bot.py - did the match block move?"
    text = help_text.help_message()
    missing = sorted(w for w in words if not re.search(rf"\b{re.escape(w)}\b", text))
    assert not missing, f"commands missing from help_text.py: {missing}"


def test_help_lists_every_usage():
    text = help_text.help_message()
    for commands in help_text.COMMANDS.values():
        for usage, _ in commands:
            assert usage in text


def test_command_reference_has_one_line_per_command():
    lines = help_text.command_reference().splitlines()
    total = sum(len(commands) for commands in help_text.COMMANDS.values())
    assert len(lines) == total
    assert all(line.startswith("- syntia ") for line in lines)
