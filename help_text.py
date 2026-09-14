"""
The list of chat commands, written ONCE. The help command, the /help slash
command, and the AI's system prompt are all built from it — add a command here
and every place that explains commands stays in sync.
"""

# (usage, what it does). Grouped into sections for the help message.
COMMANDS = {
    "Music": [
        ("syntia play <song / playlist / link>", "Replace the queue and play now"),
        ("syntia add <...>", "Append to the queue (alias: enqueue)"),
        ("syntia queue", "Show what's playing and what's next"),
        ("syntia clear", "Empty the upcoming queue"),
        ("syntia skip", "Skip to the next song"),
        ("syntia previous", "Replay the previous song (aliases: prev, back)"),
        ("syntia forward [N]", "Jump ahead N seconds, default 30 (aliases: fwd, ff)"),
        ("syntia rewind [N]", "Jump back N seconds, default 30 (alias: rw)"),
        ("syntia seek <time>", "Jump to a position, e.g. 1:02:00"),
        ("syntia volume [0-100]", "Show or set the volume, e.g. volume 10 (alias: vol)"),
        ("syntia shuffle [playlist]", "Shuffle the queue, or shuffle-play a playlist"),
        ("syntia stop", "Stop the music and clear the queue, but stay in voice"),
    ],
    "Voice": [
        ("syntia leave", "Leave the voice channel (aliases: bye, disconnect)"),
        ("syntia join", "Join your voice channel without playing (aliases: come, summon)"),
        ("syntia timeout [on | off | minutes]", "Show or change the idle auto-leave (admins)"),
    ],
    "Other": [
        ("syntia roll [N]", "Roll a die, 1 to N (default 6)"),
        ("syntia help", "Show this list (alias: commands)"),
        ("syntia <anything else>", "Chat with me - I can start music too"),
    ],
}


def help_message() -> str:
    # The full, human-readable list sent by `syntia help`, /help, and show_help.
    lines = ["**Syntia commands**"]
    for section, commands in COMMANDS.items():
        lines.append(f"\n**{section}**")
        lines += [f"`{usage}` - {what}" for usage, what in commands]
    return "\n".join(lines)[:2000]


def command_reference() -> str:
    # A compact version for the AI's system prompt, so it knows the exact syntax.
    return "\n".join(
        f"- {usage}: {what}"
        for commands in COMMANDS.values()
        for usage, what in commands
    )
