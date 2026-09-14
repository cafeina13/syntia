"""
Syntia — a learning Discord bot (entry point).

This file wires Discord events to the feature modules:
  config.py  — settings + AI/Spotify clients
  music.py   — voice playback, queue, source resolution
  ai.py      — AI brain: tools, Gemini/Ollama backends, dispatch

Run it with:  .venv\\Scripts\\python.exe bot.py
"""

import random

import discord
from discord import app_commands
from discord.ext import tasks

import ai
import music
from config import GUILD_ID, PREFIX, TOKEN
from help_text import help_message

# "Intents" tell Discord which events the bot receives. To read chat messages
# (needed for our "syntia ..." prefix), we turn on the PRIVILEGED message content
# intent here AND enable it in the Developer Portal (see README).
intents = discord.Intents.default()
intents.message_content = True


class SyntiaBot(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        # The "command tree" holds your slash commands (/ping, etc.).
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        # Runs once before the bot connects. Start the idle-timeout watcher, then
        # sync: syncing pushes your slash commands up to Discord for the / menu.
        idle_watcher.start()
        if GUILD_ID:
            guild = discord.Object(id=int(GUILD_ID))
            # Copy global commands onto this one guild, then sync just it.
            # Guild syncs are INSTANT — perfect while developing.
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            print(f"Synced {len(synced)} commands to guild {GUILD_ID} (instant).")
        else:
            synced = await self.tree.sync()
            print(
                f"Synced {len(synced)} GLOBAL commands "
                "(can take up to 1 hour to appear). "
                "Set GUILD_ID in .env for instant updates."
            )


@tasks.loop(seconds=30)
async def idle_watcher():
    # Leaves voice channels that have gone quiet (see music.check_idle).
    try:
        await music.check_idle()
    except Exception as error:
        # An exception would stop the loop for good — log it and keep watching.
        print(f"Idle check failed: {error}")


@idle_watcher.before_loop
async def before_idle_watcher():
    await client.wait_until_ready()


client = SyntiaBot()
# Give the music module the client, so its playback callbacks can reach the loop.
music.client = client


@client.event
async def on_ready():
    # Fires when the bot has finished logging in — including AGAIN after a
    # dropped session had to re-IDENTIFY, which is why we try to resume here.
    print(f"Logged in as {client.user} (id: {client.user.id})")
    print("Bot is ready! Try /ping, or type 'syntia roll 20' in chat.")
    await music.recover_all()


@client.event
async def on_resumed():
    # A briefer hiccup: the session survived, so Discord let us RESUME. Voice
    # may still have been dropped, so give the queue the same chance to recover.
    await music.recover_all()


@client.event
async def on_message(message: discord.Message):
    # Fires for EVERY message the bot can see. We decide what to do with it.

    # CRITICAL: ignore the bot's own messages, or it could reply to itself forever.
    if message.author == client.user:
        return

    # Only react to messages aimed at us (case-insensitive: "Syntia" works too).
    if not message.content.lower().startswith(PREFIX):
        return

    # Remove the prefix, then split the rest into a command word + its arguments.
    # "syntia roll 20"  ->  command = "roll", args = ["20"]
    body = message.content[len(PREFIX) :].strip()
    parts = body.split()
    if not parts:
        return
    command = parts[0].lower()
    args = parts[1:]

    match command:
        case "roll":
            sides = 6  # a normal die if they don't specify
            if args:
                # args[0] is the requested number of sides. Validate it's a number.
                if not args[0].isdigit() or int(args[0]) < 1:
                    await message.channel.send(
                        "Give me a positive number of sides, e.g. `syntia roll 20`."
                    )
                    return
                sides = int(args[0])
            result = random.randint(1, sides)  # both ends included: 1..sides
            await message.channel.send(f"🎲 You rolled a **{result}** (1–{sides}).")

        case "play":
            # "play" REPLACES the queue and starts now (song or playlist).
            query = " ".join(args)
            if not query:
                await message.channel.send(
                    "Give me a song or link, e.g. `syntia play lofi hip hop`."
                )
            else:
                await music.play_music(message, query)

        case "add" | "enqueue":
            # "add" APPENDS to the queue without interrupting the current song.
            query = " ".join(args)
            if not query:
                await message.channel.send(
                    "Give me a song or link to add, e.g. `syntia add some jazz`."
                )
            else:
                await music.add_music(message, query)

        case "clear":
            await music.clear_queue(message)

        case "stop":
            # End the music but stay in the channel.
            await music.stop_music(message)

        case "leave" | "bye" | "disconnect":
            # One case can match several words with the | (or) pattern.
            await music.leave_voice(message)

        case "pause":
            await music.pause_music(message)

        case "resume" | "unpause":
            await music.resume_music(message)

        case "skip":
            await music.skip_song(message)

        case "forward" | "fwd" | "ff":
            # Jump ahead in the current track. Optional number of seconds (default 30).
            secs = int(args[0]) if args and args[0].isdigit() else 30
            await music.seek(message, secs)

        case "rewind" | "rw":
            secs = int(args[0]) if args and args[0].isdigit() else 30
            await music.seek(message, -secs)

        case "seek":
            # Jump TO an absolute time: syntia seek 1:02:00 (or 5:25, or 325)
            secs = music.parse_timestamp(args[0]) if args else None
            if secs is None:
                await message.channel.send(
                    "Give me a time, e.g. `syntia seek 1:02:00`."
                )
            else:
                await music.seek(message, to=secs)

        case "shuffle":
            await music.shuffle_queue(message, " ".join(args))

        case "np" | "nowplaying" | "now" | "current":
            # What's playing, where we are in it, and a link to resume later.
            await music.now_playing(message)

        case "queue":
            await music.show_queue(message)

        case "previous" | "prev" | "back":
            await music.play_previous(message)

        case "join" | "come" | "summon":
            # Join voice without playing anything.
            await music.join_voice(message)

        case "timeout":
            await music.set_timeout(message, args[0] if args else "")

        case "volume" | "vol":
            await music.set_volume(message, args[0] if args else "")

        case "help" | "commands":
            await message.channel.send(help_message())

        case _:
            # Nothing matched — maybe a typo, maybe they just want to chat.
            # Hand the FULL text (command word included) to the AI assistant.
            await ai.ask_ai(message, body)


# --- Slash commands -------------------------------------------------------
# Each function below is a slash command. The @decorator registers it.


@client.tree.command(name="ping", description="Check that the bot is alive.")
async def ping(interaction: discord.Interaction):
    # interaction.response.send_message replies to the person who ran it.
    latency_ms = round(client.latency * 1000)
    await interaction.response.send_message(f"Pong! ({latency_ms}ms)")


@client.tree.command(name="help", description="List Syntia's commands.")
async def help_command(interaction: discord.Interaction):
    # ephemeral=True -> only the person who asked sees it; no channel clutter.
    await interaction.response.send_message(help_message(), ephemeral=True)


@client.tree.command(name="hello", description="Say hello to the bot.")
async def hello(interaction: discord.Interaction):
    name = interaction.user.display_name
    await interaction.response.send_message(f"Hello, {name}! 👋")


@client.tree.command(name="echo", description="Repeat back what you say.")
@app_commands.describe(text="The text you want echoed back")
async def echo(interaction: discord.Interaction, text: str):
    # `text: str` becomes a required argument in the slash command UI.
    await interaction.response.send_message(text)


# --- Start the bot --------------------------------------------------------


def main():
    if not TOKEN:
        raise SystemExit(
            "No DISCORD_TOKEN found. Copy .env.example to .env and paste "
            "your bot token into it (see README.md)."
        )
    client.run(TOKEN)


if __name__ == "__main__":
    main()
