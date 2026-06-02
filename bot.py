r"""
Syntia — a learning Discord bot (entry point).

This file wires Discord events to the feature modules:
  config.py  — settings + AI/Spotify clients
  music.py   — voice playback, queue, source resolution
  ai.py      — AI brain: tools, Gemini/Ollama backends, dispatch

Run it with:  .venv\Scripts\python.exe bot.py
"""

import random

import discord
from discord import app_commands

import ai
import music
from config import GUILD_ID, PREFIX, TOKEN

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
        # Runs once before the bot connects. Syncing pushes your slash commands
        # up to Discord so they appear in the / menu.
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


client = SyntiaBot()
# Give the music module the client, so its playback callbacks can reach the loop.
music.client = client


@client.event
async def on_ready():
    # Fires when the bot has finished logging in.
    print(f"Logged in as {client.user} (id: {client.user.id})")
    print("Bot is ready! Try /ping, or type 'syntia roll 20' in chat.")


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
    body = message.content[len(PREFIX):].strip()
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

        case "stop" | "leave" | "bye":
            # One case can match several words with the | (or) pattern.
            await music.leave_voice(message)

        case "skip":
            await music.skip_song(message)

        case "shuffle":
            await music.shuffle_queue(message, " ".join(args))

        case "queue":
            await music.show_queue(message)

        case "previous" | "prev" | "back":
            await music.play_previous(message)

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
