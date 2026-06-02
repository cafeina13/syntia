r"""
Syntia — a learning Discord bot.

Read this top-to-bottom; the comments explain each piece.
Run it with:  .venv\Scripts\python.exe bot.py
"""

import os
import random
from pathlib import Path

import discord
from discord import app_commands
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load the secret token from the .env file into the environment.
# We keep the token OUT of the code so it never gets committed to git.
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

# Optional: the ID of YOUR server, for instant command updates during development.
# Global commands take up to an hour to appear; guild commands appear instantly.
# Leave it unset to publish commands globally (slow, but visible in every server).
GUILD_ID = os.getenv("GUILD_ID")

# Google AI Studio (Gemini) — free tier. Get a key at https://aistudio.google.com/apikey
# If the key is missing we just skip AI; the rest of the bot still works.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
AI_MODEL = "gemini-2.5-flash"  # fast and free-tier friendly

# The bot's "system prompt" (personality + rules) lives in System_Prompt.md so
# you can edit the personality in plain Markdown without touching code.
# We read it once at startup. NOTE: encoding="utf-8" is required — without it,
# Windows may fail on characters like the em dash (we hit that exact error before).
PROMPT_FILE = Path(__file__).parent / "System_Prompt.md"
try:
    AI_PRE_PROMPT = PROMPT_FILE.read_text(encoding="utf-8").strip()
except FileNotFoundError:
    AI_PRE_PROMPT = "You are Syntia, a helpful Discord bot."


def build_system_instruction(message: discord.Message) -> str:
    # Personalise the system prompt per message: tell the AI who it's talking to
    # and where. This is what makes replies feel made-for-you.
    user_name = message.author.display_name
    server_name = message.guild.name if message.guild else "a direct message"
    return (
        f"{AI_PRE_PROMPT}\n\n"
        f"### Current Context\n"
        f"- The user you are talking to is named: {user_name}\n"
        f"- Server: {server_name}"
    )


# "Intents" tell Discord which kinds of events your bot wants to receive.
# The defaults are enough for slash commands. To read chat messages (needed for
# our "syntia ..." prefix commands), we must turn on the PRIVILEGED message
# content intent here AND enable it in the Developer Portal (see README).
intents = discord.Intents.default()
intents.message_content = True

# Our custom "caller" — type this (then a command) in chat to talk to the bot.
PREFIX = "syntia "


class SyntiaBot(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        # The "command tree" holds your slash commands (/ping, etc.).
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        # Runs once before the bot connects. Syncing pushes your slash
        # commands up to Discord so they appear in the / menu.
        if GUILD_ID:
            guild = discord.Object(id=int(GUILD_ID))
            # Copy the global commands onto this one guild, then sync just it.
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


@client.event
async def on_ready():
    # Fires when the bot has finished logging in.
    print(f"Logged in as {client.user} (id: {client.user.id})")
    print("Bot is ready! Try /ping, or type 'syntia roll 20' in chat.")


async def ask_ai(message: discord.Message, prompt: str):
    # The "default" behaviour: send whatever the user typed to Gemini and relay
    # the reply. Handles typos AND people just wanting to chat.
    if ai_client is None:
        await message.channel.send(
            "AI isn't set up yet — add GEMINI_API_KEY to your .env (see README)."
        )
        return

    try:
        # `typing()` shows the "Syntia is typing…" indicator while we wait.
        async with message.channel.typing():
            # .aio = the async version, so the bot stays responsive during the call.
            # system_instruction = the personality/rules; contents = the user's text.
            response = await ai_client.aio.models.generate_content(
                model=AI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=build_system_instruction(message),
                ),
            )
        reply = (response.text or "").strip() or "(the AI returned nothing)"
    except Exception as error:
        # Never let one bad API call crash the whole bot — report and move on.
        reply = f"AI error: {error}"

    # Discord rejects messages longer than 2000 characters, so trim if needed.
    await message.channel.send(reply[:2000])


@client.event
async def on_message(message: discord.Message):
    # Fires for EVERY message the bot can see. We decide what to do with it.

    # CRITICAL: ignore the bot's own messages. Without this, if the bot ever
    # said something starting with the prefix, it would reply to itself forever.
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

        case _:
            # Nothing matched — maybe a typo, maybe they just want to chat.
            # Hand the FULL text (command word included) to the AI assistant.
            await ask_ai(message, body)


# --- Commands -------------------------------------------------------------
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
