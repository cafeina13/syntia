r"""
Syntia — a learning Discord bot.

Read this top-to-bottom; the comments explain each piece.
Run it with:  .venv\Scripts\python.exe bot.py
"""

import asyncio
import os
import random
from pathlib import Path

import discord
import ollama
import yt_dlp
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

# Which AI backend to use: "gemini" (cloud, free tier) or "ollama" (local).
# Set AI_BACKEND in .env. Ollama is great for offline testing — no rate limits.
AI_BACKEND = os.getenv("AI_BACKEND", "gemini").lower()

# Gemini (cloud). Free key at https://aistudio.google.com/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
GEMINI_MODEL = "gemini-2.5-flash"  # fast and free-tier friendly

# Ollama (local, http://localhost:11434). Use a tool-capable model (qwen2.5 is).
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
ollama_client = ollama.AsyncClient()

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
        f"- Server: {server_name}\n\n"
        f"You can play or stop music in the user's voice channel by calling your "
        f"available tools whenever they want to listen to or stop something. "
        f"If a message is clearly a song, artist, or playlist, play it instead of "
        f"replying with text."
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


# The tools the AI may call, described ONCE in a neutral form. Gemini and Ollama
# want different shapes, so we build each provider's version from this list —
# edit a tool here and both backends stay in sync.
# Each property is (type, description).
TOOL_SPECS = [
    {
        "name": "play_music",
        "description": (
            "Play a song, artist, or playlist in the user's voice channel. "
            "Use whenever the user wants to listen to, put on, or play music."
        ),
        "properties": {
            "query": ("string", "What to search for: song, artist, playlist, or link."),
            "start_seconds": (
                "integer",
                "Where to start playback, in seconds. Convert 'from 10 minutes "
                "in' to 600. Use 0 to start at the beginning.",
            ),
        },
        "required": ["query"],
    },
    {
        "name": "stop_music",
        "description": "Stop playback and leave the voice channel.",
        "properties": {},
        "required": [],
    },
    {
        "name": "skip_song",
        "description": "Skip the current song and play the next one in the queue.",
        "properties": {},
        "required": [],
    },
    {
        "name": "shuffle_queue",
        "description": "Randomly shuffle the order of the upcoming songs in the queue.",
        "properties": {},
        "required": [],
    },
]

_GEMINI_TYPES = {"string": types.Type.STRING, "integer": types.Type.INTEGER}


def _build_gemini_tools():
    declarations = []
    for spec in TOOL_SPECS:
        props = {
            name: types.Schema(type=_GEMINI_TYPES[kind], description=desc)
            for name, (kind, desc) in spec["properties"].items()
        }
        declarations.append(
            types.FunctionDeclaration(
                name=spec["name"],
                description=spec["description"],
                parameters=types.Schema(
                    type=types.Type.OBJECT, properties=props, required=spec["required"]
                ),
            )
        )
    return [types.Tool(function_declarations=declarations)]


def _build_ollama_tools():
    tools = []
    for spec in TOOL_SPECS:
        props = {
            name: {"type": kind, "description": desc}
            for name, (kind, desc) in spec["properties"].items()
        }
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": spec["required"],
                    },
                },
            }
        )
    return tools


GEMINI_TOOLS = _build_gemini_tools()
OLLAMA_TOOLS = _build_ollama_tools()


async def generate_gemini(system: str, prompt: str) -> dict:
    # Returns a normalized result so ask_ai doesn't care which backend ran:
    #   {"type": "text", "text": ...}  or  {"type": "tool", "name": ..., "args": {...}}
    response = await gemini_client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system, tools=GEMINI_TOOLS),
    )
    candidates = response.candidates or []
    parts = candidates[0].content.parts if candidates and candidates[0].content else []
    for part in parts:
        if part.function_call:
            fc = part.function_call
            return {"type": "tool", "name": fc.name, "args": dict(fc.args)}
    return {"type": "text", "text": response.text or ""}


async def generate_ollama(system: str, prompt: str) -> dict:
    # Same normalized result, but talking to the LOCAL Ollama server.
    response = await ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        tools=OLLAMA_TOOLS,
    )
    msg = response.message
    if msg.tool_calls:
        call = msg.tool_calls[0]  # honor the first tool the model asked for
        return {"type": "tool", "name": call.function.name, "args": dict(call.function.arguments)}
    return {"type": "text", "text": msg.content or ""}


async def run_tool(message: discord.Message, name: str, args: dict):
    # Map the tool name the AI chose to the real bot function. One place for
    # both backends, so this dispatch isn't duplicated.
    if name == "play_music":
        start = int(args.get("start_seconds") or 0)
        await play_music(message, args.get("query", ""), start)
    elif name == "stop_music":
        await leave_voice(message)
    elif name == "skip_song":
        await skip_song(message)
    elif name == "shuffle_queue":
        await shuffle_queue(message)


async def ask_ai(message: discord.Message, prompt: str):
    # The "default" behaviour: hand the message to the AI. With tools attached it
    # can either reply with text (chatting) OR ask us to run a command (tool use).
    # AI_BACKEND in .env decides whether that AI is local (ollama) or cloud (gemini).
    if AI_BACKEND == "ollama":
        backend = generate_ollama
    elif gemini_client is not None:
        backend = generate_gemini
    else:
        await message.channel.send(
            "AI isn't set up — add GEMINI_API_KEY to .env, or set AI_BACKEND=ollama."
        )
        return

    try:
        async with message.channel.typing():
            result = await backend(build_system_instruction(message), prompt)
    except Exception as error:
        # Never let one bad AI call crash the whole bot — report and move on.
        await message.channel.send(f"AI error: {error}")
        return

    if result["type"] == "tool":
        await run_tool(message, result["name"], result["args"])
    else:
        reply = (result["text"] or "").strip() or "(the AI returned nothing)"
        await message.channel.send(reply[:2000])


# --- Music (Step 2: a per-server queue + skip + shuffle) ------------------

# How yt-dlp finds audio. "ytsearch" means plain text like "lofi hip hop" gets
# searched on YouTube; a full YouTube URL also works. noplaylist=True keeps each
# request to a single track (Spotify-playlist expansion comes in a later step).
YTDL_OPTS = {
    "format": "bestaudio/best",
    "noplaylist": True,
    "default_search": "ytsearch",
    "quiet": True,
    "no_warnings": True,
}
ytdl = yt_dlp.YoutubeDL(YTDL_OPTS)


# FFmpeg flags: -vn drops video; reconnect flags help if the stream hiccups.
# -ss <seconds> (input seek) is how we START PART-WAY into a track. A YouTube
# "&t=600" only moves the web player; to actually skip ahead we must tell FFmpeg.
def ffmpeg_options(start_seconds: int = 0) -> dict:
    before = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"
    if start_seconds > 0:
        before = f"-ss {start_seconds} " + before
    return {"before_options": before, "options": "-vn"}


# Each server gets its own list of upcoming tracks. Key = guild id, value = list.
song_queues: dict[int, list[dict]] = {}


def get_queue(guild_id: int) -> list:
    # setdefault: return the existing queue, or create an empty one on first use.
    return song_queues.setdefault(guild_id, [])


async def resolve_track(query: str, start_seconds: int, text_channel) -> dict:
    # Ask yt-dlp for the audio. Blocking work, so run it off the event loop.
    data = await asyncio.to_thread(ytdl.extract_info, query, download=False)
    if "entries" in data:  # a search returns a list of hits; take the first
        data = data["entries"][0]
    return {
        "title": data.get("title", "Unknown"),
        "stream_url": data["url"],
        "start_seconds": start_seconds,
        "channel": text_channel,  # where to announce "Now playing"
    }


def schedule_next(guild: discord.Guild):
    # Called in FFmpeg's OWN thread when a song ends — we can't await here, so we
    # hand the coroutine back to the bot's event loop to play the next track.
    asyncio.run_coroutine_threadsafe(play_next(guild), client.loop)


async def play_next(guild: discord.Guild):
    queue = get_queue(guild.id)
    voice = guild.voice_client
    if voice is None or not queue:
        return  # not connected, or nothing left to play
    track = queue.pop(0)  # take the song at the front of the line
    source = await discord.FFmpegOpusAudio.from_probe(
        track["stream_url"], **ffmpeg_options(track["start_seconds"])
    )
    # after=... runs when THIS song finishes -> kick off the next one.
    voice.play(source, after=lambda error: schedule_next(guild))
    start = track["start_seconds"]
    note = f" (from {start // 60}:{start % 60:02d})" if start else ""
    await track["channel"].send(f"▶️ Now playing: **{track['title']}**{note}")


async def play_music(message: discord.Message, query: str, start_seconds: int = 0):
    # You must be in a voice channel so the bot knows where to join.
    if not message.author.voice or not message.author.voice.channel:
        await message.channel.send("Join a voice channel first, then try again.")
        return
    channel = message.author.voice.channel

    # Connect to your channel (or move there if already connected elsewhere).
    voice = message.guild.voice_client
    if voice is None:
        voice = await channel.connect()
    elif voice.channel != channel:
        await voice.move_to(channel)

    async with message.channel.typing():
        try:
            track = await resolve_track(query, start_seconds, message.channel)
        except Exception as error:
            await message.channel.send(f"Couldn't find that: {error}")
            return

    queue = get_queue(message.guild.id)
    queue.append(track)

    # If nothing is playing, start now; otherwise the track waits its turn.
    if not voice.is_playing() and not voice.is_paused():
        await play_next(message.guild)
    else:
        await message.channel.send(
            f"➕ Added to queue (#{len(queue)}): **{track['title']}**"
        )


async def skip_song(message: discord.Message):
    voice = message.guild.voice_client
    if voice and (voice.is_playing() or voice.is_paused()):
        voice.stop()  # stopping fires the after= callback, which plays the next
        await message.channel.send("⏭️ Skipped.")
    else:
        await message.channel.send("Nothing is playing.")


async def shuffle_queue(message: discord.Message):
    queue = get_queue(message.guild.id)
    if len(queue) < 2:
        await message.channel.send("Not enough songs in the queue to shuffle.")
        return
    random.shuffle(queue)  # shuffles the list in place
    await message.channel.send("🔀 Shuffled the queue.")


async def show_queue(message: discord.Message):
    queue = get_queue(message.guild.id)
    if not queue:
        await message.channel.send("The queue is empty.")
        return
    lines = [f"{i}. {track['title']}" for i, track in enumerate(queue, 1)]
    await message.channel.send(("**Up next:**\n" + "\n".join(lines))[:2000])


async def leave_voice(message: discord.Message):
    voice = message.guild.voice_client
    if voice is None:
        await message.channel.send("I'm not in a voice channel.")
        return
    get_queue(message.guild.id).clear()  # forget the queue when we leave
    await voice.disconnect()
    await message.channel.send("Left the voice channel. 👋")


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

        case "play":
            # Everything after "play" is the song name or YouTube link.
            query = " ".join(args)
            if not query:
                await message.channel.send(
                    "Give me a song or link, e.g. `syntia play lofi hip hop`."
                )
            else:
                await play_music(message, query)

        case "stop" | "leave":
            # One case can match several words with the | (or) pattern.
            await leave_voice(message)

        case "skip":
            await skip_song(message)

        case "shuffle":
            await shuffle_queue(message)

        case "queue":
            await show_queue(message)

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
