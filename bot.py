r"""
Syntia — a learning Discord bot.

Read this top-to-bottom; the comments explain each piece.
Run it with:  .venv\Scripts\python.exe bot.py
"""

import asyncio
import os
import random
import re
from pathlib import Path

import discord
import ollama
import spotipy
import yt_dlp
from discord import app_commands
from dotenv import load_dotenv
from google import genai
from google.genai import types
from spotipy.oauth2 import SpotifyOAuth

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

# Spotify: we read a playlist/album/track's song list, then find the audio on
# YouTube (Spotify itself can't be streamed). Reading PLAYLISTS now requires a
# user login, so you log in ONCE with `python spotify_login.py` — that caches a
# token to .spotify_cache, which the bot reads and refreshes automatically.
# Free credentials: https://developer.spotify.com/dashboard
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv(
    "SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback"
)
SPOTIFY_SCOPE = "playlist-read-private playlist-read-collaborative"
SPOTIFY_CACHE = str(Path(__file__).parent / ".spotify_cache")

spotify_client = None
if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
    spotify_auth = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SPOTIFY_SCOPE,
        cache_path=SPOTIFY_CACHE,
        open_browser=False,  # the bot must NEVER block trying to open a browser
    )
    # Only enable Spotify if a login is already cached (from spotify_login.py).
    if spotify_auth.cache_handler.get_cached_token():
        spotify_client = spotipy.Spotify(auth_manager=spotify_auth)

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
    {
        "name": "play_previous",
        "description": "Go back and replay the previously played song, keeping the rest of the queue intact.",
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
        tools.append({
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
        })
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
        await shuffle_queue(message, args)
    elif name == "play_previous":
        await play_previous(message)


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

# A second extractor in "flat" mode: lists a playlist's videos quickly WITHOUT
# resolving each one (we resolve lazily at play time, like the Spotify tracks).
ytdl_flat = yt_dlp.YoutubeDL({"extract_flat": True, "quiet": True, "no_warnings": True})


# FFmpeg flags: -vn drops video; reconnect flags help if the stream hiccups.
# -ss <seconds> (input seek) is how we START PART-WAY into a track. A YouTube
# "&t=600" only moves the web player; to actually skip ahead we must tell FFmpeg.
def ffmpeg_options(start_seconds: int = 0) -> dict:
    before = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"
    if start_seconds > 0:
        before = f"-ss {start_seconds} " + before
    return {"before_options": before, "options": "-vn"}


# Each server gets its own player: what's playing now, what's queued, and what
# has already played (so we can go back to a previous song).
class GuildPlayer:
    def __init__(self):
        self.queue = []  # upcoming tracks (front of the list = next to play)
        self.history = []  # tracks already played (oldest first)
        self.current = None  # the track playing right now


players: dict[int, GuildPlayer] = {}


def get_player(guild_id: int) -> GuildPlayer:
    # setdefault: return the existing player, or create one on first use.
    return players.setdefault(guild_id, GuildPlayer())


async def resolve_stream(query: str):
    # Ask yt-dlp for a playable audio stream. Blocking, so run it in a thread.
    # Returns (stream_url, real_title). We resolve LAZILY — only when a track is
    # about to play — so adding a 50-song Spotify playlist is instant.
    data = await asyncio.to_thread(ytdl.extract_info, query, download=False)
    if "entries" in data:  # a search returns a list of hits; take the first
        data = data["entries"][0]
    return data["url"], data.get("title", query)


def schedule_next(guild: discord.Guild):
    # Called in FFmpeg's OWN thread when a song ends — we can't await here, so we
    # hand the coroutine back to the bot's event loop to play the next track.
    asyncio.run_coroutine_threadsafe(play_next(guild), client.loop)


async def play_next(guild: discord.Guild):
    player = get_player(guild.id)
    voice = guild.voice_client
    if voice is None:
        return
    # The song that just finished (if any) moves into history so we can go back.
    if player.current is not None:
        player.history.append(player.current)
        player.current = None
    # Loop so a track we can't load just gets skipped instead of stopping music.
    while player.queue:
        entry = player.queue.pop(0)  # take the song at the front of the line
        try:
            stream_url, title = await resolve_stream(entry["query"])
        except Exception as error:
            await entry["channel"].send(
                f"Skipping **{entry['title']}** (couldn't load it: {error})."
            )
            continue
        entry["title"] = title  # remember the real YouTube title for later display
        player.current = entry
        source = await discord.FFmpegOpusAudio.from_probe(
            stream_url, **ffmpeg_options(entry["start_seconds"])
        )
        # after=... runs when THIS song finishes -> kick off the next one.
        voice.play(source, after=lambda error: schedule_next(guild))
        start = entry["start_seconds"]
        note = f" (from {start // 60}:{start % 60:02d})" if start else ""
        await entry["channel"].send(f"▶️ Now playing: **{title}**{note}")
        return


def _spotify_query(track: dict) -> str:
    # Turn a Spotify track into a YouTube search string, e.g. "Queen - Bohemian Rhapsody".
    if not track:
        return ""
    name = track.get("name", "")
    artists = ", ".join(artist["name"] for artist in track.get("artists", []))
    return f"{artists} - {name}".strip(" -")


def spotify_tracks(url: str) -> list:
    # Read a Spotify track / playlist / album link and return YouTube search
    # strings. Blocking (network), so callers run it via asyncio.to_thread.
    match = re.search(r"open\.spotify\.com/(playlist|track|album)/([A-Za-z0-9]+)", url)
    if not match:
        return []
    kind, spotify_id = match.group(1), match.group(2)
    searches = []

    if kind == "track":
        searches.append(_spotify_query(spotify_client.track(spotify_id)))
    elif kind == "playlist":
        page = spotify_client.playlist_items(spotify_id, additional_types=["track"])
        while page:  # playlists come in pages; follow "next" until there's none
            for item in page["items"]:
                # Spotify now nests the track under "item" (older API used "track").
                searches.append(_spotify_query(item.get("item") or item.get("track")))
            page = spotify_client.next(page) if page.get("next") else None
    elif kind == "album":
        page = spotify_client.album_tracks(spotify_id)
        while page:
            for track in page["items"]:
                searches.append(_spotify_query(track))
            page = spotify_client.next(page) if page.get("next") else None

    return [search for search in searches if search]  # drop any empties


def is_spotify_url(text: str) -> bool:
    return "open.spotify.com" in text.lower()


def is_youtube_playlist(text: str) -> bool:
    # Only dedicated playlist links expand. A normal watch link (even one that
    # carries a "&list=...") just plays its single video.
    low = text.lower()
    return "youtube.com/playlist" in low or "music.youtube.com/playlist" in low


def youtube_playlist_entries(url: str) -> list:
    # Enumerate a YouTube / YouTube Music playlist's videos. Blocking, so callers
    # use asyncio.to_thread. Returns [{"query": watch_url, "title": ...}, ...].
    data = ytdl_flat.extract_info(url, download=False)
    entries = data.get("entries") or []
    out = []
    for entry in entries:
        if not entry:
            continue
        video = entry.get("url") or entry.get("id")
        if not video:
            continue
        if not video.startswith("http"):
            video = f"https://www.youtube.com/watch?v={video}"
        out.append({"query": video, "title": entry.get("title") or video})
    return out


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

    queue = get_player(message.guild.id).queue
    was_idle = not (voice.is_playing() or voice.is_paused())

    if is_spotify_url(query):
        # Spotify link -> expand into many YouTube searches and queue them all.
        if spotify_client is None:
            await message.channel.send(
                "Spotify playlists need a one-time login. Set SPOTIFY_CLIENT_ID/"
                "SECRET in .env, run `python spotify_login.py` once, then restart me."
            )
            return
        async with message.channel.typing():
            try:
                searches = await asyncio.to_thread(spotify_tracks, query)
            except Exception as error:
                await message.channel.send(f"Couldn't read that Spotify link: {error}")
                return
        if not searches:
            await message.channel.send("That Spotify link had no playable tracks.")
            return
        for search in searches:
            queue.append({
                "query": search,
                "title": search,
                "start_seconds": 0,
                "channel": message.channel,
            })
        await message.channel.send(
            f"➕ Queued **{len(searches)}** tracks from Spotify."
        )
    elif is_youtube_playlist(query):
        # A YouTube / YouTube Music playlist link -> enumerate all its videos.
        async with message.channel.typing():
            try:
                entries = await asyncio.to_thread(youtube_playlist_entries, query)
            except Exception as error:
                await message.channel.send(f"Couldn't read that playlist: {error}")
                return
        if not entries:
            await message.channel.send("That playlist had no playable videos.")
            return
        for entry in entries:
            queue.append({
                "query": entry["query"],
                "title": entry["title"],
                "start_seconds": 0,
                "channel": message.channel,
            })
        await message.channel.send(f"➕ Queued **{len(entries)}** tracks from YouTube.")
    else:
        # A single song name or link (YouTube, YouTube Music, or plain text). We
        # store it UNRESOLVED and look it up when it's this track's turn to play.
        queue.append({
            "query": query,
            "title": query,
            "start_seconds": start_seconds,
            "channel": message.channel,
        })
        if not was_idle:
            await message.channel.send(
                f"➕ Added to queue (#{len(queue)}): **{query}**"
            )

    # If nothing was playing, start now; otherwise tracks wait their turn.
    if was_idle:
        await play_next(message.guild)


async def skip_song(message: discord.Message):
    voice = message.guild.voice_client
    if voice and (voice.is_playing() or voice.is_paused()):
        voice.stop()  # stopping fires the after= callback, which plays the next
        await message.channel.send("⏭️ Skipped.")
    else:
        await message.channel.send("Nothing is playing.")


async def play_previous(message: discord.Message):
    player = get_player(message.guild.id)
    voice = message.guild.voice_client
    if not player.history:
        await message.channel.send("No previous song to go back to.")
        return
    target = player.history.pop()  # the previously played song we'll replay
    # The current song should play again AFTER the previous one, so push it back
    # to the front of the queue. This keeps the rest of the queue untouched.
    if player.current is not None:
        player.queue.insert(0, player.current)
        player.current = None  # cleared so play_next won't archive it again
    player.queue.insert(0, target)  # target jumps to the very front
    await message.channel.send("⏮️ Going back a song.")
    if voice and (voice.is_playing() or voice.is_paused()):
        voice.stop()  # fires the after= callback -> play_next plays target
    else:
        await play_next(message.guild)


async def shuffle_queue(message: discord.Message, args, call_play=True):
    queue = get_player(message.guild.id).queue
    if call_play and len(queue) == 0:
        query = " ".join(args)
        if not query:
            await message.channel.send(
                "Give me a song or link, e.g. `syntia play lofi hip hop`."
            )
        else:
            await play_music(message, query)
            await shuffle_queue(message, args, False)
    elif len(queue) < 2:
        await message.channel.send("Not enough songs in the queue to shuffle.")
        return
    random.shuffle(queue)  # shuffles the list in place
    await message.channel.send("🔀 Shuffled the queue.")


async def show_queue(message: discord.Message):
    player = get_player(message.guild.id)
    lines = []
    if player.current:
        lines.append(f"**Now playing:** {player.current['title']}")
    if player.queue:
        lines.append("**Up next:**")
        lines += [f"{i}. {track['title']}" for i, track in enumerate(player.queue, 1)]
    if not lines:
        await message.channel.send("Nothing playing and the queue is empty.")
        return
    await message.channel.send("\n".join(lines)[:2000])


async def leave_voice(message: discord.Message):
    voice = message.guild.voice_client
    if voice is None:
        await message.channel.send("I'm not in a voice channel.")
        return
    players.pop(message.guild.id, None)  # forget queue, history, and current
    await voice.disconnect()
    await message.channel.send("If you wanna be Alone then be Alone... Bye!")


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
            # Everything after "play" is the song name or YouTube link.
            query = " ".join(args)
            if not query:
                await message.channel.send(
                    "Give me a song or link, e.g. `syntia play lofi hip hop`."
                )
            else:
                await play_music(message, query)

        case "stop" | "leave" | "bye":
            # One case can match several words with the | (or) pattern.
            await leave_voice(message)

        case "skip":
            await skip_song(message)

        case "shuffle":
            await shuffle_queue(message, args)

        case "queue":
            await show_queue(message)

        case "previous" | "prev" | "back":
            await play_previous(message)

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
