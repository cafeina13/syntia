# Syntia — a learning Discord bot

A Discord bot built while learning, one feature at a time. It started as three
slash commands and grew into a music bot with an AI brain. The code is heavily
commented — read it top to bottom to follow how each piece works.

## What it can do

- **Music** in a voice channel from **YouTube, YouTube Music, and Spotify**
  (songs, albums, and playlists) — a queue with skip / previous / shuffle / clear,
  and seeking within a track (forward / rewind / jump to a timestamp).
- **AI chat**: anything it doesn't recognize as a command goes to an AI, which
  can reply *or* decide to run a music command itself (tool calling).
- **Switchable AI backend**: cloud **Gemini** (free tier) or local **Ollama** —
  set by one line in `.env`.
- A custom `syntia ` chat prefix, plus a few slash commands.

## Requirements

- **Python 3.13** (3.12 also fine).
- **FFmpeg** installed and on your PATH (it streams the audio). Check with
  `ffmpeg -version`.

## Setup

### 1. Install the dependencies

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> Everything is run with `.venv\Scripts\python.exe` (the environment's Python,
> which has the libraries) — **not** a bare `python`.

### 2. Create the bot and get its token

1. Go to <https://discord.com/developers/applications> → **New Application**.
2. Left sidebar → **Bot** → **Reset Token** → **Copy** it (treat it like a password).
3. Copy `.env.example` to a new file named `.env` and paste the token:
   ```
   DISCORD_TOKEN=your-real-token-here
   ```

### 3. Turn on the privileged intents

Developer Portal → your app → **Bot** → **Privileged Gateway Intents** →
turn **ON** **Message Content Intent** (needed so the bot can read the `syntia `
prefix) and save.

### 4. Invite the bot to your server

Developer Portal → **OAuth2 → URL Generator**:

- **Scopes:** `bot` and `applications.commands`
- **Bot Permissions:** `Send Messages`, plus `Connect` and `Speak` (for voice)

Open the generated URL, pick your server, authorize.

### 5. Pick an AI backend (`.env`)

```
AI_BACKEND=gemini        # "gemini" (cloud) or "ollama" (local)
```

- **Gemini** (recommended, simplest): get a free key at
  <https://aistudio.google.com/apikey> and set `GEMINI_API_KEY` in `.env`.
- **Ollama** (local, no API key): install [Ollama](https://ollama.com), pull a
  tool-capable model (`ollama pull qwen2.5:7b-instruct-q4_K_M`), and set
  `OLLAMA_MODEL` to match. The first reply is slow while the model loads.

The bot's personality and rules live in **`System_Prompt.md`** — edit that to
change how the AI behaves (restart to apply).

### 6. Spotify (optional — only for Spotify links)

1. Create a free app at <https://developer.spotify.com/dashboard>. For the
   Redirect URI use `http://127.0.0.1:8888/callback` (use `127.0.0.1`, not
   `localhost`). Copy the **Client ID** and **Client Secret** into `.env`.
2. Reading playlists needs a one-time login (Spotify blocks playlists for
   app-only tokens). Run it once:
   ```powershell
   .venv\Scripts\python.exe spotify_login.py
   ```
   A browser opens; approve, and a token is cached to `.spotify_cache`.
3. Note: in Spotify "Development Mode" you can reliably read **playlists owned by
   the logged-in account**; other people's playlists may be blocked.

## Run it

Either:

```powershell
.venv\Scripts\python.exe bot.py
```

…or just **double-click `start_syntia.bat`**. Keep the window open while the bot
runs; close it (or `Ctrl+C`) to stop.

> The bot only runs while this machine is on — there's no cloud server.

## Commands

Type these in any text channel (you must be in a voice channel for music):

| Command | What it does |
|---|---|
| `syntia play <song / playlist / link>` | **Replace** the queue and play now |
| `syntia add <…>` (or `enqueue`) | **Append** to the queue |
| `syntia queue` | Show what's playing and what's next |
| `syntia clear` | Empty the upcoming queue |
| `syntia skip` | Skip to the next song |
| `syntia previous` (or `prev` / `back`) | Replay the previous song |
| `syntia forward [N]` (or `fwd` / `ff`) | Jump ahead N seconds in the current track (default 30) |
| `syntia rewind [N]` (or `rw`) | Jump back N seconds (default 30) |
| `syntia seek <time>` | Jump to a position, e.g. `syntia seek 1:02:00` |
| `syntia shuffle` | Shuffle the queue |
| `syntia shuffle <playlist>` | Load a playlist and shuffle-play it |
| `syntia stop` (or `leave` / `bye`) | Stop and leave the voice channel |
| `syntia roll [N]` | Roll a dice (1–N, default 6) |
| `syntia <anything else>` | Talk to the AI (it may also start music) |

`<…>` can be a search ("lofi hip hop"), a YouTube / YouTube Music / Spotify link,
or a playlist link.

Slash commands also exist: `/ping`, `/hello`, `/echo`.

## Project layout

| File | What's in it |
|---|---|
| `bot.py` | Entry point: the Discord client, message dispatch, slash commands |
| `config.py` | Settings + the Gemini / Ollama / Spotify clients (reads `.env`) |
| `music.py` | Voice playback, the queue, and resolving audio from YouTube/Spotify |
| `ai.py` | AI tools, the Gemini/Ollama backends, and the chat dispatcher |
| `System_Prompt.md` | The AI's personality and rules (plain Markdown) |
| `spotify_login.py` | One-time Spotify login helper |
| `start_syntia.bat` | Double-click launcher |
| `.env` | Your secrets (gitignored — never commit) |
| `.env.example` | Template for `.env` |

## `.env` reference

```
DISCORD_TOKEN=        # required — your bot token
GUILD_ID=             # optional — your server ID for instant slash-command updates
OWNER_ID=0            # optional — your Discord user ID; the AI treats it as the verified owner
AI_BACKEND=gemini     # "gemini" or "ollama"
GEMINI_API_KEY=       # needed if AI_BACKEND=gemini
OLLAMA_MODEL=qwen2.5:7b-instruct-q4_K_M   # used if AI_BACKEND=ollama
SPOTIFY_CLIENT_ID=    # optional — for Spotify links
SPOTIFY_CLIENT_SECRET=
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
```

## Reinstalling dependencies later

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
