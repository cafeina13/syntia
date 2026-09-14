# Syntia — a learning Discord bot

A Discord bot built while learning, one feature at a time. It started as three
slash commands and grew into a music bot with an AI brain. The code is heavily
commented — read it top to bottom to follow how each piece works.

## What it can do

- **Music** in a voice channel from **YouTube, YouTube Music, and Spotify**
  (songs, albums, and playlists) — a queue with skip / previous / shuffle / clear,
  and seeking within a track (forward / rewind / jump to a timestamp).
- **Live volume control** for everyone in the channel, even mid-song.
- **Voice housekeeping**: join a channel without playing, leave automatically
  after a few quiet minutes (or when everyone's gone), and pick the queue back
  up mid-song if the voice connection drops.
- **AI chat**: anything it doesn't recognize as a command goes to an AI, which
  can reply *or* decide to run a command itself (tool calling), and points you
  to the right command when you mistype one.
- **Switchable AI backend**: cloud **Gemini** (free tier) or local **Ollama** —
  set by one line in `.env`.
- A custom `syntia ` chat prefix, a `help` command, and a few slash commands.

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
  `OLLAMA_MODEL` to match. Best as a backup for when Gemini's free tier hits its
  rate limit:
  - The first reply is slow while the model loads.
  - The 7B model holds about 5 GB of GPU memory while loaded, which competes
    with games.
  - It follows the system prompt less reliably than Gemini: expect the odd
    missed command, especially with typos or non-English messages.

The bot's personality and rules live in **`System_Prompt.md`** — edit that to
change how the AI behaves (restart to apply).

### 6. Spotify (fully optional)

**Spotify links work with no setup at all.** For any *public* playlist, album, or
track, the bot reads the track list off Spotify's public embed page and searches
each song on YouTube. No credentials, no login.

Adding a Spotify app only buys you one thing: **no track cap**. The embed page
returns a limited slice of a long playlist (50–100 tracks, varies), while the
official API returns all of it. Set it up only if you queue playlists longer
than that:

1. Create a free app at <https://developer.spotify.com/dashboard>. For the
   Redirect URI use `http://127.0.0.1:8888/callback` (use `127.0.0.1`, not
   `localhost`). Copy the **Client ID** and **Client Secret** into `.env`.
2. Log in once so the bot can read playlists with your account:
   ```powershell
   .venv\Scripts\python.exe spotify_login.py
   ```
   A browser opens; approve, and a token is cached to `.spotify_cache`.
3. In Spotify "Development Mode" the API reliably reads **playlists owned by the
   logged-in account**. Other people's playlists usually come back 403/404 —
   that's fine, the bot just falls back to the embed page automatically.

Note the embed fallback depends on the layout of Spotify's public pages, so it
can break if they change them. If Spotify links suddenly stop resolving, that's
the first place to look.

## Run it

```powershell
.venv\Scripts\python.exe bot.py
```

Keep the window open while the bot runs; close it (or `Ctrl+C`) to stop.

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
| `syntia volume [0-100]` (or `vol`) | Show or set the volume for everyone, e.g. `syntia volume 10` (applies instantly) |
| `syntia shuffle` | Shuffle the queue |
| `syntia shuffle <playlist>` | Load a playlist and shuffle-play it |
| `syntia stop` | Stop the music and clear the queue, but stay in the channel |
| `syntia leave` (or `bye` / `disconnect`) | Leave the voice channel |
| `syntia join` (or `come` / `summon`) | Join your voice channel without playing anything |
| `syntia timeout` | Show the idle auto-leave setting |
| `syntia timeout on` / `off` / `<minutes>` | Change it (owner or Manage Server only) |
| `syntia roll [N]` | Roll a dice (1–N, default 6) |
| `syntia help` (or `commands`) | List all commands |
| `syntia <anything else>` | Talk to the AI (it may also start music) |

`<…>` can be a search ("lofi hip hop"), a YouTube / YouTube Music / Spotify link,
or a playlist link.

If you mistype a command, the AI points you to the right one (or sends the help
list). The command list itself lives in `help_text.py` — the help command, `/help`,
and the AI all read from it.

**Idle timeout:** the bot leaves voice after a few minutes (default 5) with no
music playing, or with nobody left in the channel. Set the default with
`IDLE_TIMEOUT_MINUTES` in `.env` (`0` = off); `syntia timeout` changes it per
server until the next restart.

Slash commands also exist: `/help`, `/ping`, `/hello`, `/echo`.

## Project layout

| File | What's in it |
|---|---|
| `bot.py` | Entry point: the Discord client, message dispatch, slash commands |
| `config.py` | Settings + the Gemini / Ollama / Spotify clients (reads `.env`) |
| `music.py` | Voice playback, the queue, and resolving audio from YouTube/Spotify |
| `ai.py` | AI tools, the Gemini/Ollama backends, and the chat dispatcher |
| `help_text.py` | The command list behind `syntia help`, `/help`, and the AI prompt |
| `System_Prompt.md` | The AI's personality and rules (plain Markdown) |
| `spotify_login.py` | One-time Spotify login helper |
| `tests/` | Offline test suite (see [Running the tests](#running-the-tests)) |
| `pytest.ini` | Test runner settings |
| `requirements.txt` | What the bot needs to run |
| `requirements-dev.txt` | Extra packages for running the tests |
| `.env` | Your secrets (gitignored — never commit) |
| `.env.example` | Template for `.env` |

## `.env` reference

```
DISCORD_TOKEN=            # required — your bot token
GUILD_ID=                 # optional — your server ID for instant slash-command updates
OWNER_ID=0                # optional — your Discord user ID; the AI treats it as the verified owner
IDLE_TIMEOUT_MINUTES=5    # leave voice after N quiet minutes; 0 = off by default
DEFAULT_VOLUME=100        # starting playback volume in percent (0-100)
AI_BACKEND=gemini         # "gemini" or "ollama"
GEMINI_API_KEY=           # needed if AI_BACKEND=gemini
OLLAMA_MODEL=qwen2.5:7b-instruct-q4_K_M   # used if AI_BACKEND=ollama
SPOTIFY_CLIENT_ID=        # optional — only to lift the track cap on long playlists
SPOTIFY_CLIENT_SECRET=
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
```

## Running the tests

The `tests/` folder checks the bot's logic with small fake Discord objects —
no token, server, or voice connection needed, and it finishes in about a second.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt   # once
.venv\Scripts\python.exe -m pytest
```

One test plays a generated tone through FFmpeg to check the volume math; it's
skipped automatically if FFmpeg isn't installed. When you add a command, the
suite also fails if you forget to list it in `help_text.py`.

## Reinstalling dependencies later

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
