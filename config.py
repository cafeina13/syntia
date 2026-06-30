"""
Configuration and external clients for Syntia.

Loads secrets from .env and creates the AI / Spotify clients once, so the rest
of the code can just `import config` and use them. No Discord logic lives here.
"""

import os
from pathlib import Path

import ollama
import spotipy
from dotenv import load_dotenv
from google import genai
from spotipy.oauth2 import SpotifyOAuth

# Load secrets from the .env file into the environment. We keep them OUT of the
# code so they never get committed to git.
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

# Optional: the ID of YOUR server, for instant slash-command updates in dev.
# Global commands take up to an hour to appear; guild commands appear instantly.
GUILD_ID = os.getenv("GUILD_ID")

# Optional: YOUR Discord user ID (a permanent number, unlike a nickname, so it
# can't be faked). When set, the AI treats this user as the verified owner.
# 0 = no owner.
OWNER_ID = int(os.getenv("OWNER_ID") or 0)

# Our custom "caller" — type this (then a command) in chat to talk to the bot.
PREFIX = "syntia "

# Which AI backend to use: "gemini" (cloud, free tier) or "ollama" (local).
# Ollama is great for offline testing — no rate limits.
AI_BACKEND = os.getenv("AI_BACKEND", "gemini").lower()

# Gemini (cloud). Free key at https://aistudio.google.com/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
GEMINI_MODEL = "gemini-2.5-flash"  # fast and free-tier friendly

# Ollama (local, http://localhost:11434). Use a tool-capable model (qwen2.5 is).
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
ollama_client = ollama.AsyncClient()

# Spotify: read a playlist/album/track's song list, then find the audio on
# YouTube (Spotify itself can't be streamed). Reading PLAYLISTS needs a one-time
# user login (run spotify_login.py), which caches a token to .spotify_cache that
# the bot reads and refreshes. Free credentials: https://developer.spotify.com/dashboard
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
# NOTE: encoding="utf-8" is required — without it, Windows may fail on characters
# like the em dash.
PROMPT_FILE = Path(__file__).parent / "System_Prompt.md"
try:
    AI_PRE_PROMPT = PROMPT_FILE.read_text(encoding="utf-8").strip()
except FileNotFoundError:
    AI_PRE_PROMPT = "You are Syntia, a helpful Discord bot."
