r"""
One-time Spotify login.

Spotify blocks reading playlist contents with app-only credentials, so the bot
needs a USER token. Run this once:

    .venv\Scripts\python.exe spotify_login.py

It opens your browser, you approve, and the token is cached to .spotify_cache.
After that the bot reads/refreshes that token on its own — you won't need to log
in again unless you delete .spotify_cache or revoke access.

Tip: log in with the SAME account that created the Spotify app. A different
(throwaway) account works too, but must be added under the app's
Settings -> User Management in the Spotify dashboard first.
"""

import os

from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

if not CLIENT_ID or not CLIENT_SECRET:
    raise SystemExit("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env first.")

# open_browser=True + a 127.0.0.1 redirect lets spotipy run a tiny local server
# to catch the redirect automatically — no copy-pasting URLs.
auth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope="playlist-read-private playlist-read-collaborative",
    cache_path=".spotify_cache",
    open_browser=True,
)

sp = spotipy.Spotify(auth_manager=auth)
# Any authenticated call triggers the login flow and caches the token.
sp.search(q="hello", type="track", limit=1)
print("Spotify login complete. Token cached to .spotify_cache. You can start the bot now.")
