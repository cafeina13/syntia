"""
Music: voice playback, the per-server queue, and resolving audio from YouTube,
YouTube Music, and Spotify links. No Discord event wiring here — bot.py calls
these functions; the AI reaches them via ai.run_tool.
"""

import asyncio
import random
import re
import time

import discord
import yt_dlp

import config

# Set by bot.py once the client exists. schedule_next() runs inside FFmpeg's own
# thread when a song ends, and needs the bot's event loop to hop back onto.
client = None

# How yt-dlp finds audio. "ytsearch" means plain text like "lofi hip hop" gets
# searched on YouTube; a full YouTube URL also works. noplaylist=True keeps each
# request to a single track (playlists are expanded separately).
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
        self.offset = 0  # seconds into the track where the current source began
        self.started_at = 0.0  # monotonic clock time when the current source began
        self.seek_target = None  # (entry, seconds) set by seek() so play_next replays it

    def position(self) -> float:
        # How many seconds into the current track we are right now.
        return self.offset + (time.monotonic() - self.started_at)


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


async def _start_track(guild: discord.Guild, entry: dict, offset: int, announce: bool = True) -> bool:
    # Start ONE track at `offset` seconds in. Shared by normal playback and by
    # seek (which restarts the same track at a new offset). Returns True if it
    # actually started, False if the audio couldn't be loaded.
    voice = guild.voice_client
    if voice is None:
        return False
    try:
        stream_url, title = await resolve_stream(entry["query"])
    except Exception as error:
        await entry["channel"].send(
            f"Skipping **{entry['title']}** (couldn't load it: {error})."
        )
        return False
    entry["title"] = title  # remember the real YouTube title for later display
    player = get_player(guild.id)
    player.current = entry
    player.offset = int(offset)
    source = await discord.FFmpegOpusAudio.from_probe(
        stream_url, **ffmpeg_options(int(offset))
    )
    # after=... runs when THIS song finishes -> kick off the next one.
    voice.play(source, after=lambda error: schedule_next(guild))
    player.started_at = time.monotonic()  # start the position clock
    if announce:
        note = f" (from {int(offset) // 60}:{int(offset) % 60:02d})" if offset else ""
        await entry["channel"].send(f"▶️ Now playing: **{title}**{note}")
    return True


async def play_next(guild: discord.Guild):
    player = get_player(guild.id)
    voice = guild.voice_client
    if voice is None:
        return
    # A seek is in progress? Replay the SAME track at the new position, WITHOUT
    # touching history or the queue.
    if player.seek_target is not None:
        entry, target = player.seek_target
        player.seek_target = None
        await _start_track(guild, entry, target, announce=False)
        return
    # The song that just finished (if any) moves into history so we can go back.
    if player.current is not None:
        player.history.append(player.current)
        player.current = None
    # Loop so a track we can't load just gets skipped instead of stopping music.
    while player.queue:
        entry = player.queue.pop(0)  # take the song at the front of the line
        if await _start_track(guild, entry, entry["start_seconds"]):
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
        searches.append(_spotify_query(config.spotify_client.track(spotify_id)))
    elif kind == "playlist":
        page = config.spotify_client.playlist_items(spotify_id, additional_types=["track"])
        while page:  # playlists come in pages; follow "next" until there's none
            for item in page["items"]:
                # Spotify now nests the track under "item" (older API used "track").
                searches.append(_spotify_query(item.get("item") or item.get("track")))
            page = config.spotify_client.next(page) if page.get("next") else None
    elif kind == "album":
        page = config.spotify_client.album_tracks(spotify_id)
        while page:
            for track in page["items"]:
                searches.append(_spotify_query(track))
            page = config.spotify_client.next(page) if page.get("next") else None

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


async def ensure_voice(message: discord.Message):
    # Make sure the user is in a voice channel and the bot is connected to it.
    # Returns the voice client, or None (after messaging) if we can't join.
    if not message.author.voice or not message.author.voice.channel:
        await message.channel.send("Join a voice channel first, then try again.")
        return None
    channel = message.author.voice.channel
    voice = message.guild.voice_client
    if voice is None:
        voice = await channel.connect()
    elif voice.channel != channel:
        await voice.move_to(channel)
    return voice


async def enqueue(message: discord.Message, query: str, start_seconds: int = 0,
                  announce_add: bool = False) -> int:
    # APPEND the song(s) for `query` to the queue WITHOUT starting playback.
    # Handles Spotify links, YouTube/Music playlists, and single songs/searches.
    # Returns how many tracks were added (0 means nothing was, or an error).
    queue = get_player(message.guild.id).queue

    if is_spotify_url(query):
        if config.spotify_client is None:
            await message.channel.send(
                "Spotify playlists need a one-time login. Set SPOTIFY_CLIENT_ID/"
                "SECRET in .env, run `python spotify_login.py` once, then restart me."
            )
            return 0
        async with message.channel.typing():
            try:
                searches = await asyncio.to_thread(spotify_tracks, query)
            except Exception as error:
                await message.channel.send(f"Couldn't read that Spotify link: {error}")
                return 0
        if not searches:
            await message.channel.send("That Spotify link had no playable tracks.")
            return 0
        for search in searches:
            queue.append({"query": search, "title": search,
                          "start_seconds": 0, "channel": message.channel})
        await message.channel.send(f"➕ Queued **{len(searches)}** tracks from Spotify.")
        return len(searches)

    if is_youtube_playlist(query):
        async with message.channel.typing():
            try:
                entries = await asyncio.to_thread(youtube_playlist_entries, query)
            except Exception as error:
                await message.channel.send(f"Couldn't read that playlist: {error}")
                return 0
        if not entries:
            await message.channel.send("That playlist had no playable videos.")
            return 0
        for entry in entries:
            queue.append({"query": entry["query"], "title": entry["title"],
                          "start_seconds": 0, "channel": message.channel})
        await message.channel.send(f"➕ Queued **{len(entries)}** tracks from YouTube.")
        return len(entries)

    # A single song name or link. Stored unresolved; looked up when it plays.
    queue.append({"query": query, "title": query,
                  "start_seconds": start_seconds, "channel": message.channel})
    if announce_add:
        await message.channel.send(f"➕ Added to queue (#{len(queue)}): **{query}**")
    return 1


async def play_music(message: discord.Message, query: str, start_seconds: int = 0):
    # "Play now": REPLACE the queue with this song/playlist and start it fresh.
    voice = await ensure_voice(message)
    if voice is None:
        return
    player = get_player(message.guild.id)
    saved = player.queue[:]  # snapshot, so a failed lookup doesn't wipe the queue
    player.queue.clear()
    if await enqueue(message, query, start_seconds) == 0:
        player.queue[:] = saved  # restore on failure
        return
    # Start fresh: stopping the current song fires its after-callback, which plays
    # the new queue front; if nothing was playing, just begin.
    if voice.is_playing() or voice.is_paused():
        voice.stop()
    else:
        await play_next(message.guild)


async def add_music(message: discord.Message, query: str, start_seconds: int = 0):
    # "Add": append to the queue without disturbing the current song.
    voice = await ensure_voice(message)
    if voice is None:
        return
    was_idle = not (voice.is_playing() or voice.is_paused())
    if await enqueue(message, query, start_seconds, announce_add=not was_idle) == 0:
        return
    if was_idle:  # nothing playing -> start the track we just added
        await play_next(message.guild)


async def clear_queue(message: discord.Message):
    # Empty the upcoming queue but leave the current song playing.
    player = get_player(message.guild.id)
    count = len(player.queue)
    if count == 0:
        await message.channel.send("The queue is already empty.")
        return
    player.queue.clear()
    await message.channel.send(f"🗑️ Cleared {count} song(s) from the queue.")


async def skip_song(message: discord.Message):
    voice = message.guild.voice_client
    if voice and (voice.is_playing() or voice.is_paused()):
        voice.stop()  # stopping fires the after= callback, which plays the next
        await message.channel.send("⏭️ Skipped.")
    else:
        await message.channel.send("Nothing is playing.")


def fmt_time(seconds: int) -> str:
    # 3720 -> "1:02:00", 325 -> "5:25". Used in the "Jumped to ..." message.
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def parse_timestamp(text: str):
    # "1:02:00" -> 3720, "5:25" -> 325, "90" -> 90. Returns seconds, or None if
    # the text isn't a valid time.
    parts = text.split(":")
    if len(parts) > 3 or not all(p.isdigit() for p in parts):
        return None
    total = 0
    for part in parts:
        total = total * 60 + int(part)
    return total


async def seek(message: discord.Message, seconds: int = 0, to: int | None = None):
    # Move within the CURRENT track. `to` = an ABSOLUTE position to jump TO;
    # `seconds` = a RELATIVE jump (positive forward, negative back). We can't move
    # the playhead in place, so we restart the same track at the new offset;
    # seek_target tells play_next to replay rather than advance.
    player = get_player(message.guild.id)
    voice = message.guild.voice_client
    if voice is None or player.current is None or not voice.is_playing():
        await message.channel.send("Nothing is playing to seek.")
        return
    if to is not None:
        target = max(0, int(to))
        forward = target >= player.position()
    else:
        target = max(0, int(player.position() + seconds))
        forward = seconds >= 0
    player.seek_target = (player.current, target)
    voice.stop()  # fires the after= callback -> play_next replays at `target`
    arrow = "⏩" if forward else "⏪"
    await message.channel.send(f"{arrow} Jumped to {fmt_time(target)}.")


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


async def shuffle_queue(message: discord.Message, query: str = ""):
    if query:
        # Shuffle-play: REPLACE the queue, shuffle it, THEN start — so the first
        # track is random, not the playlist's original opener.
        voice = await ensure_voice(message)
        if voice is None:
            return
        player = get_player(message.guild.id)
        saved = player.queue[:]
        player.queue.clear()
        if await enqueue(message, query) == 0:
            player.queue[:] = saved
            return
        if len(player.queue) >= 2:
            random.shuffle(player.queue)
            await message.channel.send("🔀 Shuffled the queue.")
        if voice.is_playing() or voice.is_paused():
            voice.stop()  # after-callback plays the new (random) front
        else:
            await play_next(message.guild)
        return

    # No query: just shuffle whatever is already queued.
    queue = get_player(message.guild.id).queue
    if len(queue) < 2:
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
