"""
Music: voice playback, the per-server queue, and resolving audio from YouTube,
YouTube Music, and Spotify links. No Discord event wiring here — bot.py calls
these functions; the AI reaches them via ai.run_tool.
"""

import asyncio
import json
import random
import re
import time
import urllib.request

import discord
import yt_dlp

import config

# Set by bot.py once the client exists. schedule_next() runs inside FFmpeg's own
# thread when a song ends, and needs the bot's event loop to hop back onto.
client = None

# The VoiceClient class every voice connection uses. The optional voice assistant
# swaps in one that also tracks who is speaking from the moment it connects
# (assistant/voice_receive.py); the music bot itself doesn't need that.
VOICE_CLIENT_CLASS = discord.VoiceClient

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
        self.paused_at = None  # monotonic clock time of a pause, or None while playing
        self.seek_target = None  # (entry, seconds) set by seek() so play_next replays it
        # Set while playing so we can find our way back after a dropped
        # connection: where we were speaking, and where we were announcing.
        self.voice_channel_id = None
        self.text_channel = None
        self.interrupted = False  # True once voice died with tracks still queued
        self.resume_attempts = 0  # give up after MAX_RESUME_ATTEMPTS in a row

    def position(self) -> float:
        # How many seconds into the current track we are right now. While paused
        # the clock is frozen at the moment of the pause.
        now = self.paused_at if self.paused_at is not None else time.monotonic()
        return self.offset + (now - self.started_at)


# A flapping connection shouldn't have us reconnecting forever.
MAX_RESUME_ATTEMPTS = 3

players: dict[int, GuildPlayer] = {}


def get_player(guild_id: int) -> GuildPlayer:
    # setdefault: return the existing player, or create one on first use.
    return players.setdefault(guild_id, GuildPlayer())


async def resolve_stream(query: str) -> dict:
    # Ask yt-dlp for a playable audio stream. Blocking, so run it in a thread.
    # We resolve LAZILY — only when a track is about to play — so adding a
    # 50-song Spotify playlist is instant. Returns:
    #   url          the audio stream FFmpeg reads
    #   title        the real video title
    #   webpage_url  the video's page (for resume links), or None
    #   duration     length in seconds, or None (e.g. live streams)
    data = await asyncio.to_thread(ytdl.extract_info, query, download=False)
    if "entries" in data:  # a search returns a list of hits; take the first
        data = data["entries"][0]
    return {
        "url": data["url"],
        "title": data.get("title", query),
        "webpage_url": data.get("webpage_url"),
        "duration": data.get("duration"),
    }


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
        info = await resolve_stream(entry["query"])
    except Exception as error:
        await entry["channel"].send(
            f"Skipping **{entry['title']}** (couldn't load it: {error})."
        )
        return False
    title = info["title"]
    entry["title"] = title  # remember the real YouTube title for later display
    entry["duration"] = info["duration"]
    if info["webpage_url"]:
        # Pin the exact video. A search like "lofi mix" could find a DIFFERENT
        # video next time — and a seek, previous, or reconnect re-resolves it.
        entry["query"] = entry["webpage_url"] = info["webpage_url"]
    stream_url = info["url"]
    player = get_player(guild.id)
    player.current = entry
    player.offset = int(offset)
    # Breadcrumbs for resume_after_reconnect(): the voice channel to rejoin and
    # the text channel to speak in. A clean start also clears the retry budget.
    player.voice_channel_id = voice.channel.id
    player.text_channel = entry["channel"]
    player.resume_attempts = 0
    # Decode to raw PCM and wrap it in a volume transformer, so the volume can
    # change live mid-song. (Passing Opus straight through is cheaper, but its
    # loudness can't be touched.)
    source = discord.PCMVolumeTransformer(
        discord.FFmpegPCMAudio(stream_url, **ffmpeg_options(int(offset))),
        volume=get_volume(guild.id) / 100,
    )
    # The voice assistant may be mid-sentence ("Tamam, hallediyorum") when the
    # song is ready. Speech is marked is_speech; cut it so play() doesn't refuse.
    if voice.is_playing() and getattr(voice.source, "is_speech", False):
        voice.stop()
    # after=... runs when THIS song finishes -> kick off the next one.
    voice.play(source, after=lambda error: schedule_next(guild))
    player.started_at = time.monotonic()  # start the position clock
    player.paused_at = None  # a fresh source always starts out playing
    if announce:
        note = f" (from {fmt_time(offset)})" if offset else ""
        # silent=True -> the message still posts, but Discord skips the push
        # notification. This one fires on its own at every track change, so a
        # ping per song in a long queue gets old fast.
        await entry["channel"].send(f"▶️ Now playing: **{title}**{note}", silent=True)
        # A spoken request's reply channel (assistant/request.py) can react — e.g.
        # say the title out loud. A normal Discord text channel has no such hook.
        hook = getattr(entry["channel"], "on_track_started", None)
        if hook is not None:
            await hook(entry)
    return True


async def play_next(guild: discord.Guild):
    player = get_player(guild.id)
    voice = guild.voice_client
    if voice is None:
        # Voice went away under us — a gateway reconnect, a kick, or a network
        # drop. Park the queue and say so instead of going quiet.
        await _handle_voice_loss(guild, player)
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


async def _handle_voice_loss(guild: discord.Guild, player: GuildPlayer):
    # Called when the after-callback fires but we're no longer in voice. Push the
    # track that was playing back to the FRONT of the queue, tagged with how far
    # we got, so a resume picks up mid-song rather than restarting it.
    if player.interrupted:
        return  # already parked — don't announce the same drop twice
    if player.current is None and not player.queue:
        return  # nothing was going on; nothing to mourn
    if player.current is not None:
        entry = player.current
        entry["start_seconds"] = max(0, int(player.position()))
        player.queue.insert(0, entry)
        player.current = None
    player.interrupted = True
    if player.text_channel is not None:
        await player.text_channel.send(
            f"🔌 Lost the voice connection. Queue is paused with "
            f"**{len(player.queue)}** track(s) left — I'll pick it up "
            "automatically if I get back in.",
            silent=True,
        )


async def resume_after_reconnect(guild: discord.Guild):
    # Called after the gateway comes back. Rejoins the channel we were in and
    # restarts the queue. No-op unless _handle_voice_loss() flagged this guild.
    player = players.get(guild.id)
    if player is None or not player.interrupted:
        return
    if not player.queue or player.voice_channel_id is None:
        player.interrupted = False
        return
    if player.resume_attempts >= MAX_RESUME_ATTEMPTS:
        return  # stay parked; a manual play command still works
    player.resume_attempts += 1
    channel = guild.get_channel(player.voice_channel_id)
    if channel is None:  # channel deleted, or we can't see it any more
        player.interrupted = False
        return
    await asyncio.sleep(2)  # let the fresh gateway session settle before voice
    try:
        if guild.voice_client is None:
            voice = await channel.connect(cls=VOICE_CLIENT_CLASS)
            _run_in_background(_clear_speaking_ring(voice))
        else:
            await guild.voice_client.move_to(channel)
    except Exception:
        return  # leave the flag set so the next reconnect tries again
    player.interrupted = False
    if player.text_channel is not None:
        await player.text_channel.send(
            "🔁 Back online — picking up where I left off.", silent=True
        )
    await play_next(guild)


async def recover_all():
    # Fired from bot.py on every (re)connect. Guilds we were never playing in
    # skip out immediately via the interrupted flag.
    for guild_id in list(players):
        guild = client.get_guild(guild_id)
        if guild is not None:
            await resume_after_reconnect(guild)


def _spotify_query(track: dict) -> str:
    # Turn a Spotify track into a YouTube search string, e.g. "Queen - Bohemian Rhapsody".
    if not track:
        return ""
    name = track.get("name", "")
    artists = ", ".join(artist["name"] for artist in track.get("artists", []))
    return f"{artists} - {name}".strip(" -")


def spotify_tracks(url: str) -> list:
    # Read a Spotify track / playlist / album link -> YouTube search strings.
    # Blocking (network), so callers run it via asyncio.to_thread.
    # Prefer the official API (full list, no cap) when we have a login; otherwise
    # — or for playlists the API can't read, like other people's — scrape the
    # public embed page.
    match = re.search(r"open\.spotify\.com/(playlist|track|album)/([A-Za-z0-9]+)", url)
    if not match:
        return []
    kind, spotify_id = match.group(1), match.group(2)
    if config.spotify_client is not None:
        try:
            return _spotify_tracks_api(kind, spotify_id)
        except Exception:
            pass  # e.g. someone else's playlist (403) -> fall back to scraping
    return _scrape_spotify(kind, spotify_id)


def _spotify_tracks_api(kind: str, spotify_id: str) -> list:
    # Official-API path (needs a login). Full track list, no ~100 cap.
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


def _find_tracklist(obj):
    # Recursively find the "trackList" array inside the embed page's JSON.
    if isinstance(obj, dict):
        if isinstance(obj.get("trackList"), list):
            return obj["trackList"]
        for value in obj.values():
            found = _find_tracklist(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_tracklist(value)
            if found:
                return found
    return None


def _scrape_spotify(kind: str, spotify_id: str) -> list:
    # No-API fallback: the public embed page ships the track list as JSON. Works
    # for any PUBLIC item, but caps at ~100 tracks and can break if Spotify
    # changes their page. ToS gray area — fine for a personal bot.
    embed = f"https://open.spotify.com/embed/{kind}/{spotify_id}"
    request = urllib.request.Request(embed, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(request, timeout=20).read().decode("utf-8", "replace")
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not match:
        return []
    tracklist = _find_tracklist(json.loads(match.group(1))) or []
    searches = []
    for track in tracklist:
        query = f"{track.get('subtitle', '')} - {track.get('title', '')}".strip(" -")
        if query:
            searches.append(query)
    return searches


def is_spotify_url(text: str) -> bool:
    return "open.spotify.com" in text.lower()


def is_youtube_playlist(text: str) -> bool:
    # Only dedicated playlist links expand. A normal watch link (even one that
    # carries a "&list=...") just plays its single video.
    low = text.lower()
    return "youtube.com/playlist" in low or "music.youtube.com/playlist" in low


def is_youtube_url(text: str) -> bool:
    return re.search(r"(^|[/.])(youtube\.com|youtu\.be)/", text.lower()) is not None


def youtube_start_seconds(url: str) -> int:
    # The start time in a YouTube link: "t=3753", "t=3753s", or "t=1h2m33s".
    # Returns 0 when there's none (or it isn't a YouTube link).
    if not is_youtube_url(url):
        return 0
    match = re.search(r"[?&#]t=([0-9hms]+)", url)
    if not match:
        return 0
    value = match.group(1)
    if value.isdigit():
        return int(value)
    parts = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", value)
    if not parts:
        return 0
    hours, minutes, seconds = (int(part or 0) for part in parts.groups())
    return hours * 3600 + minutes * 60 + seconds


def resume_link(url: str | None, seconds: int) -> str | None:
    # The same YouTube link, set to start at `seconds`. None for anything that
    # isn't a YouTube link (we can't promise other sites honour "t=").
    if not url or not is_youtube_url(url):
        return None
    url = re.sub(r"([?&])t=[^&#]*&?", r"\1", url).rstrip("?&")  # drop an old t=
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}t={int(seconds)}"


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


async def join_voice(message: discord.Message):
    # Join the user's voice channel WITHOUT playing anything (debugging, and a
    # base for future voice features). The idle timeout still applies.
    if not message.author.voice or not message.author.voice.channel:
        await message.channel.send("Join a voice channel first, then try again.")
        return
    channel = message.author.voice.channel
    voice = message.guild.voice_client
    if voice is not None and voice.channel == channel:
        await message.channel.send("I'm already here.")
        return
    try:
        voice = await ensure_voice(message)
    except Exception as error:
        await message.channel.send(f"Couldn't join **{channel.name}**: {error}")
        return
    # Remember where we are, so the idle timeout knows where to say goodbye.
    player = get_player(message.guild.id)
    player.voice_channel_id = voice.channel.id
    player.text_channel = message.channel
    await message.channel.send(f"Joined **{voice.channel.name}**.")


# asyncio only keeps weak references to tasks, so hold on to them until done.
_background_tasks: set[asyncio.Task] = set()


def _run_in_background(coro):
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _clear_speaking_ring(voice: discord.VoiceClient):
    # A freshly connected bot shows a stuck green "speaking" ring until it sends
    # some audio: the "not speaking" signal from the handshake alone doesn't
    # clear it. Stopping a song clears it by sending a few silent frames, so we
    # do the same right after connecting.
    await asyncio.sleep(1)  # let Discord register us in the channel first
    try:
        if not voice.is_connected() or voice.is_playing():
            return  # a song already started; it handles speaking itself
        await voice.ws.speak(discord.SpeakingState.none)
        for _ in range(5):
            voice.send_audio_packet(discord.opus.OPUS_SILENCE, encode=False)
    except Exception:
        pass  # purely cosmetic — never break voice over it


async def ensure_voice(message: discord.Message):
    # Make sure the user is in a voice channel and the bot is connected to it.
    # Returns the voice client, or None (after messaging) if we can't join.
    if not message.author.voice or not message.author.voice.channel:
        await message.channel.send("Join a voice channel first, then try again.")
        return None
    channel = message.author.voice.channel
    voice = message.guild.voice_client
    if voice is None:
        voice = await channel.connect(cls=VOICE_CLIENT_CLASS)
        # In the background, so a play command doesn't wait on it.
        _run_in_background(_clear_speaking_ring(voice))
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
        async with message.channel.typing():
            try:
                searches = await asyncio.to_thread(spotify_tracks, query)
            except Exception as error:
                await message.channel.send(f"Couldn't read that Spotify link: {error}")
                return 0
        if not searches:
            await message.channel.send("Couldn't get any tracks from that Spotify link.")
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
    # A YouTube link carrying a time ("...&t=3753") starts from there, so the
    # link `syntia np` hands out picks up exactly where you left off.
    if not start_seconds:
        start_seconds = youtube_start_seconds(query)
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
    was_paused = voice is not None and voice.is_paused()
    if voice is None or player.current is None or not (voice.is_playing() or was_paused):
        await message.channel.send("Nothing is playing to seek.")
        return
    if to is not None:
        target = max(0, int(to))
        forward = target >= player.position()
    else:
        target = max(0, int(player.position() + seconds))
        forward = seconds >= 0
    player.seek_target = (player.current, target)
    # Fires the after= callback -> play_next replays at `target`. The restarted
    # track starts out playing, so seeking also un-pauses.
    voice.stop()
    arrow = "⏩" if forward else "⏪"
    note = " (and resumed)" if was_paused else ""
    await message.channel.send(f"{arrow} Jumped to {fmt_time(target)}{note}.")


async def pause_music(message: discord.Message):
    # Pause the current song in place. A paused bot counts as idle, so the idle
    # timeout still applies.
    player = get_player(message.guild.id)
    voice = message.guild.voice_client
    if voice is not None and voice.is_paused():
        await message.channel.send("Already paused. `syntia resume` to continue.")
        return
    if voice is None or not voice.is_playing():
        await message.channel.send("Nothing is playing.")
        return
    voice.pause()
    player.paused_at = time.monotonic()  # freeze the position clock
    await message.channel.send("⏸️ Paused. `syntia resume` to continue.")


async def resume_music(message: discord.Message):
    player = get_player(message.guild.id)
    voice = message.guild.voice_client
    if voice is not None and voice.is_playing():
        await message.channel.send("It's already playing.")
        return
    if voice is None or not voice.is_paused():
        await message.channel.send("Nothing is paused.")
        return
    if player.paused_at is not None:
        # Slide the start time forward by the length of the pause, so position()
        # doesn't count the paused minutes as listened.
        player.started_at += time.monotonic() - player.paused_at
        player.paused_at = None
    voice.resume()
    await message.channel.send("▶️ Resumed.")


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


async def now_playing(message: discord.Message):
    # What's playing and how far in — plus a command that restarts it from this
    # exact spot, handy before restarting the bot during a long mix.
    player = get_player(message.guild.id)
    voice = message.guild.voice_client
    if (voice is None or player.current is None
            or not (voice.is_playing() or voice.is_paused())):
        await message.channel.send("Nothing is playing.")
        return
    entry = player.current
    position = int(player.position())
    duration = entry.get("duration")
    if duration:
        position = min(position, int(duration))
        timing = f"{fmt_time(position)} / {fmt_time(duration)}"
    else:
        timing = fmt_time(position)
    state = " (paused)" if voice.is_paused() else ""
    lines = [f"🎵 **{entry['title']}**", f"`{timing}`{state}"]
    link = resume_link(entry.get("webpage_url"), position)
    if link:
        # In backticks: easy to copy, and Discord won't unfurl a big preview.
        lines.append(f"Pick up here later: `syntia play {link}`")
    await message.channel.send("\n".join(lines))


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


async def stop_music(message: discord.Message):
    # End playback and empty the queue, but STAY in the channel (leaving is
    # `syntia leave`). If nothing starts again, the idle timeout takes over.
    voice = message.guild.voice_client
    if voice is None or not (voice.is_playing() or voice.is_paused()):
        await message.channel.send("Nothing is playing.")
        return
    player = get_player(message.guild.id)
    player.queue.clear()
    player.seek_target = None  # a pending seek must not restart the track
    # Stopping fires the after-callback: play_next files the song into history
    # (so `previous` still works) and finds an empty queue, so it goes quiet.
    voice.stop()
    await message.channel.send("⏹️ Stopped. I'll hang around here.")


async def disconnect(guild: discord.Guild):
    # Shared by `stop` and the idle timeout. Forget the player FIRST: disconnecting
    # fires the after-callback, which then finds nothing to "recover".
    voice = guild.voice_client
    players.pop(guild.id, None)  # forget queue, history, and current
    idle_since.pop(guild.id, None)
    if voice is not None:
        await voice.disconnect()


async def leave_voice(message: discord.Message):
    if message.guild.voice_client is None:
        await message.channel.send("I'm not in a voice channel.")
        return
    await disconnect(message.guild)
    await message.channel.send("If you wanna be Alone then be Alone... Bye!")


# --- Volume ------------------------------------------------------------------
# Percent per server, kept outside GuildPlayer so it survives a leave. In memory
# only: a restart resets every server to DEFAULT_VOLUME.
volumes: dict[int, int] = {}


def get_volume(guild_id: int) -> int:
    return volumes.get(guild_id, config.DEFAULT_VOLUME)


async def set_volume(message: discord.Message, level: str = ""):
    # `syntia volume` shows it; `syntia volume 10` (or "10%") sets it, live.
    guild_id = message.guild.id
    text = str(level).strip().rstrip("%")
    if not text:
        await message.channel.send(f"🔊 Volume is **{get_volume(guild_id)}%**.")
        return
    if not text.isdigit() or int(text) > 100:
        await message.channel.send("Give me a volume from 0 to 100, e.g. `syntia volume 10`.")
        return
    percent = volumes[guild_id] = int(text)
    # Apply it to the song playing right now; later tracks read it on start.
    voice = message.guild.voice_client
    if voice is not None and isinstance(voice.source, discord.PCMVolumeTransformer):
        voice.source.volume = percent / 100
    icon = "🔇" if percent == 0 else "🔉" if percent < 50 else "🔊"
    await message.channel.send(f"{icon} Volume set to **{percent}%**.")


# --- Idle timeout ------------------------------------------------------------
# Per-server settings live here (not on GuildPlayer, which is thrown away on
# leave). In memory only: a restart resets every server to the .env default.
timeout_settings: dict[int, dict] = {}
# When each server's bot first went idle / was left alone (monotonic clock).
idle_since: dict[int, float] = {}

MAX_TIMEOUT_MINUTES = 240


def get_timeout(guild_id: int) -> dict:
    default = config.IDLE_TIMEOUT_MINUTES
    return timeout_settings.setdefault(
        guild_id, {"enabled": default > 0, "minutes": default if default > 0 else 5}
    )


async def check_idle():
    # Polled every 30 s by bot.py. Polling (instead of hooking every playback
    # path) means nothing slips through; a brief gap during a seek or track
    # change just gets cleared again on the next poll.
    now = time.monotonic()
    for voice in list(client.voice_clients):
        guild = voice.guild
        setting = get_timeout(guild.id)
        idle = not voice.is_playing()  # a paused song counts as idle too
        alone = not any(not member.bot for member in voice.channel.members)
        if not setting["enabled"] or not (idle or alone):
            idle_since.pop(guild.id, None)
            continue
        since = idle_since.setdefault(guild.id, now)
        if now - since < setting["minutes"] * 60:
            continue
        player = players.get(guild.id)
        channel = player.text_channel if player else None
        await disconnect(guild)
        if channel is not None:
            reason = (
                "Everyone left, so I did too."
                if alone
                else f"No music for {setting['minutes']} min, heading out."
            )
            await channel.send(reason, silent=True)


def _can_change_timeout(member: discord.Member) -> bool:
    # The verified owner (by ID, like the AI's check) or a server admin.
    is_owner = bool(config.OWNER_ID) and member.id == config.OWNER_ID
    return is_owner or member.guild_permissions.manage_guild


async def set_timeout(message: discord.Message, arg: str = ""):
    # `syntia timeout` shows the setting; on / off / <minutes> change it (admins).
    setting = get_timeout(message.guild.id)
    arg = arg.lower()
    if not arg:
        state = f"on, {setting['minutes']} min" if setting["enabled"] else "off"
        await message.channel.send(f"Idle timeout: **{state}**.")
        return
    if not _can_change_timeout(message.author):
        await message.channel.send(
            "Only the owner or someone with Manage Server can change the timeout."
        )
        return
    if arg == "off":
        setting["enabled"] = False
        idle_since.pop(message.guild.id, None)
        await message.channel.send("Idle timeout **off** - I'll stay until told to leave.")
    elif arg == "on":
        setting["enabled"] = True
        await message.channel.send(f"Idle timeout **on** ({setting['minutes']} min).")
    elif arg.isdigit() and 1 <= int(arg) <= MAX_TIMEOUT_MINUTES:
        setting["enabled"] = True
        setting["minutes"] = int(arg)
        await message.channel.send(f"Idle timeout **on** ({arg} min).")
    else:
        await message.channel.send(
            f"Use `syntia timeout on`, `off`, or a number of minutes (1-{MAX_TIMEOUT_MINUTES})."
        )
