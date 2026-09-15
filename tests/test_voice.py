"""Joining voice, the speaking-ring fix, starting a track, and volume."""

import shutil
from types import SimpleNamespace

import discord
import pytest
from conftest import FakeTextChannel, FakeVoice, FakeVoiceChannel, make_member, make_message

import music


@pytest.fixture
def no_sleep(monkeypatch):
    # _clear_speaking_ring waits a second first; tests don't need to.
    async def instant(_seconds):
        return None

    monkeypatch.setattr(music.asyncio, "sleep", instant)


# --- join ------------------------------------------------------------------


async def test_join_requires_the_user_in_voice(guild, text_channel):
    await music.join_voice(make_message(guild, make_member(), text_channel))
    assert "Join a voice channel first" in text_channel.last


async def test_join_connects_and_remembers_the_channels(guild, text_channel, monkeypatch):
    ring_cleared = []
    monkeypatch.setattr(music, "_run_in_background", lambda coro: (ring_cleared.append(1), coro.close()))
    vc = FakeVoiceChannel(guild, name="Lounge")
    await music.join_voice(make_message(guild, make_member(voice_channel=vc), text_channel))

    assert guild.voice_client is not None and guild.voice_client.channel is vc
    player = music.get_player(guild.id)
    assert player.voice_channel_id == vc.id and player.text_channel is text_channel
    assert text_channel.last == "Joined **Lounge**."
    assert ring_cleared == [1]  # the ring fix is scheduled on a fresh connect
    assert not guild.voice_client.playing  # joined silently


async def test_join_when_already_there(connected):
    voice, message = connected
    await music.join_voice(message)
    assert message.channel.last == "I'm already here."


async def test_join_moves_to_the_users_channel(connected, guild):
    voice, message = connected
    other = FakeVoiceChannel(guild, name="Other")
    message.author.voice = SimpleNamespace(channel=other)
    await music.join_voice(message)
    assert voice.channel is other and message.channel.last == "Joined **Other**."


async def test_join_reports_connection_errors(guild, text_channel):
    vc = FakeVoiceChannel(guild, name="Locked", fail_connect=RuntimeError("Missing Permissions"))
    await music.join_voice(make_message(guild, make_member(voice_channel=vc), text_channel))
    assert text_channel.last == "Couldn't join **Locked**: Missing Permissions"


async def test_speaking_ring_fix_sends_silence_when_idle(guild, no_sleep):
    voice = FakeVoice(guild, None)
    await music._clear_speaking_ring(voice)
    assert voice.speaking == [discord.SpeakingState.none]
    assert voice.packets == [discord.opus.OPUS_SILENCE] * 5


async def test_speaking_ring_fix_leaves_a_playing_song_alone(guild, no_sleep):
    voice = FakeVoice(guild, None, playing=True)
    await music._clear_speaking_ring(voice)
    assert voice.packets == [] and voice.speaking == []


async def test_speaking_ring_fix_never_raises(guild, no_sleep):
    voice = FakeVoice(guild, None)

    def broken(*args, **kwargs):
        raise OSError("socket closed")

    voice.send_audio_packet = broken
    await music._clear_speaking_ring(voice)  # must not raise


# --- starting a track --------------------------------------------------------


class SilentSource(discord.AudioSource):
    # Stands in for FFmpegPCMAudio: raw PCM frames of a constant loud value.
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def read(self):
        return (1000).to_bytes(2, "little", signed=True) * (discord.opus.Encoder.FRAME_SIZE // 2)

    def is_opus(self):
        return False


async def test_start_track_applies_volume_and_announces_silently(connected, monkeypatch):
    voice, message = connected
    music.volumes[message.guild.id] = 25

    async def fake_resolve(query):
        return {"url": "https://stream.example/audio", "title": "Real Title",
                "webpage_url": "https://www.youtube.com/watch?v=abc", "duration": 300}

    monkeypatch.setattr(music, "resolve_stream", fake_resolve)
    monkeypatch.setattr(music.discord, "FFmpegPCMAudio", SilentSource)
    song = {"query": "some song", "title": "some song", "start_seconds": 0,
            "channel": message.channel}

    assert await music._start_track(message.guild, song, 90)

    assert isinstance(voice.source, discord.PCMVolumeTransformer)
    assert voice.source.volume == 0.25
    assert voice.source.original.kwargs["before_options"].startswith("-ss 90 ")
    assert message.channel.last == "▶️ Now playing: **Real Title** (from 1:30)"
    assert message.channel.silent[-1] is True
    current = music.get_player(message.guild.id).current
    assert current["title"] == "Real Title" and current["duration"] == 300
    # The search is pinned to the exact video, so a seek replays the same one.
    assert current["query"] == current["webpage_url"] == "https://www.youtube.com/watch?v=abc"


async def test_start_track_reports_a_load_failure(connected, monkeypatch):
    voice, message = connected

    async def failing(query):
        raise RuntimeError("Video unavailable")

    monkeypatch.setattr(music, "resolve_stream", failing)
    song = {"query": "x", "title": "x", "start_seconds": 0, "channel": message.channel}
    assert not await music._start_track(message.guild, song, 0)
    assert "couldn't load it: Video unavailable" in message.channel.last


# --- volume ------------------------------------------------------------------


async def test_volume_defaults_to_config(guild):
    assert music.get_volume(guild.id) == 100


@pytest.mark.parametrize("arg, reply", [
    ("", "🔊 Volume is **100%**."),
    ("10", "🔉 Volume set to **10%**."),
    ("10%", "🔉 Volume set to **10%**."),
    ("0", "🔇 Volume set to **0%**."),
    ("80", "🔊 Volume set to **80%**."),
    ("150", "Give me a volume from 0 to 100, e.g. `syntia volume 10`."),
    ("loud", "Give me a volume from 0 to 100, e.g. `syntia volume 10`."),
])
async def test_set_volume_replies(guild, arg, reply):
    channel = FakeTextChannel()
    await music.set_volume(make_message(guild, channel=channel), arg)
    assert channel.last == reply


async def test_volume_changes_the_playing_song_live(connected):
    voice, message = connected
    voice.source = discord.PCMVolumeTransformer(SilentSource(), volume=1.0)
    await music.set_volume(message, "10")
    assert voice.source.volume == pytest.approx(0.10)
    assert music.get_volume(message.guild.id) == 10


async def test_volume_survives_leaving(connected):
    voice, message = connected
    await music.set_volume(message, "30")
    await music.leave_voice(message)
    assert music.get_volume(message.guild.id) == 30


@pytest.mark.ffmpeg
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not on PATH")
def test_volume_really_scales_ffmpeg_audio():
    # End to end on a real FFmpeg stream: a generated tone at 100%, then 10%.
    import array

    source = discord.PCMVolumeTransformer(
        discord.FFmpegPCMAudio("sine=frequency=440:duration=2",
                               before_options="-f lavfi", options="-vn"),
        volume=1.0,
    )
    try:
        def peak():
            return max(abs(x) for x in array.array("h", source.read()))

        for _ in range(5):
            loud = peak()
        source.volume = 0.1
        for _ in range(5):
            quiet = peak()
    finally:
        source.cleanup()
    assert quiet / loud == pytest.approx(0.1, abs=0.01)


# --- the track-started hook (for spoken requests) ------------------------------------


async def test_track_started_hook_runs_when_the_channel_has_one(connected, monkeypatch):
    voice, message = connected
    started = []

    class HookedChannel(FakeTextChannel):
        async def on_track_started(self, entry):
            started.append(entry["title"])

    async def fake_resolve(query):
        return {"url": "u", "title": "Şımarık", "webpage_url": None, "duration": 200}

    monkeypatch.setattr(music, "resolve_stream", fake_resolve)
    monkeypatch.setattr(music.discord, "FFmpegPCMAudio", SilentSource)
    channel = HookedChannel()
    song = {"query": "tarkan", "title": "tarkan", "start_seconds": 0, "channel": channel}
    assert await music._start_track(message.guild, song, 0)
    assert started == ["Şımarık"] and channel.last.startswith("▶️ Now playing")


async def test_plain_channels_have_no_hook_and_nothing_breaks(connected, monkeypatch):
    voice, message = connected

    async def fake_resolve(query):
        return {"url": "u", "title": "Song", "webpage_url": None, "duration": None}

    monkeypatch.setattr(music, "resolve_stream", fake_resolve)
    monkeypatch.setattr(music.discord, "FFmpegPCMAudio", SilentSource)
    assert not hasattr(message.channel, "on_track_started")
    song = {"query": "x", "title": "x", "start_seconds": 0, "channel": message.channel}
    assert await music._start_track(message.guild, song, 0)
