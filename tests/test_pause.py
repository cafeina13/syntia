"""Pause / resume, and how pausing interacts with position, seek, and timeout."""

import time

import discord
import pytest
from conftest import make_message

import music


class FakeSource(discord.AudioSource):
    def read(self):
        return b""

    def is_opus(self):
        return False


@pytest.fixture
def playing(connected):
    # A song that started 100 seconds ago and is still playing.
    voice, message = connected
    player = music.get_player(message.guild.id)
    player.current = {"query": "A", "title": "A", "start_seconds": 0, "channel": message.channel}
    player.started_at = time.monotonic() - 100
    voice.playing = True
    return voice, message, player


async def test_pause_then_resume(playing):
    voice, message, player = playing
    await music.pause_music(message)
    assert voice.is_paused() and message.channel.last.startswith("⏸️ Paused.")
    await music.resume_music(message)
    assert voice.is_playing() and message.channel.last == "▶️ Resumed."
    assert player.paused_at is None


async def test_double_pause_and_double_resume(playing):
    voice, message, player = playing
    await music.resume_music(message)
    assert message.channel.last == "It's already playing."
    await music.pause_music(message)
    await music.pause_music(message)
    assert message.channel.last.startswith("Already paused.")


async def test_nothing_to_pause_or_resume(connected):
    voice, message = connected
    await music.pause_music(message)
    assert message.channel.last == "Nothing is playing."
    await music.resume_music(message)
    assert message.channel.last == "Nothing is paused."


async def test_pause_without_a_voice_connection(guild, text_channel):
    message = make_message(guild, channel=text_channel)
    await music.pause_music(message)
    await music.resume_music(message)
    assert text_channel.sent == ["Nothing is playing.", "Nothing is paused."]


async def test_position_is_frozen_while_paused(playing):
    voice, message, player = playing
    await music.pause_music(message)
    player.paused_at -= 40  # pretend the pause began 40 seconds ago...
    # ...so 60 s were listened before it, and the paused 40 s don't count.
    assert player.position() == pytest.approx(60, abs=1)


async def test_resume_does_not_count_the_paused_time(playing):
    voice, message, player = playing
    await music.pause_music(message)
    player.paused_at -= 40  # paused 40 seconds ago, 60 s into the song
    await music.resume_music(message)
    assert player.position() == pytest.approx(60, abs=1)


async def test_seek_works_while_paused_and_resumes(playing):
    voice, message, player = playing
    await music.pause_music(message)
    player.paused_at -= 40
    await music.seek(message, 30)
    assert player.seek_target[1] == 90  # 60 listened + 30, not 100 + 30
    assert voice.stop_calls == 1
    assert message.channel.last == "⏩ Jumped to 1:30 (and resumed)."


async def test_a_new_track_starts_unpaused(playing, monkeypatch):
    voice, message, player = playing
    await music.pause_music(message)

    async def fake_resolve(query):
        return {"url": "https://stream.example/audio", "title": "Next",
                "webpage_url": None, "duration": None}

    monkeypatch.setattr(music, "resolve_stream", fake_resolve)
    monkeypatch.setattr(music.discord, "FFmpegPCMAudio", lambda *a, **k: FakeSource())
    song = {"query": "Next", "title": "Next", "start_seconds": 0, "channel": message.channel}
    assert await music._start_track(message.guild, song, 0)
    assert player.paused_at is None


async def test_skip_and_stop_work_while_paused(playing):
    voice, message, player = playing
    await music.pause_music(message)
    await music.skip_song(message)
    assert voice.stop_calls == 1 and message.channel.last == "⏭️ Skipped."
    voice.paused = True
    await music.stop_music(message)
    assert message.channel.last.startswith("⏹️ Stopped.")


async def test_paused_song_counts_as_idle(playing):
    voice, message, player = playing
    player.text_channel = message.channel
    await music.pause_music(message)

    await music.check_idle()
    assert message.guild.id in music.idle_since  # the clock started
    music.idle_since[message.guild.id] -= 6 * 60
    await music.check_idle()

    assert not voice.connected
    assert message.channel.last == "No music for 5 min, heading out."


async def test_resuming_stops_the_idle_clock(playing):
    voice, message, player = playing
    await music.pause_music(message)
    await music.check_idle()
    await music.resume_music(message)
    await music.check_idle()
    assert message.guild.id not in music.idle_since
