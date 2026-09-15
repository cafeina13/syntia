"""Syntia speaking in voice (assistant/speaker.py): alone, over music, and when not to."""

import pytest

# These tests need the optional voice assistant packages; without them, skip
# instead of crashing the whole run.
pytest.importorskip("numpy", reason="voice assistant packages not installed (pip install -r assistant/requirements.txt)")
pytest.importorskip("scipy", reason="voice assistant packages not installed (pip install -r assistant/requirements.txt)")


import asyncio

import discord
import numpy as np
from conftest import FakeVoice

import music
from assistant import speaker as sp


def frame(value: int) -> bytes:
    return np.full(sp.FRAME_BYTES // 2, value, dtype=np.int16).tobytes()


class MusicSource(discord.AudioSource):
    def __init__(self, frames=10, value=1000):
        self.left, self.value, self.cleaned = frames, value, False

    def read(self):
        if self.left <= 0:
            return b""
        self.left -= 1
        return frame(self.value)

    def is_opus(self):
        return False

    def cleanup(self):
        self.cleaned = True


def samples(data: bytes):
    return np.frombuffer(data, dtype=np.int16)


# --- text --------------------------------------------------------------------------


@pytest.mark.parametrize("raw, spoken", [
    ("▶️ Now playing: **Şımarık**", "Now playing: Şımarık"),
    ("-# 🎙️ cafeína (voice)", "cafeína"),
    ("Tamam! Açıyorum. Başka bir şey? Bir de şu var.", "Tamam! Açıyorum."),
    ("Bak: https://youtu.be/abc", "Bak:"),
])
def test_clean_for_speech(raw, spoken):
    assert sp.clean_for_speech(raw) == spoken


def test_clean_title_drops_video_tags():
    assert sp.clean_title("Tarkan - Şımarık (Official Video) [HD]") == "Tarkan, Şımarık"


def test_long_text_is_cut_at_a_word():
    spoken = sp.clean_for_speech("kelime " * 100)
    assert len(spoken) <= sp.MAX_SPEECH_CHARS + 1 and spoken.endswith("…") and "kelim…" not in spoken


def test_piper_audio_becomes_discord_audio():
    one_second_22k = np.zeros(22050, dtype=np.int16).tobytes()
    pcm = sp.to_discord_pcm(one_second_22k, 22050)
    assert len(pcm) == 48000 * 4  # 48 kHz, stereo, 16-bit


# --- sources -------------------------------------------------------------------------


def test_speech_source_pads_the_last_frame_then_ends():
    source = sp.SpeechSource(b"\x01\x00" * 1000)  # less than one frame
    assert len(source.read()) == sp.FRAME_BYTES and source.read() == b""
    assert source.is_speech and not source.is_opus()


def test_mixer_ducks_music_under_speech_then_plays_music_alone():
    music_source = MusicSource(frames=5, value=1000)
    mixer = sp.DuckingMixer(music_source, frame(500) * 2)
    first, second, third = mixer.read(), mixer.read(), mixer.read()
    assert samples(first)[0] == 1000 * sp.DUCK + 500  # 250 + 500
    assert samples(second)[0] == 750
    assert samples(third)[0] == 1000 and not mixer.speaking  # speech over, full volume
    assert len(first) == sp.FRAME_BYTES


def test_mixer_ends_when_the_song_ends_so_the_next_track_starts():
    mixer = sp.DuckingMixer(MusicSource(frames=1), frame(500) * 50)
    assert mixer.read() and mixer.read() == b""  # song over, even mid-sentence
    mixer.cleanup()
    assert mixer.original.cleaned


def test_a_released_mixer_never_kills_the_song_when_collected():
    # Regression (live): after Syntia spoke over "Mayın Tarlası" the song stopped.
    # The mixer was garbage-collected, discord.py's AudioSource.__del__ called
    # cleanup(), and that killed the song's FFmpeg process.
    import gc

    song = MusicSource()
    mixer = sp.DuckingMixer(song, frame(500))
    assert mixer.release() is song
    del mixer
    gc.collect()
    assert not song.cleaned


def test_an_active_mixer_still_cleans_up_a_song_that_ended():
    song = MusicSource(frames=0)
    mixer = sp.DuckingMixer(song, frame(500))
    assert mixer.read() == b""  # song over while Syntia was talking
    mixer.cleanup()  # what the player does when a source finishes
    assert song.cleaned


def test_mixer_never_clips_into_noise():
    mixer = sp.DuckingMixer(MusicSource(frames=1, value=32000), frame(30000))
    assert samples(mixer.read()).max() == 32767


# --- the speaker ---------------------------------------------------------------------


class FakeTTS:
    sample_rate = 48000

    def synthesize(self, text):
        return np.full(4800, 700, dtype=np.int16).tobytes()  # 0.1 s


@pytest.fixture
def fast_sleep(monkeypatch):
    real_sleep = asyncio.sleep

    async def quick(seconds):
        await real_sleep(0)

    monkeypatch.setattr(sp.asyncio, "sleep", quick)


async def test_speaks_on_its_own_when_nothing_plays(guild, fast_sleep):
    voice = FakeVoice(guild, None)
    guild.voice_client = voice
    played = []

    def play(source, *, after=None):
        played.append(source)
        after(None)  # finished at once

    voice.play = play
    await sp.Speaker(FakeTTS()).say(guild, "Tamam, hallediyorum.")
    [source] = played
    assert isinstance(source, sp.SpeechSource) and source.is_speech


async def test_speaks_over_music_by_ducking_then_gives_the_song_back(guild, fast_sleep):
    voice = FakeVoice(guild, None, playing=True)
    song = MusicSource()
    voice.source = song
    guild.voice_client = voice
    seen = []
    original_setattr = FakeVoice.__setattr__

    await_speaker = sp.Speaker(FakeTTS())
    task = asyncio.create_task(await_speaker.say(guild, "Şımarık çalıyorum."))
    for _ in range(50):
        await asyncio.sleep(0)
        if isinstance(voice.source, sp.DuckingMixer):
            seen.append(voice.source)
            break
    await task
    assert seen and seen[0].original is song
    assert voice.source is song  # handed back once the sentence is over
    assert voice.stop_calls == 0  # the song never stopped
    del seen[:]
    import gc
    gc.collect()
    assert not song.cleaned  # and nothing killed it afterwards


async def test_stays_quiet_while_music_is_paused(guild, fast_sleep):
    voice = FakeVoice(guild, None)
    voice.paused = True
    song = MusicSource()
    voice.source = song
    guild.voice_client = voice
    await sp.Speaker(FakeTTS()).say(guild, "Merhaba")
    assert voice.source is song and voice.paused  # untouched, still paused


async def test_silent_assistant_without_a_voice_model(guild):
    voice = FakeVoice(guild, None)
    guild.voice_client = voice
    await sp.Speaker(None).say(guild, "Merhaba")
    assert voice.source is None


async def test_a_synthesis_failure_never_raises(guild):
    class BrokenTTS(FakeTTS):
        def synthesize(self, text):
            raise RuntimeError("onnx exploded")

    guild.voice_client = FakeVoice(guild, None)
    await sp.Speaker(BrokenTTS()).say(guild, "Merhaba")  # must not raise


async def test_a_song_starting_cuts_speech_instead_of_crashing(connected, monkeypatch):
    voice, message = connected
    voice.playing = True
    voice.source = sp.SpeechSource(frame(1) * 10)  # Syntia is mid-sentence

    async def fake_resolve(query):
        return {"url": "u", "title": "Şımarık", "webpage_url": None, "duration": 200}

    class Silent(discord.AudioSource):
        def __init__(self, *a, **k):
            pass

        def read(self):
            return b""

        def is_opus(self):
            return False

    monkeypatch.setattr(music, "resolve_stream", fake_resolve)
    monkeypatch.setattr(music.discord, "FFmpegPCMAudio", Silent)
    song = {"query": "x", "title": "x", "start_seconds": 0, "channel": message.channel}
    assert await music._start_track(message.guild, song, 0)
    assert voice.stop_calls == 1 and isinstance(voice.source, discord.PCMVolumeTransformer)


async def test_music_connects_with_the_configured_voice_client(guild, text_channel, monkeypatch):
    from conftest import FakeVoiceChannel, make_member, make_message

    class Special:
        pass

    monkeypatch.setattr(music, "VOICE_CLIENT_CLASS", Special)
    monkeypatch.setattr(music, "_run_in_background", lambda coro: coro.close())
    vc = FakeVoiceChannel(guild)
    await music.ensure_voice(make_message(guild, make_member(voice_channel=vc), text_channel))
    assert vc.connected_with is Special
