"""`syntia assistant on|off` and one voice command end to end (assistant/session.py).

Models, the receiver and Gemini are fakes; ai.ask_ai, the music tools and the
reply hooks are the real code.
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import FakeTextChannel, FakeVoiceChannel, make_member, make_message

import ai
import config
import music
from assistant import phrases
from assistant import session as ss
from assistant.listener import CHIME, UTTERANCE

USER, STRANGER = 42, 7


class FakeReceiver:
    instances = []

    def __init__(self, voice, on_audio, *, users):
        self.voice, self.on_audio, self.users = voice, on_audio, users
        self.started = self.stopped = False
        FakeReceiver.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class FakeTranscriber:
    def __init__(self):
        self.next_text = "şarkı çal"

    def transcribe(self, audio):
        return SimpleNamespace(text=self.next_text, rejected=None, raw=self.next_text)


class FakeSpeaker:
    def __init__(self):
        self.lines, self.pcm = [], []

    async def say(self, guild, text):
        self.lines.append(text)

    async def play_pcm(self, guild, pcm):
        self.pcm.append(pcm)


class FakeListener:
    def __init__(self, user_id, on_event):
        self.user_id, self.on_event, self.fed, self.ticks = user_id, on_event, 0, 0

    def feed(self, pcm):
        self.fed += 1

    def tick(self):
        self.ticks += 1


@pytest.fixture
def rig(guild, monkeypatch):
    FakeReceiver.instances.clear()
    loads = []
    models = ss.Models(FakeTranscriber(), FakeSpeaker(), lambda uid, cb: FakeListener(uid, cb), b"chime")

    def loader(device, piper):
        loads.append((device, piper))
        return models, []

    monkeypatch.setattr(ss, "missing_packages", lambda: [])
    monkeypatch.setattr(music, "_run_in_background", lambda coro: coro.close())
    manager = ss.AssistantManager(permitted={USER}, owner_id=0, stt_device="cuda", piper_model="voice.onnx",
                                  loader=loader, make_receiver=FakeReceiver)
    vc = FakeVoiceChannel(guild, name="Sohbet")
    member = make_member(USER, voice_channel=vc, name="cafeína")
    guild.get_member = lambda user_id: member if user_id == USER else None
    text = FakeTextChannel()
    return SimpleNamespace(manager=manager, models=models, loads=loads, guild=guild, member=member,
                           text=text, message=make_message(guild, member, text))


async def settle(session):
    # Let the fire-and-forget speech tasks run.
    for _ in range(5):
        await asyncio.sleep(0)
    if session._tasks:
        await asyncio.gather(*list(session._tasks), return_exceptions=True)


# --- on / off ------------------------------------------------------------------------


async def test_status_and_permissions(rig, guild):
    await rig.manager.command(rig.message, "")
    assert "is **off**" in rig.text.last
    stranger = make_message(guild, make_member(STRANGER, voice_channel=FakeVoiceChannel(guild)), rig.text)
    await rig.manager.command(stranger, "on")
    assert rig.text.last.startswith("Only people listed in VOICE_USER_IDS")
    assert rig.loads == [] and not rig.manager.sessions


def test_owner_is_always_allowed():
    manager = ss.AssistantManager(permitted={1}, owner_id=555, loader=lambda *a: None, make_receiver=FakeReceiver)
    assert manager.allowed(555) and manager.allowed(1) and not manager.allowed(2)


async def test_on_loads_models_once_joins_and_listens(rig, guild):
    await rig.manager.command(rig.message, "on")
    session = rig.manager.sessions[guild.id]
    try:
        assert rig.loads == [("cuda", "voice.onnx")]
        assert guild.voice_client is not None  # joined the speaker's channel
        [receiver] = FakeReceiver.instances
        assert receiver.started and receiver.users == {USER}
        assert "Voice assistant **on** in **Sohbet**" in rig.text.last
        await rig.manager.command(rig.message, "on")
        assert rig.text.last.startswith("🎙️ Already listening") and len(rig.loads) == 1
    finally:
        session.stop()


async def test_off_stops_listening_and_frees_the_models(rig, guild):
    await rig.manager.command(rig.message, "on")
    receiver = FakeReceiver.instances[0]
    await rig.manager.command(rig.message, "off")
    assert receiver.stopped and not rig.manager.sessions
    assert rig.manager.models is None  # Whisper & co. released: VRAM back
    assert rig.text.last == "🎙️ Voice assistant off."
    await rig.manager.command(rig.message, "off")
    assert rig.text.last == "🎙️ The voice assistant is already off."


async def test_install_makes_music_connect_as_a_listening_client(monkeypatch):
    monkeypatch.setattr(music, "VOICE_CLIENT_CLASS", music.discord.VoiceClient)
    ss.AssistantManager.install()
    assert music.VOICE_CLIENT_CLASS is ss.ListeningVoiceClient


# --- the session's life ---------------------------------------------------------------


async def test_audio_creates_a_listener_per_speaker_and_the_chime_plays(rig, guild):
    await rig.manager.command(rig.message, "on")
    session = rig.manager.sessions[guild.id]
    try:
        receiver = FakeReceiver.instances[0]
        receiver.on_audio(USER, b"pcm", 960)
        receiver.on_audio(USER, b"pcm", 1920)
        assert session.listeners[USER].fed == 2
        session._on_event(CHIME, session.listeners[USER], {})
        await settle(session)
        assert rig.models.speaker.pcm == [b"chime"]
    finally:
        session.stop()


async def test_leaving_voice_turns_the_assistant_off(rig, guild):
    await rig.manager.command(rig.message, "on")
    session = rig.manager.sessions[guild.id]
    await guild.voice_client.disconnect()  # e.g. idle timeout, `syntia leave`, kicked
    await session.tick()
    assert not rig.manager.sessions and rig.manager.models is None
    assert rig.text.last == "🎙️ Voice assistant off (left the voice channel)."


async def test_a_reconnect_reattaches_the_receiver(rig, guild):
    await rig.manager.command(rig.message, "on")
    session = rig.manager.sessions[guild.id]
    try:
        old_voice = guild.voice_client
        new_voice = type(old_voice)(guild, old_voice.channel)
        guild.voice_client = new_voice
        await session.tick()
        assert FakeReceiver.instances[0].stopped
        assert FakeReceiver.instances[1].voice is new_voice and FakeReceiver.instances[1].started
    finally:
        session.stop()


# --- one voice command --------------------------------------------------------------


@pytest.fixture
async def listening(rig, guild, monkeypatch):
    await rig.manager.command(rig.message, "on")
    session = rig.manager.sessions[guild.id]
    for log in (rig.text.sent, rig.text.silent, rig.text.messages):
        log.clear()  # forget the "assistant on" messages
    sent = {}

    def gemini(response):
        async def generate_content(**kwargs):
            sent.update(kwargs)
            return response
        monkeypatch.setattr(config, "gemini_client",
                            SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))))

    rig.session, rig.sent, rig.gemini = session, sent, gemini
    yield rig
    session.stop()


def tool(name, args):
    call = SimpleNamespace(name=name, args=args)
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(
        parts=[SimpleNamespace(function_call=call)]))], text=None)


AUDIO = np.zeros(16000, dtype=np.int16)


async def test_nothing_understood_is_said_and_no_ai_request_is_spent(listening, monkeypatch):
    listening.models.transcriber.next_text = ""
    called = []
    monkeypatch.setattr(ai, "ask_ai", lambda *a, **k: called.append(1))
    await listening.session.handle_command(USER, AUDIO)
    await settle(listening.session)
    assert called == [] and listening.models.speaker.lines == [phrases.NOT_UNDERSTOOD]
    assert listening.text.sent == []


async def test_play_command_acknowledges_then_says_the_title(listening, guild, monkeypatch):
    listening.gemini(tool("play_music", {"query": "Tarkan Şımarık"}))

    async def fake_start(g, entry, offset, announce=True):
        entry["title"] = "Tarkan - Şımarık (Official Video)"
        await entry["channel"].send("▶️ Now playing: **Tarkan - Şımarık (Official Video)**", silent=True)
        await entry["channel"].on_track_started(entry)
        return True

    monkeypatch.setattr(music, "_start_track", fake_start)
    result = await listening.session.handle_command(USER, AUDIO)
    await settle(listening.session)

    assert result["type"] == "tools"
    assert listening.text.sent[0] == "-# 🎙️ cafeína (voice)" and listening.text.silent[0] is True
    [part] = listening.sent["contents"]
    assert part.inline_data.data[:4] == b"RIFF"  # Gemini got the recording as a WAV
    assert listening.models.speaker.lines == [phrases.ACKNOWLEDGED, "Tarkan, Şımarık çalıyorum."]


async def test_quick_actions_say_nothing(listening, guild):
    music.volumes[guild.id] = 50
    listening.gemini(tool("set_volume", {"level": 35}))
    await listening.session.handle_command(USER, AUDIO)
    await settle(listening.session)
    assert music.get_volume(guild.id) == 35 and listening.models.speaker.lines == []


async def test_a_chat_answer_is_spoken(listening):
    listening.gemini(SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(
        parts=[SimpleNamespace(function_call=None)]))], text="İyiyim, sen nasılsın?"))
    await listening.session.handle_command(USER, AUDIO)
    await settle(listening.session)
    assert listening.models.speaker.lines == ["İyiyim, sen nasılsın?"]
    assert listening.text.last == "İyiyim, sen nasılsın?"


async def test_busy_ai_is_said_out_loud(listening, monkeypatch):
    async def busy(**kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(config, "gemini_client", SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=busy))))
    result = await listening.session.handle_command(USER, AUDIO)
    await settle(listening.session)
    assert result == {"type": "error", "reason": "busy"}
    assert listening.models.speaker.lines == [phrases.BUSY]


async def test_utterance_event_runs_a_command(listening, monkeypatch):
    handled = []

    async def fake_handle(user_id, audio):
        handled.append(user_id)

    monkeypatch.setattr(listening.session, "handle_command", fake_handle)
    listening.session._on_event(UTTERANCE, SimpleNamespace(user_id=USER), {"audio": AUDIO})
    await settle(listening.session)
    assert handled == [USER]


def test_wav_bytes_is_a_real_wav():
    import io
    import wave
    data = ss.wav_bytes(np.arange(160, dtype=np.int16))
    with wave.open(io.BytesIO(data)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getnframes()) == (16000, 1, 160)


def test_ids_from_env_text():
    assert ss.parse_ids("123, 456,abc,") == {123, 456}
    assert ss.parse_ids(None) == set()


async def test_saying_assistant_off_really_turns_it_off(listening, guild):
    # Regression (live): "assistant off" played a video instead.
    listening.gemini(tool("turn_off_assistant", {}))
    receiver = FakeReceiver.instances[0]
    await listening.session.handle_command(USER, AUDIO)
    assert listening.models.speaker.lines == [phrases.ASSISTANT_OFF]  # said goodbye first
    assert not listening.manager.sessions and listening.manager.models is None and receiver.stopped
    assert listening.text.last == "🎙️ Voice assistant off."
    assert guild.voice_client is not None  # still in the channel, music untouched
