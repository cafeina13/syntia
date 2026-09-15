"""
Shared test setup: small fake stand-ins for Discord objects, so the bot's logic
can be tested without a token, a server, or a real voice connection.

Only the attributes and methods the bot actually touches are faked. If a test
fails with AttributeError on a Fake*, the code started using something new —
add it here.
"""

import os
from types import SimpleNamespace

# Pin the settings that config.py reads from .env, BEFORE anything imports it,
# so tests don't depend on whatever is in your real .env. (load_dotenv never
# overrides variables that are already set.)
os.environ["IDLE_TIMEOUT_MINUTES"] = "5"
os.environ["DEFAULT_VOLUME"] = "100"
os.environ["OWNER_ID"] = "0"

import pytest  # noqa: E402

import ai  # noqa: E402
import config  # noqa: E402
import music  # noqa: E402


class NullTyping:
    # Stands in for `async with channel.typing():`.
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSentMessage:
    # What send() returns in discord.py: a message the bot can edit or delete.
    def __init__(self, channel, content):
        self.channel, self.content, self.deleted = channel, content, False

    async def edit(self, *, content=None, **kwargs):
        self.content = content
        self.channel.edits.append(content)

    async def delete(self):
        self.deleted = True
        self.channel.deleted.append(self.content)


class FakeTextChannel:
    def __init__(self, name="general"):
        self.name = name
        self.sent = []  # every message the bot sent here, in order
        self.silent = []  # the silent= flag of each of those messages
        self.edits = []  # new content of every edited message, in order
        self.deleted = []  # content of every message the bot deleted
        self.messages = []  # the FakeSentMessage objects, in order

    async def send(self, content, *, silent=False, **kwargs):
        self.sent.append(content)
        self.silent.append(silent)
        message = FakeSentMessage(self, content)
        self.messages.append(message)
        return message

    def typing(self):
        return NullTyping()

    @property
    def last(self):
        return self.sent[-1] if self.sent else None


class FakeVoice:
    # Stands in for discord.VoiceClient.
    def __init__(self, guild, channel, playing=False):
        self.guild = guild
        self.channel = channel
        self.playing = playing
        self.paused = False
        self.connected = True
        self.source = None
        self.after = None
        self.stop_calls = 0
        self.packets = []
        self.speaking = []
        self.ws = SimpleNamespace(speak=self._speak)

    async def _speak(self, state):
        self.speaking.append(state)

    def is_playing(self):
        return self.playing

    def is_paused(self):
        return self.paused

    def is_connected(self):
        return self.connected

    def play(self, source, *, after=None):
        self.source, self.after, self.playing = source, after, True

    def stop(self):
        self.stop_calls += 1
        self.playing = self.paused = False

    def pause(self):
        # Like discord.py: is_playing() is False while paused.
        self.playing, self.paused = False, True

    def resume(self):
        self.playing, self.paused = True, False

    def send_audio_packet(self, data, *, encode=True):
        self.packets.append(data)

    async def move_to(self, channel):
        self.channel = channel

    async def disconnect(self):
        self.connected = self.playing = False
        self.guild.voice_client = None


class FakeVoiceChannel:
    _next_id = 1000

    def __init__(self, guild, name="Music", members=None, fail_connect=None):
        FakeVoiceChannel._next_id += 1
        self.id = FakeVoiceChannel._next_id
        self.guild = guild
        self.name = name
        self.members = members if members is not None else []
        self.fail_connect = fail_connect  # an exception to raise from connect()

    async def connect(self, *, cls=None):
        if self.fail_connect:
            raise self.fail_connect
        self.connected_with = cls  # which VoiceClient class music asked for
        voice = FakeVoice(self.guild, self)
        self.guild.voice_client = voice
        return voice


class FakeGuild:
    def __init__(self, guild_id=1, name="Test Server"):
        self.id = guild_id
        self.name = name
        self.voice_client = None


def make_member(member_id=42, *, bot=False, manage_guild=False, voice_channel=None,
                name="Tester"):
    return SimpleNamespace(
        id=member_id,
        bot=bot,
        display_name=name,
        guild_permissions=SimpleNamespace(manage_guild=manage_guild),
        voice=SimpleNamespace(channel=voice_channel) if voice_channel else None,
    )


def make_message(guild, author=None, channel=None):
    return SimpleNamespace(
        guild=guild,
        author=author or make_member(),
        channel=channel or FakeTextChannel(),
    )


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    # music.py keeps per-server state in module-level dicts; wipe it between
    # tests, and pin the config values tests rely on.
    music.players.clear()
    music.timeout_settings.clear()
    music.idle_since.clear()
    music.volumes.clear()
    ai._resting_until.clear()  # which Gemini models are sitting out after "busy"
    monkeypatch.setattr(ai, "assistant", None)  # no voice assistant unless a test adds one
    monkeypatch.setattr(config, "IDLE_TIMEOUT_MINUTES", 5)
    monkeypatch.setattr(config, "DEFAULT_VOLUME", 100)
    monkeypatch.setattr(config, "OWNER_ID", 0)
    monkeypatch.setattr(music, "client", SimpleNamespace(voice_clients=[], loop=None))
    yield


@pytest.fixture
def guild():
    return FakeGuild()


@pytest.fixture
def text_channel():
    return FakeTextChannel()


@pytest.fixture
def connected(guild, text_channel):
    # A bot already in a voice channel with one human listener. Returns
    # (voice, message) where the message author is that listener.
    listener = make_member(42)
    channel = FakeVoiceChannel(guild, members=[make_member(99, bot=True), listener])
    listener.voice = SimpleNamespace(channel=channel)
    voice = FakeVoice(guild, channel)
    guild.voice_client = voice
    music.client.voice_clients.append(voice)
    return voice, make_message(guild, listener, text_channel)
