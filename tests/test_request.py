"""A spoken request (assistant/request.py) running through Syntia's real brain."""

from types import SimpleNamespace

import pytest
from conftest import FakeTextChannel, FakeVoiceChannel, make_member

import ai
import config
import music
from assistant.request import ReplyChannel, VoiceRequest


def gemini_tool(name, args):
    call = SimpleNamespace(name=name, args=args)
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(
        parts=[SimpleNamespace(function_call=call)]))], text=None)


@pytest.fixture
def gemini(monkeypatch):
    script = {}

    async def generate_content(**kwargs):
        script["sent"] = kwargs
        return script["response"]

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    monkeypatch.setattr(config, "gemini_client", client)
    return script


@pytest.fixture
def request_in_voice(guild):
    text = FakeTextChannel()
    vc = FakeVoiceChannel(guild, name="Sohbet")
    member = make_member(42, voice_channel=vc, name="cafeína")
    return VoiceRequest(guild, member, ReplyChannel(text)), text


async def test_reply_channel_forwards_to_the_text_channel(request_in_voice):
    req, text = request_in_voice
    await req.channel.send("merhaba", silent=True)
    assert text.sent == ["merhaba"] and text.silent == [True]
    assert req.channel.name == "general"  # anything else falls through to the real channel


async def test_first_track_hook_fires_once(request_in_voice):
    req, text = request_in_voice
    heard = []

    async def say_title(entry):
        heard.append(entry["title"])

    channel = ReplyChannel(text, on_first_track=say_title)
    await channel.on_track_started({"title": "Şımarık"})
    await channel.on_track_started({"title": "Kuzu Kuzu"})  # next song in a playlist: silent
    assert heard == ["Şımarık"]


async def test_hook_without_a_speaker_does_nothing(request_in_voice):
    req, text = request_in_voice
    await req.channel.on_track_started({"title": "x"})  # must not raise


async def test_spoken_volume_command_changes_the_volume(request_in_voice, gemini, guild):
    req, text = request_in_voice
    music.volumes[guild.id] = 50
    gemini["response"] = gemini_tool("set_volume", {"level": 35})
    result = await ai.ask_ai(req, "", audio=b"wav")
    assert result["type"] == "tools"
    assert music.get_volume(guild.id) == 35 and text.last == "🔉 Volume set to **35%**."
    assert "cafeína" in gemini["sent"]["config"].system_instruction  # personalised like text


async def test_spoken_play_command_joins_and_queues(request_in_voice, gemini, guild, monkeypatch):
    req, text = request_in_voice
    started = []

    async def fake_start(g, entry, offset, announce=True):
        started.append((entry["query"], entry["channel"]))
        return True

    monkeypatch.setattr(music, "_start_track", fake_start)
    monkeypatch.setattr(music, "_run_in_background", lambda coro: coro.close())
    gemini["response"] = gemini_tool("play_music", {"query": "Tarkan Şımarık"})
    await ai.ask_ai(req, "", audio=b"wav")
    assert guild.voice_client is not None  # joined the speaker's channel
    [(query, channel)] = started
    assert query == "Tarkan Şımarık"
    assert channel is req.channel  # replies (and the title hook) come back to this request
