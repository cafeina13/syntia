"""The AI layer: tool definitions, dispatch, the system prompt, and ask_ai."""

from types import SimpleNamespace

import pytest
from conftest import FakeGuild, FakeTextChannel, make_member, make_message

import ai
import config
import music


@pytest.fixture
def recorded(monkeypatch):
    # Swap every music function the AI can reach for a recorder.
    calls = []
    names = ["play_music", "add_music", "clear_queue", "stop_music", "leave_voice",
             "now_playing", "pause_music", "resume_music", "skip_song", "shuffle_queue",
             "play_previous", "seek", "set_volume",
             "join_voice"]
    for name in names:
        async def record(*args, _name=name):
            calls.append((_name, args[1:]))  # drop the message argument
        monkeypatch.setattr(music, name, record)
    return calls


def test_tool_names_are_unique():
    names = [spec["name"] for spec in ai.TOOL_SPECS]
    assert len(names) == len(set(names))


def test_gemini_and_ollama_get_the_same_tools():
    gemini = [d.name for d in ai.GEMINI_TOOLS[0].function_declarations]
    ollama = [t["function"]["name"] for t in ai.OLLAMA_TOOLS]
    assert gemini == ollama == [spec["name"] for spec in ai.TOOL_SPECS]


def test_required_arguments_exist():
    for spec in ai.TOOL_SPECS:
        for arg in spec["required"]:
            assert arg in spec["properties"], f"{spec['name']} requires unknown {arg}"


@pytest.mark.parametrize("tool, args, expected", [
    ("play_music", {"query": "lofi", "start_seconds": 60}, ("play_music", ("lofi", 60))),
    ("add_to_queue", {"query": "jazz"}, ("add_music", ("jazz", 0))),
    ("clear_queue", {}, ("clear_queue", ())),
    ("stop_music", {}, ("stop_music", ())),
    ("leave_voice", {}, ("leave_voice", ())),
    ("now_playing", {}, ("now_playing", ())),
    ("pause_music", {}, ("pause_music", ())),
    ("resume_music", {}, ("resume_music", ())),
    ("skip_song", {}, ("skip_song", ())),
    ("shuffle_queue", {}, ("shuffle_queue", ())),
    ("play_previous", {}, ("play_previous", ())),
    ("seek", {"seconds": -30}, ("seek", (-30, None))),
    ("seek", {"to_seconds": 325}, ("seek", (0, 325))),
    ("set_volume", {"level": 10}, ("set_volume", ("10",))),
    ("set_volume", {"level": 400}, ("set_volume", ("100",))),  # clamped
    ("set_volume", {"level": -5}, ("set_volume", ("0",))),
    ("join_voice", {}, ("join_voice", ())),
])
async def test_run_tool_calls_the_right_function(recorded, tool, args, expected):
    await ai.run_tool(make_message(FakeGuild()), tool, args)
    assert recorded == [expected]


async def test_every_tool_is_handled(recorded, monkeypatch):
    # A tool described to the AI but missing from run_tool would silently do
    # nothing. show_help sends a message instead of calling music, so count it.
    channel = FakeTextChannel()
    for spec in ai.TOOL_SPECS:
        before = len(recorded) + len(channel.sent)
        args = {name: 1 if kind == "integer" else "x"
                for name, (kind, _) in spec["properties"].items()}
        await ai.run_tool(make_message(FakeGuild(), channel=channel), spec["name"], args)
        assert len(recorded) + len(channel.sent) > before, f"{spec['name']} did nothing"


async def test_show_help_sends_the_help(recorded):
    channel = FakeTextChannel()
    await ai.run_tool(make_message(FakeGuild(), channel=channel), "show_help", {})
    assert channel.last.startswith("**Syntia commands**")


def test_system_prompt_has_commands_and_volume():
    guild = FakeGuild()
    music.volumes[guild.id] = 35
    prompt = ai.build_system_instruction(make_message(guild, make_member(name="Ali")))
    assert "Ali" in prompt and "Test Server" in prompt
    assert "Current music volume: 35%" in prompt
    assert "- syntia seek <time>" in prompt
    assert "### Verified Owner" not in prompt


def test_owner_block_only_for_the_real_owner_id(monkeypatch):
    monkeypatch.setattr(config, "OWNER_ID", 555)
    guild = FakeGuild()
    owner = ai.build_system_instruction(make_message(guild, make_member(555)))
    impostor = ai.build_system_instruction(make_message(guild, make_member(556, name="555")))
    assert "### Verified Owner" in owner
    assert "### Verified Owner" not in impostor


@pytest.fixture
def ollama_backend(monkeypatch):
    # Route ask_ai to a scripted fake instead of a real model.
    monkeypatch.setattr(config, "AI_BACKEND", "ollama")
    reply = {}

    async def fake_generate(system, prompt):
        reply["prompt"] = prompt
        return reply["result"]

    monkeypatch.setattr(ai, "generate_ollama", fake_generate)
    return reply


async def test_ask_ai_runs_every_tool_in_order(recorded, ollama_backend):
    ollama_backend["result"] = {"type": "tools", "calls": [
        {"name": "shuffle_queue", "args": {}},
        {"name": "skip_song", "args": {}},
    ]}
    await ai.ask_ai(make_message(FakeGuild()), "shuffle then skip")
    assert [name for name, _ in recorded] == ["shuffle_queue", "skip_song"]


async def test_ask_ai_sends_text_replies(ollama_backend):
    ollama_backend["result"] = {"type": "text", "text": "x" * 2500}
    channel = FakeTextChannel()
    await ai.ask_ai(make_message(FakeGuild(), channel=channel), "hi")
    assert len(channel.last) == 2000  # trimmed to Discord's limit


async def test_ask_ai_reports_backend_errors(monkeypatch):
    monkeypatch.setattr(config, "AI_BACKEND", "ollama")

    async def broken(system, prompt):
        raise ConnectionError("ollama is not running")

    monkeypatch.setattr(ai, "generate_ollama", broken)
    channel = FakeTextChannel()
    await ai.ask_ai(make_message(FakeGuild(), channel=channel), "hi")
    assert channel.last == "AI error: ollama is not running"


async def test_ask_ai_without_any_backend(monkeypatch):
    monkeypatch.setattr(config, "AI_BACKEND", "gemini")
    monkeypatch.setattr(config, "gemini_client", None)
    channel = FakeTextChannel()
    await ai.ask_ai(make_message(FakeGuild(), channel=channel), "hi")
    assert channel.last.startswith("AI isn't set up")


def fake_gemini(monkeypatch, response):
    async def generate_content(**kwargs):
        return response

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    monkeypatch.setattr(config, "gemini_client", client)


def gemini_part(name=None, args=None):
    call = SimpleNamespace(name=name, args=args) if name else None
    return SimpleNamespace(function_call=call)


async def test_generate_gemini_keeps_every_tool_call_in_order(monkeypatch):
    response = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[
            gemini_part("shuffle_queue", {}), gemini_part(), gemini_part("seek", {"seconds": 30}),
        ]))],
        text=None,
    )
    fake_gemini(monkeypatch, response)
    result = await ai.generate_gemini("system", "shuffle then jump 30s")
    assert result == {"type": "tools", "calls": [
        {"name": "shuffle_queue", "args": {}},
        {"name": "seek", "args": {"seconds": 30}},
    ]}


async def test_generate_gemini_text_and_empty_replies(monkeypatch):
    fake_gemini(monkeypatch, SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[gemini_part()]))], text="hey"))
    assert await ai.generate_gemini("s", "hi") == {"type": "text", "text": "hey"}
    # A blocked/empty response has no candidates at all.
    fake_gemini(monkeypatch, SimpleNamespace(candidates=None, text=None))
    assert await ai.generate_gemini("s", "hi") == {"type": "text", "text": ""}


# --- voice requests (audio) and results --------------------------------------------


class CapturingGemini:
    # A fake Gemini client that records what it was sent and answers as scripted.
    def __init__(self, response=None, error=None):
        self.sent = []
        self.response, self.error = response, error
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

    async def _generate(self, **kwargs):
        self.sent.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def gemini_says_tool(name, args):
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(
        parts=[gemini_part(name, args)]))], text=None)


async def test_audio_goes_to_gemini_as_an_audio_part(recorded, monkeypatch):
    client = CapturingGemini(gemini_says_tool("play_music", {"query": "Tarkan Şımarık"}))
    monkeypatch.setattr(config, "gemini_client", client)
    result = await ai.ask_ai(make_message(FakeGuild()), "", audio=b"RIFF....WAVE")
    [request] = client.sent
    [part] = request["contents"]  # audio only: no text draft to anchor on
    assert part.inline_data.mime_type == "audio/wav" and part.inline_data.data == b"RIFF....WAVE"
    assert "### Voice Command" in request["config"].system_instruction
    assert result == {"type": "tools", "calls": [{"name": "play_music", "args": {"query": "Tarkan Şımarık"}}]}
    assert recorded == [("play_music", ("Tarkan Şımarık", 0))]


async def test_text_requests_get_no_voice_note(recorded, monkeypatch):
    client = CapturingGemini(gemini_says_tool("skip_song", {}))
    monkeypatch.setattr(config, "AI_BACKEND", "gemini")
    monkeypatch.setattr(config, "gemini_client", client)
    await ai.ask_ai(make_message(FakeGuild()), "şarkıyı geç")
    assert client.sent[0]["contents"] == "şarkıyı geç"
    assert "### Voice Command" not in client.sent[0]["config"].system_instruction


async def test_voice_never_falls_back_to_ollama(monkeypatch):
    called = []

    async def ollama(system, prompt):
        called.append(prompt)

    monkeypatch.setattr(config, "AI_BACKEND", "ollama")
    monkeypatch.setattr(config, "gemini_client", None)
    monkeypatch.setattr(ai, "generate_ollama", ollama)
    channel = FakeTextChannel()
    result = await ai.ask_ai(make_message(FakeGuild(), channel=channel), "", audio=b"wav")
    assert called == [] and result == {"type": "error", "reason": "not_set_up"}
    assert channel.last.startswith("Voice commands need Gemini")


async def test_voice_uses_gemini_even_when_text_uses_ollama(recorded, monkeypatch):
    client = CapturingGemini(gemini_says_tool("skip_song", {}))
    monkeypatch.setattr(config, "AI_BACKEND", "ollama")
    monkeypatch.setattr(config, "gemini_client", client)
    await ai.ask_ai(make_message(FakeGuild()), "", audio=b"wav")
    assert len(client.sent) == 1 and recorded == [("skip_song", ())]


@pytest.mark.parametrize("error", [
    RuntimeError("429 RESOURCE_EXHAUSTED. You exceeded your current quota"),
    RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand"),
])
async def test_busy_gemini_gets_a_friendly_message(monkeypatch, error):
    monkeypatch.setattr(config, "AI_BACKEND", "gemini")
    monkeypatch.setattr(config, "gemini_client", CapturingGemini(error=error))
    channel = FakeTextChannel()
    result = await ai.ask_ai(make_message(FakeGuild(), channel=channel), "hi")
    assert result == {"type": "error", "reason": "busy"}
    assert channel.last == "The AI is busy right now — try again in a minute."


async def test_other_errors_still_show_the_details(monkeypatch):
    monkeypatch.setattr(config, "AI_BACKEND", "gemini")
    monkeypatch.setattr(config, "gemini_client", CapturingGemini(error=ValueError("bad request")))
    channel = FakeTextChannel()
    result = await ai.ask_ai(make_message(FakeGuild(), channel=channel), "hi")
    assert result == {"type": "error", "reason": "failed"} and channel.last == "AI error: bad request"


async def test_text_reply_is_returned_too(ollama_backend):
    ollama_backend["result"] = {"type": "text", "text": "Merhaba!"}
    channel = FakeTextChannel()
    result = await ai.ask_ai(make_message(FakeGuild(), channel=channel), "selam")
    assert result == {"type": "text", "text": "Merhaba!"} and channel.last == "Merhaba!"
