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


# --- model fallback ------------------------------------------------------------------


class PerModelGemini:
    # Each model name answers from its own script: an exception to raise, or a response.
    def __init__(self, behaviour):
        self.behaviour, self.tried = behaviour, []
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

    async def _generate(self, model, **kwargs):
        self.tried.append(model)
        outcome = self.behaviour[model]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


QUOTA = RuntimeError("429 RESOURCE_EXHAUSTED. GenerateRequestsPerDayPerProjectPerModel-FreeTier")
OVERLOADED = RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand")


@pytest.fixture
def three_models(monkeypatch):
    monkeypatch.setattr(config, "AI_BACKEND", "gemini")
    monkeypatch.setattr(config, "GEMINI_MODELS", ["new-flash", "old-flash", "lite"])


async def test_falls_back_to_the_next_model_when_quota_is_used_up(recorded, three_models, monkeypatch):
    client = PerModelGemini({"new-flash": QUOTA, "old-flash": gemini_says_tool("skip_song", {}), "lite": QUOTA})
    monkeypatch.setattr(config, "gemini_client", client)
    result = await ai.ask_ai(make_message(FakeGuild()), "şarkıyı geç")
    assert client.tried == ["new-flash", "old-flash"]
    assert result["type"] == "tools" and recorded == [("skip_song", ())]
    assert ai.last_model_used == "old-flash"


async def test_a_resting_model_is_skipped_next_time(three_models, monkeypatch):
    client = PerModelGemini({"new-flash": QUOTA, "old-flash": gemini_says_tool("skip_song", {}), "lite": QUOTA})
    monkeypatch.setattr(config, "gemini_client", client)
    monkeypatch.setattr(music, "skip_song", _noop)
    await ai.ask_ai(make_message(FakeGuild()), "geç")
    client.tried.clear()
    await ai.ask_ai(make_message(FakeGuild()), "geç")
    assert client.tried == ["old-flash"]  # no wasted request on the model that's out of quota


async def test_rest_ends_and_the_preferred_model_is_used_again(three_models, monkeypatch):
    client = PerModelGemini({"new-flash": OVERLOADED, "old-flash": gemini_says_tool("skip_song", {}), "lite": QUOTA})
    monkeypatch.setattr(config, "gemini_client", client)
    monkeypatch.setattr(music, "skip_song", _noop)
    await ai.ask_ai(make_message(FakeGuild()), "geç")
    assert ai._resting_until["new-flash"] - ai._resting_until.get("old-flash", 0) > 0
    ai._resting_until["new-flash"] = 0  # pretend the rest is over
    client.behaviour["new-flash"] = gemini_says_tool("skip_song", {})
    client.tried.clear()
    await ai.ask_ai(make_message(FakeGuild()), "geç")
    assert client.tried == ["new-flash"]


async def test_all_models_busy_means_the_friendly_busy_message(three_models, monkeypatch):
    client = PerModelGemini({"new-flash": QUOTA, "old-flash": OVERLOADED, "lite": QUOTA})
    monkeypatch.setattr(config, "gemini_client", client)
    channel = FakeTextChannel()
    result = await ai.ask_ai(make_message(FakeGuild(), channel=channel), "hi")
    assert client.tried == ["new-flash", "old-flash", "lite"]
    assert result == {"type": "error", "reason": "busy"} and channel.last.startswith("The AI is busy")


async def test_when_every_model_is_resting_they_are_all_tried_anyway(three_models, monkeypatch):
    import time
    for model in config.GEMINI_MODELS:
        ai._resting_until[model] = time.monotonic() + 999
    client = PerModelGemini({"new-flash": gemini_says_tool("skip_song", {}), "old-flash": QUOTA, "lite": QUOTA})
    monkeypatch.setattr(config, "gemini_client", client)
    monkeypatch.setattr(music, "skip_song", _noop)
    result = await ai.ask_ai(make_message(FakeGuild()), "geç")
    assert result["type"] == "tools" and client.tried == ["new-flash"]


async def test_real_errors_do_not_fall_back(three_models, monkeypatch):
    client = PerModelGemini({"new-flash": ValueError("400 INVALID_ARGUMENT"), "old-flash": QUOTA, "lite": QUOTA})
    monkeypatch.setattr(config, "gemini_client", client)
    channel = FakeTextChannel()
    result = await ai.ask_ai(make_message(FakeGuild(), channel=channel), "hi")
    assert client.tried == ["new-flash"]  # a broken request would break on every model
    assert result == {"type": "error", "reason": "failed"}


async def _noop(*args, **kwargs):
    return None


# --- progress while a slow answer cooks ------------------------------------------------


class SlowGemini:
    # Per model: (seconds to think, response or exception).
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

    async def _generate(self, model, **kwargs):
        import asyncio
        delay, outcome = self.behaviour[model]
        await asyncio.sleep(delay)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def quick_steps(monkeypatch):
    # The real timeline is 3 s / 12 s / 30 s; tests use 50 / 150 / 300 ms.
    monkeypatch.setattr(ai, "PROGRESS_STEPS", [(0.05, "cooking"), (0.15, "still cooking"), (0.3, "almost")])
    monkeypatch.setattr(config, "AI_BACKEND", "gemini")
    monkeypatch.setattr(config, "GEMINI_MODELS", ["main", "backup"])


async def test_a_fast_answer_shows_no_status(quick_steps, monkeypatch):
    monkeypatch.setattr(config, "gemini_client", SlowGemini({"main": (0.0, gemini_text("selam!"))}))
    channel = FakeTextChannel()
    await ai.ask_ai(make_message(FakeGuild(), channel=channel), "selam")
    assert channel.sent == ["selam!"] and channel.deleted == []


async def test_a_slow_answer_shows_a_status_that_updates_then_disappears(quick_steps, monkeypatch):
    monkeypatch.setattr(config, "gemini_client", SlowGemini({"main": (0.2, gemini_text("tamam"))}))
    channel = FakeTextChannel()
    await ai.ask_ai(make_message(FakeGuild(), channel=channel), "bir şey")
    assert channel.sent == ["cooking", "tamam"]  # one status message, then the answer
    assert channel.edits == ["still cooking"]  # edited in place, not a new message
    assert channel.messages[0].deleted and channel.silent[0] is True  # removed, never pinged anyone


async def test_switching_models_is_mentioned_in_the_status(quick_steps, monkeypatch):
    monkeypatch.setattr(config, "gemini_client", SlowGemini({
        "main": (0.08, RuntimeError("503 UNAVAILABLE")), "backup": (0.05, gemini_text("geldim")),
    }))
    channel = FakeTextChannel()
    await ai.ask_ai(make_message(FakeGuild(), channel=channel), "hey")
    assert channel.sent[0] == "cooking"
    assert channel.edits == ["cooking" + ai.SWITCH_NOTE]
    assert channel.sent[-1] == "geldim" and channel.messages[0].deleted


async def test_status_is_cleaned_up_when_every_model_is_busy(quick_steps, monkeypatch):
    busy = RuntimeError("429 RESOURCE_EXHAUSTED")
    monkeypatch.setattr(config, "gemini_client", SlowGemini({"main": (0.08, busy), "backup": (0.02, busy)}))
    channel = FakeTextChannel()
    result = await ai.ask_ai(make_message(FakeGuild(), channel=channel), "hey")
    assert result["reason"] == "busy" and channel.messages[0].deleted
    assert channel.last.startswith("The AI is busy")


async def test_slow_answer_hook_fires_once_for_a_spoken_request(quick_steps, monkeypatch):
    from assistant.request import ReplyChannel

    spoken = []

    async def say(status):
        spoken.append(status)

    monkeypatch.setattr(config, "gemini_client", SlowGemini({"main": (0.2, gemini_text("tamam"))}))
    text = FakeTextChannel()
    reply = ReplyChannel(text, on_slow_answer=say)
    await ai.ask_ai(SimpleNamespace(guild=FakeGuild(), author=make_member(), channel=reply), "", audio=b"wav")
    assert spoken == ["cooking"]  # once, not again on every edit
    assert text.sent[-1] == "tamam"


def gemini_text(text):
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[gemini_part()]))], text=text)


async def test_text_reply_is_returned_too(ollama_backend):
    ollama_backend["result"] = {"type": "text", "text": "Merhaba!"}
    channel = FakeTextChannel()
    result = await ai.ask_ai(make_message(FakeGuild(), channel=channel), "selam")
    assert result == {"type": "text", "text": "Merhaba!"} and channel.last == "Merhaba!"


class FakeAssistant:
    def __init__(self):
        self.commands = []

    async def command(self, message, arg):
        self.commands.append(arg)


def gemini_says_tools(*names):
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(
        parts=[gemini_part(name, {}) for name in names]))], text=None)


async def test_join_and_start_the_assistant_in_one_sentence(recorded, monkeypatch):
    # Regression (live): "syntia odaya gel ve sesli asistanı başlat" joined but
    # couldn't start the assistant — there was no tool for it.
    assistant = FakeAssistant()
    monkeypatch.setattr(ai, "assistant", assistant)
    monkeypatch.setattr(config, "AI_BACKEND", "gemini")
    monkeypatch.setattr(config, "gemini_client",
                        CapturingGemini(gemini_says_tools("join_voice", "turn_on_assistant")))
    await ai.ask_ai(make_message(FakeGuild()), "odaya gel ve sesli asistanı başlat")
    assert recorded == [("join_voice", ())] and assistant.commands == ["on"]


async def test_typed_assistant_off_really_turns_it_off(monkeypatch):
    assistant = FakeAssistant()
    monkeypatch.setattr(ai, "assistant", assistant)
    await ai.run_tool(make_message(FakeGuild()), "turn_off_assistant", {})
    assert assistant.commands == ["off"]


@pytest.mark.parametrize("tool_name", ["turn_on_assistant", "turn_off_assistant"])
async def test_assistant_tools_without_the_assistant(tool_name):
    channel = FakeTextChannel()
    await ai.run_tool(make_message(FakeGuild(), channel=channel), tool_name, {})
    assert channel.last == "The voice assistant isn't available on this bot."


def test_voice_note_does_not_push_everything_towards_music():
    # Regression (live): "assistant off" by voice played a YouTube video, because
    # the note said requests are "usually a music command".
    note = ai.VOICE_NOTE
    assert "ONLY when the request clearly asks for music" in note
    assert "usually a music command" not in note
    names = [spec["name"] for spec in ai.TOOL_SPECS]
    assert "turn_off_assistant" in names and "turn_on_assistant" in names
