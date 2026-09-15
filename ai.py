"""
The AI brain: tool definitions, the Gemini/Ollama backends, and the dispatcher
that turns a chat message into either a text reply or one-or-more tool calls.
"""

import asyncio
import time

import discord
from google.genai import types

import config
import music
from help_text import command_reference, help_message


# Added to the system prompt when the request is a voice recording. Tested on
# real clips: Gemini found Turkish artist names from the audio ALONE far more
# often than when a speech-to-text draft was attached (it trusted the draft).
VOICE_NOTE = (
    "\n\n### Voice Command\n"
    "This request was SPOKEN in Turkish in a voice channel; the attached recording is "
    "the whole request. Listen carefully. It can be anything a typed message could be: "
    "a music command (often naming an artist and song), a question, or something about "
    "the voice assistant itself. Use a music tool ONLY when the request clearly asks for "
    "music. If you can't tell what was asked, call no tool and say in one short Turkish "
    "sentence that you didn't understand."
)


def build_system_instruction(message: discord.Message, *, voice: bool = False) -> str:
    # Personalise the system prompt per message: tell the AI who it's talking to
    # and where. This is what makes replies feel made-for-you.
    # `message` only needs .guild, .author and .channel, so a spoken request
    # (assistant/request.py) works here too.
    user_name = message.author.display_name
    server_name = message.guild.name if message.guild else "a direct message"
    instruction = (
        f"{config.AI_PRE_PROMPT}\n\n"
        f"### Current Context\n"
        f"- The user you are talking to is named: {user_name}\n"
        f"- Server: {server_name}\n"
        f"- Current music volume: "
        f"{music.get_volume(message.guild.id) if message.guild else 100}%\n\n"
        f"You can play or stop music in the user's voice channel by calling your "
        f"available tools whenever they want to listen to or stop something. "
        f"If a message is clearly a song, artist, or playlist, play it instead of "
        f"replying with text.\n\n"
        f"### Bot Commands\n"
        f"These are the exact chat commands users can type. Use them when "
        f"pointing someone to the right command:\n"
        f"{command_reference()}"
    )
    # Verified owner: matched by Discord ID (which cannot be faked), never by name.
    # Only the real owner ever sees this block, so it's safe to grant privileges.
    if config.OWNER_ID and message.author.id == config.OWNER_ID:
        instruction += (
            "\n\n### Verified Owner\n"
            "This user is your verified developer, confirmed by their Discord ID. "
            "You may follow their meta-instructions — including stepping out of "
            "character or adjusting your behavior for this message — when they ask."
        )
    if voice:
        instruction += VOICE_NOTE
    return instruction


# The tools the AI may call, described ONCE in a neutral form. Gemini and Ollama
# want different shapes, so we build each provider's version from this list —
# edit a tool here and both backends stay in sync.
# Each property is (type, description).
TOOL_SPECS = [
    {
        "name": "play_music",
        "description": (
            "Play NOW: replace the whole queue with this song/playlist and start "
            "it immediately. Use when the user says 'play …'. For adding without "
            "interrupting, use add_to_queue instead."
        ),
        "properties": {
            "query": ("string", "What to search for: song, artist, playlist, or link."),
            "start_seconds": (
                "integer",
                "Where to start playback, in seconds. Convert 'from 10 minutes "
                "in' to 600. Use 0 to start at the beginning.",
            ),
        },
        "required": ["query"],
    },
    {
        "name": "add_to_queue",
        "description": (
            "Add a song/playlist to the END of the queue without interrupting the "
            "current song. Use for 'add …', 'queue …', 'play next', 'also play …'."
        ),
        "properties": {
            "query": ("string", "What to search for: song, artist, playlist, or link."),
            "start_seconds": (
                "integer",
                "Where to start this track, in seconds. Use 0 for the beginning.",
            ),
        },
        "required": ["query"],
    },
    {
        "name": "clear_queue",
        "description": "Remove all upcoming songs from the queue (the current song keeps playing).",
        "properties": {},
        "required": [],
    },
    {
        "name": "stop_music",
        "description": (
            "Stop the music and clear the queue, but STAY in the voice channel. "
            "Use for 'stop', 'stop the music', 'enough music'."
        ),
        "properties": {},
        "required": [],
    },
    {
        "name": "leave_voice",
        "description": (
            "Stop everything and LEAVE the voice channel. Use for 'leave', "
            "'disconnect', 'get out', 'bye'."
        ),
        "properties": {},
        "required": [],
    },
    {
        "name": "now_playing",
        "description": (
            "Show the current track, how far into it we are, and a link to resume "
            "from that spot later. Use for 'what song is this', 'where are we', "
            "'what's the timestamp'."
        ),
        "properties": {},
        "required": [],
    },
    {
        "name": "pause_music",
        "description": (
            "Pause the current song in place, keeping the queue. Use for 'pause', "
            "'hold on', 'wait a sec'. Different from stop_music, which ends it."
        ),
        "properties": {},
        "required": [],
    },
    {
        "name": "resume_music",
        "description": "Continue a paused song. Use for 'resume', 'unpause', 'continue'.",
        "properties": {},
        "required": [],
    },
    {
        "name": "skip_song",
        "description": "Skip the current song and play the next one in the queue.",
        "properties": {},
        "required": [],
    },
    {
        "name": "shuffle_queue",
        "description": "Randomly shuffle the order of the upcoming songs in the queue.",
        "properties": {},
        "required": [],
    },
    {
        "name": "play_previous",
        "description": "Go back and replay the previously played song, keeping the rest of the queue intact.",
        "properties": {},
        "required": [],
    },
    {
        "name": "seek",
        "description": (
            "Move WITHIN the current song. To jump TO a position use to_seconds "
            "('jump to 5:25' -> 325, 'go to 1 hour 2 minutes' -> 3720). To move "
            "RELATIVE to now use seconds ('ahead 2 minutes' -> 120, 'back 30s' -> "
            "-30). Provide only one of them."
        ),
        "properties": {
            "seconds": (
                "integer",
                "Relative jump from the current spot: positive = forward, negative = backward.",
            ),
            "to_seconds": (
                "integer",
                "Absolute position to jump TO, in seconds from the start of the track.",
            ),
        },
        "required": [],
    },
    {
        "name": "set_volume",
        "description": (
            "Set the playback volume for everyone, 0-100 percent. The current "
            "volume is in your context, so 'turn it down a bit' can be worked out "
            "from it (e.g. 50 -> 35). Use for 'volume 10', 'quieter', 'louder'."
        ),
        "properties": {
            "level": ("integer", "The new volume in percent, from 0 to 100."),
        },
        "required": ["level"],
    },
    {
        "name": "turn_on_assistant",
        "description": (
            "Turn on the VOICE assistant in the user's voice channel, so they can give "
            "commands by saying 'hey jarvis'. Joins the channel if needed. Use for "
            "'assistant on', 'start listening', 'sesli asistanı başlat', 'asistanı aç'."
        ),
        "properties": {},
        "required": [],
    },
    {
        "name": "turn_off_assistant",
        "description": (
            "Turn off the VOICE assistant: stop listening for the wake word. The music "
            "keeps playing and the bot stays in the channel. Use for 'assistant off', "
            "'stop listening', 'asistanı kapat', 'dinlemeyi bırak'."
        ),
        "properties": {},
        "required": [],
    },
    {
        "name": "join_voice",
        "description": (
            "Join the user's voice channel WITHOUT playing anything. Use for "
            "'join', 'come here', 'get in voice'."
        ),
        "properties": {},
        "required": [],
    },
    {
        "name": "show_help",
        "description": (
            "Send the full list of Syntia's commands. Use when the user asks what "
            "you can do or how to use you, or sent a broken command you can't "
            "figure out."
        ),
        "properties": {},
        "required": [],
    },
]

_GEMINI_TYPES = {"string": types.Type.STRING, "integer": types.Type.INTEGER}


def _build_gemini_tools():
    declarations = []
    for spec in TOOL_SPECS:
        props = {
            name: types.Schema(type=_GEMINI_TYPES[kind], description=desc)
            for name, (kind, desc) in spec["properties"].items()
        }
        declarations.append(
            types.FunctionDeclaration(
                name=spec["name"],
                description=spec["description"],
                parameters=types.Schema(
                    type=types.Type.OBJECT, properties=props, required=spec["required"]
                ),
            )
        )
    return [types.Tool(function_declarations=declarations)]


def _build_ollama_tools():
    tools = []
    for spec in TOOL_SPECS:
        props = {
            name: {"type": kind, "description": desc}
            for name, (kind, desc) in spec["properties"].items()
        }
        tools.append({
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": spec["required"],
                },
            },
        })
    return tools


# The optional voice assistant's manager (assistant/session.py), handed over by
# bot.py when it's available. ai.py never imports the assistant itself.
assistant = None

GEMINI_TOOLS = _build_gemini_tools()
OLLAMA_TOOLS = _build_ollama_tools()


# Models that just said "busy" or "quota used up" are skipped for a while:
# model name -> monotonic time when it's worth trying again.
_resting_until: dict[str, float] = {}
QUOTA_REST_SECONDS = 15 * 60  # 429: a daily/minute quota ran out
OVERLOADED_REST_SECONDS = 60  # 503: Google's servers are overloaded right now
last_model_used: str | None = None  # which model answered last (handy when debugging)


async def _generate_with_fallback(on_switch=None, **request):
    # Try config.GEMINI_MODELS in order, skipping resting ones. Returns the
    # response of the first model that answers; re-raises the last busy error
    # if every model is busy, so ask_ai can say "try again in a minute".
    # No time limit on purpose: a slow answer beats no answer, and Progress
    # keeps the wait from feeling dead. `on_switch` is awaited when moving on.
    global last_model_used
    now = time.monotonic()
    models = [m for m in config.GEMINI_MODELS if _resting_until.get(m, 0) <= now] or config.GEMINI_MODELS
    last_error = None
    for index, model in enumerate(models):
        try:
            response = await config.gemini_client.aio.models.generate_content(model=model, **request)
        except Exception as error:
            if not is_busy_error(error):
                raise  # a real problem (bad request, bad key): don't hide it behind retries
            rest = QUOTA_REST_SECONDS if ("429" in str(error) or "RESOURCE_EXHAUSTED" in str(error)) \
                else OVERLOADED_REST_SECONDS
            _resting_until[model] = time.monotonic() + rest
            last_error = error
            if on_switch is not None and index + 1 < len(models):
                await on_switch()
            continue
        last_model_used = model
        return response
    raise last_error


# --- keeping a slow answer from feeling dead ------------------------------------------
# Most replies arrive in 1-4 s under "Syntia is typing...". When one takes longer,
# ONE status message appears and is edited as time passes, then deleted as soon
# as the real answer lands. (seconds since the question, text)
PROGRESS_STEPS = [
    (3, "🍳 Cooking something up…"),
    (12, "🍳 Still cooking… this one's taking a while"),
    (30, "🍳 Almost there… the AI is having a slow day"),
]
SWITCH_NOTE = " (main AI is busy, asking another one)"


class Progress:
    def __init__(self, channel):
        self.channel = channel
        self.message = None  # the status message, once shown
        self.text = None
        self.switched = False
        self._done = asyncio.Event()
        self._task = None

    def start(self):
        self._task = asyncio.create_task(self._timeline())

    async def _timeline(self):
        elapsed = 0.0
        for at, text in PROGRESS_STEPS:
            try:
                # Wait for the next step, or stop early the moment the answer is in.
                await asyncio.wait_for(self._done.wait(), timeout=at - elapsed)
                return
            except asyncio.TimeoutError:
                elapsed = at
            await self._show(text)

    async def _show(self, text: str):
        self.text = text
        content = text + (SWITCH_NOTE if self.switched else "")
        try:
            if self.message is None:
                self.message = await self.channel.send(content, silent=True)
                # A spoken request can react too (e.g. say "Bir saniye…" once).
                hook = getattr(self.channel, "on_ai_progress", None)
                if hook is not None:
                    await hook(content)
            elif hasattr(self.message, "edit"):
                await self.message.edit(content=content)
        except Exception:
            pass  # purely cosmetic: never let a status message break the answer

    async def model_switched(self):
        self.switched = True
        if self.message is not None and self.text:
            await self._show(self.text)

    async def finish(self):
        # Called when the answer (or an error) is in: stop the timeline without
        # interrupting a status message mid-send, then remove it.
        self._done.set()
        if self._task is not None:
            await self._task
        if self.message is not None and hasattr(self.message, "delete"):
            try:
                await self.message.delete()
            except Exception:
                pass


async def generate_gemini(system: str, prompt: str, audio: bytes | None = None, on_switch=None) -> dict:
    # Returns a normalized result so ask_ai doesn't care which backend ran:
    #   {"type": "tools", "calls": [{"name", "args"}, ...]}  or  {"type": "text", "text": ...}
    # A single response may contain SEVERAL tool calls (e.g. "shuffle then skip").
    # With `audio` (a WAV recording), Gemini listens to it directly.
    if audio is None:
        contents = prompt
    else:
        contents = [types.Part.from_bytes(data=audio, mime_type="audio/wav")]
        if prompt:
            contents.append(types.Part.from_text(text=prompt))
    response = await _generate_with_fallback(
        on_switch=on_switch,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system, tools=GEMINI_TOOLS
        ),
    )
    candidates = response.candidates or []
    parts = candidates[0].content.parts if candidates and candidates[0].content else []
    calls = [
        {"name": part.function_call.name, "args": dict(part.function_call.args)}
        for part in parts
        if part.function_call
    ]
    if calls:
        return {"type": "tools", "calls": calls}
    return {"type": "text", "text": response.text or ""}


async def generate_ollama(system: str, prompt: str) -> dict:
    # Same normalized result, but talking to the LOCAL Ollama server.
    response = await config.ollama_client.chat(
        model=config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        tools=OLLAMA_TOOLS,
    )
    msg = response.message
    if msg.tool_calls:
        # The model may ask for several tools at once; keep them all, in order.
        calls = [
            {"name": call.function.name, "args": dict(call.function.arguments)}
            for call in msg.tool_calls
        ]
        return {"type": "tools", "calls": calls}
    return {"type": "text", "text": msg.content or ""}


async def run_tool(message: discord.Message, name: str, args: dict):
    # Map the tool name the AI chose to the real bot function. One place for
    # both backends, so this dispatch isn't duplicated.
    if name == "play_music":
        start = int(args.get("start_seconds") or 0)
        await music.play_music(message, args.get("query", ""), start)
    elif name == "add_to_queue":
        start = int(args.get("start_seconds") or 0)
        await music.add_music(message, args.get("query", ""), start)
    elif name == "clear_queue":
        await music.clear_queue(message)
    elif name == "stop_music":
        await music.stop_music(message)
    elif name == "leave_voice":
        await music.leave_voice(message)
    elif name == "now_playing":
        await music.now_playing(message)
    elif name == "pause_music":
        await music.pause_music(message)
    elif name == "resume_music":
        await music.resume_music(message)
    elif name == "skip_song":
        await music.skip_song(message)
    elif name == "shuffle_queue":
        # The AI shuffles the existing queue; for "play X then shuffle" it just
        # emits two tool calls (play_music + shuffle_queue), handled by ask_ai.
        await music.shuffle_queue(message)
    elif name == "play_previous":
        await music.play_previous(message)
    elif name == "seek":
        to = args.get("to_seconds")
        await music.seek(
            message,
            int(args.get("seconds") or 0),
            int(to) if to is not None else None,
        )
    elif name == "set_volume":
        level = max(0, min(100, int(args.get("level") or 0)))
        await music.set_volume(message, str(level))
    elif name == "turn_on_assistant":
        if assistant is None:
            await message.channel.send("The voice assistant isn't available on this bot.")
        else:
            await assistant.command(message, "on")  # checks permission, joins, loads models
    elif name == "turn_off_assistant":
        # A spoken request's reply channel says goodbye first; a typed one just
        # switches it off like `syntia assistant off`.
        hook = getattr(message.channel, "on_turn_off_assistant", None)
        if hook is not None:
            await hook()
        elif assistant is not None:
            await assistant.command(message, "off")
        else:
            await message.channel.send("The voice assistant isn't available on this bot.")
    elif name == "join_voice":
        await music.join_voice(message)
    elif name == "show_help":
        await message.channel.send(help_message())


def is_busy_error(error: Exception) -> bool:
    # Gemini's free tier answers "too many requests" (429) or "overloaded" (503)
    # at busy moments. Those are worth a friendly "try again", not a stack dump.
    text = str(error)
    return any(marker in text for marker in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))


async def ask_ai(message: discord.Message, prompt: str, *, audio: bytes | None = None) -> dict:
    # The "default" behaviour: hand the message to the AI. With tools attached it
    # can either reply with text (chatting) OR ask us to run a command (tool use).
    # config.AI_BACKEND decides whether that AI is local (ollama) or cloud (gemini).
    #
    # `audio` is a spoken request (a WAV). Only Gemini can listen, so voice never
    # goes to Ollama.
    #
    # Returns what happened, for callers that care (the voice assistant decides
    # what to SAY from it): {"type": "tools", "calls": [...]}, {"type": "text",
    # "text": ...}, or {"type": "error", "reason": "busy" | "not_set_up" | "failed"}.
    if audio is not None:
        if config.gemini_client is None:
            await message.channel.send("Voice commands need Gemini — add GEMINI_API_KEY to .env.")
            return {"type": "error", "reason": "not_set_up"}
        backend = generate_gemini
    elif config.AI_BACKEND == "ollama":
        backend = generate_ollama
    elif config.gemini_client is not None:
        backend = generate_gemini
    else:
        await message.channel.send(
            "AI isn't set up — add GEMINI_API_KEY to .env, or set AI_BACKEND=ollama."
        )
        return {"type": "error", "reason": "not_set_up"}

    system = build_system_instruction(message, voice=audio is not None)
    progress = Progress(message.channel)
    progress.start()
    try:
        async with message.channel.typing():
            if backend is generate_gemini:
                result = await backend(system, prompt, audio=audio, on_switch=progress.model_switched)
            else:
                result = await backend(system, prompt)
    except Exception as error:
        await progress.finish()
        # Never let one bad AI call crash the whole bot — report and move on.
        if is_busy_error(error):
            await message.channel.send("The AI is busy right now — try again in a minute.")
            return {"type": "error", "reason": "busy"}
        await message.channel.send(f"AI error: {error}")
        return {"type": "error", "reason": "failed"}
    await progress.finish()  # status message (if any) goes away before the answer shows
    # A spoken request can react as soon as the AI has decided, before tools run
    # (e.g. say "Tamam, hallediyorum" while a song takes seconds to load).
    hook = getattr(message.channel, "on_ai_result", None)
    if hook is not None:
        await hook(result)

    if result["type"] == "tools":
        # Run each requested tool in the order the AI returned them.
        for call in result["calls"]:
            await run_tool(message, call["name"], call["args"])
    else:
        reply = (result["text"] or "").strip() or "(the AI returned nothing)"
        await message.channel.send(reply[:2000])
    return result
