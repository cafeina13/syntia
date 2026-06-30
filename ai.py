"""
The AI brain: tool definitions, the Gemini/Ollama backends, and the dispatcher
that turns a chat message into either a text reply or one-or-more tool calls.
"""

import discord
from google.genai import types

import config
import music


def build_system_instruction(message: discord.Message) -> str:
    # Personalise the system prompt per message: tell the AI who it's talking to
    # and where. This is what makes replies feel made-for-you.
    user_name = message.author.display_name
    server_name = message.guild.name if message.guild else "a direct message"
    instruction = (
        f"{config.AI_PRE_PROMPT}\n\n"
        f"### Current Context\n"
        f"- The user you are talking to is named: {user_name}\n"
        f"- Server: {server_name}\n\n"
        f"You can play or stop music in the user's voice channel by calling your "
        f"available tools whenever they want to listen to or stop something. "
        f"If a message is clearly a song, artist, or playlist, play it instead of "
        f"replying with text."
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
        "description": "Stop playback and leave the voice channel.",
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


GEMINI_TOOLS = _build_gemini_tools()
OLLAMA_TOOLS = _build_ollama_tools()


async def generate_gemini(system: str, prompt: str) -> dict:
    # Returns a normalized result so ask_ai doesn't care which backend ran:
    #   {"type": "tools", "calls": [{"name", "args"}, ...]}  or  {"type": "text", "text": ...}
    # A single response may contain SEVERAL tool calls (e.g. "shuffle then skip").
    response = await config.gemini_client.aio.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
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
        await music.leave_voice(message)
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


async def ask_ai(message: discord.Message, prompt: str):
    # The "default" behaviour: hand the message to the AI. With tools attached it
    # can either reply with text (chatting) OR ask us to run a command (tool use).
    # config.AI_BACKEND decides whether that AI is local (ollama) or cloud (gemini).
    if config.AI_BACKEND == "ollama":
        backend = generate_ollama
    elif config.gemini_client is not None:
        backend = generate_gemini
    else:
        await message.channel.send(
            "AI isn't set up — add GEMINI_API_KEY to .env, or set AI_BACKEND=ollama."
        )
        return

    try:
        async with message.channel.typing():
            result = await backend(build_system_instruction(message), prompt)
    except Exception as error:
        # Never let one bad AI call crash the whole bot — report and move on.
        await message.channel.send(f"AI error: {error}")
        return

    if result["type"] == "tools":
        # Run each requested tool in the order the AI returned them.
        for call in result["calls"]:
            await run_tool(message, call["name"], call["args"])
    else:
        reply = (result["text"] or "").strip() or "(the AI returned nothing)"
        await message.channel.send(reply[:2000])
