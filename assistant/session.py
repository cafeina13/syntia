"""
The voice assistant inside Syntia: `syntia assistant on|off`.

    you: "hey jarvis, Tarkan'dan Şımarık çal"
      VoiceReceiver      your audio, decrypted, per speaker   (voice_receive.py)
      SpeakerListener    wake word -> capture the command     (listener.py)
      Transcriber        local Whisper as a GATE: real speech or not? (stt.py)
      ai.ask_ai(audio)   Gemini listens to the clip + calls Syntia's tools
      Speaker            "Tamam, hallediyorum." ... "Tarkan, Şımarık çalıyorum."

This module is imported by bot.py at startup, so it must stay LIGHT: the heavy
packages (Whisper, openWakeWord, Piper, SciPy) are only imported when someone
actually turns the assistant on. Without them installed, the music bot runs as
usual and `syntia assistant` explains what's missing.

Models are shared across servers, loaded on the first `on`, and unloaded when
the last session ends — that frees ~1 GB of VRAM for games.
"""

import asyncio
import gc
import importlib.util
import io
import os
import wave
from dataclasses import dataclass
from typing import Any, Callable

import discord

import ai
import config
import music
from assistant import phrases
from assistant.request import ReplyChannel, VoiceRequest
from assistant.voice_receive import ListeningVoiceClient, VoiceReceiver

# Piper is optional (no voice = text-only replies), so it isn't listed here.
REQUIRED_PACKAGES = ["faster_whisper", "openwakeword", "scipy", "numpy"]


def missing_packages() -> list[str]:
    return [name for name in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]


def parse_ids(text: str | None) -> set[int]:
    return {int(part) for part in (text or "").replace(" ", "").split(",") if part.isdigit()}


def wav_bytes(audio) -> bytes:
    # 16 kHz mono int16 samples -> a complete .wav file, what Gemini accepts.
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(audio.astype("int16").tobytes())
    return buffer.getvalue()


@dataclass
class Models:
    transcriber: Any  # assistant.stt.Transcriber
    speaker: Any  # assistant.speaker.Speaker (silent if no Piper voice)
    make_listener: Callable  # (user_id, on_event) -> SpeakerListener
    chime: bytes


def load_models(stt_device: str, piper_model: str | None) -> tuple[Models, list[str]]:
    # Blocking and slow (seconds): call it in a thread. Returns the models plus
    # notes worth telling the user (e.g. "no voice: PIPER_MODEL isn't set").
    from openwakeword.model import Model as WakeModel
    from openwakeword.vad import VAD

    from assistant.listener import SpeakerListener
    from assistant.speaker import Speaker, chime_pcm
    from assistant.stt import Transcriber
    from assistant.windows import disable_power_throttling

    notes = []
    disable_power_throttling()  # a background console must not run Whisper on slow cores
    try:
        transcriber = Transcriber(device=stt_device)
    except Exception as error:
        if stt_device == "cpu":
            raise
        notes.append(f"GPU speech-to-text failed ({type(error).__name__}), using the CPU — slower.")
        transcriber = Transcriber(device="cpu")

    voice = None
    if piper_model:
        try:
            from assistant.tts import Voice
            voice = Voice(piper_model)
        except Exception as error:
            notes.append(f"Couldn't load the Piper voice ({type(error).__name__}): replies will be text only.")
    else:
        notes.append("No PIPER_MODEL in .env: replies will be text only.")

    def make_listener(user_id, on_event):
        wake = WakeModel(wakeword_models=["hey_jarvis"], inference_framework="onnx")
        return SpeakerListener(user_id, wake, VAD(), on_event)

    return Models(transcriber, Speaker(voice), make_listener, chime_pcm()), notes


class VoiceSession:
    """The assistant listening in one server's voice channel."""

    def __init__(self, manager: "AssistantManager", guild: discord.Guild,
                 text_channel, voice: discord.VoiceClient, models: Models):
        self.manager = manager
        self.guild = guild
        self.text_channel = text_channel
        self.voice = voice
        self.models = models
        self.listeners: dict[int, Any] = {}
        self.receiver = None
        self.ticker = None
        self.stopped = False
        self._command_lock = asyncio.Lock()  # one voice command at a time
        self._tasks: set[asyncio.Task] = set()

    # --- lifecycle -------------------------------------------------------------

    def start(self):
        self._attach(self.voice)
        self.ticker = asyncio.get_running_loop().create_task(self._tick_forever())

    def _attach(self, voice):
        if self.receiver is not None:
            self.receiver.stop()
        self.voice = voice
        self.receiver = self.manager.make_receiver(voice, self._on_audio, users=self.manager.permitted)
        self.receiver.start()

    def stop(self):
        self.stopped = True
        current = asyncio.current_task()
        # stop() can be called FROM the ticker (auto-off after leaving voice):
        # cancelling ourselves would abort the goodbye message, so the loop just
        # sees `stopped` and ends instead.
        if self.ticker is not None and self.ticker is not current:
            self.ticker.cancel()
        if self.receiver is not None:
            self.receiver.stop()
        for task in list(self._tasks):
            if task is not current:
                task.cancel()

    def _spawn(self, coro):
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _tick_forever(self):
        while not self.stopped:
            await asyncio.sleep(0.1)
            await self.tick()

    async def tick(self):
        current = self.guild.voice_client
        if current is None or not current.is_connected():
            await self.manager.stop(self.guild, reason="left the voice channel")
            return
        if current is not self.voice:
            self._attach(current)  # reconnected: listen on the new connection
        for listener in self.listeners.values():
            listener.tick()

    # --- audio in --------------------------------------------------------------

    def _on_audio(self, user_id: int, pcm: bytes, timestamp: int):
        listener = self.listeners.get(user_id)
        if listener is None:
            listener = self.listeners[user_id] = self.models.make_listener(user_id, self._on_event)
        listener.feed(pcm)

    def _on_event(self, kind: str, listener, info: dict):
        from assistant.listener import CHIME, UTTERANCE  # listener.py is only loaded once on
        if kind == CHIME:
            self._spawn(self.models.speaker.play_pcm(self.guild, self.models.chime))
        elif kind == UTTERANCE:
            self._spawn(self.handle_command(listener.user_id, info["audio"]))

    # --- one voice command -------------------------------------------------------

    async def handle_command(self, user_id: int, audio) -> dict | None:
        async with self._command_lock:
            speaker = self.models.speaker
            gate = await asyncio.to_thread(self.models.transcriber.transcribe, audio)
            if not gate.text:
                # No real speech (or garbage): don't spend an AI request on it.
                self._spawn(speaker.say(self.guild, phrases.NOT_UNDERSTOOD))
                return None
            member = self.guild.get_member(user_id)
            if member is None:
                return None
            await self.text_channel.send(f"-# 🎙️ {member.display_name} (voice)", silent=True)

            async def decided(result):
                line = phrases.when_decided(result)
                if line:
                    self._spawn(speaker.say(self.guild, line))  # don't hold up the tools

            async def first_track(entry):
                from assistant.speaker import clean_title
                self._spawn(speaker.say(self.guild, phrases.NOW_PLAYING.format(title=clean_title(entry["title"]))))

            async def slow(status):
                self._spawn(speaker.say(self.guild, phrases.SLOW))

            async def assistant_off():
                # Say goodbye first (and wait for it): stopping the session cancels
                # its background speech, and unloads the voice right after.
                await speaker.say(self.guild, phrases.ASSISTANT_OFF)
                await self.manager.stop(self.guild, reason=None)
                await self.text_channel.send("🎙️ Voice assistant off.")

            reply = ReplyChannel(self.text_channel, on_first_track=first_track,
                                 on_slow_answer=slow, on_decided=decided,
                                 on_assistant_off=assistant_off)
            result = await ai.ask_ai(VoiceRequest(self.guild, member, reply), "", audio=wav_bytes(audio))
            line = phrases.for_error(result)
            if line:
                self._spawn(speaker.say(self.guild, line))
            return result


class AssistantManager:
    """One per bot: permissions, shared models, and a session per server."""

    def __init__(self, *, permitted: set[int] | None = None, owner_id: int | None = None,
                 stt_device: str | None = None, piper_model: str | None = None,
                 loader: Callable = load_models, make_receiver: Callable = VoiceReceiver):
        self.owner_id = config.OWNER_ID if owner_id is None else owner_id
        self.permitted = set(parse_ids(os.getenv("VOICE_USER_IDS")) if permitted is None else permitted)
        if self.owner_id:
            self.permitted.add(self.owner_id)
        self.stt_device = stt_device or os.getenv("STT_DEVICE", "cuda")
        self.piper_model = piper_model if piper_model is not None else os.getenv("PIPER_MODEL")
        self.loader = loader
        self.make_receiver = make_receiver
        self.models: Models | None = None
        self.sessions: dict[int, VoiceSession] = {}
        self._load_lock = asyncio.Lock()

    @staticmethod
    def install():
        # Every voice connection the music bot makes will identify speakers from
        # its first moment, so turning the assistant on mid-song still works.
        music.VOICE_CLIENT_CLASS = ListeningVoiceClient

    def allowed(self, user_id: int) -> bool:
        return user_id in self.permitted

    async def command(self, message: discord.Message, arg: str = ""):
        arg = arg.lower()
        if arg not in ("on", "off"):
            session = self.sessions.get(message.guild.id)
            state = f"**on** in {session.voice.channel.name}" if session else "**off**"
            await message.channel.send(f"🎙️ Voice assistant is {state}. `syntia assistant on|off`")
            return
        if not self.allowed(message.author.id):
            await message.channel.send("Only people listed in VOICE_USER_IDS (or the owner) can do that.")
            return
        if arg == "on":
            await self.start(message)
        else:
            if message.guild.id not in self.sessions:
                await message.channel.send("🎙️ The voice assistant is already off.")
                return
            await self.stop(message.guild, reason=None)
            await message.channel.send("🎙️ Voice assistant off.")

    async def start(self, message: discord.Message):
        guild = message.guild
        if guild.id in self.sessions:
            await message.channel.send("🎙️ Already listening. `syntia assistant off` to stop.")
            return
        missing = missing_packages()
        if missing:
            await message.channel.send(
                f"Voice assistant isn't installed (missing: {', '.join(missing)}). "
                "Run `pip install -r assistant/requirements.txt`.")
            return
        voice = await music.ensure_voice(message)  # also tells them to join voice first
        if voice is None:
            return
        notes = []
        async with self._load_lock:
            if self.models is None:
                await message.channel.send("🎙️ Loading the voice models (a few seconds)...")
                try:
                    self.models, notes = await asyncio.to_thread(self.loader, self.stt_device, self.piper_model)
                except Exception as error:
                    await message.channel.send(f"Couldn't start the voice assistant: {error}")
                    return
        session = VoiceSession(self, guild, message.channel, voice, self.models)
        self.sessions[guild.id] = session
        session.start()
        extra = "".join(f"\n-# {note}" for note in notes)
        await message.channel.send(
            f"🎙️ Voice assistant **on** in **{voice.channel.name}**. Say **\"hey jarvis\"** and your "
            f"command, e.g. *hey jarvis, Tarkan'dan Şımarık çal*. `syntia assistant off` to stop.{extra}")

    async def stop(self, guild: discord.Guild, reason: str | None):
        session = self.sessions.pop(guild.id, None)
        if session is None:
            return
        session.stop()
        if reason:
            await session.text_channel.send(f"🎙️ Voice assistant off ({reason}).", silent=True)
        if not self.sessions and self.models is not None:
            # Nobody is listening anywhere: let Whisper, Piper and the wake word
            # models go, so their RAM and ~1 GB of VRAM are free again.
            self.models = None
            gc.collect()
