"""
Spikes 1, 3, 4: can we HEAR people in voice, can Syntia SPEAK, and can it WAKE?

A tiny standalone bot (not Syntia's music bot) for trying the voice pieces in
Discord. Uses the same bot token, so STOP the music bot before running this.

    .venv\\Scripts\\python.exe -m assistant.spikes.record_spike

Then in a text channel, while you're in voice:
    spike rec          record permitted speakers to WAV (spike 1)
    spike stats        packet counts so far
    spike say <text>   speak it with the Piper voice (spike 3, PIPER_MODEL in .env)
    spike listen       wake word -> command -> transcript in chat (spike 4)
    spike stop         end recording or listening, print stats, leave

.env: VOICE_USER_IDS=123,456 limits it to those users (default: whoever typed
the command).
"""

import asyncio
import io
import os
import time
import wave
from datetime import datetime
from pathlib import Path

import discord
from discord.opus import OPUS_SILENCE
from dotenv import load_dotenv

from assistant.voice_receive import BYTES_PER_SAMPLE, SAMPLES_PER_FRAME, ListeningVoiceClient, VoiceReceiver

try:
    from assistant.tts import Voice
except ImportError:  # piper-tts not installed: `spike say` just explains
    Voice = None

try:
    import numpy as np
    from openwakeword.model import Model as WakeModel
    from openwakeword.vad import VAD

    from assistant.listener import CHIME, GAVE_UP, UTTERANCE, WAKE, SpeakerListener
    from assistant.spikes.whisper_bench import peak_ram_mb
    from assistant.stt import Transcriber
except ImportError:  # assistant/requirements.txt not installed: `spike listen` explains
    WakeModel = None

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
PERMITTED = {int(x) for x in os.getenv("VOICE_USER_IDS", "").replace(" ", "").split(",") if x}
PREFIX = "spike "
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "recordings"
MAX_SECONDS = 300  # per user, so a forgotten recording can't eat all the RAM
MAX_GAP_SECONDS = 1.0  # longer silences get squashed to this in the WAV
PIPER_MODEL = os.getenv("PIPER_MODEL")
STT_DEVICE = os.getenv("STT_DEVICE", "cuda")  # "cpu" keeps the GPU free (~3 s per command)
_voice = None  # loaded on first `spike say`

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


class Recording:
    # One user's audio. Discord sends nothing while someone is silent, so we use
    # the RTP timestamp to put the pauses back (capped) — otherwise every
    # sentence would be glued to the next.
    def __init__(self):
        self.pcm = bytearray()
        self.last_timestamp = None

    def add(self, pcm: bytes, timestamp: int):
        if len(self.pcm) >= MAX_SECONDS * 48000 * BYTES_PER_SAMPLE:
            return
        if self.last_timestamp is not None:
            gap = ((timestamp - self.last_timestamp) & 0xFFFFFFFF) - SAMPLES_PER_FRAME
            if 0 < gap <= 48000 * 60:
                gap = min(gap, int(48000 * MAX_GAP_SECONDS))
                self.pcm += bytes(gap * BYTES_PER_SAMPLE)
        self.pcm += pcm
        self.last_timestamp = timestamp

    @property
    def seconds(self) -> float:
        return len(self.pcm) / (48000 * BYTES_PER_SAMPLE)


session = {}  # the one active recording: voice, receiver, recordings, clocks


def summary(guild: discord.Guild) -> str:
    receiver = session["receiver"]
    wall = time.monotonic() - session["wall_start"]
    process_cpu = time.process_time() - session["cpu_start"]
    audio = sum(r.seconds for r in session["recordings"].values())
    state = receiver.voice._connection
    lines = [
        f"Recording for {wall:.0f}s. DAVE protocol v{state.dave_protocol_version}, "
        f"transport mode `{state.mode}`.",
        f"Packets: {dict(receiver.stats) or 'none yet'}",
        f"Known speakers (SSRC -> user): {len(receiver.ssrc_to_user)}",
    ]
    for user_id, rec in session["recordings"].items():
        member = guild.get_member(user_id)
        lines.append(f"- {member.display_name if member else user_id}: {rec.seconds:.1f}s of audio")
    if wall > 0:
        lines.append(
            f"CPU: receive path {receiver.cpu_seconds * 1000:.0f} ms total"
            + (f" = {receiver.cpu_seconds / audio * 100:.2f}% of a core per second of speech"
               if audio else "")
            + f"; whole bot process {process_cpu / wall * 100:.1f}% of one core."
        )
    return "\n".join(lines)


async def start_recording(message: discord.Message):
    if session or listening:
        await message.channel.send("Already recording. `spike stop` first.")
        return
    if not message.author.voice or not message.author.voice.channel:
        await message.channel.send("Join a voice channel first.")
        return
    voice = await join(message)
    await asyncio.sleep(1)
    # Send a moment of silence: Discord has been known not to deliver audio to a
    # client that has never sent any. (Same trick as the speaking-ring fix.)
    for _ in range(5):
        voice.send_audio_packet(OPUS_SILENCE, encode=False)

    users = PERMITTED or {message.author.id}
    recordings: dict[int, Recording] = {}

    def on_audio(user_id: int, pcm: bytes, timestamp: int):
        recordings.setdefault(user_id, Recording()).add(pcm, timestamp)

    receiver = VoiceReceiver(voice, on_audio, users=users)
    receiver.start()
    session.update(receiver=receiver, recordings=recordings, voice=voice,
                   wall_start=time.monotonic(), cpu_start=time.process_time())
    names = ", ".join(str(u) for u in users)
    await message.channel.send(
        f"Recording in **{voice.channel.name}** for user(s) {names}. "
        "Talk a bit (Turkish sentences are perfect for step 2), then `spike stop`."
    )


async def stop_recording(message: discord.Message):
    if not session:
        await message.channel.send("Not recording.")
        return
    receiver = session["receiver"]
    receiver.stop()
    report = summary(message.guild)
    OUT_DIR.mkdir(exist_ok=True)
    saved = []
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for user_id, rec in session["recordings"].items():
        path = OUT_DIR / f"{stamp}-{user_id}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(48000)
            wav.writeframes(bytes(rec.pcm))
        saved.append(path.name)
    await session["voice"].disconnect()
    session.clear()
    print(report)
    await message.channel.send(
        f"{report}\nSaved: {', '.join(saved) or 'nothing (no audio arrived)'}"[:2000]
    )


async def say(message: discord.Message, text: str):
    global _voice
    if not text:
        await message.channel.send("Usage: `spike say Tamam, hallediyorum.`")
        return
    if Voice is None or not PIPER_MODEL:
        await message.channel.send("Needs `pip install piper-tts` and PIPER_MODEL=<path to .onnx> in .env.")
        return
    if not message.author.voice or not message.author.voice.channel:
        await message.channel.send("Join a voice channel first.")
        return
    voice = await join(message)
    if voice.is_playing():
        await message.channel.send("Still speaking, one moment.")
        return
    started = time.perf_counter()
    if _voice is None:
        _voice = await asyncio.to_thread(Voice, PIPER_MODEL)
    loaded = time.perf_counter()
    pcm = await asyncio.to_thread(_voice.synthesize, text)
    synthesized = time.perf_counter()
    voice.play(_voice.to_discord(pcm))
    await message.channel.send(
        f"Speaking {_voice.seconds(pcm):.1f}s of audio. Voice load {1000 * (loaded - started):.0f} ms, "
        f"synthesis {1000 * (synthesized - loaded):.0f} ms."
    )


# --- spike 4: listen for the wake word -------------------------------------------

listening = {}  # the active listen session
_transcriber = None  # Whisper, loaded on first `spike listen` and kept


def chime_pcm() -> bytes:
    # Two short rising notes, quiet, with 10 ms fades so they don't click.
    # 48 kHz stereo 16-bit: exactly what discord.PCMAudio plays.
    notes = []
    for frequency in (880, 1320):
        t = np.arange(int(48000 * 0.09)) / 48000
        fade = np.minimum(1, np.minimum(t, t[::-1]) / 0.01)
        notes.append(3000 * fade * np.sin(2 * np.pi * frequency * t))
    return np.repeat(np.concatenate(notes).astype(np.int16), 2).tobytes()


def disable_power_throttling() -> bool:
    # Windows 11 puts background processes into "efficiency mode" (EcoQoS): slow
    # efficiency cores, lower speed. A bot whose console isn't the focused window
    # is exactly that. Opt this process out. Returns True if Windows accepted.
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes

    class ThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetProcessInformation.restype = wintypes.BOOL
    PROCESS_POWER_THROTTLING = 4  # ProcessPowerThrottling
    EXECUTION_SPEED = 0x1  # control execution-speed throttling...
    state = ThrottlingState(1, EXECUTION_SPEED, 0)  # ...and turn it OFF
    return bool(kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), PROCESS_POWER_THROTTLING,
                                               ctypes.byref(state), ctypes.sizeof(state)))


def save_command_wav(audio, user_id: int) -> str:
    folder = OUT_DIR / "commands"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{user_id}.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(audio.astype(np.int16).tobytes())
    return path.name


async def join(message: discord.Message) -> ListeningVoiceClient:
    # Always connect as a ListeningVoiceClient, so speakers are identified from
    # the first moment. A leftover plain connection gets replaced.
    voice = message.guild.voice_client
    if voice is not None and not isinstance(voice, ListeningVoiceClient):
        await voice.disconnect(force=True)
        voice = None
    if voice is None:
        voice = await message.author.voice.channel.connect(cls=ListeningVoiceClient)
    elif voice.channel != message.author.voice.channel:
        await voice.move_to(message.author.voice.channel)
    return voice


def name_of(guild: discord.Guild, user_id: int) -> str:
    member = guild.get_member(user_id)
    return member.display_name if member else str(user_id)


async def start_listening(message: discord.Message):
    global _transcriber
    if session or listening:
        await message.channel.send("Already busy. `spike stop` first.")
        return
    if WakeModel is None:
        await message.channel.send("Needs `pip install -r assistant/requirements.txt`.")
        return
    if not message.author.voice or not message.author.voice.channel:
        await message.channel.send("Join a voice channel first.")
        return
    if _transcriber is None:
        await message.channel.send(f"Loading Whisper on {STT_DEVICE} (first time only)...")
        _transcriber = await asyncio.to_thread(Transcriber, device=STT_DEVICE)

    voice = await join(message)
    await asyncio.sleep(1)
    for _ in range(5):
        voice.send_audio_packet(OPUS_SILENCE, encode=False)

    guild, channel = message.guild, message.channel
    loop = asyncio.get_running_loop()
    transcribe_lock = asyncio.Lock()
    chime = chime_pcm()

    async def handle(kind: str, listener: SpeakerListener, info: dict):
        name = name_of(guild, listener.user_id)
        if kind == WAKE:
            listening["wakes"] += 1
            await channel.send(f"👂 **{name}**: wake word (score {info['score']:.2f})")
        elif kind == CHIME:
            if not voice.is_playing():
                voice.play(discord.PCMAudio(io.BytesIO(chime)))
        elif kind == GAVE_UP:
            await channel.send(f"💤 **{name}**: nothing after the chime, listening again.")
        elif kind == UTTERANCE:
            audio = info["audio"]
            saved = save_command_wav(audio, listener.user_id)  # replayable offline
            async with transcribe_lock:  # one Whisper job at a time
                started, cpu_started = time.perf_counter(), time.process_time()
                result = await asyncio.to_thread(_transcriber.transcribe, audio)
                whisper = time.perf_counter() - started
                # CPU seconds / wall seconds = how many cores were actually working.
                cores = (time.process_time() - cpu_started) / whisper
            # finished_at is when 0.8 s of silence was confirmed; speech ended before that.
            since_speech = time.monotonic() - info["finished_at"] + listener.config.end_silence_seconds
            listening["commands"] += 1
            await channel.send(
                f"🎙️ **{name}**: {result.text or '*(nothing understood)*'}\n"
                + (f"-# thrown out ({result.rejected}): {result.raw[:120]}\n" if result.rejected else "")
                + f"`{len(audio) / 16000:.1f}s captured (ended by {info['ended_by']}) · "
                f"Whisper {whisper:.2f}s ({_transcriber.device}, ~{cores:.1f} CPU cores) · "
                f"reply ~{since_speech:.1f}s after you stopped talking · {saved}`"
            )

    def on_event(kind, listener, info):
        # Called from feed()/tick(), already on the event loop.
        loop.create_task(handle(kind, listener, info))

    users = PERMITTED or {message.author.id}
    # One wake word model + VAD per speaker: both remember recent audio.
    listeners = {
        user_id: SpeakerListener(user_id, WakeModel(wakeword_models=["hey_jarvis"], inference_framework="onnx"),
                                 VAD(), on_event)
        for user_id in users
    }
    for listener in listeners.values():
        listener.debug = True  # keep what it heard + the score timeline, saved on stop

    def on_audio(user_id: int, pcm: bytes, timestamp: int):
        listeners[user_id].feed(pcm)

    async def ticker():
        while True:
            await asyncio.sleep(0.1)
            for listener in listeners.values():
                listener.tick()

    receiver = VoiceReceiver(voice, on_audio, users=users)
    receiver.start()
    listening.update(receiver=receiver, listeners=listeners, voice=voice, ticker=loop.create_task(ticker()),
                     wall_start=time.monotonic(), cpu_start=time.process_time(), wakes=0, commands=0)
    await channel.send(
        f"Listening in **{voice.channel.name}** for {', '.join(name_of(guild, u) for u in users)}. "
        f"Whisper loaded in {_transcriber.load_seconds:.1f}s.\n"
        "Try **\"hey jarvis, şarkıyı geç\"** in one breath, or **\"hey jarvis\"** → chime → command. "
        "Also just chat normally for a while to see if it wakes by mistake. `spike stop` when done."
    )


async def stop_listening(message: discord.Message):
    listening["ticker"].cancel()
    receiver = listening["receiver"]
    receiver.stop()
    wall = time.monotonic() - listening["wall_start"]
    process_cpu = time.process_time() - listening["cpu_start"]
    lines = [
        f"Listened for {wall:.0f}s: {listening['wakes']} wake(s), {listening['commands']} command(s). "
        f"Packets: {dict(receiver.stats)}",
    ]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for user_id, listener in listening["listeners"].items():
        heard = listener.audio_seconds
        lines.append(
            f"- {name_of(message.guild, user_id)}: {heard:.0f}s of audio through the pipeline, "
            f"wake word + VAD CPU {listener.cpu_seconds / heard * 100 if heard else 0:.2f}% of one core, "
            f"highest score that did NOT wake: {listener.best_idle_score:.2f}"
        )
        if listener.debug_frames:
            # Exactly what the wake word model heard, plus when scores peaked and
            # when packets were lost — so a miss can be studied offline.
            folder = OUT_DIR / "listen"
            folder.mkdir(parents=True, exist_ok=True)
            base = folder / f"{stamp}-{user_id}"
            with wave.open(str(base.with_suffix(".wav")), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(np.concatenate(listener.debug_frames).tobytes())
            with open(base.with_suffix(".csv"), "w", encoding="utf-8") as log:
                log.write("seconds,wake_score\n")
                log.writelines(f"{t:.2f},{s:.4f}\n" for t, s in listener.score_log)
            top = sorted(listener.score_log, key=lambda x: -x[1])[:5]
            losses = [f"{t:.1f}s" for u, t, _ in receiver.loss_log if u == user_id]
            lines.append(f"  top wake scores at: {', '.join(f'{t:.1f}s={s:.2f}' for t, s in top)}")
            lines.append(f"  lost packets at: {', '.join(losses[:25]) or 'none'}"
                         + (f" (+{len(losses) - 25} more)" if len(losses) > 25 else ""))
            lines.append(f"  saved {base.name}.wav/.csv")
    if receiver.loss_reasons:
        lines.append(f"DAVE errors: {dict(receiver.loss_reasons)}")
    lines.append(f"Whole process: {process_cpu / wall * 100:.1f}% of one core on average "
                 f"(includes Whisper bursts), peak RAM {peak_ram_mb():.0f} MB.")
    report = "\n".join(lines)
    await listening["voice"].disconnect()
    listening.clear()
    print(report)
    await message.channel.send(report[:2000])


@client.event
async def on_ready():
    print(f"Spike bot ready as {client.user}. Type `spike rec`, `spike say ...` or `spike listen` while in voice.")
    print(f"Windows power throttling (efficiency mode) disabled: {disable_power_throttling()}")
    print(f"Permitted users: {sorted(PERMITTED) or 'whoever types the command'}")


@client.event
async def on_message(message: discord.Message):
    if message.author.bot or not message.guild:
        return
    text = message.content.lower().strip()
    if text == PREFIX + "rec":
        await start_recording(message)
    elif text == PREFIX + "listen":
        await start_listening(message)
    elif text == PREFIX + "stop":
        if listening:
            await stop_listening(message)
        else:
            await stop_recording(message)
    elif text.startswith(PREFIX + "say"):
        # Keep the original casing and Turkish letters for the speech itself.
        await say(message, message.content.strip()[len(PREFIX + "say"):].strip())
    elif text == PREFIX + "stats":
        await message.channel.send(summary(message.guild) if session else "Not recording.")


if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("No DISCORD_TOKEN in .env")
    client.run(TOKEN)
