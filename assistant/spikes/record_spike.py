"""
Spikes 1 + 3: can we HEAR people in voice, and can Syntia SPEAK?

A tiny standalone bot (not Syntia's music bot) that joins your voice channel,
records every permitted speaker to a WAV file and reports packet stats and CPU
use, and speaks text with your Piper voice. Uses the same bot token, so STOP
the music bot before running this.

    .venv\\Scripts\\python.exe -m assistant.spikes.record_spike

Then in a text channel, while you're in voice:
    spike rec     join and start recording
    spike stats   packet counts so far
    spike stop    save WAVs to recordings/, print stats, leave
    spike say <text>   speak it with the Piper voice (PIPER_MODEL in .env)

.env: VOICE_USER_IDS=123,456 limits recording to those users (default: whoever
typed `spike rec`).
"""

import asyncio
import os
import time
import wave
from datetime import datetime
from pathlib import Path

import discord
from discord.opus import OPUS_SILENCE
from dotenv import load_dotenv

from assistant.voice_receive import BYTES_PER_SAMPLE, SAMPLES_PER_FRAME, VoiceReceiver

try:
    from assistant.tts import Voice
except ImportError:  # piper-tts not installed: `spike say` just explains
    Voice = None

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
PERMITTED = {int(x) for x in os.getenv("VOICE_USER_IDS", "").replace(" ", "").split(",") if x}
PREFIX = "spike "
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "recordings"
MAX_SECONDS = 300  # per user, so a forgotten recording can't eat all the RAM
MAX_GAP_SECONDS = 1.0  # longer silences get squashed to this in the WAV
PIPER_MODEL = os.getenv("PIPER_MODEL")
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
    if session:
        await message.channel.send("Already recording. `spike stop` first.")
        return
    if not message.author.voice or not message.author.voice.channel:
        await message.channel.send("Join a voice channel first.")
        return
    voice = message.guild.voice_client or await message.author.voice.channel.connect()
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
    voice = message.guild.voice_client or await message.author.voice.channel.connect()
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


@client.event
async def on_ready():
    print(f"Spike bot ready as {client.user}. Type `spike rec` while in voice.")
    print(f"Permitted users: {sorted(PERMITTED) or 'whoever types spike rec'}")


@client.event
async def on_message(message: discord.Message):
    if message.author.bot or not message.guild:
        return
    text = message.content.lower().strip()
    if text == PREFIX + "rec":
        await start_recording(message)
    elif text == PREFIX + "stop":
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
