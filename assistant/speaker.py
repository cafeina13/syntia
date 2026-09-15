"""
Syntia's voice in the channel: short spoken lines, over music or on their own.

Discord plays ONE audio source at a time, and play() refuses while a song is
on. So speech goes out one of two ways:

    nothing playing   voice.play(SpeechSource)  — marked is_speech, so a song
                      that becomes ready simply cuts it (music._start_track)
    music playing     the live source is swapped for a DuckingMixer: the song at
                      25% plus the speech, then the song alone again. A source
                      swap, not a new player, so the song's position, volume and
                      "play the next track when done" all carry on untouched.
    music paused      stay quiet — swapping the source would un-pause it.

Audio here is what discord.py plays: 48 kHz stereo 16-bit PCM, 20 ms frames.
"""

import asyncio
import re

import discord
import numpy as np
from scipy.signal import resample_poly

FRAME_BYTES = 3840  # 20 ms at 48 kHz, stereo, 16-bit
DUCK = 0.25  # how loud the music stays under speech
MAX_SPEECH_CHARS = 200


# --- turning text into something worth saying ---------------------------------------

# Emoji and the symbol blocks the bot's own messages use (▶ ⏸ ⏩ ⏭ 🔊 ...),
# which a TTS voice would otherwise try to read out.
_EMOJI = re.compile("[\U0001F000-\U0001FAFF←-⇿⌀-⏿■-◿☀-➿️‍]")
_BRACKETS = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")  # (Official Video), [4K], (Lyrics)...


def clean_for_speech(text: str) -> str:
    text = text or ""
    text = re.sub(r"^-#\s*", "", text, flags=re.MULTILINE)  # Discord small-text marker
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_`~|>#]", "", text)  # markdown
    text = _EMOJI.sub("", text)
    text = _BRACKETS.sub("", text)
    text = re.sub(r"\s+-\s+", ", ", text)  # "Tarkan - Şımarık" reads as "Tarkan, Şımarık"
    text = re.sub(r"\s+", " ", text).strip()
    # Keep it short: the first two sentences, and never more than ~200 characters.
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    text = " ".join(sentences[:2])
    if len(text) > MAX_SPEECH_CHARS:
        text = text[:MAX_SPEECH_CHARS].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return text


def clean_title(title: str) -> str:
    # "Tarkan - Şımarık (Official Video) [HD]" -> "Tarkan, Şımarık"
    return clean_for_speech(_BRACKETS.sub("", title or "")).rstrip(".…")


def to_discord_pcm(pcm16_mono: bytes, sample_rate: int) -> bytes:
    # Piper speaks 16-bit mono at its own rate (22050 Hz); Discord wants 48 kHz stereo.
    samples = np.frombuffer(pcm16_mono, dtype=np.int16).astype(np.float32)
    if sample_rate != 48000:
        gcd = np.gcd(48000, sample_rate)
        samples = resample_poly(samples, 48000 // gcd, sample_rate // gcd)
    mono = np.clip(samples, -32768, 32767).astype(np.int16)
    return np.repeat(mono, 2).tobytes()


def chime_pcm() -> bytes:
    # Two short rising notes, quiet, with 10 ms fades so they don't click.
    notes = []
    for frequency in (880, 1320):
        t = np.arange(int(48000 * 0.09)) / 48000
        fade = np.minimum(1, np.minimum(t, t[::-1]) / 0.01)
        notes.append(3000 * fade * np.sin(2 * np.pi * frequency * t))
    return np.repeat(np.concatenate(notes).astype(np.int16), 2).tobytes()


# --- audio sources -------------------------------------------------------------------


class SpeechSource(discord.AudioSource):
    """Plays a finished PCM clip on its own."""

    is_speech = True  # music._start_track may cut this to start a song

    def __init__(self, pcm: bytes):
        self.pcm = pcm
        self.position = 0

    def read(self) -> bytes:
        frame = self.pcm[self.position:self.position + FRAME_BYTES]
        self.position += FRAME_BYTES
        if not frame:
            return b""
        return frame.ljust(FRAME_BYTES, b"\x00")

    def is_opus(self) -> bool:
        return False


class DuckingMixer(discord.AudioSource):
    """The playing song with speech laid over it, the song ducked while speech lasts."""

    is_speech = False  # it's still the song: never cut it for another song's start

    def __init__(self, music: discord.AudioSource, speech_pcm: bytes, duck: float = DUCK):
        self.original = music
        self.speech = speech_pcm
        self.position = 0
        self.duck = duck
        self.released = False  # True once the song has been handed back to the player

    @property
    def speaking(self) -> bool:
        return self.position < len(self.speech)

    def read(self) -> bytes:
        frame = self.original.read()
        if not frame:
            return b""  # the song ended: let the player finish and move to the next track
        if not self.speaking:
            return frame
        speech = self.speech[self.position:self.position + FRAME_BYTES].ljust(len(frame), b"\x00")
        self.position += FRAME_BYTES
        mixed = (np.frombuffer(frame, dtype=np.int16).astype(np.int32) * self.duck
                 + np.frombuffer(speech[:len(frame)], dtype=np.int16))
        return np.clip(mixed, -32768, 32767).astype(np.int16).tobytes()

    def is_opus(self) -> bool:
        return False

    def release(self) -> discord.AudioSource:
        # Hand the song back. After this the mixer must never touch it again.
        self.released = True
        return self.original

    def cleanup(self):
        # discord.py calls cleanup() when a source finishes AND when Python
        # garbage-collects it (AudioSource.__del__). For a song, cleanup kills its
        # FFmpeg process — so a mixer that already handed the song back must NOT
        # pass this on, or the song dies moments after Syntia stops talking.
        if not self.released:
            self.original.cleanup()


# --- the speaker ---------------------------------------------------------------------


class Speaker:
    def __init__(self, voice_model=None):
        # voice_model: an assistant.tts.Voice, or None for a silent (text-only) assistant.
        self.voice_model = voice_model
        self._lock = asyncio.Lock()  # one line at a time, in the order they were asked

    async def say(self, guild: discord.Guild, text: str):
        text = clean_for_speech(text)
        if not text or self.voice_model is None:
            return
        try:
            pcm = await asyncio.to_thread(self._synthesize, text)
        except Exception:
            return  # a voice glitch must never break the command itself
        await self.play_pcm(guild, pcm)

    def _synthesize(self, text: str) -> bytes:
        return to_discord_pcm(self.voice_model.synthesize(text), self.voice_model.sample_rate)

    async def play_pcm(self, guild: discord.Guild, pcm: bytes):
        async with self._lock:
            try:
                await self._play(guild, pcm)
            except Exception:
                pass  # e.g. disconnected mid-sentence

    async def _play(self, guild: discord.Guild, pcm: bytes):
        voice = guild.voice_client
        if voice is None or not voice.is_connected() or voice.is_paused():
            return
        seconds = len(pcm) / (48000 * 4)
        if voice.is_playing() and not getattr(voice.source, "is_speech", False):
            mixer = DuckingMixer(voice.source, pcm)
            voice.source = mixer
            await asyncio.sleep(seconds + 0.1)
            if voice.source is mixer:  # still the same song: hand back the plain source
                voice.source = mixer.release()
            return
        if voice.is_playing():
            voice.stop()  # leftover speech; the lock makes this rare
        loop = asyncio.get_running_loop()
        done = asyncio.Event()
        voice.play(SpeechSource(pcm), after=lambda error: loop.call_soon_threadsafe(done.set))
        try:
            await asyncio.wait_for(done.wait(), timeout=seconds + 2)
        except asyncio.TimeoutError:
            pass
