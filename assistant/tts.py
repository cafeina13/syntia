"""
Syntia's voice: text -> speech with a Piper ONNX voice, played into Discord.

Piper turns text into phonemes (espeak-ng, Turkish rules), then a small neural
network turns those into audio: 16-bit mono at the voice's own sample rate
(22050 Hz for tr_TR-merve-medium). Discord wants 48 kHz stereo, so FFmpeg —
already used for music — converts it on the way out.
"""

import io
from pathlib import Path

import discord
from piper import PiperVoice


class Voice:
    def __init__(self, model_path: str | Path):
        # Loads the .onnx and its .onnx.json (looked up next to it).
        self.piper = PiperVoice.load(str(model_path))
        self.sample_rate = self.piper.config.sample_rate

    def synthesize(self, text: str) -> bytes:
        # Raw 16-bit mono PCM at self.sample_rate. Piper splits text into
        # sentences and yields a chunk for each; we join them.
        return b"".join(chunk.audio_int16_bytes for chunk in self.piper.synthesize(text))

    def seconds(self, pcm: bytes) -> float:
        return len(pcm) / 2 / self.sample_rate

    def to_discord(self, pcm: bytes) -> discord.AudioSource:
        # Feed the raw audio to FFmpeg through a pipe; FFmpeg resamples to the
        # 48 kHz stereo PCM discord.py expects.
        return discord.FFmpegPCMAudio(
            io.BytesIO(pcm), pipe=True,
            before_options=f"-f s16le -ar {self.sample_rate} -ac 1",
        )
