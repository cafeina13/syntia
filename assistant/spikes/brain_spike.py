"""
Spike 5: saved voice commands through Syntia's REAL brain, without playing anything.

For each WAV in recordings/commands/ (saved by `spike listen`):
    Whisper gate: is there real speech?   no -> skipped, no Gemini request spent
    yes -> ai.ask_ai(VoiceRequest, audio=the recording)  (Gemini listens itself)
The music functions are swapped for stubs that print what WOULD happen.

    .venv\\Scripts\\python.exe -m assistant.spikes.brain_spike [--pause 10]

Uses your Gemini free quota: one request per clip that passes the gate.
"""

import argparse
import asyncio
import os
import sys
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import ai
import music
from assistant.request import ReplyChannel, VoiceRequest
from assistant.stt import Transcriber

ROOT = Path(__file__).resolve().parent.parent.parent
STUBBED = ["play_music", "add_music", "clear_queue", "stop_music", "leave_voice", "now_playing",
           "pause_music", "resume_music", "skip_song", "shuffle_queue", "play_previous", "seek",
           "set_volume", "join_voice"]


class PrintChannel:
    # Where replies would be posted in Discord.
    async def send(self, content=None, **kwargs):
        print(f"      chat: {content}")

    def typing(self):
        class Nothing:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False
        return Nothing()


def stub_music():
    for name in STUBBED:
        async def would(request, *args, _name=name):
            print(f"      WOULD RUN: music.{_name}{args}")
        setattr(music, name, would)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pause", type=float, default=10, help="seconds between Gemini requests")
    args = parser.parse_args()

    stub_music()
    guild = SimpleNamespace(id=1, name="Test Server", voice_client=None)
    member = SimpleNamespace(id=int(os.getenv("SPIKE_USER_ID", "1")), display_name="cafeína", bot=False,
                             voice=None, guild_permissions=SimpleNamespace(manage_guild=False))
    stt = await asyncio.to_thread(Transcriber, device=os.getenv("STT_DEVICE", "cuda"))

    clips = sorted((ROOT / "recordings" / "commands").glob("*.wav"))
    print(f"{len(clips)} saved command(s)\n")
    spent = 0
    for path in clips:
        with wave.open(str(path)) as wav:
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        gate = await asyncio.to_thread(stt.transcribe, pcm)
        print(f"{path.name[:15]}  gate: {gate.text or '(nothing)'!r}"
              + (f"  [rejected: {gate.rejected}]" if gate.rejected else ""))
        if not gate.text:
            print("      skipped: no real speech, no Gemini request spent\n")
            continue
        request = VoiceRequest(guild, member, ReplyChannel(PrintChannel()))
        started = time.perf_counter()
        result = await ai.ask_ai(request, "", audio=path.read_bytes())
        spent += 1
        print(f"      result: {result['type']} in {time.perf_counter() - started:.1f}s\n")
        await asyncio.sleep(args.pause)
    print(f"Gemini requests spent: {spent}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
