"""
Spike 3a: how heavy is speaking with your Piper voice, and how does it sound?

Synthesizes typical Syntia replies with one or more voices, reports load time,
RAM, and how long each phrase takes to generate compared to how long it plays,
and saves WAVs to recordings/tts/<voice folder>/ so you can compare by ear.

    .venv\\Scripts\\python.exe -m assistant.spikes.tts_bench models/step_15000/tr_TR-merve-medium.onnx ...
"""

import argparse
import time
import wave
from pathlib import Path

from assistant.spikes.whisper_bench import peak_ram_mb
from assistant.tts import Voice

ROOT = Path(__file__).resolve().parent.parent.parent
PHRASES = [
    "Tamam, hallediyorum.",
    "Tarkan'dan Şımarık buldum, çalıyorum.",
    "Sesi yüzde otuza indirdim.",
    "Ses seviyesini 30'a ayarladım.",
    "Şarkıyı 2 dakika 15 saniye ileri aldım.",
    "Bunu bulamadım, başka bir şey dener misin?",
    "Çalma listesinden 24 şarkı sıraya eklendi, ilk şarkı başlıyor.",
]


def bench(model: Path):
    # Checkpoints share a filename, so each output folder is named after the
    # folder the voice sits in (e.g. step_15000).
    started = time.perf_counter()
    voice = Voice(model)
    print(f"=== {model.parent.name}: loaded in {time.perf_counter() - started:.1f}s "
          f"({voice.sample_rate} Hz), peak RAM {peak_ram_mb():.0f} MB")

    out = ROOT / "recordings" / "tts" / model.parent.name
    out.mkdir(parents=True, exist_ok=True)
    voice.synthesize("Isınma.")  # first call pays one-time setup costs

    total_audio = total_work = 0.0
    for i, text in enumerate(PHRASES, 1):
        started = time.perf_counter()
        pcm = voice.synthesize(text)
        work = time.perf_counter() - started
        audio = voice.seconds(pcm)
        total_audio, total_work = total_audio + audio, total_work + work
        path = out / f"{i:02d}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(voice.sample_rate)
            wav.writeframes(pcm)
        print(f"  {path.name}: {audio:.1f}s of speech in {work * 1000:.0f} ms  |  {text}")

    print(f"=> {total_work / total_audio:.3f}s of work per second of speech "
          f"({total_audio / total_work:.0f}x faster than real time), peak RAM {peak_ram_mb():.0f} MB")
    print(f"Listen: {out}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="paths to Piper .onnx voices")
    args = parser.parse_args()
    for model in args.models:
        bench(Path(model))


if __name__ == "__main__":
    main()
