"""
Spike 2: how heavy is Turkish speech-to-text on this machine's CPU?

Measures, per configuration: model load time, RAM, how long transcription
takes per second of audio (the "real-time factor"), CPU use, and the text.

    .venv\\Scripts\\python.exe -m assistant.spikes.whisper_bench
    .venv\\Scripts\\python.exe -m assistant.spikes.whisper_bench --dataset 20

Inputs: every WAV in recordings/ (from spike 1), plus optionally N clips from
a dataset folder with a `metadata.csv` of `file.wav|text` lines, scored by word
error rate. Careful: if that dataset was itself transcribed by Whisper, the
scores are biased toward the model that made it.
"""

import argparse
import ctypes
import random
import re
import time
from ctypes import wintypes
from pathlib import Path

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET = Path(r"\\wsl$\Ubuntu\home\cafeina\TTS\data\merve")
SAMPLE_RATE = 16000  # what Whisper works in
# The vocabulary hint now lives with the real transcriber. (It was tuned on the
# spike 1 recording, so scores on that same file are optimistic.)
from assistant.stt import HINT  # noqa: E402


def peak_ram_mb() -> float:
    # Peak working set of this process (Windows), in MB.
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Declare types: without them ctypes squeezes the 64-bit handle into an int.
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return float("nan")
    return counters.PeakWorkingSetSize / 1024**2


def normalize(text: str) -> list[str]:
    # Turkish-aware lowercasing (I -> ı, İ -> i), no punctuation.
    text = text.replace("I", "ı").replace("İ", "i").lower()
    return re.findall(r"[a-zçğıöşü0-9]+", text)


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    # (edit distance in words, reference word count) -> WER = errors / words.
    ref, hyp = normalize(reference), normalize(hypothesis)
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[len(hyp)], len(ref)


def load_inputs(dataset_clips: int):
    items = []  # (label, audio, reference text or None)
    for path in sorted((ROOT / "recordings").glob("*.wav")):
        items.append((f"recording {path.name}", decode_audio(str(path), SAMPLE_RATE), None))
    if dataset_clips:
        lines = (DATASET / "metadata.csv").read_text(encoding="utf-8").splitlines()
        pairs = [line.split("|", 1) for line in lines if "|" in line]
        for name, text in random.Random(42).sample(pairs, dataset_clips):
            audio = decode_audio(str(DATASET / "wavs" / name), SAMPLE_RATE)
            items.append((f"dataset {name}", audio, text))
    return items


def run(model_name: str, compute: str, threads: int, beam: int, hint: str | None, items,
        show_text: bool):
    started = time.perf_counter()
    model = WhisperModel(model_name, device="cpu", compute_type=compute, cpu_threads=threads)
    load_seconds = time.perf_counter() - started

    audio_total = wall_total = cpu_total = 0.0
    errors = words = 0
    for label, audio, reference in items:
        seconds = len(audio) / SAMPLE_RATE
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        segments, _ = model.transcribe(audio, language="tr", beam_size=beam, vad_filter=True,
                                       initial_prompt=hint)
        text = " ".join(segment.text.strip() for segment in segments)  # generator: work happens here
        wall = time.perf_counter() - wall_start
        cpu = time.process_time() - cpu_start
        audio_total, wall_total, cpu_total = audio_total + seconds, wall_total + wall, cpu_total + cpu
        line = f"    {label}: {seconds:.1f}s audio -> {wall:.1f}s"
        if reference is not None:
            e, n = word_errors(reference, text)
            errors, words = errors + e, words + n
            line += f", WER {e / max(n, 1):.0%}"
        print(line)
        if show_text:
            if reference is not None:
                print(f"      expected: {reference}")
            print(f"      heard:    {text}")

    print(f"  => load {load_seconds:.1f}s | peak RAM {peak_ram_mb():.0f} MB | "
          f"{wall_total / audio_total:.2f}s per second of audio | "
          f"CPU {cpu_total / wall_total * 100:.0f}% of one core while working"
          + (f" | WER {errors / words:.1%} over {words} words" if words else ""))
    return wall_total / audio_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium")
    parser.add_argument("--compute", default="int8")
    parser.add_argument("--threads", type=int, nargs="+", default=[4])
    parser.add_argument("--beam", type=int, nargs="+", default=[1])
    parser.add_argument("--no-hint", action="store_true", help="transcribe without the vocabulary hint")
    parser.add_argument("--dataset", type=int, default=0, help="also test N dataset clips")
    parser.add_argument("--quiet", action="store_true", help="don't print transcripts")
    args = parser.parse_args()

    items = load_inputs(args.dataset)
    total = sum(len(audio) for _, audio, _ in items) / SAMPLE_RATE
    print(f"{len(items)} input(s), {total:.0f}s of audio. Model {args.model} ({args.compute}) on CPU.\n")
    first = True
    for threads in args.threads:
        for beam in args.beam:
            hint = None if args.no_hint else HINT
            print(f"[threads={threads} beam={beam} hint={'off' if hint is None else 'on'}]")
            run(args.model, args.compute, threads, beam, hint, items, show_text=first and not args.quiet)
            first = False
            print()


if __name__ == "__main__":
    main()
