"""
Speech-to-text for voice commands: faster-whisper.

Whisper always encodes a full 30-second window, so even a 1-second command
costs one whole window. On this laptop's CPU (medium, int8, 4 threads) that's
~3.3 s per command — too slow to feel responsive — so the GPU is the default,
using ~1 GB of VRAM (int8_float16). Pass device="cpu" to keep the GPU free.

Whisper also invents text when there's little real speech: it loops ("kıs, kıs,
kıs, ...") or recites the vocabulary hint back. The settings below stop the
loops, and check_transcript() throws out text the audio can't have contained —
a rejected transcript means "didn't understand", never a command.
"""

import glob
import os
import re
import sys
import time
from dataclasses import dataclass


def _expose_nvidia_dlls():
    # CTranslate2 needs cuBLAS on the GPU. The `nvidia-cublas-cu12` pip package
    # ships the DLLs inside site-packages, where Windows won't look by itself.
    if os.name != "nt":
        return
    for folder in glob.glob(os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "*", "bin")):
        os.add_dll_directory(folder)
        os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")


_expose_nvidia_dlls()

import numpy as np  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

# A sample of the command vocabulary, given to Whisper up front. On a sick-voice
# test recording it cut word errors from 44% to 12% — but on a near-silent clip
# Whisper recited it back, hence the echo check below.
HINT_PHRASES = ["Syntia", "Müzik çal", "şarkıyı geç", "sesi kıs", "sesi aç", "ses seviyesini 20'ye ayarla",
                "şarkıyı durdur", "30 saniye ileri al", "2. dakikadan başlat"]
HINT = HINT_PHRASES[0] + ". " + ", ".join(HINT_PHRASES[1:]) + "."

SETTINGS = {
    "cuda": {"compute_type": "int8_float16"},  # int8 weights, float16 math: ~1 GB VRAM
    "cpu": {"compute_type": "int8", "cpu_threads": 4},
}

MAX_WORDS_PER_SECOND = 4.5  # fast Turkish speech is ~3-4 words a second

# Captures start a little before the wake word fired, so its tail often comes
# along. How Whisper has written "(hey) jarvis" so far: Jarvis, Herviz, H-Arviz,
# H-Arbis, Sintiye... Swap this when the wake word changes to "hey syntia".
WAKE_ECHO = re.compile(r"^\W*(?:hey\W+)?(?:h\W*)?[jcçhş]?[ae]r[vb][iıy][sz]\b\W*", re.IGNORECASE)


def strip_wake_word(text: str) -> str:
    return WAKE_ECHO.sub("", text, count=1).strip()
MAX_COMPRESSION_RATIO = 2.4  # Whisper's own "this text is suspiciously repetitive" line


@dataclass
class Transcript:
    text: str  # "" when nothing usable was heard
    rejected: str | None = None  # why the raw text was thrown out, if it was
    raw: str = ""


def check_transcript(text: str, seconds: float, compression_ratio: float = 0.0) -> str | None:
    # Returns a reason to reject, or None if the text is believable for the audio.
    words = re.findall(r"\w+", text)
    if not words:
        return None
    if compression_ratio > MAX_COMPRESSION_RATIO:
        return "repetitive"
    if len(words) > max(3.0, MAX_WORDS_PER_SECOND * seconds):
        return f"{len(words)} words can't fit in {seconds:.1f}s of audio"
    lowered = text.lower()
    echoed = sum(1 for phrase in HINT_PHRASES[1:] if phrase.lower() in lowered)
    if echoed >= 3:
        return "recited the vocabulary hint"
    return None


class Transcriber:
    def __init__(self, model: str = "medium", device: str = "cuda"):
        started = time.perf_counter()
        self.device = device
        self.model = WhisperModel(model, device=device, **SETTINGS[device])
        # The first transcription pays one-time GPU setup costs; pay them now,
        # not on someone's first command.
        self.transcribe(np.zeros(16000, dtype=np.float32))
        self.load_seconds = time.perf_counter() - started

    def transcribe(self, audio: np.ndarray) -> Transcript:
        # audio: 16 kHz mono, int16 or float32 in [-1, 1].
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768
        segments, _ = self.model.transcribe(
            audio, language="tr", initial_prompt=HINT, vad_filter=True,
            beam_size=1,
            # One decoding pass. Whisper's default retries at higher "temperatures"
            # made a near-silent clip recite the whole hint back, and cost seconds.
            temperature=0.0,
            no_repeat_ngram_size=3,  # can't repeat a 3-token phrase: kills "kıs, kıs, kıs..."
            max_new_tokens=48,  # a command is a sentence, not a paragraph
        )
        segments = list(segments)
        raw = " ".join(segment.text.strip() for segment in segments).strip()
        ratio = max((segment.compression_ratio for segment in segments), default=0.0)
        reason = check_transcript(raw, len(audio) / 16000, ratio)
        text = "" if reason else strip_wake_word(raw)
        return Transcript(text=text, rejected=reason, raw=raw)
