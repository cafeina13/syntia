"""Rejecting transcripts the audio can't have contained (assistant/stt.py)."""

import pytest

# These tests need the optional voice assistant packages; without them, skip
# instead of crashing the whole run.
pytest.importorskip("numpy", reason="voice assistant packages not installed (pip install -r assistant/requirements.txt)")
pytest.importorskip("faster_whisper", reason="voice assistant packages not installed (pip install -r assistant/requirements.txt)")



from assistant import stt


@pytest.mark.parametrize("text, seconds, ratio", [
    ("Şarkıyı geç.", 1.1, 0.6),
    ("Sesi kıs.", 1.4, 0.6),
    ("Herviz.", 1.5, 0.5),
    ("Kıs, kıs,", 1.0, 0.6),  # odd, but short and plausible
    ("Ses seviyesini 30'a ayarla ve şarkıyı geç.", 3.0, 0.9),
    ("", 1.0, 0.0),
])
def test_believable_transcripts_pass(text, seconds, ratio):
    assert stt.check_transcript(text, seconds, ratio) is None


def test_repetition_loop_is_rejected():
    loop = ", ".join(["Kıs"] * 75)
    assert stt.check_transcript(loop, 60.0, compression_ratio=23.4) == "repetitive"


def test_too_many_words_for_the_audio_is_rejected():
    # The real failure: 1 s of near-silence came back as a 15-word sentence.
    reason = stt.check_transcript(
        "Müzik çal, şarkıyı geç, sesi kıs, ses seviyesini 20 saniye ileri al, 2. dakikadan başlat.",
        0.96, compression_ratio=1.1)
    assert reason and "can't fit" in reason


def test_reciting_the_hint_is_rejected_even_when_it_would_fit():
    text = "Müzik çal, şarkıyı geç, sesi kıs."
    assert stt.check_transcript(text, 10.0) == "recited the vocabulary hint"
    assert stt.check_transcript("Müzik çal, şarkıyı geç.", 10.0) is None  # two phrases is a real request


def test_hint_is_built_from_its_phrases():
    assert stt.HINT == ("Syntia. Müzik çal, şarkıyı geç, sesi kıs, sesi aç, ses seviyesini 20'ye ayarla, "
                        "şarkıyı durdur, 30 saniye ileri al, 2. dakikadan başlat.")


@pytest.mark.parametrize("raw, command", [
    ("Jarvis, sesi kıs.", "sesi kıs."),
    ("Hey Jarvis, şarkıyı geç.", "şarkıyı geç."),
    ("Herviz, sesi kıs.", "sesi kıs."),
    ("H-Arviz şarkıyı geç", "şarkıyı geç"),
    ("H-Arbis.", ""),
    ("Carvis, müzik çal.", "müzik çal."),
    ("Sesi kıs.", "Sesi kıs."),  # nothing to strip
    ("Servis çağır.", "Servis çağır."),  # a real word that merely looks similar stays
    ("Şarkıyı geç, Jarvis.", "Şarkıyı geç, Jarvis."),  # only a LEADING wake word is removed
])
def test_strip_wake_word(raw, command):
    assert stt.strip_wake_word(raw) == command
