"""
Everything Syntia SAYS as a voice assistant, in one place.

Voice is kept to a minimum on purpose: the main actions get a short line, the
rest you hear for yourself (the song skips, the volume drops). Edit freely —
None means "say nothing".
"""

ACKNOWLEDGED = "Tamam, hallediyorum."  # the AI decided to play or queue something
NOW_PLAYING = "{title} çalıyorum."  # that request's first track started
SLOW = "Bir saniye…"  # the answer is taking a while
BUSY = "Şu an çok yoğunum, birazdan tekrar dene."  # every AI model is busy
FAILED = "Bir sorun çıktı."  # any other AI error
NOT_UNDERSTOOD = "Anlayamadım."  # no real speech in the recording
NOT_SET_UP = "Sesli komutlar için yapay zekâ ayarlı değil."  # no Gemini key
ASSISTANT_OFF = "Tamam, dinlemeyi bırakıyorum."  # "asistanı kapat" by voice

# Per tool: what to say when the AI calls it. Missing = say nothing.
TOOL_LINES = {
    "play_music": ACKNOWLEDGED,
    "add_to_queue": ACKNOWLEDGED,
}


def when_decided(result: dict) -> str | None:
    # Spoken as soon as the AI answers, before any tool runs.
    if result.get("type") == "tools":
        for call in result.get("calls", []):
            line = TOOL_LINES.get(call.get("name"))
            if line:
                return line
        return None
    if result.get("type") == "text":
        return (result.get("text") or "").strip() or NOT_UNDERSTOOD
    return None


def for_error(result: dict) -> str | None:
    if result.get("type") != "error":
        return None
    return {"busy": BUSY, "not_set_up": NOT_SET_UP}.get(result.get("reason"), FAILED)
