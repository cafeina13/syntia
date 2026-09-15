"""The wake word -> capture state machine (assistant/listener.py), on a fake clock."""

import numpy as np
import pytest

from assistant import listener as ls

CHUNK = 960  # samples per 20 ms Discord chunk at 48 kHz


def tone48(frequency, seconds=0.2, amplitude=10000):
    t = np.arange(int(48000 * seconds)) / 48000
    mono = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
    return np.repeat(mono, 2)  # stereo, interleaved


# --- resampling --------------------------------------------------------------------


def test_resampler_output_length_and_continuity():
    r = ls.Resampler()
    total = sum(len(r.process(tone48(440, 0.02).tobytes())) for _ in range(50))
    assert total == 50 * 320  # 1 s at 48 kHz -> 1 s at 16 kHz


def test_resampler_handles_odd_chunk_sizes():
    r = ls.Resampler()
    audio = tone48(440, 1.0)
    pieces = [audio[:2 * 1000], audio[2 * 1000:2 * 1961], audio[2 * 1961:]]  # 1000, 961, rest samples
    total = sum(len(r.process(p.tobytes())) for p in pieces)
    assert abs(total - 16000) <= 1


def test_resampler_keeps_speech_and_removes_what_16khz_cannot_hold():
    def level(frequency):
        r = ls.Resampler()
        out = r.process(tone48(frequency, 0.5).tobytes())[2000:]  # skip filter warm-up
        return np.sqrt(np.mean(out.astype(float) ** 2))

    assert level(440) > 6000  # a voice-range tone passes (~7070 RMS expected)
    assert level(12000) < 300  # would alias into noise without the low-pass


# --- the state machine -------------------------------------------------------------


class FakeWake:
    def __init__(self):
        self.score = 0.0
        self.resets = 0

    def predict(self, frame):
        return {"hey_jarvis": self.score}

    def reset(self):
        self.resets += 1


class FakeVad:
    def __init__(self):
        self.speech = False

    def predict(self, frame, frame_size=480):
        return 0.9 if self.speech else 0.1

    def reset_states(self):
        pass


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


@pytest.fixture
def rig():
    clock, wake, vad, events = Clock(), FakeWake(), FakeVad(), []
    listener = ls.SpeakerListener(42, wake, vad, lambda kind, who, info: events.append((kind, info)),
                                  clock=clock)

    def frames(n=1, *, speech=None, score=None):
        # Feed n 80 ms frames (4 Discord chunks each), advancing the clock.
        if speech is not None:
            vad.speech = speech
        if score is not None:
            wake.score = score
        for _ in range(n):
            for _ in range(4):
                listener.feed(np.zeros(CHUNK * 2, dtype=np.int16).tobytes())
            clock.now += 0.08
            listener.tick()

    def wait(seconds):
        # Discord sends nothing during silence: only time passes, and tick() runs.
        steps = int(round(seconds / 0.1))
        for _ in range(steps):
            clock.now += 0.1
            listener.tick()

    return listener, wake, vad, clock, events, frames, wait


def kinds(events):
    return [kind for kind, _ in events]


def test_quiet_room_never_wakes(rig):
    listener, wake, vad, clock, events, frames, wait = rig
    frames(50, score=0.3, speech=True)
    assert events == [] and listener.state == listener.LISTENING
    assert listener.best_idle_score == pytest.approx(0.3)


def test_one_breath_command(rig):
    listener, wake, vad, clock, events, frames, wait = rig
    frames(1, score=0.9)  # "hey jarvis"
    assert kinds(events) == [ls.WAKE] and listener.state == listener.JUST_WOKE
    frames(4, score=0.0, speech=True)  # "şarkıyı geç" straight after
    assert listener.state == listener.CAPTURING
    frames(10, speech=True)
    wait(1.0)  # they stop talking: Discord goes silent
    assert kinds(events) == [ls.WAKE, ls.UTTERANCE]
    info = events[-1][1]
    assert info["ended_by"] == "silence" and len(info["audio"]) >= 10 * ls.WAKE_FRAME
    assert listener.state == listener.LISTENING


def test_audio_right_after_the_wake_word_is_kept(rig):
    # Regression: the first 0.3 s after waking used to be thrown away, cutting
    # the start off one-breath commands ("hey jarvis, şarkıyı geç" -> "...kıyı geç").
    listener, wake, vad, clock, events, frames, wait = rig
    frames(1, score=0.9)
    frames(3, score=0.0, speech=True)  # inside the 0.3 s window
    frames(2, speech=True)
    wait(1.0)
    assert kinds(events) == [ls.WAKE, ls.UTTERANCE]
    # all 5 frames after the wake, plus the triggering frame from the pre-roll
    assert len(events[-1][1]["audio"]) == 6 * ls.WAKE_FRAME


def test_pre_roll_keeps_audio_from_before_the_trigger(rig):
    # Regression: a weak, late trigger (0.53) meant "sesi" of "sesi kıs" was
    # already gone. The last ~0.3 s before the trigger now starts the capture.
    listener, wake, vad, clock, events, frames, wait = rig
    frames(20, score=0.1)  # talking before the trigger
    frames(1, score=0.9)
    frames(4, score=0.0, speech=True)
    wait(1.0)
    audio = events[-1][1]["audio"]
    pre_roll_frames = round(listener.config.pre_roll_seconds / 0.08)
    assert pre_roll_frames == 4
    assert len(audio) == (pre_roll_frames + 4) * ls.WAKE_FRAME  # not all 21 earlier frames


def test_pre_roll_does_not_leak_into_the_next_command(rig):
    listener, wake, vad, clock, events, frames, wait = rig
    frames(10, score=0.1)
    frames(1, score=0.9)
    frames(4, score=0.0, speech=True)
    wait(2.5)  # command ends after 0.8 s of silence, then the 1 s cooldown
    frames(1, score=0.9)  # second wake right away: pre-roll is just this frame
    frames(4, score=0.0, speech=True)
    wait(1.0)
    assert kinds(events) == [ls.WAKE, ls.UTTERANCE, ls.WAKE, ls.UTTERANCE]
    assert len(events[-1][1]["audio"]) == 5 * ls.WAKE_FRAME


def test_wake_then_chime_then_command(rig):
    listener, wake, vad, clock, events, frames, wait = rig
    frames(1, score=0.9)
    wake.score = 0.0
    wait(1.0)  # silent after the wake word
    assert kinds(events) == [ls.WAKE, ls.CHIME] and listener.state == listener.WAITING
    wait(2.0)
    frames(8, speech=True)  # the command, 2 s after the chime
    wait(1.0)
    assert kinds(events) == [ls.WAKE, ls.CHIME, ls.UTTERANCE]


def test_chime_then_nothing_gives_up(rig):
    listener, wake, vad, clock, events, frames, wait = rig
    frames(1, score=0.9)
    wake.score = 0.0
    wait(1.0)
    wait(4.5)
    assert kinds(events) == [ls.WAKE, ls.CHIME, ls.GAVE_UP]
    assert listener.state == listener.LISTENING


def test_tail_of_the_wake_word_is_not_a_command(rig):
    listener, wake, vad, clock, events, frames, wait = rig
    frames(1, score=0.9)
    frames(2, score=0.0, speech=True)  # "...vis" still voiced, within 0.3 s
    vad.speech = False
    assert listener.state == listener.JUST_WOKE
    wait(1.0)
    assert kinds(events) == [ls.WAKE, ls.CHIME]


def test_long_command_is_cut_at_the_limit(rig):
    listener, wake, vad, clock, events, frames, wait = rig
    frames(1, score=0.9)
    frames(5, score=0.0, speech=True)
    frames(130, speech=True)  # talking for 10+ seconds
    assert kinds(events) == [ls.WAKE, ls.UTTERANCE]
    assert events[-1][1]["ended_by"] == "time limit"


def test_cooldown_blocks_an_immediate_second_wake(rig):
    listener, wake, vad, clock, events, frames, wait = rig
    frames(1, score=0.9)
    frames(5, score=0.9, speech=True)  # wake model is ignored while capturing
    vad.speech = False
    wait(1.0)
    frames(1, score=0.9)  # still inside the 1 s cooldown
    assert kinds(events) == [ls.WAKE, ls.UTTERANCE]
    wait(1.0)
    frames(1, score=0.9)
    assert kinds(events) == [ls.WAKE, ls.UTTERANCE, ls.WAKE]


def test_models_are_reset_between_commands(rig):
    listener, wake, vad, clock, events, frames, wait = rig
    frames(1, score=0.9)
    assert wake.resets == 1  # right after triggering, so it doesn't re-fire
    frames(5, score=0.0, speech=True)
    vad.speech = False
    wait(1.0)
    assert wake.resets == 2


def test_measures_cpu_and_audio(rig):
    listener, wake, vad, clock, events, frames, wait = rig
    frames(10)
    assert listener.audio_seconds == pytest.approx(0.8)
    assert listener.cpu_seconds > 0
