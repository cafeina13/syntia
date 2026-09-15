"""
From one speaker's raw Discord audio to finished voice commands.

    Discord audio, 48 kHz stereo, 20 ms chunks
      -> Resampler: 16 kHz mono (what wake word models and Whisper expect)
      -> SpeakerListener, a small state machine per speaker:

    LISTENING   the wake word model scores every 80 ms
       | wake word
    JUST_WOKE   did the command start in the same breath? ("hey jarvis, şarkıyı geç")
       | speech within 0.8 s          | nothing for 0.8 s -> chime
       v                              v
    CAPTURING  <---- speech --------  WAITING  (up to 4 s, else give up)
       | 0.8 s without speech, or 10 s total
       v
    utterance -> back to LISTENING

Discord sends NOTHING while someone is silent, so silence can't be spotted by
looking at audio — it's measured with a clock in tick(), called regularly.

The wake word model fires a moment AFTER the word (later still on a weak
score), and a quick "hey jarvis, sesi kıs" has started by then. So the last
~0.3 s of audio is always kept and becomes the start of every capture; the
transcriber strips the bit of "jarvis" that comes along.

No Discord code here: events go out through a callback, which keeps this
testable with a fake clock.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi

WAKE_FRAME = 1280  # 80 ms at 16 kHz: the chunk size openWakeWord works in


class Resampler:
    """48 kHz stereo int16 -> 16 kHz mono int16, keeping filter state between chunks."""

    def __init__(self):
        # Low-pass below the new Nyquist frequency (8 kHz) so high sounds don't
        # fold back as noise when we keep only every 3rd sample.
        self.taps = firwin(63, 7200, fs=48000)
        self.state = lfilter_zi(self.taps, 1.0) * 0
        self.phase = 0  # which sample of the next chunk is "every 3rd"

    def process(self, pcm48_stereo: bytes) -> np.ndarray:
        stereo = np.frombuffer(pcm48_stereo, dtype=np.int16).reshape(-1, 2)
        mono = stereo.mean(axis=1)
        filtered, self.state = lfilter(self.taps, 1.0, mono, zi=self.state)
        out = filtered[self.phase::3]
        self.phase = (self.phase - len(mono)) % 3
        return np.clip(out, -32768, 32767).astype(np.int16)


@dataclass
class ListenerConfig:
    wake_word: str = "hey_jarvis"
    wake_threshold: float = 0.5
    vad_threshold: float = 0.5
    same_breath_seconds: float = 0.8  # speech this soon after the wake word = one breath
    ignore_after_wake: float = 0.3  # the tail of "jarvis" itself is still speech
    wait_seconds: float = 4.0  # after the chime, how long to wait for a command
    end_silence_seconds: float = 0.8  # this much silence ends a command
    max_command_seconds: float = 10.0
    cooldown_seconds: float = 1.0  # no new wake word right after a command
    # Audio from just before the trigger to keep (4 frames). 0.48 s dragged in so much
    # of "jarvis" that Whisper prefixed junk ("Bir...", "Sürpriz...").
    pre_roll_seconds: float = 0.32


# Event kinds sent to the callback: (kind, listener, details)
WAKE, CHIME, UTTERANCE, GAVE_UP = "wake", "chime", "utterance", "gave_up"


class SpeakerListener:
    LISTENING, JUST_WOKE, WAITING, CAPTURING = "listening", "just_woke", "waiting", "capturing"

    def __init__(self, user_id: int, wake_model, vad, on_event: Callable,
                 config: ListenerConfig | None = None, clock=time.monotonic):
        self.user_id = user_id
        self.wake = wake_model  # openwakeword Model (one per speaker: it keeps state)
        self.vad = vad  # openwakeword VAD (Silero), also stateful
        self.on_event = on_event
        self.config = config or ListenerConfig()
        self.clock = clock
        self.resampler = Resampler()
        self.state = self.LISTENING
        self.pending = np.zeros(0, dtype=np.int16)  # 16 kHz audio not yet a full 80 ms frame
        # The last few frames heard while LISTENING, including the one that triggers.
        self.recent = deque(maxlen=max(1, round(self.config.pre_roll_seconds / 0.08)))
        self.captured: list[np.ndarray] = []
        self.woke_at = self.wait_started = self.capture_started = self.last_speech = 0.0
        self.cooldown_until = 0.0
        # Measurements for the spike report.
        self.cpu_seconds = 0.0
        self.audio_seconds = 0.0
        self.best_idle_score = 0.0  # highest wake score that did NOT trigger: tuning aid
        # Optional debugging: every 16 kHz frame the pipeline heard, and the wake
        # score for each LISTENING frame as (seconds into the audio, score).
        self.debug = False
        self.debug_frames: list[np.ndarray] = []
        self.score_log: list[tuple[float, float]] = []

    # --- input -------------------------------------------------------------------

    def feed(self, pcm48_stereo: bytes):
        started = time.perf_counter()
        audio = self.resampler.process(pcm48_stereo)
        self.pending = np.concatenate([self.pending, audio])
        while len(self.pending) >= WAKE_FRAME:
            frame, self.pending = self.pending[:WAKE_FRAME], self.pending[WAKE_FRAME:]
            self.audio_seconds += WAKE_FRAME / 16000
            self._frame(frame)
        self.cpu_seconds += time.perf_counter() - started

    def _frame(self, frame: np.ndarray):
        now = self.clock()
        config = self.config
        if self.debug and len(self.debug_frames) < 16000 * 600 // WAKE_FRAME:  # cap: 10 minutes
            self.debug_frames.append(frame)
        if self.state == self.LISTENING:
            self.recent.append(frame)
            score = float(self.wake.predict(frame)[config.wake_word])
            if self.debug:
                self.score_log.append((self.audio_seconds, score))
            if score >= config.wake_threshold and now >= self.cooldown_until:
                self.state, self.woke_at = self.JUST_WOKE, now
                self.captured = list(self.recent)  # pre-roll: the command may have begun
                self.recent.clear()
                self.wake.reset()
                self.vad.reset_states()
                self.on_event(WAKE, self, {"score": score})
            else:
                self.best_idle_score = max(self.best_idle_score, score)
            return

        speaking = float(self.vad.predict(frame, frame_size=640)) >= config.vad_threshold
        if self.state == self.JUST_WOKE:
            # Always KEEP the audio: in "hey jarvis, şarkıyı geç" the command can
            # start inside this window. Only the decision to start capturing waits,
            # because the tail of "jarvis" itself is still speech.
            self.captured.append(frame)
            if speaking and now - self.woke_at >= config.ignore_after_wake:
                self._start_capture(now)
        elif self.state == self.WAITING:
            self.captured.append(frame)
            if speaking:
                self._start_capture(now)
        elif self.state == self.CAPTURING:
            self.captured.append(frame)
            if speaking:
                self.last_speech = now

    def _start_capture(self, now: float):
        self.state, self.capture_started, self.last_speech = self.CAPTURING, now, now

    # --- time --------------------------------------------------------------------

    def tick(self):
        now = self.clock()
        config = self.config
        if self.state == self.JUST_WOKE and now - self.woke_at >= config.same_breath_seconds:
            self.state, self.wait_started = self.WAITING, now
            self.captured = []  # just silence/breath so far
            self.on_event(CHIME, self, {})
        elif self.state == self.WAITING and now - self.wait_started >= config.wait_seconds:
            self._reset(now)
            self.on_event(GAVE_UP, self, {})
        elif self.state == self.CAPTURING and (
                now - self.last_speech >= config.end_silence_seconds
                or now - self.capture_started >= config.max_command_seconds):
            audio = np.concatenate(self.captured) if self.captured else np.zeros(0, dtype=np.int16)
            ended_by = "silence" if now - self.last_speech >= config.end_silence_seconds else "time limit"
            self._reset(now)
            self.on_event(UTTERANCE, self, {"audio": audio, "ended_by": ended_by, "finished_at": now})

    def _reset(self, now: float):
        self.state = self.LISTENING
        self.captured = []
        self.cooldown_until = now + self.config.cooldown_seconds
        self.wake.reset()
        self.vad.reset_states()
