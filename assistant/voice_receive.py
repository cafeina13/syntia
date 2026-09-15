"""
Hearing people in a Discord voice channel — by hand, on stock discord.py.

discord.py can only SEND audio. To RECEIVE it we undo, in reverse order, what
each speaker's Discord app did to their voice before it reached us:

    UDP packet
      -> RTP header       who (SSRC), order (sequence), time (timestamp)
      -> transport crypto XChaCha20-Poly1305 with the voice session's secret key
      -> DAVE             Discord's end-to-end encryption, per user, via `davey`
      -> Opus decode      48 kHz, stereo, 16-bit PCM — 20 ms (960 samples) a packet

Packets only carry an SSRC (a stream number), not a user. Discord tells us
which user owns which SSRC over the voice websocket (op 5 SPEAKING), so we
listen there too.

Everything here uses what discord.py 2.7 already has: a UDP socket listener,
the voice websocket hook, the secret key, and the DAVE session it maintains.
Two of those are private attributes (`voice._connection`, the websocket
`_hook`), so a discord.py update could move them — the tests will say so.
"""

import asyncio
import logging
import struct
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import discord
import nacl.secret
from discord.gateway import DiscordVoiceWebSocket
from discord.opus import OPUS_SILENCE, Decoder
from discord.voice_state import VoiceConnectionState

try:
    import davey
except ImportError:  # discord.py 2.7 needs it for voice anyway
    davey = None

log = logging.getLogger(__name__)

RTP_HEADER_SIZE = 12
RTP_VERSION = 2
OPUS_PAYLOAD_TYPE = 0x78  # 120: what Discord uses for voice audio
SAMPLES_PER_FRAME = 960  # 20 ms at 48 kHz
BYTES_PER_SAMPLE = 4  # 16-bit x 2 channels

# Called with (user_id, pcm, rtp_timestamp) for every decoded 20 ms of audio.
AudioCallback = Callable[[int, bytes, int], None]


@dataclass
class RtpPacket:
    ssrc: int
    sequence: int
    timestamp: int
    header: bytes  # authenticated but NOT encrypted (the AEAD "additional data")
    payload: bytes  # encrypted: [extension body] + opus, then a 4-byte nonce
    extended: bool


def parse_rtp(data: bytes) -> RtpPacket | None:
    # Split a UDP datagram into an RTP header and payload. Returns None for
    # anything that isn't voice audio: RTCP reports, IP discovery replies, junk.
    if len(data) < RTP_HEADER_SIZE + 4:
        return None
    if data[0] >> 6 != RTP_VERSION:
        return None
    if data[1] & 0x7F != OPUS_PAYLOAD_TYPE:  # RTCP types (200-204) land here too
        return None
    csrc_count = data[0] & 0x0F
    extended = bool(data[0] & 0x10)
    header_size = RTP_HEADER_SIZE + 4 * csrc_count
    if extended:
        # "rtpsize" modes: the 4-byte extension preamble (profile + length)
        # stays in the clear; the extension BODY is encrypted with the audio.
        header_size += 4
    if len(data) < header_size + 4:
        return None
    sequence, timestamp, ssrc = struct.unpack_from(">HII", data, 2)
    return RtpPacket(ssrc, sequence, timestamp, bytes(data[:header_size]),
                     bytes(data[header_size:]), extended)


def decrypt_transport(packet: RtpPacket, secret_key) -> bytes:
    # Mirror of discord.py's _encrypt_aead_xchacha20_poly1305_rtpsize: the last
    # 4 payload bytes are the nonce (zero-padded to 24), the header is the AAD.
    nonce = bytearray(24)
    nonce[:4] = packet.payload[-4:]
    box = nacl.secret.Aead(bytes(secret_key))
    plain = box.decrypt(packet.payload[:-4], packet.header, bytes(nonce))
    if packet.extended:
        # The preamble's last 2 bytes say how many 32-bit words of extension
        # body precede the audio. Skip them.
        words = struct.unpack_from(">H", packet.header, len(packet.header) - 2)[0]
        plain = plain[4 * words:]
    return plain


def is_newer(sequence: int, last: int) -> bool:
    # RTP sequence numbers wrap at 65536; "newer" means up to half the range ahead.
    return 0 < ((sequence - last) & 0xFFFF) < 0x8000


def track_speakers(ssrc_to_user: dict[int, int], msg: dict):
    # Discord announces "user X sends audio as SSRC Y" (op 5 SPEAKING) ONCE per
    # user, the first time they talk after we join — miss it and that user's
    # audio is anonymous for the rest of the call.
    op, data = msg.get("op"), msg.get("d") or {}
    if op == DiscordVoiceWebSocket.SPEAKING and "ssrc" in data and "user_id" in data:
        ssrc_to_user[int(data["ssrc"])] = int(data["user_id"])
    elif op == DiscordVoiceWebSocket.CLIENT_DISCONNECT and "user_id" in data:
        gone = int(data["user_id"])
        for ssrc in [s for s, u in ssrc_to_user.items() if u == gone]:
            ssrc_to_user.pop(ssrc)


class ListeningVoiceClient(discord.VoiceClient):
    """
    A VoiceClient that tracks who is who from the moment it connects.

    Use it with `await channel.connect(cls=ListeningVoiceClient)`. A plain
    VoiceClient only lets a receiver start watching later, by which time
    Discord may already have sent the one message that names each speaker.
    """

    def __init__(self, client, channel):
        self.ssrc_to_user: dict[int, int] = {}  # set BEFORE super(): connecting starts there
        super().__init__(client, channel)

    def create_connection_state(self):
        # discord.py's extension point: give the voice websocket our hook up front.
        return VoiceConnectionState(self, hook=self._on_voice_message)

    async def _on_voice_message(self, ws, msg: dict):
        track_speakers(self.ssrc_to_user, msg)


class VoiceReceiver:
    """Attach to a connected VoiceClient and get decoded audio per user."""

    def __init__(self, voice: discord.VoiceClient, on_audio: AudioCallback,
                 *, users: set[int] | None = None):
        self.voice = voice
        self.on_audio = on_audio
        self.users = users  # None = everyone; otherwise only these user IDs
        self.loop = asyncio.get_running_loop()
        # A ListeningVoiceClient already knows every speaker since it connected;
        # share its live map. Otherwise we can only learn speakers from now on.
        self.ssrc_to_user: dict[int, int] = getattr(voice, "ssrc_to_user", None)
        self.tracks_itself = self.ssrc_to_user is not None
        if not self.tracks_itself:
            self.ssrc_to_user = {}
        # Keyed by (ssrc, user): if Discord hands a leaver's SSRC to someone
        # new, they get a fresh decoder instead of the old one's state.
        self.decoders: dict[tuple[int, int], Decoder] = {}
        self.last_sequence: dict[tuple[int, int], int] = {}
        self.stats: Counter = Counter()
        self.cpu_seconds = 0.0  # time spent decrypting + decoding, for measuring
        self.user_seconds: dict[int, float] = {}  # audio delivered per user so far
        self.loss_log: list[tuple[int, float, str]] = []  # (user, seconds into their audio, why)
        self.loss_reasons: Counter = Counter()  # error messages from davey, counted
        self._previous_hook = None
        self._running = False

    # --- lifecycle -------------------------------------------------------------

    def start(self):
        state = self.voice._connection
        if not self.tracks_itself:
            self._previous_hook = state.hook
            # state.hook is used when the voice websocket (re)connects; the live
            # socket copied it at creation, so patch that one too.
            state.hook = self._websocket_hook
            self.voice.ws._hook = self._websocket_hook
        state.add_socket_listener(self._on_datagram)
        self._running = True

    def stop(self):
        if not self._running:
            return
        self._running = False
        state = self.voice._connection
        state.remove_socket_listener(self._on_datagram)
        if not self.tracks_itself:
            state.hook = self._previous_hook
            if self._previous_hook is None:
                self.voice.ws.__dict__.pop("_hook", None)  # back to the class no-op
            else:
                self.voice.ws._hook = self._previous_hook

    # --- who is who (plain VoiceClient only) ------------------------------------

    async def _websocket_hook(self, ws, msg: dict):
        track_speakers(self.ssrc_to_user, msg)
        if self._previous_hook is not None:
            await self._previous_hook(ws, msg)

    # --- the packet path -------------------------------------------------------

    def _on_datagram(self, data: bytes):
        # Runs on discord.py's socket-reader THREAD. Hop onto the event loop:
        # discord.py updates the DAVE keys there, so decrypting there too means
        # we never read keys while they're half-updated.
        if self._running:
            self.loop.call_soon_threadsafe(self.handle_datagram, data)

    def handle_datagram(self, data: bytes):
        started = time.perf_counter()
        try:
            self._process(data)
        except Exception:
            self.stats["errors"] += 1
            if self.stats["errors"] == 1:
                log.exception("voice receive: failed to process a packet")
        finally:
            self.cpu_seconds += time.perf_counter() - started

    def _process(self, data: bytes):
        packet = parse_rtp(data)
        if packet is None:
            self.stats["not_audio"] += 1
            return
        user_id = self.ssrc_to_user.get(packet.ssrc)
        if user_id is None:
            self.stats["unknown_ssrc"] += 1  # audio before its SPEAKING event
            return
        if self.users is not None and user_id not in self.users:
            self.stats["ignored_user"] += 1  # skipped BEFORE any crypto work
            return
        stream = (packet.ssrc, user_id)
        last = self.last_sequence.get(stream)
        if last is not None and not is_newer(packet.sequence, last):
            self.stats["late_or_duplicate"] += 1
            return
        self.last_sequence[stream] = packet.sequence

        opus = decrypt_transport(packet, self.voice._connection.secret_key)
        opus = self._decrypt_dave(user_id, opus)
        decoder = self.decoders.get(stream)
        if opus is None:
            if decoder is None:
                return  # nothing heard from them yet to guess from
            # Packet loss concealment: Opus invents a plausible 20 ms from what came
            # before, so the audio keeps its timing instead of a word losing a slice.
            pcm = decoder.decode(None, fec=False)
            self.stats["concealed"] += 1
        else:
            if decoder is None:
                decoder = self.decoders[stream] = Decoder()
            pcm = decoder.decode(opus, fec=False)
            self.stats["frames"] += 1
        self.user_seconds[user_id] = self.user_seconds.get(user_id, 0.0) + len(pcm) / BYTES_PER_SAMPLE / 48000
        self.on_audio(user_id, pcm, packet.timestamp)

    def _decrypt_dave(self, user_id: int, opus: bytes) -> bytes | None:
        state = self.voice._connection
        session = state.dave_session
        if state.dave_protocol_version == 0 or session is None:
            return opus  # this call isn't end-to-end encrypted
        if opus == OPUS_SILENCE:
            return opus  # silence frames are sent unencrypted
        if not session.ready:
            self._log_loss(user_id, "dave_not_ready")
            return None
        try:
            return session.decrypt(user_id, davey.MediaType.audio, opus)
        except Exception as error:
            self._log_loss(user_id, "dave_failed", error)
            return None

    def _log_loss(self, user_id: int, kind: str, error: Exception | None = None):
        self.stats[kind] += 1
        if error is not None:
            self.loss_reasons[f"{type(error).__name__}: {error}"[:120]] += 1
        if len(self.loss_log) < 1000:
            # When, in that user's audio timeline, so it lines up with a recording.
            self.loss_log.append((user_id, round(self.user_seconds.get(user_id, 0.0), 2), kind))
