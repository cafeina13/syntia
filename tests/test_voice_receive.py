"""
The hand-written voice receiver (assistant/voice_receive.py).

Packets are built the way discord.py's own sender encrypts them, so if our
decryption drifts from theirs these tests fail. DAVE is faked — a real MLS
group needs Discord's servers — so the live spike covers that part.
"""

import array
import math
import os
import struct
from types import SimpleNamespace

import nacl.secret
import pytest
from discord.gateway import DiscordVoiceWebSocket
from discord.opus import OPUS_SILENCE, Encoder
from discord.voice_client import VoiceClient

from assistant import voice_receive as vr

KEY = list(os.urandom(32))
USER, OTHER = 111, 222
SSRC, OTHER_SSRC = 5000, 6000


def opus_tone(frequency=440, frame=0):
    # One real 20 ms Opus frame of a sine tone, so decoding has something to find.
    samples = array.array("h", (
        int(12000 * math.sin(2 * math.pi * frequency * (frame * 960 + i) / 48000))
        for i in range(960) for _ in range(2)
    ))
    return Encoder().encode(samples.tobytes(), 960)


def discord_py_packet(opus: bytes, *, ssrc=SSRC, sequence=1, timestamp=960, nonce=1):
    # Exactly what discord.py sends: 12-byte header, then its own encryption.
    header = bytearray(12)
    header[0], header[1] = 0x80, 0x78
    struct.pack_into(">HII", header, 2, sequence, timestamp, ssrc)
    sender = SimpleNamespace(secret_key=KEY, _incr_nonce=nonce,
                             checked_add=lambda *args: None)
    return VoiceClient._encrypt_aead_xchacha20_poly1305_rtpsize(sender, bytes(header), opus)


def client_packet_with_extension(opus: bytes, *, ssrc=SSRC, sequence=1, timestamp=960):
    # What a Discord APP sends: an RTP header extension (one 32-bit word here).
    # The preamble is authenticated in the clear; the body is encrypted with the audio.
    header = bytearray(12)
    header[0], header[1] = 0x90, 0x78  # 0x10 = extension present
    struct.pack_into(">HII", header, 2, sequence, timestamp, ssrc)
    header += struct.pack(">HH", 0xBEEF, 1)
    body = b"\x10\xff\x00\x00" + opus
    nonce = bytearray(24)
    nonce[:4] = struct.pack(">I", 7)
    sealed = nacl.secret.Aead(bytes(KEY)).encrypt(body, bytes(header), bytes(nonce)).ciphertext
    return bytes(header) + sealed + nonce[:4]


# --- parsing ---------------------------------------------------------------------


def test_parse_audio_packet():
    packet = vr.parse_rtp(discord_py_packet(b"opus", sequence=42, timestamp=1234))
    assert (packet.ssrc, packet.sequence, packet.timestamp) == (SSRC, 42, 1234)
    assert len(packet.header) == 12 and not packet.extended


@pytest.mark.parametrize("data", [
    b"",
    b"\x80\x78" + bytes(10),  # too short to hold a nonce
    bytes([0x81, 0xC9]) + bytes(30),  # RTCP receiver report (type 201)
    b"\x00\x02\x00\x46" + bytes(70),  # IP discovery reply
    bytes([0x80, 0x60]) + bytes(30),  # some other payload type
])
def test_parse_ignores_non_audio(data):
    assert vr.parse_rtp(data) is None


@pytest.mark.parametrize("sequence, last, newer", [
    (2, 1, True), (1, 1, False), (1, 2, False), (0, 65535, True), (65535, 0, False),
])
def test_sequence_wraparound(sequence, last, newer):
    assert vr.is_newer(sequence, last) is newer


# --- decryption ----------------------------------------------------------------


def test_decrypts_what_discord_py_encrypts():
    opus = opus_tone()
    packet = vr.parse_rtp(discord_py_packet(opus))
    assert vr.decrypt_transport(packet, KEY) == opus


def test_strips_the_encrypted_header_extension():
    opus = opus_tone()
    packet = vr.parse_rtp(client_packet_with_extension(opus))
    assert packet.extended and len(packet.header) == 16
    assert vr.decrypt_transport(packet, KEY) == opus


def test_wrong_key_fails_loudly():
    packet = vr.parse_rtp(discord_py_packet(opus_tone()))
    with pytest.raises(Exception):
        vr.decrypt_transport(packet, list(os.urandom(32)))


# --- the receiver ----------------------------------------------------------------


class FakeDave:
    def __init__(self, ready=True, fail=False):
        self.ready, self.fail, self.calls = ready, fail, []

    def decrypt(self, user_id, media_type, packet):
        self.calls.append(user_id)
        if self.fail:
            raise RuntimeError("no key for user")
        return packet  # pretend the E2EE layer is already stripped


def fake_voice(dave=None):
    listeners = []
    state = SimpleNamespace(
        secret_key=KEY, hook=None, mode="aead_xchacha20_poly1305_rtpsize",
        dave_session=dave, dave_protocol_version=1 if dave else 0,
        add_socket_listener=listeners.append, remove_socket_listener=listeners.remove,
    )
    ws = SimpleNamespace()
    return SimpleNamespace(_connection=state, ws=ws, listeners=listeners)


@pytest.fixture
async def receiver():
    heard = []
    voice = fake_voice()
    rx = vr.VoiceReceiver(voice, lambda user, pcm, ts: heard.append((user, pcm, ts)),
                          users={USER})
    rx.start()
    await speaking(rx, USER, SSRC)
    rx.heard = heard
    yield rx
    rx.stop()


async def speaking(rx, user_id, ssrc):
    msg = {"op": DiscordVoiceWebSocket.SPEAKING, "d": {"user_id": str(user_id), "ssrc": ssrc, "speaking": 1}}
    await rx.voice.ws._hook(rx.voice.ws, msg)


async def test_start_hooks_in_and_stop_unhooks():
    voice = fake_voice()
    rx = vr.VoiceReceiver(voice, lambda *a: None)
    rx.start()
    assert voice.listeners == [rx._on_datagram]
    assert voice._connection.hook == rx._websocket_hook and voice.ws._hook == rx._websocket_hook
    rx.stop()
    assert voice.listeners == [] and voice._connection.hook is None
    assert "_hook" not in voice.ws.__dict__


async def test_hears_a_real_decoded_tone(receiver):
    receiver.handle_datagram(discord_py_packet(opus_tone(), timestamp=1920))
    [(user, pcm, timestamp)] = receiver.heard
    assert user == USER and timestamp == 1920
    assert len(pcm) == 960 * vr.BYTES_PER_SAMPLE  # 20 ms, stereo, 16-bit
    assert max(abs(s) for s in array.array("h", pcm)) > 1000  # actual sound, not silence
    assert receiver.stats["frames"] == 1 and receiver.cpu_seconds > 0


async def test_ignores_unpermitted_users_before_decrypting(receiver):
    await speaking(receiver, OTHER, OTHER_SSRC)
    receiver.handle_datagram(b"\x80\x78" + struct.pack(">HII", 1, 960, OTHER_SSRC) + bytes(40))
    assert receiver.heard == [] and receiver.stats["ignored_user"] == 1
    assert receiver.stats["errors"] == 0  # garbage payload was never decrypted


async def test_unknown_ssrc_and_non_audio_are_counted(receiver):
    receiver.handle_datagram(discord_py_packet(opus_tone(), ssrc=9999))
    receiver.handle_datagram(bytes([0x81, 0xC9]) + bytes(30))
    assert receiver.stats["unknown_ssrc"] == 1 and receiver.stats["not_audio"] == 1


async def test_drops_late_and_duplicate_packets(receiver):
    for sequence in (5, 5, 4, 6):
        receiver.handle_datagram(discord_py_packet(opus_tone(), sequence=sequence))
    assert len(receiver.heard) == 2  # 5 and 6
    assert receiver.stats["late_or_duplicate"] == 2


async def test_a_broken_packet_never_raises(receiver):
    receiver.handle_datagram(b"\x80\x78" + struct.pack(">HII", 1, 960, SSRC) + os.urandom(40))
    assert receiver.stats["errors"] == 1 and receiver.heard == []


async def test_disconnect_forgets_the_speaker(receiver):
    receiver.handle_datagram(discord_py_packet(opus_tone()))
    await receiver.voice.ws._hook(receiver.voice.ws, {"op": DiscordVoiceWebSocket.CLIENT_DISCONNECT,
                                                      "d": {"user_id": str(USER)}})
    assert SSRC not in receiver.ssrc_to_user


async def test_previous_hook_still_runs():
    seen = []

    async def previous(ws, msg):
        seen.append(msg["op"])

    voice = fake_voice()
    voice._connection.hook = previous
    rx = vr.VoiceReceiver(voice, lambda *a: None)
    rx.start()
    await speaking(rx, USER, SSRC)
    assert seen == [DiscordVoiceWebSocket.SPEAKING]
    rx.stop()
    assert voice._connection.hook is previous and voice.ws._hook is previous


# --- DAVE ------------------------------------------------------------------------


async def make_dave_receiver(dave):
    heard = []
    rx = vr.VoiceReceiver(fake_voice(dave), lambda u, p, t: heard.append(u), users=None)
    rx.start()
    await speaking(rx, USER, SSRC)
    return rx, heard


async def test_dave_decrypts_per_user():
    dave = FakeDave()
    rx, heard = await make_dave_receiver(dave)
    rx.handle_datagram(discord_py_packet(opus_tone()))
    assert dave.calls == [USER] and heard == [USER]


async def test_dave_not_ready_or_failing_is_skipped():
    rx, heard = await make_dave_receiver(FakeDave(ready=False))
    rx.handle_datagram(discord_py_packet(opus_tone()))
    assert heard == [] and rx.stats["dave_not_ready"] == 1

    rx, heard = await make_dave_receiver(FakeDave(fail=True))
    rx.handle_datagram(discord_py_packet(opus_tone()))
    assert heard == [] and rx.stats["dave_failed"] == 1


async def test_dave_passes_silence_frames_through():
    dave = FakeDave()
    rx, heard = await make_dave_receiver(dave)
    rx.handle_datagram(discord_py_packet(OPUS_SILENCE))
    assert heard == [USER] and dave.calls == []


# --- identifying speakers from the moment of connecting ----------------------------


def test_track_speakers_learns_and_forgets():
    mapping = {}
    vr.track_speakers(mapping, {"op": DiscordVoiceWebSocket.SPEAKING,
                                "d": {"user_id": str(USER), "ssrc": SSRC, "speaking": 1}})
    vr.track_speakers(mapping, {"op": DiscordVoiceWebSocket.HEARTBEAT_ACK, "d": 123})
    assert mapping == {SSRC: USER}
    vr.track_speakers(mapping, {"op": DiscordVoiceWebSocket.CLIENT_DISCONNECT, "d": {"user_id": str(USER)}})
    assert mapping == {}


async def test_listening_client_installs_its_hook_before_connecting():
    # The real class, built without a network: the connection state it creates
    # must carry our hook, so the voice websocket calls it from its first message.
    loop = __import__("asyncio").get_running_loop()
    fake_client = SimpleNamespace(_connection=SimpleNamespace(loop=loop), user=SimpleNamespace(id=1))
    channel = SimpleNamespace(guild=SimpleNamespace(id=1), id=2)
    voice = vr.ListeningVoiceClient(fake_client, channel)
    assert voice._connection.hook == voice._on_voice_message
    await voice._on_voice_message(None, {"op": DiscordVoiceWebSocket.SPEAKING,
                                          "d": {"user_id": str(USER), "ssrc": SSRC}})
    assert voice.ssrc_to_user == {SSRC: USER}


async def test_receiver_uses_speakers_known_before_it_started():
    # Regression: the user spoke (or `spike say` had joined) BEFORE listening
    # began, so the one SPEAKING message was missed -> every packet "unknown_ssrc".
    heard = []
    voice = fake_voice()
    voice.ssrc_to_user = {SSRC: USER}  # learned at connect time by ListeningVoiceClient
    rx = vr.VoiceReceiver(voice, lambda user, pcm, ts: heard.append(user), users={USER})
    rx.start()
    assert voice._connection.hook is None and "_hook" not in voice.ws.__dict__  # nothing patched
    rx.handle_datagram(discord_py_packet(opus_tone()))
    assert heard == [USER] and rx.stats["unknown_ssrc"] == 0
    rx.stop()
    rx2 = vr.VoiceReceiver(voice, lambda user, pcm, ts: heard.append(user), users={USER})
    rx2.start()  # a second listen session on the same connection still knows them
    rx2.handle_datagram(discord_py_packet(opus_tone(), sequence=2))
    assert heard == [USER, USER]


async def test_failed_dave_packets_are_concealed_not_dropped():
    # A lost 20 ms mid-word used to leave a hole; now Opus fills in a guess so
    # the audio keeps its timing (and the failure is logged with its reason).
    dave = FakeDave()
    heard = []
    rx = vr.VoiceReceiver(fake_voice(dave), lambda u, p, t: heard.append(len(p)), users=None)
    rx.start()
    await speaking(rx, USER, SSRC)
    rx.handle_datagram(discord_py_packet(opus_tone(), sequence=1))
    dave.fail = True
    rx.handle_datagram(discord_py_packet(opus_tone(), sequence=2))
    assert heard == [960 * vr.BYTES_PER_SAMPLE] * 2  # two 20 ms chunks, one of them invented
    assert rx.stats["frames"] == 1 and rx.stats["concealed"] == 1 and rx.stats["dave_failed"] == 1
    assert rx.loss_log == [(USER, 0.02, "dave_failed")]
    assert rx.loss_reasons == {"RuntimeError: no key for user": 1}


async def test_failure_before_any_audio_is_skipped():
    rx, heard = await make_dave_receiver(FakeDave(fail=True))
    rx.handle_datagram(discord_py_packet(opus_tone()))
    assert heard == [] and rx.stats["dave_failed"] == 1 and rx.stats["concealed"] == 0
