"""`syntia np`: current track + timestamp, and YouTube links with a start time."""

import time

import pytest

import music

VIDEO = "https://www.youtube.com/watch?v=abc123"


@pytest.mark.parametrize("url, seconds", [
    (VIDEO, 0),
    (f"{VIDEO}&t=3753", 3753),
    (f"{VIDEO}&t=3753s", 3753),
    (f"{VIDEO}&t=1h2m33s", 3753),
    (f"{VIDEO}&t=2m", 120),
    ("https://youtu.be/abc123?t=90", 90),
    ("https://music.youtube.com/watch?v=abc123&t=45", 45),
    (f"{VIDEO}&list=PL1&t=10", 10),
    (f"{VIDEO}&t=soon", 0),
    ("https://example.com/video?t=90", 0),  # not YouTube
    ("lofi hip hop t=90", 0),  # a search, not a link
])
def test_youtube_start_seconds(url, seconds):
    assert music.youtube_start_seconds(url) == seconds


@pytest.mark.parametrize("url, expected", [
    (VIDEO, f"{VIDEO}&t=3753"),
    (f"{VIDEO}&t=10", f"{VIDEO}&t=3753"),  # an old time is replaced
    (f"https://www.youtube.com/watch?t=10&v=abc123", f"https://www.youtube.com/watch?v=abc123&t=3753"),
    ("https://youtu.be/abc123", "https://youtu.be/abc123?t=3753"),
    ("https://soundcloud.com/some/track", None),
    (None, None),
])
def test_resume_link(url, expected):
    assert music.resume_link(url, 3753) == expected


def test_resume_link_round_trips():
    # The link np hands out must start `play` at the same second.
    assert music.youtube_start_seconds(music.resume_link(VIDEO, 3753)) == 3753


async def test_play_link_with_time_starts_there(connected):
    voice, message = connected
    voice.playing = True  # so enqueue only queues
    await music.enqueue(message, f"{VIDEO}&t=1h2m33s")
    assert music.get_player(message.guild.id).queue[-1]["start_seconds"] == 3753


async def test_explicit_start_wins_over_the_link(connected):
    voice, message = connected
    await music.enqueue(message, f"{VIDEO}&t=100", start_seconds=5)
    assert music.get_player(message.guild.id).queue[-1]["start_seconds"] == 5


@pytest.fixture
def listening(connected):
    # 1:02:33 into a 1:45:10 album video.
    voice, message = connected
    player = music.get_player(message.guild.id)
    player.current = {"query": VIDEO, "title": "Album - Full", "webpage_url": VIDEO,
                      "duration": 6310, "start_seconds": 0, "channel": message.channel}
    player.started_at = time.monotonic() - 3753
    voice.playing = True
    return voice, message, player


async def test_np_shows_time_and_resume_command(listening):
    voice, message, player = listening
    await music.now_playing(message)
    assert message.channel.last == (
        "🎵 **Album - Full**\n"
        "`1:02:33 / 1:45:10`\n"
        f"Pick up here later: `syntia play {VIDEO}&t=3753`"
    )


async def test_np_while_paused(listening):
    voice, message, player = listening
    await music.pause_music(message)
    await music.now_playing(message)
    assert "`1:02:33 / 1:45:10` (paused)" in message.channel.last


async def test_np_after_a_seek_offset(listening):
    # A seek restarts the source at an offset; np must include it.
    voice, message, player = listening
    player.offset = 600
    player.started_at = time.monotonic() - 33
    await music.now_playing(message)
    assert "`10:33 / 1:45:10`" in message.channel.last
    assert "&t=633`" in message.channel.last


async def test_np_never_runs_past_the_end(listening):
    voice, message, player = listening
    player.started_at = time.monotonic() - 99999
    await music.now_playing(message)
    assert "`1:45:10 / 1:45:10`" in message.channel.last


async def test_np_without_duration_or_youtube_link(listening):
    voice, message, player = listening
    player.current.update(duration=None, webpage_url="https://soundcloud.com/x/y")
    await music.now_playing(message)
    assert message.channel.last == "🎵 **Album - Full**\n`1:02:33`"


async def test_np_when_nothing_plays(connected):
    voice, message = connected
    await music.now_playing(message)
    assert message.channel.last == "Nothing is playing."
