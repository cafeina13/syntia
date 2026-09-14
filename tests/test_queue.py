"""The queue: play / add / skip / previous / seek / stop / leave, and recovery."""

import pytest
from conftest import make_member, make_message

import music


def entry(title, channel=None, start=0):
    return {"query": title, "title": title, "start_seconds": start, "channel": channel}


class StartedTracks(list):
    # A list of (title, offset, announce) for every track that "started".
    def __init__(self):
        super().__init__()
        self.fail = set()  # titles that should fail to load


@pytest.fixture
def started(monkeypatch):
    # Replace the real track starter (which needs yt-dlp + FFmpeg) with one that
    # records what would have played. Titles in `started.fail` can't load.
    calls = StartedTracks()

    async def fake_start(guild, item, offset, announce=True):
        if item["title"] in calls.fail:
            return False
        player = music.get_player(guild.id)
        player.current = item
        player.offset = offset
        guild.voice_client.playing = True
        calls.append((item["title"], offset, announce))
        return True

    monkeypatch.setattr(music, "_start_track", fake_start)
    return calls


async def test_play_next_archives_current_and_starts_the_next(connected, started):
    voice, message = connected
    player = music.get_player(message.guild.id)
    player.current = entry("A")
    player.queue = [entry("B"), entry("C")]
    await music.play_next(message.guild)
    assert started == [("B", 0, True)]
    assert [t["title"] for t in player.history] == ["A"]
    assert [t["title"] for t in player.queue] == ["C"]


async def test_play_next_skips_tracks_that_fail_to_load(connected, started):
    voice, message = connected
    player = music.get_player(message.guild.id)
    player.queue = [entry("broken"), entry("good")]
    started.fail.add("broken")
    await music.play_next(message.guild)
    assert started == [("good", 0, True)]


async def test_play_next_replays_the_same_track_for_a_seek(connected, started):
    voice, message = connected
    player = music.get_player(message.guild.id)
    song = entry("A")
    player.current = song
    player.queue = [entry("B")]
    player.seek_target = (song, 120)
    await music.play_next(message.guild)
    assert started == [("A", 120, False)]  # same track, silent restart
    assert player.history == [] and len(player.queue) == 1


async def test_add_music_starts_when_idle_and_queues_when_busy(connected, started):
    voice, message = connected
    await music.add_music(message, "first song")
    assert started == [("first song", 0, True)]
    await music.add_music(message, "second song")
    assert len(started) == 1  # didn't interrupt
    assert "Added to queue (#1)" in message.channel.last


async def test_play_music_replaces_the_queue_and_restarts(connected, started):
    voice, message = connected
    player = music.get_player(message.guild.id)
    player.queue = [entry("old")]
    voice.playing = True
    await music.play_music(message, "new song", start_seconds=30)
    assert [t["title"] for t in player.queue] == ["new song"]
    assert voice.stop_calls == 1  # the after-callback would then play it


async def test_play_music_needs_the_user_in_voice(guild, text_channel):
    message = make_message(guild, make_member(voice_channel=None), text_channel)
    await music.play_music(message, "anything")
    assert "Join a voice channel first" in text_channel.last


async def test_skip_and_clear(connected):
    voice, message = connected
    await music.skip_song(message)
    assert message.channel.last == "Nothing is playing."
    voice.playing = True
    await music.skip_song(message)
    assert voice.stop_calls == 1

    await music.clear_queue(message)
    assert message.channel.last == "The queue is already empty."
    music.get_player(message.guild.id).queue = [entry("A"), entry("B")]
    await music.clear_queue(message)
    assert "Cleared 2" in message.channel.last


async def test_play_previous_puts_previous_then_current_up_front(connected, started):
    voice, message = connected
    player = music.get_player(message.guild.id)
    player.history = [entry("old")]
    player.current = entry("now")
    player.queue = [entry("next")]
    voice.playing = True
    await music.play_previous(message)
    assert [t["title"] for t in player.queue] == ["old", "now", "next"]
    assert voice.stop_calls == 1


async def test_play_previous_with_no_history(connected):
    voice, message = connected
    await music.play_previous(message)
    assert message.channel.last == "No previous song to go back to."


async def test_seek_relative_and_absolute(connected, monkeypatch):
    voice, message = connected
    player = music.get_player(message.guild.id)
    player.current = entry("A")
    voice.playing = True
    monkeypatch.setattr(music.GuildPlayer, "position", lambda self: 100.0)

    await music.seek(message, 30)
    assert player.seek_target[1] == 130 and "Jumped to 2:10" in message.channel.last
    voice.playing = True
    await music.seek(message, -500)
    assert player.seek_target[1] == 0  # never before the start
    voice.playing = True
    await music.seek(message, to=3720)
    assert player.seek_target[1] == 3720 and "1:02:00" in message.channel.last


async def test_seek_when_nothing_plays(connected):
    voice, message = connected
    await music.seek(message, 30)
    assert message.channel.last == "Nothing is playing to seek."


async def test_stop_clears_music_but_stays_connected(connected, started):
    voice, message = connected
    player = music.get_player(message.guild.id)
    player.current = entry("A")
    player.queue = [entry("B"), entry("C")]
    player.seek_target = (player.current, 30)
    voice.playing = True

    await music.stop_music(message)
    await music.play_next(message.guild)  # what the after-callback does

    assert message.guild.voice_client is voice and voice.connected
    assert player.queue == [] and player.current is None and player.seek_target is None
    assert player.history[-1]["title"] == "A"  # `previous` still works
    assert started == []  # nothing new started

    await music.stop_music(message)
    assert message.channel.last == "Nothing is playing."


async def test_leave_disconnects_and_forgets_the_queue(connected):
    voice, message = connected
    music.get_player(message.guild.id).queue = [entry("A")]
    await music.leave_voice(message)
    assert message.guild.voice_client is None
    assert message.guild.id not in music.players
    await music.leave_voice(message)
    assert message.channel.last == "I'm not in a voice channel."


async def test_shuffle_needs_two_songs(connected):
    voice, message = connected
    await music.shuffle_queue(message)
    assert "Not enough songs" in message.channel.last
    music.get_player(message.guild.id).queue = [entry(str(i)) for i in range(10)]
    await music.shuffle_queue(message)
    assert sorted(t["title"] for t in music.get_player(message.guild.id).queue) == \
        sorted(str(i) for i in range(10))


async def test_show_queue(connected):
    voice, message = connected
    await music.show_queue(message)
    assert message.channel.last == "Nothing playing and the queue is empty."
    player = music.get_player(message.guild.id)
    player.current = entry("Now")
    player.queue = [entry("Next")]
    await music.show_queue(message)
    assert message.channel.last == "**Now playing:** Now\n**Up next:**\n1. Next"


async def test_voice_loss_parks_the_song_where_it_was(guild, text_channel, monkeypatch):
    player = music.get_player(guild.id)
    player.current = entry("A", text_channel)
    player.queue = [entry("B")]
    player.text_channel = text_channel
    monkeypatch.setattr(music.GuildPlayer, "position", lambda self: 95.7)

    await music.play_next(guild)  # voice_client is None -> connection lost

    assert player.interrupted
    assert [t["title"] for t in player.queue] == ["A", "B"]
    assert player.queue[0]["start_seconds"] == 95  # resumes mid-song
    assert "Lost the voice connection" in text_channel.last
    await music.play_next(guild)
    assert len(text_channel.sent) == 1  # not announced twice
