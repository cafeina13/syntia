"""The idle timeout: when the bot leaves, and who can change the setting."""

import pytest
from conftest import FakeTextChannel, make_member, make_message

import config
import music


def went_quiet_minutes_ago(guild_id, minutes):
    # Pretend the bot went idle `minutes` ago, instead of really waiting.
    music.idle_since[guild_id] -= minutes * 60


async def test_leaves_after_the_timeout_when_nothing_plays(connected):
    voice, message = connected
    music.get_player(message.guild.id).text_channel = message.channel

    await music.check_idle()
    assert voice.connected  # first poll only starts the clock
    went_quiet_minutes_ago(message.guild.id, 4)
    await music.check_idle()
    assert voice.connected  # 4 min < 5 min
    went_quiet_minutes_ago(message.guild.id, 2)
    await music.check_idle()

    assert not voice.connected
    assert message.channel.last == "No music for 5 min, heading out."
    assert message.channel.silent[-1] is True
    assert message.guild.id not in music.players
    assert message.guild.id not in music.idle_since


async def test_leaves_when_alone_even_while_playing(connected):
    voice, message = connected
    voice.playing = True
    voice.channel.members = [make_member(99, bot=True)]  # only the bot is left
    music.get_player(message.guild.id).text_channel = message.channel

    await music.check_idle()
    went_quiet_minutes_ago(message.guild.id, 6)
    await music.check_idle()

    assert not voice.connected
    assert message.channel.last == "Everyone left, so I did too."


async def test_stays_while_playing_to_listeners(connected):
    voice, message = connected
    voice.playing = True
    await music.check_idle()
    assert voice.connected and message.guild.id not in music.idle_since


async def test_music_starting_again_resets_the_clock(connected):
    voice, message = connected
    await music.check_idle()
    went_quiet_minutes_ago(message.guild.id, 4)
    voice.playing = True  # e.g. someone played a song
    await music.check_idle()
    assert message.guild.id not in music.idle_since
    voice.playing = False
    await music.check_idle()
    went_quiet_minutes_ago(message.guild.id, 4)
    await music.check_idle()
    assert voice.connected  # counted from the new quiet spell, not the old one


async def test_disabled_timeout_never_leaves(connected):
    voice, message = connected
    music.get_timeout(message.guild.id)["enabled"] = False
    await music.check_idle()
    assert message.guild.id not in music.idle_since
    assert voice.connected


async def test_leaves_quietly_when_there_is_no_text_channel(connected):
    voice, message = connected  # the player never recorded a text channel
    await music.check_idle()
    went_quiet_minutes_ago(message.guild.id, 6)
    await music.check_idle()
    assert not voice.connected and message.channel.sent == []


def test_default_comes_from_config(guild, monkeypatch):
    assert music.get_timeout(guild.id) == {"enabled": True, "minutes": 5}
    monkeypatch.setattr(config, "IDLE_TIMEOUT_MINUTES", 0)
    assert music.get_timeout(guild.id + 1) == {"enabled": False, "minutes": 5}


# --- `syntia timeout` --------------------------------------------------------


async def run_timeout(guild, author, arg):
    channel = FakeTextChannel()
    await music.set_timeout(make_message(guild, author, channel), arg)
    return channel.last


async def test_anyone_can_see_the_setting(guild):
    assert await run_timeout(guild, make_member(), "") == "Idle timeout: **on, 5 min**."


@pytest.mark.parametrize("arg", ["off", "on", "10"])
async def test_regular_members_cannot_change_it(guild, arg):
    reply = await run_timeout(guild, make_member(), arg)
    assert reply.startswith("Only the owner or someone with Manage Server")
    assert music.get_timeout(guild.id) == {"enabled": True, "minutes": 5}


async def test_admins_can_change_it(guild):
    admin = make_member(7, manage_guild=True)
    assert "**off**" in await run_timeout(guild, admin, "off")
    assert music.get_timeout(guild.id)["enabled"] is False
    assert await run_timeout(guild, admin, "") == "Idle timeout: **off**."
    assert await run_timeout(guild, admin, "12") == "Idle timeout **on** (12 min)."
    assert music.get_timeout(guild.id) == {"enabled": True, "minutes": 12}
    await run_timeout(guild, admin, "off")
    assert await run_timeout(guild, admin, "ON") == "Idle timeout **on** (12 min)."  # keeps minutes


async def test_the_owner_can_change_it(guild, monkeypatch):
    monkeypatch.setattr(config, "OWNER_ID", 555)
    await run_timeout(guild, make_member(555), "off")
    assert music.get_timeout(guild.id)["enabled"] is False


async def test_owner_id_zero_means_no_owner(guild):
    # OWNER_ID=0 must not turn a member with id 0 into the owner.
    await run_timeout(guild, make_member(0), "off")
    assert music.get_timeout(guild.id)["enabled"] is True


@pytest.mark.parametrize("arg", ["0", "241", "soon", "-5"])
async def test_rejects_bad_values(guild, arg):
    reply = await run_timeout(guild, make_member(manage_guild=True), arg)
    assert reply.startswith("Use `syntia timeout on`")


async def test_settings_are_per_server(guild):
    admin = make_member(manage_guild=True)
    await run_timeout(guild, admin, "off")
    other = type(guild)(guild_id=2)
    assert music.get_timeout(other.id)["enabled"] is True
