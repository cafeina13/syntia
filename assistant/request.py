"""
A spoken request, shaped so Syntia's existing brain can handle it.

ai.ask_ai() and the music functions were written for Discord text messages, but
they only ever touch three things on one:

    .guild     which server
    .author    who asked (their .voice.channel, .id, .display_name)
    .channel   where to reply: .send(text, silent=...) and .typing()

A VoiceRequest has exactly those, so a voice command runs through the same code
as a typed one:

    await ai.ask_ai(VoiceRequest(guild, member, ReplyChannel(text_channel)), "", audio=wav)

The dependency only goes one way: assistant/ imports the bot's modules, never
the other way round.
"""

from dataclasses import dataclass
from typing import Awaitable, Callable

import discord


class ReplyChannel:
    """Wraps the text channel a voice request replies in.

    Text still goes to the channel as usual. `on_track_started` is the hook the
    music code calls when a track from this request starts playing — the voice
    assistant uses it to say the title out loud (only for this request's first
    track, never for every song change).
    """

    def __init__(self, text_channel: discord.abc.Messageable,
                 on_first_track: Callable[[dict], Awaitable[None]] | None = None):
        self.text_channel = text_channel
        self.on_first_track = on_first_track
        self._announced = False

    async def send(self, content=None, **kwargs):
        return await self.text_channel.send(content, **kwargs)

    def typing(self):
        return self.text_channel.typing()

    async def on_track_started(self, entry: dict):
        if self._announced or self.on_first_track is None:
            return
        self._announced = True
        await self.on_first_track(entry)

    def __getattr__(self, name):
        # Anything else (e.g. .id, .name, .mention) behaves like the real channel.
        # (Only called for attributes this object lacks; guard against looping if
        # text_channel itself isn't set yet.)
        if name == "text_channel":
            raise AttributeError(name)
        return getattr(self.text_channel, name)


@dataclass
class VoiceRequest:
    guild: discord.Guild
    author: discord.Member
    channel: ReplyChannel
