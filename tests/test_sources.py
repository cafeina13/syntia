"""Link detection, time parsing, and reading track lists (no network used)."""

import json

import pytest

import config
import music


@pytest.mark.parametrize("text, seconds", [
    ("90", 90),
    ("5:25", 325),
    ("1:02:00", 3720),
    ("0", 0),
    ("abc", None),
    ("1:xx", None),
    ("1:2:3:4", None),
    ("", None),
    ("-5", None),
])
def test_parse_timestamp(text, seconds):
    assert music.parse_timestamp(text) == seconds


@pytest.mark.parametrize("seconds, text", [(0, "0:00"), (325, "5:25"), (3720, "1:02:00")])
def test_fmt_time(seconds, text):
    assert music.fmt_time(seconds) == text


def test_ffmpeg_options_only_seeks_when_asked():
    assert "-ss" not in music.ffmpeg_options(0)["before_options"]
    assert music.ffmpeg_options(600)["before_options"].startswith("-ss 600 ")


def test_is_spotify_url():
    assert music.is_spotify_url("https://open.spotify.com/playlist/abc")
    assert not music.is_spotify_url("lofi hip hop")


@pytest.mark.parametrize("url, expected", [
    ("https://www.youtube.com/playlist?list=PL123", True),
    ("https://music.youtube.com/playlist?list=PL123", True),
    # A normal video that happens to carry a list= stays a single video.
    ("https://www.youtube.com/watch?v=abc&list=PL123", False),
    ("lofi hip hop", False),
])
def test_is_youtube_playlist(url, expected):
    assert music.is_youtube_playlist(url) is expected


def test_spotify_query():
    track = {"name": "Bohemian Rhapsody", "artists": [{"name": "Queen"}]}
    assert music._spotify_query(track) == "Queen - Bohemian Rhapsody"
    assert music._spotify_query({"name": "Solo", "artists": []}) == "Solo"
    assert music._spotify_query(None) == ""


def test_find_tracklist_digs_through_nested_json():
    data = {"props": {"pageProps": [{"x": 1}, {"state": {"trackList": [{"title": "A"}]}}]}}
    assert music._find_tracklist(data) == [{"title": "A"}]
    assert music._find_tracklist({"nothing": "here"}) is None


def test_spotify_tracks_ignores_non_spotify_links():
    assert music.spotify_tracks("https://example.com/playlist/1") == []


def test_spotify_tracks_falls_back_to_scraping_when_api_fails(monkeypatch):
    class BrokenApi:
        def playlist_items(self, *args, **kwargs):
            raise RuntimeError("403 - not your playlist")

    monkeypatch.setattr(config, "spotify_client", BrokenApi())
    monkeypatch.setattr(music, "_scrape_spotify", lambda kind, sid: [f"{kind}:{sid}"])
    url = "https://open.spotify.com/playlist/AbC123?si=xyz"
    assert music.spotify_tracks(url) == ["playlist:AbC123"]


def test_scrape_spotify_reads_the_embed_page(monkeypatch):
    payload = {"props": {"state": {"data": {"entity": {"trackList": [
        {"title": "Song One", "subtitle": "Artist A"},
        {"title": "Song Two", "subtitle": ""},
        {"title": "", "subtitle": ""},  # empty entries are dropped
    ]}}}}}
    html = f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script></html>'

    class FakeResponse:
        def read(self):
            return html.encode("utf-8")

    monkeypatch.setattr(music.urllib.request, "urlopen", lambda request, timeout: FakeResponse())
    assert music._scrape_spotify("playlist", "abc") == ["Artist A - Song One", "Song Two"]


def test_scrape_spotify_without_data_returns_nothing(monkeypatch):
    class FakeResponse:
        def read(self):
            return b"<html>no data here</html>"

    monkeypatch.setattr(music.urllib.request, "urlopen", lambda request, timeout: FakeResponse())
    assert music._scrape_spotify("track", "abc") == []


# --- Spotify must never freeze the bot (regression, live) ---------------------------


def test_an_expired_login_fails_instead_of_asking_on_the_console(monkeypatch):
    # spotipy's default asks for a new login with input() and waits forever.
    import builtins

    from spotipy.cache_handler import MemoryCacheHandler
    from spotipy.oauth2 import SpotifyOauthError

    def no_prompts(*args, **kwargs):
        raise AssertionError("the bot must never wait for console input")

    monkeypatch.setattr(builtins, "input", no_prompts)
    auth = config.BotSpotifyOAuth(client_id="id", client_secret="secret",
                                  redirect_uri="http://127.0.0.1:8888/callback",
                                  cache_handler=MemoryCacheHandler(), open_browser=False)
    with pytest.raises(SpotifyOauthError):
        auth.get_access_token(as_dict=False)  # nothing cached -> would have prompted


def test_the_configured_client_never_waits_out_a_rate_limit():
    if config.spotify_client is None:
        pytest.skip("no Spotify login configured")
    assert config.spotify_client.status_retries == 0
    assert config.spotify_client.requests_timeout <= 10
    assert isinstance(config.spotify_auth, config.BotSpotifyOAuth)


async def test_a_stuck_spotify_request_times_out(monkeypatch):
    import time as real_time

    from conftest import FakeGuild, make_message

    def stuck(url):
        real_time.sleep(0.5)  # a request that never comes back (for this test: 0.5 s)
        return ["never used"]

    monkeypatch.setattr(music, "spotify_tracks", stuck)
    monkeypatch.setattr(music, "SPOTIFY_TIMEOUT_SECONDS", 0.05)
    message = make_message(FakeGuild())
    added = await music.enqueue(message, "https://open.spotify.com/playlist/abc")
    assert added == 0
    assert message.channel.last == "Spotify took too long to answer — try again in a bit."


async def test_spotify_errors_are_one_short_line_in_chat(monkeypatch):
    from conftest import FakeGuild, make_message

    def broken(url):
        raise RuntimeError("http status: 404, code: -1 - https://api.spotify.com/v1/playlists/x\n"
                           " Resource not found, reason: None\n" + "more detail\n" * 20)

    monkeypatch.setattr(music, "spotify_tracks", broken)
    message = make_message(FakeGuild())
    await music.enqueue(message, "https://open.spotify.com/playlist/abc")
    assert "\n" not in message.channel.last and len(message.channel.last) < 220


def test_short_error_strips_ytdlp_noise():
    long = "ERROR: [youtube] abc: Sign in to confirm you're not a bot. " + "x" * 300
    short = music.short_error(RuntimeError(long))
    assert short.startswith("[youtube] abc: Sign in") and len(short) <= 150
