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
