import json

import pandas as pd

import scripts.preprocess_events as preprocessing


def test_clean_text():
    result = preprocessing.clean_text(
        "<p>Concert</p>   à   Tours"
    )
    assert result == "Concert à Tours"


def test_clean_text_empty():
    assert preprocessing.clean_text(None) == ""
    assert preprocessing.clean_text("") == ""


def test_load_events_list(tmp_path, monkeypatch):
    events = [{"title_fr": "Concert"}]
    file_path = tmp_path / "events.json"
    file_path.write_text(
        json.dumps(events),
        encoding="utf-8"
    )

    monkeypatch.setattr(
        preprocessing,
        "INPUT_FILE",
        file_path
    )

    assert preprocessing.load_events() == events


def test_load_events_dictionary(tmp_path, monkeypatch):
    events = [{"title_fr": "Festival"}]
    file_path = tmp_path / "events.json"
    file_path.write_text(
        json.dumps({"results": events}),
        encoding="utf-8"
    )

    monkeypatch.setattr(
        preprocessing,
        "INPUT_FILE",
        file_path
    )

    assert preprocessing.load_events() == events


def test_preprocess_events():
    events = [
        {
            "uid": "123",
            "title_fr": "<b>Concert</b> à Tours",
            "description_fr": "<p>Musique</p>",
            "longdescription_fr": "Description longue",
            "firstdate_begin": "2026-12-15",
            "firstdate_end": "2026-12-15",
            "location_city": "Tours",
            "location_department": "Indre-et-Loire",
            "location_region": "Centre-Val de Loire",
            "location_address": "Place Plumereau",
            "location_postalcode": "37000",
            "category": "Musique",
            "keywords_fr": "concert, musique",
            "location_website": "https://example.com",
            "status": "Publié"
        },
        {
            "uid": "456",
            "title_fr": "",
            "description_fr": "Sans titre"
        }
    ]

    result = preprocessing.preprocess_events(events)

    assert len(result) == 1
    assert result[0]["title"] == "Concert à Tours"
    assert result[0]["description"] == "Musique"
    assert result[0]["city"] == "Tours"
    assert result[0]["department"] == "Indre-et-Loire"


def test_filter_events():
    now = pd.Timestamp.now(tz="UTC")

    df = pd.DataFrame([
        {
            "title": "Concert à Tours",
            "date_begin": now + pd.Timedelta(days=10),
            "date_end": now + pd.Timedelta(days=10),
            "city": "Tours",
            "department": "Indre-et-Loire"
        },
        {
            "title": "Festival dans le département",
            "date_begin": now + pd.Timedelta(days=20),
            "date_end": now + pd.Timedelta(days=20),
            "city": "Amboise",
            "department": "Indre-et-Loire"
        },
        {
            "title": "Événement hors zone",
            "date_begin": now + pd.Timedelta(days=15),
            "date_end": now + pd.Timedelta(days=15),
            "city": "Paris",
            "department": "Paris"
        },
        {
            "title": "Événement sans date",
            "date_begin": None,
            "date_end": None,
            "city": "Tours",
            "department": "Indre-et-Loire"
        }
    ])

    result = preprocessing.filter_events(df)

    assert len(result) == 2
    assert set(result["title"]) == {
        "Concert à Tours",
        "Festival dans le département"
    }
