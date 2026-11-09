import json
import re
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/events_raw.json")
OUTPUT_FILE = Path("data/events_clean.csv")


def clean_text(text):
    """Nettoie un texte pour faciliter son utilisation future dans le RAG."""

    if not text:
        return ""

    text = str(text)

    # Suppression des balises HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # Suppression des espaces multiples
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_events():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    # fetch_events.py sauvegarde directement une liste
    if isinstance(data, list):
        return data

    # Compatibilité avec l'ancien format OpenDataSoft
    if isinstance(data, dict):
        return data.get("results", [])

    return []


def preprocess_events(events):
    """Nettoie et structure les événements."""

    cleaned_events = []

    for event in events:
        title = clean_text(event.get("title_fr"))
        description = clean_text(event.get("description_fr"))
        long_description = clean_text(event.get("longdescription_fr"))

        city = clean_text(event.get("location_city"))
        department = clean_text(event.get("location_department"))
        region = clean_text(event.get("location_region"))

        category = clean_text(event.get("category"))
        keywords = clean_text(event.get("keywords_fr"))

        date_begin = event.get("firstdate_begin")
        date_end = event.get("firstdate_end")

        # Un événement doit au minimum avoir un titre
        if not title:
            continue

        cleaned_events.append(
            {
                "uid": event.get("uid"),
                "title": title,
                "description": description,
                "long_description": long_description,
                "date_begin": date_begin,
                "date_end": date_end,
                "city": city,
                "department": department,
                "region": region,
                "address": clean_text(event.get("location_address")),
                "postal_code": clean_text(event.get("location_postalcode")),
                "category": category,
                "keywords": keywords,
                "website": event.get("location_website"),
                "status": clean_text(event.get("status")),
            }
        )

    return cleaned_events


def filter_events(df):
    """Filtre les événements par localisation et période."""

    # Conversion des dates
    df["date_begin"] = pd.to_datetime(
        df["date_begin"],
        errors="coerce",
        utc=True
    )

    df["date_end"] = pd.to_datetime(
        df["date_end"],
        errors="coerce",
        utc=True
    )

    # Suppression des événements sans date valide
    df = df.dropna(subset=["date_begin"])

    # Date actuelle
    now = pd.Timestamp.now(tz="UTC")

    # Un an dans le passé
    one_year_ago = now - pd.DateOffset(years=1)

    # Garder :
    # - les événements datant de moins d'un an
    # - les événements futurs
    df = df[
        (df["date_begin"] >= one_year_ago)
        & (df["date_begin"] <= now + pd.DateOffset(years=1))
    ]

    # Zone géographique :
    # Tours ou Indre-et-Loire
    city_filter = df["city"].str.lower().str.contains(
        "tours",
        na=False
    )

    department_filter = df["department"].str.lower().str.contains(
        "indre-et-loire|indre et loire|37",
        regex=True,
        na=False
    )

    df = df[city_filter | department_filter]

    return df


def main():

    print("Chargement des événements...")

    events = load_events()

    print(f"Événements récupérés : {len(events)}")

    cleaned_events = preprocess_events(events)

    print(
        f"Événements après nettoyage : "
        f"{len(cleaned_events)}"
    )

    df = pd.DataFrame(cleaned_events)

    df = filter_events(df)

    print(
        f"Événements après filtrage : "
        f"{len(df)}"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"✅ Dataset final sauvegardé : "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()