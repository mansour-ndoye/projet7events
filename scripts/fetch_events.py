import json
from pathlib import Path
from datetime import datetime, timezone
import requests

API_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/evenements-publics-openagenda/records"
)
OUTPUT_FILE  = Path("data/events_raw.json")
DEPARTMENT   = "Indre-et-Loire"
PAGE_SIZE    = 100    # max autorisé par l'API
MAX_EVENTS   = 1000   # nombre total visé


def fetch_events():
    print("=== RÉCUPÉRATION DES ÉVÉNEMENTS OPENAGENDA ===")

    all_results = []
    offset      = 0

    while len(all_results) < MAX_EVENTS:
        params = {
            "limit":    PAGE_SIZE,
            "offset":   offset,
            "order_by": "firstdate_begin desc",
             "where":    'location_department="Indre-et-Loire"',
        }

        try:
            response = requests.get(API_URL, params=params, timeout=60)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"❌ Erreur HTTP : {e}")
            print(f"   Détail : {response.text[:300]}")
            break
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur réseau : {e}")
            break

        data        = response.json()
        results     = data.get("results", [])
        total_count = data.get("total_count", 0)

        if offset == 0:
            print(f"Total disponible sur l'API : {total_count:,}")

        if not results:
            break  # plus rien à récupérer

        all_results.extend(results)
        offset += PAGE_SIZE

        print(f"  Page {offset // PAGE_SIZE} — {len(all_results)} événements récupérés...")

        # Stop si on a tout récupéré
        if offset >= total_count:
            break

    print(f"\nTotal brut récupéré : {len(all_results)}")

    # ── Filtrage département ──────────────────────────────────────
    events_dept = [
        e for e in all_results
        if str(e.get("location_department", "")).strip() == DEPARTMENT
    ]
    print(f"Événements en Indre-et-Loire : {len(events_dept)}")

    # ── Filtrage dates futures ────────────────────────────────────
    now = datetime.now(timezone.utc)
    filtered = []

    for event in events_dept:
        date_str = event.get("firstdate_begin")
        if not date_str:
            continue
        try:
            event_date = datetime.fromisoformat(
                date_str.replace("Z", "+00:00")
            )
            if event_date >= now:
                filtered.append(event)
        except ValueError:
            continue

    print(f"Événements futurs conservés : {len(filtered)}")

    # ── Sauvegarde ───────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"✅ Fichier sauvegardé : {OUTPUT_FILE}")


if __name__ == "__main__":
    fetch_events()