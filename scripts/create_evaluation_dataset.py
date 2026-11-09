from pathlib import Path
import pandas as pd

SOURCE = Path("data/events_clean.csv")
DESTINATION = Path("tests/jeu_test_annote.csv")

df = pd.read_csv(SOURCE).fillna("")
df["date_begin"] = pd.to_datetime(df["date_begin"], errors="coerce", utc=True)

# Recherche dans les informations textuelles de chaque événement.
df["search_text"] = (
    df["title"].astype(str) + " " +
    df["description"].astype(str) + " " +
    df["long_description"].astype(str) + " " +
    df["keywords"].astype(str) + " " +
    df["city"].astype(str)
).str.lower()

tests = []

def add_test(category, question, mask, criteria):
    matches = df[mask].sort_values("date_begin")

    if matches.empty:
        print(f"Non trouvé : {question}")
        return

    # Un événement réel sert de référence pour ce test.
    event = matches.iloc[0]

    tests.append({
        "test_id": f"T{len(tests) + 1:02d}",
        "category": category,
        "question": question,
        "expected_event_titles": event["title"],
        "expected_event_uids": event["uid"],
        "expected_answer_criteria": criteria,
        "actual_answer": "",
        "retrieved_event_titles": "",
        "relevance_score": "",
        "faithfulness_score": "",
        "comments": ""
    })

# Tests géographiques
for city in [
    "Tours", "Chinon", "Amboise", "Joué-lès-Tours",
    "Loches", "Château-Renault"
]:
    add_test(
        "Géographie",
        f"Quels événements sont proposés à {city} ?",
        df["city"].str.lower().eq(city.lower()),
        f"Retrouver un événement situé à {city} et restituer son titre."
    )

# Tests thématiques
themes = [
    ("emploi", "emploi", "Quels événements concernent l'emploi ?"),
    ("recrutement", "recrutement", "Quels événements proposent du recrutement ?"),
    ("formation", "formation", "Quels événements concernent la formation ?"),
    ("musique", "musique", "Quels événements sont liés à la musique ?"),
    ("patrimoine", "patrimoine", "Quels événements permettent de découvrir le patrimoine ?"),
    ("exposition", "exposition", "Quelles expositions sont proposées ?"),
    ("visite", "visite", "Quelles visites peut-on découvrir ?"),
    ("famille", "petite enfance", "Quels événements sont adaptés aux jeunes enfants ?")
]

for category, term, question in themes:
    add_test(
        "Thématique",
        question,
        df["search_text"].str.contains(term, regex=False),
        f"Identifier un événement associé au thème : {category}."
    )

# Tests de recherche par période
add_test(
    "Date",
    "Quels événements sont prévus en décembre 2026 ?",
    df["date_begin"].dt.month.eq(12) & df["date_begin"].dt.year.eq(2026),
    "Retrouver un événement daté de décembre 2026 et donner sa date."
)

add_test(
    "Date",
    "Quels événements sont prévus en janvier 2027 ?",
    df["date_begin"].dt.month.eq(1) & df["date_begin"].dt.year.eq(2027),
    "Retrouver un événement daté de janvier 2027 et donner sa date."
)

add_test(
    "Date",
    "Quels événements sont prévus en juillet 2027 ?",
    df["date_begin"].dt.month.eq(7) & df["date_begin"].dt.year.eq(2027),
    "Retrouver un événement daté de juillet 2027 et donner sa date."
)

# Tests combinant ville et thème
add_test(
    "Recherche combinée",
    "Quels événements musicaux sont organisés à Joué-lès-Tours ?",
    df["city"].str.lower().eq("joué-lès-tours") &
    df["search_text"].str.contains("musique", regex=False),
    "Trouver un événement musical dans la ville demandée."
)

add_test(
    "Recherche combinée",
    "Quels événements liés à l'emploi sont organisés à Chinon ?",
    df["city"].str.lower().eq("chinon") &
    df["search_text"].str.contains("emploi", regex=False),
    "Trouver un événement lié à l'emploi à Chinon."
)

# Test sans résultat attendu : vérifier que le chatbot ne fabrique rien.
add_test(
    "Absence de résultat",
    "Quels événements proposent une compétition de fusées intergalactiques à Tours ?",
    df["title"].str.contains(
        "compétition de fusées intergalactiques",
        case=False,
        regex=False
    ),
    "Le chatbot doit signaler qu'aucun événement correspondant n'est trouvé."
)

DESTINATION.parent.mkdir(parents=True, exist_ok=True)

result = pd.DataFrame(tests)
result.to_csv(DESTINATION, index=False, encoding="utf-8-sig")

print("\nJeu de test créé :", DESTINATION)
print("Nombre de tests générés :", len(result))
print("\nRépartition par catégorie :")
print(result["category"].value_counts().to_string())
print("\nFichier prêt pour l'évaluation manuelle.")
