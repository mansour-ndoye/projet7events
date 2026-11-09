from types import SimpleNamespace
from unittest.mock import patch
from subprocess import CompletedProcess

from fastapi.testclient import TestClient


# On simule le chargement du RAG pour ne pas charger
# les modèles ni contacter Mistral pendant les tests.
with patch("scripts.rag_chatbot.load_vectorstore", return_value=object()), \
     patch("scripts.rag_chatbot.create_llm", return_value=object()):
    import api


client = TestClient(api.app)


def test_root():
    """Vérifie que la route d'accueil fonctionne."""

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data
    assert data["message"].startswith("API Puls-Events")


def test_ask():
    """Vérifie la réponse à une question avec un RAG simulé."""

    document = SimpleNamespace(
        metadata={
            "title": "Concert à Tours",
            "city": "Tours",
            "date_begin": "2026-12-15",
            "address": "Tours"
        }
    )

    with patch(
        "api.ask_chatbot",
        return_value=("Voici un concert à Tours.", [document])
    ):
        response = client.post(
            "/ask",
            json={"question": "Je cherche un concert à Tours"}
        )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "Je cherche un concert à Tours"
    assert data["answer"] == "Voici un concert à Tours."
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) == 1
    assert data["sources"][0]["city"] == "Tours"


def test_empty_question():
    """Vérifie qu'une question vide est refusée."""

    response = client.post(
        "/ask",
        json={"question": ""}
    )

    assert response.status_code == 400


def test_ask_without_rag():
    """Vérifie le comportement lorsque le RAG est indisponible."""

    old_vectorstore = api.vectorstore
    old_llm = api.llm

    try:
        api.vectorstore = None
        api.llm = None

        response = client.post(
            "/ask",
            json={"question": "Je cherche un concert"}
        )

        assert response.status_code == 503

    finally:
        api.vectorstore = old_vectorstore
        api.llm = old_llm


def test_rebuild():
    """Vérifie la reconstruction de l'index sans exécuter le script."""

    fake_result = CompletedProcess(
        args=[],
        returncode=0,
        stdout="Index reconstruit",
        stderr=""
    )

    with patch("api.subprocess.run", return_value=fake_result), \
         patch("api.load_vectorstore", return_value=object()):

        response = client.post("/rebuild")

    assert response.status_code == 200
    assert "message" in response.json()
    assert "Index reconstruit" in response.json()["details"]
