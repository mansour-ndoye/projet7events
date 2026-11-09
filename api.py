import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from scripts.rag_chatbot import (
    load_vectorstore,
    create_llm,
    ask_chatbot,
)


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

app = FastAPI(
    title="Puls-Events RAG API",
    description=(
        "API REST permettant d'interroger le chatbot "
        "de recommandation d'événements culturels Puls-Events."
    ),
    version="1.0.0",
)


# ============================================================
# MODÈLE DE REQUÊTE
# ============================================================

class QuestionRequest(BaseModel):
    question: str


# ============================================================
# CHARGEMENT DU SYSTÈME RAG
# ============================================================

print("Chargement du système RAG...")

try:
    vectorstore = load_vectorstore()
    llm = create_llm()
    print("Système RAG chargé avec succès.")

except Exception as error:
    print(f"Erreur lors du chargement du système RAG : {error}")
    vectorstore = None
    llm = None


# ============================================================
# ROUTE RACINE
# ============================================================

@app.get("/")
def root():
    """
    Vérifie que l'API fonctionne.
    """
    return {
        "message": "API Puls-Events opérationnelle",
        "docs": "/docs",
    }


# ============================================================
# ROUTE /ASK
# ============================================================

@app.post("/ask")
def ask(request: QuestionRequest):
    """
    Pose une question au chatbot RAG.

    Exemple de requête :

    {
        "question": "Je cherche un concert à Tours"
    }
    """

    global vectorstore
    global llm

    # Vérification du système RAG
    if vectorstore is None or llm is None:
        raise HTTPException(
            status_code=503,
            detail="Le système RAG n'est pas disponible.",
        )

    # Nettoyage de la question
    question = request.question.strip()

    # Vérification question vide
    if not question:
        raise HTTPException(
            status_code=400,
            detail="La question ne peut pas être vide.",
        )

    try:
        # Appel du système RAG
        answer, documents = ask_chatbot(
            question,
            vectorstore,
            llm,
        )

        # Gestion d'une indisponibilité de Mistral
        if answer is None:
            raise HTTPException(
                status_code=503,
                detail="Le service Mistral est temporairement indisponible.",
            )

        # Préparation des sources
        sources = []

        for document in documents:
            sources.append(
                {
                    "title": document.metadata.get("title"),
                    "city": document.metadata.get("city"),
                    "date_begin": document.metadata.get("date_begin"),
                    "address": document.metadata.get("address"),
                }
            )

        # Réponse JSON
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement de la question : {error}",
        )


# ============================================================
# ROUTE /REBUILD
# ============================================================

@app.post("/rebuild")
def rebuild():
    """
    Reconstruit la base vectorielle FAISS
    à partir du dataset nettoyé.
    """

    global vectorstore

    script_path = Path("scripts/build_vectorstore.py")

    # Vérification du script
    if not script_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Le script de reconstruction est introuvable.",
        )

    try:
        print("Reconstruction de la base vectorielle...")

        # Force l'encodage UTF-8 sous Windows
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        # Exécution du script de reconstruction
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            check=True,
        )

        # Recharge l'index FAISS après reconstruction
        vectorstore = load_vectorstore()

        print("Base vectorielle reconstruite avec succès.")

        return {
            "message": "Base vectorielle reconstruite avec succès.",
            "details": result.stdout,
        }

    except subprocess.CalledProcessError as error:
        error_details = error.stderr or error.stdout

        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur lors de la reconstruction "
                f"de la base vectorielle : {error_details}"
            ),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur lors du rechargement "
                f"de la base vectorielle : {error}"
            ),
        )


# ============================================================
# LANCEMENT DE L'API
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )