import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI


VECTORSTORE_DIR = Path("vectorstore/events_faiss")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_vectorstore():
    """Charge la base vectorielle FAISS."""

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def create_llm():
    """Initialise Mistral."""

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError("MISTRAL_API_KEY introuvable")

    return ChatMistralAI(
        model="ministral-3b-2512",
        temperature=0.2,
        max_tokens=300,
        api_key=api_key
    )


def build_context(documents):
    """Construit un contexte compact pour le RAG."""

    context_parts = []

    for i, document in enumerate(documents, start=1):

        title = document.metadata.get("title", "Titre inconnu")
        date = document.metadata.get("date_begin", "Date inconnue")
        city = document.metadata.get("city", "Ville inconnue")
        address = document.metadata.get("address", "Adresse inconnue")

        # On limite volontairement le texte transmis au LLM.
        description = document.page_content[:800]

        context_parts.append(
            f"""
ÉVÉNEMENT {i}
Titre : {title}
Date : {date}
Ville : {city}
Adresse : {address}
Informations : {description}
""".strip()
        )

    return "\n\n".join(context_parts)


def ask_chatbot(question, vectorstore, llm):
    """Recherche les événements puis génère la réponse."""

    # Recherche sémantique
    documents = vectorstore.similarity_search(
        question,
        k=10
    )

    # Construction du contexte
    context = build_context(documents)

    prompt = f"""
Tu es un assistant de recommandation d'événements culturels
pour Puls-Events.

Réponds uniquement à partir du CONTEXTE fourni.

Règles :
- N'invente aucune information.
- Ne crée aucun événement qui n'existe pas dans le contexte.
- Si les informations sont insuffisantes, dis-le.
- Réponds en français.
- Sois bref et clair.
- Pour chaque recommandation, indique le titre, la date et la ville
  lorsque ces informations sont disponibles.

CONTEXTE :
{context}

QUESTION :
{question}
"""

    try:

        response = llm.invoke(prompt)

        return response.content, documents

    except httpx.HTTPStatusError as error:

        if error.response.status_code == 429:

            print("\n❌ Mistral a atteint une limite de débit (429).")
            print("Attends quelques instants puis réessaie.")

            return None, documents

        raise


def main():

    load_dotenv()

    print("=== CHATBOT RAG PULS-EVENTS ===")

    if not os.getenv("MISTRAL_API_KEY"):
        print("❌ MISTRAL_API_KEY introuvable")
        return

    print("Chargement de FAISS...")
    vectorstore = load_vectorstore()

    print("Chargement de Mistral...")
    llm = create_llm()

    question = input(
        "\nPosez votre question sur les événements : "
    )

    answer, documents = ask_chatbot(
        question,
        vectorstore,
        llm
    )

    if answer is not None:

        print("\n=== RÉPONSE DU CHATBOT ===\n")
        print(answer)

    print("\n=== ÉVÉNEMENTS RÉCUPÉRÉS PAR FAISS ===\n")

    for i, document in enumerate(documents, start=1):

        print(
            f"{i}. "
            f"{document.metadata.get('title')} | "
            f"{document.metadata.get('city')} | "
            f"{document.metadata.get('date_begin')}"
        )


if __name__ == "__main__":
    main()