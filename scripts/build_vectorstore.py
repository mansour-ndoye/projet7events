from pathlib import Path

import pandas as pd

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


INPUT_FILE = Path("data/events_clean.csv")
VECTORSTORE_DIR = Path("vectorstore/events_faiss")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_events():
    """Charge les événements nettoyés."""
    df = pd.read_csv(INPUT_FILE)

    print(f"Événements chargés : {len(df)}")

    return df


def create_documents(df):
    """Transforme les événements en Documents LangChain."""
    documents = []

    for _, row in df.iterrows():

        title = str(row.get("title", ""))
        description = str(row.get("description", ""))
        longdescription = str(row.get("longdescription", ""))
        conditions = str(row.get("conditions", ""))
        keywords = str(row.get("keywords", ""))

        city = str(row.get("city", ""))
        department = str(row.get("department", ""))
        address = str(row.get("address", ""))
        date_begin = str(row.get("date_begin", ""))
        date_end = str(row.get("date_end", ""))

        text = f"""
Titre : {title}

Description :
{description}

Description détaillée :
{longdescription}

Conditions :
{conditions}

Mots-clés :
{keywords}

Lieu :
{city}, {department}

Adresse :
{address}

Date de début :
{date_begin}

Date de fin :
{date_end}
""".strip()

        metadata = {
            "uid": str(row.get("uid", "")),
            "title": title,
            "date_begin": date_begin,
            "date_end": date_end,
            "city": city,
            "department": department,
            "address": address,
            "category": str(row.get("category", "")),
        }

        documents.append(
            Document(
                page_content=text,
                metadata=metadata
            )
        )

    print(f"Documents créés : {len(documents)}")

    return documents


def split_documents(documents):
    """Découpe les documents en chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"Chunks créés : {len(chunks)}")

    return chunks


def build_vectorstore(chunks):
    """Crée et sauvegarde l'index FAISS."""

    print("Chargement du modèle d'embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Création de l'index FAISS...")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(VECTORSTORE_DIR))

    print(f"✅ Index FAISS sauvegardé dans : {VECTORSTORE_DIR}")

    return vectorstore


def main():
    print("=== Construction de la base vectorielle ===")

    df = load_events()

    documents = create_documents(df)

    chunks = split_documents(documents)

    build_vectorstore(chunks)

    print("=== Indexation terminée ===")


if __name__ == "__main__":
    main()