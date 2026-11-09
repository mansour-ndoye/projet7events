from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


VECTORSTORE_DIR = Path("vectorstore/events_faiss")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print("Chargement de la base vectorielle...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    query = "Je cherche un concert à Tours"

    print(f"\nRecherche : {query}\n")

    results = vectorstore.similarity_search(query, k=5)

    for i, document in enumerate(results, start=1):
        print(f"--- Résultat {i} ---")
        print("Titre :", document.metadata.get("title"))
        print("Ville :", document.metadata.get("city"))
        print("Date :", document.metadata.get("date_begin"))
        print()


if __name__ == "__main__":
    main()