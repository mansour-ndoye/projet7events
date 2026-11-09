\# Puls-Events - Assistant de recommandation d'événements culturels



\## Objectif



Ce projet consiste à développer un assistant conversationnel capable

de répondre aux questions des utilisateurs concernant des événements

culturels à venir.



Le système repose sur une architecture RAG (Retrieval-Augmented Generation)

combinant :



\- une collecte de données d'événements culturels ;

\- un traitement et nettoyage des données ;

\- des embeddings vectoriels ;

\- FAISS pour la recherche de similarité ;

\- LangChain pour l'orchestration du pipeline ;

\- Mistral pour la génération des réponses ;

\- une API REST avec FastAPI.



\## Installation



\### Prérequis



\- Python >= 3.8

\- pip

\- connexion Internet



\### Création de l'environnement virtuel



```bash

python -m venv .venv

