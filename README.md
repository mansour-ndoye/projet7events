# Puls-Events — Assistant de recommandation d'événements culturels

## 1. Présentation du projet

Puls-Events est un assistant conversationnel qui aide les utilisateurs à découvrir des événements culturels à venir, notamment dans la ville de Tours et le département d'Indre-et-Loire.

Le projet repose sur une architecture **RAG (Retrieval-Augmented Generation)** qui combine la recherche d'informations dans une base documentaire et la génération de réponses par un modèle de langage.

L'objectif est de fournir des réponses contextualisées à partir des données d'événements collectées et indexées.

## 2. Technologies utilisées

- **Python** : développement du projet.
- **OpenAgenda / OpenDataSoft** : source des événements culturels.
- **Pandas** : nettoyage et préparation des données.
- **Sentence Transformers** : génération des embeddings.
- **FAISS** : recherche vectorielle par similarité.
- **LangChain** : orchestration du pipeline RAG.
- **Mistral AI** : génération des réponses.
- **FastAPI** : création de l'API REST.
- **Docker** : conteneurisation de l'application.
- **Pytest** : tests automatisés.

## 3. Architecture du système RAG

Le fonctionnement du projet se déroule en plusieurs étapes :

1. **Collecte des données** : récupération des événements depuis OpenAgenda / OpenDataSoft.
2. **Prétraitement** : nettoyage des textes, suppression des événements incomplets et filtrage géographique et temporel.
3. **Transformation documentaire** : conversion des événements en documents exploitables par LangChain.
4. **Découpage** : segmentation des documents avec des chunks de 600 caractères et un chevauchement de 100 caractères.
5. **Vectorisation** : création des embeddings avec `sentence-transformers/all-MiniLM-L6-v2`.
6. **Indexation** : stockage des vecteurs dans FAISS.
7. **Recherche** : récupération des 10 documents les plus similaires à la question.
8. **Génération** : transmission du contexte récupéré à Mistral pour produire une réponse.

### Schéma simplifié

```text
OpenAgenda / OpenDataSoft
          |
          v
   Collecte des événements
          |
          v
    Nettoyage des données
          |
          v
   Documents LangChain
          |
          v
     Découpage en chunks
          |
          v
  Embeddings Sentence Transformers
          |
          v
       Index FAISS
          |
          v
Question utilisateur
          |
          v
 Recherche vectorielle (Top 10)
          |
          v
 Contexte documentaire récupéré
          |
          v
       Modèle Mistral
          |
          v
    Réponse contextualisée
```

## 4. Structure du dépôt

```text
projet7events/
│
├── data/
│   ├── events_raw.json
│   └── events_clean.csv
│
├── scripts/
│   ├── fetch_events.py
│   ├── preprocess_events.py
│   ├── build_vectorstore.py
│   ├── search_vectorstore.py
│   └── rag_chatbot.py
│
├── tests/
│   ├── test_api.py
│   ├── test_environment.py
│   ├── test_preprocess_events.py
│   └── test_rag_chatbot.py
│
├── api.py
├── mistral_manual.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

Le répertoire `vectorstore/events_faiss` est généré localement lors de la construction de l'index et n'est pas versionné sur GitHub.

## 5. Installation

### Prérequis

- Python 3.11 recommandé
- pip
- Git
- Une connexion Internet
- Une clé API Mistral

### Cloner le dépôt

```bash
git clone https://github.com/mansour-ndoye/projet7events.git
cd projet7events
```

### Créer un environnement virtuel

Sous Windows PowerShell :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Installer les dépendances

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Configuration de la clé API Mistral

Créer un fichier `.env` à la racine du projet.

Y ajouter :

```text
MISTRAL_API_KEY=VOTRE_CLE_API_MISTRAL
```

Remplacer la valeur par votre propre clé API.

**Important :** ne jamais publier le fichier `.env` sur GitHub. Il est exclu du dépôt grâce au `.gitignore`.

## 7. Préparation des données et construction de l'index

Les données sont disponibles dans le dossier `data/`.

Pour récupérer de nouveaux événements :

```powershell
python scripts/fetch_events.py
```

Pour nettoyer et filtrer les événements :

```powershell
python scripts/preprocess_events.py
```

Pour construire la base vectorielle FAISS :

```powershell
python scripts/build_vectorstore.py
```

L'index est sauvegardé dans :

```text
vectorstore/events_faiss/
```

La première exécution peut télécharger le modèle d'embeddings depuis Hugging Face.

## 8. Lancement de l'API REST

Après avoir configuré la clé Mistral et construit l'index FAISS :

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

L'API est accessible à l'adresse :

http://127.0.0.1:8000

La documentation interactive Swagger est disponible ici :

http://127.0.0.1:8000/docs

### Endpoints disponibles

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/` | Vérification du fonctionnement de l'API |
| POST | `/ask` | Interrogation du chatbot |
| POST | `/rebuild` | Reconstruction de l'index FAISS |

### Exemple de requête au chatbot

```json
{
  "question": "Je cherche un concert à Tours"
}
```

Exemple de réponse :

```json
{
  "question": "Je cherche un concert à Tours",
  "answer": "Voici les événements correspondant à votre recherche...",
  "sources": [
    {
      "title": "Exemple d'événement",
      "city": "Tours",
      "date_begin": "2026-12-15",
      "address": "Tours"
    }
  ]
}
```

Les valeurs ci-dessus illustrent le format de réponse ; les événements retournés dépendent des données indexées.

## 9. Tests automatisés

Le projet dispose de tests unitaires avec Pytest.

Pour exécuter tous les tests :

```powershell
python -m pytest -v
```

Les tests couvrent :

- les endpoints de l'API REST ;
- l'environnement et les dépendances ;
- le nettoyage et le filtrage des événements ;
- la construction du contexte RAG ;
- la recherche documentaire ;
- la gestion des erreurs de l'API Mistral ;
- le chargement du modèle et de l'index vectoriel.

**Résultat obtenu lors de la validation du projet : 25 tests réussis sur 25.**

Deux avertissements de dépréciation ont été observés, sans échec de test.

## 10. Exécution avec Docker

Après avoir installé Docker et construit l'index FAISS localement :

Construire l'image :

```powershell
docker build -t puls-events-api .
```

Lancer le conteneur :

```powershell
docker run --env-file .env -p 8000:8000 --name puls-events-container puls-events-api
```

L'API est alors accessible sur :

http://localhost:8000/docs

Le fichier `.env` est transmis au conteneur au démarrage et ne doit pas être intégré à l'image.

## 11. Limites et perspectives d'amélioration

Plusieurs axes peuvent améliorer le projet :

- améliorer la précision des recherches géographiques et temporelles ;
- ajouter des filtres structurés par ville, date et catégorie ;
- améliorer la fiabilité de l'affichage des dates ;
- enrichir le jeu de test avec des évaluations humaines ;
- ajouter une interface utilisateur dédiée ;
- mettre en place une intégration continue avec GitHub Actions.

## 12. Auteur et dépôt

**Projet :** Puls-Events

**Dépôt GitHub :** https://github.com/mansour-ndoye/projet7events

**Description :** assistant conversationnel de recommandation d'événements culturels basé sur une architecture RAG.