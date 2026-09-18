# Projet 7 - Système RAG

Projet de conception et déploiement d'un système RAG appliqué aux événements publics 2026 de la région Hauts-de-France (données OpenAgenda).


---

## 🚀 Démarrage

### 1. Installation

Le projet utilise [`uv`](https://github.com/astral-sh/uv) pour gérer les dépendances et l'environnement virtuel :

```bash
uv venv
uv sync
```

### 2. Configuration

Ajouter un fichier `.env` dans le dossier `src/` (ou à la racine) avec votre clé d'API Mistral :

```env
MISTRAL_API_KEY=votre_cle_api
```

---

## 💻 Utilisation et Architecture

Le chatbot est conçu sous forme de service autonome dans [`src/chatbot.py`](src/chatbot.py) (`EventRAGChatbot`), totalement prêt à être branché sur la future interface graphique (Streamlit, Chainlit, etc.) :

```python
from src.chatbot import EventRAGChatbot

# Initialisation du service RAG
bot = EventRAGChatbot()

# Génération d'une recommandation pour l'interface graphique
reponse = bot.chat("Quels concerts de jazz sont prévus cet été ?")
```

### 1. Vérifier l'état du projet
```bash
uv run python main.py
```

### 2. Inspecter la base vectorielle (100% hors-ligne - 0 crédit)
Pour auditer les données locales (nombre de vecteurs, villes, mots-clés) sans appel API :
```bash
uv run python main.py --inspect
# Ou avec recherche de mot-clé :
uv run python main.py --inspect --keyword jazz
```

### 3. Lancer l'Interface Graphique Web (Très légère)
Pour interagir visuellement avec le chatbot et tester les recommandations en direct dans le navigateur :
```bash
uv run python main.py --ui
```
Cette commande démarre le serveur et ouvre automatiquement votre navigateur sur **[http://127.0.0.1:8000](http://127.0.0.1:8000)** (disponible directement à la racine et sur `/ui`).
- 💬 **Conversation fluide & interactive** : suggestions en 1 clic, saisie libre, sélecteur de Top-K.
- 📚 **Sources FAISS détaillées** : cartes déroulantes pour chaque recommandation (titre, ville, score distance, lien OpenAgenda).
- ⏱️ **Indicateurs de performance** : mesure de la latence en secondes et statut en direct de l'API.
- ⚙️ **Administration intégrée** : bouton *Rebuild* protégé par confirmation pour recharger ou basculer l'index à chaud.

### 4. Lancer l'API REST FastAPI
Pour lancer uniquement le serveur d'API (sans ouvrir automatiquement le navigateur) :
```bash
uv run python main.py --serve
# ou avec rechargement à chaud : uv run uvicorn src.api:app --reload --port 8000
```
- **Interface Graphique Web** : [http://127.0.0.1:8000](http://127.0.0.1:8000) (ou `/ui`)
- **Documentation interactive Swagger UI** : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Documentation ReDoc** : [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Santé & Métadonnées** : `GET http://127.0.0.1:8000/health`
- **Poser une question** : `POST http://127.0.0.1:8000/ask` (body : `{"question": "...", "top_k": 4}`)
- **Recharger / Reconstruire l'index** : `POST http://127.0.0.1:8000/rebuild` ou `GET http://127.0.0.1:8000/rebuild`
  *(Protégé par pop-up de confirmation dans l'UI et header `X-Admin-Key` si `ADMIN_API_KEY` est configuré)*

### 5. Exécuter les tests fonctionnels de l'API
Pour valider l'ensemble des endpoints HTTP, l'interface graphique à la racine `/` et `/ui`, les codes d'erreurs (400, 422, 403), et Swagger :
```bash
uv run python api_test.py
```

### 6. Évaluer les scénarios d'interaction RAG (Vérité terrain & Rapport HTML)
Pour exécuter la suite de 5 scénarios de test et les comparer aux **réponses de référence annotées par l'humain** :
```bash
uv run python src/evaluer_scenarios.py
# ou : uv run python main.py --eval
# génère par défaut : rapport/rapport_evaluation.html
```

### 7. Mode console de test (CLI isolé)
Pour échanger avec le bot directement dans le terminal (outil de test) :
```bash
uv run python src/cli.py
# ou : uv run python main.py --cli
```

### 8. Exécuter l'ensemble des tests automatisés
```bash
uv run pytest
```

---

## 📊 Premiers résultats (Flat L2 vs HNSW)

Sur un corpus de ~24 700 événements indexés :
- **Flat L2 :** recherche exacte, latence moyenne d'environ **16.4 ms / requête**.
- **HNSW :** recherche approximative, latence moyenne d'environ **0.55 ms / requête** (~x30 plus rapide).

---

## 📁 Structure du projet

```text
projet_7_systeme_RAG/
├── rapport/                                        # Rapports d'évaluation HTML générés
│   └── rapport_evaluation.html
├── resources/                                      # Données brutes CSV
├── src/
│   ├── api.py                                      # API REST FastAPI (/ask, /rebuild, /docs)
│   ├── chatbot.py                                  # Moteur RAG métier (classe EventRAGChatbot)
│   ├── cli.py                                      # Interface console de test isolée
│   ├── evaluer_scenarios.py                       # Démonstration sur 5 scénarios & rapport HTML
│   ├── inspect_index.py                            # Inspection locale hors-ligne (0 crédit)
│   ├── recuperation_data_et_vectorisation.ipynb   # Notebook de préparation et benchmarks
│   ├── static/
│   │   └── index.html                              # Interface Web graphique interactive
│   ├── mon_index_langchain_evenements/             # Index FAISS Flat L2
│   ├── mon_index_langchain_evenements_hnsw_rapide/ # Index FAISS HNSW optimisé
│   └── .env                                        # Clé API Mistral (non versionné)
├── tests/
│   ├── conftest.py
│   ├── test_api.py                                 # Tests d'intégration API REST
│   └── test_chatbot.py                             # Tests unitaires et RAG (100% offline)
├── api_test.py                                     # Script de test fonctionnel autonome de l'API
├── main.py                                         # Point d'orchestration (CLI, Serveur, Audit)
├── pyproject.toml
├── uv.lock
└── README.md
```
