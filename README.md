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

Créer un fichier `.env` à la racine du projet à partir du modèle fourni :

```bash
cp .env.example .env   # ou sous Windows: copy .env.example .env
```

Puis renseigner votre clé d'API Mistral et vos identifiants d'administration dans ce fichier `.env` unique :

```env
MISTRAL_API_KEY=votre_cle_api_mistral
ADMIN_USERNAME=admin
ADMIN_PASSWORD=votre_mot_de_passe
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

### 1. Création de la base vectorielle HNSW (Première étape)
Avant de lancer le chatbot ou l'API, initialisez la base vectorielle FAISS HNSW en interrogeant l'API publique OpenAgenda (OpenDataSoft) :
```bash
# Création complète de la base vectorielle HNSW
uv run python src/creer_index_hnsw.py

# Ou création d'un index réduit pour tester (ex: 50 événements)
uv run python src/creer_index_hnsw.py --limit 50

# Mode simulation (teste la récupération sans appeler l'API d'embeddings)
uv run python src/creer_index_hnsw.py --dry --limit 10
```

### 2. Vérifier l'état du projet
```bash
uv run python main.py
```

### 3. Inspecter la base vectorielle (100% hors-ligne - 0 crédit)
Pour auditer les données locales (nombre de vecteurs, villes, mots-clés) sans appel API :
```bash
uv run python main.py --inspect
# Ou avec recherche de mot-clé :
uv run python main.py --inspect --keyword jazz
```

### 4. Lancer l'Interface Graphique Web (Très légère)
Pour interagir visuellement avec le chatbot et tester les recommandations en direct dans le navigateur :
```bash
uv run python main.py --ui
```
Cette commande démarre le serveur et ouvre automatiquement votre navigateur sur **[http://127.0.0.1:8000](http://127.0.0.1:8000)** (disponible directement à la racine et sur `/ui`).
- 💬 **Conversation fluide & interactive** : suggestions en 1 clic, saisie libre, sélecteur de Top-K.
- 📚 **Sources FAISS détaillées** : cartes déroulantes pour chaque recommandation (titre, ville, score distance, lien OpenAgenda).
- ⏱️ **Indicateurs de performance** : mesure de la latence en secondes et statut en direct de l'API.
- ⚙️ **Administration intégrée** : bouton *Rebuild* protégé par confirmation pour recharger ou basculer l'index à chaud.

### 5. Lancer l'API REST FastAPI
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
- **Recharger / Reconstruire l'index** : `POST http://127.0.0.1:8000/rebuild`
  *(Protégé par authentification HTTP Basic Auth ou identifiants JSON via `ADMIN_USERNAME` et `ADMIN_PASSWORD`)*

### 6. Déploiement avec Docker & Docker Compose

Le chatbot est entièrement conteneurisé. Au démarrage du conteneur, l'application affiche immédiatement dans les logs Docker les URL directes d'accès à l'application.

#### Méthode A : Avec Docker Compose (Recommandé)
Le fichier `docker-compose.yml` configure automatiquement le port, la politique de redémarrage, le healthcheck et charge vos variables d'environnement directement depuis le fichier `.env` à la racine :

```bash
# Construire et lancer le conteneur en arrière-plan
docker compose up --build -d

# Consulter les logs de démarrage (affiche l'URL d'accès)
docker compose logs -f

# Arrêter le conteneur
docker compose down
```

#### Méthode B : Avec Docker CLI classique
```bash
# 1. Construction de l'image Docker
docker build -t rag-evenements-chatbot .

# 2. Lancement du conteneur avec le fichier d'environnement 
docker run -d \
  --name rag-chatbot \
  -p 8000:8000 \
  --env-file .env \
  rag-evenements-chatbot

# 3. Affichage des logs et des URL d'accès
docker logs -f rag-chatbot
```

#### Accès aux services conteneurisés :
- 🌐 **Interface Web (Chatbot RAG)** : [http://localhost:8000](http://localhost:8000) (ou `http://127.0.0.1:8000/ui`)
- 📖 **Documentation Swagger OpenAPI** : [http://localhost:8000/docs](http://localhost:8000/docs)
- 🩺 **Contrôle de santé (Healthcheck)** : [http://localhost:8000/health](http://localhost:8000/health)

---

### 7. Exécuter les tests fonctionnels de l'API
Pour valider l'ensemble des endpoints HTTP, l'interface graphique à la racine `/` et `/ui`, les codes d'erreurs (400, 422, 403), et Swagger :
```bash
uv run python tests/test_api.py
```

### 8. Évaluer les scénarios d'interaction RAG (Vérité terrain & Rapport HTML)
Pour exécuter la suite de 5 scénarios de test et les comparer aux **réponses de référence annotées par l'humain** :
```bash
uv run python src/evaluation/evaluer_scenarios.py
# ou : uv run python main.py --eval
# génère par défaut : rapport/rapport_evaluation.html
```

### 9. Mode console de test (CLI isolé)
Pour échanger avec le bot directement dans le terminal (outil de test) :
```bash
uv run python src/cli.py
# ou : uv run python main.py --cli
```

### 10. Exécuter l'ensemble des tests automatisés
```bash
uv run pytest

# Ou avec génération du rapport interactif HTML dans tests/rapport/ :
uv run pytest --html=tests/rapport/rapport_tests_unitaires.html --self-contained-html
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
├── .env                                            # Configuration locale & Docker (non versionné)
├── .env.example                                    # Modèle documenté des variables d'environnement
├── .gitattributes                                  # Configuration Git (binaires & fins de ligne)
├── diagram_uml.png                                 # Schéma d'architecture global du système RAG généré avec plantuml
├── Dockerfile                                      # Image Docker multi-plateforme optimisée
├── docker-compose.yml                              # Orchestration Docker Compose
├── RAPPORT_TECHNIQUE.md                            # Rapport technique d'ingénierie et de soutenance
├── rapport/                                        # Rapports d'évaluation HTML générés
│   └── rapport_evaluation.html
├── resources/                                      # Données brutes CSV
├── src/
│   ├── api.py                                      # API REST FastAPI (/ask, /rebuild, /docs, /)
│   ├── chatbot.py                                  # Moteur RAG métier (classe EventRAGChatbot)
│   ├── cli.py                                      # Interface console de test isolée
│   ├── creer_index_hnsw.py                         # Création autonome de l'index FAISS HNSW via l'API
│   ├── inspect_index.py                            # Inspection locale hors-ligne (0 crédit)
│   ├── evaluation/                                 # Suite d'évaluation et de benchmark
│   │   ├── __init__.py
│   │   ├── scenarios.py                            # Définition des scénarios et vérités terrain
│   │   ├── evaluate_rag.py                         # Évaluation standardisée Ragas (LLM-as-a-judge)
│   │   ├── evaluer_scenarios.py                    # Démonstration sur 5 scénarios & rapport HTML
│   │   └── rapport_template.html                   # Gabarit HTML Jinja2 du rapport d'évaluation
│   ├── static/
│   │   └── ui-chatbot.html                         # Interface Web graphique interactive
│   └── mon_index_langchain_evenements_hnsw_rapide/ # Index FAISS HNSW optimisé (inclus dans l'image)
├── tests/
│   ├── conftest.py
│   ├── test_api.py                                 # Tests fonctionnels et d'intégration de l'API REST
│   ├── test_chatbot.py                             # Tests unitaires et RAG (100% offline)
│   ├── test_creer_index_hnsw.py                    # Tests unitaires du script d'indexation (mode dry)
│   └── rapport/                                    # Rapports de tests automatisés (HTML & XML)
├── main.py                                         # Point d'orchestration (CLI, Serveur, Audit)
├── pyproject.toml
├── uv.lock
└── README.md
```
