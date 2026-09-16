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

### 3. Évaluer les scénarios d'interaction RAG (Vérité terrain & Rapport HTML)
Pour exécuter la suite de 5 scénarios de test et les comparer aux **réponses de référence annotées par l'humain** (calcul de l'Exact Match factuel, similarité lexicale F1, classification qualitative et génération du rapport HTML avec horodatage d'exécution) :
```bash
uv run python src/evaluer_scenarios.py
# ou : uv run python main.py --eval
# génère par défaut : rapport/rapport_evaluation.html
# avec chemin personnalisé : uv run python src/evaluer_scenarios.py --output rapport/mon_rapport.html
```

### 4. Mode console de test (CLI isolé)
Pour échanger avec le bot directement dans le terminal (outil de test) :
```bash
uv run python src/cli.py
# ou : uv run python main.py --cli
```

### 5. Exécuter les tests automatisés
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
│   ├── chatbot.py                                  # Moteur RAG métier (classe EventRAGChatbot pour GUI/API)
│   ├── cli.py                                      # Interface console de test isolée
│   ├── evaluer_scenarios.py                       # Démonstration sur 5 scénarios
│   ├── inspect_index.py                            # Inspection locale hors-ligne (0 crédit)
│   ├── recuperation_data_et_vectorisation.ipynb   # Notebook de préparation et benchmarks
│   ├── mon_index_langchain_evenements/             # Index FAISS Flat L2
│   ├── mon_index_langchain_evenements_hnsw_rapide/ # Index FAISS HNSW optimisé
│   └── .env                                        # Clé API Mistral
├── tests/
│   ├── conftest.py
│   └── test_chatbot.py                             # Tests unitaires et d'intégration
├── main.py                                         # Point d'orchestration et vérification d'état
├── pyproject.toml
├── uv.lock
└── README.md
```
