"""Tests fonctionnels et d'intégration pour l'API REST FastAPI.

Ce fichier peut être exécuté :
1. Via pytest :
       uv run pytest tests/test_api.py
       uv run pytest
2. Directement en script autonome avec rapport de latence dans la console :
       uv run python tests/test_api.py

Il valide tous les endpoints requis :
- GET  /         (Interface Web HTML ou JSON si Accept: application/json)
- GET  /ui       (Interface Web Graphique HTML)
- GET  /health   (État de santé et statistiques de l'index)
- GET  /docs     (Documentation interactive Swagger)
- GET  /openapi.json (Spécification OpenAPI 3.x)
- POST /ask      (Questions valides, gestion des questions vides 400 et validation 422)
- POST /rebuild  (Reconstruction / rechargement de la base vectorielle HNSW)
- GET  /rebuild  (Rechargement via méthode GET)
- Règles métier  (Modèle HNSW exclusif & phrase de confirmation obligatoire)
- Sécurité       (Protection de /rebuild par clé d'administration X-Admin-Key)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Generator, List, Optional

# Encodage Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Assurer l'accès à la racine du projet
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# En environnement de test, forcer les identifiants d'administration pour les endpoints sécurisés
os.environ["ADMIN_USERNAME"] = "admin"
os.environ["ADMIN_PASSWORD"] = "le_mot_de_passe"

import pytest
from fastapi.testclient import TestClient
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.api import app, get_bot
from src.chatbot import EventRAGChatbot


# ============================================================================
# Doubles de test (Mock déterministe 100% offline pour les tests de l'API)
# ============================================================================

class MockChatMistralAI(BaseChatModel):
    """Simulateur de LLM Mistral pour tests d'API sans consommation de crédits."""

    model_name: str = "mock-open-mistral-nemo"

    @property
    def _llm_type(self) -> str:
        return "mock-mistral-ai"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        user_prompt = ""
        for m in reversed(messages):
            if getattr(m, "type", "") == "human" or getattr(m, "role", "") == "user":
                user_prompt = str(m.content)
                break
        if not user_prompt and messages:
            user_prompt = str(messages[-1].content)

        reply = (
            f"En réponse à votre question « {user_prompt} », nous vous recommandons "
            "le Morty Jazz Festival à Mortefontaine qui se tiendra en juillet 2026. "
            "Cet événement en plein air est gratuit et accessible à tous."
        )

        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=reply))
            ]
        )


def build_test_chatbot() -> EventRAGChatbot:
    """Crée une instance de chatbot autonome et légère dédiée aux tests fonctionnels de l'API."""
    docs = [
        Document(
            page_content="Morty Jazz Festival à Mortefontaine dans le sud de l'Oise. Festival de jazz en plein air.",
            metadata={
                "Titre": "Morty Jazz Festival",
                "Ville": "Mortefontaine",
                "URL canonique": "https://openagenda.com/events/morty-jazz-2026",
                "Première date - Début": "2026-07-24T18:00:00",
                "Dernière date - Fin": "2026-07-25T23:30:00",
                "Conditions d'accès": "Gratuit en plein air",
            },
        ),
        Document(
            page_content="Atelier numérique pour débutants à Vervins. Initiation aux outils informatiques.",
            metadata={
                "Titre": "Les RDV numériques du vendredi",
                "Ville": "Vervins",
                "URL canonique": "https://openagenda.com/events/rdv-numeriques-vervins-2026",
                "Première date - Début": "2026-05-22T14:00:00",
                "Dernière date - Fin": "2026-07-24T16:00:00",
                "Conditions d'accès": "5 € la séance",
            },
        ),
    ]
    embeddings = FakeEmbeddings(size=32)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Initialisation de l'objet métier sans réseau
    os.environ["ADMIN_USERNAME"] = "admin"
    os.environ["ADMIN_PASSWORD"] = "le_mot_de_passe"
    os.environ.setdefault("MISTRAL_API_KEY", "mock_key_for_testing")
    bot = EventRAGChatbot(
        vectorstore=vectorstore,
        llm=MockChatMistralAI(),
        embeddings=embeddings,
        top_k=2,
    )
    bot.index_path = Path("mock_index")
    return bot


# ============================================================================
# Fixture de Client de Test
# ============================================================================

@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Fournit un TestClient FastAPI configuré avec le bot de test."""
    test_bot = build_test_chatbot()
    app.dependency_overrides[get_bot] = lambda: test_bot

    with TestClient(app) as tc:
        yield tc

    app.dependency_overrides.clear()


# ============================================================================
# Tests des Endpoints
# ============================================================================

def test_root_endpoint(client: TestClient) -> None:
    """Vérifie que la racine GET / sert l'interface Web HTML (ou JSON si demandé avec Accept: application/json)."""
    # 1. Vérification que la racine GET / renvoie bien l'UI HTML pour le navigateur
    response = client.get("/")
    assert response.status_code == 200, f"Erreur statut {response.status_code}: {response.text}"
    assert "text/html" in response.headers.get("content-type", "")
    assert "<!DOCTYPE html>" in response.text or "<html" in response.text
    assert "Hauts-de-France" in response.text

    # 2. Vérification du fallback JSON avec en-tête Accept: application/json
    json_response = client.get("/", headers={"accept": "application/json"})
    assert json_response.status_code == 200
    data = json_response.json()
    assert "documentation_swagger" in data
    assert data["documentation_swagger"] == "/docs"
    assert "interface_web" in data
    assert "total_documents" in data
    assert data["status"] == "online"


def test_ui_endpoint(client: TestClient) -> None:
    """Vérifie que GET /ui renvoie l'interface graphique HTML avec succès."""
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "<!DOCTYPE html>" in response.text or "<html" in response.text
    assert "Hauts-de-France" in response.text
    assert "chatContainer" in response.text
    assert "openSettingsBtn" not in response.text


def test_health_endpoint(client: TestClient) -> None:
    """Vérifie que GET /health renvoie l'état healthy et les métadonnées."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["total_documents"] >= 2
    assert "llm_model" in data
    assert data["documentation_url"] == "/docs"


def test_swagger_documentation(client: TestClient) -> None:
    """Vérifie que la documentation Swagger UI (/docs) et OpenAPI (/openapi.json) sont accessibles."""
    # Test Swagger UI
    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200
    assert "swagger" in docs_resp.text.lower() or "html" in docs_resp.text.lower()

    # Test OpenAPI JSON schema
    openapi_resp = client.get("/openapi.json")
    assert openapi_resp.status_code == 200
    schema = openapi_resp.json()
    assert "openapi" in schema
    assert "/ask" in schema["paths"]
    assert "/rebuild" in schema["paths"]
    assert "/health" in schema["paths"]
    assert "/ui" in schema["paths"]


def test_ask_valid_question(client: TestClient) -> None:
    """Vérifie que POST /ask traite une question valide et renvoie la réponse augmentée."""
    payload = {
        "question": "Où assister à un festival de jazz en plein air en juillet ?",
        "top_k": 2,
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200, f"Erreur statut {response.status_code}: {response.text}"
    data = response.json()

    assert data["question"] == payload["question"]
    assert "Morty Jazz Festival" in data["answer"]
    assert "2026" in data["answer"]
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0
    assert data["total_sources"] == len(data["sources"])
    assert "latency_seconds" in data
    assert data["latency_seconds"] >= 0.0

    # Vérification des métadonnées de la première source
    first_source = data["sources"][0]
    assert "titre" in first_source
    assert "ville" in first_source
    assert "score_distance" in first_source
    assert "url" in first_source


def test_ask_empty_question_returns_400(client: TestClient) -> None:
    """Vérifie qu'une question vide ou composée uniquement d'espaces renvoie HTTP 400."""
    # Chaîne avec espaces uniquement
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "vide" in detail.lower()

    # Chaîne vide directe
    response_empty = client.post("/ask", json={"question": ""})
    assert response_empty.status_code in (400, 422)


def test_ask_invalid_payload_returns_422(client: TestClient) -> None:
    """Vérifie qu'un payload sans le champ obligatoire 'question' renvoie HTTP 422."""
    response = client.post("/ask", json={})
    assert response.status_code == 422

    # Test avec top_k négatif ou hors limites (ge=1, le=20)
    response_invalid_k = client.post("/ask", json={"question": "Bonjour", "top_k": 999})
    assert response_invalid_k.status_code == 422


def test_rebuild_post_endpoint(client: TestClient) -> None:
    """Vérifie que POST /rebuild recharge la base vectorielle HNSW avec succès lorsque les identifiants admin sont fournis."""
    response = client.post(
        "/rebuild",
        json={
            "index_type": "hnsw",
            "user": "admin",
            "password": "le_mot_de_passe",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "hnsw" in data["message"].lower() or "rechargée" in data["message"].lower() or "succès" in data["message"].lower()
    assert data["total_documents"] >= 2
    assert "timestamp" in data


def test_rebuild_get_method_not_allowed(client: TestClient) -> None:
    """Vérifie que GET /rebuild est désormais interdit (HTTP 405 Method Not Allowed) conformément aux standards REST."""
    response = client.get(
        "/rebuild",
        params={"index_type": "hnsw"},
        auth=("admin", "le_mot_de_passe"),
    )
    assert response.status_code == 405


def test_rebuild_triggers_reconstruction_pipeline(client: TestClient) -> None:
    """Vérifie que /rebuild déclenche bien la fonction de reconstruction avec le paramètre limit."""
    from unittest.mock import patch

    with patch("src.api.reconstruire_index_hnsw") as mock_rebuild:
        mock_rebuild.return_value = 50
        bot = app.dependency_overrides[get_bot]()
        orig_path = bot.index_path
        try:
            bot.index_path = Path("production_index_path")
            with patch.object(bot, "reload_index", return_value=50):
                resp = client.post(
                    "/rebuild",
                    json={
                        "index_type": "hnsw",
                        "user": "admin",
                        "password": "le_mot_de_passe",
                        "limit": 25,
                    },
                )
                assert resp.status_code == 200
                mock_rebuild.assert_called_once_with(limit=25)
                data = resp.json()
                assert "reconstruite" in data["message"].lower()
                assert data["total_documents"] == 50
        finally:
            bot.index_path = orig_path


def test_rebuild_validation_rules(client: TestClient) -> None:
    """Vérifie le rejet strict si les identifiants sont absents/invalides ou si l'index n'est pas HNSW."""
    # 1. POST sans identifiants -> 401
    resp_no_creds = client.post("/rebuild", json={"index_type": "hnsw"})
    assert resp_no_creds.status_code == 401
    assert "obligatoire" in resp_no_creds.json().get("detail", "").lower()

    # 2. POST avec mot de passe invalide -> 401
    resp_wrong_pwd = client.post(
        "/rebuild",
        json={"index_type": "hnsw", "user": "admin", "password": "mauvais_mot_de_passe"},
    )
    assert resp_wrong_pwd.status_code == 401
    assert "invalide" in resp_wrong_pwd.json().get("detail", "").lower()

    # 3. POST avec nom d'utilisateur invalide -> 401
    resp_wrong_user = client.post(
        "/rebuild",
        json={"index_type": "hnsw", "user": "mauvais_user", "password": "le_mot_de_passe"},
    )
    assert resp_wrong_user.status_code == 401
    assert "invalide" in resp_wrong_user.json().get("detail", "").lower()

    # 4. POST avec modèle d'index interdit (ex: flat) avec identifiants valides -> 400
    resp_flat = client.post(
        "/rebuild",
        json={
            "index_type": "flat",
            "user": "admin",
            "password": "le_mot_de_passe",
        },
    )
    assert resp_flat.status_code == 400
    assert "hnsw" in resp_flat.json().get("detail", "").lower()


def test_rebuild_security_protection(client: TestClient) -> None:
    """Vérifie la protection de l'endpoint /rebuild par utilisateur et mot de passe (Basic Auth ou JSON)."""
    user_backup = os.environ.get("ADMIN_USERNAME")
    pass_backup = os.environ.get("ADMIN_PASSWORD")
    valid_payload = {
        "index_type": "hnsw",
    }
    try:
        os.environ["ADMIN_USERNAME"] = "custom_admin"
        os.environ["ADMIN_PASSWORD"] = "custom_secret_pass_2026"

        # 1. Requête sans authentification -> refusée avec HTTP 401
        resp_unauth = client.post("/rebuild", json=valid_payload)
        assert resp_unauth.status_code == 401
        assert "refusé" in resp_unauth.json().get("detail", "").lower()

        # 2. Requête avec mauvais mot de passe en Basic Auth -> refusée avec HTTP 401
        resp_wrong_auth = client.post(
            "/rebuild",
            json=valid_payload,
            auth=("custom_admin", "mauvais_mot_de_passe"),
        )
        assert resp_wrong_auth.status_code == 401

        # 3. Requête avec mauvais mot de passe dans body JSON -> refusée avec HTTP 401
        resp_wrong_body = client.post(
            "/rebuild",
            json={"index_type": "hnsw", "user": "custom_admin", "password": "mauvais_mot_de_passe"},
        )
        assert resp_wrong_body.status_code == 401

        # 4. Requête avec les bons identifiants via HTTP Basic Auth -> HTTP 200
        resp_ok_basic = client.post(
            "/rebuild",
            json=valid_payload,
            auth=("custom_admin", "custom_secret_pass_2026"),
        )
        assert resp_ok_basic.status_code == 200
        assert resp_ok_basic.json()["status"] == "success"

        # 5. Requête avec les bons identifiants dans le corps JSON ('user') -> HTTP 200
        resp_ok_user = client.post(
            "/rebuild",
            json={"index_type": "hnsw", "user": "custom_admin", "password": "custom_secret_pass_2026"},
        )
        assert resp_ok_user.status_code == 200
        assert resp_ok_user.json()["status"] == "success"

        # 6. Requête avec les bons identifiants dans le corps JSON ('username') -> HTTP 200
        resp_ok_username = client.post(
            "/rebuild",
            json={"index_type": "hnsw", "username": "custom_admin", "password": "custom_secret_pass_2026"},
        )
        assert resp_ok_username.status_code == 200
        assert resp_ok_username.json()["status"] == "success"

    finally:
        # Nettoyage et restauration de l'environnement
        if user_backup is not None:
            os.environ["ADMIN_USERNAME"] = user_backup
        else:
            os.environ.pop("ADMIN_USERNAME", None)
        if pass_backup is not None:
            os.environ["ADMIN_PASSWORD"] = pass_backup
        else:
            os.environ.pop("ADMIN_PASSWORD", None)


# ============================================================================
# Exécution Directe (Script CLI)
# ============================================================================

def run_functional_test_suite() -> bool:
    """Exécute la suite de tests fonctionnels et affiche un compte-rendu élégant."""
    print("=" * 80)
    print("🧪 EXÉCUTION DES TESTS FONCTIONNELS DE L'API REST (FastAPI)")
    print("=" * 80)

    test_bot = build_test_chatbot()
    app.dependency_overrides[get_bot] = lambda: test_bot

    tests = [
        ("GET  / (Racine & Liens)", test_root_endpoint),
        ("GET  /ui (Interface Web Graphique)", test_ui_endpoint),
        ("GET  /health (Santé & Métadonnées)", test_health_endpoint),
        ("GET  /docs & /openapi.json (Swagger UI)", test_swagger_documentation),
        ("POST /ask (Question valide & Recommandation)", test_ask_valid_question),
        ("POST /ask (Gestion des questions vides - 400)", test_ask_empty_question_returns_400),
        ("POST /ask (Validation de schéma - 422)", test_ask_invalid_payload_returns_422),
        ("POST /rebuild (Rechargement base vectorielle HNSW)", test_rebuild_post_endpoint),
        ("POST /rebuild (Pipeline de reconstruction)", test_rebuild_triggers_reconstruction_pipeline),
        ("GET  /rebuild (Interdiction HTTP 405 Method Not Allowed)", test_rebuild_get_method_not_allowed),
        ("VALIDATION /rebuild (Modèle HNSW exclusif & Rejet identifiants)", test_rebuild_validation_rules),
        ("SÉCURITÉ /rebuild (Authentification user/password Basic Auth & JSON 401)", test_rebuild_security_protection),
    ]

    success_count = 0
    start_total = time.perf_counter()

    with TestClient(app) as tc:
        for name, test_fn in tests:
            t0 = time.perf_counter()
            try:
                test_fn(tc)
                elapsed = (time.perf_counter() - t0) * 1000
                print(f"  ✅ PASS [{elapsed:5.1f} ms] : {name}")
                success_count += 1
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000
                print(f"  ❌ FAIL [{elapsed:5.1f} ms] : {name}")
                print(f"     👉 Détail : {exc}")

    app.dependency_overrides.clear()
    total_time = time.perf_counter() - start_total

    print("=" * 80)
    print(f"📊 RÉSULTAT : {success_count}/{len(tests)} tests réussis en {total_time:.3f} s.")
    if success_count == len(tests):
        print("🎉 TOUS LES TESTS DE L'API ONT RÉUSSI AVEC SUCCÈS !")
        print("=" * 80)
        return True
    else:
        print("⚠️ CERTAINS TESTS ONT ÉCHOUÉ !")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = run_functional_test_suite()
    sys.exit(0 if success else 1)
