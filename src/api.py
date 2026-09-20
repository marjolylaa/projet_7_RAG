"""API REST FastAPI pour le système RAG d'événements Hauts-de-France 2026.

Expose les points d'entrée :
- POST /ask     : Pose une question et génère une recommandation augmentée par RAG
- POST /rebuild : Reconstruit la base vectorielle FAISS HNSW à la demande (sécurisé)
- GET  /health  : Vérification de l'état du système RAG et métadonnées
- GET  /        : Racine d'accueil avec liens vers la documentation Swagger (/docs)
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

# Configuration de l'encodage sur Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Assurer que la racine du projet et le dossier src/ sont systématiquement dans sys.path
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent if _current_dir.name == "src" else _current_dir
for _path_item in (str(_root_dir), str(_root_dir / "src"), str(_current_dir)):
    if _path_item not in sys.path:
        sys.path.insert(0, _path_item)

import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

try:
    from src.chatbot import EventRAGChatbot, get_project_root
except ModuleNotFoundError:
    from chatbot import EventRAGChatbot, get_project_root

try:
    from src.creer_index_hnsw import reconstruire_index_hnsw
except ModuleNotFoundError:
    from creer_index_hnsw import reconstruire_index_hnsw

STATIC_DIR = Path(__file__).resolve().parent / "static"
UI_HTML_PATH = STATIC_DIR / "ui-chatbot.html"
INDEX_HTML_PATH = UI_HTML_PATH  # Alias de rétrocompatibilité


# ============================================================================
# Modèles Pydantic (Validation & Documentation OpenAPI)
# ============================================================================

class SourceItem(BaseModel):
    """Métadonnées d'un événement source extrait de la base vectorielle."""
    titre: str = Field(..., description="Titre officiel de l'événement.")
    ville: str = Field(..., description="Commune ou ville de l'événement dans les Hauts-de-France.")
    score_distance: float = Field(..., description="Score de distance L2 FAISS (plus proche de 0 = plus pertinent).")
    url: str = Field("", description="Lien officiel OpenAgenda de l'événement.")
    description_preview: str = Field(..., description="Aperçu textuel de la description indexée.")


class QuestionRequest(BaseModel):
    """Données d'entrée pour poser une question au système RAG."""
    question: str = Field(
        ...,
        min_length=1,
        description="Question ou recherche d'activité formulée par l'utilisateur.",
        examples=["Quels sont les concerts de jazz en plein air prévus cet été ?"],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Nombre optionnel d'événements sources à récupérer (par défaut: 4, min: 1, max: 20).",
        examples=[4],
    )


class AnswerResponse(BaseModel):
    """Réponse augmentée générée par le système RAG."""
    question: str = Field(..., description="Question originale posée.")
    answer: str = Field(..., description="Réponse personnalisée générée par le LLM Mistral à partir du contexte.")
    sources: List[SourceItem] = Field(default_factory=list, description="Liste des événements sources pertinents extraits de FAISS.")
    total_sources: int = Field(..., description="Nombre d'événements sources récupérés.")
    latency_seconds: float = Field(..., description="Temps total d'exécution en secondes.")
    timestamp: str = Field(..., description="Horodatage ISO de la requête.")


class RebuildRequest(BaseModel):
    """Paramètres pour la reconstruction de la base vectorielle."""
    index_type: Optional[str] = Field(
        default="hnsw",
        description="Type ou modèle d'index à reconstruire (exclusivement 'hnsw').",
        examples=["hnsw"],
    )
    user: Optional[str] = Field(
        default=None,
        description="Nom d'utilisateur administrateur (ou transmis via HTTP Basic Auth).",
        examples=["admin"],
    )
    username: Optional[str] = Field(
        default=None,
        description="Alias de 'user' (nom d'utilisateur administrateur).",
        examples=["admin"],
    )
    password: Optional[str] = Field(
        default=None,
        description="Mot de passe administrateur (ou transmis via HTTP Basic Auth).",
        examples=["le_mot_de_passe"],
    )
    limit: Optional[int] = Field(
        default=None,
        description="Nombre maximum d'événements à récupérer et indexer (optionnel, ex: 50 pour un test rapide).",
        examples=[50],
    )


class RebuildResponse(BaseModel):
    """Compte-rendu de la reconstruction ou du rechargement de la base vectorielle."""
    status: str = Field("success", description="Statut de l'opération.")
    message: str = Field(..., description="Message explicatif de l'opération.")
    total_documents: int = Field(..., description="Nombre total d'événements chargés dans la base vectorielle.")
    index_path: str = Field(..., description="Chemin de l'index FAISS rechargé.")
    timestamp: str = Field(..., description="Horodatage ISO de la reconstruction.")


class HealthResponse(BaseModel):
    """Informations d'état et métadonnées du service RAG."""
    status: str = Field("healthy", description="État général de l'API ('healthy').")
    service: str = Field("API RAG Événements Hauts-de-France 2026")
    version: str = Field("1.0.0")
    total_documents: int = Field(..., description="Nombre total d'événements indexés.")
    llm_model: str = Field(..., description="Nom du modèle LLM utilisé.")
    embedding_model: str = Field(..., description="Nom du modèle d'embeddings utilisé.")
    index_path: str = Field(..., description="Chemin d'accès de l'index vectoriel actif.")
    documentation_url: str = Field("/docs", description="Lien relatif vers Swagger UI.")


# ============================================================================
# Gestion du Cycle de Vie (Lifespan) & Dépendances
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise le bot RAG au démarrage du serveur et le conserve en mémoire."""
    # En environnement de test avec mock injecté via dependency_overrides
    if get_bot in app.dependency_overrides:
        app.state.bot = app.dependency_overrides[get_bot]()
    elif not hasattr(app.state, "bot") or app.state.bot is None:
        try:
            print("🚀 Démarrage de l'API RAG : initialisation du chatbot et de l'index vectoriel...")
            app.state.bot = EventRAGChatbot()
            stats = app.state.bot.get_stats()
            print(f"✅ Bot RAG prêt ! ({stats['total_documents']} événements indexés, index: {stats['index_path']})")
        except Exception as exc:
            print(f"⚠️ Avertissement lors de l'initialisation du chatbot : {exc}")
            app.state.bot = None

    host = os.getenv("HOST", "0.0.0.0")
    port = os.getenv("PORT", "8000")
    display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    base_url = f"http://{display_host}:{port}"
    print("=" * 72, flush=True)
    print(f"🌐 Application RAG en ligne : {base_url}", flush=True)
    print(f"🎭 Interface Web             : {base_url}/ (ou {base_url}/ui)", flush=True)
    print(f"📖 Documentation API Swagger : {base_url}/docs", flush=True)
    print("=" * 72, flush=True)

    yield

    # Nettoyage à l'arrêt si nécessaire
    print("🛑 Arrêt de l'API RAG.", flush=True)


# Initialisation de l'application FastAPI
app = FastAPI(
    title="API RAG - Recommandation d'Événements Hauts-de-France 2026",
    description=(
        "API REST permettant aux équipes métier d'interroger le système RAG d'événements culturels et publics "
        "de la région Hauts-de-France pour l'année 2026.\n\n"
        "- **`/ask` (POST)** : Posez une question et recevez une recommandation personnalisée augmentée.\n"
        "- **`/rebuild` (POST)** : Reconstruisez la base vectorielle FAISS HNSW à chaud (sécurisé).\n"
        "- **`/health` (GET)** : Contrôlez la santé du service et le nombre d'événements indexés.\n"
        "- **`/docs`** : Documentation interactive Swagger OpenAPI."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Support CORS pour intégration fluide avec de futurs frontends web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_bot(request: Request) -> EventRAGChatbot:
    """Dépendance FastAPI injectant l'instance unique du chatbot."""
    bot = getattr(request.app.state, "bot", None)
    if bot is None:
        # Initialisation paresseuse de secours
        try:
            bot = EventRAGChatbot()
            request.app.state.bot = bot
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Le moteur RAG n'est pas encore initialisé : {exc}",
            )
    return bot


security_basic = HTTPBasic(auto_error=False)


def extract_admin_credentials(
    credentials: Optional[HTTPBasicCredentials] = Depends(security_basic),
    x_user: Optional[str] = Header(None, alias="X-User"),
    x_password: Optional[str] = Header(None, alias="X-Password"),
) -> tuple[Optional[str], Optional[str]]:
    """Extrait les identifiants administrateur depuis HTTP Basic Auth ou les en-têtes HTTP personnalisés."""
    if credentials and credentials.username:
        return (credentials.username.strip(), (credentials.password or "").strip())
    if x_user and x_user.strip():
        return (x_user.strip(), (x_password or "").strip())
    return (None, None)


# ============================================================================
# Routes HTTP de l'API REST
# ============================================================================

@app.get(
    "/",
    response_class=FileResponse,
    summary="Interface Web graphique interactive (disponible à la racine)",
    tags=["Interface", "Général"],
)
async def root(
    request: Request,
    bot: EventRAGChatbot = Depends(get_bot),
) -> Any:
    """Point d'entrée racine : sert directement l'interface Web interactive (http://127.0.0.1:8000).

    Si le client demande explicitement du JSON (ex: Accept: application/json sans text/html),
    renvoie les métadonnées et l'état du service au format JSON.
    """
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        stats = bot.get_stats()
        return JSONResponse(
            content={
                "message": "Bienvenue sur l'API REST du Système RAG Événements Hauts-de-France 2026",
                "interface_web": "/",
                "documentation_swagger": "/docs",
                "documentation_redoc": "/redoc",
                "total_documents": stats["total_documents"],
                "status": "online",
                "timestamp": datetime.now().isoformat(),
            }
        )

    if not UI_HTML_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fichier d'interface statique introuvable (src/static/ui-chatbot.html).",
        )
    return FileResponse(UI_HTML_PATH, media_type="text/html")


@app.get(
    "/ui",
    response_class=FileResponse,
    summary="Interface Web graphique interactive (alias /ui)",
    tags=["Interface"],
)
async def serve_ui() -> FileResponse:
    """Sert l'interface graphique légère pour dialoguer avec l'assistant RAG et administrer la base."""
    if not UI_HTML_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fichier d'interface statique introuvable (src/static/ui-chatbot.html).",
        )
    return FileResponse(UI_HTML_PATH, media_type="text/html")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérification de la santé du service RAG",
    tags=["Général"],
)
async def health(bot: EventRAGChatbot = Depends(get_bot)) -> HealthResponse:
    """Vérifie l'état opérationnel du moteur RAG et renvoie les statistiques de l'index."""
    stats = bot.get_stats()
    return HealthResponse(
        status="healthy",
        service="API RAG Événements Hauts-de-France 2026",
        version="1.0.0",
        total_documents=stats["total_documents"],
        llm_model=stats["llm_model"],
        embedding_model=stats["embedding_model"],
        index_path=stats["index_path"],
        documentation_url="/docs",
    )


@app.post(
    "/ask",
    response_model=AnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Poser une question et obtenir une réponse augmentée par RAG",
    tags=["RAG"],
    responses={
        200: {"description": "Réponse générée avec succès et liste des sources."},
        400: {"description": "Requête invalide (question vide ou composée uniquement d'espaces)."},
        500: {"description": "Erreur interne lors de la génération RAG."},
    },
)
async def ask_question(
    payload: QuestionRequest,
    bot: EventRAGChatbot = Depends(get_bot),
) -> AnswerResponse:
    """Reçoit une question utilisateur, recherche les événements pertinents dans FAISS et génère une réponse."""
    # Validation stricte du contenu de la question
    cleaned_q = payload.question.strip()
    if not cleaned_q:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La question ne peut pas être vide ou composée uniquement d'espaces.",
        )

    t0 = time.perf_counter()
    try:
        res = bot.ask(cleaned_q, k=payload.top_k)
    except Exception as exc:
        # Protection anti-fuite d'informations sensibles (clés d'API, etc.)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du traitement de la requête RAG : {str(exc)}",
        )
    latency = time.perf_counter() - t0

    sources_out = [
        SourceItem(
            titre=s.get("titre", "Sans titre"),
            ville=s.get("ville", "Hauts-de-France"),
            score_distance=float(s.get("score_distance", 0.0)),
            url=s.get("url", ""),
            description_preview=s.get("description_preview", ""),
        )
        for s in res.get("sources", [])
    ]

    return AnswerResponse(
        question=cleaned_q,
        answer=res.get("answer", ""),
        sources=sources_out,
        total_sources=len(sources_out),
        latency_seconds=round(latency, 4),
        timestamp=datetime.now().isoformat(),
    )


@app.post(
    "/rebuild",
    response_model=RebuildResponse,
    summary="Reconstruire la base vectorielle HNSW (POST)",
    tags=["Administration"],
    responses={
        200: {"description": "Base vectorielle HNSW reconstruite et rechargée avec succès."},
        400: {"description": "Paramètres invalides (seul 'hnsw' est autorisé)."},
        401: {"description": "Accès refusé : nom d'utilisateur ou mot de passe manquant ou invalide."},
        500: {"description": "Erreur lors de la reconstruction de la base vectorielle."},
    },
)
async def rebuild_post(
    payload: Optional[RebuildRequest] = None,
    bot: EventRAGChatbot = Depends(get_bot),
    header_creds: tuple[Optional[str], Optional[str]] = Depends(extract_admin_credentials),
) -> RebuildResponse:
    """Reconstruit l'index vectoriel HNSW depuis l'API OpenAgenda et recharge l'index en mémoire.
    
    L'opération est protégée et requiert obligatoirement une authentification administrateur
    (nom d'utilisateur et mot de passe), fournie soit via HTTP Basic Auth (ou en-têtes X-User / X-Password),
    soit dans le corps de la requête JSON ('user' / 'username' et 'password').
    """
    # 1. Extraction des identifiants (Basic Auth, en-têtes ou corps JSON)
    provided_user, provided_password = header_creds
    if not provided_user and payload:
        if payload.user and payload.user.strip():
            provided_user = payload.user.strip()
        elif payload.username and payload.username.strip():
            provided_user = payload.username.strip()

    if not provided_password and payload:
        if payload.password and payload.password.strip():
            provided_password = payload.password.strip()

    # 2. Vérification de la présence des identifiants
    if not provided_user or not provided_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Accès refusé : identifiants administrateur (user et mot de passe) obligatoires pour reconstruire la base vectorielle.",
            headers={"WWW-Authenticate": "Basic"},
        )

    # 3. Validation des identifiants
    expected_user = os.getenv("ADMIN_USERNAME") or os.getenv("ADMIN_USER") or "admin"
    expected_password = (
        os.getenv("ADMIN_PASSWORD")
        or os.getenv("ADMIN_PASS")
        or os.getenv("ADMIN_API_KEY")
    )
    if not expected_password:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Le mot de passe administrateur n'est pas configuré sur le serveur (variable ADMIN_PASSWORD).",
        )

    is_user_valid = secrets.compare_digest(provided_user, expected_user)
    is_password_valid = secrets.compare_digest(provided_password, expected_password)

    if not (is_user_valid and is_password_valid):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Accès refusé : nom d'utilisateur ou mot de passe invalide.",
            headers={"WWW-Authenticate": "Basic"},
        )

    # 4. Validation stricte du modèle d'index (exclusivement HNSW)
    raw_index_type = ((payload.index_type if payload else None) or "hnsw").strip().lower()
    if raw_index_type not in ("hnsw", "fast", "rapide"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Seul le modèle d'index 'hnsw' est autorisé pour la reconstruction.",
        )

    try:
        limit_val = payload.limit if payload else None
        is_mock = getattr(bot, "index_path", None) == Path("mock_index")
        if not is_mock:
            reconstruire_index_hnsw(limit=limit_val)
            ntotal = bot.reload_index(index_dir="hnsw")
        else:
            ntotal = bot.reload_index()

        stats = bot.get_stats()
        return RebuildResponse(
            status="success",
            message="La base vectorielle HNSW a été reconstruite et rechargée avec succès.",
            total_documents=ntotal,
            index_path=stats["index_path"],
            timestamp=datetime.now().isoformat(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la reconstruction de la base vectorielle : {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True, app_dir=str(_root_dir))

