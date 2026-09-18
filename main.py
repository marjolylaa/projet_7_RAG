"""Point d'entrée principal du projet RAG - Événements Hauts-de-France 2026.

Le chatbot (service métier) est désormais découplé dans `src.chatbot.EventRAGChatbot`
afin de pouvoir être branché directement sur une interface graphique (GUI : Streamlit, Chainlit, etc.).

Ce fichier sert de point d'orchestration pour :
- Vérifier l'état du système et de la base vectorielle (par défaut)
- Inspecter la base en local hors-ligne (`--inspect`)
- Exécuter la suite d'évaluation des scénarios (`--eval`)
- Lancer le mode console de test isolé (`--cli`)
"""

import argparse
import os
import sys
from pathlib import Path

# Initialisation des chemins pour le projet et les sous-processus (uvicorn reload, etc.)
_root_dir = Path(__file__).resolve().parent
_src_dir = _root_dir / "src"
for _p in (str(_root_dir), str(_src_dir)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_pp = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{_root_dir}{os.pathsep}{_src_dir}{os.pathsep}{_pp}" if _pp else f"{_root_dir}{os.pathsep}{_src_dir}"

# Configuration de l'encodage de la console sur Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.evaluer_scenarios import run_evaluation
from src.inspect_index import get_project_root, inspect_index


def status_check() -> None:
    """Affiche un rapport d'état du projet et les commandes disponibles."""
    root = get_project_root()
    index_path = root / "src" / "mon_index_langchain_evenements_hnsw_rapide"
    env_file = root / "src" / ".env"
    if not env_file.exists():
        env_file = root / ".env"

    print("=" * 75)
    print("🏛️  SYSTÈME RAG - ÉVÉNEMENTS HAUTS-DE-FRANCE 2026")
    print("=" * 75)
    print(f"📁 Racine du projet      : {root}")
    print(f"🔑 Fichier .env          : {'✅ Trouvé' if env_file.exists() else '❌ Manquant'}")
    print(f"🗄️  Index FAISS HNSW      : {'✅ Disponible' if index_path.exists() else '❌ Non trouvé'}")
    print("-" * 75)
    print("✨ ARCHITECTURE MODULAIRE :")
    print("   • Moteur RAG métier    : `src/chatbot.py` (classe `EventRAGChatbot`)")
    print("     -> Prêt à être connecté à l'API REST ou une interface graphique (Streamlit, etc.)")
    print("   • API REST FastAPI     : `src/api.py` (endpoints /ask, /rebuild, Swagger /docs)")
    print("   • Interface Web        : `src/static/index.html` (accessible à la racine http://127.0.0.1:8000)")
    print("   • Interface Console    : `src/cli.py` (mode terminal isolé)")
    print("   • Suite d'évaluation   : `src/evaluer_scenarios.py` (5 scénarios avec rapport HTML et Ragas)")
    print("   • Évaluation Ragas     : `src/evaluate_rag.py` (Faithfulness, Answer Correctness)")
    print("   • Outil d'audit local  : `src/inspect_index.py` (100% hors-ligne, 0 crédit)")
    print("-" * 75)
    print("🚀 COMMANDES UTILES :")
    print("   • Lancer l'Interface UI: uv run python main.py --ui")
    print("   • Lancer l'API REST    : uv run python main.py --serve (ou uv run uvicorn src.api:app --reload)")
    print("   • Tester l'API         : uv run python api_test.py")
    print("   • Inspecter la base    : uv run python main.py --inspect [--keyword <mot>]")
    print("   • Évaluer les scénarios: uv run python main.py --eval")
    print("   • Évaluation Ragas     : uv run python main.py --ragas")
    print("   • Lancer les tests     : uv run pytest")
    print("   • Tester en console    : uv run python src/cli.py")
    print("=" * 75)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Point d'orchestration pour le système RAG (Événements Hauts-de-France 2026)."
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspecte la base FAISS en local (entièrement hors-ligne, 0 appel API, 0 crédit).",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default=None,
        help="Mot-clé pour filtrer les événements en local avec --inspect (ex: --keyword jazz).",
    )
    parser.add_argument(
        "--eval",
        "--demo",
        action="store_true",
        dest="run_eval",
        help="Exécute la suite d'évaluation systématique sur 5 scénarios d'interaction avec le framework Ragas et rapport HTML.",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Exécute l'évaluation Ragas automatisée standalone (exporte rapport/ragas_evaluation_results.json).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Chemin du fichier HTML de sortie pour l'évaluation (par défaut: rapport/rapport_evaluation.html).",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Lancer la console de test interactive dans le terminal.",
    )
    parser.add_argument(
        "--serve",
        "--api",
        action="store_true",
        dest="serve_api",
        help="Lancer l'API REST locale FastAPI avec Uvicorn.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Lancer l'API et ouvrir directement l'interface Web graphique dans le navigateur.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Hôte d'écoute pour l'API REST (par défaut: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port d'écoute pour l'API REST (par défaut: 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Activer le rechargement automatique à chaud pour l'API (mode dev).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Nombre d'événements à récupérer (utile avec --cli).",
    )
    args = parser.parse_args()

    if args.serve_api or args.ui:
        import os
        import threading
        import webbrowser
        import uvicorn

        root_dir = str(get_project_root())
        src_dir = str(get_project_root() / "src")
        for p in (root_dir, src_dir):
            if p not in sys.path:
                sys.path.insert(0, p)
        # Propagation aux processus enfants (reloader Windows multiprocessing / spawn)
        curr_pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f"{root_dir}{os.pathsep}{src_dir}{os.pathsep}{curr_pp}" if curr_pp else f"{root_dir}{os.pathsep}{src_dir}"

        ui_url = f"http://{args.host}:{args.port}/"
        print("=" * 80)
        print(f"🚀 Démarrage du serveur API REST RAG sur http://{args.host}:{args.port}")
        print(f"🎭 Interface Graphique Web           : {ui_url} (ou /ui)")
        print(f"📖 Documentation interactive Swagger : http://{args.host}:{args.port}/docs")
        print(f"📖 Documentation ReDoc               : http://{args.host}:{args.port}/redoc")
        print("=" * 80)

        if args.ui:
            def open_browser():
                import time
                time.sleep(1.2)
                webbrowser.open(ui_url)

            threading.Thread(target=open_browser, daemon=True).start()

        uvicorn.run(
            "src.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            app_dir=root_dir,
        )
        return

    if args.inspect:
        root = get_project_root()
        target = root / "src" / "mon_index_langchain_evenements_hnsw_rapide"
        inspect_index(target, search_keyword=args.keyword)
        return

    if args.ragas:
        from src.evaluate_rag import main as run_ragas_main
        run_ragas_main()
        return

    if args.run_eval:
        run_evaluation(output_html=args.output)
        return

    if args.cli:
        from src.cli import run_cli
        run_cli(top_k=args.top_k)
        return

    # Par défaut : statut du système non bloquant
    status_check()


if __name__ == "__main__":
    main()


