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
import sys
from pathlib import Path

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
    print("     -> Prêt à être connecté à la future interface graphique (Streamlit, etc.)")
    print("   • Interface Console    : `src/cli.py` (mode terminal isolé)")
    print("   • Suite d'évaluation   : `src/evaluer_scenarios.py` (5 scénarios de test)")
    print("   • Outil d'audit local  : `src/inspect_index.py` (100% hors-ligne, 0 crédit)")
    print("-" * 75)
    print("🚀 COMMANDES UTILES :")
    print("   • Inspecter la base    : uv run python main.py --inspect [--keyword <mot>]")
    print("   • Évaluer les scénarios: uv run python main.py --eval")
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
        help="Inspecte la base FAISS en local (100% hors-ligne, 0 appel API, 0 crédit).",
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
        help="Exécute la suite de tests et de démonstration sur 5 scénarios d'interaction (génère un rapport HTML).",
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
        "--top-k",
        type=int,
        default=4,
        help="Nombre d'événements à récupérer (utile avec --cli).",
    )
    args = parser.parse_args()

    if args.inspect:
        root = get_project_root()
        target = root / "src" / "mon_index_langchain_evenements_hnsw_rapide"
        inspect_index(target, search_keyword=args.keyword)
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


