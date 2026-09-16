"""Interface en ligne de commande (CLI) pour tester le chatbot RAG dans le terminal.

Ce module isole la boucle interactive textuelle (input/print) afin de ne pas
encombrer le service métier (src/chatbot.py) ni le point d'entrée principal (main.py).
"""

from __future__ import annotations

import argparse
import sys

# Configuration de l'encodage de la console sur Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.chatbot import EventRAGChatbot


def run_cli(top_k: int = 4, model: str = "open-mistral-nemo") -> None:
    """Lance le chatbot en mode conversationnel dans le terminal."""
    print("=" * 70)
    print("🤖 ASSISTANT RAG - HAUTS-DE-FRANCE 2026 (Console de Test)")
    print("=" * 70)
    print("Posez vos questions sur les sorties, ateliers, concerts et festivals.")
    print("Tapez 'quit' ou 'exit' pour quitter.\n")

    try:
        bot = EventRAGChatbot(top_k=top_k, llm_model=model)
        print("✅ Base vectorielle FAISS prête et modèle connecté !\n")
    except Exception as e:
        print(f"❌ Impossible d'initialiser le chatbot : {e}")
        sys.exit(1)

    while True:
        try:
            query = input("👤 Vous : ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("Au revoir !")
                break

            print("\n⏳ Recherche et génération en cours...")
            res = bot.ask(query)
            print("\n" + "=" * 70)
            print(res["answer"])
            print("=" * 70)
            print(
                f"📚 {res['total_sources']} événement(s) pertinent(s) sélectionné(s) (parmi 24 705 dans la base).\n"
            )

        except KeyboardInterrupt:
            print("\nFermeture du terminal.")
            break
        except Exception as e:
            print(f"\n❌ Erreur : {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interface console pour tester le Chatbot RAG.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Nombre d'événements similaires à récupérer dans la base FAISS (défaut: 4).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="open-mistral-nemo",
        help="Modèle LLM Mistral à utiliser (défaut: open-mistral-nemo).",
    )
    args = parser.parse_args()
    run_cli(top_k=args.top_k, model=args.model)
