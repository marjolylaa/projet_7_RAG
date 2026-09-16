"""Script d'inspection 100 % HORS-LIGNE et GRATUIT de la base vectorielle FAISS.

AUCUN appel réseau, AUCUNE clé API requise, 0 CRÉDIT consommé.
Lit directement les fichiers binaires locaux (index.faiss et index.pkl).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

# Configuration de l'encodage console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import faiss


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    if current.parent.name == "src":
        return current.parent.parent
    return current.parent


def inspect_index(index_dir: Path, search_keyword: str | None = None, sample_count: int = 5) -> None:
    faiss_file = index_dir / "index.faiss"
    pkl_file = index_dir / "index.pkl"

    if not faiss_file.exists() or not pkl_file.exists():
        print(f"❌ Dossier d'index introuvable ou incomplet : {index_dir}")
        return

    print("=" * 75)
    print("🔍 INSPECTION LOCALE DE L'INDEX FAISS (100 % HORS-LIGNE - 0 CRÉDIT API)")
    print("=" * 75)
    print(f"📁 Chemin : {index_dir}\n")

    # 1. Lecture de l'index FAISS (C++ natif, 0 réseau)
    index = faiss.read_index(str(faiss_file))
    print(f"📊 [FAISS] Nombre de vecteurs indexés : {index.ntotal:,}")
    print(f"📐 [FAISS] Dimension des vecteurs      : {index.d}")
    print(f"🏷️  [FAISS] Type d'index FAISS         : {type(index).__name__}")
    print(f"💾 [Disque] Taille index.faiss        : {faiss_file.stat().st_size / (1024*1024):.2f} Mo")

    # 2. Lecture du Docstore (Pickle local, 0 réseau)
    with open(pkl_file, "rb") as f:
        docstore, id_to_doc = pickle.load(f)

    # Récupération de tous les documents en mémoire
    docs = list(docstore._dict.values())
    print(f"📚 [Docstore] Documents enregistrés    : {len(docs):,}")
    print(f"💾 [Disque] Taille index.pkl          : {pkl_file.stat().st_size / (1024*1024):.2f} Mo")

    # 3. Statistiques géographiques rapides
    villes = [d.metadata.get("Ville") for d in docs if d.metadata.get("Ville")]
    top_villes = Counter(villes).most_common(10)
    print("\n🏙️ Top 10 des villes les plus représentées dans la base :")
    for ville, count in top_villes:
        print(f"   • {ville:<25} : {count:,} événements")

    # 4. Échantillon aléatoire d'événements
    import random
    print(f"\n🎲 Échantillon aléatoire de {sample_count} événements dans la base :")
    sample_docs = random.sample(docs, min(sample_count, len(docs)))
    for i, d in enumerate(sample_docs, 1):
        titre = d.metadata.get("Titre", "Sans titre")
        ville = d.metadata.get("Ville", "Ville non précisée")
        dates = (
            d.metadata.get("Première date - Début")
            or d.metadata.get("Premire date - Dbut")
            or "Date non précisée"
        )
        print(f"   [{i}] {titre}")
        print(f"       Ville : {ville} | Date début : {dates}")

    # 5. Recherche textuelle locale directe (sans embedding, filtrage purement textuel)
    if search_keyword:
        kw = search_keyword.lower()
        matches = [
            d for d in docs
            if kw in (d.metadata.get("Titre") or "").lower()
            or kw in d.page_content.lower()
        ]
        print(f"\n🔎 Recherche textuelle locale pour le mot-clé « {search_keyword} » :")
        print(f"   Total trouvé : {len(matches):,} événement(s)")
        for i, m in enumerate(matches[:5], 1):
            t = m.metadata.get("Titre", "Sans titre")
            v = m.metadata.get("Ville", "")
            print(f"   [{i}] {t} ({v})")
        if len(matches) > 5:
            print(f"   ... et {len(matches) - 5} autres résultats.")

    print("\n" + "=" * 75)
    print("✅ Inspection terminée sans aucun appel externe ni consommation de jetons.")
    print("=" * 75)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspecte l'index FAISS local sans aucun appel API ni coût."
    )
    parser.add_argument(
        "--keyword",
        "-k",
        type=str,
        default=None,
        help="Recherche un mot-clé textuel dans les métadonnées (100% hors-ligne).",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Inspecter l'index Flat L2 au lieu de l'index HNSW.",
    )
    args = parser.parse_args()

    root = get_project_root()
    sub_dir = "mon_index_langchain_evenements" if args.flat else "mon_index_langchain_evenements_hnsw_rapide"
    target = root / "src" / sub_dir
    if not target.exists():
        target = root / sub_dir

    inspect_index(target, search_keyword=args.keyword)


if __name__ == "__main__":
    main()
