"""Script de création de la base vectorielle FAISS HNSW.

Reprend les étapes du notebook `src/recuperation_data_et_vectorisation.ipynb` :
1. Récupération des événements 2026 en Hauts-de-France depuis l'API OpenDataSoft (OpenAgenda).
2. Extraction des descriptions et calcul des embeddings par lots avec Mistral AI (`mistral-embed`).
3. Création de l'index optimisé FAISS HNSW (`IndexHNSWFlat`, M=32).
4. Sauvegarde dans `src/mon_index_langchain_evenements_hnsw_rapide`.

Options CLI :
    --limit N   : Limite le nombre d'événements à récupérer pour créer un index réduit (ex: --limit 50).
    --dry       : Mode simulation (récupère les données sans appeler l'API d'embeddings ni écrire sur disque).
"""

import argparse
import os
import sys
import time
import uuid
import warnings
from pathlib import Path
from typing import Optional, Union

import faiss
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from mistralai.client import Mistral
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

# Configuration de l'encodage console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def reconstruire_index_hnsw(
    limit: Optional[int] = None,
    dry: bool = False,
    output_path: Optional[Union[Path, str]] = None,
    api_key: Optional[str] = None,
) -> int:
    """Récupère les événements 2026 depuis l'API OpenAgenda, calcule les embeddings Mistral et sauvegarde l'index FAISS HNSW.

    Args:
        limit: Nombre maximum d'événements à récupérer pour créer un index réduit (ex: 50).
        dry: Mode simulation (récupère les données sans appeler l'API d'embeddings ni écrire sur disque).
        output_path: Chemin du répertoire de sauvegarde de l'index (par défaut src/mon_index_langchain_evenements_hnsw_rapide).
        api_key: Clé API Mistral (si non fournie, lue depuis la variable d'environnement MISTRAL_API_KEY).

    Returns:
        int: Nombre de documents indexés (ou 0 si mode dry ou aucun document).
    """
    if not api_key:
        env_file = Path(__file__).resolve().parent / ".env"
        if env_file.exists():
            load_dotenv(dotenv_path=env_file)
        else:
            load_dotenv()
        api_key = os.environ.get("MISTRAL_API_KEY")

    if not api_key and not dry:
        raise ValueError("La variable MISTRAL_API_KEY est manquante dans l'environnement.")

    model = "mistral-embed"

    # ============================================================================
    # 1. Récupération des données via l'API OpenAgenda
    # ============================================================================

    headers = {"Content-Type": "application/json"}
    base_url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records/"
    where = "YEAR(firstdate_begin) = 2026 AND YEAR(lastdate_begin) = 2026 AND location_region = 'Hauts-de-France'"

    print("Récupération des données depuis l'API OpenAgenda...")
    liste_resultats = []
    offset = 0

    while True:
        current_limit = 100
        if limit:
            remaining = limit - len(liste_resultats)
            if remaining <= 0:
                break
            current_limit = min(100, remaining)

        params = {
            "lang": "fr",
            "limit": current_limit,
            "offset": offset,
            "where": where,
        }
        response = requests.get(base_url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        donnees = response.json()

        batch = donnees.get("results", [])
        if not batch:
            break
        liste_resultats.extend(batch)
        offset += len(batch)

        total_count = donnees.get("total_count", 0)
        target = min(total_count, limit) if limit else total_count
        print(f"Téléchargement : {len(liste_resultats)} / {target} événements", end="\r")
        if offset >= total_count or (limit and len(liste_resultats) >= limit):
            break

    print()
    evenement_pd = pd.DataFrame(liste_resultats)
    print(f"taille data : {len(evenement_pd)}")

    # ============================================================================
    # 2. Vectorisation des descriptions
    # ============================================================================

    col_desc = "description_fr" if "description_fr" in evenement_pd.columns else "Description"
    if col_desc in evenement_pd.columns:
        evenement_pd = evenement_pd.dropna(subset=[col_desc])
        descriptions_to_embed = evenement_pd[col_desc].to_list()
    else:
        descriptions_to_embed = []
    print(f"Descriptions valides à vectoriser : {len(descriptions_to_embed)}")

    if dry:
        print("🔍 Mode DRY activé : simulation terminée (aucun appel d'embeddings ni écriture sur disque).")
        return 0

    if not descriptions_to_embed:
        print("⚠️ Aucune description valide à vectoriser.")
        return 0

    client = Mistral(api_key=api_key)
    taille_du_lot = 100
    tous_les_embeddings = []

    # Exécution par paquets à cause des limitations de l'api
    for i in tqdm(range(0, len(descriptions_to_embed), taille_du_lot), desc="Génération des vecteurs"):
        lot = descriptions_to_embed[i : i + taille_du_lot]

        reponse = client.embeddings.create(
            model=model,
            inputs=lot,
        )

        for element in reponse.data:
            tous_les_embeddings.append(element.embedding)

        # Pause de sécurité pour ne pas dépasser les limites de l'API
        time.sleep(1)

    # Insertion dans FAISS
    vecteurs_numpy = np.array(tous_les_embeddings).astype("float32")
    dimension = 1024  # Le modèle 'mistral-embed' génère des vecteurs de 1024 dimensions
    liens_par_noeud = 32  # Paramètre HNSW

    df_metadata = evenement_pd.drop(columns=[col_desc])
    metadatas = df_metadata.to_dict(orient="records")

    textes_et_vecteurs = list(zip(descriptions_to_embed, vecteurs_numpy))
    embeddings = MistralAIEmbeddings(model=model, mistral_api_key=api_key)

    # ============================================================================
    # 3. Création et sauvegarde de l'index HNSW
    # ============================================================================

    index_optimise = faiss.IndexHNSWFlat(dimension, liens_par_noeud)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index_optimise,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # LangChain demande un identifiant unique (ID) pour chaque document ajouté manuellement
    ids = [str(uuid.uuid4()) for _ in range(len(descriptions_to_embed))]

    vector_store.add_embeddings(
        text_embeddings=textes_et_vecteurs,
        metadatas=metadatas,
        ids=ids,
    )

    # Sauvegarde de la nouvelle base super-rapide
    if output_path is None:
        target_path = Path(__file__).resolve().parent / "mon_index_langchain_evenements_hnsw_rapide"
    else:
        target_path = Path(output_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(target_path))
    print("✅ Index HNSW optimisé et sauvegardé !")
    return len(descriptions_to_embed)


def main(argv=None):
    # Arguments CLI
    parser = argparse.ArgumentParser(description="Création de la base vectorielle FAISS HNSW.")
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Nombre maximum d'événements à récupérer pour un index réduit (ex: --limit 50).",
    )
    parser.add_argument(
        "--dry", "--dry-run",
        action="store_true",
        dest="dry",
        help="Mode simulation : teste la récupération des données sans appeler l'API d'embeddings.",
    )
    args = parser.parse_args(argv)

    reconstruire_index_hnsw(limit=args.limit, dry=args.dry)
    return 0


if __name__ == "__main__":
    sys.exit(main())

