"""Module Chatbot RAG pour la recommandation d'événements culturels en Hauts-de-France (2026).

Orchestration avec LangChain, base vectorielle FAISS et LLM Mistral AI.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configuration de l'encodage de la console sur Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings


def get_project_root() -> Path:
    """Retourne la racine du projet."""
    current = Path(__file__).resolve()
    # Si le fichier est dans src/, la racine est un niveau au-dessus
    if current.parent.name == "src":
        return current.parent.parent
    return current.parent


def setup_environment(env_file: Optional[Path] = None) -> None:
    """Charge les variables d'environnement (.env)."""
    root = get_project_root()
    candidates = [
        env_file,
        root / "src" / ".env",
        root / ".env",
        Path.cwd() / "src" / ".env",
        Path.cwd() / ".env",
    ]
    for path in candidates:
        if path and Path(path).exists():
            load_dotenv(path)
            break

    if not os.getenv("MISTRAL_API_KEY"):
        raise ValueError(
            "La variable MISTRAL_API_KEY est introuvable. "
            "Veuillez définir votre clé dans un fichier .env (ex: src/.env)."
        )


def clean_text(value: Any) -> str:
    """Nettoie une chaîne de texte (valeurs nan, caractères invalides)."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    # Remplacement des caractères de remplacement utf-8 résiduels
    text = text.replace("\ufffd", "e")
    # Suppression des balises HTML basiques éventuelles
    text = re.sub(r"<[^>]+>", " ", text)
    # Réduction des espaces multiples
    text = re.sub(r"\s+", " ", text).strip()
    return text


MONTHS_MAP: Dict[str, str] = {
    "janvier": "01",
    "février": "02",
    "fevrier": "02",
    "mars": "03",
    "avril": "04",
    "mai": "05",
    "juin": "06",
    "juillet": "07",
    "août": "08",
    "aout": "08",
    "septembre": "09",
    "octobre": "10",
    "novembre": "11",
    "décembre": "12",
    "decembre": "12",
}


def get_temporal_filter(query: str) -> Optional[Tuple[Any, str]]:
    """Détecte la mention d'un mois dans la requête et génère un filtre sur les métadonnées."""
    q_lower = query.lower()
    for month_name, month_code in MONTHS_MAP.items():
        pattern = r"\b" + re.escape(month_name) + r"\b"
        if re.search(pattern, q_lower):
            def month_filter(meta: dict) -> bool:
                combined = (
                    str(meta.get("Première date - Début", ""))
                    + str(meta.get("Premire date - Dbut", ""))
                    + str(meta.get("Dernière date - Fin", ""))
                    + str(meta.get("Dernire date - Fin", ""))
                    + str(meta.get("Résumé horaires", ""))
                    + str(meta.get("Rsum horaires", ""))
                ).lower()
                return f"2026-{month_code}" in combined or month_name in combined

            return month_filter, month_name
    return None


def format_date_with_year(horaires: str, date_debut: str, date_fin: str) -> str:
    """Garantit que l'année (ex: 2026) est explicitement mentionnée dans la date."""
    year = "2026"
    for d_str in (date_debut, date_fin):
        if d_str and len(d_str) >= 4 and d_str[:4].isdigit():
            year = d_str[:4]
            break

    if horaires:
        if year not in horaires:
            if "," in horaires:
                parts = horaires.split(",", 1)
                return f"{parts[0].strip()} {year},{parts[1]}"
            else:
                return f"{horaires} {year}"
        return horaires
    elif date_debut or date_fin:
        d1 = date_debut[:10] if date_debut else ""
        d2 = date_fin[:10] if date_fin else ""
        if d1 and d2 and d1 != d2:
            return f"du {d1} au {d2}"
        return f"le {d1 or d2}"
    return f"Année {year}"


def compute_temporal_status(date_debut_iso: str, date_fin_iso: str) -> Tuple[str, str]:
    """Détermine si l'événement est passé, en cours ou à venir par rapport à aujourd'hui."""
    now = datetime.now().astimezone()
    dt_start = None
    dt_end = None

    if date_debut_iso:
        try:
            dt_start = datetime.fromisoformat(date_debut_iso)
            if dt_start.tzinfo is None:
                dt_start = dt_start.replace(tzinfo=now.tzinfo)
        except Exception:
            pass

    if date_fin_iso:
        try:
            dt_end = datetime.fromisoformat(date_fin_iso)
            if dt_end.tzinfo is None:
                dt_end = dt_end.replace(tzinfo=now.tzinfo)
        except Exception:
            pass

    target_dt = dt_end or dt_start
    if target_dt:
        if target_dt < now:
            return (
                "PASSÉ",
                f"Événement terminé (date : {target_dt.strftime('%d/%m/%Y')}) -> Employer IMPÉRATIVEMENT des temps du passé (ex: 's'est tenu le', 'a eu lieu en', 'a proposé').",
            )
        elif dt_start and dt_start <= now and dt_end and dt_end >= now:
            return (
                "EN COURS",
                f"Événement en cours jusqu'au {dt_end.strftime('%d/%m/%Y')} -> Employer le présent (ex: 'se tient actuellement').",
            )
        else:
            return (
                "À VENIR",
                f"Événement futur (prévu le {target_dt.strftime('%d/%m/%Y')}) -> Employer le futur ou le présent d'anticipation (ex: 'aura lieu le', 'est prévu en').",
            )

    return "NON DÉTERMINÉ", "Adapter la concordance des temps selon la date indiquée."


def format_document_context(docs: List[Document]) -> str:
    """Formate une liste de documents LangChain en texte structuré pour le prompt."""
    if not docs:
        return "Aucun événement pertinent trouvé dans la base."

    formatted_entries: List[str] = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        titre = clean_text(meta.get("Titre") or meta.get("title_fr") or "Événement sans titre")
        ville = clean_text(meta.get("Ville") or meta.get("location_city") or "Hauts-de-France")
        nom_lieu = clean_text(meta.get("Nom du lieu") or meta.get("location_name") or "")
        adresse = clean_text(meta.get("Adresse") or meta.get("location_address") or "")
        conditions = clean_text(
            meta.get("Détail des conditions")
            or meta.get("Dtail des conditions")
            or meta.get("conditions_fr")
            or ""
        )
        date_debut = clean_text(
            meta.get("Première date - Début")
            or meta.get("Premire date - Dbut")
            or meta.get("firstdate_begin")
            or ""
        )
        date_fin = clean_text(
            meta.get("Dernière date - Fin")
            or meta.get("Dernire date - Fin")
            or meta.get("lastdate_end")
            or ""
        )
        horaires = clean_text(
            meta.get("Résumé horaires")
            or meta.get("Rsum horaires")
            or meta.get("daterange_fr")
            or ""
        )
        url = clean_text(meta.get("URL canonique") or meta.get("canonicalurl") or "")
        age_min = meta.get("Age minimum") or meta.get("age_min")
        age_max = meta.get("Age maximum") or meta.get("age_max")
        description = clean_text(doc.page_content)

        lines = [f"[ÉVÉNEMENT {i}]"]
        lines.append(f"- Titre : {titre}")
        if nom_lieu and ville:
            lines.append(f"- Lieu : {nom_lieu}, {ville}")
        elif ville:
            lines.append(f"- Ville : {ville}")
        if adresse:
            lines.append(f"- Adresse : {adresse}")

        # Date avec mention explicite de l'année
        date_complete = format_date_with_year(horaires, date_debut, date_fin)
        lines.append(f"- Date (avec année) : {date_complete}")

        # Statut temporel et consigne de rédaction grammaticale
        statut_temporel, consigne_temps = compute_temporal_status(date_debut, date_fin)
        lines.append(f"- Statut temporel : {statut_temporel}")
        lines.append(f"- Règle de conjugaison : {consigne_temps}")

        if conditions:
            lines.append(f"- Conditions d'accès & Tarifs : {conditions}")

        if age_min is not None and str(age_min) != "nan":
            age_str = f"À partir de {age_min} ans"
            if age_max is not None and str(age_max) != "nan" and int(age_max) < 99:
                age_str += f" jusqu'à {age_max} ans"
            lines.append(f"- Public conseillé : {age_str}")

        if url:
            lines.append(f"- Lien officiel : {url}")

        if description:
            lines.append(f"- Description : {description}")

        formatted_entries.append("\n".join(lines))

    return "\n\n".join(formatted_entries)


SYSTEM_PROMPT = """Tu es un conseiller culturel et guide intelligent expert des événements publics et culturels de la région Hauts-de-France pour l'année 2026.
Date actuelle de référence : {current_date}

Ton objectif est de fournir des recommandations personnalisées, chaleureuses et pertinentes en te basant EXCLUSIVEMENT sur les événements fournis ci-dessous dans la section CONTEXTE.

Consignes impératives :
1. Mention OBLIGATOIRE de l'année :
   - Tu dois SYSTÉMATIQUEMENT mentionner l'année pour chaque événement cité (ex : "en mai 2026", "le 24 novembre 2026", "au printemps 2026"). Ne cite JAMAIS un jour et un mois sans expliciter l'année.
2. Concordance des temps (Passé vs Futur) :
   - Pour les événements passés (dont la date est antérieure à la date actuelle {current_date}, ou indiqués comme PASSÉ dans le contexte) : Tu dois OBLIGATOIREMENT employer des temps du passé (passé composé ou imparfait).
     Exemples : "Cet événement s'est déroulé le 7 juin 2026", "Il a eu lieu à Senlis", "Les participants ont pu découvrir...".
     Ne présente JAMAIS un événement passé au futur ou au présent d'anticipation !
   - Pour les événements à venir (marqués comme À VENIR) : Utilise le futur ou le présent d'anticipation (ex : "Cet événement aura lieu le 24 novembre 2026", "Il se déroulera à Lille").
   - Pour les événements en cours (marqués comme EN COURS) : Utilise le présent (ex : "Cet événement se tient actuellement jusqu'au...").
3. Pertinence et personnalisation :
   - Sélectionne parmi le contexte les événements les plus appropriés à la demande de l'utilisateur.
   - Explique clairement et de façon motivante POURQUOI chaque événement répond à son besoin (thématique, ambiance, public cible, accessibilité).
4. Précision factuelle :
   - Pour chaque événement suggéré, indique systématiquement : le titre exact, la ville / le lieu, la date complète AVEC l'année, ainsi que les conditions d'accès (gratuit, réservation, tarif) et le lien officiel s'ils sont disponibles.
   - N'invente JAMAIS d'événement, de date, de lieu ou de tarif qui ne figure pas dans le contexte.
5. Règle de refus et de périmètre :
   - Si la demande est totalement hors de propos (ex: autre sujet que des sorties/événements, question théorique hors domaine, ou demande pour une autre région comme Marseille ou l'étranger), décline poliment en rappelant ton périmètre (les événements 2026 dans les Hauts-de-France).
   - Si le contexte ne contient aucun événement pertinent pour satisfaire la requête, indique-le avec courtoisie en précisant qu'aucun événement correspondant n'a été trouvé dans l'agenda actuel des Hauts-de-France pour 2026, et invite l'utilisateur à reformuler sa recherche.
6. Style et mise en page :
   - Rédige en français soigné, engageant, bienveillant et aéré (listes à puces, mise en gras des titres et lieux).

CONTEXTE DES ÉVÉNEMENTS DISPONIBLES :
{context}
"""


class EventRAGChatbot:
    """Chatbot RAG pour la recommandation d'événements avec FAISS et Mistral AI."""

    def __init__(
        self,
        index_dir: Optional[str | Path] = None,
        embedding_model: str = "mistral-embed",
        llm_model: str = "open-mistral-nemo",
        temperature: float = 0.2,
        top_k: int = 4,
        vectorstore: Optional[FAISS] = None,
        llm: Optional[Any] = None,
        embeddings: Optional[Any] = None,
    ) -> None:
        setup_environment()
        self.root = get_project_root()
        self.top_k = top_k
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.temperature = temperature

        # 1. Initialisation des embeddings
        if embeddings is not None:
            self.embeddings = embeddings
        elif vectorstore is not None and hasattr(vectorstore, "embeddings") and vectorstore.embeddings is not None:
            self.embeddings = vectorstore.embeddings
        else:
            self.embeddings = MistralAIEmbeddings(model=embedding_model)

        # 2. Chargement ou injection de l'index vectoriel FAISS
        if vectorstore is not None:
            self.vectorstore = vectorstore
            self.index_path = Path("mock_index")
        else:
            self.index_path = self._resolve_index_path(index_dir)
            self.vectorstore = FAISS.load_local(
                folder_path=str(self.index_path),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )

        # 3. Initialisation du modèle Mistral
        if llm is not None:
            self.llm = llm
        else:
            self.llm = self._init_llm(llm_model, temperature)

        # 4. Construction de la chaîne LangChain (LCEL)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        self.chain = (
            {
                "context": lambda x: format_document_context(x["docs"]),
                "question": lambda x: x["question"],
                "current_date": lambda x: datetime.now().strftime("%d/%m/%Y"),
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def _resolve_index_path(self, path: Optional[str | Path]) -> Path:
        """Détermine le chemin absolu de l'index vectoriel."""
        if path:
            candidate = Path(path)
            if candidate.is_absolute() and candidate.exists():
                return candidate
            if (self.root / candidate).exists():
                return self.root / candidate

        # Recherche par défaut de l'index HNSW, puis fallback Flat L2
        hnsw_candidates = [
            self.root / "src" / "mon_index_langchain_evenements_hnsw_rapide",
            self.root / "mon_index_langchain_evenements_hnsw_rapide",
        ]
        for p in hnsw_candidates:
            if p.exists() and (p / "index.faiss").exists():
                return p

        flat_candidates = [
            self.root / "src" / "mon_index_langchain_evenements",
            self.root / "mon_index_langchain_evenements",
        ]
        for p in flat_candidates:
            if p.exists() and (p / "index.faiss").exists():
                return p

        raise FileNotFoundError(
            f"Aucun index FAISS trouvé dans les emplacements attendus ({self.root}/src/...)."
        )

    def _init_llm(self, model_name: str, temperature: float) -> ChatMistralAI:
        """Initialise le LLM Mistral avec gestion d'erreurs éventuelles."""
        try:
            return ChatMistralAI(model=model_name, temperature=temperature)
        except Exception:
            # Fallback vers open-mistral-7b si le modèle demandé n'est pas accessible
            return ChatMistralAI(model="open-mistral-7b", temperature=temperature)

    def retrieve_documents(
        self,
        query: str,
        k: Optional[int] = None,
        deduplicate: bool = True,
        filter_func: Optional[Any] = None,
    ) -> List[Tuple[Document, float]]:
        """Effectue une recherche par similarité avec calcul de distance L2.
        
        Si un critère temporel (ex: mois de novembre) est détecté dans la requête,
        applique automatiquement un filtrage sur les métadonnées de dates avec un
        espace de recherche élargi (fetch_k) pour garantir des résultats exacts.
        
        Si deduplicate=True, élimine les doublons de titre d'événements récurrents.
        """
        limit = k or self.top_k

        # Détection automatique de critère temporel dans la requête
        if filter_func is None:
            temp_filter = get_temporal_filter(query)
            if temp_filter:
                filter_func, _month = temp_filter

        fetch_k = 600 if filter_func else (limit * 3 if deduplicate else limit)
        try:
            raw_results = self.vectorstore.similarity_search_with_score(
                query, k=limit * 3 if deduplicate else limit, filter=filter_func, fetch_k=fetch_k
            )
        except Exception:
            # Fallback direct en cas de problème de filtre
            raw_results = self.vectorstore.similarity_search_with_score(query, k=limit * 3)

        if not deduplicate:
            return raw_results[:limit]

        seen_titles = set()
        unique_results = []
        for doc, score in raw_results:
            title = clean_text(
                doc.metadata.get("Titre") or doc.metadata.get("title_fr") or doc.page_content[:60]
            ).lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_results.append((doc, score))
                if len(unique_results) == limit:
                    break

        return unique_results if unique_results else raw_results[:limit]

    def ask(self, question: str, k: Optional[int] = None) -> Dict[str, Any]:
        """Traite une question utilisateur et génère une réponse augmentée par RAG.

        Retourne un dictionnaire contenant la question, la réponse du LLM,
        le contexte textuel formaté, ainsi que les documents sources avec leurs scores.
        """
        limit = k or self.top_k
        docs_with_scores = self.retrieve_documents(question, k=limit)
        docs = [doc for doc, _score in docs_with_scores]

        answer = self.chain.invoke({
            "question": question,
            "docs": docs,
        })

        sources = []
        for doc, score in docs_with_scores:
            sources.append({
                "titre": doc.metadata.get("Titre") or "Sans titre",
                "ville": doc.metadata.get("Ville") or "Hauts-de-France",
                "score_distance": float(score),
                "url": doc.metadata.get("URL canonique") or "",
                "description_preview": doc.page_content[:180] + "...",
            })

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "total_sources": len(sources),
        }

    def chat(self, question: str) -> str:
        """Retourne directement le texte de la réponse générée."""
        res = self.ask(question)
        return res["answer"]

