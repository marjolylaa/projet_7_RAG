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
        dept = clean_text(meta.get("Département") or meta.get("department") or "")
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
        
        # Prise en compte de la description longue et complète
        desc_longue = clean_text(meta.get("Description longue") or meta.get("longdescription_fr") or "")
        content_clean = clean_text(doc.page_content)
        if desc_longue and len(desc_longue) > len(content_clean):
            description = desc_longue
        else:
            description = content_clean

        # Extraction des coordonnées et modalités d'inscription
        reg = meta.get("Registration")
        reg_contacts = []
        if reg and str(reg) != "nan":
            try:
                reg_data = json.loads(reg) if isinstance(reg, str) else reg
                if isinstance(reg_data, list):
                    for item in reg_data:
                        val = item.get("value") or item.get("link")
                        if val:
                            reg_contacts.append(str(val))
            except Exception:
                pass

        lines = [f"[ÉVÉNEMENT {i}]"]
        lines.append(f"- Titre : {titre}")
        if nom_lieu and ville:
            lines.append(f"- Lieu : {nom_lieu}, {ville}")
        elif ville:
            lines.append(f"- Ville : {ville}")
        if dept:
            lines.append(f"- Département : {dept}")
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

        if reg_contacts:
            lines.append(f"- Contact / Inscription : {', '.join(reg_contacts)}")

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

Ton objectif est de fournir des réponses précises, personnalisées, chaleureuses et pertinentes en te basant EXCLUSIVEMENT sur les événements fournis ci-dessous dans la section CONTEXTE.

Consignes impératives :
1. Prise en compte des questions ciblées (Résultat unique et exclusif) :
   - Si la question de l'utilisateur porte sur un événement précis, un lieu spécifique ou un atelier identifié (ex: un atelier numérique à Vervins, le Mercredi des tout-petits à Don, un festival à Mortefontaine, un spectacle à Lille) :
     -> Réponds STRICTEMENT et UNIQUEMENT sur cet événement spécifique en puisant dans tous ses champs (lieu, département, dates, horaires, tarifs, modalités et contacts d'inscription, description).
     -> N'ajoute AUCUNE suggestion ou alternative dans d'autres villes non sollicitées à la fin de ta réponse ! Limite-toi à l'événement demandé, même si l'événement comporte la mention 'COMPLET' dans son titre ou est déjà passé.
   - Si la question est générale ou exploratoire (ex: "quelles sorties en famille ce week-end ?"), présente alors plusieurs recommandations pertinentes issues du contexte.

2. Validité intégrale des événements de l'année 2026 (Passé vs Futur) :
   - Tous les événements de l'année 2026, qu'ils soient passés (antérieurs à {current_date}) ou à venir, constituent tous des informations VALIDES et RÉELLES de l'agenda régional.
   - Si l'utilisateur demande des informations sur un événement passé de 2026, fournis TOUS ses détails pratiques en utilisant la concordance des temps passés (passé composé ou imparfait, ex: "Cet atelier s'est tenu du 22 mai au 24 juillet 2026", "Il a proposé des initiations...").
   - Ne prétends JAMAIS qu'un événement n'est pas organisé ou n'existe pas sous prétexte que sa date est passée !
   - Pour les événements à venir (marqués comme À VENIR) : Utilise le futur ou le présent d'anticipation (ex: "aura lieu le", "se tiendra").
   - Pour les événements en cours (marqués comme EN COURS) : Utilise le présent.

3. Précision factuelle et anti-hallucination :
   - Mentionne OBLIGATOIREMENT l'année (2026) pour chaque date citée.
   - Indique fidèlement les tarifs, horaires et conditions d'accès indiqués dans le contexte.
   - N'invente JAMAIS d'événement, de date, de lieu ou de tarif qui ne figure pas dans le contexte.
   - Règle de fidélité stricte (anti-hallucination) : Ne formule AUCUNE affirmation factuelle qui ne soit directement déductible du CONTEXTE. Si une information demandée (tarif, contact, horaires précis) n'est pas spécifiée dans le CONTEXTE, indique clairement qu'elle n'est pas renseignée plutôt que de l'extrapoler ou de la deviner.
   - Évite les ajouts non sourcés ou conseils généraux non documentés pour garantir un ancrage factuel irréprochable.

4. Règle de refus et de périmètre :
   - Si la demande est totalement hors de propos (ex: mécanique automobile, question théorique hors domaine, ou demande pour une autre région comme Marseille ou l'étranger) : décline poliment et fermement en rappelant ton périmètre (les événements culturels et sorties 2026 dans les Hauts-de-France). Réoriente brièvement de manière courtoise vers les organismes adaptés ou des ateliers participatifs locaux (ex: Repair Cafés ou ateliers d'aide à la réparation de vélos), sans détailler de longues fiches de recrutement ou d'offres d'emploi sans rapport.
   - Si le contexte ne contient aucun événement pertinent pour satisfaire la requête, indique-le avec courtoisie en précisant qu'aucun événement correspondant n'a été trouvé dans l'agenda actuel des Hauts-de-France pour 2026.

5. Style, hiérarchie visuelle et indentation stricte :
   - Rédige en français soigné, clair, engageant et parfaitement structuré.
   - Structure hiérarchique OBLIGATOIRE pour les programmes, plannings, horaires et déroulés d'activités :
     • L'horaire / le créneau horaire et le titre de l'animation/atelier DOIVENT TOUJOURS constituer une puce principale de premier niveau (ex: `- **20h30 – 22h30 : Découverte de la Crypte Néogothique**`).
     • Tous les détails associés (description, intervenants/artistes, morceaux/œuvres au programme, conditions d'accès, réservations) DOIVENT TOUJOURS être indentés sous forme de sous-puces au niveau inférieur avec 2 espaces et un tiret (`  - `).
     • Exemple impératif de structure attendue :
       - **20h30 – 22h30 : Découverte de la Crypte Néogothique**
         - Visite guidée par Les Amis de la Cathédrale ND de la Treille.
         - Accès depuis le Transept Sud – Places limitées, prévoir une file d’attente.
       - **21h00 : Vivaldi-Express avec Quinte&sens (dirigé par Olivier Molendi-Coste)**
         - Médley musical inspiré de Harry Potter (J. Williams, P. Doyle), Concerto pour 4 violons (A. Vivaldi), Beethoven, Einaudi...
         - Formation : Septuor incluant violon, violoncelle, piano, hautbois et basson.
     • INTERDICTION FORMELLE : Ne jamais mettre les horaires et les lignes de détails au même niveau d'indentation, et ne jamais écrire les détails sous forme de texte brut collé à la marge sans sous-puce indentée.


CONTEXTE DES ÉVÉNEMENTS DISPONIBLES :
{context}
"""


def normalize_title_for_dedup(title: str) -> str:
    """Normalise un titre d'événement pour éviter qu'un cycle récurrent ne monopolise le top-k."""
    cleaned = clean_text(title).lower()
    # Élimination des suffixes de séance du type 1/6, 2/6, session 3, partie 2, etc.
    cleaned = re.sub(r"\s*[-/]?\s*\d+\s*/\s*\d+\s*", "", cleaned)
    cleaned = re.sub(r"\s*[-/]?\s*(?:session|séance|seance|épisode|partie|volet|n°?)\s*\d+\s*", "", cleaned)
    cleaned = re.sub(r"\s*[-/]\s*$", "", cleaned)
    return cleaned.strip()


class EventRAGChatbot:
    """Chatbot RAG pour la recommandation d'événements avec FAISS et Mistral AI."""

    def __init__(
        self,
        index_dir: Optional[str | Path] = None,
        embedding_model: str = "mistral-embed",
        llm_model: str = "open-mistral-nemo",
        temperature: float = 0.0,
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
            p_str = str(path).strip().lower()
            if p_str in ("hnsw", "fast", "rapide"):
                return self.root / "src" / "mon_index_langchain_evenements_hnsw_rapide"
            if p_str in ("flat", "l2", "exact"):
                return self.root / "src" / "mon_index_langchain_evenements"

            candidate = Path(path)
            if candidate.is_absolute() and candidate.exists():
                return candidate
            if (self.root / candidate).exists():
                return self.root / candidate
            if (self.root / "src" / candidate).exists():
                return self.root / "src" / candidate

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
        """Effectue une recherche par similarité avec déduplication stricte et re-ranking géographique."""
        limit = k or self.top_k

        # Détection automatique de critère temporel dans la requête
        if filter_func is None:
            temp_filter = get_temporal_filter(query)
            if temp_filter:
                filter_func, _month = temp_filter

        # Élargissement substantiel du pool de candidats pour permettre la déduplication et le re-ranking
        fetch_k = 600 if filter_func else max(limit * 8, 30)
        try:
            raw_results = self.vectorstore.similarity_search_with_score(
                query, k=fetch_k if deduplicate else limit, filter=filter_func, fetch_k=fetch_k
            )
        except Exception:
            raw_results = self.vectorstore.similarity_search_with_score(query, k=limit * 4)

        if not deduplicate:
            return raw_results[:limit]

        q_lower = query.lower()
        seen_titles = set()
        candidates = []

        for doc, score in raw_results:
            raw_title = doc.metadata.get("Titre") or doc.metadata.get("title_fr") or doc.page_content[:60]
            norm_title = normalize_title_for_dedup(raw_title)

            # Élimination des quasi-doublons (ex: ateliers 1/6, 2/6, 3/6)
            if norm_title and norm_title in seen_titles:
                continue
            seen_titles.add(norm_title)

            # Re-ranking géographique, département et entités :
            doc_city = clean_text(doc.metadata.get("Ville") or "").lower()
            doc_venue = clean_text(doc.metadata.get("Nom du lieu") or "").lower()
            doc_dept = clean_text(doc.metadata.get("Département") or "").lower()
            title_clean = clean_text(raw_title).lower()

            boost = 0.0
            # Boost ville / commune
            if doc_city and len(doc_city) >= 3 and doc_city in q_lower:
                boost += 0.15  # Réduit la distance L2 pour faire remonter la ville expressément demandée
            # Boost lieu
            if doc_venue and len(doc_venue) >= 4 and doc_venue in q_lower:
                boost += 0.10
            # Boost département (ex: 'dans l'Oise')
            if doc_dept and len(doc_dept) >= 4 and doc_dept in q_lower:
                boost += 0.08
            # Boost titre exact ou inclusion
            if title_clean and len(title_clean) >= 4 and (title_clean in q_lower or q_lower in title_clean):
                boost += 0.25
            else:
                # Boost mots significatifs du titre
                title_words = [
                    w for w in re.findall(r"\w+", title_clean)
                    if len(w) >= 4 and w not in {
                        "pour", "dans", "avec", "sans", "tout", "tous", "festival",
                        "atelier", "spectacle", "exposition", "concert", "seance", "session"
                    }
                ]
                matches = sum(1 for w in title_words if w in q_lower)
                if matches > 0:
                    boost += min(0.20, 0.10 * matches)

            adjusted_score = max(0.0, float(score) - boost)
            candidates.append((doc, adjusted_score, float(score)))

        # Tri par score ajusté (plus petite distance = plus pertinent)
        candidates.sort(key=lambda x: x[1])

        return [(doc, orig_score) for doc, adj_score, orig_score in candidates[:limit]]

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

        contexts = [format_document_context([doc]) for doc in docs]
        context_text = format_document_context(docs)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "context_text": context_text,
            "total_sources": len(sources),
        }

    def chat(self, question: str) -> str:
        """Retourne directement le texte de la réponse générée."""
        res = self.ask(question)
        return res["answer"]

    def reload_index(self, index_dir: Optional[str | Path] = None) -> int:
        """Recharge ou bascule l'index FAISS depuis le disque et actualise le retriever.

        Retourne le nombre total de documents indexés.
        """
        if index_dir is not None:
            self.index_path = self._resolve_index_path(index_dir)
        elif self.index_path == Path("mock_index"):
            # En environnement de mock / test injecté, préserver l'index existant
            ntotal = getattr(getattr(self.vectorstore, "index", None), "ntotal", 0)
            if ntotal == 0 and hasattr(self.vectorstore, "docstore") and hasattr(self.vectorstore.docstore, "_dict"):
                ntotal = len(self.vectorstore.docstore._dict)
            return int(ntotal)
        else:
            self.index_path = self._resolve_index_path(self.index_path)

        if self.index_path != Path("mock_index") and self.index_path.exists():
            self.vectorstore = FAISS.load_local(
                folder_path=str(self.index_path),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.top_k}
            )

        ntotal = getattr(getattr(self.vectorstore, "index", None), "ntotal", 0)
        return int(ntotal)

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les métadonnées et statistiques du bot sans exposer d'informations sensibles."""
        ntotal = getattr(getattr(self.vectorstore, "index", None), "ntotal", 0)
        if ntotal == 0 and hasattr(self.vectorstore, "docstore") and hasattr(self.vectorstore.docstore, "_dict"):
            ntotal = len(self.vectorstore.docstore._dict)

        return {
            "status": "ready",
            "total_documents": int(ntotal),
            "index_path": str(self.index_path),
            "llm_model": self.llm_model_name,
            "embedding_model": self.embedding_model_name,
            "top_k": self.top_k,
            "temperature": self.temperature,
        }

