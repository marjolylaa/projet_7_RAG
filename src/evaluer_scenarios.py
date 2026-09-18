"""Script d'évaluation rigoureuse et automatisée pour le Chatbot RAG avec le framework Ragas.

Ce script teste la robustesse du chatbot sur 5 scénarios précis :
1. Recherche thématique ciblée (Atelier numérique débutant à Vervins en 2026)
2. Sortie en famille avec enfants (Éveil nature pour tout-petits à Don en septembre 2026)
3. Événement festif ou musical (Morty Jazz Festival à Mortefontaine à l'été 2026)
4. Recherche patrimoniale ciblée (La Nuit des cathédrales 2026 à Lille)
5. Cas limite / Hors périmètre (Réparation mécanique automobile à Marseille -> refus poli et rebond)

L'évaluation repose à 100 % de manière autonome et objective sur le framework standardisé Ragas :
- Answer Correctness (Justesse factuelle et sémantique par décomposition propositionnelle vs vérité terrain)
- Faithfulness (Fidélité au contexte documentaire récupéré / Anti-hallucination)
- Answer Relevancy (Pertinence directe de la réponse vis-à-vis de la question posée)
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Template

# Configuration des modules stubs pour la compatibilité VertexAI de Ragas sous LangChain Community
for _mod in [
    "langchain_community.chat_models.vertexai",
    "langchain_community.llms.vertexai",
    "langchain_community.embeddings.vertexai",
]:
    if _mod not in sys.modules:
        import types
        sys.modules[_mod] = types.SimpleNamespace(
            ChatVertexAI=None,
            VertexAI=None,
            VertexAIEmbeddings=None,
        )

# Assurer la présence des chemins projet dans sys.path
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent if _current_dir.name == "src" else _current_dir
for _path_item in (str(_root_dir), str(_root_dir / "src"), str(_current_dir)):
    if _path_item not in sys.path:
        sys.path.insert(0, _path_item)

try:
    from src.chatbot import EventRAGChatbot, get_project_root
except ModuleNotFoundError:
    from chatbot import EventRAGChatbot, get_project_root

from src.evaluate_rag import evaluate_with_ragas

# Configuration de l'encodage console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


try:
    from src.scenarios import SCENARIOS
except ModuleNotFoundError:
    from scenarios import SCENARIOS



def classify_ragas_evaluation(
    ragas_scores: Dict[str, Optional[float]],
    is_out_of_scope: bool = False,
) -> Tuple[str, str, str]:
    """Attribue une classification qualitative basée exclusivement et rigoureusement sur les métriques standardisées Ragas."""
    correctness = ragas_scores.get("answer_correctness")
    faithfulness = ragas_scores.get("faithfulness")

    corr_val = correctness if (correctness is not None and not math.isnan(correctness)) else 0.0
    faith_val = faithfulness if (faithfulness is not None and not math.isnan(faithfulness)) else 0.0

    if is_out_of_scope:
        if corr_val >= 0.25:
            return (
                "Correcte",
                "badge-success",
                f"Refus cadré et réorientation validés par Ragas (Answer Correctness : {corr_val*100:.1f}%).",
            )
        elif corr_val >= 0.15:
            return (
                "Partiellement correcte",
                "badge-warning",
                f"Refus partiel ou réorientation incomplète (Answer Correctness : {corr_val*100:.1f}%).",
            )
        else:
            return (
                "Incorrecte",
                "badge-danger",
                f"Échec de refus ou hallucination hors domaine (Answer Correctness : {corr_val*100:.1f}%).",
            )

    # Scénarios standards dans le périmètre
    if corr_val >= 0.50:
        return (
            "Correcte",
            "badge-success",
            f"Réponse factuellement juste et conforme aux sources (Ragas Correctness : {corr_val*100:.1f}%).",
        )
    elif corr_val >= 0.30:
        return (
            "Partiellement correcte",
            "badge-warning",
            f"Réponse globalement pertinente mais omettant certains détails factuels (Ragas Correctness : {corr_val*100:.1f}%).",
        )
    else:
        return (
            "Incorrecte",
            "badge-danger",
            f"Réponse erronée ou non conforme à la vérité terrain (Ragas Correctness : {corr_val*100:.1f}%).",
        )


def format_markdown_to_html(text: str) -> str:
    """Convertit du texte Markdown enrichi (listes hiérarchiques à 2 niveaux, gras, italique, titres) en HTML sécurisé."""
    if not text:
        return ""

    safe = html.escape(text)
    # Gras et italique
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", safe)
    # Liens Markdown [titre](url)
    safe = re.sub(
        r"\[([^\]]+)\]\((https?://[^\)]+)\)",
        r'<a href="\2" target="_blank" rel="noopener noreferrer" style="color: var(--primary); text-decoration: underline;">\1</a>',
        safe,
    )

    lines = safe.split("\n")
    html_parts: List[str] = []
    level = 0  # 0: hors liste, 1: niveau 1 (<ul><li>), 2: sous-liste (<ul><li><ul><li>)

    def close_to_level(target: int) -> None:
        nonlocal level
        while level > target:
            html_parts.append("</li></ul>")
            level -= 1

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            close_to_level(0)
            continue

        if stripped.startswith("---") or stripped.startswith("___"):
            close_to_level(0)
            html_parts.append('<hr style="border: none; border-top: 1px solid #e2e8f0; margin: 0.8rem 0;">')
            continue

        if stripped.startswith("### "):
            close_to_level(0)
            title = stripped[4:].strip()
            html_parts.append(f'<h4 style="font-size: 1rem; font-weight: 700; color: #1e293b; margin: 0.8rem 0 0.35rem 0;">{title}</h4>')
            continue

        if stripped.startswith("## "):
            close_to_level(0)
            title = stripped[3:].strip()
            html_parts.append(f'<h3 style="font-size: 1.08rem; font-weight: 700; color: #1e293b; margin: 0.9rem 0 0.4rem 0;">{title}</h3>')
            continue

        m_bullet = re.match(r"^(\s*)([-*•]|\d+\.)\s+(.*)$", raw_line)
        if m_bullet:
            indent_spaces = len(m_bullet.group(1).expandtabs(2))
            content = m_bullet.group(3).strip()

            if indent_spaces >= 2:
                if level == 0:
                    html_parts.append('<ul style="margin: 0.4rem 0 0.6rem 1.4rem; padding: 0;">')
                    html_parts.append('<li style="margin-bottom: 0.3rem;">')
                    html_parts.append('<ul style="margin: 0.2rem 0 0.35rem 1.2rem; padding: 0; list-style-type: circle;">')
                    level = 2
                elif level == 1:
                    html_parts.append('<ul style="margin: 0.2rem 0 0.35rem 1.2rem; padding: 0; list-style-type: circle;">')
                    level = 2
                elif level == 2:
                    html_parts.append("</li>")
                html_parts.append(f'<li style="margin-bottom: 0.25rem;">{content}')
            else:
                close_to_level(1)
                if level == 1:
                    html_parts.append("</li>")
                elif level == 0:
                    html_parts.append('<ul style="margin: 0.4rem 0 0.6rem 1.4rem; padding: 0;">')
                    level = 1
                html_parts.append(f'<li style="margin-bottom: 0.35rem;">{content}')
            continue

        # Ligne indentée sans tiret (ex: détail indenté de 2+ espaces sous une puce)
        m_indent = re.match(r"^(\s{2,})(.*)$", raw_line)
        if m_indent and level > 0:
            sub_content = m_indent.group(2).strip()
            if sub_content:
                if level == 1:
                    html_parts.append('<ul style="margin: 0.2rem 0 0.35rem 1.2rem; padding: 0; list-style-type: circle;">')
                    level = 2
                elif level == 2:
                    html_parts.append("</li>")
                html_parts.append(f'<li style="margin-bottom: 0.25rem;">{sub_content}')
            continue

        close_to_level(0)
        html_parts.append(f'<p style="margin-bottom: 0.55rem;">{stripped}</p>')

    close_to_level(0)
    return "\n".join(html_parts)


def get_faithfulness_explanation(score: Optional[float], is_out_of_scope: bool) -> str:
    """Génère l'explication contextuelle et dynamique du score de Faithfulness (0 mot-clé manuel)."""
    if is_out_of_scope:
        return (
            "Demande hors périmètre (refus poli) : la fidélité documentaire est non applicable (N/A) "
            "car le chatbot ne doit invoquer aucun événement du catalogue régional pour une question hors domaine."
        )
    if score is None or math.isnan(score):
        return "Score non disponible pour ce scénario."
    pct = score * 100
    if pct >= 80:
        return (
            f"Excellente fidélité ({pct:.1f}%) : Ragas a décomposé la réponse en affirmations atomiques et validé "
            "que la quasi-totalité des faits cités (lieu, dates 2026, tarifs, programme) sont directement étayés "
            "par le contexte documentaire sans aucune hallucination."
        )
    elif pct >= 50:
        return (
            f"Fidélité modérée ({pct:.1f}%) : la majorité des faits reposent sur les documents, mais certaines propositions "
            "ou déductions ne sont pas explicitement vérifiables dans les extraits récupérés."
        )
    else:
        return (
            f"Fidélité faible ({pct:.1f}%) : Ragas a détecté des affirmations ou détails qui ne figurent pas "
            "dans le contexte documentaire fourni."
        )


def get_correctness_explanation(score: Optional[float], is_out_of_scope: bool) -> str:
    """Génère l'explication contextuelle et dynamique du score d'Answer Correctness (0 mot-clé manuel)."""
    if score is None or math.isnan(score):
        return "Score non disponible pour ce scénario."
    pct = score * 100
    if is_out_of_scope:
        return (
            f"Justesse du refus ({pct:.1f}%) : Ragas valide la conformité du refus avec la référence humaine "
            "(rappel du périmètre régional Hauts-de-France 2026 et réorientation courtoise)."
        )
    if pct >= 70:
        return (
            f"Très bonne justesse ({pct:.1f}%) : forte correspondance entre les faits générés et la réponse de référence humaine "
            "(vrais positifs TP élevés, très peu d'omissions FN) combinée à une similarité sémantique élevée (75% faits + 25% embeddings)."
        )
    elif pct >= 45:
        return (
            f"Justesse partielle ({pct:.1f}%) : les éléments principaux sont présents, mais certains détails attendus dans la référence "
            "(horaires exacts, contacts précis ou conditions) manquent ou sont formulés différemment."
        )
    else:
        return (
            f"Justesse insuffisante ({pct:.1f}%) : décalage notable entre les faits fournis et la réponse de référence."
        )


def get_relevancy_explanation(score: Optional[float], is_out_of_scope: bool) -> str:
    """Génère l'explication contextuelle et dynamique du score d'Answer Relevancy (0 mot-clé manuel)."""
    if score is None or math.isnan(score):
        return "Score non disponible pour ce scénario."
    pct = score * 100
    if is_out_of_scope:
        return (
            f"Score de pertinence ({pct:.1f}%) : sur un refus hors périmètre, Ragas applique fréquemment la pénalité "
            "« noncommittal » si la réponse indique une impossibilité de répondre au sujet initial."
        )
    if pct >= 80:
        return (
            f"Forte pertinence ({pct:.1f}%) : le juge Ragas a rétro-généré des questions à partir de la réponse et constaté "
            "qu'elles ciblent précisément la demande initiale de l'utilisateur."
        )
    elif pct > 0:
        return (
            f"Pertinence modérée ({pct:.1f}%) : la réponse aborde le sujet mais comporte du contenu périphérique "
            "ou omet certains volets spécifiques de la question."
        )
    else:
        return (
            "Score à 0.0% : Ragas a classé la réponse comme « noncommittal » (évasive ou renvoyant vers l'extérieur) "
            "en raison de phrases de conclusion périphériques ou d'omissions sur certains points demandés."
        )


def load_html_report_template() -> str:
    """Charge le template HTML externe du rapport d'évaluation."""
    root = get_project_root()
    candidates = [
        root / "src" / "templates" / "rapport_template.html",
        root / "templates" / "rapport_template.html",
        Path(__file__).resolve().parent / "templates" / "rapport_template.html",
    ]
    for p in candidates:
        if p.exists():
            return p.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Le fichier de template rapport_template.html est introuvable. Emplacements testés : {[str(c) for c in candidates]}"
    )


def generate_html_report(
    execution_dt: datetime,
    results: List[Dict[str, Any]],
    total_duration: float,
    init_duration: float,
    model_name: str,
    ragas_global: Optional[Dict[str, Optional[float]]] = None,
) -> str:
    """Génère le document HTML autonome du rapport d'évaluation complet centré sur Ragas."""
    formatted_date = execution_dt.strftime("%d/%m/%Y à %H:%M:%S")
    total_scenarios = len(results)
    correct_count = sum(1 for r in results if r["classification"] == "Correcte")
    partial_count = sum(1 for r in results if r["classification"] == "Partiellement correcte")
    incorrect_count = sum(1 for r in results if r["classification"] == "Incorrecte")

    correct_rate = (correct_count / total_scenarios * 100) if total_scenarios else 0.0
    avg_latency = (
        sum(r["latency"] for r in results) / total_scenarios if total_scenarios else 0.0
    )

    ragas_global = ragas_global or {}
    ragas_faith = ragas_global.get("faithfulness")
    ragas_corr = ragas_global.get("answer_correctness")
    ragas_relev = ragas_global.get("answer_relevancy")

    ragas_faith_str = f"{ragas_faith * 100:.1f}%" if ragas_faith is not None else "N/A"
    ragas_corr_str = f"{ragas_corr * 100:.1f}%" if ragas_corr is not None else "N/A"
    ragas_relev_str = f"{ragas_relev * 100:.1f}%" if ragas_relev is not None else "N/A"

    # Préparation des données pour le template Jinja2 (100% logique Python, 0 balise HTML)
    prepared_results = []
    for r in results:
        r_ragas = r.get("ragas", {})
        f_val = r_ragas.get("faithfulness")
        c_val = r_ragas.get("answer_correctness")
        rel_val = r_ragas.get("answer_relevancy")
        is_oos = r.get("is_out_of_scope", False)

        prepared_results.append({
            **r,
            "reponse_reference_html": format_markdown_to_html(r.get("reponse_reference", "")),
            "answer_html": format_markdown_to_html(r.get("answer", "")),
            "faith_str": "N/A*" if is_oos else (f"{f_val * 100:.1f}%" if (f_val is not None and not math.isnan(f_val)) else "-"),
            "corr_str": f"{c_val * 100:.1f}%" if (c_val is not None and not math.isnan(c_val)) else "-",
            "relev_str": f"{rel_val * 100:.1f}%" if (rel_val is not None and not math.isnan(rel_val)) else "-",
            "faith_explain": get_faithfulness_explanation(f_val, is_oos),
            "corr_explain": get_correctness_explanation(c_val, is_oos),
            "relev_explain": get_relevancy_explanation(rel_val, is_oos),
        })

    template_content = load_html_report_template()
    template = Template(template_content)
    return template.render(
        formatted_date=formatted_date,
        total_duration=f"{total_duration:.2f}",
        model_name=html.escape(model_name),
        total_scenarios=total_scenarios,
        correct_rate=f"{correct_rate:.0f}",
        correct_count=correct_count,
        partial_count=partial_count,
        incorrect_count=incorrect_count,
        ragas_corr_str=f"{ragas_corr * 100:.1f}%" if ragas_corr is not None else "N/A",
        ragas_faith_str=f"{ragas_faith * 100:.1f}%" if ragas_faith is not None else "N/A",
        ragas_relev_str=f"{ragas_relev * 100:.1f}%" if ragas_relev is not None else "N/A",
        avg_latency=f"{avg_latency:.2f}",
        results=prepared_results,
    )


def run_evaluation(
    output_html: Optional[str | Path] = None,
    bot: Optional[EventRAGChatbot] = None,
) -> Path:
    """Exécute la série de scénarios et génère un rapport HTML complet évalué de façon 100% autonome par Ragas."""
    execution_dt = datetime.now()
    t_global_start = time.perf_counter()

    print("=" * 80)
    print("🧪 ÉVALUATION RIGUREUSE ET 100% AUTONOME DU CHATBOT RAG AVEC RAGAS")
    print(f"📅 Date d'exécution : {execution_dt.strftime('%d/%m/%Y à %H:%M:%S')}")
    print(f"🎯 Nombre de scénarios : {len(SCENARIOS)}")
    print("=" * 80)

    init_duration = 0.0
    if bot is None:
        print("Chargement du modèle et de la base vectorielle...\n")
        t0 = time.perf_counter()
        bot = EventRAGChatbot(top_k=4)
        init_duration = time.perf_counter() - t0
        print(f"✅ Modèle et base initialisés en {init_duration:.2f} s.\n")

    results: List[Dict[str, Any]] = []

    for i, scen in enumerate(SCENARIOS, 1):
        print("\n" + "=" * 80)
        print(f"🔹 Scénario {i}/{len(SCENARIOS)} : {scen['nom']}")
        print(f"📝 Objectif : {scen['description']}")
        print(f"💬 Question utilisateur : « {scen['question']} »")
        print("-" * 80)

        t_start = time.perf_counter()
        res = bot.ask(scen["question"])
        elapsed = time.perf_counter() - t_start

        print(f"🔍 {len(res['sources'])} sources récupérées par FAISS (avec re-ranking) :")
        for j, src in enumerate(res["sources"], 1):
            titre = src["titre"]
            ville = src["ville"]
            dist = src["score_distance"]
            print(f"   [{j}] {titre} ({ville}) - Distance L2: {dist:.4f}")

        print("\n🤖 Réponse générée par le chatbot :")
        print("-" * 40)
        print(res["answer"])
        print("-" * 40)
        print(f"⏱️ Temps de réponse : {elapsed:.2f} s")

        results.append({
            "id": scen["id"],
            "nom": scen["nom"],
            "description": scen["description"],
            "is_out_of_scope": scen.get("is_out_of_scope", False),
            "question": scen["question"],
            "reponse_reference": scen["reponse_reference_humaine"],
            "answer": res["answer"],
            "sources": res["sources"],
            "contexts": res.get("contexts", []),
            "latency": elapsed,
        })

        time.sleep(0.3)

    # Évaluation 100% autonome par Ragas
    print("\n" + "=" * 80)
    print("🔬 CALCUL DES MÉTRIQUES RAGAS (FAITHFULNESS, CORRECTNESS, RELEVANCY)")
    print("=" * 80)

    samples_data = []
    for r in results:
        contexts = r.get("contexts")
        if not contexts:
            contexts = [
                f"{s['titre']} ({s['ville']}) : {s['description_preview']}"
                for s in r.get("sources", [])
            ]
        samples_data.append({
            "id": r["id"],
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": contexts,
            "reference": r["reponse_reference"],
            "is_out_of_scope": r.get("is_out_of_scope", False),
        })

    ragas_res = evaluate_with_ragas(bot, samples_data)
    ragas_global = ragas_res.get("global_scores", {})
    per_scen_ragas = ragas_res.get("per_scenario_scores", [])

    for idx, r in enumerate(results):
        r_ragas = per_scen_ragas[idx] if idx < len(per_scen_ragas) else {}
        r["ragas"] = r_ragas

        # Classification purement dérivée des métriques Ragas
        classification, badge_class, justification = classify_ragas_evaluation(
            r_ragas, is_out_of_scope=r["is_out_of_scope"]
        )
        r["classification"] = classification
        r["badge_class"] = badge_class
        r["justification"] = justification

    total_duration = time.perf_counter() - t_global_start

    print("\n" + "=" * 80)
    print("📋 RÉCAPITULATIF DE L'ÉVALUATION 100% RAGAS")
    print("=" * 80)
    print(f"{'Scénario':<35} | {'Correctness':<12} | {'Faithful':<10} | {'Relevancy':<10} | {'Classification'}")
    print("-" * 80)
    for r in results:
        r_rag = r.get("ragas", {})
        c_val = r_rag.get("answer_correctness")
        f_val = r_rag.get("faithfulness")
        rel_val = r_rag.get("answer_relevancy")

        c_s = f"{c_val*100:.1f}%" if (c_val is not None and not math.isnan(c_val)) else "N/A"
        f_s = f"{f_val*100:.1f}%" if (f_val is not None and not math.isnan(f_val)) else "N/A"
        rel_s = f"{rel_val*100:.1f}%" if (rel_val is not None and not math.isnan(rel_val)) else "N/A"

        print(f"{r['nom'][:33]:<35} | {c_s:>12} | {f_s:>10} | {rel_s:>10} | {r['classification']}")
    print("=" * 80)

    if output_html:
        out_path = Path(output_html)
        if not out_path.is_absolute():
            out_path = get_project_root() / out_path
    else:
        out_path = get_project_root() / "rapport" / "rapport_evaluation.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = getattr(bot, "llm_model_name", "open-mistral-nemo")
    html_content = generate_html_report(
        execution_dt=execution_dt,
        results=results,
        total_duration=total_duration,
        init_duration=init_duration,
        model_name=model_name,
        ragas_global=ragas_global,
    )
    out_path.write_text(html_content, encoding="utf-8")

    print(f"\n📄 Rapport HTML généré avec succès dans le dossier rapport :")
    print(f"👉 {out_path.resolve()}\n")
    print("🎉 Évaluation 100% Ragas achevée avec succès !")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évaluation autonome des scénarios d'interaction pour le Chatbot RAG avec Ragas et rapport HTML."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Chemin du fichier HTML de sortie (par défaut: rapport/rapport_evaluation.html).",
    )
    args = parser.parse_args()
    run_evaluation(output_html=args.output)


if __name__ == "__main__":
    main()
