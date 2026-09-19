"""Module d'évaluation Ragas (Retrieval Augmented Generation Assessment).

Ce module implémente l'évaluation automatisée du système RAG selon les standards Ragas :
- Faithfulness (Fidélité au contexte documentaire / Anti-hallucination)
- Answer Correctness (Justesse factuelle et sémantique par rapport à la vérité terrain humaine)
- Answer Relevancy (Pertinence de la réponse vis-à-vis de la question utilisateur)

Compatible avec Python 3.13, LangChain et Mistral AI (modèle ChatMistralAI + Embeddings).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configuration des modules stubs pour la compatibilité VertexAI de Ragas sous LangChain Community
for _mod in [
    "langchain_community.chat_models.vertexai",
    "langchain_community.llms.vertexai",
    "langchain_community.embeddings.vertexai",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.SimpleNamespace(
            ChatVertexAI=None,
            VertexAI=None,
            VertexAIEmbeddings=None,
        )

# Assurer la présence des chemins projet dans sys.path
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent.parent if _current_dir.name == "evaluation" else (_current_dir.parent if _current_dir.name == "src" else _current_dir)
for _path_item in (str(_root_dir), str(_root_dir / "src"), str(_current_dir)):
    if _path_item not in sys.path:
        sys.path.insert(0, _path_item)

from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerCorrectness, AnswerRelevancy, Faithfulness
from ragas.run_config import RunConfig

try:
    from src.chatbot import EventRAGChatbot, get_project_root
except ModuleNotFoundError:
    from chatbot import EventRAGChatbot, get_project_root

try:
    from src.evaluation.scenarios import SCENARIOS
except ModuleNotFoundError:
    try:
        from src.scenarios import SCENARIOS
    except ModuleNotFoundError:
        from scenarios import SCENARIOS

logger = logging.getLogger("ragas_eval")


def build_ragas_dataset(
    samples_data: List[Dict[str, Any]]
) -> EvaluationDataset:
    """Construit un objet EvaluationDataset Ragas à partir d'une liste de dictionnaires.

    Chaque dictionnaire doit contenir :
    - user_input (str) : Question utilisateur
    - response (str) : Réponse générée par le chatbot RAG
    - retrieved_contexts (List[str]) : Extraits textuels des sources documentaires
    - reference (str) : Réponse humaine de référence (Ground Truth)
    """
    samples: List[SingleTurnSample] = []
    for item in samples_data:
        contexts = item.get("retrieved_contexts", [])
        if not contexts:
            # Pour les requêtes hors périmètre sans source documentaire
            contexts = ["Aucun document trouvé dans la base d'événements pour cette question hors périmètre."]

        sample = SingleTurnSample(
            user_input=str(item.get("user_input", "")),
            response=str(item.get("response", "")),
            retrieved_contexts=[str(c) for c in contexts],
            reference=str(item.get("reference", "")),
        )
        samples.append(sample)

    return EvaluationDataset(samples=samples)


def evaluate_with_ragas(
    bot: EventRAGChatbot,
    samples_data: List[Dict[str, Any]],
    timeout: int = 120,
    max_workers: int = 1,
    output_json: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Évalue une série d'échantillons RAG avec le framework Ragas et le modèle Mistral AI.

    Args:
        bot: Instance du chatbot EventRAGChatbot.
        samples_data: Liste de dictionnaires contenant user_input, response, retrieved_contexts, reference.
        timeout: Délai d'attente maximum par appel LLM (en secondes).
        max_workers: Nombre d'appels concurrents (1 pour préserver les quotas de l'API Mistral).
        output_json: Chemin d'enregistrement du rapport JSON.

    Returns:
        Dictionnaire avec les scores globaux, les scores par scénario et le chemin du fichier JSON.
    """
    t0 = time.perf_counter()
    print("\n" + "=" * 80)
    print("🔬 DÉMARRAGE DE L'ÉVALUATION AUTOMATISÉE RAGAS")
    print("=" * 80)
    print(f"📊 Métriques évaluées : Faithfulness, Answer Correctness, Answer Relevancy")
    print(f"🤖 Modèle LLM d'évaluation : Mistral AI ({getattr(bot, 'llm_model_name', 'open-mistral-nemo')})")
    print(f"📁 Nombre de scénarios     : {len(samples_data)}")
    print("-" * 80)

    dataset = build_ragas_dataset(samples_data)

    metrics = [
        Faithfulness(),
        AnswerCorrectness(max_retries=3),
        AnswerRelevancy(strictness=1),
    ]

    ragas_llm = LangchainLLMWrapper(bot.llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(bot.embeddings)

    run_cfg = RunConfig(
        timeout=timeout,
        max_workers=max_workers,
        max_retries=3,
    )

    try:
        raw_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_cfg,
            raise_exceptions=False,
            show_progress=True,
        )
    except Exception as exc:
        logger.error(f"Erreur lors de l'exécution de Ragas evaluate: {exc}")
        raw_result = None

    elapsed = time.perf_counter() - t0

    # Extraction des scores par échantillon et calcul des moyennes
    per_scenario_scores: List[Dict[str, Optional[float]]] = []
    global_scores: Dict[str, Optional[float]] = {
        "faithfulness": None,
        "answer_correctness": None,
        "answer_relevancy": None,
    }

    if raw_result is not None:
        try:
            df_res = raw_result.to_pandas()
            for idx, row in df_res.iterrows():
                def _clean_val(v: Any) -> Optional[float]:
                    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                        return None
                    return round(float(v), 4)

                per_scenario_scores.append({
                    "faithfulness": _clean_val(row.get("faithfulness")),
                    "answer_correctness": _clean_val(row.get("answer_correctness")),
                    "answer_relevancy": _clean_val(row.get("answer_relevancy")),
                })

            for m_key in ("faithfulness", "answer_correctness", "answer_relevancy"):
                if m_key == "faithfulness":
                    # Pour Faithfulness, la fidélité documentaire s'évalue sur les scénarios dans le périmètre
                    # disposant de documents sources (les refus hors périmètre n'ont pas de source à ancrer).
                    in_scope_vals = [
                        s[m_key]
                        for idx, s in enumerate(per_scenario_scores)
                        if idx < len(samples_data)
                        and s[m_key] is not None
                        and not samples_data[idx].get("is_out_of_scope", False)
                    ]
                    if in_scope_vals:
                        global_scores[m_key] = round(sum(in_scope_vals) / len(in_scope_vals), 4)
                    else:
                        vals = [s[m_key] for s in per_scenario_scores if s[m_key] is not None]
                        if vals:
                            global_scores[m_key] = round(sum(vals) / len(vals), 4)
                else:
                    vals = [s[m_key] for s in per_scenario_scores if s[m_key] is not None]
                    if vals:
                        global_scores[m_key] = round(sum(vals) / len(vals), 4)
        except Exception as e:
            logger.warning(f"Erreur lors de la conversion des scores Ragas : {e}")

    # Si certains scénarios n'ont pas de score, combler
    while len(per_scenario_scores) < len(samples_data):
        per_scenario_scores.append({
            "faithfulness": None,
            "answer_correctness": None,
            "answer_relevancy": None,
        })

    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(elapsed, 2),
        "total_samples": len(samples_data),
        "global_averages": global_scores,
        "scenarios": [
            {
                "id": samples_data[i].get("id", f"scenario_{i+1}"),
                "question": samples_data[i].get("user_input"),
                "ragas_scores": per_scenario_scores[i],
            }
            for i in range(len(samples_data))
        ],
    }

    # Sauvegarde JSON
    if output_json is None:
        output_json = get_project_root() / "rapport" / "ragas_evaluation_results.json"
    else:
        output_json = Path(output_json)
        if not output_json.is_absolute():
            output_json = get_project_root() / output_json

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Évaluation Ragas achevée avec succès en {elapsed:.1f} s.")
    print(f"📈 Scores moyens Ragas :")
    for k, v in global_scores.items():
        val_str = f"{v * 100:.1f}%" if v is not None else "N/A"
        print(f"   • {k:<20} : {val_str}")
    print(f"📁 Fichier de résultats : {output_json}\n")

    return {
        "global_scores": global_scores,
        "per_scenario_scores": per_scenario_scores,
        "duration_seconds": elapsed,
        "output_path": output_json,
    }


def main() -> None:
    """Exécute l'évaluation Ragas autonome en ligne de commande."""
    parser = argparse.ArgumentParser(description="Évaluation Ragas du système RAG Événements Hauts-de-France.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Chemin de sauvegarde du rapport JSON Ragas (par défaut : rapport/ragas_evaluation_results.json)",
    )
    args = parser.parse_args()

    bot = EventRAGChatbot(top_k=4)

    # Répondre aux questions des scénarios pour générer les contextes et réponses
    samples_data = []
    for scen in SCENARIOS:
        print(f"Requête : {scen['question']}")
        res = bot.ask(scen["question"])
        contexts = res.get("contexts")
        if not contexts:
            contexts = [
                f"{s['titre']} ({s['ville']}) : {s['description_preview']}"
                for s in res.get("sources", [])
            ]
        samples_data.append({
            "id": scen["id"],
            "user_input": scen["question"],
            "response": res["answer"],
            "retrieved_contexts": contexts,
            "reference": scen["reponse_reference_humaine"],
            "is_out_of_scope": scen.get("is_out_of_scope", False),
        })

    evaluate_with_ragas(bot, samples_data, output_json=args.output)


if __name__ == "__main__":
    main()
