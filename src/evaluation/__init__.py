"""Package d'évaluation et de benchmark du système RAG.

Regroupe :
- scenarios : Définition des cas de test et vérités terrain
- evaluate_rag : Évaluation standardisée Ragas (Faithfulness, Answer Correctness, Answer Relevancy)
- evaluer_scenarios : Exécution du benchmark complet avec génération du rapport HTML interactif
"""

from src.evaluation.scenarios import SCENARIOS
from src.evaluation.evaluate_rag import evaluate_with_ragas
from src.evaluation.evaluer_scenarios import run_evaluation

__all__ = [
    "SCENARIOS",
    "evaluate_with_ragas",
    "run_evaluation",
]
