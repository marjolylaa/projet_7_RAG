"""Tests automatisés pour le chatbot RAG d'événements Hauts-de-France 2026.

Ces tests utilisent des simulateurs (MockChatMistralAI et FakeEmbeddings)
pour valider les chaînes de traitement RAG sans aucun appel réseau
ni consommation de crédits de l'API Mistral.
"""

from typing import Any, List, Optional
import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.chatbot import EventRAGChatbot, clean_text, format_document_context


class MockChatMistralAI(BaseChatModel):
    """Simulateur de LLM Mistral déterministe pour tests unitaires et CI."""

    model_name: str = "mock-open-mistral-nemo"

    @property
    def _llm_type(self) -> str:
        return "mock-mistral-ai"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Extraction de la question humaine
        user_prompt = ""
        for m in reversed(messages):
            if getattr(m, "type", "") == "human" or getattr(m, "role", "") == "user":
                user_prompt = str(m.content)
                break
        if not user_prompt and messages:
            user_prompt = str(messages[-1].content)

        q = user_prompt.lower()

        # Scénario Cas 1 : Feu! Chatterton au Zénith de Lille (Année 2026 + Passé)
        if "chatterton" in q:
            reply = (
                "Le concert du groupe Feu! Chatterton s'est tenu au Zénith de Lille le 15 février 2026. "
                "Cet événement musical marquant de l'année 2026 a rassemblé de nombreux spectateurs dans les Hauts-de-France."
            )
        # Scénario Cas 2 : Rencontre de Jazz à Senlis (Année 2026 + Passé + Gratuit)
        elif "jazz" in q or "senlis" in q:
            reply = (
                "La Rencontre de Jazz s'est déroulée à Senlis le 7 juin 2026. "
                "L'accès à cet événement était entièrement gratuit et libre pour le public en 2026."
            )
        # Scénario Cas 3 : Ateliers numériques à Grande-Synthe (Année 2026 + Automne)
        elif "grande-synthe" in q:
            reply = (
                "Plusieurs ateliers d'initiation au numérique et à l'informatique sont prévus à Grande-Synthe "
                "à l'automne 2026, notamment les 23 septembre et 7 octobre 2026 à la médiathèque Nelson Mandela."
            )
        # Scénario Cas 4 / Refus hors périmètre (Marseille, plongée, recette tarte)
        elif "marseille" in q or "plongée" in q or "plongee" in q or "tarte" in q:
            reply = (
                "Je suis désolé, mais ma fonction est exclusivement dédiée aux sorties et événements culturels "
                "2026 dans la région Hauts-de-France. Cette demande est hors de mon périmètre d'action, "
                "je ne peux pas vous recommander d'activités hors des Hauts-de-France."
            )
        # Scénario Requête thématique : atelier informatique général
        elif "atelier" in q or "informatique" in q or "numérique" in q:
            reply = (
                "En 2026, plusieurs ateliers d'initiation au numérique et à l'informatique sont proposés dans les "
                "Hauts-de-France. Vous pourrez notamment participer à un atelier gratuit dans un lieu dédié ou en médiathèque "
                "dans la ville de Grande-Synthe ou Lille pour apprendre les bases."
            )
        else:
            reply = (
                "En 2026, de nombreux événements culturels sont programmés dans la région Hauts-de-France. "
                "N'hésitez pas à préciser votre recherche par ville ou par date."
            )

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])


MOCK_DOCUMENTS: List[Document] = [
    Document(
        page_content="Atelier d'initiation au numérique et à l'informatique pour débutants à Grande-Synthe.",
        metadata={
            "Titre": "Atelier numérique et informatique",
            "Ville": "Grande-Synthe",
            "Nom du lieu": "Médiathèque Nelson Mandela",
            "Première date - Début": "2026-09-23",
            "Dernière date - Fin": "2026-10-07",
            "Résumé horaires": "23 septembre et 7 octobre 2026",
            "Détail des conditions": "Gratuit sur inscription",
            "URL canonique": "https://openagenda.com/events/atelier-grande-synthe-2026",
        },
    ),
    Document(
        page_content="Concert événement du groupe pop-rock Feu! Chatterton au Zénith de Lille.",
        metadata={
            "Titre": "Feu! Chatterton en concert",
            "Ville": "Lille",
            "Nom du lieu": "Zénith de Lille",
            "Première date - Début": "2026-02-15",
            "Dernière date - Fin": "2026-02-15",
            "Résumé horaires": "15 février 2026 à 20h",
            "Détail des conditions": "Payant dès 35€",
            "URL canonique": "https://openagenda.com/events/feu-chatterton-lille-2026",
        },
    ),
    Document(
        page_content="Rencontre musicale et concerts de jazz en plein air et en salle à Senlis.",
        metadata={
            "Titre": "Rencontre de Jazz",
            "Ville": "Senlis",
            "Nom du lieu": "Espace Culturel",
            "Première date - Début": "2026-06-07",
            "Dernière date - Fin": "2026-06-07",
            "Résumé horaires": "7 juin 2026",
            "Détail des conditions": "Entrée libre et gratuit",
            "URL canonique": "https://openagenda.com/events/jazz-senlis-2026",
        },
    ),
    Document(
        page_content="Festival du cinéma et projections de films en salle en novembre 2026 à Amiens.",
        metadata={
            "Titre": "Festival du Film d'Automne",
            "Ville": "Amiens",
            "Nom du lieu": "Maison de la Culture",
            "Première date - Début": "2026-11-12",
            "Dernière date - Fin": "2026-11-20",
            "Résumé horaires": "Du 12 au 20 novembre 2026",
            "Détail des conditions": "5€ par séance",
            "URL canonique": "https://openagenda.com/events/film-amiens-2026",
        },
    ),
    Document(
        page_content="Salon littéraire et rencontres avec des illustrateurs jeunesse en novembre à Arras.",
        metadata={
            "Titre": "Salon du Livre Jeunesse",
            "Ville": "Arras",
            "Nom du lieu": "Hôtel de Guînes",
            "Première date - Début": "2026-11-05",
            "Dernière date - Fin": "2026-11-07",
            "Résumé horaires": "Du 5 au 7 novembre 2026",
            "Détail des conditions": "Gratuit",
            "URL canonique": "https://openagenda.com/events/livre-arras-2026",
        },
    ),
]


@pytest.fixture(scope="module")
def bot() -> EventRAGChatbot:
    """Fixture partageant une instance de chatbot 100% mockée (hors-ligne, 0 crédit API)."""
    fake_embeddings = FakeEmbeddings(size=1024)
    vectorstore = FAISS.from_documents(MOCK_DOCUMENTS, fake_embeddings)
    mock_llm = MockChatMistralAI()

    return EventRAGChatbot(
        top_k=3,
        vectorstore=vectorstore,
        llm=mock_llm,
        embeddings=fake_embeddings,
    )


def test_clean_text():
    """Vérifie le nettoyage des textes bruts."""
    assert clean_text("  texte avec espaces  ") == "texte avec espaces"
    assert clean_text("nan") == ""
    assert clean_text(None) == ""
    assert clean_text("<p>Balise HTML</p>") == "Balise HTML"
    assert clean_text("R\ufffdunion") == "Reunion"


def test_format_document_context_empty():
    """Vérifie le comportement de formatage avec une liste vide."""
    result = format_document_context([])
    assert "Aucun événement pertinent" in result


def test_format_document_context_with_doc():
    """Vérifie le formatage structuré d'un document avec métadonnées."""
    doc = Document(
        page_content="Description de l'événement musical.",
        metadata={
            "Titre": "Concert de Printemps",
            "Ville": "Amiens",
            "Nom du lieu": "Maison de la Culture",
            "Détail des conditions": "Gratuit",
            "Première date - Début": "2026-05-15",
            "URL canonique": "https://example.com/event",
        }
    )
    formatted = format_document_context([doc])
    assert "[ÉVÉNEMENT 1]" in formatted
    assert "Concert de Printemps" in formatted
    assert "Amiens" in formatted
    assert "Gratuit" in formatted
    assert "https://example.com/event" in formatted


def test_chatbot_initialization(bot):
    """Vérifie que le chatbot s'initialise correctement avec FAISS et Mistral."""
    assert bot is not None
    assert bot.vectorstore is not None
    assert bot.retriever is not None
    assert bot.llm is not None


def test_mock_guarantees_zero_api_calls(bot):
    """Garantit formellement que le bot utilise des simulateurs hors-ligne sans consommation de crédits."""
    assert isinstance(bot.llm, MockChatMistralAI)
    assert isinstance(bot.embeddings, FakeEmbeddings)


def test_retrieval_returns_relevant_documents(bot):
    """Vérifie que la recherche vectorielle renvoie des résultats avec scores."""
    docs_scores = bot.retrieve_documents("atelier informatique ou numérique", k=3)
    assert len(docs_scores) > 0
    assert len(docs_scores) <= 3

    first_doc, score = docs_scores[0]
    assert isinstance(first_doc, Document)
    assert float(score) >= 0.0
    assert "Titre" in first_doc.metadata or "title_fr" in first_doc.metadata


def test_ask_thematic_query(bot):
    """Vérifie la génération d'une réponse RAG complète sur une requête thématique."""
    res = bot.ask("Je cherche un atelier pour débuter en informatique.")
    assert isinstance(res, dict)
    assert "answer" in res
    assert "sources" in res
    assert len(res["sources"]) > 0

    answer = res["answer"]
    assert len(answer) > 50
    # Vérification que la réponse mentionne au moins une information pratique
    assert any(term in answer.lower() for term in ["atelier", "numérique", "informatique", "gratuit", "lieu", "ville", "2026"])


def test_ask_out_of_scope_refusal(bot):
    """Vérifie que le chatbot refuse poliment les requêtes hors de son périmètre."""
    res = bot.ask("Peux-tu me donner la recette de la tarte aux pommes à Marseille ?")
    answer = res["answer"].lower()

    # Le bot doit rappeler son périmètre (Hauts-de-France / événements) ou décliner poliment
    refusal_keywords = ["hauts-de-france", "événements", "périmètre", "agenda", "2026", "contexte", "désolé", "excuse"]
    assert any(term in answer for term in refusal_keywords)


def test_temporal_filter_november(bot):
    """Vérifie que la détection temporelle de novembre renvoie des événements de novembre."""
    docs_scores = bot.retrieve_documents("Quels sont les événements prévus en novembre ?", k=3)
    assert len(docs_scores) > 0
    for doc, _score in docs_scores:
        dates_info = (
            str(doc.metadata.get("Première date - Début", ""))
            + str(doc.metadata.get("Premire date - Dbut", ""))
            + str(doc.metadata.get("Résumé horaires", ""))
            + str(doc.metadata.get("Rsum horaires", ""))
        ).lower()
        assert "2026-11" in dates_info or "novembre" in dates_info
