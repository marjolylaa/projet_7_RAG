"""Tests d'intégration de l'API REST FastAPI pour pytest.

Exécute la suite de tests définie dans api_test.py dans le cadre du runner de tests global.
"""

from api_test import (
    client,
    test_root_endpoint,
    test_ui_endpoint,
    test_health_endpoint,
    test_swagger_documentation,
    test_ask_valid_question,
    test_ask_empty_question_returns_400,
    test_ask_invalid_payload_returns_422,
    test_rebuild_post_endpoint,
    test_rebuild_get_endpoint,
    test_rebuild_security_protection,
)

__all__ = [
    "client",
    "test_root_endpoint",
    "test_ui_endpoint",
    "test_health_endpoint",
    "test_swagger_documentation",
    "test_ask_valid_question",
    "test_ask_empty_question_returns_400",
    "test_ask_invalid_payload_returns_422",
    "test_rebuild_post_endpoint",
    "test_rebuild_get_endpoint",
    "test_rebuild_security_protection",
]
