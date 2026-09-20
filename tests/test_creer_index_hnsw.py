"""Tests unitaires pour le script de création de la base vectorielle FAISS HNSW.

Valide spécifiquement le fonctionnement du mode `--dry` (simulation) :
- Vérifie la récupération et le filtrage des données depuis l'API OpenAgenda.
- Vérifie la prise en compte des paramètres CLI (--dry, --dry-run, --limit, -l).
- S'assure qu'aucun appel à l'API Mistral d'embeddings ni écriture sur disque n'est effectué.
- Valide la gestion de l'absence de clé API en mode dry.
- Valide l'exécution du point d'entrée __main__.
- Tous les tests s'exécutent 100% hors-ligne via simulation (mock) de l'API OpenAgenda.
"""

import os
import runpy
from unittest.mock import MagicMock, patch

import pytest

from src.creer_index_hnsw import main, reconstruire_index_hnsw


@pytest.fixture
def fake_openagenda_records():
    """Jeu de données simulé représentatif des données OpenAgenda."""
    return {
        "total_count": 3,
        "results": [
            {
                "uid": "evt-1",
                "title_fr": "Concert Jazz en plein air",
                "description_fr": "Un magnifique concert de jazz dans le parc.",
                "location_city": "Lille",
                "firstdate_begin": "2026-06-15T20:00:00+02:00",
            },
            {
                "uid": "evt-2",
                "title_fr": "Festival des arts de la rue",
                "description_fr": "Spectacles vivants gratuits pour tous les âges.",
                "location_city": "Amiens",
                "firstdate_begin": "2026-07-20T14:00:00+02:00",
            },
            {
                "uid": "evt-3",
                "title_fr": "Exposition photographique",
                "description_fr": None,  # Description manquante pour tester le filtrage
                "location_city": "Arras",
                "firstdate_begin": "2026-09-10T10:00:00+02:00",
            },
        ],
    }


def test_dry_mode_basic_execution(fake_openagenda_records, capsys):
    """Vérifie que le mode --dry s'exécute, traite les données et s'arrête sans appel d'embedding."""
    mock_response = MagicMock()
    mock_response.json.return_value = fake_openagenda_records
    mock_response.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_response) as mock_get:
        code = main(["--dry"])

    assert code == 0
    mock_get.assert_called_once()

    captured = capsys.readouterr().out
    assert "Récupération des données depuis l'API OpenAgenda" in captured
    assert "taille data : 3" in captured
    # 2 descriptions valides sur 3 (1 description manquante filtrée)
    assert "Descriptions valides à vectoriser : 2" in captured
    assert "Mode DRY activé : simulation terminée" in captured


def test_dry_mode_does_not_call_mistral_or_faiss(fake_openagenda_records):
    """Vérifie expressément que ni le client Mistral ni FAISS ne sont sollicités en mode --dry."""
    mock_response = MagicMock()
    mock_response.json.return_value = fake_openagenda_records
    mock_response.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_response):
        with patch("src.creer_index_hnsw.Mistral") as mock_mistral:
            with patch("src.creer_index_hnsw.faiss.IndexHNSWFlat") as mock_faiss:
                code = main(["--dry"])

    assert code == 0
    mock_mistral.assert_not_called()
    mock_faiss.assert_not_called()


def test_dry_mode_flag_alias_dry_run(fake_openagenda_records, capsys):
    """Vérifie que l'alias --dry-run fonctionne à l'identique de --dry."""
    mock_response = MagicMock()
    mock_response.json.return_value = fake_openagenda_records
    mock_response.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_response):
        code = main(["--dry-run"])

    assert code == 0
    captured = capsys.readouterr().out
    assert "Mode DRY activé : simulation terminée" in captured


def test_dry_mode_with_limit(capsys):
    """Vérifie que le paramètre --limit restreint le volume d'événements récupérés."""
    records_batch = {
        "total_count": 50,
        "results": [
            {"uid": f"evt-{i}", "description_fr": f"Description événement {i}"}
            for i in range(1, 3)
        ],
    }
    mock_response = MagicMock()
    mock_response.json.return_value = records_batch
    mock_response.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_response) as mock_get:
        code = main(["--dry", "--limit", "2"])

    assert code == 0
    assert mock_get.call_count == 1
    call_params = mock_get.call_args[1]["params"]
    assert call_params["limit"] == 2

    captured = capsys.readouterr().out
    assert "taille data : 2" in captured
    assert "Descriptions valides à vectoriser : 2" in captured


def test_dry_mode_short_limit_flag(capsys):
    """Vérifie le fonctionnement du flag raccourci -l."""
    records_batch = {
        "total_count": 50,
        "results": [{"uid": "evt-1", "description_fr": "Desc 1"}],
    }
    mock_response = MagicMock()
    mock_response.json.return_value = records_batch
    mock_response.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_response) as mock_get:
        code = main(["--dry", "-l", "1"])

    assert code == 0
    assert mock_get.call_args[1]["params"]["limit"] == 1


def test_dry_mode_pagination_with_limit(capsys):
    """Vérifie la pagination séquentielle jusqu'à atteindre la limite demandée."""
    page_1 = {
        "total_count": 5,
        "results": [{"uid": f"evt-{i}", "description_fr": f"Desc {i}"} for i in range(1, 3)],
    }
    page_2 = {
        "total_count": 5,
        "results": [{"uid": "evt-3", "description_fr": "Desc 3"}],
    }
    mock_resp1 = MagicMock()
    mock_resp1.json.return_value = page_1
    mock_resp1.raise_for_status.return_value = None

    mock_resp2 = MagicMock()
    mock_resp2.json.return_value = page_2
    mock_resp2.raise_for_status.return_value = None

    with patch("requests.get", side_effect=[mock_resp1, mock_resp2]) as mock_get:
        code = main(["--dry", "--limit", "3"])

    assert code == 0
    assert mock_get.call_count == 2
    captured = capsys.readouterr().out
    assert "taille data : 3" in captured
    assert "Descriptions valides à vectoriser : 3" in captured


def test_dry_mode_without_api_key(fake_openagenda_records):
    """Vérifie que le mode --dry n'exige PAS la variable MISTRAL_API_KEY."""
    mock_response = MagicMock()
    mock_response.json.return_value = fake_openagenda_records
    mock_response.raise_for_status.return_value = None

    with patch("dotenv.load_dotenv"):
        with patch.dict(os.environ, {"MISTRAL_API_KEY": ""}, clear=False):
            with patch("requests.get", return_value=mock_response):
                code = main(["--dry"])

    assert code == 0


def test_missing_api_key_raises_error_without_dry():
    """Vérifie que sans --dry, l'absence de clé API lève une ValueError explicite."""
    with patch("dotenv.load_dotenv"):
        with patch.dict(os.environ, {"MISTRAL_API_KEY": ""}, clear=False):
            with pytest.raises(ValueError, match="La variable MISTRAL_API_KEY est manquante"):
                main([])


def test_dry_mode_fallback_column_description(capsys):
    """Vérifie que la colonne alternative 'Description' est utilisée si 'description_fr' est absente."""
    records = {
        "total_count": 1,
        "results": [
            {
                "uid": "evt-fallback",
                "Description": "Description issue du champ alternatif.",
            }
        ],
    }
    mock_response = MagicMock()
    mock_response.json.return_value = records
    mock_response.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_response):
        code = main(["--dry"])

    assert code == 0
    captured = capsys.readouterr().out
    assert "Descriptions valides à vectoriser : 1" in captured


def test_dry_mode_empty_api_results(capsys):
    """Vérifie que le script gère convenablement une réponse vide de l'API sans planter."""
    empty_records = {
        "total_count": 0,
        "results": [],
    }
    mock_response = MagicMock()
    mock_response.json.return_value = empty_records
    mock_response.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_response):
        code = main(["--dry"])

    assert code == 0
    captured = capsys.readouterr().out
    assert "taille data : 0" in captured
    assert "Descriptions valides à vectoriser : 0" in captured
    assert "Mode DRY activé : simulation terminée" in captured


def test_dry_mode_sys_argv_invocation(fake_openagenda_records, capsys):
    """Vérifie l'invocation via sys.argv sans argument passé à main()."""
    mock_response = MagicMock()
    mock_response.json.return_value = fake_openagenda_records
    mock_response.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_response):
        with patch("sys.argv", ["creer_index_hnsw.py", "--dry", "--limit", "2"]):
            code = main()

    assert code == 0
    captured = capsys.readouterr().out
    assert "Mode DRY activé : simulation terminée" in captured


def test_dry_mode_runpy_entrypoint(fake_openagenda_records, capsys):
    """Vérifie l'exécution complète du fichier en tant que module __main__ (CLI réel)."""
    mock_response = MagicMock()
    mock_response.json.return_value = fake_openagenda_records
    mock_response.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_response):
        with patch("sys.argv", ["creer_index_hnsw.py", "--dry"]):
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_path("src/creer_index_hnsw.py", run_name="__main__")
            assert exc_info.value.code == 0

    captured = capsys.readouterr().out
    assert "Mode DRY activé : simulation terminée" in captured


def test_reconstruire_index_hnsw_dry_mode(fake_openagenda_records):
    """Vérifie que la fonction reconstruire_index_hnsw s'exécute en mode dry et retourne 0."""
    mock_response = MagicMock()
    mock_response.json.return_value = fake_openagenda_records
    mock_response.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_response):
        count = reconstruire_index_hnsw(dry=True, api_key="test_key")

    assert count == 0


def test_reconstruire_index_hnsw_full_pipeline_mocked(fake_openagenda_records, tmp_path):
    """Vérifie l'exécution complète de reconstruire_index_hnsw avec mocks sans appel réseau réel."""
    mock_response = MagicMock()
    mock_response.json.return_value = fake_openagenda_records
    mock_response.raise_for_status.return_value = None

    # Simuler le client Mistral retournant 2 embeddings (2 descriptions valides)
    mock_mistral_client = MagicMock()
    mock_mistral_client.embeddings.create.return_value = MagicMock(
        data=[
            MagicMock(embedding=[0.05] * 1024),
            MagicMock(embedding=[0.10] * 1024),
        ]
    )

    out_dir = tmp_path / "test_rebuilt_hnsw"

    with patch("requests.get", return_value=mock_response):
        with patch("src.creer_index_hnsw.Mistral", return_value=mock_mistral_client):
            with patch("src.creer_index_hnsw.time.sleep"):  # Pas d'attente réelle
                with patch("src.creer_index_hnsw.FAISS") as mock_faiss_class:
                    mock_vs = MagicMock()
                    mock_faiss_class.return_value = mock_vs
                    count = reconstruire_index_hnsw(
                        limit=2,
                        dry=False,
                        output_path=out_dir,
                        api_key="mock_key",
                    )

    assert count == 2
    mock_faiss_class.assert_called_once()
    mock_vs.add_embeddings.assert_called_once()
    mock_vs.save_local.assert_called_once_with(str(out_dir))

