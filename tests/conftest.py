import os
import sys
from pathlib import Path

# Clé factice pour les tests afin de ne jamais exiger de clé réelle ni consommer de crédits
os.environ.setdefault("MISTRAL_API_KEY", "mock-mistral-api-key-for-offline-tests")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
