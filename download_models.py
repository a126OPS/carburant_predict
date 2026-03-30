"""Telecharge les fichiers modele depuis le repo Hugging Face du modele."""

import urllib.request
from pathlib import Path

MODEL_REPO_URL = "https://huggingface.co/a126OPS/carburant_price_predict"
MODEL_URL = (
    "https://huggingface.co/a126OPS/carburant_price_predict/"
    "resolve/main/modele_carburant.joblib?download=true"
)
CACHE_URL = (
    "https://huggingface.co/a126OPS/carburant_price_predict/"
    "resolve/main/runtime_cache_2026.joblib?download=true"
)


def download_if_missing():
    """Telecharge les fichiers s'ils n'existent pas localement."""
    files = {
        "modele_carburant.joblib": MODEL_URL,
        "runtime_cache_2026.joblib": CACHE_URL,
    }

    for filename, url in files.items():
        filepath = Path(filename)

        if filepath.exists():
            print(f"[OK] {filename} already present")
            continue

        print(f"[DOWNLOAD] {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"[OK] {filename} downloaded")
        except Exception as exc:
            print(f"[ERROR] {exc}")
            print(f"Assurez-vous que les fichiers existent dans {MODEL_REPO_URL}.")


if __name__ == "__main__":
    download_if_missing()
