"""Telecharge les fichiers modele depuis GitHub Releases."""

import urllib.request
from pathlib import Path

MODEL_URL = "https://github.com/a126OPS/projet_public/releases/download/v1.0/modele_carburant.joblib"
CACHE_URL = "https://github.com/a126OPS/projet_public/releases/download/v1.0/runtime_cache_2026.joblib"


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
            print("Assurez-vous que la release GitHub v1.0 existe.")


if __name__ == "__main__":
    download_if_missing()
