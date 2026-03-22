"""Télécharge les fichiers modèle depuis GitHub Releases"""
import os
import urllib.request
from pathlib import Path

MODEL_URL = "https://github.com/a126OPS/projet_public/releases/download/v1.0/modele_carburant.joblib"
CACHE_URL = "https://github.com/a126OPS/projet_public/releases/download/v1.0/runtime_cache_2026.joblib"

def download_if_missing():
    """Télécharge les fichiers s'ils n'existent pas localement"""
    
    files = {
        "modele_carburant.joblib": MODEL_URL,
        "runtime_cache_2026.joblib": CACHE_URL,
    }
    
    for filename, url in files.items():
        filepath = Path(filename)
        
        if not filepath.exists():
            print(f"⏬ Téléchargement {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"✅ {filename} téléchargé")
            except Exception as e:
                print(f"❌ Erreur: {e}")
                print(f"   Assurez-vous que la release v1.0 existe sur GitHub")
        else:
            print(f"✓ {filename} déjà présent")

if __name__ == "__main__":
    download_if_missing()
