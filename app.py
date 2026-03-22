#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio interface for carburant prediction
Hugging Face Spaces deployment
"""

import os
import sys

# Télécharger les modèles s'ils n'existent pas
try:
    from download_models import download_if_missing
    download_if_missing()
except Exception as e:
    print(f"[AVERTISSEMENT] {e}")

# Charger et lancer l'interface
from interface import demo

if __name__ == "__main__":
    demo.launch()
