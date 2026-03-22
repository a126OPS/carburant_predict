---
title: Carburant Predict
emoji: ⛽
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
license: mit
---

# Carburant Predict

Prediction du prix des carburants en France pour les 14 prochains jours.

## Fonctionnement

- Modele de regression Ridge
- Donnees par departement et type de carburant
- Source: data.economie.gouv.fr
- Intervalle de confiance base sur la MAE du modele

## Utilisation

1. Choisissez un departement.
2. Choisissez un carburant.
3. Choisissez un horizon de prediction entre 1 et 14 jours.
4. Lancez la prediction pour obtenir la tendance et le conseil associe.

## Notes

- Les fichiers `modele_carburant.joblib` et `runtime_cache_2026.joblib` sont telecharges au demarrage si besoin.
- Le Space utilise `app.py` comme point d'entree.
- Les dependances runtime sont definies dans `requirements.txt`.
