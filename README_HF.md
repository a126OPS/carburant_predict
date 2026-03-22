---
title: Carburant Predict
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
---

# PREDICTION PRIX CARBURANT

Prédictions IA du prix du carburant en France pour les 14 prochains jours.

## Fonctionnement

- Modèle Ridge Regression entraîné sur **33,000+ observations** (2022-2026)
- Prédiction par département et type de carburant
- Incertitude estimée (MAE ~2-3 centimes)
- Source : [data.economie.gouv.fr](https://data.economie.gouv.fr)

## Utilisation

1. Sélectionnez votre **département**
2. Choisissez le **type de carburant**
3. Définissez l'**horizon de prédiction** (1-14 jours)
4. Consultez la tendance et les recommandations

## Disponibilité

- 95 départements français
- 5 types de carburant : Diesel, SP95, SP98, E85, GPL
- Mis à jour régulièrement

## Limitations

- ⚠️ Ne prédit pas les chocs géopolitiques
- ⚠️ Données brutes (pas de lissage officiel)
- ⚠️ Marges de distribution non incluses

---

**Source Code** : [GitHub](https://github.com/a126OPS/projet_public)
