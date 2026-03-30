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

# 📊 PREDICTION PRIX CARBURANT - France

Prédictions IA du prix du carburant en France pour les **14 prochains jours**.

## 🚀 Accès rapide

L'application est **déployée et prête à l'emploi** sur cet espace.

## ⚙️ Fonctionnement

- **Modèle** : Ridge Regression avec 31 features
- **Données** : 33,000+ observations (2022-2026)
- **Source** : data.economie.gouv.fr
- **Précision** : ±2-3 centimes (MAE)
- **Couverture** : 95 départements, 5 carburants

## 📋 Comment utiliser

1. Sélectionnez votre **département**
2. Choisissez le **type de carburant** (Diesel, SP95, SP98, E85, GPL)
3. Définissez l'**horizon** de prédiction (1-14 jours)
4. Consultez la **tendance** et les **recommandations**

## 📊 Carburants disponibles

- **Diesel** : Carburant véhicules commerciaux
- **SP95** : Carburant essence standard (en déclin)
- **SP98** : Essence premium (disponible partout)
- **E85** : Supercarburant éthanol (stations sélectionnées)
- **GPL** : Gaz liquéfié (stations sélectionnées)

## ⚠️ Limitations

- ❌ Ne prédit pas les chocs géopolitiques
- ❌ Données brutes (pas de lissage officiel)
- ❌ Pas de marges de distribution
- ❌ Coupure de données possible les jours fériés

## 📦 Technologies

- Python 3.13
- Gradio (interface web)
- scikit-learn (Ridge Regression)
- Pandas & NumPy (traitement données)
- Joblib (sérialisation modèle)

## 🔧 Code source

- **GitHub** : https://github.com/a126OPS/projet_public
- **License** : MIT

## 📈 Métriques du modèle

| Carburant | MAE (€/L) |
|-----------|-----------|
| Diesel    | 0.024     |
| SP95      | 0.022     |
| SP98      | 0.023     |
| E85       | 0.024     |
| GPL       | 0.015     |

---

**Dernière mise à jour** : 22 mars 2026  
**Prochain réentraînement** : 22 juin 2026
