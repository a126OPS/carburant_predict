---
language:
- fr
license: mit
tags:
- tabular-regression
- price-prediction
- fuel
- carburant
- scikit-learn
- joblib
metrics:
- rmse
- r2
---

# ⛽ Prédiction du Prix du Carburant

## Description

Ce modèle prédit le **prix du carburant à la pompe** (SP95, SP98, Diesel, E10, E85) en France à partir de variables temporelles et géographiques. Il permet d'anticiper les évolutions de prix et de comparer les tarifs selon les régions.

## Utilisation

```python
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# Chargement du modèle
model_path = hf_hub_download(repo_id="a126OPS/carburant_predict", filename="model.joblib")
model = joblib.load(model_path)

# Exemple de prédiction
# [type_carburant, departement, mois, annee]
features = np.array([[1, 71, 3, 2025]])  # SP95, Saône-et-Loire, mars 2025
predicted_price = model.predict(features)
print(f"Prix estimé : {predicted_price[0]:.3f} €/L")
```

## Données d'entraînement

- **Source :** Open data gouvernemental — prix des carburants en France (data.gouv.fr)
- **Variables d'entrée :** type de carburant, département, période (mois/année)
- **Variable cible :** prix en €/litre

## Performances

| Métrique | Valeur |
|----------|--------|
| RMSE | À compléter |
| R² | À compléter |

## Limites

- Les chocs ponctuels (crises géopolitiques, taxes exceptionnelles) ne sont pas modélisés
- La précision varie selon le type de carburant
- Modèle entraîné sur données françaises uniquement

## Auteur

Développé par [a126OPS](https://huggingface.co/a126OPS)  
🔗 Démo interactive : [carburant_predict](https://huggingface.co/spaces/a126OPS/carburant_predict)

## Licence

[MIT](https://opensource.org/licenses/MIT)
