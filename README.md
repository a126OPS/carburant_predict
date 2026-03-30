# ⛽ Prédiction du Prix du Carburant

## Description

Ce modèle prédit le **prix du carburant à la pompe** (SP95, SP98, Diesel, E85, GPL) en France à partir de variables temporelles et géographiques. Il permet d'anticiper les évolutions de prix et de comparer les tarifs selon les régions.

- **Repo modèle Hugging Face :** https://huggingface.co/a126OPS/carburant_price_predict
- **Space Hugging Face :** https://huggingface.co/spaces/a126OPS/carburant_predict

## Utilisation

```python
import joblib
from huggingface_hub import hf_hub_download

# Chargement des artefacts du modèle
model_path = hf_hub_download(
    repo_id="a126OPS/carburant_price_predict",
    filename="modele_carburant.joblib",
)
model_data = joblib.load(model_path)
pipeline = model_data["pipeline"]

print(type(pipeline).__name__)
```

## Données d'entraînement

- **Source :** Open data gouvernemental — prix des carburants en France (data.gouv.fr)
- **Variables d'entrée :** type de carburant, département, période (mois/année)
- **Variable cible :** prix en €/litre

## API

L'application Gradio déployée sur le Space Hugging Face expose aussi une **API JSON** pour l'intégrer facilement dans un portfolio ou un site vitrine.

- **Space :** `https://huggingface.co/spaces/a126OPS/carburant_predict`
- **Endpoint runtime :** `POST https://<ton-space>.hf.space/api/predict`
- **Entrées :** `departement`, `carburant`, `horizon`
- **Formats acceptés pour le département :** `"75"` ou `"75 — Paris"`

Exemple de requête :

```json
{
  "data": ["75", "Diesel", 7]
}
```

Exemple JavaScript :

```js
const response = await fetch("https://<ton-space>.hf.space/api/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    data: ["75", "Diesel", 7],
  }),
});

const payload = await response.json();
const result = payload.data[0];

if (!result.ok) {
  console.error(result.error);
} else {
  console.log(result.prediction.prix_predit_eur_l);
  console.log(result.tendance.conseil);
}
```

## Limites

- Les chocs ponctuels (crises géopolitiques, taxes exceptionnelles) ne sont pas modélisés
- La précision varie selon le type de carburant
- Modèle entraîné sur données françaises uniquement

## Auteur

Développé par [a126OPS](https://huggingface.co/a126OPS)  
🔗 Modèle : [carburant_price_predict](https://huggingface.co/a126OPS/carburant_price_predict)  
🔗 Démo interactive et API : [carburant_predict](https://huggingface.co/spaces/a126OPS/carburant_predict)

## Licence

[MIT](https://opensource.org/licenses/MIT)
