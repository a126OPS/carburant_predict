# -*- coding: utf-8 -*-
# !pip install gradio -q
import gradio as gr
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Télécharger les modèles s'ils n'existent pas (pour Hugging Face Spaces)
try:
    from download_models import download_if_missing
    download_if_missing()
except Exception as e:
    print(f"[AVERTISSEMENT] Impossible de télécharger les modèles: {e}")

# ============================================================
# CHARGEMENT DU MODÈLE ET DES DONNÉES
# ============================================================

# Charger le modèle entraîné
try:
    model_data = joblib.load('modele_carburant.joblib')
    print('[OK] Modele charge')
except FileNotFoundError:
    raise FileNotFoundError('Impossible de trouver modele_carburant.joblib. '
                            'Exécute d\'abord le notebook pour entraîner le modèle.')

# Extraire les composants du modèle
pipe_ridge = model_data['pipeline']
FEATURES_FINALES = model_data['features']
le_carb = model_data['le_carb']
le_dept = model_data['le_dept']
HORIZON = model_data['horizon']
mae_carb = model_data['mae_par_carb']
aujourd_hui = pd.Timestamp(model_data['date_entrainement'])
date_prochain = pd.Timestamp(model_data['date_prochain_reentrain'])

# Charger le cache runtime pour les données
try:
    cache = joblib.load('runtime_cache_2026.joblib')
    df_dept_jour = cache['df_dept_jour']
    df_complet = df_dept_jour.copy()  # Approximation : on utilise df_dept_jour comme df_complet
    n_observations = cache.get('n_rows', len(df_dept_jour))
    print('[OK] Donnees cachees chargees')
except FileNotFoundError:
    raise FileNotFoundError('Impossible de trouver runtime_cache_2026.joblib. '
                            'Exécute d\'abord le notebook pour générer le cache.')

# ============================================================
# FONCTION DE PRÉDICTION
# ============================================================

def predire_depuis_aujourd_hui(dept, carburant, horizon=7):
    """
    Prédit le prix d'un carburant dans 'horizon' jours.

    Utilise les 30 derniers jours comme features.
    Retourne None si données trop anciennes (> 60 jours).

    Note : mae_carb est en EUR (pas de division par 100 necessaire)
    """
    mask = (
        (df_dept_jour['departement'] == dept) &
        (df_dept_jour['carburant']   == carburant)
    )
    hist = df_dept_jour[mask].sort_values('date').tail(60)

    if len(hist) < 14:
        return None, None, None, None

    prix_serie    = hist.set_index('date')['prix_median']
    derniere_date = prix_serie.index[-1]

    # Vérification fraîcheur des données
    jours_ecart = (aujourd_hui - derniere_date).days
    if jours_ecart > 60:
        return None, None, None, None

    prix_actuel  = float(prix_serie.iloc[-1])
    date_predite = derniere_date + timedelta(days=horizon)
    p            = prix_serie.values

    features = {
        'prix_lag_1' : p[-1],
        'prix_lag_2' : p[-2]  if len(p)>=2  else p[-1],
        'prix_lag_3' : p[-3]  if len(p)>=3  else p[-1],
        'prix_lag_4' : p[-4]  if len(p)>=4  else p[-1],
        'prix_lag_5' : p[-5]  if len(p)>=5  else p[-1],
        'prix_lag_6' : p[-6]  if len(p)>=6  else p[-1],
        'prix_lag_7' : p[-7]  if len(p)>=7  else p[-1],
        'prix_lag_10': p[-10] if len(p)>=10 else p[-1],
        'prix_lag_14': p[-14] if len(p)>=14 else p[-1],
        'prix_lag_21': p[-21] if len(p)>=21 else p[-1],
        'prix_lag_30': p[-30] if len(p)>=30 else p[-1],
        'ma_3j'      : float(np.mean(p[-3:])),
        'ma_7j'      : float(np.mean(p[-7:])),
        'ma_14j'     : float(np.mean(p[-14:])),
        'ma_30j'     : float(np.mean(p[-30:])),
        'tendance_1j' : float(p[-1]-p[-2])  if len(p)>=2  else 0,
        'tendance_7j' : float(p[-1]-p[-8])  if len(p)>=8  else 0,
        'tendance_14j': float(p[-1]-p[-15]) if len(p)>=15 else 0,
        'tendance_30j': float(p[-1]-p[-31]) if len(p)>=31 else 0,
        'volatilite_7j' : float(np.std(p[-7:]))  if len(p)>=7  else 0,
        'volatilite_30j': float(np.std(p[-30:])) if len(p)>=30 else 0,
        'mois'              : date_predite.month,
        'semaine'           : date_predite.isocalendar()[1],
        'jour_semaine'      : date_predite.weekday(),
        'trimestre'         : date_predite.quarter,
        'annee'             : date_predite.year,
        'est_ete'           : int(date_predite.month in [6,7,8]),
        'est_hiver'         : int(date_predite.month in [12,1,2]),
        'vacances_scolaires': int(date_predite.month in [7,8]),
        'carburant_enc'     : (
            int(le_carb.transform([carburant])[0])
            if carburant in le_carb.classes_ else 0
        ),
        'dept_enc'          : (
            int(le_dept.transform([dept])[0])
            if dept in le_dept.classes_ else 0
        )
    }

    df_pred     = pd.DataFrame([features])[FEATURES_FINALES]
    prix_predit = float(pipe_ridge.predict(df_pred)[0])
    mae_val     = mae_carb.get(carburant, 0.03)  # en EUR

    return prix_actuel, prix_predit, mae_val, date_predite

NOMS_DEPTS = {
    '01':'Ain','02':'Aisne','03':'Allier',
    '04':'Alpes-de-Haute-Provence','05':'Hautes-Alpes',
    '06':'Alpes-Maritimes','07':'Ardèche','08':'Ardennes',
    '09':'Ariège','10':'Aube','11':'Aude','12':'Aveyron',
    '13':'Bouches-du-Rhône','14':'Calvados','15':'Cantal',
    '16':'Charente','17':'Charente-Maritime','18':'Cher',
    '19':'Corrèze','21':"Côte-d'Or",'22':"Côtes-d'Armor",
    '23':'Creuse','24':'Dordogne','25':'Doubs','26':'Drôme',
    '27':'Eure','28':'Eure-et-Loir','29':'Finistère',
    '30':'Gard','31':'Haute-Garonne','32':'Gers',
    '33':'Gironde','34':'Hérault','35':'Ille-et-Vilaine',
    '36':'Indre','37':'Indre-et-Loire','38':'Isère',
    '39':'Jura','40':'Landes','41':'Loir-et-Cher',
    '42':'Loire','43':'Haute-Loire','44':'Loire-Atlantique',
    '45':'Loiret','46':'Lot','47':'Lot-et-Garonne',
    '48':'Lozère','49':'Maine-et-Loire','50':'Manche',
    '51':'Marne','52':'Haute-Marne','53':'Mayenne',
    '54':'Meurthe-et-Moselle','55':'Meuse','56':'Morbihan',
    '57':'Moselle','58':'Nièvre','59':'Nord','60':'Oise',
    '61':'Orne','62':'Pas-de-Calais','63':'Puy-de-Dôme',
    '64':'Pyrénées-Atlantiques','65':'Hautes-Pyrénées',
    '66':'Pyrénées-Orientales','67':'Bas-Rhin',
    '68':'Haut-Rhin','69':'Rhône','70':'Haute-Saône',
    '71':'Saône-et-Loire','72':'Sarthe','73':'Savoie',
    '74':'Haute-Savoie','75':'Paris','76':'Seine-Maritime',
    '77':'Seine-et-Marne','78':'Yvelines',
    '79':'Deux-Sèvres','80':'Somme','81':'Tarn',
    '82':'Tarn-et-Garonne','83':'Var','84':'Vaucluse',
    '85':'Vendée','86':'Vienne','87':'Haute-Vienne',
    '88':'Vosges','89':'Yonne',
    '90':'Territoire de Belfort','91':'Essonne',
    '92':'Hauts-de-Seine','93':'Seine-Saint-Denis',
    '94':'Val-de-Marne','95':"Val-d'Oise"
}

# Récupérer les départements disponibles
DEPTS_DISPO = sorted(df_dept_jour['departement'].unique())
CHOIX_DEPTS = [f"{d} — {NOMS_DEPTS.get(d, d)}" for d in DEPTS_DISPO]
CARBURANTS_DISPO = ['Diesel', 'SP95', 'SP98', 'E85', 'GPL']
LABELS_DEPTS = {d: f"{d} — {NOMS_DEPTS.get(d, d)}" for d in DEPTS_DISPO}
DEPTS_PAR_NOM = {NOMS_DEPTS.get(d, d).casefold(): d for d in DEPTS_DISPO}
CARBURANTS_PAR_NOM = {carb.casefold(): carb for carb in CARBURANTS_DISPO}


def normaliser_departement(dept):
    """Accepte un code departement, un label UI ou un nom de departement."""
    if dept is None:
        return None

    if isinstance(dept, (int, np.integer)):
        dept = f"{int(dept):02d}"
    else:
        dept = str(dept).strip()
        if '—' in dept:
            dept = dept.split('—', 1)[0].strip()
        if dept.isdigit():
            dept = dept.zfill(2)
        else:
            dept = DEPTS_PAR_NOM.get(dept.casefold())

    if dept in DEPTS_DISPO:
        return dept
    return None


def normaliser_carburant(carburant):
    """Accepte un carburant en casse libre."""
    if carburant is None:
        return None
    return CARBURANTS_PAR_NOM.get(str(carburant).strip().casefold())


def construire_reponse_prediction(dept, carburant, horizon):
    """Construit une reponse JSON stable reutilisable par Gradio et un portfolio."""
    dept_normalise = normaliser_departement(dept)
    if dept_normalise is None:
        return {
            'ok': False,
            'code': 'departement_invalide',
            'error': 'Departement invalide. Utilise un code a 2 chiffres comme "75" ou un libelle comme "75 — Paris".',
        }

    carburant_normalise = normaliser_carburant(carburant)
    if carburant_normalise is None:
        return {
            'ok': False,
            'code': 'carburant_invalide',
            'error': f'Carburant invalide. Valeurs acceptees : {", ".join(CARBURANTS_DISPO)}.',
        }

    try:
        horizon_val = int(horizon)
    except (TypeError, ValueError):
        return {
            'ok': False,
            'code': 'horizon_invalide',
            'error': 'Horizon invalide. Utilise un nombre entier entre 1 et 14.',
        }

    if not 1 <= horizon_val <= 14:
        return {
            'ok': False,
            'code': 'horizon_hors_limite',
            'error': 'Horizon invalide. Utilise un nombre entier entre 1 et 14.',
        }

    res = predire_depuis_aujourd_hui(dept_normalise, carburant_normalise, horizon_val)
    if res[0] is None:
        return {
            'ok': False,
            'code': 'donnees_indisponibles',
            'error': (
                'Donnees insuffisantes ou trop anciennes pour cette combinaison. '
                'SP95 peut etre rare dans certains departements : essaie SP98 si besoin.'
            ),
            'input': {
                'departement': dept_normalise,
                'departement_nom': NOMS_DEPTS.get(dept_normalise, dept_normalise),
                'departement_label': LABELS_DEPTS.get(dept_normalise, dept_normalise),
                'carburant': carburant_normalise,
                'horizon_jours': horizon_val,
            },
        }

    prix_actuel, prix_predit, mae_val, date_pred = res
    variation = prix_predit - prix_actuel
    variation_pct = (variation / prix_actuel) * 100 if prix_actuel else 0.0
    nom_dept = NOMS_DEPTS.get(dept_normalise, dept_normalise)

    if variation > 0.005:
        direction = 'hausse'
        tendance = f'Hausse prevue : {variation:+.3f} EUR/L ({variation_pct:+.2f}%)'
        conseil = 'Faire le plein maintenant avant la hausse'
    elif variation < -0.005:
        direction = 'baisse'
        tendance = f'Baisse prevue : {variation:+.3f} EUR/L ({variation_pct:+.2f}%)'
        conseil = 'Attendre : le prix devrait baisser'
    else:
        direction = 'stable'
        tendance = f'Prix stable ({variation:+.3f} EUR/L)'
        conseil = 'Pas d urgence particuliere : prix stable'

    return {
        'ok': True,
        'input': {
            'departement': dept_normalise,
            'departement_nom': nom_dept,
            'departement_label': LABELS_DEPTS.get(dept_normalise, dept_normalise),
            'carburant': carburant_normalise,
            'horizon_jours': horizon_val,
        },
        'prediction': {
            'date_reference': aujourd_hui.strftime('%Y-%m-%d'),
            'date_prediction': date_pred.strftime('%Y-%m-%d'),
            'prix_actuel_eur_l': round(prix_actuel, 3),
            'prix_predit_eur_l': round(prix_predit, 3),
            'variation_eur_l': round(variation, 3),
            'variation_pct': round(variation_pct, 2),
            'borne_basse_eur_l': round(prix_predit - mae_val, 3),
            'borne_haute_eur_l': round(prix_predit + mae_val, 3),
            'incertitude_eur_l': round(mae_val, 3),
            'incertitude_centimes': round(mae_val * 100, 1),
        },
        'tendance': {
            'direction': direction,
            'resume': tendance,
            'conseil': conseil,
        },
        'meta': {
            'observations': int(n_observations),
            'source': 'data.economie.gouv.fr',
            'date_entrainement': aujourd_hui.strftime('%Y-%m-%d'),
            'date_prochain_reentrainement': date_prochain.strftime('%Y-%m-%d'),
        },
    }


def predire_api(dept, carburant, horizon):
    """Endpoint API JSON pour un portfolio ou une integration front."""
    return construire_reponse_prediction(dept, carburant, horizon)


def predire_interface(dept_str, carburant, horizon):
    prediction = construire_reponse_prediction(dept_str, carburant, horizon)

    if not prediction['ok']:
        return f"[ERREUR] {prediction['error']}"

    entree = prediction['input']
    resultat = prediction['prediction']
    tendance = prediction['tendance']

    return f"""
## {entree['carburant']} — {entree['departement_nom']} ({entree['departement']})

| Aujourd'hui ({aujourd_hui.strftime('%d/%m/%Y')}) | Le {pd.Timestamp(resultat['date_prediction']).strftime('%d/%m/%Y')} |
|:---:|:---:|
| **{resultat['prix_actuel_eur_l']:.3f} EUR/L** | **{resultat['prix_predit_eur_l']:.3f} EUR/L** |

### [TENDANCE] {tendance['resume']}

### [CONSEIL] {tendance['conseil']}

### [CONFIANCE] Fourchette de confiance
| A la baisse | **Prediction** | A la hausse |
|:-----------:|:--------------:|:-----------:|
| {resultat['borne_basse_eur_l']:.3f} EUR/L | **{resultat['prix_predit_eur_l']:.3f} EUR/L** | {resultat['borne_haute_eur_l']:.3f} EUR/L |

*(±{resultat['incertitude_centimes']:.1f} centimes d'incertitude)*

---
*Modele : {n_observations:,} observations (2022 → {aujourd_hui.strftime('%d/%m/%Y')})*
*Source : data.economie.gouv.fr | [LIMITE] Ne predit pas les chocs geopolitiques*
    """


with gr.Blocks(
    title='PRIX CARBURANT — Prediction J+7',
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(f"""
    # PREDICTION PRIX CARBURANT
    ## France entière — Mis à jour le {aujourd_hui.strftime('%d/%m/%Y')}
    > Entraîné sur **{n_observations:,} observations** (2022 → aujourd'hui)
    > Source : data.economie.gouv.fr
    ---
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown('### LOCALISATION')
            dept_input = gr.Dropdown(
                choices=CHOIX_DEPTS, value='75 — Paris',
                label='Département'
            )
            carb_input = gr.Dropdown(
                choices=CARBURANTS_DISPO,
                value='Diesel', label='Carburant'
            )
            horizon_input = gr.Slider(
                minimum=1, maximum=14, value=7, step=1,
                label='Prédire dans combien de jours ?'
            )

    # Composants caches pour exposer une API JSON simple via Gradio.
    api_dept_input = gr.Textbox(label='Code departement', visible=False)
    api_carb_input = gr.Textbox(label='Carburant', visible=False)
    api_horizon_input = gr.Number(label='Horizon', value=7, precision=0, visible=False)
    api_output = gr.JSON(label='Prediction API', visible=False)
    api_btn = gr.Button('API predict', visible=False)

    gr.Markdown('---')
    btn    = gr.Button('[PREDICT] Predire le prix futur', variant='primary', size='lg')
    output = gr.Markdown('_Selectionne un departement et clique sur Predire_')

    btn.click(
        fn=predire_interface,
        inputs=[dept_input, carb_input, horizon_input],
        outputs=output,
        api_name=False,
        show_api=False
    )

    api_btn.click(
        fn=predire_api,
        inputs=[api_dept_input, api_carb_input, api_horizon_input],
        outputs=api_output,
        api_name='predict',
        queue=False
    )

    gr.Markdown('---\n### [EXAMPLES] Exemples')
    gr.Examples(
        examples=[
            ['75 — Paris',            'Diesel', 7],
            ['75 — Paris',            'SP98',   7],
            ['71 — Saône-et-Loire',   'E85',   14],
            ['69 — Rhône',            'SP98',   3],
            ['13 — Bouches-du-Rhône', 'E85',    7],
            ['33 — Gironde',          'GPL',    7],
        ],
        inputs=[dept_input, carb_input, horizon_input]
    )

    gr.Markdown(f"""
    ---
    ### [INFO] Signification des carburants
    | Sigle | Nom | Pour qui ? |
    |:------|:----|:-----------|
    | **Diesel** | Gazole | Moteurs diesel |
    | **SP95** | Sans Plomb 95 (5% ethanol) | Moteurs essence standard |
    | **SP98** | Sans Plomb 98 | Moteurs haut de gamme / sportifs |
    | **E85** | Superethanol 85% | Vehicules flex-fuel uniquement |
    | **GPL** | Gaz de Petrole Liquefie | Vehicules avec kit GPL |

    ### [PREC] Precision du modele (test sur 3 derniers mois)
    | Carburant | MAE |
    |:----------|:---:|
    | GPL | ~1.6 centimes |
    | E85 | ~2.1 centimes |
    | Diesel | ~2.4 centimes |
    | SP98 | ~2.4 centimes |
    | SP95 | ~2.5 centimes |

    ### [LIMITS] Limites
    - Ne predit pas les **chocs geopolitiques** (guerre, OPEP, decisions politiques)
    - SP95 rare dans certains departements -> preferer SP98
    - **Reentrainer le {date_prochain.strftime('%d/%m/%Y')}** pour maintenir la precision
    """)


if __name__ == "__main__":
    demo.launch()
