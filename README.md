# CAC 40 — Prédiction de cours financiers par LSTM

Modèle de prédiction de séries temporelles financières basé sur une architecture LSTM (Long Short-Term Memory).  
Adapté depuis un projet de prédiction météorologique (meteo_LSTM) — l'architecture du réseau de neurones est conservée à l'identique ; seules les données et le contexte changent.

## Présentation

Le modèle apprend les dynamiques historiques de l'indice **CAC 40 (^FCHI)** sur 5 ans et prédit les **5 prochains cours de clôture** (horizon J+1 … J+5) à partir d'une fenêtre glissante de 60 jours.

Les entrées combinent les prix OHLCV bruts et des indicateurs techniques calculés à la volée :

| Feature | Description |
|---|---|
| `close` | Cours de clôture (cible, colonne 0) |
| `returns` | Rendement journalier (%) |
| `log_returns` | Log-rendement |
| `volatility_5` | Volatilité rolling 5 jours |
| `ma5 / ma20 / ma60` | Moyennes mobiles normalisées par le prix |
| `rsi` | Relative Strength Index (14 jours) |
| `volume_norm` | Volume normalisé (z-score rolling 20j) |
| `hl_range` | Amplitude Haut-Bas relative |

## Architecture LSTM

Architecture **identique** au projet météo original :

```
LSTM(64, return_sequences=True)
Dropout(0.2)
LSTM(32, return_sequences=False)
Dropout(0.2)
Dense(5)                         ← 5 jours de prédiction
```

- Optimiseur : Adam
- Loss : MSE
- Early stopping (patience=15) + ReduceLROnPlateau

## Résultats

| Métrique | Valeur |
|---|---|
| RMSE global (test set) | **161 pts** (~2 % du cours) |
| MAE global (test set) | **128 pts** (~1,6 % du cours) |
| Fenêtre d'entrée | 60 jours |
| Horizon de prédiction | 5 jours ouvrés |
| Données | CAC 40, 5 ans (août 2021 → mai 2026), ~1 250 séances |

Les graphiques de résultats sont sauvegardés dans `plots/` après l'entraînement :
- `loss_curve.png` — courbe de loss train/val
- `predictions_vs_real.png` — cours réels vs prédictions (J+1) sur le test set
- `rmse_per_horizon.png` — RMSE par horizon de prédiction

## Utilisation

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Entraîner le modèle
```bash
python train.py
```
Télécharge automatiquement les données CAC 40 via yfinance, entraîne le LSTM et sauvegarde les artefacts dans `models/cac40/`.

### 3. Lancer l'API FastAPI
```bash
uvicorn src.api:app --reload
```
Documentation interactive : http://localhost:8000/docs

### 4. Exemple d'appel API
```bash
# Prédiction des 5 prochains jours
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"asset": "cac40", "horizon": 5}'
```

Réponse :
```json
{
  "asset": "cac40",
  "ticker": "^FCHI",
  "horizon": 5,
  "predictions": [7520.3, 7535.1, 7498.7, 7512.0, 7541.8],
  ...
}
```

## Structure du projet

```
meteo_LSTM/
├── train.py                  # Script d'entraînement (MODIFIÉ)
├── requirements.txt          # Dépendances (MODIFIÉ : yfinance)
├── models/
│   └── cac40/                # Artefacts du modèle (NEW)
│       ├── model.keras
│       ├── scaler.pkl
│       ├── feature_order.json
│       └── meta.json
├── plots/                    # Visualisations générées
├── cache/yfinance/           # Cache local des données
└── src/
    ├── data.py               # Téléchargement yfinance (MODIFIÉ)
    ├── features.py           # Indicateurs techniques (MODIFIÉ)
    ├── inference.py          # FinancialForecaster (MODIFIÉ)
    └── api.py                # API FastAPI (MODIFIÉ)
```

## Technologies

- **Python 3.x** · TensorFlow/Keras · scikit-learn
- **yfinance** — données OHLCV (remplace Meteostat)
- **FastAPI** — API REST de prédiction
- **pandas / numpy / matplotlib**

---

*Projet dérivé de [meteo_LSTM](https://github.com/Nesho-k/meteo_LSTM) — architecture LSTM conservée, domaine applicatif transposé vers la finance.*
