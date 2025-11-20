# Quand on va récupérer les données, vu qu'on a crée des features supplémentaires,
# il faut aussi les calculer avec les données récentes

# src/features.py
from __future__ import annotations
import requests, pandas as pd, numpy as np, json
from datetime import date, timedelta
from pathlib import Path

from src.data import fetch_daily_history

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()

    # --- Index temporel --- (vérifier que l'index est bien datetime)
    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"])
        out = out.set_index("time")

    out['dayofyear'] = out.index.dayofyear
    out['month'] = out.index.month

    out['month_sin'] = np.sin(2 * np.pi * out['month']/12)
    out['month_cos'] = np.cos(2 * np.pi * out['month']/12)
    out['day_sin'] = np.sin(2 * np.pi * out['dayofyear']/365)
    out['day_cos'] = np.cos(2 * np.pi * out['dayofyear']/365)

    out['tavg_roll_3'] = out['tavg'].rolling(window=3).mean()
    out['tavg_roll_7'] = out['tavg'].rolling(window=7).mean()
    out['tsun_roll_7'] = out['tsun'].rolling(window=7).mean()

    features = ["tavg", "tmin", "tmax", "prcp", "tsun", "tavg_roll_3", "tavg_roll_7", "tsun_roll_7", "month_sin", "month_cos", "day_sin", "day_cos"
]

    out = out[features]
    out = out.ffill().bfill()
    # ffill = forward fill (remplit à partir de la dernière valeur connue), bfill = backward fill (remplit à partir de la prochaine valeur connue, utile si les premières valeurs sont NaN)

    return out

def build_feature_matrix(city: str, feature_order_path: str, window: int = 14) -> np.ndarray:
    raw = fetch_daily_history(city, days=window + 6)  # on prend un peu plus pour les rolling
    feats = engineer_features(raw)

    # Résoudre le chemin relatif à partir de la racine du projet
    if not Path(feature_order_path).is_absolute():
        project_root = Path(__file__).parent.parent.parent  # remonte de src/ à meteo_app/ à meteo/
        feature_order_path = project_root / feature_order_path

    with open(feature_order_path) as f:
        order = json.load(f)
    X = feats.tail(window)[order].to_numpy(dtype=np.float32)
    return X

#if __name__ == "__main__":
#    x = build_feature_matrix("berlin", "models/berlin/feature_order.json", window=30)
#    print(x.shape)