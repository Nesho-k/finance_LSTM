# src/features.py — Feature engineering pour les données financières CAC 40
# [MODIFIÉ] Remplace les features météo (tavg, tmin, tmax, prcp, tsun…)
#           par des indicateurs techniques financiers (returns, RSI, moving averages…)

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.data import fetch_daily_history


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI (Relative Strength Index) — indicateur de momentum."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    return 100 - 100 / (1 + rs)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les indicateurs techniques à partir des colonnes OHLCV brutes.
    Convention : 'close' est la colonne 0 (cible), même rôle que 'tavg' dans le modèle météo.
    La logique ffill/bfill/fillna est conservée identique à l'original.
    """
    out = df.copy()

    # Mise en index temporel (même logique que l'original sur 'time')
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date")

    # --- Returns et volatilité ---
    out["returns"]      = out["close"].pct_change()
    out["log_returns"]  = np.log(out["close"] / out["close"].shift(1))
    out["volatility_5"] = out["returns"].rolling(5).std()

    # --- Moyennes mobiles normalisées par le prix (scale-invariant) ---
    out["ma5"]  = out["close"].rolling(5).mean()  / out["close"]
    out["ma20"] = out["close"].rolling(20).mean() / out["close"]
    out["ma60"] = out["close"].rolling(60).mean() / out["close"]

    # --- RSI (14 jours) ---
    out["rsi"] = compute_rsi(out["close"], window=14)

    # --- Volume normalisé (z-score rolling 20j) ---
    out["volume_norm"] = (
        (out["volume"] - out["volume"].rolling(20).mean())
        / (out["volume"].rolling(20).std() + 1e-9)
    )

    # --- Amplitude Haut-Bas relative ---
    out["hl_range"] = (out["high"] - out["low"]) / out["close"]

    # Ordre des features — 'close' en position 0 (cible, identique à 'tavg' dans météo)
    features = [
        "close",
        "returns",
        "log_returns",
        "volatility_5",
        "ma5",
        "ma20",
        "ma60",
        "rsi",
        "volume_norm",
        "hl_range",
    ]

    out = out[features]
    # Même stratégie de remplissage que l'original (ffill → bfill → 0)
    out = out.ffill().bfill().fillna(0)
    return out


def build_feature_matrix(
    ticker: str = "cac40",
    feature_order_path: str = "models/cac40/feature_order.json",
    window: int = 60,
) -> np.ndarray:
    """
    Télécharge les données CAC 40 et retourne la matrice numpy (window, n_features)
    prête pour l'inférence — même interface que la fonction météo originale.
    """
    # On télécharge window + 80 jours pour absorber les rolling windows (max=60j)
    raw   = fetch_daily_history(days=window + 80)
    feats = engineer_features(raw)

    if not Path(feature_order_path).is_absolute():
        project_root       = Path(__file__).parent.parent
        feature_order_path = project_root / feature_order_path

    with open(feature_order_path) as f:
        order = json.load(f)

    X = feats.tail(window)[order].to_numpy(dtype=np.float32)
    return X
