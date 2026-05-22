# src/inference.py — Moteur d'inférence LSTM pour la prédiction du CAC 40
# [MODIFIÉ] WeatherForecaster → FinancialForecaster ; Paris/Berlin → CAC 40
# La logique de chargement, scaling et dénormalisation est identique à l'original.

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from src.features import build_feature_matrix

ROOT_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
ASSETS     = ("cac40",)          # [MODIFIÉ] "paris"/"berlin" → "cac40"


@dataclass
class AssetArtifacts:
    model:               object
    scaler:              object
    feature_order_path:  Path
    meta:                dict


class FinancialForecaster:
    """
    Moteur d'inférence LSTM pour la prédiction de cours financiers (CAC 40).
    Interface identique à WeatherForecaster — seul le domaine change.

    Usage :
        f = FinancialForecaster()
        result = f.predict("cac40", horizon=5)
    """

    def __init__(self, models_dir: str | Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.artifacts: dict[str, AssetArtifacts] = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        for asset in ASSETS:
            asset_dir = self.models_dir / asset
            try:
                model              = keras.models.load_model(asset_dir / "model.keras")
                scaler             = joblib.load(asset_dir / "scaler.pkl")
                feature_order_path = asset_dir / "feature_order.json"
                meta_path          = asset_dir / "meta.json"
                meta               = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            except Exception as e:
                raise RuntimeError(f"Impossible de charger les artefacts pour {asset}: {e}")

            self.artifacts[asset] = AssetArtifacts(
                model=model,
                scaler=scaler,
                feature_order_path=feature_order_path,
                meta=meta,
            )

    def _build_X(self, asset: str) -> np.ndarray:
        a           = self.artifacts[asset]
        window_size = a.meta.get("window_size", 60)
        X           = build_feature_matrix(
            ticker=asset,
            feature_order_path=str(a.feature_order_path),
            window=window_size,
        )
        # Scaling identique à l'entraînement (même logique que WeatherForecaster)
        Xs = a.scaler.transform(X)
        if np.isnan(Xs).any():
            raise ValueError(f"NaN détectés après scaling pour {asset}. Vérifiez les features / le scaler.")
        return Xs

    def predict(self, asset: str = "cac40", horizon: int = 5) -> dict:
        asset = asset.lower()
        if asset not in self.artifacts:
            raise ValueError(f"Asset doit être parmi {list(self.artifacts.keys())}")

        window_size = self.artifacts[asset].meta.get("window_size", 60)
        Xs          = self._build_X(asset)
        Xs_batch    = np.expand_dims(Xs, axis=0)   # (1, window_size, n_features)

        model  = self.artifacts[asset].model
        scaler = self.artifacts[asset].scaler

        y_scaled   = model.predict(Xs_batch, verbose=0).reshape(-1)  # (H,)
        n_features = Xs.shape[1]

        # Dénormalisation — identique à WeatherForecaster :
        # on reconstruit (H, n_features) avec 0 partout sauf colonne 0 (close/tavg)
        y_full       = np.zeros((len(y_scaled), n_features))
        y_full[:, 0] = y_scaled
        y_denorm     = scaler.inverse_transform(y_full)[:, 0]

        max_h  = self.artifacts[asset].meta.get("horizon", 5)
        y_pred = y_denorm[:min(horizon, max_h)].tolist()

        return {
            "asset":       asset,
            "ticker":      self.artifacts[asset].meta.get("ticker", "^FCHI"),
            "horizon":     len(y_pred),
            "predictions": y_pred,
            "n_inputs":    int(Xs.shape[0]),
            "n_features":  int(Xs.shape[1]),
            "meta":        self.artifacts[asset].meta,
        }
