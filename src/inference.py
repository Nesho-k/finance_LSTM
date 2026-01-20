# au lieu de faire les prédictions dans l'API, on les fait ici 
# pour éviter d'exposer les modèles et surcharger l'API

# src/inference.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import json
import numpy as np
import joblib
from tensorflow import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error only --> pour ne plus avoir les warnings TensorFlow

from src.features import build_feature_matrix


ROOT_DIR = Path(__file__).resolve().parent.parent  # meteo_app/
MODELS_DIR = ROOT_DIR / "models"  # meteo_app/models/

CITIES = ("berlin", "paris")


@dataclass
class CityArtifacts:
    model: object
    scaler: object
    feature_order_path: Path
    meta: dict


class WeatherForecaster:
    """
    Serveur d'inférence (côté Python). À appeler depuis l'API plus tard.
    Usage:
        f = WeatherForecaster(models_dir="models")
        y = f.predict("paris", horizon=7)
    """
    def __init__(self, models_dir: str | Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.artifacts: dict[str, CityArtifacts] = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        for city in CITIES:
            city_dir = self.models_dir / city
            try:
                model = keras.models.load_model(city_dir / "model.keras")
                scaler = joblib.load(city_dir / "scaler.pkl")
                feature_order_path = city_dir / "feature_order.json"
                meta_path = city_dir / "meta.json"
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            except Exception as e:
                raise RuntimeError(f"Impossible de charger les artefacts pour {city}: {e}")

            self.artifacts[city] = CityArtifacts(
                model=model,
                scaler=scaler,
                feature_order_path=feature_order_path,
                meta=meta,
            )

    def _build_X(self, city: str, window: int = 14) -> np.ndarray:
        a = self.artifacts[city]
        # récupère 36j (window+6), feature eng, coupe à 30, vérifs internes
        X = build_feature_matrix(
            city=city,
            feature_order_path=str(a.feature_order_path),
            window=window
        )  # -> (30, n_features), dtype float32
        # scaling identique à l'entraînement
        Xs = a.scaler.transform(X)
        if np.isnan(Xs).any():
            raise ValueError(f"NaN après scaling pour {city}. Vérifie tes features / scaler.")
        return Xs

    def predict(self, city: str, horizon: int = 7) -> dict:
        city = city.lower()
        if city not in self.artifacts:
            raise ValueError("city doit être 'paris' ou 'berlin'")

        # Récupère le window_size depuis meta.json (défaut: 30)
        window_size = self.artifacts[city].meta.get("window_size", 30)

        Xs = self._build_X(city, window=window_size)            # (window_size, F)
        Xs_batch = np.expand_dims(Xs, axis=0)                   # (1, window_size, F) - batch de 1 échantillon

        model = self.artifacts[city].model
        scaler = self.artifacts[city].scaler

        # Modèle Keras - prédiction directe
        y_scaled = model.predict(Xs_batch, verbose=0).reshape(-1)  # (H,) - valeurs normalisées

        # Dénormalisation : on prédit uniquement tavg (1ère colonne)
        # On crée un tableau (H, n_features) avec les prédictions dans la colonne tavg
        n_features = Xs.shape[1]
        y_full = np.zeros((len(y_scaled), n_features))
        y_full[:, 0] = y_scaled  # tavg est la 1ère colonne

        # Inverse transform pour récupérer les vraies températures
        y_denorm = scaler.inverse_transform(y_full)[:, 0]  # On garde seulement la colonne tavg

        y_pred = y_denorm[:horizon].tolist()
        out = {
            "city": city,
            "horizon": horizon,
            "predictions": y_pred,
            "n_inputs": int(Xs.shape[0]),
            "n_features": int(Xs.shape[1]),
            "meta": self.artifacts[city].meta,
        }
        return out
    


####### TEST #######

#if __name__ == "__main__":
    """import argparse, pprint
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", choices=CITIES, default="berlin")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--models_dir", type=str, default="models")
    args = parser.parse_args()"""

#    f = WeatherForecaster(models_dir=MODELS_DIR)
#    res = f.predict("berlin", horizon=7)
#    print(res)