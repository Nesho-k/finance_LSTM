# src/api.py — API FastAPI pour la prédiction de cours du CAC 40 via LSTM
# [MODIFIÉ] Contexte météo → finance ; WeatherForecaster → FinancialForecaster
# Structure des endpoints conservée à l'identique.

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from src.inference import ASSETS, MODELS_DIR, FinancialForecaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# [MODIFIÉ] Titre et description reflètent le contexte financier
app = FastAPI(
    title="Financial Forecast API — CAC 40 LSTM",
    description="API de prédiction de cours financiers (CAC 40) via un modèle LSTM entraîné sur 5 ans de données.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:5174",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

forecaster: FinancialForecaster | None = None


@app.on_event("startup")
async def startup_event():
    global forecaster
    try:
        logger.info("Chargement du modèle LSTM CAC 40...")
        forecaster = FinancialForecaster(models_dir=MODELS_DIR)
        logger.info(f"Modèle chargé pour les actifs : {list(forecaster.artifacts.keys())}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {e}")
        raise


# --- Modèles Pydantic ---

class PredictionRequest(BaseModel):
    asset:   str = Field("cac40", description="Actif financier (cac40)")
    horizon: int = Field(5, ge=1, le=5, description="Horizon de prédiction en jours ouvrés (1–5)")

    @field_validator("asset")
    @classmethod
    def validate_asset(cls, v: str) -> str:
        v = v.lower()
        if v not in ASSETS:
            raise ValueError(f"Asset doit être parmi {ASSETS}")
        return v


class PredictionResponse(BaseModel):
    asset:       str
    ticker:      str
    horizon:     int
    predictions: List[float]
    n_inputs:    int
    n_features:  int
    meta:        Dict[str, Any]
    timestamp:   str


class HealthResponse(BaseModel):
    status:        str
    timestamp:     str
    models_loaded: List[str]
    version:       str


# --- Endpoints ---

@app.get("/", tags=["Root"])
async def root():
    return {
        "name":    "Financial Forecast API — CAC 40 LSTM",
        "version": "2.0.0",
        "status":  "running",
        "endpoints": {
            "health":  "/health",
            "predict": "/predict",
            "assets":  "/assets",
            "docs":    "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    if forecaster is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Modèle non chargé")
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=list(forecaster.artifacts.keys()),
        version="2.0.0",
    )


@app.get("/assets", tags=["Info"])
async def get_assets():
    """Retourne les actifs disponibles avec leurs métadonnées de performance."""
    if forecaster is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Modèle non chargé")
    return {
        "assets": {
            asset: {
                "name":        artifacts.meta.get("asset", asset.upper()),
                "ticker":      artifacts.meta.get("ticker", "^FCHI"),
                "window_size": artifacts.meta.get("window_size", 60),
                "horizon":     artifacts.meta.get("horizon", 5),
                "metrics":     artifacts.meta.get("metrics", {}),
            }
            for asset, artifacts in forecaster.artifacts.items()
        }
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Prédit les cours du CAC 40 pour les N prochains jours ouvrés.

    - **asset** : actif financier (cac40)
    - **horizon** : nombre de jours à prédire (1–5, défaut 5)
    """
    if forecaster is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Modèle non chargé")
    try:
        logger.info(f"Prédiction demandée — asset={request.asset}, horizon={request.horizon}")
        result = forecaster.predict(asset=request.asset, horizon=request.horizon)
        result["timestamp"] = datetime.utcnow().isoformat()
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur prédiction : {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erreur interne : {str(e)}")


@app.get("/predict/{asset}", response_model=PredictionResponse, tags=["Prediction"])
async def predict_get(asset: str, horizon: int = 5):
    """Endpoint GET alternatif (facilite les tests rapides)."""
    request = PredictionRequest(asset=asset, horizon=horizon)
    return await predict(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
