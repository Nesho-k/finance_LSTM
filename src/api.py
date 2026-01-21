# src/api.py
"""
API FastAPI pour les prédictions météo LSTM et reconnaissance de chiffres (LeCun CNN)
Fournit des endpoints pour prédire la température à Paris et Berlin
et pour reconnaître des chiffres manuscrits
"""
from __future__ import annotations
from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
from datetime import datetime
import logging
from pathlib import Path
import io

import os

from meteo_app.src import lecun_model
from meteo_app.src.lecun_model import RN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error only --> pour ne plus avoir les warnings TensorFlow

#import torch
#from PIL import Image

#from src.inference import WeatherForecaster, MODELS_DIR, CITIES
#from src.lecun_model import RN

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialisation de l'API
app = FastAPI(
    title="Weather Forecast API",
    description="API de prédiction météo utilisant des modèles LSTM pour Paris et Berlin",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS pour permettre les requêtes depuis le frontend
# Récupérer les origines autorisées depuis les variables d'environnement
import os
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:5174"  # Dev par défaut
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Origines autorisées (frontend)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation du forecaster au démarrage
#forecaster: WeatherForecaster | None = None

# --- Chargement du modèle LeCun (reconnaissance de chiffres) ---
#LECUN_WEIGHTS_PATH = MODELS_DIR / "lecun" / "weights.pt"
#lecun_model: RN | None = None

#def load_lecun_model():
#    """Charge le modèle LeCun pour la reconnaissance de chiffres"""
#    global lecun_model
#    if not LECUN_WEIGHTS_PATH.exists():
#        logger.warning(f"Poids LeCun non trouvés: {LECUN_WEIGHTS_PATH}")
#        return False
#    lecun_model = RN()
#    lecun_model.load_state_dict(torch.load(LECUN_WEIGHTS_PATH, map_location="cpu", weights_only=True))
#    lecun_model.eval()
#    logger.info("Modèle LeCun chargé avec succès")
#    return True

#def preprocess_digit_image(image_bytes: bytes) -> torch.Tensor:
#    """
#    Prétraitement de l'image pour reconnaissance de chiffres:
#    - Conversion en niveaux de gris
#    - Redimensionnement en 28x28
#    - Normalisation entre -1 et 1
#    """
#    image = Image.open(io.BytesIO(image_bytes))
#    image = image.convert("L")
#    image = image.resize((28, 28), Image.Resampling.BOX)
#    tensor = torch.tensor(list(image.getdata()), dtype=torch.float32)
#    tensor = tensor.reshape(28, 28)
#    tensor = (tensor / 255.0) * 2 - 1
#    tensor = tensor.unsqueeze(0).unsqueeze(0)
#    return tensor


@app.on_event("startup")
async def startup_event():
    """Charge les modèles au démarrage de l'application"""
    global forecaster
    try:
        logger.info("Chargement des modèles météo...")
        forecaster = WeatherForecaster(models_dir=MODELS_DIR)
        logger.info(f"Modèles météo chargés avec succès pour les villes: {list(forecaster.artifacts.keys())}")

        # Charger le modèle LeCun
        logger.info("Chargement du modèle LeCun...")
        load_lecun_model()
    except Exception as e:
        logger.error(f"Erreur lors du chargement des modèles: {e}")
        raise


# --- Modèles Pydantic pour la validation ---

class PredictionRequest(BaseModel):
    """Modèle de requête pour la prédiction"""
    city: str = Field(..., description="Ville (paris ou berlin)")
    horizon: int = Field(7, ge=1, le=14, description="Horizon de prédiction (1-14 jours)")

    @field_validator('city')
    @classmethod
    def validate_city(cls, v: str) -> str:
        v = v.lower()
        if v not in CITIES:
            raise ValueError(f"City doit être parmi {CITIES}")
        return v


class PredictionResponse(BaseModel):
    """Modèle de réponse pour la prédiction"""
    city: str
    horizon: int
    predictions: List[float]
    n_inputs: int
    n_features: int
    meta: Dict[str, Any]
    timestamp: str


class HealthResponse(BaseModel):
    """Modèle de réponse pour le health check"""
    status: str
    timestamp: str
    models_loaded: List[str]
    version: str


# --- Endpoints ---

@app.get("/", tags=["Root"])
async def root():
    """Endpoint racine avec informations de l'API"""
    return {
        "name": "Weather Forecast API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "cities": "/cities",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Vérifie l'état de santé de l'API et des modèles chargés
    """
    if forecaster is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèles non chargés"
        )

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=list(forecaster.artifacts.keys()),
        version="1.0.0"
    )


@app.get("/cities", tags=["Info"])
async def get_cities():
    """
    Retourne la liste des villes disponibles avec leurs métadonnées
    """
    if forecaster is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèles non chargés"
        )

    cities_info = {}
    for city, artifacts in forecaster.artifacts.items():
        cities_info[city] = {
            "name": city.capitalize(),
            "meta": artifacts.meta,
            "features_count": artifacts.meta.get("n_features", "unknown"),
            "window_size": artifacts.meta.get("window_size", "unknown"),
            "max_horizon": artifacts.meta.get("horizon", 7)
        }

    return {
        "cities": cities_info,
        "total": len(cities_info)
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Effectue une prédiction météo pour une ville donnée

    - **city**: Ville pour laquelle faire la prédiction (paris ou berlin)
    - **horizon**: Nombre de jours à prédire (1-14, par défaut 7)

    Retourne les températures moyennes prédites pour les N prochains jours
    """
    if forecaster is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèles non chargés"
        )

    try:
        logger.info(f"Prédiction demandée pour {request.city}, horizon={request.horizon}")

        # Effectue la prédiction
        result = forecaster.predict(city=request.city, horizon=request.horizon)

        # Ajoute le timestamp
        result["timestamp"] = datetime.utcnow().isoformat()

        logger.info(f"Prédiction réussie pour {request.city}")
        return PredictionResponse(**result)

    except ValueError as e:
        logger.error(f"Erreur de validation: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne: {str(e)}"
        )


@app.get("/predict/{city}", response_model=PredictionResponse, tags=["Prediction"])
async def predict_get(city: str, horizon: int = 7):
    """
    Endpoint GET alternatif pour les prédictions (pour faciliter les tests)

    - **city**: Ville (paris ou berlin)
    - **horizon**: Nombre de jours (1-14, par défaut 7)
    """
    request = PredictionRequest(city=city, horizon=horizon)
    return await predict(request)


# --- Endpoints LeCun (reconnaissance de chiffres) ---

#@app.post("/lecun/predict", tags=["LeCun"])
#async def lecun_predict(file: UploadFile = File(...)):
    """
    Reconnaissance de chiffres manuscrits (0-9).
    Reçoit une image et retourne la prédiction + scores.
    """
    if lecun_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle LeCun non chargé"
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")

    try:
        image_bytes = await file.read()
        tensor = preprocess_digit_image(image_bytes)

        with torch.no_grad():
            logits = lecun_model(tensor)

        scores = logits[0].tolist()
        prediction = int(torch.argmax(logits, dim=1).item())

        return {"prediction": prediction, "scores": scores}

    except Exception as e:
        logger.error(f"Erreur LeCun: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


#@app.get("/lecun/health", tags=["LeCun"])
#async def lecun_health():
    """Vérifie que le modèle LeCun est chargé."""
    return {
        "status": "ok" if lecun_model is not None else "not_loaded",
        "model": "LeCun CNN (MNIST)"
    }


# --- Point d'entrée pour exécution directe ---

if __name__ == "__main__":
    import uvicorn

    # Configuration du serveur
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload en développement
        log_level="info"
    )
