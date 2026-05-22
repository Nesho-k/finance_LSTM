"""
Script d'entraînement du modèle LSTM pour la prédiction du CAC 40.

Modifié depuis meteo_LSTM : remplacement des données météo (Meteostat)
par des données financières (yfinance / CAC 40 ^FCHI).
Architecture LSTM identique à l'original : LSTM → Dropout → LSTM → Dropout → Dense.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import yfinance as yf

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Paramètres ───────────────────────────────────────────────────────────────
# [MODIFIÉ] Paramètres adaptés aux séries financières (ticker + fenêtres plus longues)
TICKER      = "^FCHI"   # CAC 40
PERIOD      = "5y"      # 5 ans de données historiques
WINDOW_SIZE = 60        # jours d'historique en entrée (vs 14 météo → 60 finance)
HORIZON     = 5         # jours de cours à prédire (J+1 … J+5)
TEST_RATIO  = 0.15
VAL_RATIO   = 0.10
EPOCHS      = 100
BATCH_SIZE  = 32

MODEL_DIR = Path("models/cac40")
PLOTS_DIR = Path("plots")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Téléchargement des données ─────────────────────────────────────────────
# [MODIFIÉ] yfinance remplace Meteostat
print("[DATA] Téléchargement des données CAC 40 via yfinance...")
raw = yf.download(TICKER, period=PERIOD, auto_adjust=True, progress=False)

if raw.empty:
    raise RuntimeError("yfinance n'a renvoyé aucune donnée. Vérifiez la connexion / le ticker.")

raw = raw.reset_index()

# Aplatir les colonnes MultiIndex que yfinance peut retourner
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = [col[0].lower() for col in raw.columns]
else:
    raw.columns = [str(c).lower() for c in raw.columns]

df = raw[["date", "open", "high", "low", "close", "volume"]].dropna()
df = df.sort_values("date").reset_index(drop=True)
print(f"[DATA] {len(df)} jours ({df['date'].min().date()} → {df['date'].max().date()})")


# ── 2. Feature engineering ────────────────────────────────────────────────────
# [MODIFIÉ] Indicateurs techniques financiers remplacent les features météo
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    return 100 - 100 / (1 + rs)


print("[FEATURES] Calcul des indicateurs techniques...")
df["returns"]      = df["close"].pct_change()
df["log_returns"]  = np.log(df["close"] / df["close"].shift(1))
df["volatility_5"] = df["returns"].rolling(5).std()

# Moyennes mobiles normalisées par le prix courant (scale-invariant)
df["ma5"]  = df["close"].rolling(5).mean()  / df["close"]
df["ma20"] = df["close"].rolling(20).mean() / df["close"]
df["ma60"] = df["close"].rolling(60).mean() / df["close"]

df["rsi"] = compute_rsi(df["close"], window=14)

# Volume normalisé par z-score rolling
df["volume_norm"] = (
    (df["volume"] - df["volume"].rolling(20).mean())
    / (df["volume"].rolling(20).std() + 1e-9)
)

# Amplitude Haut-Bas relative
df["hl_range"] = (df["high"] - df["low"]) / df["close"]

df = df.dropna().reset_index(drop=True)
print(f"[FEATURES] {len(df)} lignes après suppression des NaN")

# ── 3. Sélection des features ─────────────────────────────────────────────────
# [MODIFIÉ] "close" est la cible (colonne 0) — même convention que "tavg" dans météo
FEATURE_COLS = [
    "close",        # target (colonne 0, identique à tavg dans l'original)
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

with open(MODEL_DIR / "feature_order.json", "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)
print(f"[FEATURES] {len(FEATURE_COLS)} features : {FEATURE_COLS}")


# ── 4. Normalisation (fit sur train uniquement — pas de data leakage) ─────────
X_raw   = df[FEATURE_COLS].to_numpy(dtype=np.float32)
n_total = len(X_raw)
n_test  = int(n_total * TEST_RATIO)
n_val   = int(n_total * VAL_RATIO)
n_train = n_total - n_test - n_val

scaler   = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit(X_raw[:n_train]).transform(X_raw)

joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
print(f"[SCALE] Scaler ajusté sur {n_train} jours (train set uniquement)")


# ── 5. Fenêtres glissantes ────────────────────────────────────────────────────
# [MODIFIÉ] Même logique de sliding window que l'original météo
def create_sequences(data: np.ndarray, window: int, horizon: int):
    X, y = [], []
    for i in range(len(data) - window - horizon + 1):
        X.append(data[i : i + window])
        y.append(data[i + window : i + window + horizon, 0])  # close = colonne 0
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


X_seq, y_seq = create_sequences(X_scaled, WINDOW_SIZE, HORIZON)
print(f"[SEQ] X_seq: {X_seq.shape}  y_seq: {y_seq.shape}")

n_samples   = len(X_seq)
n_test_seq  = int(n_samples * TEST_RATIO)
n_val_seq   = int(n_samples * VAL_RATIO)
n_train_seq = n_samples - n_test_seq - n_val_seq

X_train, y_train = X_seq[:n_train_seq], y_seq[:n_train_seq]
X_val,   y_val   = X_seq[n_train_seq : n_train_seq + n_val_seq], y_seq[n_train_seq : n_train_seq + n_val_seq]
X_test,  y_test  = X_seq[-n_test_seq:], y_seq[-n_test_seq:]

print(f"[SPLIT] Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")


# ── 6. Architecture LSTM ──────────────────────────────────────────────────────
# Architecture IDENTIQUE à l'original : LSTM → Dropout → LSTM → Dropout → Dense
# Seul l'input_shape change (window_size=60, n_features=10 vs 14/12 météo)
n_features = X_train.shape[2]

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, n_features)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(HORIZON),
], name="cac40_lstm")

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()


# ── 7. Entraînement ───────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
]

print("\n[TRAIN] Démarrage de l'entraînement...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)


# ── 8. Évaluation sur le test set ─────────────────────────────────────────────
# [MODIFIÉ] Dénormalisation identique au modèle météo (colonne 0 = cible)
def denormalize(y_scaled_arr: np.ndarray, sc: MinMaxScaler, n_feat: int) -> np.ndarray:
    """Reconstruit un tableau (N*H, n_feat) avec 0 sauf colonne 0, puis inverse_transform."""
    n, h   = y_scaled_arr.shape
    buf    = np.zeros((n * h, n_feat), dtype=np.float32)
    buf[:, 0] = y_scaled_arr.reshape(-1)
    return sc.inverse_transform(buf)[:, 0].reshape(n, h)


y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = denormalize(y_pred_scaled, scaler, n_features)
y_true = denormalize(y_test,        scaler, n_features)

rmse_per_step = [float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))) for i in range(HORIZON)]
mae_per_step  = [float(mean_absolute_error(y_true[:, i],  y_pred[:, i])) for i in range(HORIZON)]
rmse_global   = float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel())))
mae_global    = float(mean_absolute_error(y_true.ravel(), y_pred.ravel()))

print(f"\n[METRICS] RMSE global : {rmse_global:.2f} pts")
print(f"[METRICS] MAE global  : {mae_global:.2f} pts")
print(f"[METRICS] RMSE / step : {[round(v, 2) for v in rmse_per_step]}")
print(f"[METRICS] MAE  / step : {[round(v, 2) for v in mae_per_step]}")


# ── 9. Sauvegarde des artefacts ───────────────────────────────────────────────
model.save(MODEL_DIR / "model.keras")

train_end_idx = n_train + n_val - 1
meta = {
    "version":        "2.0.0",
    "ticker":         TICKER,
    "asset":          "CAC 40",
    "train_end_date": str(df["date"].iloc[min(train_end_idx, len(df) - 1)].date()),
    "window_size":    WINDOW_SIZE,
    "horizon":        HORIZON,
    "n_features":     n_features,
    "metrics": {
        "rmse_global":   rmse_global,
        "mae_global":    mae_global,
        "rmse_per_step": rmse_per_step,
        "mae_per_step":  mae_per_step,
    },
    "framework": {
        "tensorflow": tf.__version__,
        "python":     "3.x",
    },
}

with open(MODEL_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n[SAVE] Modèle  → {MODEL_DIR}/model.keras")
print(f"[SAVE] Scaler  → {MODEL_DIR}/scaler.pkl")
print(f"[SAVE] Meta    → {MODEL_DIR}/meta.json")
print(f"[SAVE] Features→ {MODEL_DIR}/feature_order.json")


# ── 10. Visualisations ────────────────────────────────────────────────────────
# [MODIFIÉ] Graphiques financiers remplacent les graphiques de température

# Courbe de loss
plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"],     label="Train loss (MSE)")
plt.plot(history.history["val_loss"], label="Val loss (MSE)")
plt.xlabel("Époque")
plt.ylabel("MSE")
plt.title("Courbe de loss — LSTM CAC 40")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "loss_curve.png", dpi=150)
plt.close()
print(f"\n[PLOT] Courbe de loss → {PLOTS_DIR}/loss_curve.png")

# Cours réels vs prédictions (horizon J+1)
n_display = min(250, len(y_true))
plt.figure(figsize=(14, 5))
plt.plot(y_true[-n_display:, 0], label="Cours réels CAC 40 (J+1)",   color="steelblue", linewidth=1.5)
plt.plot(y_pred[-n_display:, 0], label="Prédictions LSTM (J+1)",      color="tomato",    linewidth=1.2, linestyle="--")
plt.xlabel("Jours (test set)")
plt.ylabel("Points CAC 40")
plt.title(f"CAC 40 — Cours réels vs Prédictions LSTM  |  RMSE={rmse_per_step[0]:.1f} pts")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "predictions_vs_real.png", dpi=150)
plt.close()
print(f"[PLOT] Prédictions vs réels → {PLOTS_DIR}/predictions_vs_real.png")

# RMSE par horizon
plt.figure(figsize=(7, 4))
plt.bar([f"J+{i+1}" for i in range(HORIZON)], rmse_per_step, color="steelblue")
plt.xlabel("Horizon de prédiction")
plt.ylabel("RMSE (points CAC 40)")
plt.title("RMSE par horizon — LSTM CAC 40")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "rmse_per_horizon.png", dpi=150)
plt.close()
print(f"[PLOT] RMSE par horizon → {PLOTS_DIR}/rmse_per_horizon.png")

# ── Résumé final ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ENTRAÎNEMENT TERMINÉ")
print(f"  Asset      : CAC 40 (^FCHI)")
print(f"  Période    : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  RMSE global: {rmse_global:.2f} pts")
print(f"  MAE global : {mae_global:.2f} pts")
print("=" * 60)
