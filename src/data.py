# src/data.py — Téléchargement des cours du CAC 40 via yfinance avec cache local parquet
# [MODIFIÉ] Remplace Meteostat (météo Paris/Berlin) par yfinance (CAC 40 ^FCHI)

from __future__ import annotations

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timezone

TICKER    = "^FCHI"
CACHE_DIR = Path("cache/yfinance")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path() -> Path:
    return CACHE_DIR / "daily_cac40.parquet"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def fetch_daily_history(ticker: str = TICKER, days: int = 200) -> pd.DataFrame:
    """
    Retourne les 'days' derniers jours de cours OHLCV du CAC 40.
    Utilise un cache parquet local (valide 12h) pour limiter les appels yfinance.
    """
    cache_fp = _cache_path()

    # --- Tentative de lecture du cache ---
    if cache_fp.exists():
        mtime      = datetime.fromtimestamp(cache_fp.stat().st_mtime, timezone.utc)
        age_hours  = (_now_utc() - mtime).total_seconds() / 3600
        if age_hours < 12:
            try:
                df_cache = pd.read_parquet(cache_fp)
                df_cache["date"] = pd.to_datetime(df_cache["date"])
                if len(df_cache) >= days:
                    print(f"[CACHE ✅] CAC 40 — cache récent utilisé ({age_hours:.1f}h, {len(df_cache)} lignes).")
                    return df_cache.tail(days).reset_index(drop=True)
            except Exception as e:
                print(f"[WARN] Cache illisible : {e}. Refetch complet.")

    # --- Appel yfinance ---
    print(f"[API 🔵] Téléchargement CAC 40 via yfinance (period=2y)...")
    raw = yf.download(TICKER, period="2y", auto_adjust=True, progress=False)

    if raw is None or raw.empty:
        raise RuntimeError("[ERROR] yfinance n'a renvoyé aucune donnée pour ^FCHI.")

    raw = raw.reset_index()

    # Aplatir les colonnes MultiIndex (yfinance peut retourner MultiIndex)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [str(c).lower() for c in raw.columns]

    df = raw[["date", "open", "high", "low", "close", "volume"]].dropna()
    df = (
        df.sort_values("date")
          .drop_duplicates(subset=["date"])
          .reset_index(drop=True)
    )
    df["date"] = pd.to_datetime(df["date"])

    # Sauvegarde du cache
    try:
        df.to_parquet(cache_fp, index=False)
        print(f"[CACHE 💾] Cache mis à jour — {len(df)} lignes "
              f"({df['date'].min().date()} → {df['date'].max().date()}).")
    except Exception as e:
        print(f"[WARN] Impossible de sauvegarder le cache : {e}")

    return df.tail(days).reset_index(drop=True)
