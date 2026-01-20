# src/features.py
from __future__ import annotations
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

from meteostat import Point, Daily  
import os
# Désactiver le cache pickle de Meteostat (on utilise notre propre cache parquet)
os.environ['METEOSTAT_CACHE_DIR'] = ''

# Config villes
CITY_CFG = {
    "paris":  {"lat": 48.8566, "lon": 2.3522},
    "berlin": {"lat": 52.5200, "lon": 13.4050},
}

# Dossier cache local
CACHE_DIR = Path("cache/meteostat")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_path(city: str) -> Path:
    return CACHE_DIR / f"daily_{city.lower()}.parquet"

def _now_utc():
    return datetime.now(timezone.utc)

def fetch_daily_history(city: str, days: int = 30) -> pd.DataFrame:
    """
    Récupère les 'days' derniers jours (J-29..J) via Meteostat.
    Si un cache récent (< 1 jour) existe, l'utilise directement.
    Sinon, refetch complet et écrase le cache.
    """
    city = city.lower()
    if city not in CITY_CFG:
        raise ValueError("city must be 'paris' or 'berlin'")

    # Fenêtre voulue
    end = _now_utc().date()
    start = end - timedelta(days=days - 1)

    cache_fp = _cache_path(city)
    use_cache = False
    df_cache = None

    # --- Vérifie si un cache valide existe ---
    if cache_fp.exists():
        mtime = datetime.fromtimestamp(cache_fp.stat().st_mtime, timezone.utc)
        age_hours = (_now_utc() - mtime).total_seconds() / 3600
        if age_hours < 24:  # cache de moins de 24h → on le réutilise
            try:
                df_cache = pd.read_parquet(cache_fp)
                df_cache["time"] = pd.to_datetime(df_cache["time"]).dt.tz_localize(None)
                use_cache = True
                print(f"[CACHE ✅] {city.capitalize()} — cache récent utilisé ({age_hours:.1f}h).")
            except Exception as e:
                print(f"[WARN] Cache illisible pour {city}: {e}. On refait un fetch.")
                use_cache = False

    if use_cache and df_cache is not None and not df_cache.empty:
        # Vérifie que le cache couvre bien les dates demandées
        dmin, dmax = df_cache["time"].dt.date.min(), df_cache["time"].dt.date.max()
        if dmin <= start and dmax >= end:
            df_window = df_cache[
                (df_cache["time"].dt.date >= start) & (df_cache["time"].dt.date <= end)
            ].copy().reset_index(drop=True)
            print(f"[OK ✅] {city.capitalize()} — données chargées depuis le cache ({len(df_window)} jours).")
            return df_window

    # --- Sinon : appel API complet ---
    lat, lon = CITY_CFG[city]["lat"], CITY_CFG[city]["lon"]
    loc = Point(lat, lon)
    print(f"[API 🔵] {city.capitalize()} — fetch complet depuis Meteostat ({start} → {end}).")

    attempts, backoff = 3, 1.5
    fetched, last_err = None, None

    for i in range(attempts):
        try:
            ms = datetime(start.year, start.month, start.day)
            me = datetime(end.year, end.month, end.day)
            print(f"[API DEMANDE 📤] Période demandée à Meteostat: {ms.date()} → {me.date()}")
            print(f"[API DEMANDE 📤] Nombre de jours demandés: {(me.date() - ms.date()).days + 1}")
            df_new = Daily(loc, ms, me).fetch()
            if not df_new.empty:
                df_new = df_new.reset_index().rename(columns={"time": "time"})
                df_new["time"] = pd.to_datetime(df_new["time"]).dt.tz_localize(None)
                print(f"[API REÇU 📥] Meteostat a renvoyé {len(df_new)} lignes")
                print(f"[API REÇU 📥] Première date: {df_new['time'].dt.date.min()}")
                print(f"[API REÇU 📥] Dernière date: {df_new['time'].dt.date.max()}")
                print(f"[API REÇU 📥] Dates complètes:\n{df_new['time'].dt.date.tolist()}")
            fetched = df_new
            break
        except Exception as e:
            last_err = e
            print(f"[WARN] Tentative {i+1}/{attempts} échouée pour {city}: {e}")
            time.sleep(backoff**i)

    if fetched is None or fetched.empty:
        if df_cache is not None and not df_cache.empty:
            print(f"[FALLBACK 🔴] {city.capitalize()} — API indisponible, utilisation du cache existant.")
            return df_cache
        raise RuntimeError(f"[ERROR] Echec Meteostat pour {city}: {last_err}")

    # Nettoyage + déduplication
    fetched = (
        fetched.sort_values("time")
               .drop_duplicates(subset=["time"], keep="last")
               .reset_index(drop=True)
    )

    # Debug : afficher ce qui a été reçu
    print(f"[DEBUG 🔍] Dates reçues de l'API: {fetched['time'].dt.date.min()} → {fetched['time'].dt.date.max()}")
    print(f"[DEBUG 🔍] Nombre de lignes reçues: {len(fetched)} (attendu: {(end - start).days + 1})")

    # Sauvegarde du nouveau cache
    try:
        fetched.to_parquet(cache_fp, index=False)
        print(f"[CACHE 💾] {city.capitalize()} — cache mis à jour ({len(fetched)} lignes).")
        print(f"[CACHE ENREGISTRÉ 💾] Première date: {fetched['time'].dt.date.min()}")
        print(f"[CACHE ENREGISTRÉ 💾] Dernière date: {fetched['time'].dt.date.max()}")
        print(f"[CACHE ENREGISTRÉ 💾] Dates complètes:\n{fetched['time'].dt.date.tolist()}")
    except Exception as e:
        print(f"[WARN] Impossible de sauvegarder le cache pour {city}: {e}")

    return fetched


###### TEST #########
#if __name__ == "__main__":
#    df = fetch_daily_history("berlin", days=30)
#    print(f"=== BERLIN ===")
#    print(df.tail(5))