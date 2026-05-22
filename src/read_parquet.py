# Utilitaire de debug — lit le cache local des données CAC 40
# [MODIFIÉ] Adapté pour le cache yfinance (daily_cac40.parquet)

import pandas as pd

df = pd.read_parquet("cache/yfinance/daily_cac40.parquet")

print(df.info())
print(df.head())
print(df.tail())
print("Nombre de jours :", len(df))
print("Dates couvertes :", df["date"].min(), "→", df["date"].max())
