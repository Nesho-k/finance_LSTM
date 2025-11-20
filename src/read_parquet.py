import pandas as pd

df = pd.read_parquet("cache_bis/meteostat/daily_berlin.parquet")

print(df.info())
print(df.head())
print(df.tail())
print("Nombre de jours :", len(df))
print("Dates couvertes :", df['time'].min(), "→", df['time'].max())