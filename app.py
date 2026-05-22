"""
app.py — Frontend Streamlit pour la prédiction de cours du CAC 40.
Communique avec le backend FastAPI via HTTP (appels REST).
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CAC 40 - Prediction LSTM",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="metric-container"] { background: #1e2130; border-radius: 8px; padding: 12px; }
.block-container { padding-top: 1.5rem; }
div.stButton > button {
    width: 100%; font-size: 1rem; font-weight: 600;
    padding: 0.55rem 1.5rem; border-radius: 8px;
    background: linear-gradient(90deg, #4C9BE8, #2d6fb5);
    color: white; border: none;
}
div.stButton > button:hover { opacity: 0.88; }
/* Espacement vertical autour de la zone boutons */
div[data-testid="column"] > div { padding: 0 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ── URL du backend ─────────────────────────────────────────────────────────────
# En local : http://localhost:8000
# En production : défini dans .streamlit/secrets.toml → API_URL = "https://..."
API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))


# ── Appels API ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def api_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def api_assets() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/assets", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_predict(horizon: int) -> dict | None:
    """Appel POST /predict — non mis en cache (résultat à la demande)."""
    try:
        r = requests.post(
            f"{API_URL}/predict",
            json={"asset": "cac40", "horizon": horizon},
            timeout=30,
        )
        if r.ok:
            return r.json()
        st.error(f"Erreur API ({r.status_code}) : {r.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Impossible de joindre le backend ({API_URL}). Vérifiez que le serveur FastAPI est démarré.")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue : {e}")
        return None


# ── Données historiques (yfinance directement côté Streamlit) ─────────────────
@st.cache_data(ttl=3600, show_spinner="Chargement des données CAC 40…")
def load_history() -> pd.DataFrame:
    try:
        raw = yf.download("^FCHI", period="1y", auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            raise ValueError("yfinance a retourné un DataFrame vide.")
        raw = raw.reset_index()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0].lower() for col in raw.columns]
        else:
            raw.columns = [str(c).lower() for c in raw.columns]
        raw["date"] = pd.to_datetime(raw["date"])
        df = raw[["date", "close"]].dropna().reset_index(drop=True)
        if df.empty:
            raise ValueError("Aucune donnée apres nettoyage.")
        return df
    except Exception as e:
        st.error(f"Impossible de charger les donnees CAC 40 : {e}")
        st.stop()


def next_trading_days(last: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    days, d = [], last
    while len(days) < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            days.append(d)
    return days


# ── Session state ─────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None

# ── Chargement données ────────────────────────────────────────────────────────
hist_df    = load_history()
last_date  = pd.Timestamp(hist_df["date"].iloc[-1])
last_close = float(hist_df["close"].iloc[-1])
prev_close = float(hist_df["close"].iloc[-2])
delta_pts  = last_close - prev_close
delta_pct  = delta_pts / prev_close * 100

# ── En-tête ───────────────────────────────────────────────────────────────────
st.title("CAC 40 — Prediction LSTM")
st.caption(f"Données : **{last_date.strftime('%d/%m/%Y')}** · Backend : `{API_URL}`")

# Statut du backend
health = api_health()
if health:
    st.success(f"Backend connecté · modèles chargés : {health.get('models_loaded', [])}")
else:
    st.warning(f"Backend non joignable ({API_URL}) — lancez `uvicorn src.api:app --reload`")

# ── KPIs cours actuel ─────────────────────────────────────────────────────────
assets_info = api_assets()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dernier cours",  f"{last_close:,.0f} pts",
          f"{delta_pts:+.0f} pts ({delta_pct:+.2f}%)")

if assets_info:
    m = assets_info["assets"]["cac40"]["metrics"]
    c2.metric("RMSE modèle", f"{m['rmse_global']:.0f} pts",
              "~{:.1f} % d'erreur".format(m['rmse_global'] / last_close * 100))
    c3.metric("MAE modèle",  f"{m['mae_global']:.0f} pts")
    c4.metric("Horizon max", f"{assets_info['assets']['cac40']['horizon']} jours ouvrés")

st.divider()

# ── Graphique historique ──────────────────────────────────────────────────────
n_hist = st.slider("Jours d'historique affichés", 30, 250, 90, 10)
hist   = hist_df.tail(n_hist)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist["date"], y=hist["close"],
    mode="lines", name="Cours réels CAC 40",
    line=dict(color="#4C9BE8", width=2),
))

# Overlay des prédictions si disponibles
if st.session_state.result:
    res        = st.session_state.result
    preds      = res["predictions"]
    pred_dates = next_trading_days(last_date, len(preds))
    rmse_steps = res["meta"]["metrics"]["rmse_per_step"]

    fig.add_trace(go.Scatter(
        x=[last_date, pred_dates[0]], y=[last_close, preds[0]],
        mode="lines", line=dict(color="#F4A261", width=1.5, dash="dot"),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=pred_dates, y=preds,
        mode="lines+markers", name="Prédictions LSTM",
        line=dict(color="#F4A261", width=2.5, dash="dash"),
        marker=dict(size=9, symbol="diamond"),
    ))
    upper = [p + r for p, r in zip(preds, rmse_steps)]
    lower = [p - r for p, r in zip(preds, rmse_steps)]
    fig.add_trace(go.Scatter(
        x=pred_dates + pred_dates[::-1],
        y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(244,162,97,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Intervalle ±RMSE",
    ))

fig.update_layout(
    height=430,
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    font=dict(color="#fafafa"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=True, gridcolor="#2a2d3e"),
    yaxis=dict(showgrid=True, gridcolor="#2a2d3e", title="Points CAC 40"),
    hovermode="x unified",
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Contrôles prédiction ──────────────────────────────────────────────────────
st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)
col_horizon, col_btn, col_reset = st.columns([2, 3, 1], vertical_alignment="bottom")

with col_horizon:
    horizon = st.selectbox(
        "Horizon de prédiction",
        options=[1, 2, 3, 4, 5],
        index=4,
        format_func=lambda x: f"J+{x} ({x} jour{'s' if x > 1 else ''} ouvre{'s' if x > 1 else ''})",
    )

with col_btn:
    predict_clicked = st.button("Predire les prochains jours ouvres")

with col_reset:
    if st.button("Effacer"):
        st.session_state.result = None
        st.rerun()

st.markdown("<div style='margin-bottom: 0.8rem;'></div>", unsafe_allow_html=True)

if predict_clicked:
    with st.spinner("Appel au backend FastAPI…"):
        result = api_predict(horizon)
    if result:
        st.session_state.result = result
        st.success(f"Prédiction reçue du backend ({API_URL}/predict) ✓")
        st.rerun()

# ── Résultats ─────────────────────────────────────────────────────────────────
if st.session_state.result:
    res        = st.session_state.result
    preds      = res["predictions"]
    pred_dates = next_trading_days(last_date, len(preds))
    rmse_steps = res["meta"]["metrics"]["rmse_per_step"][:len(preds)]
    mae_steps  = res["meta"]["metrics"]["mae_per_step"][:len(preds)]

    st.subheader("Resultats")

    pred_df = pd.DataFrame({
        "Horizon":          [f"J+{i+1}" for i in range(len(preds))],
        "Date":             [d.strftime("%A %d %b %Y") for d in pred_dates],
        "Cours prédit":     [f"{p:,.0f} pts" for p in preds],
        "Variation / J0":   [
            f"{p - last_close:+.0f} pts  ({(p-last_close)/last_close*100:+.2f}%)"
            for p in preds
        ],
        "Incertitude RMSE": [f"± {r:.0f} pts" for r in rmse_steps],
    })
    st.dataframe(pred_df, hide_index=True, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("RMSE par horizon")
        fig2 = go.Figure(go.Bar(
            x=[f"J+{i+1}" for i in range(len(rmse_steps))],
            y=rmse_steps,
            marker_color=["#4C9BE8", "#5aabf0", "#68baf8", "#90d0ff", "#b8e4ff"][:len(rmse_steps)],
            text=[f"{v:.0f}" for v in rmse_steps], textposition="outside",
        ))
        fig2.update_layout(
            height=280, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#fafafa"),
            yaxis=dict(title="pts", gridcolor="#2a2d3e"),
            xaxis=dict(gridcolor="#2a2d3e"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.subheader("Écart prédit vs cours actuel")
        deltas = [p - last_close for p in preds]
        fig3 = go.Figure(go.Bar(
            x=[f"J+{i+1}" for i in range(len(preds))],
            y=deltas,
            marker_color=["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas],
            text=[f"{d:+.0f} pts" for d in deltas], textposition="outside",
        ))
        fig3.update_layout(
            height=280, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#fafafa"),
            yaxis=dict(title="pts vs J0", gridcolor="#2a2d3e",
                       zeroline=True, zerolinecolor="#555"),
            xaxis=dict(gridcolor="#2a2d3e"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Détail de la réponse API brute (pour montrer l'intégration backend sur le CV)
    with st.expander("Reponse brute de l'API (JSON)"):
        st.json(res)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Backend")
    st.code(API_URL, language="bash")
    st.caption("Configurable via `.streamlit/secrets.toml` :\n`API_URL = \"https://...\"`")

    if assets_info:
        a = assets_info["assets"]["cac40"]
        st.header("Modele")
        st.markdown(f"""
| Paramètre | Valeur |
|---|---|
| Asset | **CAC 40** (^FCHI) |
| Fenêtre | **{a['window_size']} jours** |
| Horizon max | **{a['horizon']} jours** |
| RMSE | **{a['metrics']['rmse_global']:.0f} pts** |
| MAE | **{a['metrics']['mae_global']:.0f} pts** |
        """)

    st.header("Architecture")
    st.code("""LSTM(64, return_seq=True)
Dropout(0.2)
LSTM(32)
Dropout(0.2)
Dense(5)""", language="python")

    st.divider()
    if st.button("Rafraichir"):
        st.cache_data.clear()
        st.session_state.result = None
        st.rerun()
