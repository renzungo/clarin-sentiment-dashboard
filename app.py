import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Clarin Sentiment Analysis", layout="wide")

LANGS = {"es": "Español", "en": "English"}
STRINGS = {
    "es": {
        "title": "Clarin - Análisis de Sentimiento",
        "subtitle": "Explora la evolución del sentimiento y las secciones del diario",
        "language": "Idioma",
        "date": "Fecha",
        "section": "Sección",
        "kpi_notes": "Notas",
        "kpi_mean": "Sentimiento medio",
        "kpi_pos": "% positivas",
        "time_series": "Tendencia diaria del sentimiento",
        "dist": "Distribución de puntajes",
        "recent": "Titulares recientes",
        "download_csv": "Descargar CSV filtrado",
        "no_data": "Sin datos para los filtros seleccionados.",
    },
    "en": {
        "title": "Clarin - Sentiment Analysis",
        "subtitle": "Explore sentiment trends and newspaper sections",
        "language": "Language",
        "date": "Date",
        "section": "Section",
        "kpi_notes": "Articles",
        "kpi_mean": "Avg sentiment",
        "kpi_pos": "% positive",
        "time_series": "Daily sentiment trend",
        "dist": "Score distribution",
        "recent": "Recent headlines",
        "download_csv": "Download filtered CSV",
        "no_data": "No data for selected filters.",
    },
}
def t(key: str) -> str:
    lang = st.session_state.get("lang", "es")
    return STRINGS.get(lang, STRINGS["es"]).get(key, key)

if "lang" not in st.session_state:
    st.session_state.lang = "es"
with st.sidebar:
    st.selectbox(STRINGS[st.session_state.lang]["language"], options=list(LANGS.keys()),
                 format_func=lambda k: LANGS[k], key="lang")

@st.cache_data(ttl=300)
def load_data():
    data_url = os.getenv("DATA_URL", "").strip()
    if data_url:
        df = pd.read_csv(data_url)
    else:
        df = pd.read_csv("data/clarin_sentiment.csv")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

df = load_data()
st.title(t("title"))
st.caption(t("subtitle"))
if df.empty:
    st.info(t("no_data"))
    st.stop()

min_d = pd.to_datetime(df["date"]).min()
max_d = pd.to_datetime(df["date"]).max()
with st.sidebar:
    date_range = st.date_input(t("date"), (min_d.date(), max_d.date()))
    sections = sorted(df["section"].dropna().unique().tolist() if "section" in df.columns else [])
    selected_sections = st.multiselect(t("section"), sections, default=sections[:5])

mask = df["date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
if selected_sections:
    mask &= df["section"].isin(selected_sections)
dff = df.loc[mask].copy()
if dff.empty:
    st.info(t("no_data"))
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric(t("kpi_notes"), int(len(dff)))
c2.metric(t("kpi_mean"), round(dff["sentiment_score"].mean(), 3))
pos_ratio = (dff["sentiment_label"].astype(str).str.lower().eq("positive").mean()) * 100
c3.metric(t("kpi_pos"), f"{pos_ratio:.1f}%")

ts = dff.set_index("date").resample("D")["sentiment_score"].mean().reset_index()
st.plotly_chart(px.line(ts, x="date", y="sentiment_score", title=t("time_series")), use_container_width=True)
st.plotly_chart(px.histogram(dff, x="sentiment_score", nbins=40, title=t("dist")), use_container_width=True)

show_cols = [c for c in ["date","section","title","sentiment_label","sentiment_score","url"] if c in dff.columns]
st.subheader(t("recent"))
st.dataframe(dff.sort_values("date", ascending=False)[show_cols].head(300), use_container_width=True)

st.download_button(t("download_csv"), data=dff.to_csv(index=False).encode("utf-8"),
                   file_name="clarin_sentiment_filtered.csv", mime="text/csv")
