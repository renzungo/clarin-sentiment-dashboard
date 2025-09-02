import os
import pandas as pd
import streamlit as st
import plotly.express as px
st.write("DATA_URL:", os.getenv("DATA_URL"))
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

# Sidebar language toggle
if "lang" not in st.session_state:
    st.session_state.lang = "es"
with st.sidebar:
    st.selectbox(
        STRINGS[st.session_state.lang]["language"],
        options=list(LANGS.keys()),
        format_func=lambda k: LANGS[k],
        key="lang"
    )

REQUIRED_COLS = {"date", "sentiment_score"}
OPTIONAL_COLS = {"section", "title", "sentiment_label", "url"}

@st.cache_data(ttl=600, show_spinner=True)
import os, io, csv
import pandas as pd
import streamlit as st
import requests

REQUIRED_COLS = {"date", "sentiment_score"}

@st.cache_data(ttl=600, show_spinner=True)
def load_data() -> pd.DataFrame:
    data_url = os.getenv("DATA_URL", "").strip()
    local_fallback = "data/clarin_sentiment.csv"

    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip().lower() for c in df.columns]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "sentiment_score" in df.columns:
            df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
        if "sentiment_label" in df.columns:
            df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()
        df = df.dropna(subset=["date"]).sort_values("date")
        return df

    def read_from_bytes(b: bytes) -> pd.DataFrame:
        # 1) Excel?
        if b[:4] == b"PK\x03\x04":  # XLSX is a zip
            return normalize(pd.read_excel(io.BytesIO(b), engine="openpyxl"))

        # 2) Try python engine with auto sep sniffing (handles ; , \t)
        try:
            return normalize(pd.read_csv(
                io.BytesIO(b),
                engine="python",
                sep=None,              # auto-detect
                encoding="utf-8-sig",  # handle BOM
                on_bad_lines="skip"    # skip malformed lines
            ))
        except Exception:
            pass

        # 3) Fallbacks: explicit separators
        for sep in [",", ";", "\t", "|"]:
            try:
                return normalize(pd.read_csv(
                    io.BytesIO(b),
                    engine="python",
                    sep=sep,
                    encoding="utf-8-sig",
                    on_bad_lines="skip"
                ))
            except Exception:
                continue

        raise ValueError("Could not parse the file. Check delimiter/encoding.")

    def fetch_bytes(url: str) -> bytes:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return r.content

    # Case A: HTTP(S) URL provided
    if data_url.lower().startswith(("http://", "https://")):
        b = fetch_bytes(data_url)
        return read_from_bytes(b)

    # Case B: Local path fallback for dev
    if os.path.exists(local_fallback):
        return normalize(pd.read_csv(local_fallback, encoding="utf-8-sig"))

    # Nothing worked
    raise FileNotFoundError(f"DATA_URL not set to a valid URL and no local fallback at {local_fallback}")

df = load_data()

# Basic schema check
missing = REQUIRED_COLS - set(df.columns)
if missing:
    st.error(f"Missing required column(s): {', '.join(sorted(missing))}")
    st.stop()

st.title(t("title"))
st.caption(t("subtitle"))

if df.empty:
    st.info(t("no_data"))
    st.stop()

# Sidebar filters
min_d = pd.to_datetime(df["date"]).min()
max_d = pd.to_datetime(df["date"]).max()
# Ensure safe defaults if min/max are NaT
if pd.isna(min_d) or pd.isna(max_d):
    st.info(t("no_data"))
    st.stop()

with st.sidebar:
    # Streamlit returns a single date when min==max; normalize to a tuple
    date_sel = st.date_input(
        t("date"),
        value=(min_d.date(), max_d.date()),
        min_value=min_d.date(),
        max_value=max_d.date()
    )
    if isinstance(date_sel, tuple):
        start_d, end_d = date_sel
    else:
        start_d = end_d = date_sel

    sections = sorted(df["section"].dropna().unique().tolist()) if "section" in df.columns else []
    default_sections = sections[:5] if sections else []
    selected_sections = st.multiselect(t("section"), sections, default=default_sections)

# Row filtering
mask = df["date"].between(pd.to_datetime(start_d), pd.to_datetime(end_d))
if selected_sections and "section" in df.columns:
    mask &= df["section"].isin(selected_sections)
dff = df.loc[mask].copy()

if dff.empty:
    st.info(t("no_data"))
    st.stop()

# KPIs
c1, c2, c3 = st.columns(3)
c1.metric(t("kpi_notes"), int(len(dff)))
mean_sent = dff["sentiment_score"].mean()
c2.metric(t("kpi_mean"), f"{mean_sent:.3f}" if pd.notna(mean_sent) else "—")

pos_ratio = None
if "sentiment_label" in dff.columns:
    pos_ratio = (dff["sentiment_label"].eq("positive").mean()) * 100
c3.metric(t("kpi_pos"), f"{pos_ratio:.1f}%" if pos_ratio is not None else "—")

# Charts
ts = (
    dff.set_index("date")["sentiment_score"]
    .resample("D").mean()
    .reset_index()
    .rename(columns={"sentiment_score": "avg_sentiment"})
)
st.plotly_chart(px.line(ts, x="date", y="avg_sentiment", title=t("time_series")), use_container_width=True)
st.plotly_chart(px.histogram(dff, x="sentiment_score", nbins=40, title=t("dist")), use_container_width=True)

# Table
show_cols = [c for c in ["date", "section", "title", "sentiment_label", "sentiment_score", "url"] if c in dff.columns]
st.subheader(t("recent"))
st.dataframe(dff.sort_values("date", ascending=False)[show_cols].head(300), use_container_width=True)

# Download
st.download_button(
    t("download_csv"),
    data=dff.to_csv(index=False).encode("utf-8"),
    file_name="clarin_sentiment_filtered.csv",
    mime="text/csv"
)
