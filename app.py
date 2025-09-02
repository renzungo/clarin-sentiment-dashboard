import os
import io
import re
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

try:
    # optional: only needed if you use HF_DATASET/HF_FILE
    from huggingface_hub import hf_hub_download  # type: ignore
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# -----------------------------
# Page + i18n
# -----------------------------
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
        "missing_cols": "Faltan columnas requeridas: ",
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
        "missing_cols": "Missing required columns: ",
    },
}
def T(key: str) -> str:
    lang = st.session_state.get("lang", "es")
    return STRINGS.get(lang, STRINGS["es"]).get(key, key)

if "lang" not in st.session_state:
    st.session_state.lang = "es"

with st.sidebar:
    st.selectbox(
        STRINGS[st.session_state.lang]["language"],
        options=list(LANGS.keys()),
        format_func=lambda k: LANGS[k],
        key="lang"
    )

# -----------------------------
# Config / Debug
# -----------------------------
REQUIRED_COLS = {"date", "overall_score"}
DEBUG = os.getenv("DEBUG", "0").strip() in {"1", "true", "True"}

DATA_URL = os.getenv("DATA_URL", "").strip()           # e.g., https://.../clarin_sentiment.csv
HF_DATASET = os.getenv("HF_DATASET", "").strip()       # e.g., renzungo/cover_master
HF_FILE = os.getenv("HF_FILE", "").strip()             # e.g., clarin_sentiment.csv
LOCAL_FALLBACK = "data/clarin_sentiment.csv"

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(ttl=600, show_spinner=True)
def load_data() -> pd.DataFrame:
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip().lower() for c in df.columns]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "overall_score" in df.columns:
            df["overall_score"] = pd.to_numeric(df["overall_score"], errors="coerce")
        if "sentiment_label" in df.columns:
            df["sentiment_label"] = (
                df["sentiment_label"].astype(str).str.strip().str.lower()
            )
        # drop rows without date; sort for resampling
        if "date" in df.columns:
            df = df.dropna(subset=["date"]).sort_values("date")
        return df

    def read_from_bytes(b: bytes) -> pd.DataFrame:
        # XLSX signature (zip)
        if len(b) >= 4 and b[:4] == b"PK\x03\x04":
            try:
                import openpyxl  # noqa: F401
                return normalize(pd.read_excel(io.BytesIO(b), engine="openpyxl"))
            except Exception as e:
                raise ValueError(f"XLSX detected but couldn't parse: {e}")

        # Try robust CSV parsing (auto-detect sep, handle BOM, skip bad lines)
        try:
            return normalize(pd.read_csv(
                io.BytesIO(b),
                engine="python",
                sep=None,               # sniff delimiter
                encoding="utf-8-sig",
                on_bad_lines="skip"
            ))
        except Exception:
            pass

        # Explicit separators fallback
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

        raise ValueError("Could not parse file (unknown format/encoding).")

    def fetch_bytes(url: str) -> bytes:
        r = requests.get(url, timeout=60)
        r.raise_for_status()  # raises HTTPError for 4xx/5xx
        return r.content

    # Case 1: Direct HTTP(S) URL (recommended)
    if DATA_URL and re.match(r"^https?://", DATA_URL):
        if DEBUG: st.write("🔗 Loading via DATA_URL:", DATA_URL)
        b = fetch_bytes(DATA_URL)
        return read_from_bytes(b)

    # Case 2: DATA_URL like "user/repo:filename.csv" (short form)
    if DATA_URL and "/" in DATA_URL and ":" in DATA_URL:
        repo, file_ = DATA_URL.split(":", 1)
        if DEBUG: st.write("📦 Loading via DATA_URL (HF repo:file):", repo, file_)
        if not HF_AVAILABLE:
            raise RuntimeError("huggingface_hub not installed; add it to requirements.txt.")
        local_path = hf_hub_download(repo_id=repo, filename=file_, repo_type="dataset")
        return normalize(pd.read_csv(local_path, encoding="utf-8-sig", engine="python", on_bad_lines="skip"))

    # Case 3: HF_DATASET + HF_FILE envs
    if HF_DATASET and HF_FILE:
        if DEBUG: st.write("📦 Loading via HF_DATASET/HF_FILE:", HF_DATASET, HF_FILE)
        if not HF_AVAILABLE:
            raise RuntimeError("huggingface_hub not installed; add it to requirements.txt.")
        local_path = hf_hub_download(repo_id=HF_DATASET, filename=HF_FILE, repo_type="dataset")
        return normalize(pd.read_csv(local_path, encoding="utf-8-sig", engine="python", on_bad_lines="skip"))

    # Case 4: Local fallback (for dev)
    if os.path.exists(LOCAL_FALLBACK):
        if DEBUG: st.write("📁 Loading local fallback:", LOCAL_FALLBACK)
        return normalize(pd.read_csv(LOCAL_FALLBACK, encoding="utf-8-sig", engine="python", on_bad_lines="skip"))

    # Nothing worked
    raise FileNotFoundError(
        "No valid data source. Set DATA_URL to a direct CSV URL, "
        "or set HF_DATASET + HF_FILE, or include data/clarin_sentiment.csv."
    )

# -----------------------------
# App
# -----------------------------
# Surface debug info (optional)
if DEBUG:
    st.sidebar.markdown("### 🐞 Debug")
    st.sidebar.write("DATA_URL =", DATA_URL)
    st.sidebar.write("HF_DATASET =", HF_DATASET)
    st.sidebar.write("HF_FILE =", HF_FILE)
    st.sidebar.write("HF_AVAILABLE =", HF_AVAILABLE)

df = load_data()

# Basic schema check
missing = REQUIRED_COLS - set(df.columns)
if missing:
    st.error(T("missing_cols") + ", ".join(sorted(missing)))
    if DEBUG:
        st.write("Columns present:", list(df.columns))
        st.dataframe(df.head(10))
    st.stop()

st.title(T("title"))
st.caption(T("subtitle"))

if df.empty:
    st.info(T("no_data"))
    st.stop()

# -----------------------------
# Filters
# -----------------------------
min_d = pd.to_datetime(df["date"]).min()
max_d = pd.to_datetime(df["date"]).max()
if pd.isna(min_d) or pd.isna(max_d):
    st.info(T("no_data"))
    if DEBUG:
        st.write("min_d or max_d is NaT")
    st.stop()

with st.sidebar:
    date_sel = st.date_input(
        T("date"),
        value=(min_d.date(), max_d.date()),
        min_value=min_d.date(),
        max_value=max_d.date()
    )
    if isinstance(date_sel, tuple):
        start_d, end_d = date_sel
    else:  # when streamlit returns a single date
        start_d = end_d = date_sel

    sections = sorted(df["section"].dropna().unique().tolist()) if "section" in df.columns else []
    default_sections = sections[:5] if sections else []
    selected_sections = st.multiselect(T("section"), sections, default=default_sections)

mask = df["date"].between(pd.to_datetime(start_d), pd.to_datetime(end_d))
if selected_sections and "section" in df.columns:
    mask &= df["section"].isin(selected_sections)
dff = df.loc[mask].copy()

if dff.empty:
    st.info(T("no_data"))
    if DEBUG:
        st.write("Filters removed all rows. start/end:", start_d, end_d, "sections:", selected_sections)
    st.stop()

# -----------------------------
# KPIs
# -----------------------------
c1, c2, c3 = st.columns(3)
c1.metric(T("kpi_notes"), int(len(dff)))

mean_sent = dff["overall_score"].mean()
c2.metric(T("kpi_mean"), f"{mean_sent:.3f}" if pd.notna(mean_sent) else "—")

pos_ratio = None
if "sentiment_label" in dff.columns:
    pos_ratio = (dff["sentiment_label"].eq("positive").mean()) * 100
c3.metric(T("kpi_pos"), f"{pos_ratio:.1f}%" if pos_ratio is not None else "—")

# -----------------------------
# Charts
# -----------------------------
ts = (
    dff.set_index("date")["overall_score"]
    .resample("D").mean()
    .reset_index()
    .rename(columns={"overall_score": "avg_sentiment"})
)
st.plotly_chart(px.line(ts, x="date", y="avg_sentiment", title=T("time_series")), use_container_width=True)
st.plotly_chart(px.histogram(dff, x="overall_score", nbins=40, title=T("dist")), use_container_width=True)

# -----------------------------
# Table + Download
# -----------------------------
show_cols = [c for c in ["date", "section", "title", "sentiment_label", "overall_score", "url"] if c in dff.columns]
st.subheader(T("recent"))
st.dataframe(dff.sort_values("date", ascending=False)[show_cols].head(300), use_container_width=True)

st.download_button(
    T("download_csv"),
    data=dff.to_csv(index=False).encode("utf-8"),
    file_name="clarin_sentiment_filtered.csv",
    mime="text/csv"
)

# Optional: show sample head in DEBUG
if DEBUG:
    st.markdown("### 🔎 Sample rows")
    st.dataframe(dff.head(20))
