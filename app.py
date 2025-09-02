import os, io, re, requests
import pandas as pd
import streamlit as st
import plotly.express as px

try:
    from huggingface_hub import hf_hub_download  # opcional, solo si usas HF_DATASET/HF_FILE
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

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
    st.selectbox(STRINGS[st.session_state.lang]["language"], list(LANGS.keys()),
                 format_func=lambda k: LANGS[k], key="lang")

# -------- Config / Debug --------
DEBUG = os.getenv("DEBUG", "0").strip() in {"1", "true", "True"}
DATA_URL   = os.getenv("DATA_URL", "").strip()          # https://.../clarin_sentiment.csv
HF_DATASET = os.getenv("HF_DATASET", "").strip()        # p.ej. renzungo/cover_master
HF_FILE    = os.getenv("HF_FILE", "").strip()           # p.ej. clarin_sentiment.csv
DATE_COL_ENV  = os.getenv("DATE_COL", "").strip().lower()
SCORE_COL_ENV = os.getenv("SCORE_COL", "").strip().lower()
LOCAL_FALLBACK = "data/clarin_sentiment.csv"

# Requisitos canónicos
REQUIRED_COLS = {"date", "sentiment_score"}

# -------- Loader robusto --------
@st.cache_data(ttl=600, show_spinner=True)
def load_data() -> pd.DataFrame:
    import io

    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        original_cols = list(df.columns)
        df.columns = [c.strip().lower() for c in df.columns]

        # 1) Overrides por env (si existen)
        rename_map = {}
        if DATE_COL_ENV and DATE_COL_ENV in df.columns:
            rename_map[DATE_COL_ENV] = "date"
        if SCORE_COL_ENV and SCORE_COL_ENV in df.columns:
            rename_map[SCORE_COL_ENV] = "sentiment_score"

        # 2) Sinónimos comunes (solo si faltan los canónicos)
        # date
        if "date" not in df.columns:
            for cand in ["fecha", "cover_date", "dia", "día"]:
                if cand in df.columns: rename_map[cand] = "date"; break
        # sentiment_score
        if "sentiment_score" not in df.columns:
            for cand in ["overall_score", "score", "compound", "beto_score"]:
                if cand in df.columns: rename_map[cand] = "sentiment_score"; break
            # Si hay columna 'sentiment' numérica, úsala como score
            if "sentiment" in df.columns:
                try:
                    if pd.to_numeric(df["sentiment"], errors="coerce").notna().mean() > 0.8:
                        rename_map["sentiment"] = "sentiment_score"
                except Exception:
                    pass
        # section / title / url / label
        if "section" not in df.columns:
            for cand in ["seccion", "sección", "category", "categoria"]:
                if cand in df.columns: rename_map[cand] = "section"; break
        if "title" not in df.columns:
            for cand in ["titulo", "título", "headline", "titular"]:
                if cand in df.columns: rename_map[cand] = "title"; break
        if "url" not in df.columns:
            for cand in ["link", "href", "cover_url"]:
                if cand in df.columns: rename_map[cand] = "url"; break
        if "sentiment_label" not in df.columns:
            for cand in ["label", "pred_label", "polarity", "sentimentclass", "class"]:
                if cand in df.columns: rename_map[cand] = "sentiment_label"; break

        if rename_map:
            df = df.rename(columns=rename_map)

        if DEBUG:
            st.sidebar.markdown("### 🐞 Debug columnas")
            st.sidebar.write("Original:", original_cols)
            st.sidebar.write("Después de renombrar:", list(df.columns))
            st.sidebar.write("Renames aplicados:", rename_map)

        # Coerciones
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "sentiment_score" in df.columns:
            df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
        if "sentiment_label" in df.columns:
            df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()

        if "date" in df.columns:
            df = df.dropna(subset=["date"]).sort_values("date")

        return df

    def read_from_bytes(b: bytes) -> pd.DataFrame:
        # XLSX (zip signature)
        if len(b) >= 4 and b[:4] == b"PK\x03\x04":
            import openpyxl  # ensure installed
            return normalize(pd.read_excel(io.BytesIO(b), engine="openpyxl"))

        # CSV autodetect
        try:
            return normalize(pd.read_csv(io.BytesIO(b), engine="python",
                                         sep=None, encoding="utf-8-sig",
                                         on_bad_lines="skip"))
        except Exception:
            pass
        for sep in [",", ";", "\t", "|"]:
            try:
                return normalize(pd.read_csv(io.BytesIO(b), engine="python",
                                             sep=sep, encoding="utf-8-sig",
                                             on_bad_lines="skip"))
            except Exception:
                continue
        raise ValueError("No se pudo parsear el archivo (formato/encoding desconocido).")

    def fetch_bytes(url: str) -> bytes:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return r.content

    # A) URL directa
    if DATA_URL and re.match(r"^https?://", DATA_URL):
        if DEBUG: st.write("🔗 Cargando por DATA_URL:", DATA_URL)
        return read_from_bytes(fetch_bytes(DATA_URL))

    # B) DATA_URL estilo "user/repo:file.csv"
    if DATA_URL and "/" in DATA_URL and ":" in DATA_URL:
        if not HF_AVAILABLE:
            raise RuntimeError("Falta huggingface_hub en requirements.txt")
        repo, file_ = DATA_URL.split(":", 1)
        if DEBUG: st.write("📦 HF corto:", repo, file_)
        local_path = hf_hub_download(repo_id=repo, filename=file_, repo_type="dataset")
        return normalize(pd.read_csv(local_path, encoding="utf-8-sig", engine="python", on_bad_lines="skip"))

    # C) HF_DATASET + HF_FILE
    if HF_DATASET and HF_FILE:
        if not HF_AVAILABLE:
            raise RuntimeError("Falta huggingface_hub en requirements.txt")
        if DEBUG: st.write("📦 HF envs:", HF_DATASET, HF_FILE)
        local_path = hf_hub_download(repo_id=HF_DATASET, filename=HF_FILE, repo_type="dataset")
        return normalize(pd.read_csv(local_path, encoding="utf-8-sig", engine="python", on_bad_lines="skip"))

    # D) Local fallback
    if os.path.exists(LOCAL_FALLBACK):
        if DEBUG: st.write("📁 Local fallback:", LOCAL_FALLBACK)
        return normalize(pd.read_csv(LOCAL_FALLBACK, encoding="utf-8-sig", engine="python", on_bad_lines="skip"))

    raise FileNotFoundError("Configura DATA_URL (URL directa) o HF_DATASET+HF_FILE, o añade data/clarin_sentiment.csv")

# -------- App --------
if DEBUG:
    st.sidebar.markdown("### ⚙️ Debug env")
    st.sidebar.write("DATA_URL =", DATA_URL)
    st.sidebar.write("HF_DATASET =", HF_DATASET)
    st.sidebar.write("HF_FILE =", HF_FILE)
    st.sidebar.write("DATE_COL =", DATE_COL_ENV or "(auto)")
    st.sidebar.write("SCORE_COL =", SCORE_COL_ENV or "(auto)")
    st.sidebar.write("HF_AVAILABLE =", HF_AVAILABLE)

df = load_data()

missing = REQUIRED_COLS - set(df.columns)
if missing:
    st.error(T("missing_cols") + ", ".join(sorted(missing)))
    if DEBUG:
        st.write("Columnas presentes:", list(df.columns))
        st.dataframe(df.head(20))
    st.stop()

st.title(T("title"))
st.caption(T("subtitle"))

if df.empty:
    st.info(T("no_data"))
    st.stop()

# Filtros
min_d = pd.to_datetime(df["date"]).min()
max_d = pd.to_datetime(df["date"]).max()
if pd.isna(min_d) or pd.isna(max_d):
    st.info(T("no_data")); st.stop()

with st.sidebar:
    date_sel = st.date_input(T("date"), value=(min_d.date(), max_d.date()),
                             min_value=min_d.date(), max_value=max_d.date())
    if isinstance(date_sel, tuple): start_d, end_d = date_sel
    else: start_d = end_d = date_sel

    sections = sorted(df["section"].dropna().unique().tolist()) if "section" in df.columns else []
    selected_sections = st.multiselect(T("section"), sections, default=sections[:5] if sections else [])

mask = df["date"].between(pd.to_datetime(start_d), pd.to_datetime(end_d))
if selected_sections and "section" in df.columns:
    mask &= df["section"].isin(selected_sections)
dff = df.loc[mask].copy()

if dff.empty:
    st.info(T("no_data")); st.stop()

# KPIs
c1, c2, c3 = st.columns(3)
c1.metric(T("kpi_notes"), int(len(dff)))
mean_sent = dff["sentiment_score"].mean()
c2.metric(T("kpi_mean"), f"{mean_sent:.3f}" if pd.notna(mean_sent) else "—")
pos_ratio = (dff["sentiment_label"].eq("positive").mean()*100) if "sentiment_label" in dff.columns else None
c3.metric(T("kpi_pos"), f"{pos_ratio:.1f}%" if pos_ratio is not None else "—")

# Gráficos
ts = (dff.set_index("date")["sentiment_score"].resample("D").mean()
      .reset_index().rename(columns={"sentiment_score":"avg_sentiment"}))
st.plotly_chart(px.line(ts, x="date", y="avg_sentiment", title=T("time_series")), use_container_width=True)
st.plotly_chart(px.histogram(dff, x="sentiment_score", nbins=40, title=T("dist")), use_container_width=True)

# Tabla + descarga
cols = [c for c in ["date","section","title","sentiment_label","sentiment_score","url"] if c in dff.columns]
st.subheader(T("recent"))
st.dataframe(dff.sort_values("date", ascending=False)[cols].head(300), use_container_width=True)
st.download_button(T("download_csv"), dff.to_csv(index=False).encode("utf-8"),
                   file_name="clarin_sentiment_filtered.csv", mime="text/csv")
