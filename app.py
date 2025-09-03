import os, io, re, requests
import pandas as pd
import streamlit as st
import plotly.express as px
import analytics

# --- Optional HF download (for DATA_URL shortcuts or HF_DATASET/HF_FILE) ---
try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

st.set_page_config(page_title="Clarin Sentiment Analysis", layout="wide")

# =============================================================================
# I18N
# =============================================================================
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
        "time_series": "Tendencia mensual del sentimiento",
        "dist": "Distribución de puntajes",
        "recent": "Titulares recientes",
        "download_csv": "Descargar CSV filtrado",
        "no_data": "Sin datos para los filtros seleccionados.",
        "missing_cols": "Faltan columnas requeridas: ",
        "overview": "General",
        "section_tab": "Por sección",
        "headline_tab": "Titulares destacados",
        "avg_by_section": "Sentimiento medio por sección",
        "counts_by_section": "Distribución de sentimiento por sección",
        "top_pos": "Top positivos",
        "top_neg": "Top negativos",
        "no_titles": "No hay títulos disponibles",

        # New for topic view
        "topic": "Tema",
        "all_topics": "Todas",
        "economy": "Economía",
        "politics": "Política",
        "smoothing": "Suavizado (meses)",
        "global_avg_label": "Promedio global",
        "last_month_change": "Último cambio mensual",
        "months_in_series": "Meses en la serie",
        "no_topic_data": "No hay datos para el filtro seleccionado.",
        "no_topic_col": "No hay columna de tema/sección; se muestra el total.",
        "series_title_base": "Sentimiento en tapas por mes",
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
        "time_series": "Monthly sentiment trend",
        "dist": "Score distribution",
        "recent": "Recent headlines",
        "download_csv": "Download filtered CSV",
        "no_data": "No data for selected filters.",
        "missing_cols": "Missing required columns: ",
        "overview": "Overview",
        "section_tab": "By section",
        "headline_tab": "Top headlines",
        "avg_by_section": "Average sentiment by section",
        "counts_by_section": "Sentiment distribution by section",
        "top_pos": "Top positive headlines",
        "top_neg": "Top negative headlines",
        "no_titles": "No titles available",

        # New for topic view
        "topic": "Topic",
        "all_topics": "All",
        "economy": "Economy",
        "politics": "Politics",
        "smoothing": "Smoothing (months)",
        "global_avg_label": "Global average",
        "last_month_change": "Last monthly change",
        "months_in_series": "Months in series",
        "no_topic_data": "No data for the selected filter.",
        "no_topic_col": "No topic/section column found; showing totals.",
        "series_title_base": "Monthly cover sentiment",
    },
}

def T(key: str) -> str:
    lang = st.session_state.get("lang", "es")
    return STRINGS.get(lang, STRINGS["es"]).get(key, key)

# Language selector
if "lang" not in st.session_state:
    st.session_state.lang = "es"
with st.sidebar:
    st.selectbox(STRINGS[st.session_state.lang]["language"], list(LANGS.keys()),
                 format_func=lambda k: LANGS[k], key="lang")

# =============================================================================
# Config / Debug
# =============================================================================
DEBUG = os.getenv("DEBUG", "0").strip() in {"1", "true", "True"}
DATA_URL   = os.getenv("DATA_URL", "").strip()          # https://.../clarin_sentiment.csv or "user/repo:file.csv"
HF_DATASET = os.getenv("HF_DATASET", "").strip()        # e.g., renzungo/cover_master
HF_FILE    = os.getenv("HF_FILE", "").strip()           # e.g., clarin_sentiment.csv
DATE_COL_ENV  = os.getenv("DATE_COL", "").strip().lower()
SCORE_COL_ENV = os.getenv("SCORE_COL", "").strip().lower()
LOCAL_FALLBACK = "data/clarin_sentiment.csv"

REQUIRED_COLS = {"date", "sentiment_score"}

# =============================================================================
# Data Loader (robust)
# =============================================================================
@st.cache_data(ttl=600, show_spinner=True)
def load_data() -> pd.DataFrame:
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        original_cols = list(df.columns)
        df.columns = [c.strip().lower() for c in df.columns]

        # Env overrides
        rename_map = {}
        if DATE_COL_ENV and DATE_COL_ENV in df.columns:
            rename_map[DATE_COL_ENV] = "date"
        if SCORE_COL_ENV and SCORE_COL_ENV in df.columns:
            rename_map[SCORE_COL_ENV] = "sentiment_score"

        # Heuristics
        if "date" not in df.columns:
            for cand in ["fecha", "cover_date", "dia", "día"]:
                if cand in df.columns: rename_map[cand] = "date"; break
        if "sentiment_score" not in df.columns:
            for cand in ["overall_score", "score", "compound", "beto_score", "sentiment"]:
                if cand in df.columns: rename_map[cand] = "sentiment_score"; break
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

        # Coercions
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
        # XLSX signature
        if len(b) >= 4 and b[:4] == b"PK\x03\x04":
            import openpyxl
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

    # A) Direct URL
    if DATA_URL and re.match(r"^https?://", DATA_URL):
        if DEBUG: st.write("🔗 Cargando por DATA_URL:", DATA_URL)
        return read_from_bytes(fetch_bytes(DATA_URL))

    # B) Short HF spec "user/repo:file.csv"
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

    raise FileNotFoundError("Configura DATA_URL (URL) o HF_DATASET+HF_FILE, o añade data/clarin_sentiment.csv")

# =============================================================================
# Sentiment Over Time (monthly) with global average + topic toggle
# =============================================================================
def _normalize_topic_values(series: pd.Series) -> pd.Series:
    """Lowercase & strip; robust against None."""
    return series.astype(str).str.lower().str.strip()

def _filter_by_topic_choice(df: pd.DataFrame, topic_col: str | None, choice: str) -> pd.DataFrame:
    """choice ∈ {All/Economy/Politics} with synonyms; falls back to no filter if column missing."""
    if topic_col is None or topic_col not in df.columns:
        return df

    norm = _normalize_topic_values(df[topic_col])

    economy_syn = {
        "economia", "economía", "economy", "economics", "eco",
        "economia / economía", "finanzas", "economy & business"
    }
    politics_syn = {
        "politica", "política", "politics", "pol", "gobierno", "elecciones"
    }

    if choice == T("economy"):
        return df[norm.isin(economy_syn)]
    if choice == T("politics"):
        return df[norm.isin(politics_syn)]
    return df  # All

def sentiment_over_time_view(
    df: pd.DataFrame,
    date_col: str = "date",
    score_col: str = "sentiment_score",
    topic_col_candidates: tuple[str, ...] = ("topic", "section"),
    key_suffix: str = "sent_time"
):
    # Pick topic column if available
    topic_col = next((c for c in topic_col_candidates if c in df.columns), None)

    # Controls
    left, right = st.columns([1, 1])
    with left:
        topic_choice = st.radio(
            T("topic"),
            options=[T("all_topics"), T("economy"), T("politics")],
            index=0,
            horizontal=True,
            key=f"topic_choice_{key_suffix}",
        )
    with right:
        smooth = st.select_slider(
            T("smoothing"),
            options=[1, 2, 3, 4, 6, 12],
            value=1,
            help="Rolling mean sobre el promedio mensual.",
            key=f"smooth_{key_suffix}",
        )

    # Topic filter
    data = _filter_by_topic_choice(df, topic_col, topic_choice)
    if data.empty:
        st.warning(T("no_topic_data"))
        return

    # Monthly aggregation
    data = data.dropna(subset=[date_col, score_col]).copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    monthly = (
        data.set_index(date_col)[score_col]
        .resample("M").mean()
        .rename("sent_monthly")
        .reset_index()
        .sort_values(date_col)
    )
    if monthly.empty:
        st.warning(T("no_topic_data"))
        return

    # Smoothing (moving average on monthly mean)
    if smooth and smooth > 1:
        monthly["sent_smoothed"] = monthly["sent_monthly"].rolling(smooth, min_periods=1).mean()
        y_col = "sent_smoothed"
    else:
        y_col = "sent_monthly"

    # Global (all-time) average under current topic filter (on monthly means)
    global_avg = monthly["sent_monthly"].mean()

    # Title
    subtitle = "" if topic_choice == T("all_topics") else f" • {topic_choice}"
    title = f"{T('series_title_base')}{subtitle}"

    fig = px.line(
        monthly,
        x=date_col,
        y=y_col,
        markers=True,
        labels={date_col: T("date"), y_col: T("kpi_mean")},
        title=title,
    )
    fig.add_hline(
        y=global_avg,
        line_dash="dot",
        annotation_text=f"{T('global_avg_label')}: {global_avg:.3f}",
        annotation_position="top left",
    )
    fig.update_layout(
        xaxis=dict(tickformat="%Y-%m"),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Mini KPIs for context
    c1, c2, c3 = st.columns(3)
    c1.metric(T("global_avg_label"), f"{global_avg:.3f}")
    if len(monthly) >= 2:
        delta = monthly[y_col].iloc[-1] - monthly[y_col].iloc[-2]
        c2.metric(T("last_month_change"), f"{monthly[y_col].iloc[-1]:.3f}", f"{delta:+.3f}")
    else:
        c2.metric(T("last_month_change"), f"{monthly[y_col].iloc[-1]:.3f}")
    c3.metric(T("months_in_series"), f"{len(monthly)}")

# =============================================================================
# App
# =============================================================================
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

# Sidebar filters
min_d = pd.to_datetime(df["date"]).min()
max_d = pd.to_datetime(df["date"]).max()
if pd.isna(min_d) or pd.isna(max_d):
    st.info(T("no_data")); st.stop()

with st.sidebar:
    date_sel = st.date_input(
        T("date"),
        value=(min_d.date(), max_d.date()),
        min_value=min_d.date(),
        max_value=max_d.date(),
        key="date_filter",
    )
    if isinstance(date_sel, tuple):
        start_d, end_d = date_sel
    else:
        start_d = end_d = date_sel

    sections = sorted(df["section"].dropna().unique().tolist()) if "section" in df.columns else []
    default_secs = sections[:5] if sections else []
    selected_sections = st.multiselect(
        T("section"), sections, default=default_secs, key="section_filter"
    )

mask = df["date"].between(pd.to_datetime(start_d), pd.to_datetime(end_d))
if selected_sections and "section" in df.columns:
    mask &= df["section"].isin(selected_sections)
dff = df.loc[mask].copy()

if dff.empty:
    st.info(T("no_data")); st.stop()

overview_tab, section_tab, headline_tab = st.tabs([
    T("overview"),
    T("section_tab"),
    T("headline_tab"),
])

with overview_tab:
    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric(T("kpi_notes"), int(len(dff)))
    mean_sent = dff["sentiment_score"].mean()
    c2.metric(T("kpi_mean"), f"{mean_sent:.3f}" if pd.notna(mean_sent) else "—")
    pos_ratio = (
        dff["sentiment_label"].eq("positive").mean() * 100
        if "sentiment_label" in dff.columns else None
    )
    c3.metric(T("kpi_pos"), f"{pos_ratio:.1f}%" if pos_ratio is not None else "—")

    # --- Monthly sentiment view with dotted global avg + topic toggle ---
    sentiment_over_time_view(
        dff,
        date_col="date",
        score_col="sentiment_score",
        topic_col_candidates=("topic", "section"),  # tries 'topic' then 'section'
        key_suffix="overview"
    )

    # Distribution
    hist_df = dff.assign(
        score_sign=dff["sentiment_score"].apply(
            lambda x: "positive" if x > 0 else ("negative" if x < 0 else "neutral")
        )
    )
    color_map = {"positive": "#2ca02c", "negative": "#d62728", "neutral": "#7f7f7f"}
    hist_fig = px.histogram(
        hist_df,
        x="sentiment_score",
        nbins=40,
        title=T("dist"),
        color="score_sign",
        color_discrete_map=color_map,
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    # Table + download
    cols = [
        c for c in ["date", "section", "title", "sentiment_label", "sentiment_score", "url"]
        if c in dff.columns
    ]
    st.subheader(T("recent"))

    def color_score(val: float) -> str:
        if val > 0: return "color: #2ca02c"
        if val < 0: return "color: #d62728"
        return "color: #7f7f7f"

    table_df = dff.sort_values("date", ascending=False)[cols].head(300)
    st.dataframe(
        table_df.style.applymap(color_score, subset=["sentiment_score"]) if "sentiment_score" in cols else table_df,
        use_container_width=True,
    )
    st.download_button(
        T("download_csv"),
        dff.to_csv(index=False).encode("utf-8"),
        file_name="clarin_sentiment_filtered.csv",
        mime="text/csv",
        key="download_filtered_csv",
    )

with section_tab:
    sec_avg = analytics.section_avg_sentiment(dff)
    if not sec_avg.empty:
        st.plotly_chart(
            px.bar(sec_avg, x="section", y="avg_sentiment", title=T("avg_by_section")),
            use_container_width=True,
        )
    sec_counts = analytics.section_label_counts(dff)
    if not sec_counts.empty:
        st.plotly_chart(
            px.bar(
                sec_counts,
                x="section",
                y="count",
                color="sentiment_label",
                title=T("counts_by_section"),
                barmode="stack",
            ),
            use_container_width=True,
        )

with headline_tab:
    if "title" in dff.columns:
        pos_df = dff.sort_values("sentiment_score", ascending=False).head(10)
        neg_df = dff.sort_values("sentiment_score", ascending=True).head(10)
        col1, col2 = st.columns(2)
        col1.subheader(T("top_pos"))
        col1.dataframe(
            pos_df[[c for c in ["date", "section", "title", "sentiment_score"] if c in pos_df.columns]]
        )
        col2.subheader(T("top_neg"))
        col2.dataframe(
            neg_df[[c for c in ["date", "section", "title", "sentiment_score"] if c in neg_df.columns]]
        )
    else:
        st.info(T("no_titles"))
