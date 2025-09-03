import os, io, re, requests
import pandas as pd
import streamlit as st
import plotly.express as px
import analytics

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
        if "sentiment_label" in dff.columns
        else None
    )
    c3.metric(T("kpi_pos"), f"{pos_ratio:.1f}%" if pos_ratio is not None else "—")

    # Gráficos principales
    ts = (
        dff.set_index("date")["sentiment_score"].resample("M").mean()
        .reset_index()
        .rename(columns={"sentiment_score": "avg_sentiment"})
    )
    st.plotly_chart(
        px.line(ts, x="date", y="avg_sentiment", title=T("time_series"), markers=True),
        use_container_width=True,
    )

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

    # Tabla + descarga
    cols = [
        c
        for c in ["date", "section", "title", "sentiment_label", "sentiment_score", "url"]
        if c in dff.columns
    ]
    st.subheader(T("recent"))

    def color_score(val: float) -> str:
        if val > 0:
            return "color: #2ca02c"
        if val < 0:
            return "color: #d62728"
        return "color: #7f7f7f"

    table_df = dff.sort_values("date", ascending=False)[cols].head(300)
    st.dataframe(
        table_df.style.applymap(color_score, subset=["sentiment_score"]),
        use_container_width=True,
    )
    st.download_button(
        T("download_csv"),
        dff.to_csv(index=False).encode("utf-8"),
        file_name="clarin_sentiment_filtered.csv",
        mime="text/csv",
    )

with section_tab:
    sec_avg = analytics.section_avg_sentiment(dff)
    if not sec_avg.empty:
        st.plotly_chart(
            px.bar(
                sec_avg,
                x="section",
                y="avg_sentiment",
                title=T("avg_by_section"),
            ),
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


# --- Sentiment Over Time (Monthly) with Global Avg + Topic Toggle ---
# Requirements: pandas, plotly.express, streamlit
# Expected columns in df:
#   - 'date' (datetime or string)
#   - 'sentiment' (numeric; e.g., polarity in [-1, 1] or [0,1])
#   - 'topic' (string; e.g., 'economy' or 'politics')
# If your column names differ, adjust the CONFIG below.

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

CONFIG = {
    "date_col": "date",
    "sentiment_col": "sentiment",
    "topic_col": "topic",
    # Map UI choices → values in your data
    "topic_values": {
        "Todas": None,          # None means don't filter by topic
        "Economía": ["economy", "economics", "economia", "eco"], 
        "Política": ["politics", "politica", "pol"]
    },
    # Chart labels
    "y_label": "Sentimiento promedio mensual",
    "x_label": "Mes",
    "title_base": "Sentimiento en tapas por mes",
    "global_avg_label": "Promedio global",
}

def _coerce_date(series):
    # Robust date parsing
    s = pd.to_datetime(series, errors="coerce", utc=True)
    # Drop tz for cleaner axis in Plotly (or localize to your tz if you prefer)
    return s.dt.tz_convert(None) if hasattr(s.dt, "tz_convert") else s

def _filter_by_topic(df, topic_col, selected_values_or_none):
    if selected_values_or_none is None:
        return df
    values = set(v.lower() for v in selected_values_or_none)
    if topic_col not in df.columns:
        return df  # no topic in data; treat as "All"
    # Normalize topic values for matching
    topic_norm = df[topic_col].astype(str).str.lower().str.strip()
    return df[topic_norm.isin(values)]

def sentiment_over_time_view(df: pd.DataFrame, key_suffix: str = "sent_time"):
    c = CONFIG

    # --- Basic validation / coercion ---
    if c["date_col"] not in df.columns or c["sentiment_col"] not in df.columns:
        st.error(
            f"No encuentro las columnas requeridas '{c['date_col']}' y '{c['sentiment_col']}'. "
            "Ajustá CONFIG o el dataframe."
        )
        return

    data = df.copy()
    data[c["date_col"]] = _coerce_date(data[c["date_col"]])
    data = data.dropna(subset=[c["date_col"], c["sentiment_col"]])

    # Optional: cap extreme values if your model produces outliers
    # data[c["sentiment_col"]] = data[c["sentiment_col"]].clip(-1, 1)

    # --- Controls ---
    with st.container():
        left, right = st.columns([1, 1])
        with left:
            topic_choice = st.radio(
                "Tema",
                options=list(c["topic_values"].keys()),
                index=0,  # "Todas"
                horizontal=True,
                key=f"topic_choice_{key_suffix}"
            )
        with right:
            # Optional smoothing window (moving average on monthly series)
            smooth = st.select_slider(
                "Suavizado (meses)",
                options=[1, 2, 3, 4, 6, 12],
                value=1,
                help="Promedio móvil sobre el promedio mensual.",
                key=f"smooth_{key_suffix}"
            )

    # --- Topic filter ---
    data_topic = _filter_by_topic(data, c["topic_col"], c["topic_values"][topic_choice])

    if data_topic.empty:
        st.warning("No hay datos para el filtro seleccionado.")
        return

    # --- Monthly aggregation ---
    # Ensure a clean month key
    data_topic["_month"] = data_topic[c["date_col"]].values.astype("datetime64[M]")
    monthly = (
        data_topic
        .groupby("_month", as_index=False)[c["sentiment_col"]]
        .mean()
        .rename(columns={c["sentiment_col"]: "sent_monthly"})
        .sort_values("_month")
    )

    # --- Optional smoothing (simple moving average on monthly means) ---
    if smooth and smooth > 1:
        monthly["sent_smoothed"] = monthly["sent_monthly"].rolling(smooth, min_periods=1).mean()
        y_col = "sent_smoothed"
    else:
        y_col = "sent_monthly"

    # --- Global (all-time) average on the currently filtered data ---
    global_avg = monthly["sent_monthly"].mean()

    # --- Figure ---
    subtitle = "" if topic_choice == "Todas" else f" • {topic_choice}"
    title = f"{c['title_base']}{subtitle}"

    fig = px.line(
        monthly,
        x="_month",
        y=y_col,
        markers=True,
        labels={"_month": c["x_label"], y_col: c["y_label"]},
        title=title
    )

    # Add dotted global average reference line
    fig.add_hline(
        y=global_avg,
        line_dash="dot",
        annotation_text=f"{c['global_avg_label']}: {global_avg:.3f}",
        annotation_position="top left"
    )

    # Format axes
    fig.update_layout(
        xaxis=dict(tickformat="%Y-%m"),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Helpful context / KPIs ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Promedio global (filtro actual)", f"{global_avg:.3f}")
    with col2:
        if len(monthly) >= 2:
            delta = monthly[y_col].iloc[-1] - monthly[y_col].iloc[max(0, len(monthly)-2)]
            st.metric("Último cambio mensual", f"{monthly[y_col].iloc[-1]:.3f}", f"{delta:+.3f}")
        else:
            st.metric("Último mes", f"{monthly[y_col].iloc[-1]:.3f}")
    with col3:
        st.metric("Meses en la serie", f"{len(monthly)}")

# -------------------------
# Example usage in your app:
# df = load_your_dataframe_somehow()
# sentiment_over_time_view(df)
# -------------------------
