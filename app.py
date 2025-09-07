# app.py — Clarin Sentiment Dashboard (completo)
import os, io, re, ast, json, requests
from typing import List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ---- Optional: analytics helpers (si no están, usamos fallbacks) ----
try:
    import analytics  # section_avg_sentiment(df), section_label_counts(df)
    HAS_ANALYTICS = True
except Exception:
    HAS_ANALYTICS = False

# ---- Optional: HuggingFace hub (para DATA_URL "user/repo:file.csv" o HF_DATASET/HF_FILE) ----
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
        "subtitle": "Explora la evolución del sentimiento, entidades y secciones del diario",
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
        "entities_tab": "Entidades",
        "section_tab": "Por sección",
        "headline_tab": "Titulares",
        "avg_by_section": "Sentimiento medio por sección",
        "counts_by_section": "Distribución de sentimiento por sección",
        "top_pos": "Top positivos",
        "top_neg": "Top negativos",
        "no_titles": "No hay títulos disponibles",

        # Topic view
        "topic": "Tema",
        "all_topics": "Todas",
        "economy": "Economía",
        "politics": "Política",
        "smoothing": "Suavizado (meses)",
        "global_avg_label": "Promedio global",
        "last_month_change": "Último cambio mensual",
        "months_in_series": "Meses en la serie",
        "no_topic_data": "No hay datos para el filtro seleccionado.",
        "series_title_base": "Sentimiento en tapas por mes",

        # Entities
        "entities_title": "Entidades principales por sentimiento",
        "entity_type": "Tipo de entidad",
        "min_mentions": "Mín. menciones",
        "top_n": "Top N",
        "compute_ner": "Calcular NER con spaCy (lento) si faltan entidades",
        "avg_sentiment": "Sentimiento medio",
        "mentions": "Menciones",

        # Keywords
        "keywords_tab": "Palabras clave",
        "keywords_title": "Términos y frases más frecuentes",
        "use_wordcloud": "Mostrar WordCloud (si está disponible)",
        "ngram_size": "Tamaño de n-grama",
        "remove_stopwords": "Remover stopwords",
        "text_cols": "Columnas de texto",
        "no_text_cols": "No se encontraron columnas de texto. Definí TEXT_COLS o elegí otras.",

        # Coverage
        "coverage_tab": "Cobertura",
        "coverage_title": "Participación de cobertura por tema (mensual)",
        "bucket": "Agrupar en",
        "bucket_econ": "Economía",
        "bucket_politics": "Política",
        "bucket_other": "Otros",
    },
    "en": {
        "title": "Clarin - Sentiment Analysis",
        "subtitle": "Explore sentiment trends, entities and sections",
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
        "entities_tab": "Entities",
        "section_tab": "By section",
        "headline_tab": "Headlines",
        "avg_by_section": "Average sentiment by section",
        "counts_by_section": "Sentiment distribution by section",
        "top_pos": "Top positive",
        "top_neg": "Top negative",
        "no_titles": "No titles available",

        # Topic view
        "topic": "Topic",
        "all_topics": "All",
        "economy": "Economy",
        "politics": "Politics",
        "smoothing": "Smoothing (months)",
        "global_avg_label": "Global average",
        "last_month_change": "Last monthly change",
        "months_in_series": "Months in series",
        "no_topic_data": "No data for the selected filter.",
        "series_title_base": "Monthly cover sentiment",

        # Entities
        "entities_title": "Top entities by sentiment",
        "entity_type": "Entity type",
        "min_mentions": "Min. mentions",
        "top_n": "Top N",
        "compute_ner": "Run spaCy NER (slow) if entities missing",
        "avg_sentiment": "Avg sentiment",
        "mentions": "Mentions",

        # Keywords
        "keywords_tab": "Keywords",
        "keywords_title": "Most frequent terms and phrases",
        "use_wordcloud": "Show WordCloud (if available)",
        "ngram_size": "N-gram size",
        "remove_stopwords": "Remove stopwords",
        "text_cols": "Text columns",
        "no_text_cols": "No text columns found. Set TEXT_COLS or pick others.",

        # Coverage
        "coverage_tab": "Coverage",
        "coverage_title": "Coverage share by topic (monthly)",
        "bucket": "Group into",
        "bucket_econ": "Economy",
        "bucket_politics": "Politics",
        "bucket_other": "Other",
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

# =============================================================================
# Config / Debug
# =============================================================================
DEBUG = os.getenv("DEBUG", "0").strip() in {"1", "true", "True"}
DATA_URL   = os.getenv("DATA_URL", "").strip()          # https://.../file.csv | "user/repo:file.csv"
HF_DATASET = os.getenv("HF_DATASET", "").strip()
HF_FILE    = os.getenv("HF_FILE", "").strip()
DATE_COL_ENV  = os.getenv("DATE_COL", "").strip().lower()
SCORE_COL_ENV = os.getenv("SCORE_COL", "").strip().lower()
LOCAL_FALLBACK = "data/clarin_sentiment.csv"

# Texto / Entidades por ENV (con defaults a tus columnas JSON)
TEXT_COLS_ENV      = os.getenv("TEXT_COLS", "").strip()        # opcional (p.ej. "title,ocr_text")
ENTITIES_COLS_ENV  = os.getenv("ENTITIES_COLS", "").strip()    # aquí vacío (no hay 'entities' global)
PERSON_COLS_ENV    = os.getenv("PERSON_COLS", "people_json").strip()
ORG_COLS_ENV       = os.getenv("ORG_COLS", "orgs_json").strip()
LOC_COLS_ENV       = os.getenv("LOC_COLS", "places_json").strip()

# =============================================================================
# Requisitos
# =============================================================================
REQUIRED_COLS = {"date"}  # las columnas de puntaje se resuelven dinámicamente

# =============================================================================
# Loader
# =============================================================================
@st.cache_data(ttl=600, show_spinner=True)
def load_data() -> pd.DataFrame:
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        original_cols = list(df.columns)
        df.columns = [c.strip().lower() for c in df.columns]

        # Renames por ENV
        rename_map = {}
        if DATE_COL_ENV and DATE_COL_ENV in df.columns:
            rename_map[DATE_COL_ENV] = "date"
        if SCORE_COL_ENV and SCORE_COL_ENV in df.columns:
            rename_map[SCORE_COL_ENV] = "sentiment_score"

        # Heurística de nombres comunes
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

        # Coerciones
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
        if "sentiment_label" in df.columns:
            df["sentiment_label"] = df["sentiment_label"].astype(str).str.lower().str.strip()
        return df

    def read_from_bytes(b: bytes) -> pd.DataFrame:
        # XLSX por firma ZIP
        if len(b) >= 4 and b[:4] == b"PK\x03\x04":
            try:
                import openpyxl  # noqa: F401
                return normalize(pd.read_excel(io.BytesIO(b), engine="openpyxl"))
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "Falta la dependencia 'openpyxl' para leer archivos .xlsx. "
                    "Agrega 'openpyxl>=3.1.2' a requirements.txt o usa CSV."
                ) from e

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

    # B) DATA_URL "user/repo:file.csv"
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

    raise FileNotFoundError("Configura DATA_URL o HF_DATASET+HF_FILE, o añade data/clarin_sentiment.csv")

# =============================================================================
# Resolución de columnas por tema (métrica activa)
# =============================================================================
SCORE_CANDIDATES = {
    "all": ["sentiment_score", "overall_score", "score", "compound", "beto_score"],
    "economy": ["eco_sent_score", "economy_sent_score", "economy_score", "eco_score", "eco_sentiment", "eco_polarity"],
    "politics": ["gov_sent_score", "pol_sent_score", "politics_sent_score", "government_sent_score", "politics_score", "pol_score", "gov_score"],
}
LABEL_CANDIDATES = {
    "all": ["sentiment_label", "label"],
    "economy": ["eco_sent_label", "eco_label"],
    "politics": ["gov_sent_label", "pol_sent_label", "politics_label"],
}
def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None
def _ui_choice_key(choice: str) -> str:
    if choice == T("economy"): return "economy"
    if choice == T("politics"): return "politics"
    return "all"
def resolve_score_and_label_cols(df: pd.DataFrame, ui_choice: str) -> Tuple[str, Optional[str], Optional[str]]:
    key = _ui_choice_key(ui_choice)
    score_col = _first_present(df, SCORE_CANDIDATES[key]) or _first_present(df, SCORE_CANDIDATES["all"])
    label_col = _first_present(df, LABEL_CANDIDATES[key]) or _first_present(df, LABEL_CANDIDATES["all"])
    return key, score_col, label_col

# =============================================================================
# Texto y NER helpers
# =============================================================================
DEFAULT_TEXT_CANDIDATES = [
    "title","titulo","título","headline","titular",
    "cover_text","ocr_text","ocr_clean","ocr","text","body",
    "desc","description","resumen","bajada","copete","lead","subtitulo","subtítulo"
]
EXCLUDE_TEXT_COLS = {"url","section","topic","entities","sentiment_label"}
def _split_env_cols(s: str) -> List[str]:
    return [c.strip().lower() for c in s.split(",") if c.strip()]
def pick_text_columns(df: pd.DataFrame) -> List[str]:
    # 1) ENV
    cols_env = _split_env_cols(TEXT_COLS_ENV) if TEXT_COLS_ENV else []
    cols_env = [c for c in cols_env if c in df.columns]
    # 2) Nombres típicos
    cand_named = [c for c in DEFAULT_TEXT_CANDIDATES if c in df.columns]
    # 3) Otros object
    others = [
        c for c in df.columns
        if df[c].dtype == "object" and c not in EXCLUDE_TEXT_COLS and c not in cand_named and c not in cols_env
    ]
    # Merge sin duplicados
    seen, ordered = set(), []
    for c in cols_env + cand_named + others:
        if c not in seen:
            seen.add(c); ordered.append(c)
    return ordered

# Stopwords mínimas (ES/EN)
STOPWORDS_ES = set("""
de la que el en y a los del se las por un para con no una su al lo como más o pero sus le ya
esta este fue porque entre cuando muy sin sobre también me hasta hay donde quien desde todo nos
durante todos uno les ni contra otros ese eso ante ellos e esto mi antes algunos qué unos yo otro
otras otra cual cuales
""".split())
STOPWORDS_EN = set("""
the of and to in a is that for it on with as was at by an be this are from or have has not but
you they he she we were their his her your its our also if when which where how what who
""".split())
def get_stopwords():
    return STOPWORDS_ES if st.session_state.get("lang","es")=="es" else STOPWORDS_EN
def tokenize(s: str) -> List[str]:
    s = re.sub(r"http\S+|www\.\S+", " ", str(s))
    s = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\s]", " ", s)
    toks = [t.lower() for t in s.split() if t.strip()]
    sw = get_stopwords()
    if st.session_state.get("remove_sw", True):
        toks = [t for t in toks if t not in sw and len(t) > 2]
    return toks
def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# =============================================================================
# Entidades (parser declarativo por ENV) + defensas
# =============================================================================
def _looks_like_filename(s: str) -> bool:
    s = str(s or "").strip().lower()
    if not s:
        return False
    if "/" in s or "\\" in s:
        return True
    if re.search(r"\.(txt|csv|json|pdf|docx?)$", s):
        return True
    if re.match(r"(cover|tapa|portada)[ _-]?\d{8}\.txt$", s):
        return True
    return False

def parse_entities_column(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Devuelve ['row_id','entity','entity_norm','label'] leyendo EXCLUSIVAMENTE:
      - ENTITIES_COLS_ENV (si se declaró) -> cada item puede ser lista/dict o string "PER: Milei"
      - PERSON_COLS_ENV, ORG_COLS_ENV, LOC_COLS_ENV -> SOPORTAN JSON/listas/dicts o strings separados por ; , | /
    """
    if df.empty:
        return None

    out = []

    def _emit(label: str, ent: str, rid: int):
        ent = str(ent).strip()
        if not ent or _looks_like_filename(ent):
            return
        out.append({
            "row_id": rid,
            "entity": ent,
            "entity_norm": re.sub(r"\s+", " ", ent).strip().title(),
            "label": (label or "").upper()
        })

    def _maybe_parse_json_like(val):
        if isinstance(val, (list, tuple, dict)):
            return val
        s = str(val)
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            return ast.literal_eval(s)
        except Exception:
            return None

    def _extract_items_any(val) -> List[str]:
        """
        Devuelve lista de strings desde:
        - lista/tupla de strings o dicts (usa keys text/name/entity/value si dict)
        - string con JSON/lista Python
        - string plano separado por ; , | /
        """
        if isinstance(val, (list, tuple)):
            seq = val
        else:
            parsed = _maybe_parse_json_like(val)
            if isinstance(parsed, (list, tuple)):
                seq = parsed
            else:
                return [x.strip() for x in re.split(r"[;,\|/]\s*", str(val)) if x.strip()]

        items = []
        for it in seq:
            if isinstance(it, dict):
                items.append(it.get("text") or it.get("name") or it.get("entity") or it.get("value") or "")
            else:
                items.append(str(it))
        return [x.strip() for x in items if str(x).strip()]

    # 1) ENTITIES_COLS exactas (si se declararon)
    entity_like_cols = [c for c in (ENTITIES_COLS_ENV.split(",") if ENTITIES_COLS_ENV else [])]
    entity_like_cols = [c.strip().lower() for c in entity_like_cols if c.strip()]
    entity_like_cols = [c for c in entity_like_cols if c in df.columns]

    for col in entity_like_cols:
        for idx, row in df.iterrows():
            rid = int(row.get("row_id", idx))
            raw = row[col]
            if pd.isna(raw):
                continue
            parsed = _maybe_parse_json_like(raw)
            if isinstance(parsed, (list, tuple)):
                for item in parsed:
                    if isinstance(item, dict):
                        _emit(item.get("label",""), item.get("text", item.get("entity","")), rid)
                    else:
                        _emit("", item, rid)
                continue
            parts = re.split(r"[;,\|]\s*", str(raw))
            for p in parts:
                m = re.match(r"(?P<label>\w+)\s*[:|-]\s*(?P<ent>.+)$", p)
                if m: _emit(m.group("label"), m.group("ent"), rid)
                else: _emit("", p, rid)

    # 2) Columnas separadas por tipo (EXACTAS por ENV)
    def _cols_from_env(s: str) -> List[str]:
        return [c.strip().lower() for c in s.split(",") if c.strip()] if s else []

    person_cols = [c for c in _cols_from_env(PERSON_COLS_ENV) if c in df.columns]
    org_cols    = [c for c in _cols_from_env(ORG_COLS_ENV)    if c in df.columns]
    loc_cols    = [c for c in _cols_from_env(LOC_COLS_ENV)    if c in df.columns]

    for cols, lab in [(person_cols, "PERSON"), (org_cols, "ORG"), (loc_cols, "LOC")]:
        for col in cols:
            for idx, row in df.iterrows():
                rid = int(row.get("row_id", idx))
                raw = row[col]
                if pd.isna(raw):
                    continue
                for ent in _extract_items_any(raw):
                    _emit(lab, ent, rid)

    if not out:
        return None

    ents = pd.DataFrame(out)
    ents["label"] = ents["label"].fillna("").str.upper()
    return ents[["row_id","entity","entity_norm","label"]]

@st.cache_data(ttl=1200, show_spinner=True)
def spacy_ner_from_rows(dff: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
    """NER con spaCy sobre columnas de texto por fila → ['row_id','entity','entity_norm','label']."""
    try:
        import spacy
        try:
            nlp = spacy.load("es_core_news_md")
        except Exception:
            import spacy.cli as spcli
            spcli.download("es_core_news_md")
            nlp = spacy.load("es_core_news_md")
    except Exception:
        return pd.DataFrame(columns=["row_id","entity","entity_norm","label"])

    out = []
    for idx, row in dff.iterrows():
        rid = int(row.get("row_id", idx))
        text = " ".join([str(row[c]) for c in text_cols if c in dff.columns and pd.notna(row[c])])
        if not text.strip(): continue
        doc = nlp(text)
        for ent in doc.ents:
            out.append({"row_id": rid, "entity": ent.text, "entity_norm": ent.text.strip().title(), "label": ent.label_.upper()})
    return pd.DataFrame(out)

# =============================================================================
# Topic buckets (Cobertura)
# =============================================================================
ECON_SYNS = {"economia","economía","economy","economics","eco","finanzas","economy & business"}
POLI_SYNS = {"politica","política","politics","pol","gobierno","elecciones","congreso"}
def bucket_topic(val: str) -> str:
    s = str(val).strip().lower()
    if s in ECON_SYNS: return T("bucket_econ")
    if s in POLI_SYNS: return T("bucket_politics")
    return T("bucket_other")

# =============================================================================
# Vistas
# =============================================================================
def sentiment_over_time_view(df: pd.DataFrame, date_col: str = "date", key_suffix: str = "sent_time"):
    left, right = st.columns([1, 1])
    with left:
        topic_choice = st.radio(T("topic"),
            options=[T("all_topics"), T("economy"), T("politics")],
            index=0, horizontal=True, key=f"topic_choice_{key_suffix}")
    with right:
        smooth = st.select_slider(T("smoothing"),
            options=[1,2,3,4,6,12], value=1,
            help="Rolling mean sobre el promedio mensual.", key=f"smooth_{key_suffix}")

    scope_key, score_col, label_col = resolve_score_and_label_cols(df, topic_choice)
    if not score_col:
        st.error("No encontré una columna de puntaje para el tema seleccionado.")
        return {"scope_key": scope_key, "score_col": None, "label_col": label_col, "topic_choice": topic_choice}

    st.session_state[f"active_scope_{key_suffix}"] = scope_key
    st.session_state[f"active_score_col_{key_suffix}"] = score_col
    st.session_state[f"active_label_col_{key_suffix}"] = label_col
    st.session_state[f"active_choice_label_{key_suffix}"] = topic_choice

    data = df.dropna(subset=[date_col, score_col]).copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")

    monthly = (data.set_index(date_col)[score_col]
               .resample("M").mean().rename("sent_monthly")
               .reset_index().sort_values(date_col))
    if monthly.empty:
        st.warning(T("no_topic_data"))
        return {"scope_key": scope_key, "score_col": score_col, "label_col": label_col, "topic_choice": topic_choice}

    y_col = "sent_monthly"
    if smooth and smooth > 1:
        monthly["sent_smoothed"] = monthly["sent_monthly"].rolling(smooth, min_periods=1).mean()
        y_col = "sent_smoothed"

    global_avg = monthly["sent_monthly"].mean()
    subtitle = "" if topic_choice == T("all_topics") else f" • {topic_choice}"
    title = f"{T('series_title_base')}{subtitle}"

    fig = px.line(monthly, x=date_col, y=y_col, markers=True,
                  labels={date_col: T("date"), y_col: T("kpi_mean")}, title=title)
    fig.add_hline(y=global_avg, line_dash="dot",
                  annotation_text=f"{T('global_avg_label')}: {global_avg:.3f}",
                  annotation_position="top left")
    fig.update_layout(xaxis=dict(tickformat="%Y-%m"), hovermode="x unified",
                      margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric(T("global_avg_label"), f"{global_avg:.3f}")
    if len(monthly) >= 2:
        delta = monthly[y_col].iloc[-1] - monthly[y_col].iloc[-2]
        c2.metric(T("last_month_change"), f"{monthly[y_col].iloc[-1]:.3f}", f"{delta:+.3f}")
    else:
        c2.metric(T("last_month_change"), f"{monthly[y_col].iloc[-1]:.3f}")
    c3.metric(T("months_in_series"), f"{len(monthly)}")

    if DEBUG:
        st.caption(f"🛠️ Columna activa: **{score_col}** (scope: {scope_key}) | Label: {label_col or '—'}")

    return {"scope_key": scope_key, "score_col": score_col, "label_col": label_col, "topic_choice": topic_choice}

def distribution_view(dff: pd.DataFrame, active_col: str):
    hist_df = dff.assign(score_sign=dff[active_col].apply(lambda x: "positive" if x > 0 else ("negative" if x < 0 else "neutral")))
    color_map = {"positive": "#2ca02c", "negative": "#d62728", "neutral": "#7f7f7f"}
    st.plotly_chart(
        px.histogram(hist_df, x=active_col, nbins=40, title=T("dist"),
                     color="score_sign", color_discrete_map=color_map),
        use_container_width=True
    )

def keywords_view(dff: pd.DataFrame, key_suffix: str = "kw"):
    st.subheader(T("keywords_title"))
    candidates = pick_text_columns(dff)
    if not candidates:
        st.info(T("no_text_cols")); return
    default_pick = [c for c in candidates if c in DEFAULT_TEXT_CANDIDATES][:2] or candidates[:1]

    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        chosen_cols = st.multiselect(T("text_cols"), options=candidates, default=default_pick, key=f"text_cols_{key_suffix}")
    with col_b:
        n = st.select_slider(T("ngram_size"), options=[1,2,3], value=1, key=f"ng_{key_suffix}")
    with col_c:
        st.checkbox(T("remove_stopwords"), value=True, key="remove_sw")
    if not chosen_cols:
        st.info(T("no_text_cols")); return

    texts = []
    for col in chosen_cols:
        texts.extend(dff[col].dropna().astype(str).map(str.strip).replace("", np.nan).dropna().tolist())
    if not texts:
        st.info(T("no_text_cols")); return

    tokens = []
    for t in texts:
        tokens.extend(tokenize(t))
    grams = [" ".join(g) for g in ngrams(tokens, n)] if n > 1 else tokens
    if not grams:
        st.info(T("no_text_cols")); return

    counts = Counter(grams).most_common(50)
    kw_df = pd.DataFrame(counts, columns=["term", "freq"])

    # WordCloud sin matplotlib
    use_wc = st.checkbox(T("use_wordcloud"), value=False, key=f"use_wc_{key_suffix}")
    if use_wc:
        try:
            from wordcloud import WordCloud
            FONT_CANDIDATES = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            ]
            font_path = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
            wc = WordCloud(
                width=900, height=400, background_color="white",
                collocations=False, prefer_horizontal=0.9,
                font_path=font_path,
                regexp=r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+(?:\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+)*"
            ).generate_from_frequencies(dict(counts))
            st.image(wc.to_array(), use_container_width=True, caption=T("keywords_title"))
        except Exception as e:
            st.info("WordCloud no disponible; mostrando tabla/gráfico.")
            if DEBUG: st.exception(e)

    bar = px.bar(kw_df.head(20), x="term", y="freq", title=T("keywords_title"))
    bar.update_layout(xaxis={"categoryorder":"total descending"}, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(bar, use_container_width=True)
    st.dataframe(kw_df, use_container_width=True)

def entities_view(dff: pd.DataFrame, active_col: str, key_suffix: str = "ents"):
    st.subheader(T("entities_title"))
    l, m, r = st.columns([1,1,1])
    with l:
        ent_type = st.selectbox(T("entity_type"), options=["PERSON", "ORG", "LOC", "ALL"], index=1, key=f"etype_{key_suffix}")
    with m:
        min_m = st.slider(T("min_mentions"), 1, 20, 3, key=f"minm_{key_suffix}")
    with r:
        compute = st.checkbox(T("compute_ner"), value=False, key=f"ner_{key_suffix}")

    if DEBUG:
        st.caption("🔎 Diagnóstico Entities")
        st.write({
            "rows_filtrados": len(dff),
            "people_cols_env": PERSON_COLS_ENV,
            "org_cols_env": ORG_COLS_ENV,
            "loc_cols_env": LOC_COLS_ENV,
            "present_people_cols": [c for c in _split_env_cols(PERSON_COLS_ENV) if c in dff.columns],
            "present_org_cols": [c for c in _split_env_cols(ORG_COLS_ENV) if c in dff.columns],
            "present_loc_cols": [c for c in _split_env_cols(LOC_COLS_ENV) if c in dff.columns],
            "texto_detectado": [c for c in dff.columns if dff[c].dtype=="object"][:10],
            "puntaje_activo": active_col,
        })

    ents = parse_entities_column(dff)

    # Si no hay entidades y el usuario lo habilita, corré NER sobre columnas de texto
    if (ents is None or ents.empty) and compute:
        text_candidates = pick_text_columns(dff)
        chosen = [c for c in text_candidates if c in DEFAULT_TEXT_CANDIDATES][:2] or text_candidates[:1]
        ents = spacy_ner_from_rows(dff, chosen)

    if ents is None or ents.empty:
        st.info("No hay entidades disponibles. Subí columnas (people_json/orgs_json/places_json) o activa NER.")
        return

    if ent_type != "ALL":
        ents = ents[ents["label"].eq(ent_type)]

    if "row_id" not in dff.columns or active_col not in dff.columns:
        st.info("Falta 'row_id' o la columna de puntaje activa para calcular promedios.")
        return
    merged = ents.merge(dff[["row_id", active_col]], on="row_id", how="left")

    agg = (merged.groupby("entity_norm", as_index=False)
           .agg(avg_sentiment=(active_col,"mean"), mentions=("entity","count"))
           .sort_values("mentions", ascending=False))
    agg = agg[agg["mentions"] >= min_m]

    top_n = st.slider(T("top_n"), 5, 50, 20, key=f"topn_{key_suffix}")
    show = agg.sort_values("avg_sentiment", ascending=False).head(top_n)

    fig = px.bar(show, x="entity_norm", y="avg_sentiment", color="avg_sentiment",
                 color_continuous_scale=["#d62728","#7f7f7f","#2ca02c"],
                 title=T("entities_title"))
    fig.update_layout(xaxis={"categoryorder": "total ascending"}, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(agg.head(200), use_container_width=True)

def coverage_view(dff: pd.DataFrame, key_suffix: str = "cov"):
    # bucket por topic/section
    if "topic" in dff.columns:
        base = dff.assign(bucket=dff["topic"].apply(bucket_topic))
    elif "section" in dff.columns:
        base = dff.assign(bucket=dff["section"].apply(bucket_topic))
    else:
        st.info("No hay columna de 'topic' o 'section' para agrupar cobertura."); return

    base["month"] = pd.to_datetime(base["date"]).values.astype("datetime64[M]")
    counts = (base.groupby(["month","bucket"], as_index=False)
              .size().rename(columns={"size":"count"}))
    totals = counts.groupby("month", as_index=False)["count"].sum().rename(columns={"count":"total"})
    shares = counts.merge(totals, on="month", how="left")
    shares["share"] = shares["count"] / shares["total"]

    fig = px.area(shares, x="month", y="share", color="bucket",
                  title=T("coverage_title"), groupnorm="fraction")
    fig.update_layout(xaxis=dict(tickformat="%Y-%m"), hovermode="x unified", margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(fig, use_container_width=True)

# Fallbacks por si no está el módulo analytics
def _section_avg_sentiment_fallback(dff: pd.DataFrame, active_col: str) -> pd.DataFrame:
    if "section" not in dff.columns or active_col not in dff.columns:
        return pd.DataFrame(columns=["section", "avg_sentiment"])
    out = dff.groupby("section", as_index=False)[active_col].mean().rename(columns={active_col:"avg_sentiment"})
    return out.sort_values("avg_sentiment", ascending=False)
def _section_label_counts_fallback(dff: pd.DataFrame) -> pd.DataFrame:
    if "section" not in dff.columns or "sentiment_label" not in dff.columns:
        return pd.DataFrame(columns=["section", "sentiment_label", "count"])
    out = (dff.groupby(["section","sentiment_label"], as_index=False).size()
           .rename(columns={"size":"count"}))
    return out.sort_values(["section","count"], ascending=[True, False])

# =============================================================================
# App
# =============================================================================
if DEBUG:
    st.sidebar.markdown("### ⚙️ Debug env")
    st.sidebar.write("DATA_URL =", DATA_URL)
    st.sidebar.write("HF_DATASET =", HF_DATASET)
    st.sidebar.write("HF_FILE =", HF_FILE)
    st.sidebar.write("HF_AVAILABLE =", HF_AVAILABLE)
    st.sidebar.write("TEXT_COLS_ENV =", TEXT_COLS_ENV or "(auto)")
    st.sidebar.write("PERSON_COLS_ENV =", PERSON_COLS_ENV or "(auto)")
    st.sidebar.write("ORG_COLS_ENV =", ORG_COLS_ENV or "(auto)")
    st.sidebar.write("LOC_COLS_ENV =", LOC_COLS_ENV or "(auto)")

df = load_data()
# id estable por fila para unir entidades ↔ puntajes
df["row_id"] = np.arange(len(df))

missing = REQUIRED_COLS - set(df.columns)
if missing:
    st.error(T("missing_cols") + ", ".join(sorted(missing)))
    if DEBUG:
        st.write("Columnas presentes:", list(df.columns)); st.dataframe(df.head(20))
    st.stop()

st.title(T("title"))
st.caption(T("subtitle"))

if df.empty:
    st.info(T("no_data")); st.stop()

# Sidebar filters
min_d = pd.to_datetime(df["date"]).min()
max_d = pd.to_datetime(df["date"]).max()
with st.sidebar:
    date_sel = st.date_input(T("date"),
                             value=(min_d.date(), max_d.date()),
                             min_value=min_d.date(), max_value=max_d.date(),
                             key="date_filter")
    if isinstance(date_sel, tuple): start_d, end_d = date_sel
    else: start_d = end_d = date_sel

    sections = sorted(df["section"].dropna().unique().tolist()) if "section" in df.columns else []
    default_secs = sections[:5] if sections else []
    selected_sections = st.multiselect(T("section"), sections, default=default_secs, key="section_filter")

mask = df["date"].between(pd.to_datetime(start_d), pd.to_datetime(end_d))
if selected_sections and "section" in df.columns:
    mask &= df["section"].isin(selected_sections)
dff = df.loc[mask].copy()

if dff.empty:
    st.info(T("no_data")); st.stop()

# Tabs SIEMPRE visibles, cada vista maneja ausencias con mensajes claros
overview_tab, entities_tab, section_tab, headline_tab = st.tabs([
    T("overview"), T("entities_tab"), T("section_tab"), T("headline_tab")
])

with overview_tab:
    sel = sentiment_over_time_view(dff, date_col="date", key_suffix="overview")
    active_col = (sel.get("score_col") if isinstance(sel, dict)
                  else st.session_state.get("active_score_col_overview"))
    if not active_col:
        active_col = _first_present(dff, SCORE_CANDIDATES["all"])

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric(T("kpi_notes"), int(len(dff)))
    mean_sent = dff[active_col].mean() if active_col in dff.columns else float("nan")
    c2.metric(T("kpi_mean"), f"{mean_sent:.3f}" if pd.notna(mean_sent) else "—")
    label_col = st.session_state.get("active_label_col_overview")
    if label_col and label_col in dff.columns:
        pos_ratio = dff[label_col].astype(str).str.lower().str.strip().eq("positive").mean() * 100
    elif active_col in dff.columns:
        pos_ratio = (dff[active_col] > 0).mean() * 100
    else:
        pos_ratio = None
    c3.metric(T("kpi_pos"), f"{pos_ratio:.1f}%" if pos_ratio is not None else "—")

    if active_col in dff.columns:
        distribution_view(dff, active_col)

    keywords_view(dff, key_suffix="overview_kw")

    # Tabla + descarga
    text_cols = pick_text_columns(dff)
    main_text_col = text_cols[0] if text_cols else None
    cols = [c for c in ["date","section", main_text_col, label_col, active_col, "url"] if c and c in dff.columns]
    st.subheader(T("recent"))
    def _color_score(val: float) -> str:
        if val > 0: return "color: #2ca02c"
        if val < 0: return "color: #d62728"
        return "color: #7f7f7f"
    table_df = dff.sort_values("date", ascending=False)[cols].head(300) if cols else dff.head(0)
    if active_col in table_df.columns:
        st.dataframe(table_df.style.applymap(_color_score, subset=[active_col]), use_container_width=True)
    else:
        st.dataframe(table_df, use_container_width=True)
    st.download_button(T("download_csv"),
                       dff.to_csv(index=False).encode("utf-8"),
                       file_name="clarin_sentiment_filtered.csv",
                       mime="text/csv", key="download_filtered_csv")

with entities_tab:
    active_col = st.session_state.get("active_score_col_overview") or _first_present(dff, SCORE_CANDIDATES["all"])
    if active_col not in dff.columns:
        st.info("No hay columnas de puntaje activas para calcular entidades.")
    else:
        entities_view(dff, active_col, key_suffix="ents")

with section_tab:
    coverage_view(dff, key_suffix="cov")

    active_col = st.session_state.get("active_score_col_overview") or _first_present(dff, SCORE_CANDIDATES["all"])
    if HAS_ANALYTICS and hasattr(analytics, "section_avg_sentiment"):
        sec_avg = analytics.section_avg_sentiment(dff.rename(columns={active_col:"sentiment_score"}))
    else:
        sec_avg = _section_avg_sentiment_fallback(dff, active_col)
    if not sec_avg.empty:
        st.plotly_chart(px.bar(sec_avg, x="section", y="avg_sentiment", title=T("avg_by_section")),
                        use_container_width=True)

    if HAS_ANALYTICS and hasattr(analytics, "section_label_counts"):
        sec_counts = analytics.section_label_counts(dff)
    else:
        sec_counts = _section_label_counts_fallback(dff)
    if not sec_counts.empty:
        st.plotly_chart(
            px.bar(sec_counts, x="section", y="count", color="sentiment_label",
                   title=T("counts_by_section"), barmode="stack"),
            use_container_width=True,
        )

with headline_tab:
    active_col = st.session_state.get("active_score_col_overview") or _first_present(dff, SCORE_CANDIDATES["all"])
    text_cols = pick_text_columns(dff)
    main_text_col = text_cols[0] if text_cols else None
    if main_text_col and active_col in dff.columns:
        pos_df = dff.sort_values(active_col, ascending=False).head(10)
        neg_df = dff.sort_values(active_col, ascending=True).head(10)
        col1, col2 = st.columns(2)
        col1.subheader(T("top_pos"))
        col1.dataframe(pos_df[[c for c in ["date","section", main_text_col, active_col] if c in pos_df.columns]])
        col2.subheader(T("top_neg"))
        col2.dataframe(neg_df[[c for c in ["date","section", main_text_col, active_col] if c in neg_df.columns]])
    else:
        st.info(T("no_titles"))
