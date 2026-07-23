"""
bertopic_gender_viz.py
-----------------------------------------------------------------------------
Módulo de visualizaciones para la sección BERTopic de la presentación
"NLP y narrativa cinematográfica: Análisis de la representación femenina y
de la evolución discursiva en los guiones ganadores del Óscar (2000-2025)".

Qué resuelve:
  1) ¿Los diálogos masculinos y femeninos hablan de cosas distintas?
     ¿En qué se parecen (convergen) y en qué difieren?
  2) Comparación de "scope" (temas universales/transversales vs. temas
     específicos de una trama concreta).
  3) Exploración jerárquica (treemap) de los tópicos de cada género.
  4) Cruce opcional con el género de dirección/guion de la película
     "dominante" de cada tópico (usando Dataset_final_NLP.pkl).

Cómo integrarlo en tu app Streamlit:

    import streamlit as st
    from bertopic_gender_viz import load_data, render_gender_bertopic_section

    male_df, female_df, combined_df, movies_df = load_data(
        "Bertopic_topics_male_categorized.csv",
        "Bertopic_topics_female_categorized.csv",
        "Bertopic_topics_combined_categorized.csv",
        "Dataset_final_NLP.pkl",
    )
    render_gender_bertopic_section(male_df, female_df, combined_df, movies_df)

También puedes ejecutar este archivo directamente:
    streamlit run bertopic_gender_viz.py
(asumiendo que los 4 archivos de datos están en el mismo directorio)
-----------------------------------------------------------------------------
"""

import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================

@st.cache_data
def load_data(male_path, female_path, combined_path, pkl_path=None):
    """Carga los CSV de tópicos (y opcionalmente el pkl con metadata de películas)."""
    male_df = pd.read_csv(male_path)
    female_df = pd.read_csv(female_path)
    combined_df = pd.read_csv(combined_path)

    movies_df = None
    if pkl_path is not None and Path(pkl_path).exists():
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        # El pkl trae columnas con diccionarios (Script_Dict, Characters_Genders,
        # scores de sentimiento por personaje, etc.). Esas columnas no son
        # "hasheables", y @st.cache_data falla al intentar cachear un DataFrame
        # que las contenga. Nos quedamos solo con lo que necesitamos para el
        # cruce director/guion.
        needed_cols = ["Title", "gender_director", "male_writer", "female_writer"]
        movies_df = raw[needed_cols].copy()
        # Puede haber duplicados de título (una fila por variante); nos quedamos
        # con la primera ocurrencia de cada película para el cruce director/guion.
        movies_df = movies_df.drop_duplicates(subset="Title").set_index("Title")

    return male_df, female_df, combined_df, movies_df


# =============================================================================
# 2. MAPEO A MACRO-TEMAS
# -----------------------------------------------------------------------------
# Las categorías de BERTopic vienen etiquetadas por separado para cada género
# y con nombres ligeramente distintos (p. ej. "Family (father-son)" vs.
# "Family / father relationship"). Para poder comparar "de qué hablan más
# hombres y mujeres" agrupamos las categorías en macro-temas comunes.
# Ajusta este diccionario libremente según el marco teórico de tu tesis/TFM.
# =============================================================================

MACRO_THEME_MAP = {
    # --- Familia y vínculos afectivos ---
    "Family relationships": "Familia y vínculos",
    "Family (father-son)": "Familia y vínculos",
    "Family (mother)": "Familia y vínculos",
    "Marriage / wedding": "Familia y vínculos",
    "Romantic love": "Familia y vínculos",
    "Family / father relationship": "Familia y vínculos",
    "Family / domestic service": "Familia y vínculos",
    "Parenting / protection": "Familia y vínculos",
    "Motherhood / pregnancy": "Familia y vínculos",
    "Romantic relationship": "Familia y vínculos",
    "Love / friendship": "Familia y vínculos",
    "Home / housing": "Familia y vínculos",

    # --- Trabajo y dinero ---
    "Money and business": "Trabajo y dinero",
    "Finance / 2008 crisis": "Trabajo y dinero",
    "Work / young adult life": "Trabajo y dinero",
    "Work / office": "Trabajo y dinero",
    "Household economy": "Trabajo y dinero",
    "Money (rupees, plot-specific)": "Trabajo y dinero",

    # --- Crimen, violencia y justicia ---
    "Race and racism": "Crimen, violencia y justicia",
    "Crime / police": "Crimen, violencia y justicia",
    "Legal system": "Crimen, violencia y justicia",
    "Armed violence": "Crimen, violencia y justicia",
    "Drug trafficking (plot-specific)": "Crimen, violencia y justicia",
    "Specific scene (possible violence)": "Crimen, violencia y justicia",
    "Fear / danger": "Crimen, violencia y justicia",
    "Boxing": "Crimen, violencia y justicia",
    "Nazism / WWII (plot-specific)": "Crimen, violencia y justicia",
    "Slavery (plot-specific)": "Crimen, violencia y justicia",
    "Colony / war (plot-specific)": "Crimen, violencia y justicia",
    "Period piece / slavery (plot-specific)": "Crimen, violencia y justicia",

    # --- Ciencia, tecnología y conocimiento ---
    "Nuclear science": "Ciencia, tecnología y saber",
    "Technology / social media": "Ciencia, tecnología y saber",
    "Literature / writing": "Ciencia, tecnología y saber",
    "Cryptography (plot-specific)": "Ciencia, tecnología y saber",
    "Education / reading": "Ciencia, tecnología y saber",
    "Philosophy / existential": "Ciencia, tecnología y saber",

    # --- Cultura y artes ---
    "Music industry": "Cultura y artes",
    "Film industry": "Cultura y artes",
    "Music": "Cultura y artes",
    "Poetic / aesthetic": "Cultura y artes",
    "Poetic / dreamlike": "Cultura y artes",
    "Biblical language": "Cultura y artes",
    "Epic fantasy (plot-specific)": "Cultura y artes",

    # --- Cuerpo, salud y emoción ---
    "Medicine / health": "Cuerpo, salud y emoción",
    "Intense emotion": "Cuerpo, salud y emoción",
    "Emotion / introspection": "Cuerpo, salud y emoción",
    "Virtue / morality (possible single scene)": "Cuerpo, salud y emoción",
    "Honor / duty (period piece)": "Cuerpo, salud y emoción",

    # --- Vida cotidiana y comunicación ---
    "Food and drink (everyday)": "Vida cotidiana y comunicación",
    "Everyday social life": "Vida cotidiana y comunicación",
    "Everyday objects": "Vida cotidiana y comunicación",
    "Urban life": "Vida cotidiana y comunicación",
    "Routine / sleep": "Vida cotidiana y comunicación",
    "Food / everyday life": "Vida cotidiana y comunicación",
    "Everyday communication": "Vida cotidiana y comunicación",
    "Filler / decision-making": "Vida cotidiana y comunicación",
    "Problem-solving": "Vida cotidiana y comunicación",
    "Gratitude / speech": "Vida cotidiana y comunicación",
    "Accent / speech (plot-specific)": "Vida cotidiana y comunicación",
    "Speech / stammering (plot-specific)": "Vida cotidiana y comunicación",
    "Lies / secrets": "Vida cotidiana y comunicación",
    "Diplomatic rescue (plot-specific)": "Vida cotidiana y comunicación",
    "Conflict / apology": "Vida cotidiana y comunicación",

    # --- Religión y política ---
    "Religion / church": "Religión y política",
    "LGBT political activism": "Religión y política",

    # --- Outliers ---
    "Outliers (no clear topic)": "Sin tema claro (outliers)",
}


def outlier_toggle(label="Outliers", key=None, default_exclude=True):
    """
    Control tipo 'pill' (como los filtros de categoría de Oscar en tu app)
    para incluir/excluir outliers, en vez de un checkbox simple.
    Usa st.segmented_control (Streamlit >= 1.36); si no está disponible en tu
    versión, cae automáticamente a un st.radio horizontal con el mismo efecto.
    """
    options = ["Incluir outliers", "Excluir outliers"]
    default = "Excluir outliers" if default_exclude else "Incluir outliers"

    if hasattr(st, "segmented_control"):
        choice = st.segmented_control(label, options=options, default=default, key=key)
    else:
        choice = st.radio(label, options=options,
                           index=options.index(default), horizontal=True, key=key)

    # Si el usuario deselecciona el pill (segmented_control permite quedar en
    # None), volvemos al comportamiento por defecto en vez de romper el resto.
    if choice is None:
        return default_exclude
    return choice == "Excluir outliers"


def add_macro_theme(df):
    df = df.copy()
    df["Macro_Theme"] = df["Category"].map(MACRO_THEME_MAP).fillna("Otros")
    return df


# =============================================================================
# 2 ter. RESUMEN (KPIs) + DIVERSIDAD VS. DOMINANCIA DE PELÍCULA POR TEMA
# -----------------------------------------------------------------------------
# Sección de arriba del todo de la página: 4 métricas rápidas + un scatter
# donde cada punto es un tema (topic) de BERTopic:
#   - Eje X (N_Movies): en cuántas películas distintas aparece ese tema.
#     Más a la derecha = tema más "repartido" entre películas.
#   - Eje Y (Dominant_Movie_Pct): qué % del tema aporta su película dominante.
#     Más arriba = una sola película concentra casi todo el tema (menos
#     generalizable / más específico de esa trama).
#   - Tamaño del punto: Count (nº de frases del tema).
#   - Color: Scope (transversal vs. específico).
# Los outliers (Topic == -1) se excluyen del scatter, pero sí cuentan para
# el KPI de "% Outliers".
# =============================================================================

def compute_topic_summary_kpis(df):
    n_topics = int((df["Topic"] != -1).sum())
    n_transversal = int((df["Scope"] == "transversal").sum())
    total_phrases = int(df["Count"].sum())
    outlier_count = int(df.loc[df["Topic"] == -1, "Count"].sum())
    pct_outliers = (outlier_count / total_phrases * 100) if total_phrases else 0.0
    return {
        "n_topics": n_topics,
        "n_transversal": n_transversal,
        "total_phrases": total_phrases,
        "pct_outliers": pct_outliers,
    }


def build_diversity_vs_dominance_chart(df, gender_label):
    plot_df = df[df["Topic"] != -1].copy()
    scope_labels = {"transversal": "Transversal", "specific": "Específico"}
    plot_df["Alcance"] = plot_df["Scope"].map(scope_labels).fillna(plot_df["Scope"])

    fig = px.scatter(
        plot_df,
        x="N_Movies", y="Dominant_Movie_Pct", size="Count", color="Alcance",
        color_discrete_map={"Transversal": "#3B6EA5", "Específico": "#E8A33D"},
        hover_name="Category",
        hover_data={
            "N_Movies": True, "Dominant_Movie_Pct": ":.1f", "Count": True,
            "Dominant_Movie": True, "Alcance": False,
        },
        title=f"Temas — {gender_label}",
        size_max=32,
    )
    fig.add_hline(
        y=50, line_dash="dot", line_color="gray",
        annotation_text="Umbral 50% (transversal/específico)",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="Nº de películas",
        yaxis_title="% dominado por 1 película",
        height=480,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig


def render_topic_diversity_summary(df, gender_label, table_key=None):
    kpis = compute_topic_summary_kpis(df)
    st.subheader(f"Resumen — diálogo {gender_label.lower()}", anchor=False)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temas encontrados", kpis["n_topics"])
    c2.metric("Temas transversales", kpis["n_transversal"])
    c3.metric("Frases totales", f"{kpis['total_phrases']:,}")
    c4.metric("% Outliers", f"{kpis['pct_outliers']:.1f}%")
    
    render_full_topics_table(
        df,
        key=table_key or f"tabla_{gender_label.lower()}",
        show_gender_filter=False,
    )
    st.divider()
    st.markdown("#### Diversidad vs. dominancia de película")
    st.markdown(
        "Cada punto es un tema. Más a la derecha = más películas distintas "
        "contribuyen. Más arriba = una sola película domina el tema (menos "
        "generalizable)."
    )
    st.plotly_chart(build_diversity_vs_dominance_chart(df, gender_label), width="stretch")


# =============================================================================
# 2 quater. TABLA COMPLETA DE TEMAS (AMBOS GÉNEROS)
# -----------------------------------------------------------------------------
# Una fila por tema (topic) de BERTopic, para hombres y mujeres a la vez,
# con nombres de columna en español y formato listo para mostrar.
# Pensada para usarse con `combined_df` (el CSV que ya trae ambos géneros
# juntos en una columna 'Gender'), pero también funciona si le pasas solo
# male_df o female_df (en ese caso simplemente no habrá columna Género).
# =============================================================================

def build_full_topics_table(df):
    scope_labels = {"transversal": "Transversal", "specific": "Específico", "outlier": "Outlier"}
    gender_labels = {"male": "Masculino", "female": "Femenino"}

    table = df.copy()
    has_gender = "Gender" in table.columns

    sort_cols = (["Gender", "Topic"] if has_gender else ["Topic"])
    table = table.sort_values(sort_cols).reset_index(drop=True)

    table["Scope"] = table["Scope"].map(scope_labels).fillna(table["Scope"])
    table["Dominant_Movie_Pct"] = table["Dominant_Movie_Pct"].round(1)

    rename_map = {
        "Topic": "ID",
        "Category": "Category",
        "Scope": "Alcance",
        "Count": "Nº Frases",
        "N_Movies": "Nº Películas",
        "Dominant_Movie": "Película Dominante",
        "Dominant_Movie_Pct": "% Dominancia",
        "Words": "Palabras representativas",
    }
    if has_gender:
        table["Gender"] = table["Gender"].map(gender_labels).fillna(table["Gender"])
        rename_map["Gender"] = "Género"

    table = table.rename(columns=rename_map)

    cols = (["Género"] if has_gender else []) + [
        "ID", "Category", "Alcance", "Nº Frases", "Nº Películas",
        "Película Dominante", "% Dominancia", "Palabras representativas",
    ]
    return table[cols]


def render_full_topics_table(df, key=None, show_toggle=True, include_outliers=True,
                              show_gender_filter=True, show_header=True):
    if show_header:
        st.subheader("Tabla completa de temas", anchor=False)

    table = build_full_topics_table(df)
    has_gender = "Género" in table.columns

    if has_gender and not show_gender_filter:
        # Dentro de una pestaña ya dedicada a un género, la columna/filtro de
        # Género es redundante: la quitamos para no repetir información.
        table = table.drop(columns="Género")
        has_gender = False

    if has_gender:
        filt_col1, filt_col2 = st.columns([1, 1])
        with filt_col1:
            gender_choice = st.segmented_control(
                "Género", options=["Todos", "Masculino", "Femenino"],
                default="Todos", key=f"{key}_gender" if key else "topics_table_gender",
            ) or "Todos"
        with filt_col2:
            exclude = outlier_toggle(
                "Categoría", key=f"{key}_toggle" if key else "topics_table_toggle",
                default_exclude=not include_outliers,
            )
    else:
        gender_choice = "Todos"
        exclude = outlier_toggle(
            "Categoría", key=f"{key}_toggle" if key else "topics_table_toggle",
            default_exclude=not include_outliers,
        ) if show_toggle else (not include_outliers)

    if has_gender and gender_choice != "Todos":
        table = table[table["Género"] == gender_choice]
    if exclude:
        table = table[table["ID"] != -1]

    st.dataframe(table, width="stretch", hide_index=True, key=key)


# =============================================================================
# 2 bis. LEYENDA: QUÉ CATEGORÍAS COMPONEN CADA MACRO-TEMA
# =============================================================================

def build_macro_theme_legend(male_df, female_df):
    """
    Devuelve un DataFrame largo (una fila por categoría) con: Macro-tema,
    Género, Categoría, Count y % dentro de ese género. Sirve como tabla de
    referencia para que quede claro qué agrupa cada macro-tema.
    """
    m = add_macro_theme(male_df).assign(Gender="Masculino")
    f = add_macro_theme(female_df).assign(Gender="Femenino")
    both = pd.concat([m, f], ignore_index=True)
    both = both[both["Macro_Theme"] != "Sin tema claro (outliers)"]

    both["pct_within_gender"] = both.groupby("Gender")["Count"].transform(
        lambda s: (s / s.sum() * 100).round(2)
    )

    legend = (
        both[["Macro_Theme", "Gender", "Category", "Count", "pct_within_gender"]]
        .rename(columns={
            "Macro_Theme": "Macro-tema",
            "Gender": "Género",
            "Category": "Categoría original (BERTopic)",
            "Count": "Nº de palabras",
            "pct_within_gender": "% dentro de su género",
        })
        .sort_values(["Macro-tema", "Género", "Nº de palabras"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return legend


def render_macro_theme_legend(male_df, female_df):
    """Muestra la leyenda como expanders (uno por macro-tema) con tabla dentro."""
    legend = build_macro_theme_legend(male_df, female_df)
    with st.expander("📋 Ver qué categorías componen cada macro-tema"):
        for theme in sorted(legend["Macro-tema"].unique()):
            st.markdown(f"**{theme}**")
            subset = legend[legend["Macro-tema"] == theme].drop(columns="Macro-tema")
            st.dataframe(subset, width="stretch", hide_index=True)


# =============================================================================
# 3. GRÁFICO 1 — BARRA DIVERGENTE POR MACRO-TEMA (¿de qué habla cada género?)
# =============================================================================

def build_diverging_theme_chart(male_df, female_df, exclude_outliers=True):
    m = add_macro_theme(male_df)
    f = add_macro_theme(female_df)

    if exclude_outliers:
        m = m[m["Macro_Theme"] != "Sin tema claro (outliers)"]
        f = f[f["Macro_Theme"] != "Sin tema claro (outliers)"]

    m_share = m.groupby("Macro_Theme")["Count"].sum()
    f_share = f.groupby("Macro_Theme")["Count"].sum()

    m_pct = (m_share / m_share.sum() * 100).round(1)
    f_pct = (f_share / f_share.sum() * 100).round(1)

    themes = sorted(set(m_pct.index) | set(f_pct.index),
                     key=lambda t: (m_pct.get(t, 0) + f_pct.get(t, 0)))

    m_vals = [-m_pct.get(t, 0) for t in themes]   # negativo -> se dibuja a la izquierda
    f_vals = [f_pct.get(t, 0) for t in themes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=themes, x=m_vals, name="Diálogo masculino", orientation="h",
        marker_color="#3B6EA5",
        text=[f"{abs(v):.1f}%" for v in m_vals], textposition="outside",
        hovertemplate="%{y}<br>Masculino: %{customdata:.1f}%<extra></extra>",
        customdata=[abs(v) for v in m_vals],
    ))
    fig.add_trace(go.Bar(
        y=themes, x=f_vals, name="Diálogo femenino", orientation="h",
        marker_color="#C1447E",
        text=[f"{v:.1f}%" for v in f_vals], textposition="outside",
        hovertemplate="%{y}<br>Femenino: %{x:.1f}%<extra></extra>",
    ))

    max_val = max(max(abs(v) for v in m_vals), max(f_vals)) * 1.35
    fig.update_layout(
        barmode="overlay",
        title="¿De qué hablan más los diálogos masculinos y femeninos?<br>"
              "<sup>% de palabras de tópicos (excluyendo outliers), por macro-tema</sup>",
        xaxis=dict(title="% del diálogo del género", range=[-max_val, max_val],
                   tickvals=[-max_val*0.66, -max_val*0.33, 0, max_val*0.33, max_val*0.66],
                   ticktext=[f"{max_val*0.66:.0f}%", f"{max_val*0.33:.0f}%", "0",
                             f"{max_val*0.33:.0f}%", f"{max_val*0.66:.0f}%"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0.25),
        height=520,
        margin=dict(l=10, r=10, t=90, b=10),
    )
    return fig


# =============================================================================
# 4. GRÁFICO 2 — PUNTOS DE ENCUENTRO Y DE DIVERGENCIA (categorías compartidas)
# =============================================================================

def build_convergence_table(male_df, female_df, exclude_outliers=True):
    """
    Identifica categorías (nombre exacto) que aparecen en ambos corpus
    (p. ej. 'Legal system', 'Medicine / health') -> puntos de encuentro,
    frente a categorías exclusivas de un género -> puntos de divergencia.

    Por defecto se excluye 'Outliers (no clear topic)' de la comparación:
    al ser, con mucho, la categoría más numerosa en ambos géneros, domina el
    gráfico y no aporta información temática real.
    """
    m = male_df.copy()
    f = female_df.copy()
    m["pct"] = m["Count"] / m["Count"].sum() * 100
    f["pct"] = f["Count"] / f["Count"].sum() * 100

    shared_cats = sorted(set(m["Category"]) & set(f["Category"]))
    if exclude_outliers and "Outliers (no clear topic)" in shared_cats:
        shared_cats.remove("Outliers (no clear topic)")
    rows = []
    for cat in shared_cats:
        rows.append({
            "Categoría": cat,
            "% en diálogo masculino": round(m.loc[m.Category == cat, "pct"].sum(), 2),
            "% en diálogo femenino": round(f.loc[f.Category == cat, "pct"].sum(), 2),
        })
    shared_df = pd.DataFrame(rows).sort_values("% en diálogo femenino", ascending=False)

    male_only = sorted(set(m["Category"]) - set(f["Category"]) - {"Outliers (no clear topic)"})
    female_only = sorted(set(f["Category"]) - set(m["Category"]) - {"Outliers (no clear topic)"})

    return shared_df, male_only, female_only


def build_convergence_chart(shared_df):
    fig = px.bar(
        shared_df.melt(id_vars="Categoría", var_name="Género", value_name="Porcentaje"),
        x="Porcentaje", y="Categoría", color="Género", barmode="group",
        orientation="h",
        color_discrete_map={
            "% en diálogo masculino": "#3B6EA5",
            "% en diálogo femenino": "#C1447E",
        },
        title="Puntos de encuentro: categorías presentes en ambos guiones",
    )
    fig.update_layout(height=max(320, 55 * len(shared_df)), margin=dict(l=10, r=10, t=60, b=10))
    return fig


# =============================================================================
# 5. GRÁFICO 3 — TREEMAPS (exploración jerárquica de tópicos por género)
# =============================================================================

def build_treemap(df, gender_label, color):
    d = add_macro_theme(df)
    d = d[d["Macro_Theme"] != "Sin tema claro (outliers)"]
    fig = px.treemap(
        d,
        path=[px.Constant(gender_label), "Macro_Theme", "Category"],
        values="Count",
        color="Macro_Theme",
        hover_data={"Words": True, "N_Movies": True},
        title=f"Mapa de tópicos — diálogo {gender_label.lower()}",
    )
    fig.update_traces(root_color="lightgrey")
    fig.update_layout(margin=dict(l=5, r=5, t=45, b=5), height=480)
    return fig


# =============================================================================
# 6. GRÁFICO 4 — SCOPE: TEMAS UNIVERSALES vs. ESPECÍFICOS DE TRAMA
# =============================================================================

def build_scope_chart(male_df, female_df, exclude_outliers=False):
    def scope_pct(df, label):
        d = df[df["Scope"] != "outlier"] if exclude_outliers else df
        s = d.groupby("Scope")["Count"].sum()
        pct = (s / s.sum() * 100).round(1)
        return pd.DataFrame({"Género": label, "Scope": pct.index, "Porcentaje": pct.values})

    combined = pd.concat([scope_pct(male_df, "Masculino"), scope_pct(female_df, "Femenino")])
    scope_labels = {"transversal": "Temas universales/transversales",
                    "specific": "Específicos de una trama",
                    "outlier": "Sin tema claro"}
    combined["Scope"] = combined["Scope"].map(scope_labels)

    fig = px.bar(
        combined, x="Género", y="Porcentaje", color="Scope", barmode="stack",
        color_discrete_map={
            "Temas universales/transversales": "#4C9F70",
            "Específicos de una trama": "#C1447E",
            "Sin tema claro": "#B0B0B0",
        },
        title="¿Hablan de temas universales o atados a la trama de su película?",
        text_auto=".1f",
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=60, b=10))
    return fig


# =============================================================================
# 7. GRÁFICO 5 (OPCIONAL) — CRUCE CON GÉNERO DE DIRECCIÓN / GUION
# =============================================================================

def build_director_writer_cross(combined_df, movies_df):
    """
    Cruza cada tópico con el género de dirección/guion de su 'película
    dominante' (la que más contribuye a ese tópico). Es una lectura de
    contexto -- indica en qué tipo de películas (por género de quien dirige/
    escribe) tienden a concentrarse ciertos macro-temas -- no una atribución
    directa de autoría de las líneas de diálogo.
    """
    if movies_df is None:
        return None, None

    d = add_macro_theme(combined_df)
    d = d[d["Macro_Theme"] != "Sin tema claro (outliers)"].copy()

    def lookup(col, title):
        if title in movies_df.index:
            return movies_df.loc[title, col]
        return None

    d["Director_Gender"] = d["Dominant_Movie"].apply(lambda t: lookup("gender_director", t))
    d["Male_Writers"] = d["Dominant_Movie"].apply(lambda t: lookup("male_writer", t))
    d["Female_Writers"] = d["Dominant_Movie"].apply(lambda t: lookup("female_writer", t))

    def writer_profile(row):
        mw, fw = row["Male_Writers"], row["Female_Writers"]
        if pd.isna(mw) and pd.isna(fw):
            return "Sin dato"
        mw, fw = (mw or 0), (fw or 0)
        if fw > 0 and mw > 0:
            return "Guion mixto"
        if fw > 0:
            return "Guion femenino"
        return "Guion masculino"

    d["Writer_Profile"] = d.apply(writer_profile, axis=1)
    d["Director_Gender"] = d["Director_Gender"].map(
        {"M": "Director hombre", "F": "Directora mujer", "U": "Sin dato"}
    ).fillna("Sin dato")

    director_ct = (
        d.groupby(["Gender", "Director_Gender"])["Count"].sum()
        .reset_index()
    )
    writer_ct = (
        d.groupby(["Gender", "Writer_Profile"])["Count"].sum()
        .reset_index()
    )

    gender_label = {"male": "Diálogo masculino", "female": "Diálogo femenino"}
    director_ct["Gender"] = director_ct["Gender"].map(gender_label)
    writer_ct["Gender"] = writer_ct["Gender"].map(gender_label)

    fig_dir = px.bar(
        director_ct, x="Gender", y="Count", color="Director_Gender", barmode="stack",
        title="Palabras de diálogo por género, según quién dirigió su película 'dominante'",
        color_discrete_map={"Director hombre": "#3B6EA5", "Directora mujer": "#C1447E",
                             "Sin dato": "#B0B0B0"},
    )
    fig_dir.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10),
                           yaxis_title="Nº de palabras (Count)", xaxis_title="")

    fig_writer = px.bar(
        writer_ct, x="Gender", y="Count", color="Writer_Profile", barmode="stack",
        title="Palabras de diálogo por género, según composición del equipo de guionistas",
        color_discrete_map={"Guion masculino": "#3B6EA5", "Guion femenino": "#C1447E",
                             "Guion mixto": "#8E6BAE", "Sin dato": "#B0B0B0"},
    )
    fig_writer.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10),
                              yaxis_title="Nº de palabras (Count)", xaxis_title="")

    return fig_dir, fig_writer


# =============================================================================
# 8. RENDER PRINCIPAL PARA STREAMLIT
# =============================================================================

def render_gender_bertopic_section(male_df, female_df, combined_df, movies_df=None):
    st.header("BERTopic: ¿de qué hablan los diálogos masculinos y femeninos?")

    st.markdown(
    """
    **BERTopic** descubre automáticamente los temas presentes en el diálogo,
    combinando *embeddings* semánticos (redes neuronales tipo *transformer*),
    reducción de dimensiones (UMAP) y agrupamiento por densidad (HDBSCAN).
    Entrenamos **dos modelos separados** — uno para diálogo masculino, otro
    para femenino — y luego revisamos cada tema manualmente para asignarle
    una categoría y clasificarlo como:

    - **Transversal**: aparece repartido entre muchas películas distintas (patrón generalizable).
    - **Específico**: dominado por una sola película (refleja su trama particular, no un patrón general).
    """
    )

    
    # --- 0. Resumen + tabla completa + diversidad vs. dominancia (arriba del todo) ---
    tab_m, tab_f = st.tabs(["Diálogo masculino", "Diálogo femenino"])
    with tab_m:
        render_topic_diversity_summary(male_df, "Masculino", table_key="tabla_masc")
    with tab_f:
        render_topic_diversity_summary(female_df, "Femenino", table_key="tabla_fem")

    st.divider()

    st.markdown('#### ¿Y si reunimos por Macro-temas?')
    st.markdown(
        "Cada tópico de BERTopic se agrupó en **macro-temas** para poder "
        "comparar los dos corpus (los nombres de categoría difieren entre "
        "hombres y mujeres, así que agregarlos permite ver el panorama general)."
    )

    # --- 1. Barra divergente ---
    exclude_outliers_1 = outlier_toggle(
        "Categoría", key="toggle_diverging", default_exclude=True
    )
    st.plotly_chart(
        build_diverging_theme_chart(male_df, female_df, exclude_outliers_1),
        width="stretch",
    )

    # --- Leyenda de macro-temas ---
    render_macro_theme_legend(male_df, female_df)

    st.divider()

    # --- 2. Convergencia / divergencia ---
    st.subheader("Puntos de encuentro y de divergencia")
    exclude_outliers_2 = outlier_toggle(
        "Categoría", key="toggle_convergence", default_exclude=True
    )
    shared_df, male_only, female_only = build_convergence_table(
        male_df, female_df, exclude_outliers_2
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(build_convergence_chart(shared_df), width="stretch")
    with col2:
        st.markdown("**Temas exclusivos del diálogo masculino**")
        st.write(", ".join(male_only) if male_only else "—")
        st.markdown("**Temas exclusivos del diálogo femenino**")
        st.write(", ".join(female_only) if female_only else "—")

    st.divider()

    # --- 3. Treemaps ---
    st.subheader("Explora los tópicos de cada género")
    t1, t2 = st.columns(2)
    with t1:
        st.plotly_chart(build_treemap(male_df, "Masculino", "#3B6EA5"), width="stretch")
    with t2:
        st.plotly_chart(build_treemap(female_df, "Femenino", "#C1447E"), width="stretch")

    st.divider()

    # --- 4. Scope ---
    st.subheader("Temas universales vs. específicos de la trama")
    exclude_outliers_3 = outlier_toggle(
        "Categoría", key="toggle_scope", default_exclude=False
    )
    st.plotly_chart(
        build_scope_chart(male_df, female_df, exclude_outliers_3),
        width="stretch",
    )

    # --- 5. Cruce con dirección/guion (opcional) ---
    if movies_df is not None:
        st.divider()
        st.subheader("Contexto: dirección y guion de la película más representativa de cada tópico")
        st.caption(
            "Lectura de contexto, no de autoría directa: indica el género de quien "
            "dirigió/escribió la película que más contribuyó a cada tópico."
        )
        fig_dir, fig_writer = build_director_writer_cross(combined_df, movies_df)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_dir, width="stretch")
        with c2:
            st.plotly_chart(fig_writer, width="stretch")


# =============================================================================
# 9. EJECUCIÓN STANDALONE (streamlit run bertopic_gender_viz.py)
# =============================================================================

def _find_data_dir(filename_to_check="Bertopic_topics_male_categorized.csv"):
    """
    Busca los archivos de datos en varias ubicaciones típicas, en este orden:
      1) La misma carpeta que este script (pages/ si vives en un multipage app)
      2) Una carpeta 'data/' junto a este script
      3) La carpeta padre del script (raíz del proyecto, si pages/ está un nivel abajo)
      4) Una carpeta 'data/' en la raíz del proyecto
    Ajusta esta lista si tu estructura de carpetas es distinta.
    """
    here = Path(__file__).parent
    candidates = [
        here,
        here / "data",
        here.parent,
        here.parent / "data",
    ]
    for c in candidates:
        if (c / filename_to_check).exists():
            return c
    # Si no se encuentra en ningún candidato, devolvemos el primero
    # (el error de FileNotFoundError dirá exactamente dónde buscó).
    return candidates[0]


if __name__ == "__main__":
    st.set_page_config(page_title="NLP y narrativa cinematográfica — BERTopic", layout="wide")

    st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    DATA_DIR = _find_data_dir()

    male_df, female_df, combined_df, movies_df = load_data(
        DATA_DIR / "Bertopic_topics_male_categorized.csv",
        DATA_DIR / "Bertopic_topics_female_categorized.csv",
        DATA_DIR / "Bertopic_topics_combined_categorized.csv",
        DATA_DIR / "Dataset_final_NLP.pkl",
    )
    render_gender_bertopic_section(male_df, female_df, combined_df, movies_df)