import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Modelado de Tópicos", page_icon="🗂️", layout="wide")

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

st.title("¿De qué hablan los personajes (temas)?")
st.markdown("### Modelado de tópicos")

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

COLORES_SCOPE = {"transversal": "#2a78d6", "specific": "#e8a23a", "outlier": "#b0b0b0"}
NOMBRES_SCOPE = {"transversal": "Transversal", "specific": "Específico", "outlier": "Outlier"}

# ----------------------------
# CARGA DE DATOS
# ----------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("Bertopic_topics_combined_categorized.csv")
    df["Scope_ES"] = df["Scope"].map(NOMBRES_SCOPE)
    return df

df = cargar_datos()

st.divider()

# ----------------------------
# FILTROS
# ----------------------------
st.subheader("🎛️ Filtros")

genero_seleccionado = st.segmented_control(
    "Género del diálogo",
    options=["Masculino", "Femenino"],
    default="Masculino",
)
genero_map = {"Masculino": "male", "Femenino": "female"}

alcance_seleccionado = st.radio(
    "Alcance del tema", ["Todos", "Transversal", "Específico"], horizontal=True
)

df_genero = df[df["Gender"] == genero_map[genero_seleccionado]]
if alcance_seleccionado != "Todos":
    df_genero = df_genero[df_genero["Scope_ES"] == alcance_seleccionado]

st.divider()

# ----------------------------
# MÉTRICAS RESUMEN
# ----------------------------
st.subheader(f"📌 Resumen — diálogo {genero_seleccionado.lower()}")

n_temas = (df[df["Gender"] == genero_map[genero_seleccionado]]["Topic"] != -1).sum()
total_frases = df[df["Gender"] == genero_map[genero_seleccionado]]["Count"].sum()
outliers = df[(df["Gender"] == genero_map[genero_seleccionado]) & (df["Topic"] == -1)]["Count"].sum()
pct_outliers = outliers / total_frases * 100 if total_frases > 0 else 0
n_transversales = (df[(df["Gender"] == genero_map[genero_seleccionado]) & (df["Scope"] == "transversal")]).shape[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Temas encontrados", f"{n_temas}")
col2.metric("Temas transversales", f"{n_transversales}")
col3.metric("Frases totales", f"{int(total_frases):,}")
col4.metric("% Outliers", f"{pct_outliers:.1f}%")

st.divider()

# ----------------------------
# SCATTER: DIVERSIDAD VS. DOMINANCIA (explica transversal/específico)
# ----------------------------
st.subheader("🔍 Diversidad vs. dominancia de película")
st.caption(
    "Cada punto es un tema. Más a la derecha = más películas distintas contribuyen. "
    "Más arriba = una sola película domina el tema (menos generalizable)."
)

df_scatter = df_genero[df_genero["Topic"] != -1]

fig_scatter = px.scatter(
    df_scatter, x="N_Movies", y="Dominant_Movie_Pct", size="Count", color="Scope_ES",
    hover_data=["Category", "Dominant_Movie", "Words"],
    title=f"Temas — {genero_seleccionado}",
    labels={"N_Movies": "Nº de películas", "Dominant_Movie_Pct": "% dominado por 1 película", "Scope_ES": "Alcance"},
    color_discrete_map={"Transversal": "#2a78d6", "Específico": "#e8a23a"}
)
fig_scatter.add_hline(y=50, line_dash="dot", line_color="gray",
                       annotation_text="Umbral 50% (transversal/específico)")
st.plotly_chart(fig_scatter, width="stretch")

st.divider()

# ----------------------------
# TOP CATEGORÍAS POR NÚMERO DE FRASES
# ----------------------------
st.subheader("📊 Categorías con más frases")

top_categorias = df_genero[df_genero["Topic"] != -1].sort_values("Count", ascending=False).head(15)

fig_top_cat = px.bar(
    top_categorias.sort_values("Count"),
    x="Count", y="Category", orientation="h", color="Scope_ES",
    title=f"Top 15 categorías — {genero_seleccionado}",
    labels={"Count": "Número de frases", "Category": "Categoría", "Scope_ES": "Alcance"},
    color_discrete_map={"Transversal": "#2a78d6", "Específico": "#e8a23a"}
)
st.plotly_chart(fig_top_cat, width="stretch")

st.divider()

# ----------------------------
# CATEGORÍAS PRESENTES EN AMBOS GÉNEROS
# ----------------------------
st.subheader("⚖️ Categorías presentes en ambos géneros")
st.caption(
    "Solo unas pocas categorías coinciden textualmente entre los dos modelos "
    "(cada uno se entrenó de forma independiente), pero vale la pena comparar estas."
)

cat_male = set(df[df["Gender"] == "male"]["Category"])
cat_female = set(df[df["Gender"] == "female"]["Category"])
categorias_comunes = sorted(cat_male & cat_female - {"Outliers (no clear topic)"})

if categorias_comunes:
    df_comunes = df[df["Category"].isin(categorias_comunes)].copy()
    df_comunes["Gender_ES"] = df_comunes["Gender"].map({"male": "Masculino", "female": "Femenino"})
    fig_comunes = px.bar(
        df_comunes, x="Category", y="Count", color="Gender_ES", barmode="group",
        title="Comparación de categorías presentes en ambos géneros",
        labels={"Category": "Categoría", "Gender_ES": "Género"},
        color_discrete_map={"Masculino": "#2a78d6", "Femenino": "#e87ba4"}
    )
    st.plotly_chart(fig_comunes, width="stretch")
else:
    st.info("No hay categorías con el mismo nombre exacto en ambos géneros.")

st.divider()

# ----------------------------
# TABLA COMPLETA
# ----------------------------
st.subheader("📋 Tabla completa de temas")

tabla_mostrar = df_genero[df_genero["Topic"] != -1][
    ["Topic", "Category", "Scope_ES", "Count", "N_Movies", "Dominant_Movie", "Dominant_Movie_Pct", "Words"]
].sort_values("Count", ascending=False)

tabla_mostrar = tabla_mostrar.rename(columns={
    "Topic": "ID", "Scope_ES": "Alcance", "Count": "Nº Frases",
    "N_Movies": "Nº Películas", "Dominant_Movie": "Película Dominante",
    "Dominant_Movie_Pct": "% Dominancia", "Words": "Palabras representativas"
})

st.dataframe(tabla_mostrar, width="stretch", hide_index=True)
