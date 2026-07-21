import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Análisis de Sentimiento", page_icon="💬", layout="wide")

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

st.title("Análisis de Sentimiento")

st.markdown(
    """
    El análisis de sentimiento asigna a cada personaje un tono predominante
    (positivo/negativo) a partir de su diálogo. Usamos dos enfoques distintos:

    - **VADER**: basado en un diccionario de palabras con carga emocional predefinida,
      da 3 categorías (positivo, negativo, neutro).
    - **DistilBERT**: un modelo de red neuronal (*transformer*) que entiende el
      contexto de la frase, con 2 categorías (positivo, negativo).
    """
)

COLORES_GENERO = {"Masculino": "#2a78d6", "Femenino": "#e87ba4", "Desconocido": "#b0b0b0"}
NOMBRES_GENERO = {"male": "Masculino", "female": "Femenino", "unknown": "Desconocido"}
TRADUCCION_AWARD = {
    "Best Picture": "Mejor Película",
    "Original Screenplay": "Mejor Guion Original",
    "Adapted Screenplay": "Mejor Guion Adaptado",
}
TRADUCCION_AWARD_INV = {v: k for k, v in TRADUCCION_AWARD.items()}

# ----------------------------
# CARGA DE DATOS
# ----------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_pickle("df_sentiment_flat.pkl")
    df["Gender_ES"] = df["Gender"].map(NOMBRES_GENERO)
    return df

df = cargar_datos()

st.divider()

# ----------------------------
# FILTROS
# ----------------------------
st.subheader("🎛️ Filtros")

modelo = st.segmented_control(
    "Modelo de sentimiento",
    options=["VADER", "DistilBERT", "Ambos"],
    default="Ambos",
)

award_seleccionado = st.segmented_control(
    "Categoría de Oscar",
    options=["Todas"] + list(TRADUCCION_AWARD.values()),
    default="Todas",
)

col_a, col_b = st.columns(2)
with col_a:
    generos_seleccionados = st.multiselect(
        "Género",
        options=["Masculino", "Femenino", "Desconocido"],
        default=["Masculino", "Femenino"]
    )
with col_b:
    corte_palabras = st.slider("Palabras mínimas del personaje", 0, int(df["Words"].max()), 20, step=5)

df_filtrado = df[
    (df["Gender_ES"].isin(generos_seleccionados)) &
    (df["Words"] >= corte_palabras)
]
if award_seleccionado and award_seleccionado != "Todas":
    df_filtrado = df_filtrado[df_filtrado["Award"] == TRADUCCION_AWARD_INV[award_seleccionado]]

st.caption(f"Personajes incluidos con estos filtros: **{len(df_filtrado):,}**")

st.divider()

# ----------------------------
# VADER
# ----------------------------
if modelo in ("VADER", "Ambos"):
    st.subheader("VADER")

    col1, col2 = st.columns(2)
    with col1:
        fig_vader_box = px.box(
            df_filtrado, x="Gender_ES", y="Vader_Compound", color="Gender_ES",
            title="Distribución de Vader_Compound por género",
            labels={"Gender_ES": "Género", "Vader_Compound": "Compound (-1 a +1)"},
            color_discrete_map=COLORES_GENERO
        )
        st.plotly_chart(fig_vader_box, width="stretch")
    with col2:
        conteo_vader_label = df_filtrado.groupby(["Gender_ES", "Vader_Label"]).size().reset_index(name="Cantidad")
        fig_vader_label = px.bar(
            conteo_vader_label, x="Gender_ES", y="Cantidad", color="Vader_Label", barmode="group",
            title="Etiqueta VADER (POSITIVE/NEGATIVE/NEUTRAL) por género",
            labels={"Gender_ES": "Género", "Vader_Label": "Etiqueta"}
        )
        st.plotly_chart(fig_vader_label, width="stretch")

    st.markdown("**Proporciones promedio pos/neu/neg de VADER, por género**")
    proporciones_vader = df_filtrado.groupby("Gender_ES")[["Vader_Pos", "Vader_Neu", "Vader_Neg"]].mean().reset_index()
    proporciones_largo = proporciones_vader.melt(id_vars="Gender_ES", var_name="Componente", value_name="Promedio")
    fig_proporciones = px.bar(
        proporciones_largo, x="Gender_ES", y="Promedio", color="Componente", barmode="group",
        title="Proporción promedio de palabras positivas/neutras/negativas (VADER)",
        labels={"Gender_ES": "Género"}
    )
    st.plotly_chart(fig_proporciones, width="stretch")

    st.divider()

# ----------------------------
# DISTILBERT
# ----------------------------
if modelo in ("DistilBERT", "Ambos"):
    st.subheader("DistilBERT")

    col1, col2 = st.columns(2)
    with col1:
        fig_distil_box = px.box(
            df_filtrado, x="Gender_ES", y="Distilbert_Score", color="Gender_ES",
            title="Distribución de Distilbert_Score por género",
            labels={"Gender_ES": "Género", "Distilbert_Score": "Score (-1 a +1)"},
            color_discrete_map=COLORES_GENERO
        )
        st.plotly_chart(fig_distil_box, width="stretch")
    with col2:
        conteo_distil_label = df_filtrado.groupby(["Gender_ES", "Distilbert_Label"]).size().reset_index(name="Cantidad")
        fig_distil_label = px.bar(
            conteo_distil_label, x="Gender_ES", y="Cantidad", color="Distilbert_Label", barmode="group",
            title="Etiqueta DistilBERT (POSITIVE/NEGATIVE) por género",
            labels={"Gender_ES": "Género", "Distilbert_Label": "Etiqueta"}
        )
        st.plotly_chart(fig_distil_label, width="stretch")

    st.divider()

# ----------------------------
# COMPARACIÓN ENTRE MODELOS (solo si se seleccionó "Ambos")
# ----------------------------
if modelo == "Ambos":
    st.subheader("Comparación VADER vs. DistilBERT")
    st.caption(
        "Cada punto es un personaje. Si ambos modelos coincidieran perfectamente, "
        "todos los puntos caerían sobre la diagonal."
    )

    fig_scatter = px.scatter(
        df_filtrado, x="Vader_Compound", y="Distilbert_Score", color="Gender_ES",
        hover_data=["Character", "Title"],
        title="Vader_Compound vs. Distilbert_Score, por personaje",
        labels={"Vader_Compound": "VADER (Compound)", "Distilbert_Score": "DistilBERT (Score)"},
        color_discrete_map=COLORES_GENERO,
        opacity=0.6
    )
    st.plotly_chart(fig_scatter, width="stretch")

    coincidencia = (
        (df_filtrado["Vader_Label"] != "NEUTRAL") &
        (
            ((df_filtrado["Vader_Label"] == "POSITIVE") & (df_filtrado["Distilbert_Label"] == "POSITIVE")) |
            ((df_filtrado["Vader_Label"] == "NEGATIVE") & (df_filtrado["Distilbert_Label"] == "NEGATIVE"))
        )
    )
    total_comparable = (df_filtrado["Vader_Label"] != "NEUTRAL").sum()
    if total_comparable > 0:
        pct_coincidencia = coincidencia.sum() / total_comparable * 100
        st.metric(
            "Coincidencia entre modelos",
            f"{pct_coincidencia:.1f}%",
            help="Solo se comparan personajes donde VADER dio POSITIVE o NEGATIVE (excluye NEUTRAL)."
        )

st.divider()

# ----------------------------
# TABLA: PERSONAJES MÁS POSITIVOS / NEGATIVOS
# ----------------------------
st.subheader("Personajes más extremos")

columna_score = "Vader_Compound" if modelo == "VADER" else "Distilbert_Score"
if modelo == "Ambos":
    columna_score = st.radio("Ordenar según:", ["Vader_Compound", "Distilbert_Score"], horizontal=True)

col_pos, col_neg = st.columns(2)
with col_pos:
    st.markdown("**Más positivos**")
    top_pos = df_filtrado.sort_values(columna_score, ascending=False).head(10)[["Title", "Character", "Gender_ES", columna_score]]
    st.dataframe(top_pos, width="stretch", hide_index=True)
with col_neg:
    st.markdown("**Más negativos**")
    top_neg = df_filtrado.sort_values(columna_score, ascending=True).head(10)[["Title", "Character", "Gender_ES", columna_score]]
    st.dataframe(top_neg, width="stretch", hide_index=True)
