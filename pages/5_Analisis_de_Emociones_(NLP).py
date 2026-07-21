import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Análisis de Emociones", page_icon="🎭", layout="wide")

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

st.title("¿Cómo se sienten los personajes (emociones)?")
st.markdown("### Análisis de emociones")

st.markdown(
    """
    Usamos el modelo **`emotion-english-distilroberta-base`** (Hartmann et al.),
    un *transformer* fine-tuned para clasificar texto en **7 emociones básicas**
    (basadas en el modelo de Ekman): alegría, tristeza, miedo, ira, asco, sorpresa
    y neutro. A diferencia de VADER/DistilBERT (solo positivo/negativo), esto
    permite ver **qué tipo** de emoción predomina en cada personaje.
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

EMOCIONES = ["Emotion_Anger", "Emotion_Disgust", "Emotion_Fear", "Emotion_Joy",
             "Emotion_Neutral", "Emotion_Sadness", "Emotion_Surprise"]
NOMBRES_EMOCIONES = {
    "Emotion_Anger": "Ira", "Emotion_Disgust": "Asco", "Emotion_Fear": "Miedo",
    "Emotion_Joy": "Alegría", "Emotion_Neutral": "Neutro",
    "Emotion_Sadness": "Tristeza", "Emotion_Surprise": "Sorpresa",
}

# ----------------------------
# CARGA DE DATOS
# ----------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("df_hartmann_emotions.csv")
    df["Gender_ES"] = df["Gender"].map(NOMBRES_GENERO)
    return df

df = cargar_datos()

st.divider()

# ----------------------------
# FILTROS
# ----------------------------
st.subheader("🎛️ Filtros")
st.caption(
    "Este dataset no tiene un filtro fijo de palabras mínimas — ajusta el corte "
    "tú mismo según lo que quieras analizar."
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
# EMOCIÓN DOMINANTE POR GÉNERO
# ----------------------------
st.subheader("Emoción dominante por género")

conteo_dominante = df_filtrado.groupby(["Gender_ES", "Emotion_Dominant"]).size().reset_index(name="Cantidad")
conteo_dominante["Emotion_Dominant_ES"] = conteo_dominante["Emotion_Dominant"].str.capitalize().map({
    "Anger": "Ira", "Disgust": "Asco", "Fear": "Miedo", "Joy": "Alegría",
    "Neutral": "Neutro", "Sadness": "Tristeza", "Surprise": "Sorpresa"
})

fig_dominante = px.bar(
    conteo_dominante, x="Emotion_Dominant_ES", y="Cantidad", color="Gender_ES", barmode="group",
    title="Número de personajes según su emoción dominante",
    labels={"Emotion_Dominant_ES": "Emoción dominante", "Gender_ES": "Género"},
    color_discrete_map=COLORES_GENERO
)
st.plotly_chart(fig_dominante, width="stretch")

st.divider()

# ----------------------------
# PERFIL EMOCIONAL PROMEDIO: GRÁFICO DE RADAR
# ----------------------------
st.subheader("Perfil emocional promedio por género")
st.caption("Promedio de las 7 probabilidades de emoción, comparado entre géneros.")

promedio_emociones = df_filtrado.groupby("Gender_ES")[EMOCIONES].mean()

fig_radar = go.Figure()
for genero in promedio_emociones.index:
    valores = promedio_emociones.loc[genero, EMOCIONES].tolist()
    valores.append(valores[0])  # cerrar el polígono
    etiquetas = [NOMBRES_EMOCIONES[e] for e in EMOCIONES] + [NOMBRES_EMOCIONES[EMOCIONES[0]]]
    fig_radar.add_trace(go.Scatterpolar(
        r=valores, theta=etiquetas, fill='toself', name=genero,
        line_color=COLORES_GENERO.get(genero, "#888888")
    ))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    title="Perfil emocional promedio por género",
    showlegend=True
)
st.plotly_chart(fig_radar, width="stretch")

st.divider()

# ----------------------------
# COMPARACIÓN DETALLADA POR EMOCIÓN (barras)
# ----------------------------
st.subheader("Comparación detallada, emoción por emoción")

promedio_largo = promedio_emociones.reset_index().melt(
    id_vars="Gender_ES", var_name="Emoción", value_name="Probabilidad promedio"
)
promedio_largo["Emoción"] = promedio_largo["Emoción"].map(NOMBRES_EMOCIONES)

fig_barras_emociones = px.bar(
    promedio_largo, x="Emoción", y="Probabilidad promedio", color="Gender_ES", barmode="group",
    title="Probabilidad promedio de cada emoción, por género",
    color_discrete_map=COLORES_GENERO
)
st.plotly_chart(fig_barras_emociones, width="stretch")

st.divider()

# ----------------------------
# TOP PERSONAJES POR EMOCIÓN ESPECÍFICA
# ----------------------------
st.subheader("Top personajes por emoción")

emocion_elegida = st.selectbox(
    "Elige una emoción",
    options=EMOCIONES,
    format_func=lambda e: NOMBRES_EMOCIONES[e]
)

top_emocion = df_filtrado.sort_values(emocion_elegida, ascending=False).head(10)[
    ["Title", "Character", "Gender_ES", emocion_elegida]
]
top_emocion = top_emocion.rename(columns={emocion_elegida: NOMBRES_EMOCIONES[emocion_elegida]})

col_tabla, col_grafico = st.columns([1, 1])
with col_tabla:
    st.dataframe(top_emocion, width="stretch", hide_index=True)
with col_grafico:
    fig_top = px.bar(
        top_emocion.sort_values(NOMBRES_EMOCIONES[emocion_elegida]),
        x=NOMBRES_EMOCIONES[emocion_elegida], y="Character", orientation="h",
        title=f"Top 10 personajes — {NOMBRES_EMOCIONES[emocion_elegida]}",
        color_discrete_sequence=["#7a5cf0"]
    )
    st.plotly_chart(fig_top, width="stretch")
