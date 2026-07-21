import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Agencia Narrativa", page_icon="🧭", layout="wide")

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

st.title("¿Cómo actuan los personajes (agencia)?")
st.markdown("### Agencia narrativa")

st.markdown(
    """
    En narratología - na disciplina que estudia cómo funcionan las histórias - 
    **agencia** es la capacidad de un personaje de **tomar deciciones y causar eventos**
    en lugar de ser el receptor de las acciones de otros. Un personaje con agencia alta **mueve la trama**.
    Un personaje con agencia baja es **movido por la trama**. El objetivo es responder: ¿los personajes femeninos hacen cosas o les pasan cosas?"

    El resultado es el Índice de Agencia: de 0 (siempre pasivo/objeto) a 1 (siempre activo/sujeto). Solo se calcula para personajes con al menos 15 palabras de diálogo.
    
    """
)
st.divider()
st.markdown(
    """
    **spaCy** analiza la estructura gramatical de cada frase y etiqueta el rol sintáctico de cada palabra. De esas etiquetas nos interesan dos:

            • nsubj (nominal subject) — el personaje es el sujeto activo del verbo: "She runs", "She decides", "She kills"
    
            • nsubjpass (nominal subject passive) — el personaje es sujeto de una pasiva: "She is saved", "She is told", "She is chosen"
    
    Con eso calculamos para cada personaje un ratio de agencia: agencia = frases donde es nsubj / (nsubj + nsubjpass).Luego, comparamos ese ratio entre personajes masculinos y femeninos, por película y por categoría de premio.
    """
)    


    # Usamos **spaCy** (concretamente su *parser* de dependencias gramaticales) para
    # medir la agencia narrativa: con qué frecuencia un personaje se presenta
    # como sujeto activo ("I did this") frente a objeto/receptor de la acción
    # ("you did this to me") cuando habla en primera persona.
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
    df = pd.read_csv("Spacy_agencia.csv")
    df["Gender_ES"] = df["Gender"].map(NOMBRES_GENERO)
    return df

df = cargar_datos()

st.divider()

# ----------------------------
# FILTROS
# ----------------------------
st.subheader("🎛️ Filtros")

award_seleccionado = st.segmented_control(
    "Categoría de Oscar",
    options=["Todas"] + list(TRADUCCION_AWARD.values()),
    default="Todas",
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    generos_seleccionados = st.multiselect(
        "Género",
        options=["Masculino", "Femenino", "Desconocido"],
        default=["Masculino", "Femenino"]
    )
with col_b:
    solo_confiables = st.checkbox("Solo casos confiables (≥5 menciones en 1ª persona)", value=True)
with col_c:
    min_menciones = st.slider(
        "Mínimo de menciones en 1ª persona", 0,
        int(df["N_FirstPerson_Mentions"].max()), 0
    )

df_filtrado = df[
    (df["Gender_ES"].isin(generos_seleccionados)) &
    (df["N_FirstPerson_Mentions"] >= min_menciones)
]
if solo_confiables:
    df_filtrado = df_filtrado[df_filtrado["Reliable"]]
if award_seleccionado and award_seleccionado != "Todas":
    df_filtrado = df_filtrado[df_filtrado["Award"] == TRADUCCION_AWARD_INV[award_seleccionado]]

df_filtrado = df_filtrado.dropna(subset=["Agency_Index"])

st.caption(f"Personajes incluidos con estos filtros: **{len(df_filtrado):,}**")

st.divider()

# ----------------------------
# MÉTRICAS RESUMEN
# ----------------------------
st.subheader("Índice de agencia promedio por género")

resumen = df_filtrado.groupby("Gender_ES")["Agency_Index"].agg(["mean", "median", "count"]).round(3)
resumen = resumen.rename(columns={"mean": "Media", "median": "Mediana", "count": "N"})

cols = st.columns(len(resumen))
for i, (genero, fila) in enumerate(resumen.iterrows()):
    cols[i].metric(f"{genero} (n={int(fila['N'])})", f"{fila['Media']:.3f}", help=f"Mediana: {fila['Mediana']:.3f}")

st.divider()

# ----------------------------
# DISTRIBUCIÓN DEL ÍNDICE DE AGENCIA
# ----------------------------
st.subheader("Distribución del Índice de Agencia por género")

col1, col2 = st.columns(2)
with col1:
    fig_box = px.box(
        df_filtrado, x="Gender_ES", y="Agency_Index", color="Gender_ES",
        title="Boxplot del Índice de Agencia",
        labels={"Gender_ES": "Género", "Agency_Index": "Índice de Agencia"},
        color_discrete_map=COLORES_GENERO
    )
    st.plotly_chart(fig_box, width="stretch")
with col2:
    fig_hist = px.histogram(
        df_filtrado, x="Agency_Index", color="Gender_ES", barmode="overlay", opacity=0.6,
        title="Histograma del Índice de Agencia",
        labels={"Agency_Index": "Índice de Agencia"},
        color_discrete_map=COLORES_GENERO
    )
    st.plotly_chart(fig_hist, width="stretch")

st.divider()

# ----------------------------
# RELACIÓN ENTRE PALABRAS Y AGENCIA
# ----------------------------
st.subheader("Relación entre palabras del personaje y agencia")

fig_scatter = px.scatter(
    df_filtrado, x="Words", y="Agency_Index", color="Gender_ES",
    hover_data=["Character", "Title"],
    title="Palabras totales vs. Índice de Agencia",
    labels={"Words": "Palabras de diálogo", "Agency_Index": "Índice de Agencia"},
    color_discrete_map=COLORES_GENERO,
    opacity=0.6,
    log_x=True
)
st.plotly_chart(fig_scatter, width="stretch")

st.divider()

# ----------------------------
# PERSONAJES POR CATEGORÍA DE OSCAR
# ----------------------------
st.subheader("Índice de agencia por categoría de Oscar")

resumen_award = df_filtrado.groupby(["Award", "Gender_ES"])["Agency_Index"].mean().reset_index()
fig_award = px.bar(
    resumen_award, x="Award", y="Agency_Index", color="Gender_ES", barmode="group",
    title="Índice de Agencia promedio por categoría de Oscar y género",
    labels={"Award": "Categoría de Oscar", "Agency_Index": "Índice de Agencia promedio"},
    color_discrete_map=COLORES_GENERO
)
st.plotly_chart(fig_award, width="stretch")

st.divider()

# ----------------------------
# TABLA: PERSONAJES MÁS Y MENOS AGENTES
# ----------------------------
st.subheader("Personajes con mayor y menor agencia")

col_alta, col_baja = st.columns(2)
with col_alta:
    st.markdown("**Mayor agencia (más activos)**")
    top_alta = df_filtrado.sort_values("Agency_Index", ascending=False).head(10)[
        ["Title", "Character", "Gender_ES", "Agency_Index", "N_FirstPerson_Mentions"]
    ]
    st.dataframe(top_alta, width="stretch", hide_index=True)
with col_baja:
    st.markdown("**Menor agencia (más pasivos/objeto)**")
    top_baja = df_filtrado.sort_values("Agency_Index", ascending=True).head(10)[
        ["Title", "Character", "Gender_ES", "Agency_Index", "N_FirstPerson_Mentions"]
    ]
    st.dataframe(top_baja, width="stretch", hide_index=True)
