import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import mannwhitneyu, chi2_contingency

st.set_page_config(page_title="Significancia Estadística", page_icon="📐", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("¿Las diferencias que observamos son reales o pueden ser fruto del azar?")
st.markdown("### Significancia estadística")

st.markdown(
    """
    Hasta ahora hemos visto **diferencias descriptivas** entre personajes masculinos
    y femeninos (medias, medianas, distribuciones). Esta página responde la pregunta
    que falta: **¿esas diferencias son reales, o podrían deberse al azar?**

    Usamos el **test de Mann-Whitney U** para variables numéricas continuas (no asume
    que los datos siguen una distribución normal, algo que ya vimos que no se cumple
    en nuestros datos) y el **test de Chi-cuadrado** para variables categóricas
    (por ejemplo, si la etiqueta POSITIVE/NEGATIVE se relaciona con el género).

    Un resultado se considera **estadísticamente significativo** si el p-valor es
    menor a 0.05 — es el umbral convencional en ciencias sociales y humanidades.
    """
)

COLORES_GENERO = {"Masculino": "#2a78d6", "Femenino": "#e87ba4"}
NOMBRES_GENERO = {"male": "Masculino", "female": "Femenino"}

# ----------------------------
# CARGA DE DATOS
# ----------------------------
@st.cache_data
def cargar_datos():
    sentimiento = pd.read_pickle("df_sentiment_flat.pkl")
    sentimiento["Gender_ES"] = sentimiento["Gender"].map(NOMBRES_GENERO)

    agencia = pd.read_csv("Spacy_agencia.csv")
    agencia["Gender_ES"] = agencia["Gender"].map(NOMBRES_GENERO)

    emociones = pd.read_csv("df_hartmann_emotions.csv")
    emociones["Gender_ES"] = emociones["Gender"].map(NOMBRES_GENERO)

    return sentimiento, agencia, emociones

df_sentimiento, df_agencia, df_emociones = cargar_datos()

st.divider()

# ----------------------------
# FILTRO GLOBAL DE PALABRAS MÍNIMAS (aplica a sentimiento y emociones)
# ----------------------------
st.subheader("🎛️ Filtro")
corte_palabras = st.slider(
    "Palabras mínimas del personaje (aplica a Sentimiento y Emociones)",
    0, 200, 20, step=5
)
st.caption(
    "El índice de Agencia usa su propio filtro de confiabilidad "
    "(≥5 menciones en primera persona), independiente de este slider."
)

# ----------------------------
# PREPARAR SUBCONJUNTOS MASCULINO/FEMENINO PARA CADA MÉTRICA
# ----------------------------
sent_f = df_sentimiento[
    (df_sentimiento["Gender_ES"].isin(["Masculino", "Femenino"])) &
    (df_sentimiento["Words"] >= corte_palabras)
]
agencia_f = df_agencia[
    (df_agencia["Gender_ES"].isin(["Masculino", "Femenino"])) &
    (df_agencia["Reliable"])
].dropna(subset=["Agency_Index"])
emo_f = df_emociones[
    (df_emociones["Gender_ES"].isin(["Masculino", "Femenino"])) &
    (df_emociones["Words"] >= corte_palabras)
]

def test_mannwhitney(df, columna, gender_col="Gender_ES"):
    m = df[df[gender_col] == "Masculino"][columna].dropna()
    f = df[df[gender_col] == "Femenino"][columna].dropna()
    if len(m) < 2 or len(f) < 2:
        return None
    stat, p = mannwhitneyu(m, f, alternative="two-sided")
    return {
        "Media Masculino": m.mean(), "Media Femenino": f.mean(),
        "N Masculino": len(m), "N Femenino": len(f),
        "p-valor": p
    }

def test_chi2(df, columna, gender_col="Gender_ES"):
    tabla = pd.crosstab(df[gender_col], df[columna])
    chi2, p, dof, expected = chi2_contingency(tabla)
    return {"p-valor": p, "tabla": tabla}

st.divider()

# ----------------------------
# TABLA RESUMEN CONSOLIDADA
# ----------------------------
st.subheader("Resumen consolidado")

metricas_numericas = {
    "Sentimiento — VADER (Compound)": (sent_f, "Vader_Compound"),
    "Sentimiento — DistilBERT (Score)": (sent_f, "Distilbert_Score"),
    "Agencia Narrativa (Índice)": (agencia_f, "Agency_Index"),
    "Emoción — Ira": (emo_f, "Emotion_Anger"),
    "Emoción — Asco": (emo_f, "Emotion_Disgust"),
    "Emoción — Miedo": (emo_f, "Emotion_Fear"),
    "Emoción — Alegría": (emo_f, "Emotion_Joy"),
    "Emoción — Tristeza": (emo_f, "Emotion_Sadness"),
    "Emoción — Sorpresa": (emo_f, "Emotion_Surprise"),
}

filas_resumen = []
for nombre, (df_metrica, columna) in metricas_numericas.items():
    resultado = test_mannwhitney(df_metrica, columna)
    if resultado is None:
        continue
    filas_resumen.append({
        "Métrica": nombre,
        "Media (M)": round(resultado["Media Masculino"], 4),
        "Media (F)": round(resultado["Media Femenino"], 4),
        "N (M)": resultado["N Masculino"],
        "N (F)": resultado["N Femenino"],
        "p-valor": round(resultado["p-valor"], 4),
        "¿Significativo? (p<0.05)": "Sí" if resultado["p-valor"] < 0.05 else "No",
    })

tabla_resumen = pd.DataFrame(filas_resumen)


def resaltar_significativos(fila):
    color = "background-color: #d4edda" if fila["¿Significativo? (p<0.05)"] == "Sí" else ""
    return [color] * len(fila)

st.dataframe(
    tabla_resumen.style.apply(resaltar_significativos, axis=1),
    width="stretch", hide_index=True
)

st.divider()

# ----------------------------
# TESTS DE CHI-CUADRADO (variables categóricas)
# ----------------------------
st.subheader("Variables categóricas (Chi-cuadrado)")

col1, col2, col3 = st.columns(3)

with col1:
    res_vader = test_chi2(sent_f, "Vader_Label")
    st.markdown(f"**VADER Label vs. Género**  \np-valor: `{res_vader['p-valor']:.4f}`")
    st.dataframe(res_vader["tabla"], width="stretch")

with col2:
    res_distil = test_chi2(sent_f, "Distilbert_Label")
    st.markdown(f"**DistilBERT Label vs. Género**  \np-valor: `{res_distil['p-valor']:.4f}`")
    st.dataframe(res_distil["tabla"], width="stretch")

with col3:
    res_emo = test_chi2(emo_f, "Emotion_Dominant")
    st.markdown(f"**Emoción Dominante vs. Género**  \np-valor: `{res_emo['p-valor']:.4f}`")
    st.dataframe(res_emo["tabla"], width="stretch")

st.divider()

# ----------------------------
# DETALLE VISUAL DE CADA MÉTRICA SIGNIFICATIVA
# ----------------------------
st.subheader("Detalle visual por métrica")

metrica_elegida = st.selectbox("Elige una métrica para ver su distribución:", list(metricas_numericas.keys()))
df_detalle, col_detalle = metricas_numericas[metrica_elegida]

fig_detalle = px.box(
    df_detalle, x="Gender_ES", y=col_detalle, color="Gender_ES",
    title=f"Distribución: {metrica_elegida}",
    labels={"Gender_ES": "Género"},
    color_discrete_map=COLORES_GENERO
)
st.plotly_chart(fig_detalle, width="stretch")

st.divider()

# ----------------------------
# CONCLUSIONES
# ----------------------------
st.subheader("Conclusiones generales")

n_significativos = (tabla_resumen["¿Significativo? (p<0.05)"] == "Sí").sum()
n_total = len(tabla_resumen)

st.markdown(
    f"""
    De las **{n_total} métricas numéricas** comparadas entre personajes masculinos
    y femeninos, **{n_significativos}** mostraron una diferencia estadísticamente
    significativa (p < 0.05).

    *(Añade aquí, a mano, la interpretación cualitativa final de tu proyecto: qué
    patrones consistentes encontraste a través de las distintas técnicas —
    sentimiento, agencia, emociones y tópicos —, y qué limitaciones metodológicas
    conviene señalar en la discusión de tu artículo.)*
    """
)
