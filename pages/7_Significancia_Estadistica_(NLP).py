import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import mannwhitneyu, chi2_contingency, spearmanr
import statsmodels.formula.api as smf

st.set_page_config(page_title="Significancia Estadística", layout="wide")

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

COLORES_GENERO = {"Masculino": "#2a78d6", "Femenino": "#C1447E"}
NOMBRES_GENERO = {"male": "Masculino", "female": "Femenino"}

# ----------------------------
# CARGA DE DATOS
# ----------------------------

# Columnas que DEBEN ser numéricas en cada dataframe. Se fuerzan aquí, una
# sola vez, en vez de en cada test/modelo por separado: si el pickle/csv de
# origen guardó alguna con dtype "object" (texto residual, None mezclado con
# floats, etc.), esto lo corrige antes de que llegue a scipy/statsmodels,
# que fallan de formas distintas y confusas ante columnas no numéricas
# (TypeError en isnan, AttributeError en corrcoef, endog categórico en OLS...).
COLUMNAS_NUMERICAS = {
    "sentimiento": ["Words", "Vader_Compound", "Distilbert_Score"],
    "agencia": ["Words", "Agency_Index"],
    "emociones": [
        "Words", "Emotion_Anger", "Emotion_Disgust", "Emotion_Fear",
        "Emotion_Joy", "Emotion_Sadness", "Emotion_Surprise",
    ],
}


def _forzar_numericas(df, columnas):
    for col in columnas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def cargar_datos():
    sentimiento = pd.read_pickle("df_sentiment_flat.pkl")
    sentimiento["Gender_ES"] = sentimiento["Gender"].map(NOMBRES_GENERO)
    sentimiento = _forzar_numericas(sentimiento, COLUMNAS_NUMERICAS["sentimiento"])

    agencia = pd.read_pickle("spacy_agencia.pkl")
    agencia["Gender_ES"] = agencia["Gender"].map(NOMBRES_GENERO)
    agencia = _forzar_numericas(agencia, COLUMNAS_NUMERICAS["agencia"])

    emociones = pd.read_csv("df_hartmann_emotions.csv")
    emociones["Gender_ES"] = emociones["Gender"].map(NOMBRES_GENERO)
    emociones = _forzar_numericas(emociones, COLUMNAS_NUMERICAS["emociones"])

    return sentimiento, agencia, emociones


# -----------------------------------------------------------------------------
# Carga adicional a nivel película (equipo de dirección/guion), necesaria para
# los modelos de regresión de más abajo. Mismo archivo que en las páginas de
# Emociones, Sentimiento y Agencia.
# -----------------------------------------------------------------------------
@st.cache_data
def cargar_datos_peliculas():
    return pd.read_pickle("dataset_final_75.pkl").drop_duplicates(subset="IMDb_ID")


df_sentimiento, df_agencia, df_emociones = cargar_datos()
df_peliculas = cargar_datos_peliculas()

st.divider()

# ----------------------------
# FILTRO GLOBAL DE PALABRAS MÍNIMAS (aplica a sentimiento y emociones)
# ----------------------------
st.subheader("Filtro")
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
# -----------------------------------------------------------------------------
# 19 de las 56 películas están nominadas en 2 categorías de Oscar, así que sus
# personajes aparecen duplicados (una fila por categoría) en los 3 datasets de
# origen. Deduplicamos por Título+Personaje en los tres para contar cada
# personaje una única vez (película única) — mismo criterio que ya aplican
# las páginas de Sentimiento, Agencia y Emociones por separado.
# -----------------------------------------------------------------------------
sent_f = df_sentimiento[
    (df_sentimiento["Gender_ES"].isin(["Masculino", "Femenino"])) &
    (df_sentimiento["Words"] >= corte_palabras)
].drop_duplicates(subset=["Title", "Character"])
agencia_f = df_agencia[
    (df_agencia["Gender_ES"].isin(["Masculino", "Femenino"])) &
    (df_agencia["Reliable"])
].dropna(subset=["Agency_Index"]).drop_duplicates(subset=["Title", "Character"])
emo_f = df_emociones[
    (df_emociones["Gender_ES"].isin(["Masculino", "Femenino"])) &
    (df_emociones["Words"] >= corte_palabras)
].drop_duplicates(subset=["Title", "Character"])

def test_mannwhitney(df, columna, gender_col="Gender_ES"):
    # Forzamos a numérico: cualquier valor no convertible (texto residual,
    # None mezclado con floats, etc.) se convierte en NaN y se descarta con
    # el dropna(). Esto evita el TypeError de np.isnan cuando la columna
    # llega con dtype "object" en vez de float.
    serie = pd.to_numeric(df[columna], errors="coerce")
    m = serie[df[gender_col] == "Masculino"].dropna()
    f = serie[df[gender_col] == "Femenino"].dropna()
    if len(m) < 2 or len(f) < 2:
        return None
    stat, p = mannwhitneyu(m.to_numpy(dtype=float), f.to_numpy(dtype=float), alternative="two-sided")
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
st.subheader("Resumen consolidado · Mann-Whitney U")

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


def resaltar_columna(df_estilo, columna, valor_resaltado="Sí"):
    def estilo(fila):
        color = "background-color: #d4edda" if fila[columna] == valor_resaltado else ""
        return [color] * len(fila)
    return df_estilo.style.apply(estilo, axis=1)


st.dataframe(
    resaltar_columna(tabla_resumen, "¿Significativo? (p<0.05)"),
    width="stretch", hide_index=True
)

st.divider()

# ----------------------------
# TESTS DE CHI-CUADRADO (variables categóricas)
# ----------------------------
st.subheader("Variables categóricas · Chi-cuadrado")

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
# CORRELACIÓN DE SPEARMAN (PALABRAS VS. OTRAS MÉTRICAS)
# -----------------------------------------------------------------------------
# Mann-Whitney compara grupos y Chi-cuadrado compara categóricas, pero
# ninguno responde si los personajes que hablan MÁS tienden a tener más/menos
# agencia o un sentimiento más extremo — eso es una relación entre DOS
# variables continuas. Spearman mide relación monótona (no asume linealidad
# ni normalidad), lo cual es apropiado porque "Words" está muy sesgada.
# -----------------------------------------------------------------------------
st.subheader("Correlación de Spearman: palabras vs. otras métricas")
st.markdown(
    """
    Mann-Whitney compara grupos (M vs. F) y Chi-cuadrado compara variables
    categóricas, pero ninguno de los dos responde: **¿los personajes que
    hablan más tienden a tener más o menos agencia, o un sentimiento más
    extremo?** Eso es una relación entre **dos variables continuas**
    (palabras y agencia/sentimiento), y para eso usamos el **coeficiente de
    Spearman (ρ)**: mide si la relación es monótona (creciente o decreciente),
    sin asumir que sea lineal — útil aquí porque el número de palabras está
    muy sesgado (pocos personajes hablan muchísimo, la mayoría habla poco).
    """
)


def fila_spearman(nombre, df, col_x, col_y):
    # Igual que en test_mannwhitney: forzamos ambas columnas a numérico para
    # evitar que un dtype "object" residual (texto, None mezclado con
    # floats, etc.) rompa spearmanr/np.corrcoef más abajo.
    datos = df[[col_x, col_y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(datos) < 3:
        return None
    rho, p = spearmanr(datos[col_x].to_numpy(dtype=float), datos[col_y].to_numpy(dtype=float))
    return {
        "Relación": nombre,
        "ρ (rho)": round(rho, 3),
        "p-valor": round(p, 4),
        "N": len(datos),
        "¿Significativo? (p<0.05)": "Sí" if p < 0.05 else "No",
    }


sent_f_spearman = sent_f.copy()
sent_f_spearman["Vader_Compound"] = pd.to_numeric(sent_f_spearman["Vader_Compound"], errors="coerce")
sent_f_spearman["Distilbert_Score"] = pd.to_numeric(sent_f_spearman["Distilbert_Score"], errors="coerce")
sent_f_spearman["|Vader_Compound|"] = sent_f_spearman["Vader_Compound"].abs()
sent_f_spearman["|Distilbert_Score|"] = sent_f_spearman["Distilbert_Score"].abs()

filas_spearman = [
    fila
    for fila in [
        fila_spearman("Palabras vs. Índice de Agencia", agencia_f, "Words", "Agency_Index"),
        fila_spearman("Palabras vs. |Vader_Compound| (extremidad)", sent_f_spearman, "Words", "|Vader_Compound|"),
        fila_spearman("Palabras vs. |Distilbert_Score| (extremidad)", sent_f_spearman, "Words", "|Distilbert_Score|"),
    ]
    if fila is not None
]

tabla_spearman = pd.DataFrame(filas_spearman)
st.dataframe(
    resaltar_columna(tabla_spearman, "¿Significativo? (p<0.05)"),
    width="stretch", hide_index=True
)
st.caption(
    "ρ cercano a 0 = no hay relación monótona; ρ cercano a ±1 = relación "
    "monótona fuerte. El signo indica la dirección: positivo = cuando una "
    "variable sube, la otra tiende a subir; negativo = tiende a bajar. "
    "\"Índice de Agencia\" usa el filtro de confiabilidad propio (no el "
    "slider de arriba); las dos filas de sentimiento sí usan el slider."
)

st.divider()

# =============================================================================
# STATSMODELS: MODELOS DE REGRESIÓN
# =============================================================================
st.header("Regresión con statsmodels: ¿importa quién dirige y quién escribe?")

st.markdown(
    """
    Todo lo de arriba compara **dos grupos a la vez** en **una variable a la vez**
    (¿difiere el sentimiento entre M y F? ¿y la emoción de ira? etc.). Pero no
    responde una pregunta distinta: **¿el género del equipo de dirección/guion
    se relaciona con cómo se representa a los personajes**, y si es así, **¿ese
    efecto es el mismo para personajes masculinos que femeninos?**

    Para esto usamos **[statsmodels](https://www.statsmodels.org/)**, una librería
    de Python para estadística clásica (la misma familia de herramientas que R o
    SPSS). A diferencia de librerías de *machine learning* como scikit-learn
    (pensadas para predecir con la mayor precisión posible), statsmodels está
    pensada para **explicar e interpretar**: te da coeficientes, errores estándar,
    p-valores e intervalos de confianza para cada variable, no solo una predicción.

    **Cómo lo usamos aquí — regresión lineal (OLS, *Ordinary Least Squares*):**
    en vez de comparar M vs. F en una sola tabla, ajustamos un modelo tipo

    ```
    Palabras ~ Género * (directoras_mujeres + directores_hombres + guionistas_mujeres + guionistas_hombres)
    ```

    Esto nos permite tener **varias variables explicativas a la vez** (género del
    personaje, cuántas mujeres/hombres dirigieron, cuántas mujeres/hombres
    escribieron el guion) y ver el efecto de cada una **controlando por las
    demás** — algo que Mann-Whitney y Chi-cuadrado no pueden hacer, porque solo
    trabajan con una variable de entrada.

    **Cómo leer los resultados:**
    - **Coeficiente**: cuánto cambia la métrica (en promedio) por cada unidad
      adicional de esa variable (por ejemplo, por cada directora mujer más),
      manteniendo el resto constante.
    - **p-valor**: la probabilidad de ver un coeficiente así de grande (o más)
      si en realidad no hubiera ningún efecto. Menor a 0.05 → lo llamamos
      "estadísticamente significativo".
    - **R²**: qué proporción de la variación total explica el modelo (de 0 a 1).
      En datos de ciencias sociales/humanidades, valores de 0.01–0.10 son
      normales y no indican que el modelo esté "mal" — el comportamiento humano
      es difícil de explicar con pocas variables.
    - Los términos con `:` (por ejemplo `Género[Masculino]:directoras_mujeres`)
      son **interacciones**: miden si el efecto de "tener una directora mujer"
      es distinto para personajes masculinos que femeninos.

    Con solo **56 películas**, estos modelos tienen poca potencia estadística
    (es fácil que un efecto real no llegue a p<0.05), así que interpreta los
    p-valores altos como "no hay evidencia suficiente todavía", no como
    "seguro que no hay relación".
    """
)

st.caption(
    "Sobre los nombres de las tablas de abajo: **\"Gender_ES\"** es solo el "
    "nombre de nuestra columna con el género traducido al español (Masculino/"
    "Femenino); el \"_ES\" no tiene ningún significado estadístico, es solo "
    "nuestra convención de nombres. Para poder comparar, statsmodels toma "
    "**Femenino como categoría de referencia**: por eso el Intercept representa "
    "a un personaje femenino, y \"Masculino (vs. Femenino)\" es cuánto cambia "
    "al pasar de Femenino a Masculino. Cuando ves dos nombres unidos por "
    "**\"×\"** (por ejemplo \"Masculino × Guionistas (mujeres)\"), es una "
    "*interacción*: mide si el efecto de esa variable del equipo creativo es "
    "distinto para personajes masculinos que para femeninos."
)


# -----------------------------------------------------------------------------
# statsmodels nombra las variables con su nombre de columna tal cual ("Gender_ES"
# es solo nuestra columna con Género traducido a español, no tiene significado
# estadístico especial). Aquí traducimos esos nombres técnicos a etiquetas
# legibles para las tablas: quitamos "Gender_ES[T....]", usamos "×" para
# interacciones en vez de ":", y damos nombres más claros al equipo creativo.
# -----------------------------------------------------------------------------
ETIQUETAS_VARIABLES = {
    "directoras_mujeres": "Directoras (mujeres)",
    "directores_hombres": "Directores (hombres)",
    "guionistas_mujeres": "Guionistas (mujeres)",
    "guionistas_hombres": "Guionistas (hombres)",
}


def etiqueta_variable(nombre, es_parte_de_interaccion=False):
    if nombre == "Intercept":
        return "Intercept (caso base: personaje Femenino, equipo creativo = 0)"
    if nombre.startswith("Gender_ES[T.") and nombre.endswith("]"):
        genero = nombre[len("Gender_ES[T."):-1]
        return genero if es_parte_de_interaccion else f"{genero} (vs. Femenino)"
    return ETIQUETAS_VARIABLES.get(nombre, nombre.replace("_", " ").capitalize())


def nombre_legible(variable_tecnica):
    if ":" in variable_tecnica:
        partes = variable_tecnica.split(":")
        return " × ".join(etiqueta_variable(p, es_parte_de_interaccion=True) for p in partes)
    return etiqueta_variable(variable_tecnica)


def tabla_coeficientes(modelo):
    """Tabla legible de coeficientes/p-valores/IC para cualquier modelo OLS."""
    tabla = pd.DataFrame({
        "Coeficiente": modelo.params,
        "Error estándar": modelo.bse,
        "p-valor": modelo.pvalues,
        "IC 95% (inf)": modelo.conf_int()[0],
        "IC 95% (sup)": modelo.conf_int()[1],
    }).round(4)
    tabla["¿Significativo? (p<0.05)"] = tabla["p-valor"].apply(lambda p: "Sí" if p < 0.05 else "No")
    tabla = tabla.reset_index().rename(columns={"index": "Término técnico"})
    tabla.insert(0, "Variable", tabla["Término técnico"].apply(nombre_legible))
    tabla = tabla.drop(columns=["Término técnico"])
    return tabla


# -----------------------------------------------------------------------------
# Variables de equipo creativo a nivel película: conteos de directores/as y
# guionistas por género (ya vienen así en dataset_final_75.pkl).
# -----------------------------------------------------------------------------
df_peliculas_modelo = df_peliculas.rename(columns={
    "female_director": "directoras_mujeres",
    "male_director": "directores_hombres",
    "female_writer": "guionistas_mujeres",
    "male_writer": "guionistas_hombres",
}).copy()
df_peliculas_modelo["pct_personajes_femeninos"] = (
    df_peliculas_modelo["Female_Characters_Count"] /
    (df_peliculas_modelo["Female_Characters_Count"] + df_peliculas_modelo["Male_Characters_Count"])
)
df_peliculas_modelo["gap_palabras_F_menos_M"] = (
    df_peliculas_modelo["AverageWords_female"] - df_peliculas_modelo["AverageWords_male"]
)

COLUMNAS_EQUIPO = ["Title", "directoras_mujeres", "directores_hombres", "guionistas_mujeres", "guionistas_hombres"]
FORMULA_EQUIPO = "directoras_mujeres + directores_hombres + guionistas_mujeres + guionistas_hombres"

# ----------------------------
# MODELO 1 (a nivel película): ¿el equipo creativo se relaciona con CUÁNTOS
# personajes femeninos hay?
# ----------------------------
st.subheader("Equipo creativo y proporción de personajes femeninos")
st.caption(
    "Nivel: película (56 filas, una por película). Variable dependiente: "
    "% de personajes que son femeninos, de todos los personajes con género "
    "conocido de esa película."
)

modelo_representacion = smf.ols(
    f"pct_personajes_femeninos ~ {FORMULA_EQUIPO}", data=df_peliculas_modelo
).fit()
st.dataframe(tabla_coeficientes(modelo_representacion), width="stretch", hide_index=True)
st.caption(f"N = {int(modelo_representacion.nobs)} películas · R² = {modelo_representacion.rsquared:.3f}")

st.divider()

# ----------------------------
# MODELO 2 (a nivel película): ¿el equipo creativo se relaciona con la
# BRECHA de palabras entre personajes femeninos y masculinos?
# ----------------------------
st.subheader("Equipo creativo y brecha de palabras (Femenino − Masculino)")
st.caption(
    "Nivel: película. Variable dependiente: palabras promedio de los "
    "personajes femeninos menos palabras promedio de los masculinos (valores "
    "negativos = los personajes femeninos hablan menos, en promedio)."
)

modelo_brecha_palabras = smf.ols(
    f"gap_palabras_F_menos_M ~ {FORMULA_EQUIPO}", data=df_peliculas_modelo
).fit()
st.dataframe(tabla_coeficientes(modelo_brecha_palabras), width="stretch", hide_index=True)
st.caption(f"N = {int(modelo_brecha_palabras.nobs)} películas · R² = {modelo_brecha_palabras.rsquared:.3f}")

st.divider()

# ----------------------------
# MODELO 3 (a nivel personaje): Palabras del personaje ~ Género × equipo
# creativo
# ----------------------------
st.subheader("Palabras del personaje, según género y equipo creativo")
st.caption(
    "Nivel: personaje (usa el mismo recorte de Sentimiento, con el slider de "
    "arriba). Variable dependiente: palabras de diálogo del personaje. Se "
    "cruzan Género y equipo creativo con interacciones, para ver si el efecto "
    "de tener una directora/guionista mujer es distinto para personajes "
    "masculinos que femeninos."
)

sent_equipo = sent_f.merge(
    df_peliculas_modelo[COLUMNAS_EQUIPO], on="Title", how="left"
)

modelo_palabras_personaje = smf.ols(
    f"Words ~ Gender_ES * ({FORMULA_EQUIPO})", data=sent_equipo
).fit()
st.dataframe(tabla_coeficientes(modelo_palabras_personaje), width="stretch", hide_index=True)
st.caption(
    f"N = {int(modelo_palabras_personaje.nobs)} personajes · "
    f"R² = {modelo_palabras_personaje.rsquared:.3f}"
)

st.divider()

# ----------------------------
# MODELO 4 (a nivel personaje, consolidado): Sentimiento y emociones ~
# Género × equipo creativo
# ----------------------------
st.subheader("Sentimiento y emociones, según género y equipo creativo")
st.caption(
    "Mismo tipo de modelo que arriba (Género × equipo creativo, con "
    "interacciones), repetido para cada métrica de sentimiento y emoción. "
    "La tabla solo muestra los 4 efectos principales del equipo creativo "
    "(sin las interacciones); usa el selector de más abajo para ver el "
    "modelo completo de cualquiera de ellas."
)

emo_equipo = emo_f.merge(df_peliculas_modelo[COLUMNAS_EQUIPO], on="Title", how="left")

metricas_para_regresion = {
    "Sentimiento — VADER (Compound)": (sent_equipo, "Vader_Compound"),
    "Sentimiento — DistilBERT (Score)": (sent_equipo, "Distilbert_Score"),
    "Emoción — Ira": (emo_equipo, "Emotion_Anger"),
    "Emoción — Asco": (emo_equipo, "Emotion_Disgust"),
    "Emoción — Miedo": (emo_equipo, "Emotion_Fear"),
    "Emoción — Alegría": (emo_equipo, "Emotion_Joy"),
    "Emoción — Tristeza": (emo_equipo, "Emotion_Sadness"),
    "Emoción — Sorpresa": (emo_equipo, "Emotion_Surprise"),
}


modelos_equipo = {}
for nombre, (df_metrica, columna) in metricas_para_regresion.items():
    datos_modelo = df_metrica.dropna(subset=[columna])
    if datos_modelo["Gender_ES"].nunique() < 2:
        continue
    modelos_equipo[nombre] = smf.ols(
        f"{columna} ~ Gender_ES * ({FORMULA_EQUIPO})", data=datos_modelo
    ).fit()

TERMINOS_EQUIPO = [
    ("Directoras (mujeres)", "directoras_mujeres"),
    ("Directores (hombres)", "directores_hombres"),
    ("Guionistas (mujeres)", "guionistas_mujeres"),
    ("Guionistas (hombres)", "guionistas_hombres"),
]

filas_equipo = []
for nombre, modelo in modelos_equipo.items():
    fila = {"Métrica": nombre, "N": int(modelo.nobs), "R²": round(modelo.rsquared, 3)}
    p_vals = []
    for etiqueta, termino in TERMINOS_EQUIPO:
        if termino in modelo.params.index:
            coef = modelo.params[termino]
            p = modelo.pvalues[termino]
            fila[etiqueta] = f"{coef:.3f} (p={p:.3f})"
            p_vals.append(p)
        else:
            fila[etiqueta] = "—"
    fila["¿Algún efecto significativo? (p<0.05)"] = "Sí" if any(p < 0.05 for p in p_vals) else "No"
    filas_equipo.append(fila)

tabla_equipo_regresion = pd.DataFrame(filas_equipo)
st.dataframe(
    resaltar_columna(tabla_equipo_regresion, "¿Algún efecto significativo? (p<0.05)"),
    width="stretch", hide_index=True
)

st.markdown("**Ver el modelo completo (con interacciones) de una métrica:**")
metrica_modelo_completo = st.selectbox(
    "Elige una métrica para ver su modelo completo:",
    list(modelos_equipo.keys()), key="selector_modelo_completo"
)
st.dataframe(
    tabla_coeficientes(modelos_equipo[metrica_modelo_completo]),
    width="stretch", hide_index=True
)

st.divider()

# ----------------------------
# CONCLUSIONES
# ----------------------------
st.subheader("Conclusiones generales")

n_significativos = (tabla_resumen["¿Significativo? (p<0.05)"] == "Sí").sum()
n_total = len(tabla_resumen)
n_significativos_equipo = (tabla_equipo_regresion["¿Algún efecto significativo? (p<0.05)"] == "Sí").sum()
n_total_equipo = len(tabla_equipo_regresion)

st.markdown(
    f"""
    De las **{n_total} métricas numéricas** comparadas entre personajes masculinos
    y femeninos (Mann-Whitney), **{n_significativos}** mostraron una diferencia
    estadísticamente significativa (p < 0.05).

    De los **{n_total_equipo} modelos de regresión** que cruzan sentimiento/emoción
    con el género del equipo de dirección y guion, **{n_significativos_equipo}**
    mostraron al menos un efecto estadísticamente significativo del equipo
    creativo (p < 0.05).
    """
)