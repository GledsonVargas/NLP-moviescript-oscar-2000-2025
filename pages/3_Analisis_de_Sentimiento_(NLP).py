import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Análisis de Sentimiento", layout="wide")

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

COLORES_GENERO = {"Masculino": "#2a78d6", "Femenino": "#C1447E", "Desconocido": "#b0b0b0"}
COLORES_ETIQUETA = {"POSITIVE": "#ffc000", "NEGATIVE": "#e7e6e6"}
NOMBRES_GENERO = {"male": "Masculino", "female": "Femenino", "unknown": "Desconocido"}
TRADUCCION_AWARD = {
    "Best Picture": "Mejor Película",
    "Original Screenplay": "Mejor Guion Original",
    "Adapted Screenplay": "Mejor Guion Adaptado",
}
TRADUCCION_AWARD_INV = {v: k for k, v in TRADUCCION_AWARD.items()}
VADER_LABEL_ES = {"POSITIVE": "Positivo", "NEGATIVE": "Negativo", "NEUTRAL": "Neutro"}
DISTIL_LABEL_ES = {"POSITIVE": "Positivo", "NEGATIVE": "Negativo"}

# ----------------------------
# CARGA DE DATOS
# ----------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_pickle("df_sentiment_flat.pkl")
    df["Gender_ES"] = df["Gender"].map(NOMBRES_GENERO)
    return df


# -----------------------------------------------------------------------------
# Carga adicional a nivel película (director/guionista), necesaria para el
# cruce con "equipo creativo". df_sentiment_flat.pkl no trae estas columnas,
# así que se cargan aparte desde dataset_final_75.pkl (mismo patrón que en la
# página de Emociones, aunque ahí el archivo pueda tener otro nombre).
# -----------------------------------------------------------------------------
@st.cache_data
def cargar_datos_peliculas():
    return pd.read_pickle("dataset_final_75.pkl")


df = cargar_datos()
df_peliculas_unicas = cargar_datos_peliculas().drop_duplicates(subset="IMDb_ID")
peliculas_con_dir_f = set(df_peliculas_unicas[df_peliculas_unicas["female_director"] > 0]["Title"])
peliculas_con_guion_f = set(df_peliculas_unicas[df_peliculas_unicas["female_writer"] > 0]["Title"])

st.divider()

# ----------------------------
# FILTROS
# ----------------------------
st.subheader("Filtros")

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
        default=["Masculino", "Femenino", "Desconocido"]
    )
with col_b:
    corte_palabras = st.slider("Palabras mínimas del personaje", 0, int(df["Words"].max()), 0, step=5)



# --- Base: género + award ya aplicados, SIN el corte de palabras todavía ---
# (para poder calcular cuántos personajes se excluyen exactamente por el
# slider, igual que en la página de Estadísticas de Género)
df_base = df[df["Gender_ES"].isin(generos_seleccionados)]
if award_seleccionado and award_seleccionado != "Todas":
    df_base = df_base[df_base["Award"] == TRADUCCION_AWARD_INV[award_seleccionado]]

# 19 de las 56 películas están nominadas en 2 categorías de Oscar, así que
# aparecen duplicadas (una fila por categoría, por personaje) cuando
# "Categoría de Oscar" = Todas. Nos quedamos con una sola fila por
# Título+Personaje para contar cada personaje una única vez (películas
# únicas), no una vez por nominación.
df_base = df_base.drop_duplicates(subset=["Title", "Character"])

conteo_antes = df_base["Gender_ES"].value_counts()

df_filtrado = df_base[df_base["Words"] >= corte_palabras]
conteo_despues = df_filtrado["Gender_ES"].value_counts()

# st.caption(
#     "Cada personaje se cuenta una sola vez, aunque su película esté nominada "
#     "en más de una categoría de Oscar (películas únicas)."
# )
st.caption(f"Personajes incluidos con estos filtros: **{len(df_filtrado):,}**")


def unir_con_y(partes):
    if not partes:
        return ""
    if len(partes) == 1:
        return partes[0]
    return ", ".join(partes[:-1]) + " y " + partes[-1]


partes_exclusion = []
for genero, etiqueta in [
    ("Masculino", "personajes masculinos"),
    ("Femenino", "personajes femeninos"),
    ("Desconocido", "personajes desconocidos"),
]:
    total_antes = conteo_antes.get(genero, 0)
    if total_antes > 0:
        excluidos = total_antes - conteo_despues.get(genero, 0)
        pct = excluidos / total_antes * 100
        partes_exclusion.append(f"**{excluidos:,}** {etiqueta} **({pct:.1f}%)**")

if partes_exclusion:
    st.caption(f"Se excluyen {unir_con_y(partes_exclusion)} con menos de **{corte_palabras}** palabras.")

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
        conteo_vader_label = (
            df_filtrado[df_filtrado["Vader_Label"] != "NEUTRAL"]
            .groupby(["Gender_ES", "Vader_Label"]).size().reset_index(name="Cantidad")
        )
        fig_vader_label = px.bar(
            conteo_vader_label, x="Gender_ES", y="Cantidad", color="Vader_Label", barmode="group",
            title="Etiqueta VADER (POSITIVE/NEGATIVE) por género",
            labels={"Gender_ES": "Género", "Vader_Label": "Etiqueta"},
            color_discrete_map=COLORES_ETIQUETA,
        )
        st.plotly_chart(fig_vader_label, width="stretch")
        n_neutral = (df_filtrado["Vader_Label"] == "NEUTRAL").sum()
        st.caption(f"No incluye los **{n_neutral:,}** personajes con etiqueta NEUTRAL.")

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
            labels={"Gender_ES": "Género", "Distilbert_Label": "Etiqueta"},
            color_discrete_map=COLORES_ETIQUETA,
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

    # --- % de acuerdo entre modelos, desglosado por género ---
    st.markdown("**¿Los modelos coinciden igual para todos los géneros?**")
    st.caption(
        "Mismo cálculo que la métrica de arriba, pero por separado para cada "
        "género (excluye personajes donde VADER dio NEUTRAL, ya que ahí no hay "
        "nada que comparar con DistilBERT)."
    )

    df_comparable = df_filtrado[df_filtrado["Vader_Label"] != "NEUTRAL"].copy()
    df_comparable["Coincide"] = (
        ((df_comparable["Vader_Label"] == "POSITIVE") & (df_comparable["Distilbert_Label"] == "POSITIVE")) |
        ((df_comparable["Vader_Label"] == "NEGATIVE") & (df_comparable["Distilbert_Label"] == "NEGATIVE"))
    )
    acuerdo_genero = (
        df_comparable.groupby("Gender_ES")["Coincide"].mean().reset_index()
    )
    acuerdo_genero["% Coincidencia"] = acuerdo_genero["Coincide"] * 100

    fig_acuerdo = px.bar(
        acuerdo_genero, x="Gender_ES", y="% Coincidencia", color="Gender_ES",
        title="% de acuerdo entre VADER y DistilBERT, por género",
        labels={"Gender_ES": "Género"},
        color_discrete_map=COLORES_GENERO,
        text="% Coincidencia",
    )
    fig_acuerdo.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig_acuerdo.update_layout(showlegend=False, yaxis_range=[0, 100])
    st.plotly_chart(fig_acuerdo, width="stretch")

    st.divider()

# ----------------------------
# EVOLUCIÓN TEMPORAL DEL SENTIMIENTO
# -----------------------------------------------------------------------------
# Media de Vader_Compound y/o Distilbert_Score por año y género, para ver si
# el tono del diálogo ha cambiado desde 2000.
# -----------------------------------------------------------------------------
st.subheader("Evolución temporal del sentimiento")
st.markdown(
    """
    ¿El tono del diálogo (positivo/negativo) se ha mantenido igual entre
    géneros a lo largo de los años, o ha cambiado? Cada punto es la media de
    ese año y género entre los personajes que cumplen los filtros de arriba.
    """
)

columnas_evolucion = []
if modelo in ("VADER", "Ambos"):
    columnas_evolucion.append(("Vader_Compound", "VADER (Compound)"))
if modelo in ("DistilBERT", "Ambos"):
    columnas_evolucion.append(("Distilbert_Score", "DistilBERT (Score)"))

col_evol = st.columns(len(columnas_evolucion)) if len(columnas_evolucion) > 1 else [st]
for col_widget, (columna, etiqueta) in zip(col_evol, columnas_evolucion):
    evolucion = (
        df_filtrado.groupby(["Oscar_Year", "Gender_ES"])[columna]
        .mean()
        .reset_index()
    )
    fig_evolucion = px.line(
        evolucion, x="Oscar_Year", y=columna, color="Gender_ES",
        title=f"Media de {etiqueta} por año y género",
        labels={"Oscar_Year": "Año", columna: etiqueta, "Gender_ES": "Género"},
        color_discrete_map=COLORES_GENERO,
        markers=True,
    )
    col_widget.plotly_chart(fig_evolucion, width="stretch")

st.caption(
    "Algunos años tienen pocos personajes, así que las oscilaciones bruscas de "
    "un año a otro pueden deberse al tamaño de muestra, no a una tendencia real."
)

st.divider()

# ----------------------------
# SENTIMIENTO SEGÚN EQUIPO CREATIVO
# -----------------------------------------------------------------------------
# ¿El tono del diálogo femenino cambia según si hay una mujer en dirección o
# guion? Mismo patrón que en la página de Emociones.
# -----------------------------------------------------------------------------
st.subheader("Sentimiento femenino, según equipo creativo")
st.markdown(
    """
    Nos quedamos solo con los personajes femeninos y los separamos según si su
    película tiene una mujer en dirección o guion, o ninguna de las dos. Así
    vemos si el tono del diálogo femenino cambia con quién dirige o escribe.
    """
)

df_equipo = df_filtrado[df_filtrado["Gender_ES"] == "Femenino"].copy()
if df_equipo.empty:
    st.info("No hay personajes femeninos que cumplan los filtros generales de arriba.")
else:
    df_equipo["Equipo creativo"] = df_equipo["Title"].apply(
        lambda t: "Con mujer en dirección o guion"
        if (t in peliculas_con_dir_f or t in peliculas_con_guion_f)
        else "Sin mujer en dirección ni guion"
    )
    orden_equipo = ["Sin mujer en dirección ni guion", "Con mujer en dirección o guion"]

    col_eq1, col_eq2 = st.columns(2)
    with col_eq1:
        if modelo in ("VADER", "Ambos"):
            fig_eq_vader = px.box(
                df_equipo, x="Equipo creativo", y="Vader_Compound", color="Equipo creativo",
                title="Vader_Compound femenino, según equipo creativo",
                category_orders={"Equipo creativo": orden_equipo},
                color_discrete_sequence=["#b0b0b0", "#C1447E"],
            )
            fig_eq_vader.update_layout(showlegend=False, xaxis_title="")
            st.plotly_chart(fig_eq_vader, width="stretch")
    with col_eq2:
        if modelo in ("DistilBERT", "Ambos"):
            fig_eq_distil = px.box(
                df_equipo, x="Equipo creativo", y="Distilbert_Score", color="Equipo creativo",
                title="Distilbert_Score femenino, según equipo creativo",
                category_orders={"Equipo creativo": orden_equipo},
                color_discrete_sequence=["#b0b0b0", "#C1447E"],
            )
            fig_eq_distil.update_layout(showlegend=False, xaxis_title="")
            st.plotly_chart(fig_eq_distil, width="stretch")

    n_por_equipo = df_equipo["Equipo creativo"].value_counts()
    st.caption(
        "Personajes femeninos en cada grupo: "
        + ", ".join(f"**{g}**: {n_por_equipo.get(g, 0):,}" for g in orden_equipo)
    )

st.divider()

# ----------------------------
# PALABRAS VS. EXTREMIDAD DEL SENTIMIENTO
# -----------------------------------------------------------------------------
# ¿Los personajes con más diálogo tienen un sentimiento más extremo o más
# moderado? VADER y DistilBERT se comportan de forma opuesta aquí.
# -----------------------------------------------------------------------------
st.subheader("Palabras vs. extremidad del sentimiento")
st.markdown(
    """
    ¿Hablar más hace que el tono de un personaje sea más extremo o más
    moderado? Aquí comparamos el número de palabras de cada personaje con el
    valor absoluto de su score (qué tan lejos de cero, sin importar el signo).
    """
)

col_ext1, col_ext2 = st.columns(2)
with col_ext1:
    if modelo in ("VADER", "Ambos"):
        df_ext = df_filtrado.copy()
        df_ext["|Vader_Compound|"] = df_ext["Vader_Compound"].abs()
        fig_ext_vader = px.scatter(
            df_ext, x="Words", y="|Vader_Compound|", color="Gender_ES",
            title="Palabras vs. |Vader_Compound|",
            labels={"Words": "Palabras", "|Vader_Compound|": "|Compound|"},
            color_discrete_map=COLORES_GENERO,
            opacity=0.5,
        )
        st.plotly_chart(fig_ext_vader, width="stretch")
with col_ext2:
    if modelo in ("DistilBERT", "Ambos"):
        df_ext2 = df_filtrado.copy()
        df_ext2["|Distilbert_Score|"] = df_ext2["Distilbert_Score"].abs()
        fig_ext_distil = px.scatter(
            df_ext2, x="Words", y="|Distilbert_Score|", color="Gender_ES",
            title="Palabras vs. |Distilbert_Score|",
            labels={"Words": "Palabras", "|Distilbert_Score|": "|Score|"},
            color_discrete_map=COLORES_GENERO,
            opacity=0.5,
        )
        st.plotly_chart(fig_ext_distil, width="stretch")

st.caption(
    "Si ves la línea de tendencia subir en VADER y bajar en DistilBERT (con "
    "los filtros por defecto), no es un error: son dos formas distintas de "
    "procesar texto largo. VADER acumula carga emocional sobre el texto "
    "completo (a más palabras, compound más extremo); DistilBERT trocea el "
    "diálogo en fragmentos y promedia sus scores (a más palabras/fragmentos, "
    "el promedio tiende a moderarse)."
)

st.divider()

# ----------------------------
# PERSONAJES DESTACADOS Y EQUIPO CREATIVO
# -----------------------------------------------------------------------------
# Sección independiente de los filtros generales de arriba (Género, Award,
# palabras mínimas): cada botón de abajo aplica su propio recorte de datos.
# Las 6 vistas comparten la misma estructura (scatter + coincidencia entre
# modelos + tabla), mostrando siempre como máximo los 10 (o 20) personajes
# con más palabras de cada género para no saturar el gráfico. Las 4 últimas
# vistas cruzan esto con la composición del equipo de dirección/guion a
# nivel película, usando dataset_final_75.pkl (no df_sentiment_flat.pkl).
# -----------------------------------------------------------------------------
st.subheader("Personajes destacados y equipo creativo")
st.markdown(
    """
    Esta sección **no depende de los filtros generales de arriba**: cada botón
    aplica su propio recorte. Los dos primeros muestran a los personajes con
    más palabras de diálogo de cada género (10 o 20 por género, sobre todo el
    dataset); los otros cuatro se quedan solo con las películas cuyo equipo de
    dirección o guion incluye a una mujer, y dentro de ese recorte muestran a
    los 10 personajes con más palabras de cada género (para no saturar el
    gráfico con demasiados puntos).
    """
)

opciones_destacados = [
    "Top 10 por género",
    "Top 20 por género",
    "Top 10 con mujer en dirección o guion",
    "Top 10 con directora mujer",
    "Top 10 con guionista mujer",
    "Top 10 con directora y guionista mujer",
]

opcion_destacados = st.pills(
    "Selecciona una vista",
    options=opciones_destacados,
    selection_mode="single",
    default="Top 10 por género",
    key="pill_destacados",
    label_visibility="collapsed",
)
opcion_destacados = opcion_destacados or "Top 10 por género"


def mostrar_top_personajes(df_personajes, n_top, texto_contexto):
    """
    Muestra, para un subconjunto de personajes ya recortado (df_personajes):
    - los top n_top masculinos y n_top femeninos por palabras,
    - un scatter Palabras vs. Vader_Compound / Distilbert_Score,
    - la coincidencia entre modelos (VADER vs. DistilBERT) para esos personajes,
    - una tabla con el detalle de cada uno.
    """
    df_dedup = df_personajes.drop_duplicates(subset=["Title", "Character"])
    top_m = df_dedup[df_dedup["Gender_ES"] == "Masculino"].sort_values("Words", ascending=False).head(n_top)
    top_f = df_dedup[df_dedup["Gender_ES"] == "Femenino"].sort_values("Words", ascending=False).head(n_top)
    df_top = pd.concat([top_m, top_f])

    st.caption(texto_contexto)

    col_top1, col_top2 = st.columns(2)
    with col_top1:
        if modelo in ("VADER", "Ambos"):
            fig_top_vader = px.scatter(
                df_top, x="Words", y="Vader_Compound", color="Gender_ES",
                hover_data=["Character", "Title"],
                title="Palabras vs. Vader_Compound",
                labels={"Words": "Palabras", "Vader_Compound": "Compound (-1 a +1)", "Gender_ES": "Género"},
                color_discrete_map=COLORES_GENERO,
            )
            st.plotly_chart(fig_top_vader, width="stretch")
    with col_top2:
        if modelo in ("DistilBERT", "Ambos"):
            fig_top_distil = px.scatter(
                df_top, x="Words", y="Distilbert_Score", color="Gender_ES",
                hover_data=["Character", "Title"],
                title="Palabras vs. Distilbert_Score",
                labels={"Words": "Palabras", "Distilbert_Score": "Score (-1 a +1)", "Gender_ES": "Género"},
                color_discrete_map=COLORES_GENERO,
            )
            st.plotly_chart(fig_top_distil, width="stretch")

    coincide_top = (
        (df_top["Vader_Label"] != "NEUTRAL") &
        (
            ((df_top["Vader_Label"] == "POSITIVE") & (df_top["Distilbert_Label"] == "POSITIVE")) |
            ((df_top["Vader_Label"] == "NEGATIVE") & (df_top["Distilbert_Label"] == "NEGATIVE"))
        )
    )
    total_comparable_top = (df_top["Vader_Label"] != "NEUTRAL").sum()
    if total_comparable_top > 0:
        pct_coincidencia_top = coincide_top.sum() / total_comparable_top * 100
        st.metric(
            "Coincidencia entre modelos",
            f"{pct_coincidencia_top:.1f}%",
            help=(
                "Solo se comparan personajes donde VADER dio POSITIVE o NEGATIVE "
                "(excluye NEUTRAL). Calculado sobre los personajes de esta vista."
            ),
        )

    tabla_top = df_top[[
        "Character", "Gender_ES", "Title", "Words", "Vader_Label", "Distilbert_Label"
    ]].copy()
    tabla_top["Vader_Label"] = tabla_top["Vader_Label"].map(VADER_LABEL_ES)
    tabla_top["Distilbert_Label"] = tabla_top["Distilbert_Label"].map(DISTIL_LABEL_ES)
    tabla_top = tabla_top.sort_values(["Gender_ES", "Words"], ascending=[True, False])
    tabla_top.columns = [
        "Personaje", "Género", "Película", "Palabras",
        "Sentimiento (VADER)", "Sentimiento (DistilBERT)",
    ]
    st.dataframe(tabla_top, width="stretch", hide_index=True)


if opcion_destacados in ("Top 10 por género", "Top 20 por género"):
    n_top_general = 10 if opcion_destacados == "Top 10 por género" else 20
    mostrar_top_personajes(
        df, n_top_general,
        f"Los {n_top_general} personajes masculinos y los {n_top_general} femeninos con más "
        f"palabras de diálogo de todo el dataset ({n_top_general * 2} puntos en total). "
        f"No usa el slider de palabras mínimas ni el resto de filtros de arriba.",
    )

else:
    if opcion_destacados == "Top 10 con mujer en dirección o guion":
        peliculas_filtradas = df_peliculas_unicas[
            (df_peliculas_unicas["female_director"] > 0) | (df_peliculas_unicas["female_writer"] > 0)
        ]
        descripcion_equipo = "con al menos una mujer en dirección o en guion"
    elif opcion_destacados == "Top 10 con directora mujer":
        peliculas_filtradas = df_peliculas_unicas[df_peliculas_unicas["female_director"] > 0]
        descripcion_equipo = "con al menos una mujer en la dirección"
    elif opcion_destacados == "Top 10 con guionista mujer":
        peliculas_filtradas = df_peliculas_unicas[df_peliculas_unicas["female_writer"] > 0]
        descripcion_equipo = "con al menos una mujer en el guion"
    else:  # "Top 10 con directora y guionista mujer"
        peliculas_filtradas = df_peliculas_unicas[
            (df_peliculas_unicas["female_director"] > 0) & (df_peliculas_unicas["female_writer"] > 0)
        ]
        descripcion_equipo = "con mujer en dirección y en guion, a la vez"

    titulos_filtrados = set(peliculas_filtradas["Title"])
    df_personajes_equipo = df[df["Title"].isin(titulos_filtrados)]

    if df_personajes_equipo.empty:
        st.info("No hay películas que cumplan esta combinación de filtros.")
    else:
        n_personajes_total = df_personajes_equipo.drop_duplicates(subset=["Title", "Character"]).shape[0]
        mostrar_top_personajes(
            df_personajes_equipo, 10,
            f"{len(peliculas_filtradas)} películas {descripcion_equipo} (de "
            f"{len(df_peliculas_unicas)} en total). Se muestran los 10 personajes "
            f"masculinos y los 10 femeninos con más palabras dentro de ese "
            f"subconjunto (de {n_personajes_total} personajes en total). No usa "
            f"el slider de palabras mínimas ni el resto de filtros de arriba.",
        )

st.divider()