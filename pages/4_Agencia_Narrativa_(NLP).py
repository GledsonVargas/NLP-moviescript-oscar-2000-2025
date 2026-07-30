import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Agencia Narrativa", layout="wide")

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
    En narratología - la disciplina que estudia cómo funcionan las histórias -
    **agencia** es la capacidad de un personaje de **tomar decisiones y causar eventos**
    en lugar de ser el receptor de las acciones de otros. Un personaje con agencia alta **mueve la trama**.
    Un personaje con agencia baja es **movido por la trama**. El objetivo es responder: ¿los personajes femeninos hacen cosas o les pasan cosas?
    Tener en cuenta que **solamente consideramos personajes que se hayan referido a sí mismos por lo menos 5 veces**, para formar un grupo donde podemos
    afirmar con solidez su agencia.

    El resultado es el Índice de Agencia: de 0 (siempre pasivo/objeto) a 1 (siempre activo/sujeto). Solo se calcula para personajes con al menos 15 palabras de diálogo.
    """
)
st.divider()
st.markdown(
    """
    **spaCy** analiza la estructura gramatical de cada frase y etiqueta el rol sintáctico de cada palabra. De esas etiquetas nos interesan dos:

            • nsubj (nominal subject) — el personaje es el sujeto activo del verbo: "She runs", "She decides", "She kills"

            • nsubjpass (nominal subject passive) — el personaje es sujeto de una pasiva: "She is saved", "She is told", "She is chosen"

    Con eso calculamos para cada personaje un ratio de agencia: agencia = frases donde es nsubj / (nsubj + nsubjpass). Luego, comparamos ese ratio entre personajes masculinos y femeninos, por película y por categoría de premio.
    """
)

COLORES_GENERO = {"Masculino": "#2a78d6", "Femenino": "#C1447E", "Desconocido": "#b0b0b0"}
NOMBRES_GENERO = {"male": "Masculino", "female": "Femenino", "unknown": "Desconocido"}
TRADUCCION_AWARD = {
    "Best Picture": "Mejor Película",
    "Original Screenplay": "Mejor Guion Original",
    "Adapted Screenplay": "Mejor Guion Adaptado",
}
TRADUCCION_AWARD_INV = {v: k for k, v in TRADUCCION_AWARD.items()}

OPCIONES_VISTA = [
    "Según nº de menciones",
    "Top 10 por género",
    "Top 20 por género",
    "Top 10 con mujer en dirección o guion",
    "Top 10 con directora mujer",
    "Top 10 con guionista mujer",
    "Top 10 con directora y guionista mujer",
]

# ----------------------------
# CARGA DE DATOS
# ----------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_pickle("spacy_agencia.pkl")
    df["Gender_ES"] = df["Gender"].map(NOMBRES_GENERO)
    return df


# -----------------------------------------------------------------------------
# Carga adicional a nivel película (director/guionista), necesaria para las
# vistas de equipo creativo. spacy_agencia.pkl no trae estas columnas, así
# que se cargan aparte desde dataset_final_75.pkl (mismo patrón que en las
# páginas de Emociones y Sentimiento).
# -----------------------------------------------------------------------------
@st.cache_data
def cargar_datos_peliculas():
    return pd.read_pickle("dataset_final_75.pkl")


df = cargar_datos()
df_peliculas_unicas = cargar_datos_peliculas().drop_duplicates(subset="IMDb_ID")

st.divider()


# -----------------------------------------------------------------------------
# Resuelve, para un gráfico concreto, qué subconjunto de datos usar según la
# vista elegida en su propio pill. "Según nº de menciones" usa el recorte
# que sale del panel de Filtros de arriba (df_filtrado). Las demás vistas
# ignoran ese panel: se quedan con los 10/20 personajes con más palabras de
# cada género, del dataset completo o de un recorte de películas según quién
# dirige/escribe (dataset_final_75.pkl).
# -----------------------------------------------------------------------------
def resolver_vista_agencia(vista, df_filtrado_slider, texto_exclusion):
    if vista == "Según nº de menciones":
        caption = f"Personajes en este recorte: **{len(df_filtrado_slider):,}**. " + texto_exclusion
        return df_filtrado_slider, caption

    # Requisito de validez del dato (no depende del checkbox de arriba):
    # solo personajes confiables (>=5 menciones en 1a persona), que es la
    # condición bajo la que se puede calcular un Índice de Agencia con
    # solidez estadística.
    df_validos = df[df["Reliable"] & df["Agency_Index"].notna()]

    if vista in ("Top 10 por género", "Top 20 por género"):
        n_top = 10 if vista == "Top 10 por género" else 20
        df_pool = df_validos
        contexto_pool = "de todo el dataset"
        n_peliculas_pool = df_pool["Title"].nunique()
    else:
        n_top = 10
        if vista == "Top 10 con mujer en dirección o guion":
            peliculas_filtradas = df_peliculas_unicas[
                (df_peliculas_unicas["female_director"] > 0) | (df_peliculas_unicas["female_writer"] > 0)
            ]
            contexto_pool = "en películas con al menos una mujer en dirección o en guion"
        elif vista == "Top 10 con directora mujer":
            peliculas_filtradas = df_peliculas_unicas[df_peliculas_unicas["female_director"] > 0]
            contexto_pool = "en películas con al menos una mujer en la dirección"
        elif vista == "Top 10 con guionista mujer":
            peliculas_filtradas = df_peliculas_unicas[df_peliculas_unicas["female_writer"] > 0]
            contexto_pool = "en películas con al menos una mujer en el guion"
        else:  # "Top 10 con directora y guionista mujer"
            peliculas_filtradas = df_peliculas_unicas[
                (df_peliculas_unicas["female_director"] > 0) & (df_peliculas_unicas["female_writer"] > 0)
            ]
            contexto_pool = "en películas con mujer en dirección y en guion, a la vez"

        titulos_filtrados = set(peliculas_filtradas["Title"])
        df_pool = df_validos[df_validos["Title"].isin(titulos_filtrados)]
        n_peliculas_pool = len(peliculas_filtradas)

    df_dedup = df_pool.drop_duplicates(subset=["Title", "Character"])
    top_m = df_dedup[df_dedup["Gender_ES"] == "Masculino"].sort_values("Words", ascending=False).head(n_top)
    top_f = df_dedup[df_dedup["Gender_ES"] == "Femenino"].sort_values("Words", ascending=False).head(n_top)
    df_vista = pd.concat([top_m, top_f])

    if vista in ("Top 10 por género", "Top 20 por género"):
        caption = (
            f"Los {n_top} personajes masculinos y los {n_top} femeninos con más "
            f"palabras de diálogo {contexto_pool} ({len(df_vista)} puntos en total; "
            f"de un total de {len(df_pool):,} personajes con Índice de Agencia "
            f"confiable en {n_peliculas_pool} películas). No usa los filtros de arriba."
        )
    else:
        caption = (
            f"**{n_peliculas_pool}** películas {contexto_pool} (de "
            f"**{len(df_peliculas_unicas)}** en total). Se muestran los {n_top} "
            f"personajes masculinos y los {n_top} femeninos con más palabras "
            f"dentro de ese subconjunto (de {len(df_pool):,} personajes con "
            f"Índice de Agencia confiable en total). No usa los filtros de arriba."
        )
    return df_vista, caption


def selector_vista(key):
    """Pill de vista para un gráfico concreto. Devuelve la opción elegida."""
    vista = st.pills(
        "Vista",
        options=OPCIONES_VISTA,
        selection_mode="single",
        default="Según nº de menciones",
        key=key,
        label_visibility="collapsed",
    )
    return vista or "Según nº de menciones"


def mostrar_tabla_personajes_agencia(df_vista):
    """Tabla de detalle para las vistas Top 10/20 y equipo creativo (no se
    muestra en 'Según nº de menciones', que puede tener cientos de filas)."""
    tabla = df_vista[[
        "Character", "Gender_ES", "Title", "Words", "N_FirstPerson_Mentions", "Agency_Index"
    ]].copy()
    tabla["Agency_Index"] = tabla["Agency_Index"].round(3)
    tabla = tabla.sort_values(["Gender_ES", "Words"], ascending=[True, False])
    tabla.columns = [
        "Personaje", "Género", "Película", "Palabras",
        "Menciones en 1ª persona", "Índice de Agencia",
    ]
    st.dataframe(tabla, width="stretch", hide_index=True)


# ----------------------------
# FILTROS
# ----------------------------
st.subheader("Filtros")

award_seleccionado = st.segmented_control(
    "Categoría de Oscar",
    options=["Todas"] + list(TRADUCCION_AWARD.values()),
    default="Todas",
)
st.caption(
    "Con \"Todas\", cada personaje se cuenta una sola vez aunque su película "
    "esté nominada en más de una categoría (algunas películas repiten en 2 "
    "categorías de Oscar; no las contamos dos veces)."
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

# --- Base: género + award ya aplicados, SIN confiabilidad/mínimo de menciones
# todavía (para poder calcular cuántos personajes se excluyen exactamente por
# esos dos filtros) ---
df_base = df[df["Gender_ES"].isin(generos_seleccionados)]
if award_seleccionado and award_seleccionado != "Todas":
    df_base = df_base[df_base["Award"] == TRADUCCION_AWARD_INV[award_seleccionado]]

# Una película nominada en 2 categorías de Oscar aparece dos veces en el
# dataset (una fila por categoría) cuando Award = "Todas". Nos quedamos con
# una sola fila por Título+Personaje para que cada personaje cuente una
# única vez (películas únicas), en vez de tener doble peso en las medias.
df_base = df_base.drop_duplicates(subset=["Title", "Character"])

conteo_antes = df_base["Gender_ES"].value_counts()

# Paso 1: filtros que controla el usuario (checkbox de confiabilidad + slider
# de menciones mínimas).
df_tras_controles = df_base[df_base["N_FirstPerson_Mentions"] >= min_menciones]
if solo_confiables:
    df_tras_controles = df_tras_controles[df_tras_controles["Reliable"]]
conteo_tras_controles = df_tras_controles["Gender_ES"].value_counts()

# Paso 2: exclusión obligatoria, INDEPENDIENTE de los controles de arriba. Un
# personaje con 0 menciones en 1a persona tiene Nsubj_I = 0 y Object_MeUs = 0,
# así que su Índice de Agencia es 0/0 (indefinido, NaN) y no se puede graficar,
# aunque el checkbox esté desactivado y el slider en 0.
df_filtrado = df_tras_controles.dropna(subset=["Agency_Index"])
conteo_despues = df_filtrado["Gender_ES"].value_counts()

total_antes_m = conteo_antes.get("Masculino", 0)
total_antes_f = conteo_antes.get("Femenino", 0)

excl_controles_m = total_antes_m - conteo_tras_controles.get("Masculino", 0)
excl_controles_f = total_antes_f - conteo_tras_controles.get("Femenino", 0)

excl_indefinido_m = conteo_tras_controles.get("Masculino", 0) - conteo_despues.get("Masculino", 0)
excl_indefinido_f = conteo_tras_controles.get("Femenino", 0) - conteo_despues.get("Femenino", 0)
pct_indefinido_m = (excl_indefinido_m / total_antes_m * 100) if total_antes_m > 0 else 0
pct_indefinido_f = (excl_indefinido_f / total_antes_f * 100) if total_antes_f > 0 else 0

st.caption(f"Personajes incluidos con estos filtros: **{len(df_filtrado):,}**")

frases_exclusion = []
if excl_controles_m > 0 or excl_controles_f > 0:
    frases_exclusion.append(
        f"Se excluyen **{excl_controles_m:,}** personajes masculinos y "
        f"**{excl_controles_f:,}** femeninos por el checkbox de confiabilidad "
        f"y/o el mínimo de menciones que elegiste arriba."
    )
frases_exclusion.append(
    f"Independientemente de esos controles, **{excl_indefinido_m:,}** personajes "
    f"masculinos **({pct_indefinido_m:.1f}%)** y **{excl_indefinido_f:,}** femeninos "
    f"**({pct_indefinido_f:.1f}%)** no tienen ninguna mención en primera persona, así "
    f"que su Índice de Agencia queda indefinido (0/0) y siempre se excluyen, aunque "
    f"el checkbox esté desactivado y el slider en 0."
)
texto_exclusion = " ".join(frases_exclusion)
st.caption(texto_exclusion)

st.divider()

# ----------------------------
# MÉTRICAS RESUMEN
# ----------------------------
st.subheader("Índice de agencia promedio por género")

vista_metricas = selector_vista("pill_vista_metricas")
df_vista_metricas, caption_metricas = resolver_vista_agencia(vista_metricas, df_filtrado, texto_exclusion)
st.caption(caption_metricas)

resumen = df_vista_metricas.groupby("Gender_ES")["Agency_Index"].agg(["mean", "median", "count"]).round(3)
resumen = resumen.rename(columns={"mean": "Media", "median": "Mediana", "count": "N"})

if resumen.empty:
    st.info("No hay personajes que cumplan esta combinación de filtros.")
else:
    col_metricas, _ = st.columns([1, 3])
    with col_metricas:
        for genero, fila in resumen.iterrows():
            st.metric(f"{genero} (n={int(fila['N'])})", f"{fila['Media']:.3f}", help=f"Mediana: {fila['Mediana']:.3f}")

    resumen_reset = resumen.reset_index()

    col_media, col_mediana = st.columns(2)
    with col_media:
        fig_media = px.bar(
            resumen_reset, x="Gender_ES", y="Media", color="Gender_ES",
            title="Índice de Agencia — media por género",
            labels={"Gender_ES": "Género", "Media": "Índice de Agencia (media)"},
            color_discrete_map=COLORES_GENERO, text="Media",
        )
        fig_media.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
        fig_media.update_layout(showlegend=False, yaxis_range=[0, 1], xaxis_title="")
        st.plotly_chart(fig_media, width="stretch")
    with col_mediana:
        fig_mediana = px.bar(
            resumen_reset, x="Gender_ES", y="Mediana", color="Gender_ES",
            title="Índice de Agencia — mediana por género",
            labels={"Gender_ES": "Género", "Mediana": "Índice de Agencia (mediana)"},
            color_discrete_map=COLORES_GENERO, text="Mediana",
        )
        fig_mediana.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
        fig_mediana.update_layout(showlegend=False, yaxis_range=[0, 1], xaxis_title="")
        st.plotly_chart(fig_mediana, width="stretch")

    if vista_metricas != "Según nº de menciones":
        mostrar_tabla_personajes_agencia(df_vista_metricas)

st.divider()

# ----------------------------
# DISTRIBUCIÓN DEL ÍNDICE DE AGENCIA
# ----------------------------
st.subheader("Distribución del Índice de Agencia por género")

vista_distribucion = selector_vista("pill_vista_distribucion")
df_vista_distribucion, caption_distribucion = resolver_vista_agencia(vista_distribucion, df_filtrado, texto_exclusion)
st.caption(caption_distribucion)

col1, col2 = st.columns(2)
with col1:
    fig_box = px.box(
        df_vista_distribucion, x="Gender_ES", y="Agency_Index", color="Gender_ES",
        title="Boxplot del Índice de Agencia",
        labels={"Gender_ES": "Género", "Agency_Index": "Índice de Agencia"},
        color_discrete_map=COLORES_GENERO
    )
    # Plotly a veces autoajusta el eje Y por debajo del rango real de los
    # datos (recortando bigotes/outliers). Forzamos el rango para que el
    # boxplot completo quede siempre visible, con un pequeño margen.
    margen_eje = 0.05
    minimo_agencia = df_vista_distribucion["Agency_Index"].min()
    maximo_agencia = df_vista_distribucion["Agency_Index"].max()
    fig_box.update_yaxes(
        range=[max(0, minimo_agencia - margen_eje), min(1, maximo_agencia + margen_eje)]
    )
    st.plotly_chart(fig_box, width="stretch")
with col2:
    fig_hist = px.histogram(
        df_vista_distribucion, x="Agency_Index", color="Gender_ES", barmode="overlay", opacity=0.6,
        title="Histograma del Índice de Agencia",
        labels={"Agency_Index": "Índice de Agencia"},
        color_discrete_map=COLORES_GENERO
    )
    st.plotly_chart(fig_hist, width="stretch")

if vista_distribucion != "Según nº de menciones":
    mostrar_tabla_personajes_agencia(df_vista_distribucion)

st.divider()

# ----------------------------
# RELACIÓN ENTRE PALABRAS Y AGENCIA
# ----------------------------
st.subheader("Relación entre palabras del personaje y agencia")

vista_scatter = selector_vista("pill_vista_scatter")
df_vista_scatter, caption_scatter = resolver_vista_agencia(vista_scatter, df_filtrado, texto_exclusion)
st.caption(caption_scatter)

fig_scatter = px.scatter(
    df_vista_scatter, x="Words", y="Agency_Index", color="Gender_ES",
    hover_data=["Character", "Title"],
    title="Palabras totales vs. Índice de Agencia",
    labels={"Words": "Palabras de diálogo", "Agency_Index": "Índice de Agencia"},
    color_discrete_map=COLORES_GENERO,
    opacity=0.6,
    log_x=True
)
st.plotly_chart(fig_scatter, width="stretch")

if vista_scatter != "Según nº de menciones":
    mostrar_tabla_personajes_agencia(df_vista_scatter)

st.divider()

# ----------------------------
# PERSONAJES POR CATEGORÍA DE OSCAR
# ----------------------------
st.subheader("Índice de agencia por categoría de Oscar")

vista_award = selector_vista("pill_vista_award")
df_vista_award, caption_award = resolver_vista_agencia(vista_award, df_filtrado, texto_exclusion)
st.caption(caption_award)

resumen_award = df_vista_award.groupby(["Award", "Gender_ES"])["Agency_Index"].mean().reset_index()
fig_award = px.bar(
    resumen_award, x="Award", y="Agency_Index", color="Gender_ES", barmode="group",
    title="Índice de Agencia promedio por categoría de Oscar y género",
    labels={"Award": "Categoría de Oscar", "Agency_Index": "Índice de Agencia promedio"},
    color_discrete_map=COLORES_GENERO,
    text="Agency_Index",
)
fig_award.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
st.plotly_chart(fig_award, width="stretch")

if vista_award != "Según nº de menciones":
    mostrar_tabla_personajes_agencia(df_vista_award)

st.divider()