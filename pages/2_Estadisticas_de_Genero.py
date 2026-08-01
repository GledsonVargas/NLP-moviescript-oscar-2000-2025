import re
import pickle

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Estadísticas de Género", layout="wide")

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


def divider_grueso(color="#0A0A0A", grosor_px=4):
    """Un st.divider() más grueso que el estándar, para separar secciones clave."""
    st.markdown(
        f"<hr style='height:{grosor_px}px;border:none;background-color:{color};"
        f"border-radius:2px;margin-top:1.2rem;margin-bottom:1.2rem;'>",
        unsafe_allow_html=True,
    )


st.title("1. Estadísticas de Género")
st.markdown(
    """
    Esta página explora cómo se reparte el diálogo entre personajes masculinos y
    femeninos: cuántos personajes hay de cada género, cuántas palabras hablan,
    y cómo varía esto según la categoría de Oscar y si la dirección o guion está escrito por diferente género.
    """
)

COLORES_GENERO = {"male": "#3B6EA5", "female": "#C1447E", "unknown": "#b0b0b0"}
NOMBRES_GENERO = {"male": "Masculino", "female": "Femenino", "unknown": "Desconocido"}
COLORES_GENERO_ES = {"Masculino": "#3B6EA5", "Femenino": "#C1447E", "Desconocido": "#b0b0b0"}

# ----------------------------
# CARGA DE DATOS
# ----------------------------

def normalize_name(name):
    """
    Unifica variantes de apóstrofe (' vs ' vs ') y espacios repetidos, para
    que el nombre de personaje en Script_Dict empareje correctamente con el
    de Characters_Genders (sin esto, algunos personajes con apóstrofe caen
    silenciosamente en género "unknown" aunque sí tengan género asignado).
    """
    name = name.replace("’", "'").replace("‘", "'")
    name = re.sub(r"\s+", " ", name).strip()
    return name.upper()


@st.cache_data
def cargar_datos():
    with open("dataset_final_75.pkl", "rb") as f:
        df_raw = pickle.load(f)

    rows = []
    for _, row in df_raw.iterrows():
        script = row["Script_Dict"]
        genders = row["Characters_Genders"]
        if not isinstance(script, dict):
            continue
        genders = genders if isinstance(genders, dict) else {}
        genders_norm = {normalize_name(k): v for k, v in genders.items()}
        for character, text in script.items():
            n_words = len(str(text).split())
            gender_raw = genders_norm.get(normalize_name(character), "unknown")
            rows.append({
                "IMDb_ID": row["IMDb_ID"],
                "Title": row["Title"],
                "Oscar_Year": row["Oscar_Year"],
                "Award": row["Award"],
                "Character": character,
                "Gender": gender_raw,
                "Words": n_words,
            })

    df = pd.DataFrame(rows)
    df = df[df["Words"] > 0].reset_index(drop=True)
    df["Gender_ES"] = df["Gender"].map(NOMBRES_GENERO)
    return df


df = cargar_datos()

# ----------------------------
# CARGA DE DATOS A NIVEL PELÍCULA (director/guionista)
# -----------------------------------------------------------------------------
# Varias secciones de esta página (Directores/guionistas, Relación de género,
# Una mirada femenina, y el gráfico de "Personajes por categoría de Oscar")
# necesitan columnas que solo existen en dataset_final_75 "en crudo"
# (male_director, female_director, male_writer, female_writer, Director,
# Writers...), no en la tabla `df` aplanada por personaje. Por eso se carga
# aparte, con nombres distintos (df_peliculas / df_peliculas_unicas) para no
# pisar el `df` de arriba.
# -----------------------------------------------------------------------------
@st.cache_data
def cargar_datos_peliculas():
    return pd.read_pickle("dataset_final_75.pkl")


df_peliculas = cargar_datos_peliculas()
df_peliculas_unicas = df_peliculas.drop_duplicates(subset="IMDb_ID")

n_peliculas_unicas_total = df["IMDb_ID"].nunique()
n_nominaciones_total = df[["Title", "Award"]].drop_duplicates().shape[0]

st.markdown(
    f"Nota: esta página usa {n_peliculas_unicas_total} películas únicas "
    f"({n_nominaciones_total} nominaciones en total, ya que algunas películas "
    f"están nominadas en más de una categoría de Oscar). El total de personajes "
    f"aquí es {df.drop_duplicates(subset=['Title', 'Character']).shape[0]:,} "
    f"(películas únicas) o {df.shape[0]:,} (contando cada nominación por separado)."
)

st.divider()

# ----------------------------
# FILTRO INTERACTIVO POR NÚMERO MÍNIMO DE PALABRAS
# ----------------------------
st.subheader("Personajes por número mínimo de palabras")
st.caption(
    "Ajusta el umbral para ver cuántos personajes de cada género tienen "
    "al menos esa cantidad de palabras de diálogo."
)

TRADUCCION_AWARD = {
    "Best Picture": "Mejor Película",
    "Original Screenplay": "Mejor Guion Original",
    "Adapted Screenplay": "Mejor Guion Adaptado",
}
TRADUCCION_AWARD_INV = {v: k for k, v in TRADUCCION_AWARD.items()}

# --- Modo de conteo: películas únicas vs. total de películas (nominaciones) ---
# Varias películas están nominadas en más de una categoría de Oscar, así que
# aparecen dos veces en el dataset "aplanado". "Total de películas" cuenta
# cada nominación por separado (una misma película puede sumar dos veces);
# "Películas únicas" deduplica por Título+Personaje para no contar dos veces
# al mismo personaje.
modo_conteo = st.segmented_control(
    "Modo de conteo",
    options=["Películas únicas", "Total de películas"],
    default="Películas únicas",
    key="modo_conteo",
) or "Películas únicas"

award_seleccionado = st.segmented_control(
    "Categoría de Oscar",
    options=["Todas"] + list(TRADUCCION_AWARD.values()),
    default="Todas",
)

# Base: aplicar primero el filtro de Award, luego el modo de conteo
df_base_award = df.copy()
if award_seleccionado and award_seleccionado != "Todas":
    df_base_award = df_base_award[df_base_award["Award"] == TRADUCCION_AWARD_INV[award_seleccionado]]

if modo_conteo == "Películas únicas":
    df_base_award = df_base_award.drop_duplicates(subset=["Title", "Character"])
    n_peliculas_seleccionadas = df_base_award["Title"].nunique()
else:
    n_peliculas_seleccionadas = df_base_award[["Title", "Award"]].drop_duplicates().shape[0]

st.caption(f"**{n_peliculas_seleccionadas:,} películas seleccionadas**")

max_palabras = int(df["Words"].max())
corte = st.slider("Número mínimo de palabras", min_value=0, max_value=max_palabras, value=20, step=5)

conteo_antes = df_base_award["Gender_ES"].value_counts()

df_filtrado = df_base_award[df_base_award["Words"] >= corte]
conteo_genero = df_filtrado["Gender_ES"].value_counts()


def unir_con_y(partes):
    if not partes:
        return ""
    if len(partes) == 1:
        return partes[0]
    return ", ".join(partes[:-1]) + " y " + partes[-1]


# if modo_conteo == "Películas únicas":
#     st.caption(
#         "Cada personaje se cuenta una sola vez, aunque su película esté "
#         "nominada en más de una categoría de Oscar (películas únicas)."
#     )
# else:
#     st.caption(
#         "Cada nominación cuenta por separado: si una película está nominada "
#         "en 2 categorías de Oscar, sus personajes se cuentan dos veces (una "
#         "vez por categoría)."
#     )

st.caption(f"Personajes incluidos con estos filtros: **{len(df_filtrado):,}**")

partes_exclusion = []
for genero, etiqueta in [
    ("Masculino", "personajes masculinos"),
    ("Femenino", "personajes femeninos"),
    ("Desconocido", "personajes desconocidos"),
]:
    total_antes_genero = conteo_antes.get(genero, 0)
    if total_antes_genero > 0:
        excluidos_genero = total_antes_genero - conteo_genero.get(genero, 0)
        pct_genero = excluidos_genero / total_antes_genero * 100
        partes_exclusion.append(f"**{excluidos_genero:,}** {etiqueta} **({pct_genero:.1f}%)**")

if partes_exclusion:
    st.caption(f"Se excluyen {unir_con_y(partes_exclusion)} con menos de **{corte}** palabras.")

col1, col2, col3 = st.columns(3)
col1.metric("Personajes masculinos", f"{conteo_genero.get('Masculino', 0):,}")
col2.metric("Personajes femeninos", f"{conteo_genero.get('Femenino', 0):,}")
col3.metric("Personajes desconocidos", f"{conteo_genero.get('Desconocido', 0):,}")

titulo_corte = f"Personajes con {corte}+ palabras, por género"
if award_seleccionado and award_seleccionado != "Todas":
    titulo_corte += f" — {award_seleccionado}"
titulo_corte += f" ({modo_conteo.lower()})"

fig_corte = px.bar(
    conteo_genero.reset_index(),
    x="Gender_ES", y="count",
    title=titulo_corte,
    labels={"Gender_ES": "Género", "count": "Número de personajes"},
    color="Gender_ES",
    color_discrete_map=COLORES_GENERO_ES,
)
st.plotly_chart(fig_corte, width="stretch")

st.divider()

# ----------------------------
# TOTALES VS. PROMEDIOS: PERSONAJES POR PELÍCULA
# ----------------------------
st.subheader("De media, solamente 1 de cada 3 personajes es mujer")
st.markdown(
    f"""
    Analizamos la proporción de personajes masculinos y femeninos en las
    {n_peliculas_unicas_total} películas premiadas de la muestra. Este análisis nos
    permite identificar patrones de representación de género y ver cómo varían
    según la categoría de Oscar y el volumen de diálogo de cada personaje.
    """
)

conteo_personajes_total = df_filtrado.groupby("Gender_ES")["Character"].count().reset_index(name="Cantidad")

# Promedio de personajes POR PELÍCULA, calculado dinámicamente sobre los datos
# ya filtrados (corte de palabras + categoría de Oscar), deduplicando por
# Título+Personaje para no contar dos veces a un personaje de una película
# nominada en varias categorías
df_dedup_pelicula = df_filtrado.drop_duplicates(subset=["Title", "Character"])
conteo_por_pelicula_genero = (
    df_dedup_pelicula.groupby(["Title", "Gender_ES"]).size().unstack(fill_value=0)
)
media_personajes_m = (
    conteo_por_pelicula_genero["Masculino"].mean() if "Masculino" in conteo_por_pelicula_genero else 0
)
media_personajes_f = (
    conteo_por_pelicula_genero["Femenino"].mean() if "Femenino" in conteo_por_pelicula_genero else 0
)

col_donut1, col_texto1 = st.columns([1, 1])
with col_donut1:
    fig_personajes_total = px.pie(
        conteo_personajes_total, names="Gender_ES", values="Cantidad", hole=0.45,
        title="Personajes por género (total)",
        color="Gender_ES",
        color_discrete_map=COLORES_GENERO_ES,
    )
    fig_personajes_total.update_traces(textinfo="value+percent", texttemplate="%{value:,}<br>%{percent}")
    st.plotly_chart(fig_personajes_total, width="stretch")
with col_texto1:
    st.markdown("<div style='padding-top: 3.5rem'></div>", unsafe_allow_html=True)
    st.markdown("**Independientemente del filtro...**")
    st.markdown(
        """
        Observamos un patrón persistente en el cine premiado: casi siempre hay muchos más personajes masculinos 
        que femeninos por película, sin importar la categoría de Oscar ni el umbral de palabras que apliques. 
        La brecha se mantiene incluso cuando se filtra por los personajes con más peso narrativo (más palabras de diálogo), 
        lo que sugiere que no es solo un efecto de personajes secundarios sin nombre, sino un patrón estructural en cómo 
        se reparten los papeles con voz propia.
        """
    )
    st.metric("Media de personajes masculinos por película", f"{media_personajes_m:.1f}")
    st.metric("Media de personajes femeninos por película", f"{media_personajes_f:.1f}")

st.divider()

# ----------------------------
# TOTALES VS. PROMEDIOS: PALABRAS
# ----------------------------
st.subheader("Palabras por género")
st.markdown(
    """
    Más palabras para los hombres, pero una historia distinta por personaje.
    Aunque el total de palabras favorece claramente a los personajes masculinos
    (hay muchos más personajes masculinos, y hablan más en conjunto), la media
    de palabras por personaje individual es mucho más equilibrada entre géneros.
    Esto sugiere que la desigualdad está más en el número de oportunidades
    (cuántos personajes femeninos existen) que en cuánto habla cada uno una vez
    que tiene un papel.
    """
)
palabras_por_genero = df_filtrado.groupby("Gender_ES")["Words"].sum().reset_index()

# Media de palabras POR PERSONAJE (no por película) — para comparar con el total
df_dedup_personaje = df_filtrado.drop_duplicates(subset=["Title", "Character"])
media_palabras_personaje = df_dedup_personaje.groupby("Gender_ES")["Words"].mean()

col_donut2, col_donut3 = st.columns(2)
with col_donut2:
    fig_palabras = px.pie(
        palabras_por_genero, names="Gender_ES", values="Words", hole=0.45,
        title=f"Palabras totales por género (personajes con {corte}+ palabras)",
        color="Gender_ES",
        color_discrete_map=COLORES_GENERO_ES,
    )
    fig_palabras.update_traces(textinfo="value+percent", texttemplate="%{value:,}<br>%{percent}")
    st.plotly_chart(fig_palabras, width="stretch")
with col_donut3:
    media_df = media_palabras_personaje.reset_index()
    media_df.columns = ["Gender_ES", "Media"]
    fig_media_personaje = px.pie(
        media_df, names="Gender_ES", values="Media", hole=0.45,
        title="Media de palabras por personaje (no por película)",
        color="Gender_ES",
        color_discrete_map=COLORES_GENERO_ES,
    )
    fig_media_personaje.update_traces(textinfo="value+percent", texttemplate="%{value:.1f}<br>%{percent}")
    st.plotly_chart(fig_media_personaje, width="stretch")

st.divider()

# ----------------------------
# DISTRIBUCIÓN DE PALABRAS POR PERSONAJE (boxplot)
# ----------------------------
st.subheader("Distribución de palabras por personaje")
mediana_m = df_filtrado[df_filtrado["Gender_ES"] == "Masculino"]["Words"].median()
mediana_f = df_filtrado[df_filtrado["Gender_ES"] == "Femenino"]["Words"].median()

personaje_max = df_filtrado.loc[df_filtrado["Words"].idxmax()]
st.markdown(
    f"Cada punto representa un personaje. La mediana de palabras es **{mediana_m:.0f}** "
    f"para los masculinos y **{mediana_f:.0f}** para los femeninos. Como algunos "
    f"personajes hablan muchísimo más que la mayoría — el caso más extremo aquí es "
    f"**{personaje_max['Character']}** ({personaje_max['Title']}, con "
    f"**{personaje_max['Words']:,.0f}** palabras) — la caja aparece comprimida cerca "
    f"de cero: la mayoría de los personajes tienen papeles secundarios con poco "
    f"diálogo, y solo un puñado concentra la mayor parte de las palabras."
)

fig_box = px.box(
    df_filtrado, x="Gender_ES", y="Words", color="Gender_ES",
    title="Distribución de palabras por personaje (según género)",
    labels={"Gender_ES": "Género", "Words": "Palabras"},
    color_discrete_map=COLORES_GENERO_ES,
)
st.plotly_chart(fig_box, width="stretch")

st.divider()

# ----------------------------
# PERSONAJES POR CATEGORÍA DE OSCAR (Award) Y GÉNERO
# ----------------------------
st.subheader("Personajes por categoría de Oscar y género")
st.markdown(
    "Este gráfico compara, dentro de cada categoría de Oscar, cómo cambia el reparto de " \
    "personajes por género según quién dirige o escribe. Para cada categoría se muestran tres grupos: " \
    "todos los personajes, solo los de películas con mujer en la dirección, y solo los de películas con " \
    "mujer en el guion. Así se puede ver si la presencia femenina detrás de cámara coincide con más " \
    "personajes femeninos delante de cámara, categoría por categoría. "
)

# OJO: aquí NO se puede reutilizar df_filtrado, porque ese dataframe puede
# venir deduplicado por Título+Personaje SIN tener en cuenta la categoría
# (modo "Películas únicas"). Si una película está nominada en dos categorías
# (p. ej. Best Picture + Adapted Screenplay), esa deduplicación cruzada le
# "quita" sus personajes a una de las dos categorías, aunque cada categoría
# por sí sola no tenga ningún duplicado que eliminar. Para este gráfico
# partimos de `df` sin deduplicar entre categorías, y solo aplicamos el
# umbral de palabras.
df_award_view = df[df["Words"] >= corte].copy()
df_award_view["Award_ES"] = df_award_view["Award"].map(TRADUCCION_AWARD)

peliculas_con_dir_f = df_peliculas_unicas[df_peliculas_unicas["female_director"] > 0]["Title"].unique()
peliculas_con_guion_f = df_peliculas_unicas[df_peliculas_unicas["female_writer"] > 0]["Title"].unique()

# --- 3 sub-grupos por cada categoría de Oscar: Todas / Mujer en dirección /
# Mujer en guion. Cada sub-grupo se calcula DENTRO de esa categoría de Oscar
# (no mezcla nominaciones de otras categorías). ---
base_todas = df_award_view.groupby(["Award_ES", "Gender_ES"]).size().reset_index(name="Cantidad")
base_todas.insert(1, "Subgrupo", "Todas")

base_dir = (
    df_award_view[df_award_view["Title"].isin(peliculas_con_dir_f)]
    .groupby(["Award_ES", "Gender_ES"]).size().reset_index(name="Cantidad")
)
base_dir.insert(1, "Subgrupo", "Mujer en dirección")

base_guion = (
    df_award_view[df_award_view["Title"].isin(peliculas_con_guion_f)]
    .groupby(["Award_ES", "Gender_ES"]).size().reset_index(name="Cantidad")
)
base_guion.insert(1, "Subgrupo", "Mujer en guion")

conteo_award_genero = pd.concat([base_todas, base_dir, base_guion], ignore_index=True)
orden_subgrupo = ["Todas", "Mujer en dirección", "Mujer en guion"]

fig_award = px.bar(
    conteo_award_genero, x="Subgrupo", y="Cantidad", color="Gender_ES", barmode="group",
    facet_col="Award_ES",
    title="Número de personajes por categoría de Oscar y género",
    labels={"Subgrupo": "", "Gender_ES": "Género"},
    color_discrete_map=COLORES_GENERO_ES,
    category_orders={
        "Gender_ES": ["Masculino", "Femenino", "Desconocido"],
        "Award_ES": list(TRADUCCION_AWARD.values()),
        "Subgrupo": orden_subgrupo,
    },
    text="Cantidad",
)
fig_award.update_traces(textposition="outside", cliponaxis=False)
fig_award.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
# fig_award.update_xaxes(tickangle=-15)
st.plotly_chart(fig_award, width="stretch")

st.divider()

# # ----------------------------
# # TOP 10 PERSONAJES CON MÁS PALABRAS, POR GÉNERO
# # ----------------------------
# st.subheader("Top 10 personajes con más palabras")

# # Deduplicar por pelicula+personaje para no contar dos veces las nominaciones repetidas
# df_top = df.drop_duplicates(subset=["Title", "Character"])

# genero_top = st.radio(
#     "Selecciona género:", ["Masculino", "Femenino", "Desconocido"], horizontal=True
# )

# COLOR_TOP = {
#     "Masculino": "#3B6EA5",
#     "Femenino": "#C1447E",
#     "Desconocido": "#7f7f7f",
# }

# top10 = (
#     df_top[df_top["Gender_ES"] == genero_top]
#     .sort_values("Words", ascending=False)
#     .head(10)[["Title", "Character", "Words"]]
# )
# top10.columns = ["Película", "Personaje", "Palabras"]

# col_izq, col_der = st.columns([1, 1])
# with col_izq:
#     st.dataframe(top10, width="stretch", hide_index=True)
# with col_der:
#     fig_top10 = px.bar(
#         top10.sort_values("Palabras"),
#         x="Palabras", y="Personaje", orientation="h",
#         title=f"Top 10 personajes {genero_top.lower()}s por palabras",
#         color_discrete_sequence=[COLOR_TOP[genero_top]],
#         text="Palabras"
#     )
#     fig_top10.update_traces(textposition="outside", cliponaxis=False)
#     st.plotly_chart(fig_top10, width="stretch")

# divider_grueso()

# ----------------------------
# DIRECTORES Y GUIONISTAS POR GÉNERO
# ----------------------------
st.subheader("Directores y guionistas por género")

rol_seleccionado = st.segmented_control(
    "Selecciona qué rol quieres ver:",
    options=["Directores", "Guionistas"],
    default="Directores",
    key="rol_seleccionado_dyg",
)

award_rol_seleccionado = st.segmented_control(
    "Categoría de Oscar",
    options=["Todas"] + list(TRADUCCION_AWARD.values()),
    default="Todas",
    key="award_rol_seleccionado_dyg",
)

columna_male = "male_director" if rol_seleccionado == "Directores" else "male_writer"
columna_female = "female_director" if rol_seleccionado == "Directores" else "female_writer"

if award_rol_seleccionado and award_rol_seleccionado != "Todas":
    # Filtramos por categoria de Oscar: cada fila ya es una nominacion, no hace falta deduplicar
    base_rol = df_peliculas[df_peliculas["Award"] == TRADUCCION_AWARD_INV[award_rol_seleccionado]]
    sufijo_titulo = f" — {award_rol_seleccionado}"
else:
    # Sin filtro de Award: usamos peliculas unicas para no contar dos veces
    # a un mismo director/guionista si su pelicula fue nominada en varias categorias
    base_rol = df_peliculas_unicas
    sufijo_titulo = " (todas las categorías, por película única)"

resumen_rol = pd.DataFrame({
    "Género": ["Masculino", "Femenino"],
    "Cantidad": [base_rol[columna_male].sum(), base_rol[columna_female].sum()]
})
titulo_rol = f"Distribución de {rol_seleccionado.lower()} por género{sufijo_titulo}"

col_grafico, col_texto = st.columns([2, 1])

with col_grafico:
    fig_rol = px.bar(
        resumen_rol, x="Género", y="Cantidad",
        title=titulo_rol, color="Género", color_discrete_map=COLORES_GENERO_ES,
        text="Cantidad",
    )
    fig_rol.update_traces(textposition="outside", cliponaxis=False)
    fig_rol.update_layout(showlegend=False)
    st.plotly_chart(fig_rol, width="stretch")

with col_texto:
    total_directores = base_rol["male_director"].sum() + base_rol["female_director"].sum()
    total_guionistas = base_rol["male_writer"].sum() + base_rol["female_writer"].sum()
    pct_directoras = (base_rol["female_director"].sum() / total_directores * 100) if total_directores > 0 else 0
    pct_guionistas = (base_rol["female_writer"].sum() / total_guionistas * 100) if total_guionistas > 0 else 0

    st.metric("% Mujeres directoras", f"{pct_directoras:.1f}%")
    st.metric("% Mujeres en guion", f"{pct_guionistas:.1f}%")

    st.markdown(
        """
        El primer resultado relevante es que, tanto en dirección como en guion,
        la representatividad femenina se mantiene en un rango similar
        (frente a la representatividad de hombres), algo que sugiere que la
        subrepresentación femenina no es exclusiva de un rol creativo en
        particular, sino que se replica de forma consistente en ambas
        funciones dentro de las películas premiadas de la muestra.
        """
    )

st.divider()

# ----------------------------
# RELACIÓN DE GÉNERO ENTRE DIRECCIÓN Y GUION
# ----------------------------
st.subheader("Relación de género entre dirección y guion")
st.markdown(
    "Se busca conocer si existe una tendencia a formar equipos del mismo "
    "género o si por el contrario hay diversidad en los dos campos. Este "
    f"apartado trabaja siempre por **película única** ({df_peliculas_unicas.shape[0]} películas), "
    "independientemente de cualquier otro filtro de esta página: la "
    "composición de género del equipo de dirección y guion es una propiedad "
    "de la película, no de la nominación."
)


def clasificar_relacion(row):
    md, fd = row["male_director"], row["female_director"]
    mw, fw = row["male_writer"], row["female_writer"]
    if md > 0 and fd == 0 and mw > 0 and fw == 0:
        return "Masculina"
    if fd > 0 and md == 0 and fw > 0 and mw == 0:
        return "Femenina"
    return "Mixta"


df_relacion = df_peliculas_unicas.copy()
df_relacion["Relación"] = df_relacion.apply(clasificar_relacion, axis=1)

colores_relacion = {"Masculina": "#3B6EA5", "Mixta": "#8E6BAE", "Femenina": "#C1447E"}
orden_relacion = ["Masculina", "Mixta", "Femenina"]

col_izq_rel, col_der_rel = st.columns(2)

with col_izq_rel:
    conteo_relacion = (
        df_relacion["Relación"].value_counts()
        .reindex(orden_relacion, fill_value=0)
        .reset_index()
    )
    conteo_relacion.columns = ["Relación", "Cantidad"]
    fig_relacion = px.pie(
        conteo_relacion, names="Relación", values="Cantidad", hole=0.45,
        title=f"Relación de género en dirección y guion ({df_peliculas_unicas.shape[0]} películas únicas)",
        color="Relación", color_discrete_map=colores_relacion,
        category_orders={"Relación": orden_relacion},
    )
    fig_relacion.update_traces(textinfo="label+value+percent")
    st.plotly_chart(fig_relacion, width="stretch")

with col_der_rel:
    relacion_seleccionada = st.segmented_control(
        "Filtrar por relación de género",
        options=orden_relacion,
        default="Masculina",
        key="relacion_genero_filtro",
    ) or "Masculina"

    df_relacion_filtrada = df_relacion[df_relacion["Relación"] == relacion_seleccionada]
    n_relacion = df_relacion_filtrada.shape[0]
    pct_relacion = (n_relacion / df_peliculas_unicas.shape[0] * 100) if df_peliculas_unicas.shape[0] > 0 else 0

    st.metric(
        f"Películas — relación {relacion_seleccionada.lower()}",
        f"{n_relacion}",
        f"{pct_relacion:.1f}% del total",
    )

    # Una película única puede tener más de una nominación (p. ej. Mejor
    # Película + Mejor Guion Adaptado), así que sacamos todas sus categorías
    # de Oscar desde `df_peliculas` completo (no `df_peliculas_unicas`),
    # aunque esta tabla siga siendo por película única.
    premios_por_pelicula = (
        df_peliculas.groupby("IMDb_ID")["Award"]
        .apply(lambda s: ", ".join(TRADUCCION_AWARD.get(a, a) for a in sorted(s.unique())))
    )

    tabla_relacion = df_relacion_filtrada[[
        "IMDb_ID", "Title", "Director", "male_director", "female_director",
        "Writers", "male_writer", "female_writer",
    ]].copy()
    tabla_relacion["Categoría(s) de Oscar"] = tabla_relacion["IMDb_ID"].map(premios_por_pelicula)
    tabla_relacion = tabla_relacion.rename(columns={
        "Title": "Película",
        "Director": "Director/a(s)",
        "male_director": "Directores (H)",
        "female_director": "Directoras (M)",
        "Writers": "Guionista(s)",
        "male_writer": "Guionistas (H)",
        "female_writer": "Guionistas (M)",
    })
    tabla_relacion = tabla_relacion[[
        "Película", "Director/a(s)", "Directores (H)", "Directoras (M)",
        "Guionista(s)", "Guionistas (H)", "Guionistas (M)", "Categoría(s) de Oscar",
    ]]

    with st.expander(f"Ver tabla ({n_relacion} películas)"):
        st.dataframe(tabla_relacion, width="stretch", hide_index=True)

divider_grueso()

# ----------------------------
# UNA MIRADA FEMENINA
# -----------------------------------------------------------------------------
# Subconjunto de películas donde una mujer participó como directora y/o
# guionista. Las 5 categorías son condiciones INDEPENDIENTES (no excluyentes
# entre sí) — una misma película puede cumplir varias a la vez:
#   - Películas con mujer en la dirección: al menos una directora, sin
#     importar si hay también hombres (dirección mixta incluida).
#   - Películas con mujer guionista: al menos una guionista, sin importar si
#     hay también hombres (guion mixto incluido).
#   - Solamente mujer directora: dirección 100% femenina (sin director
#     hombre), sin importar el guion.
#   - Solamente mujer guionista: guion 100% femenino (sin guionista hombre),
#     sin importar la dirección.
#   - Mujer directora y guionista: dirección Y guion 100% femeninos a la vez.
# -----------------------------------------------------------------------------
st.title("2. Mujeres Directoras / Guionistas")
st.markdown(
    """
    Aquí nos centramos solo en las películas donde una mujer formó parte del
    equipo de dirección o de guion. ¿Cambia la representación de los
    personajes femeninos cuando ellas mismas participan en las decisiones
    creativas detrás de cámara?
    """
)

CONDICIONES_MIRADA_FEMENINA = {
    "Películas con mujer en la dirección": lambda d: (d["female_director"] > 0),
    "Películas con mujer guionista": lambda d: (d["female_writer"] > 0),
    "Solamente mujer directora": lambda d: (
        (d["female_director"] > 0) & (d["male_director"] == 0)
    ),
    "Solamente mujer guionista": lambda d: (
        (d["female_writer"] > 0) & (d["male_writer"] == 0)
    ),
    "Mujer directora y guionista": lambda d: (
        (d["female_director"] > 0) & (d["male_director"] == 0)
        & (d["female_writer"] > 0) & (d["male_writer"] == 0)
    ),
}
orden_patron = list(CONDICIONES_MIRADA_FEMENINA.keys())

patron_seleccionado = st.segmented_control(
    "Filtrar por patrón de participación femenina",
    options=orden_patron,
    default=orden_patron[0],
    key="patron_mirada_femenina",
)
patron_seleccionado = patron_seleccionado or orden_patron[0]

mascara = CONDICIONES_MIRADA_FEMENINA[patron_seleccionado](df_peliculas_unicas)
df_femenina_filtrada = df_peliculas_unicas[mascara]
n_femenina = df_femenina_filtrada.shape[0]
pct_femenina = (n_femenina / df_peliculas_unicas.shape[0] * 100) if df_peliculas_unicas.shape[0] > 0 else 0

st.metric(
    f"Películas — {patron_seleccionado.lower()}",
    f"{n_femenina}",
    f"{pct_femenina:.1f}% del total",
)

if n_femenina == 0:
    st.info("No hay ninguna película en este dataset con este patrón concreto.")
else:
    titulos_patron = df_femenina_filtrada["Title"].unique()
    df_personajes_patron = df[df["Title"].isin(titulos_patron)].drop_duplicates(subset=["Title", "Character"])

    # --- KPIs: personajes, palabras y media, por sexo ---
    dfp_m = df_personajes_patron[df_personajes_patron["Gender_ES"] == "Masculino"]
    dfp_f = df_personajes_patron[df_personajes_patron["Gender_ES"] == "Femenino"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Personajes Masculinos", f"{len(dfp_m)}")
        st.metric("Personajes Femeninos", f"{len(dfp_f)}")
    with col2:
        st.metric("Palabras Masculinas", f"{int(dfp_m['Words'].sum()):,}")
        st.metric("Palabras Femeninas", f"{int(dfp_f['Words'].sum()):,}")
    with col3:
        st.metric("Media de Palabras Masculinas", f"{dfp_m['Words'].mean():.0f}" if len(dfp_m) > 0 else "—")
        st.metric("Media de Palabras Femeninas", f"{dfp_f['Words'].mean():.0f}" if len(dfp_f) > 0 else "—")

    st.markdown("")

    # --- Donuts: total de personajes, de palabras y media, por sexo ---
    col_donut1, col_donut2, col_donut3 = st.columns(3)
    with col_donut1:
        st.markdown("**Total de personajes por género**")
        conteo_personajes_fem = df_personajes_patron["Gender_ES"].value_counts().reset_index()
        conteo_personajes_fem.columns = ["Gender_ES", "Cantidad"]
        fig_personajes_fem = px.pie(
            conteo_personajes_fem, names="Gender_ES", values="Cantidad", hole=0.45,
            color="Gender_ES", color_discrete_map=COLORES_GENERO_ES,
        )
        fig_personajes_fem.update_traces(textinfo="value+percent", texttemplate="%{value}<br>%{percent}")
        st.plotly_chart(fig_personajes_fem, width="stretch")
    with col_donut2:
        st.markdown("**Total de palabras por género**")
        conteo_palabras_fem = df_personajes_patron.groupby("Gender_ES")["Words"].sum().reset_index()
        fig_palabras_fem = px.pie(
            conteo_palabras_fem, names="Gender_ES", values="Words", hole=0.45,
            color="Gender_ES", color_discrete_map=COLORES_GENERO_ES,
        )
        fig_palabras_fem.update_traces(textinfo="value+percent", texttemplate="%{value:,}<br>%{percent}")
        st.plotly_chart(fig_palabras_fem, width="stretch")
    with col_donut3:
        st.markdown("**Media de palabras por género**")
        media_fem = pd.DataFrame({
            "Gender_ES": ["Masculino", "Femenino"],
            "Media": [
                dfp_m["Words"].mean() if len(dfp_m) > 0 else 0,
                dfp_f["Words"].mean() if len(dfp_f) > 0 else 0,
            ],
        })
        fig_media_fem = px.pie(
            media_fem, names="Gender_ES", values="Media", hole=0.45,
            color="Gender_ES", color_discrete_map=COLORES_GENERO_ES,
        )
        fig_media_fem.update_traces(textinfo="value+percent", texttemplate="%{value:.0f}<br>%{percent}")
        st.plotly_chart(fig_media_fem, width="stretch")

    # --- Tabla de películas del patrón seleccionado ---
    premios_por_pelicula_fem = (
        df_peliculas.groupby("IMDb_ID")["Award"]
        .apply(lambda s: ", ".join(TRADUCCION_AWARD.get(a, a) for a in sorted(s.unique())))
    )

    tabla_femenina = df_femenina_filtrada[[
        "IMDb_ID", "Title", "Director", "male_director", "female_director",
        "Writers", "male_writer", "female_writer",
    ]].copy()
    tabla_femenina["Categoría(s) de Oscar"] = tabla_femenina["IMDb_ID"].map(premios_por_pelicula_fem)
    tabla_femenina = tabla_femenina.rename(columns={
        "Title": "Película",
        "Director": "Director/a(s)",
        "male_director": "Directores (H)",
        "female_director": "Directoras (M)",
        "Writers": "Guionista(s)",
        "male_writer": "Guionistas (H)",
        "female_writer": "Guionistas (M)",
    })
    tabla_femenina = tabla_femenina[[
        "Película", "Director/a(s)", "Directores (H)", "Directoras (M)",
        "Guionista(s)", "Guionistas (H)", "Guionistas (M)", "Categoría(s) de Oscar",
    ]]

    with st.expander(f"Ver tabla ({n_femenina} películas)"):
        st.dataframe(tabla_femenina, width="stretch", hide_index=True)

    st.divider()

    # --- Comparativa fija entre grupos: ¿cambia el peso narrativo por sexo
    # según quién dirige/escribe? Estos dos gráficos NO dependen del pill de
    # arriba: comparan siempre los mismos grupos entre sí. ---
    st.markdown("#### ¿Cambia el peso narrativo según quién dirige o escribe?")
    st.markdown(
        "Comparación fija entre grupos de películas (no cambia con el filtro "
        "de arriba), para ver si la media de palabras por personaje y el % "
        "de personajes femeninos varían según la composición del equipo "
        "creativo."
    )

    GRUPOS_COMPARATIVA = {
        "Equipo 100% masculino": lambda d: (
            (d["male_director"] > 0) & (d["female_director"] == 0)
            & (d["male_writer"] > 0) & (d["female_writer"] == 0)
        ),
        "Todas las películas": lambda d: pd.Series(True, index=d.index),
        **CONDICIONES_MIRADA_FEMENINA,
    }

    filas_comparativa = []
    for nombre_grupo, condicion in GRUPOS_COMPARATIVA.items():
        titulos_grupo = df_peliculas_unicas[condicion(df_peliculas_unicas)]["Title"].unique()
        sub_grupo = df[df["Title"].isin(titulos_grupo)].drop_duplicates(subset=["Title", "Character"])
        m_grupo = sub_grupo[sub_grupo["Gender_ES"] == "Masculino"]
        f_grupo = sub_grupo[sub_grupo["Gender_ES"] == "Femenino"]
        total_mf = len(m_grupo) + len(f_grupo)
        filas_comparativa.append({
            "Grupo": nombre_grupo,
            "Media Masculino": m_grupo["Words"].mean() if len(m_grupo) > 0 else 0,
            "Media Femenino": f_grupo["Words"].mean() if len(f_grupo) > 0 else 0,
            "% personajes femeninos": (len(f_grupo) / total_mf * 100) if total_mf > 0 else 0,
        })
    df_comparativa = pd.DataFrame(filas_comparativa)

    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        df_comp_media = df_comparativa.melt(
            id_vars="Grupo", value_vars=["Media Masculino", "Media Femenino"],
            var_name="Serie", value_name="Media de palabras",
        )
        df_comp_media["Serie"] = df_comp_media["Serie"].map({
            "Media Masculino": "Masculino", "Media Femenino": "Femenino",
        })
        fig_comp_media = px.bar(
            df_comp_media, x="Grupo", y="Media de palabras", color="Serie", barmode="group",
            title="Media de palabras por personaje, según equipo creativo",
            color_discrete_map=COLORES_GENERO_ES,
            text="Media de palabras",
        )
        fig_comp_media.update_traces(texttemplate="%{text:.0f}", textposition="outside", cliponaxis=False)
        fig_comp_media.update_layout(xaxis_title="", xaxis_tickangle=-20)
        st.plotly_chart(fig_comp_media, width="stretch")
    with col_comp2:
        fig_comp_pct = px.bar(
            df_comparativa, x="Grupo", y="% personajes femeninos",
            title="% de personajes femeninos, según equipo creativo",
            text="% personajes femeninos",
        )
        fig_comp_pct.update_traces(
            texttemplate="%{text:.1f}%", textposition="outside",
            marker_color="#C1447E", cliponaxis=False,
        )
        fig_comp_pct.update_layout(xaxis_title="", xaxis_tickangle=-20)
        st.plotly_chart(fig_comp_pct, width="stretch")

    st.divider()

    # --- Sunburst: top 10 personajes (por palabras) dentro de este patrón ---
    top10_patron = df_personajes_patron.sort_values("Words", ascending=False).head(10)

    fig_sunburst_fem = px.sunburst(
        top10_patron, path=["Gender_ES", "Character"], values="Words",
        color="Gender_ES", color_discrete_map=COLORES_GENERO_ES,
        title=f"Top 10 personajes por palabras — {patron_seleccionado.lower()}",
        height=550,
    )
    fig_sunburst_fem.update_traces(textinfo="label", textfont_size=14)
    fig_sunburst_fem.update_layout(margin=dict(t=60, l=10, r=10, b=10))
    st.plotly_chart(fig_sunburst_fem, width="stretch")

divider_grueso()

# ----------------------------
# MIRANDO CON LUPA
# ----------------------------
st.title("3. Mirando con lupa...")

st.markdown(
    """
    En esta sección analizamos cada película por separado. Selecciona una
    película específica del dataset y explora en detalle cómo se reparten los
    personajes y el diálogo entre hombres y mujeres.
    """
)

COLOR_MASCULINO = "#3B6EA5"
COLOR_FEMENINO = "#C1447E"
COLORES_GENERO_LUPA = {"Masculino": COLOR_MASCULINO, "Femenino": COLOR_FEMENINO, "Desconocido": "#7f7f7f"}

st.divider()

# ----------------------------
# SELECTOR DE PELÍCULA
# ----------------------------
st.subheader("¡A explorar! 👇 Selecciona una película")

peliculas_disponibles = sorted(df["Title"].unique())
pelicula_seleccionada = st.selectbox("Película", peliculas_disponibles)

# Deduplicar por personaje (por si la película está nominada en varias categorías de Oscar)
df_pelicula = df[df["Title"] == pelicula_seleccionada].drop_duplicates(subset="Character")

df_m = df_pelicula[df_pelicula["Gender_ES"] == "Masculino"]
df_f = df_pelicula[df_pelicula["Gender_ES"] == "Femenino"]

st.divider()

# ----------------------------
# MÉTRICAS: UN DATO VALE MÁS QUE MIL PALABRAS
# ----------------------------
st.subheader("Un dato vale más que mil palabras")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Personajes Masculinos", f"{len(df_m)}")
    st.metric("Personajes Femeninos", f"{len(df_f)}")
with col2:
    st.metric("Palabras Masculinas", f"{int(df_m['Words'].sum()):,}")
    st.metric("Palabras Femeninas", f"{int(df_f['Words'].sum()):,}")
with col3:
    st.metric("Media de Palabras Masculinas", f"{df_m['Words'].mean():.0f}" if len(df_m) > 0 else "—")
    st.metric("Media de Palabras Femeninas", f"{df_f['Words'].mean():.0f}" if len(df_f) > 0 else "—")

st.divider()

# ----------------------------
# DONUTS: TOTAL DE PERSONAJES Y PALABRAS POR SEXO
# ----------------------------
col_donut1, col_donut2 = st.columns(2)

with col_donut1:
    st.markdown("**Total de personajes por sexo**")
    conteo_personajes = df_pelicula["Gender_ES"].value_counts().reset_index()
    conteo_personajes.columns = ["Gender_ES", "Cantidad"]
    fig_personajes = px.pie(
        conteo_personajes, names="Gender_ES", values="Cantidad", hole=0.45,
        color="Gender_ES", color_discrete_map=COLORES_GENERO_LUPA
    )
    fig_personajes.update_traces(textinfo="value+percent", texttemplate="%{value}<br>%{percent}")
    st.plotly_chart(fig_personajes, width="stretch")

with col_donut2:
    st.markdown("**Total de palabras por género**")
    conteo_palabras = df_pelicula.groupby("Gender_ES")["Words"].sum().reset_index()
    fig_palabras = px.pie(
        conteo_palabras, names="Gender_ES", values="Words", hole=0.45,
        color="Gender_ES", color_discrete_map=COLORES_GENERO_LUPA
    )
    fig_palabras.update_traces(textinfo="value+percent", texttemplate="%{value:,}<br>%{percent}")
    st.plotly_chart(fig_palabras, width="stretch")

st.divider()

# ----------------------------
# SUNBURST: REPARTO DE PERSONAJES PRINCIPALES
# ----------------------------
st.subheader(f"Cómo es la distribución en {pelicula_seleccionada}")
st.markdown(
    """
    Al seleccionar los diez personajes con más diálogo de la película, los
    hombres, siguiendo la tónica general, suelen dominar la lista. Ese mayor
    peso de personajes masculinos influye en la narrativa, dando más espacio
    y profundidad a las voces masculinas mientras que las femeninas son menos
    representadas.
    """
)

top10_pelicula = df_pelicula.sort_values("Words", ascending=False).head(10)

fig_sunburst = px.sunburst(
    top10_pelicula, path=["Gender_ES", "Character"], values="Words",
    color="Gender_ES", color_discrete_map=COLORES_GENERO_LUPA,
    title="Reparto de personajes principales",
    height=550
)
fig_sunburst.update_traces(textinfo="label", textfont_size=14)
fig_sunburst.update_layout(margin=dict(t=60, l=10, r=10, b=10))
st.plotly_chart(fig_sunburst, width="stretch")