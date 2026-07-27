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

st.title("Estadísticas de Género")
st.markdown(
    """
    Esta página explora cómo se reparte el diálogo entre personajes masculinos y
    femeninos: cuántos personajes hay de cada género, cuántas palabras hablan,
    y cómo varía esto según la categoría de Oscar.
    """
)
st.caption(
    "Nota: esta página usa el corpus de NLP (75 nominaciones), que excluye "
    "*Talk to Her* y *Anatomy of a Fall* por no estar predominantemente en inglés. "
    "Por eso el total de personajes aquí (3,119) es algo menor que en la landing page (3,186)."
)

COLORES_GENERO = {"male": "#3B6EA5", "female": "#C1447E", "unknown": "#b0b0b0"}
NOMBRES_GENERO = {"male": "Masculino", "female": "Femenino", "unknown": "Desconocido"}

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

award_seleccionado = st.segmented_control(
    "Categoría de Oscar",
    options=["Todas"] + list(TRADUCCION_AWARD.values()),
    default="Todas",
)

max_palabras = int(df["Words"].max())
corte = st.slider("Número mínimo de palabras", min_value=0, max_value=max_palabras, value=20, step=5)

# Base sin el corte de palabras (pero ya con el filtro de Award aplicado), para poder
# calcular cuántos personajes se excluyen exactamente por el slider
df_base_award = df.copy()
if award_seleccionado and award_seleccionado != "Todas":
    df_base_award = df_base_award[df_base_award["Award"] == TRADUCCION_AWARD_INV[award_seleccionado]]

conteo_antes = df_base_award["Gender_ES"].value_counts()

df_filtrado = df_base_award[df_base_award["Words"] >= corte]
conteo_genero = df_filtrado["Gender_ES"].value_counts()

excluidos_m = conteo_antes.get("Masculino", 0) - conteo_genero.get("Masculino", 0)
excluidos_f = conteo_antes.get("Femenino", 0) - conteo_genero.get("Femenino", 0)

total_antes_m = conteo_antes.get("Masculino", 0)
total_antes_f = conteo_antes.get("Femenino", 0)

pct_masc = (excluidos_m / total_antes_m * 100) if total_antes_m > 0 else 0
pct_fem = (excluidos_f / total_antes_f * 100) if total_antes_f > 0 else 0

st.caption(
    f"Se excluyen **{excluidos_m:,}** personajes masculinos **({pct_masc:.1f}%)** y "
    f"**{excluidos_f:,}** personajes femeninos **({pct_fem:.1f}%)** con menos de **{corte}** palabras."
)

col1, col2, col3 = st.columns(3)
col1.metric("Personajes masculinos", f"{conteo_genero.get('Masculino', 0):,}")
col2.metric("Personajes femeninos", f"{conteo_genero.get('Femenino', 0):,}")
col3.metric("Personajes desconocidos", f"{conteo_genero.get('Desconocido', 0):,}")

titulo_corte = f"Personajes con {corte}+ palabras, por género"
if award_seleccionado and award_seleccionado != "Todas":
    titulo_corte += f" — {award_seleccionado}"

fig_corte = px.bar(
    conteo_genero.reset_index(),
    x="Gender_ES", y="count",
    title=titulo_corte,
    labels={"Gender_ES": "Género", "count": "Número de personajes"},
    color="Gender_ES",
    color_discrete_map={"Masculino": "#3B6EA5", "Femenino": "#C1447E", "Desconocido": "#b0b0b0"}
)
st.plotly_chart(fig_corte, width="stretch")

st.divider()

# ----------------------------
# TOTALES VS. PROMEDIOS: PERSONAJES POR PELÍCULA
# ----------------------------
st.subheader("De media, solamente 1 de cada 3 personajes es mujer")
st.markdown(
    """
    Analizamos la proporción de personajes masculinos y femeninos en las 58 películas premiadas de la muestra. Este análisis nos permite identificar patrones de representación de género y ver cómo varían según la categoría de Oscar y el volumen de diálogo de cada personaje.
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
        title="Personajes por sexo (total)",
        color="Gender_ES",
        color_discrete_map={"Masculino": "#3B6EA5", "Femenino": "#C1447E", "Desconocido": "#b0b0b0"}
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
st.subheader("Palabras por sexo")
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
        title=f"Palabras totales por sexo (personajes con {corte}+ palabras)",
        color="Gender_ES",
        color_discrete_map={"Masculino": "#3B6EA5", "Femenino": "#C1447E", "Desconocido": "#b0b0b0"}
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
        color_discrete_map={"Masculino": "#3B6EA5", "Femenino": "#C1447E", "Desconocido": "#b0b0b0"}
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
    color_discrete_map={"Masculino": "#3B6EA5", "Femenino": "#C1447E", "Desconocido": "#b0b0b0"}
)
st.plotly_chart(fig_box, width="stretch")

st.divider()

# ----------------------------
# PERSONAJES POR CATEGORÍA DE OSCAR (Award) Y GÉNERO
# ----------------------------
st.subheader("Personajes por categoría de Oscar y género")
 
conteo_award_genero = df_filtrado.groupby(["Award", "Gender_ES"]).size().reset_index(name="Cantidad")
 
fig_award = px.bar(
    conteo_award_genero, x="Award", y="Cantidad", color="Gender_ES", barmode="group",
    title="Número de personajes por categoría de Oscar y género",
    labels={"Award": "Categoría de Oscar", "Gender_ES": "Género"},
    color_discrete_map={"Masculino": "#3B6EA5", "Femenino": "#C1447E", "Desconocido": "#b0b0b0"},
    category_orders={"Gender_ES": ["Masculino", "Femenino", "Desconocido"]},
    text="Cantidad"
)
fig_award.update_traces(textposition="outside", cliponaxis=False)
st.plotly_chart(fig_award, width="stretch")
 
st.divider()
 
# ----------------------------
# TOP 10 PERSONAJES CON MÁS PALABRAS, POR GÉNERO
# ----------------------------
st.subheader("Top 10 personajes con más palabras")
 
# Deduplicar por pelicula+personaje para no contar dos veces las nominaciones repetidas
df_top = df.drop_duplicates(subset=["Title", "Character"])
 
genero_top = st.radio("Selecciona género:", ["Masculino", "Femenino"], horizontal=True)
 
top10 = (
    df_top[df_top["Gender_ES"] == genero_top]
    .sort_values("Words", ascending=False)
    .head(10)[["Title", "Character", "Words"]]
)
top10.columns = ["Película", "Personaje", "Palabras"]
 
col_izq, col_der = st.columns([1, 1])
with col_izq:
    st.dataframe(top10, width="stretch", hide_index=True)
with col_der:
    fig_top10 = px.bar(
        top10.sort_values("Palabras"),
        x="Palabras", y="Personaje", orientation="h",
        title=f"Top 10 personajes {genero_top.lower()}s por palabras",
        color_discrete_sequence=["#3B6EA5" if genero_top == "Masculino" else "#C1447E"],
        text="Palabras"
    )
    fig_top10.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig_top10, width="stretch")

# ----------------------------
# MIRANDO CON LUPA
# ----------------------------
st.title("Mirando con lupa...")
 
st.markdown(
    """
    En esta sección analizamos cada película por separado. Selecciona una
    película específica del dataset y explora en detalle cómo se reparten los
    personajes y el diálogo entre hombres y mujeres.
    """
)
 
COLOR_MASCULINO = "#3B6EA5"
COLOR_FEMENINO = "#C1447E"
COLORES_GENERO = {"Masculino": COLOR_MASCULINO, "Femenino": COLOR_FEMENINO, "Desconocido": "#7f7f7f"}
NOMBRES_GENERO = {"male": "Masculino", "female": "Femenino", "unknown": "Desconocido"}
 
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
        color="Gender_ES", color_discrete_map=COLORES_GENERO
    )
    fig_personajes.update_traces(textinfo="value+percent", texttemplate="%{value}<br>%{percent}")
    st.plotly_chart(fig_personajes, width="stretch")
 
with col_donut2:
    st.markdown("**Total de palabras por sexo**")
    conteo_palabras = df_pelicula.groupby("Gender_ES")["Words"].sum().reset_index()
    fig_palabras = px.pie(
        conteo_palabras, names="Gender_ES", values="Words", hole=0.45,
        color="Gender_ES", color_discrete_map=COLORES_GENERO
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
    color="Gender_ES", color_discrete_map=COLORES_GENERO,
    title="Reparto de personajes principales",
    height=550
)
fig_sunburst.update_traces(textinfo="label", textfont_size=14)
fig_sunburst.update_layout(margin=dict(t=60, l=10, r=10, b=10))
st.plotly_chart(fig_sunburst, width="stretch")

