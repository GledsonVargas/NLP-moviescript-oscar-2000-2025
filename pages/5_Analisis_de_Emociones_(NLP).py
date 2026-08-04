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

COLORES_GENERO = {"Masculino": "#2a78d6", "Femenino": "#C1447E", "Desconocido": "#b0b0b0"}
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


def pill_excluir_neutro(key):
    """Pill toggle reutilizable: devuelve True si el usuario activó 'Excluir neutro'."""
    seleccion = st.pills(
        "Filtro de emociones",
        options=["Excluir neutro"],
        selection_mode="multi",
        default=[],
        key=key,
        label_visibility="collapsed",
    )
    return "Excluir neutro" in seleccion


# ----------------------------
# CARGA DE DATOS
# ----------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("df_hartmann_emotions.csv")
    df["Gender_ES"] = df["Gender"].map(NOMBRES_GENERO)
    return df


# -----------------------------------------------------------------------------
# Carga adicional a nivel película (director/guionista), necesaria para los
# pills de "Equipo creativo" del radar. df_hartmann_emotions.csv no trae estas
# columnas, así que se cargan aparte desde Dataset_final.pkl.
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
        default=["Masculino", "Femenino", "Desconocido"]
    )
with col_b:
    corte_palabras = st.slider("Palabras mínimas del personaje", 0, int(df["Words"].max()), 0, step=5)

# --- Base: género + award ya aplicados, SIN el corte de palabras todavía ---
# (para poder calcular cuántos personajes se excluyen exactamente por el slider)
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
    total_antes_genero = conteo_antes.get(genero, 0)
    if total_antes_genero > 0:
        excluidos_genero = total_antes_genero - conteo_despues.get(genero, 0)
        pct_genero = excluidos_genero / total_antes_genero * 100
        partes_exclusion.append(f"**{excluidos_genero:,}** {etiqueta} **({pct_genero:.1f}%)**")

texto_exclusion = ""
if partes_exclusion:
    texto_exclusion = f"Se excluyen {unir_con_y(partes_exclusion)} con menos de **{corte_palabras}** palabras."

st.caption(f"Personajes incluidos con estos filtros: **{len(df_filtrado):,}**")
if texto_exclusion:
    st.caption(texto_exclusion)

st.divider()

# ----------------------------
# EMOCIÓN DOMINANTE POR GÉNERO
# ----------------------------
st.subheader("Emoción dominante por género")
st.caption(f"Personajes en este recorte: **{len(df_filtrado):,}**. " + texto_exclusion)

excluir_neutro_dominante = pill_excluir_neutro("pill_dominante")

df_dominante_base = df_filtrado.copy()
if excluir_neutro_dominante:
    df_dominante_base = df_dominante_base[df_dominante_base["Emotion_Dominant"].str.lower() != "neutral"]

conteo_dominante = df_dominante_base.groupby(["Gender_ES", "Emotion_Dominant"]).size().reset_index(name="Cantidad")
conteo_dominante["Emotion_Dominant_ES"] = conteo_dominante["Emotion_Dominant"].str.capitalize().map({
    "Anger": "Ira", "Disgust": "Asco", "Fear": "Miedo", "Joy": "Alegría",
    "Neutral": "Neutro", "Sadness": "Tristeza", "Surprise": "Sorpresa"
})

fig_dominante = px.bar(
    conteo_dominante, x="Emotion_Dominant_ES", y="Cantidad", color="Gender_ES", barmode="group",
    title="Número de personajes según su emoción dominante",
    labels={"Emotion_Dominant_ES": "Emoción dominante", "Gender_ES": "Género"},
    color_discrete_map=COLORES_GENERO,
    text="Cantidad",
)
fig_dominante.update_traces(textposition="outside", cliponaxis=False)
st.plotly_chart(fig_dominante, width="stretch")

st.divider()

# ----------------------------
# PERFIL EMOCIONAL PROMEDIO: GRÁFICO DE RADAR
# -----------------------------------------------------------------------------
# Este radar tiene sus propios filtros (independientes de los de arriba), para
# poder comparar de forma rápida distintos recortes sin perder los filtros
# generales de la página.
# -----------------------------------------------------------------------------
st.subheader("Perfil emocional promedio por género")
st.markdown(
    """
    Cada eje es una de las 7 emociones de Hartmann; cada línea es el promedio
    de esa emoción para todos los personajes de ese género. Cuanto más lejos
    del centro, más presente está esa emoción en el diálogo. Usa los dos
    filtros de abajo para ver si el perfil cambia según la categoría de Oscar
    o según si hay una mujer en el equipo de dirección o guion.
    """
)

col_radar_1, col_radar_2 = st.columns(2)
with col_radar_1:
    award_radar = st.segmented_control(
        "Categoría de Oscar",
        options=["Todas"] + list(TRADUCCION_AWARD.values()),
        default="Todas",
        key="award_radar",
    ) or "Todas"
with col_radar_2:
    equipo_radar = st.segmented_control(
        "Equipo creativo",
        options=["Todas las películas", "Mujer en dirección", "Mujer en guion"],
        default="Todas las películas",
        key="equipo_radar",
    ) or "Todas las películas"

excluir_neutro_radar = pill_excluir_neutro("pill_radar")
emociones_radar = [e for e in EMOCIONES if not (excluir_neutro_radar and e == "Emotion_Neutral")]

df_radar = df_filtrado.copy()
if award_radar != "Todas":
    df_radar = df_radar[df_radar["Award"] == TRADUCCION_AWARD_INV[award_radar]]
if equipo_radar == "Mujer en dirección":
    df_radar = df_radar[df_radar["Title"].isin(peliculas_con_dir_f)]
elif equipo_radar == "Mujer en guion":
    df_radar = df_radar[df_radar["Title"].isin(peliculas_con_guion_f)]

if df_radar.empty:
    st.info("No hay personajes que cumplan esta combinación de filtros.")
else:
    promedio_emociones = df_radar.groupby("Gender_ES")[emociones_radar].mean()

    fig_radar = go.Figure()
    for genero in promedio_emociones.index:
        valores = promedio_emociones.loc[genero, emociones_radar].tolist()
        valores.append(valores[0])  # cerrar el polígono
        etiquetas = [NOMBRES_EMOCIONES[e] for e in emociones_radar] + [NOMBRES_EMOCIONES[emociones_radar[0]]]
        fig_radar.add_trace(go.Scatterpolar(
            r=valores, theta=etiquetas, fill='toself', name=genero,
            line_color=COLORES_GENERO.get(genero, "#888888")
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title=f"Perfil emocional promedio por género — {award_radar} / {equipo_radar}",
        showlegend=True,
    )
    st.plotly_chart(fig_radar, width="stretch")
    st.caption(f"Personajes en este recorte: **{len(df_radar):,}**. " + texto_exclusion)


st.divider()

# ----------------------------
# PERFIL EMOCIONAL FEMENINO, SEGÚN EQUIPO CREATIVO (DIAGRAMA DE CALOR)
# -----------------------------------------------------------------------------
# Compara el perfil de las 7 emociones en personajes FEMENINOS, según si su
# película tiene o no una mujer en dirección o guion. Usa los filtros
# generales de arriba (Award, Words), no los pills del radar.
# -----------------------------------------------------------------------------
st.subheader("Perfil emocional femenino, según equipo creativo")
st.markdown(
    """
    Aquí nos quedamos solo con los personajes femeninos y los separamos en dos
    grupos: los de películas con una mujer en dirección o guion, y los del
    resto. El color de cada celda es el % promedio de esa emoción en ese
    grupo — así se ve de un vistazo qué emociones ganan o pierden peso cuando
    hay una mujer en el equipo creativo.
    """
)

excluir_neutro_heatmap = pill_excluir_neutro("pill_heatmap")
emociones_heatmap = [e for e in EMOCIONES if not (excluir_neutro_heatmap and e == "Emotion_Neutral")]

df_heatmap = df_filtrado[df_filtrado["Gender_ES"] == "Femenino"].copy()
if df_heatmap.empty:
    st.info("No hay personajes femeninos que cumplan los filtros generales de arriba.")
else:
    df_heatmap["Grupo"] = df_heatmap["Title"].apply(
        lambda t: "Con mujer en dirección o guion"
        if (t in peliculas_con_dir_f or t in peliculas_con_guion_f)
        else "Sin mujer en dirección ni guion"
    )

    tabla_heatmap = (df_heatmap.groupby("Grupo")[emociones_heatmap].mean() * 100)
    tabla_heatmap = tabla_heatmap.rename(columns=NOMBRES_EMOCIONES)
    orden_grupo = ["Sin mujer en dirección ni guion", "Con mujer en dirección o guion"]
    tabla_heatmap = tabla_heatmap.reindex([g for g in orden_grupo if g in tabla_heatmap.index])

    fig_heatmap = px.imshow(
        tabla_heatmap,
        text_auto=".1f",
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(x="Emoción", y="", color="% promedio"),
    )
    fig_heatmap.update_layout(title="Perfil emocional femenino, según equipo creativo")
    st.plotly_chart(fig_heatmap, width="stretch")

    n_por_grupo = df_heatmap["Grupo"].value_counts()
    st.caption(
        "Personajes femeninos en cada grupo: "
        + ", ".join(f"**{g}**: {n_por_grupo.get(g, 0):,}" for g in orden_grupo)
    )

st.divider()

# ----------------------------
# PERFIL EMOCIONAL: TOP 10 PERSONAJES CON MÁS PALABRAS (M vs F)
# -----------------------------------------------------------------------------
# No usa df_filtrado ni el slider de palabras de arriba (usarlo aquí sería
# circular, ya que este gráfico se basa precisamente en quiénes tienen más
# palabras). Se parte de `df` completo, deduplicado por Título+Personaje para
# no contar dos veces a un personaje de una película nominada en varias
# categorías de Oscar (56 películas únicas).
#
# Nota: "BERTIE" (The King's Speech) estaba etiquetado como "female" en
# df_hartmann_emotions.csv (personaje masculino, Jorge VI) — ya corregido a
# "male" en el CSV.
# -----------------------------------------------------------------------------
st.subheader("Perfil emocional: Top 10 personajes con más palabras (M vs F)")
st.markdown(
    """
    Aquí comparamos el perfil emocional de los 10 personajes masculinos y los
    10 personajes femeninos con más palabras de diálogo de todo el dataset
    (no depende de los filtros de arriba). Sirve para ver si los perfiles
    protagónicos más grandes —los papeles con más peso— siguen el mismo
    patrón que el promedio general, o si se comportan distinto.
    """
)

excluir_neutro_top10 = pill_excluir_neutro("pill_top10")
emociones_top10 = [e for e in EMOCIONES if not (excluir_neutro_top10 and e == "Emotion_Neutral")]

df_dedup_top = df.drop_duplicates(subset=["Title", "Character"])
top10_m = df_dedup_top[df_dedup_top["Gender_ES"] == "Masculino"].sort_values("Words", ascending=False).head(10)
top10_f = df_dedup_top[df_dedup_top["Gender_ES"] == "Femenino"].sort_values("Words", ascending=False).head(10)

tabla_top10 = pd.DataFrame({
    "Top 10 masculino": (top10_m[emociones_top10].mean() * 100),
    "Top 10 femenino": (top10_f[emociones_top10].mean() * 100),
}).T
tabla_top10 = tabla_top10.rename(columns=NOMBRES_EMOCIONES)

fig_heatmap_top10 = px.imshow(
    tabla_top10,
    text_auto=".1f",
    color_continuous_scale="Blues",
    aspect="auto",
    labels=dict(x="Emoción", y="", color="% promedio"),
)
fig_heatmap_top10.update_layout(title="Perfil emocional — Top 10 personajes con más palabras")
st.plotly_chart(fig_heatmap_top10, width="stretch")

st.caption(
    f"Basado en los 10 personajes masculinos y los 10 femeninos con más "
    f"palabras del dataset completo (top masculino: "
    f"{int(top10_m['Words'].min()):,}–{int(top10_m['Words'].max()):,} palabras; "
    f"top femenino: {int(top10_f['Words'].min()):,}–{int(top10_f['Words'].max()):,} palabras)."
)

EMOTION_DOMINANT_ES = {
    "anger": "Ira", "disgust": "Asco", "fear": "Miedo", "joy": "Alegría",
    "neutral": "Neutro", "sadness": "Tristeza", "surprise": "Sorpresa",
}


def tabla_top10_personajes(top_df):
    tabla = top_df[["Character", "Title", "Words", "Emotion_Dominant"]].copy()
    tabla["Emotion_Dominant"] = tabla["Emotion_Dominant"].str.lower().map(EMOTION_DOMINANT_ES)
    tabla.insert(0, "#", range(1, len(tabla) + 1))
    tabla.columns = ["#", "Personaje", "Película", "Palabras", "Emoción dominante"]
    return tabla


col_tabla_m, col_tabla_f = st.columns(2)
with col_tabla_m:
    st.markdown("**Top 10 masculino**")
    st.dataframe(tabla_top10_personajes(top10_m), width="stretch", hide_index=True)
with col_tabla_f:
    st.markdown("**Top 10 femenino**")
    st.dataframe(tabla_top10_personajes(top10_f), width="stretch", hide_index=True)

st.divider()

# ----------------------------
# EVOLUCIÓN TEMPORAL DE LAS EMOCIONES
# -----------------------------------------------------------------------------
# Media de una emoción (elegida por el usuario) a lo largo de los años, por
# género, para ver si la brecha emocional entre géneros se ha reducido o
# ampliado desde 2000.
#
# Opciones especiales del selector:
#   - "TODAS": grid de 7 paneles (uno por emoción), cada uno con 2 líneas
#     (Masculino/Femenino) — para comparar géneros emoción por emoción.
#   - "SIETE_F" / "SIETE_M": UN solo gráfico con 7 líneas de color (una por
#     emoción), para UN género a la vez — para comparar las 7 emociones
#     entre sí dentro del mismo género.
# -----------------------------------------------------------------------------
st.subheader("Evolución temporal de las emociones")
st.markdown(
    """
    ¿La brecha emocional entre personajes masculinos y femeninos se mantiene
    igual a lo largo de los años, o ha cambiado? Elige una emoción y compara
    su media año a año (2000–2025) entre géneros.
    """
)

OPCIONES_EVOLUCION = ["TODAS", "SIETE_F", "SIETE_M"] + EMOCIONES
NOMBRES_OPCIONES_EVOLUCION = {
    "TODAS": "Las 7 emociones (masculino y femenino)",
    "SIETE_F": "Las 7 emociones (femenino)",
    "SIETE_M": "Las 7 emociones (masculino)",
}

excluir_neutro_evolucion = pill_excluir_neutro("pill_evolucion")
emociones_evolucion_activas = [e for e in EMOCIONES if not (excluir_neutro_evolucion and e == "Emotion_Neutral")]
opciones_evolucion_activas = [
    o for o in OPCIONES_EVOLUCION if not (excluir_neutro_evolucion and o == "Emotion_Neutral")
]

# Si la emoción actualmente elegida deja de estar disponible (p.ej. "Neutro"
# al activar el pill), volvemos a "TODAS" para no romper el selectbox.
if st.session_state.get("emocion_evolucion") not in opciones_evolucion_activas:
    st.session_state["emocion_evolucion"] = "TODAS"

emocion_evolucion = st.selectbox(
    "Elige una emoción",
    options=opciones_evolucion_activas,
    format_func=lambda e: NOMBRES_OPCIONES_EVOLUCION.get(e) or NOMBRES_EMOCIONES[e],
    key="emocion_evolucion",
)

if emocion_evolucion == "TODAS":
    evolucion_todas = (
        df_filtrado.groupby(["Oscar_Year", "Gender_ES"])[emociones_evolucion_activas]
        .mean()
        .reset_index()
        .melt(id_vars=["Oscar_Year", "Gender_ES"], var_name="Emoción_col", value_name="Media")
    )
    evolucion_todas["Emoción"] = evolucion_todas["Emoción_col"].map(NOMBRES_EMOCIONES)

    fig_evolucion = px.line(
        evolucion_todas, x="Oscar_Year", y="Media", color="Gender_ES",
        facet_col="Emoción", facet_col_wrap=4,
        title="Media de cada emoción por año y género",
        labels={"Oscar_Year": "Año", "Media": "Media", "Gender_ES": "Género"},
        color_discrete_map=COLORES_GENERO,
        markers=True,
        category_orders={"Emoción": [NOMBRES_EMOCIONES[e] for e in emociones_evolucion_activas]},
    )
    fig_evolucion.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig_evolucion.update_yaxes(matches=None, showticklabels=True)
    fig_evolucion.update_layout(height=600)
    st.plotly_chart(fig_evolucion, width="stretch")

elif emocion_evolucion in ("SIETE_F", "SIETE_M"):
    genero_elegido = "Femenino" if emocion_evolucion == "SIETE_F" else "Masculino"
    df_genero_evolucion = df_filtrado[df_filtrado["Gender_ES"] == genero_elegido]

    evolucion_genero = (
        df_genero_evolucion.groupby("Oscar_Year")[emociones_evolucion_activas]
        .mean()
        .reset_index()
        .melt(id_vars="Oscar_Year", var_name="Emoción_col", value_name="Media")
    )
    evolucion_genero["Emoción"] = evolucion_genero["Emoción_col"].map(NOMBRES_EMOCIONES)

    fig_evolucion = px.line(
        evolucion_genero, x="Oscar_Year", y="Media", color="Emoción",
        title=f"Media de cada emoción por año — personajes {genero_elegido.lower()}s",
        labels={"Oscar_Year": "Año", "Media": "Media", "Emoción": "Emoción"},
        markers=True,
        category_orders={"Emoción": [NOMBRES_EMOCIONES[e] for e in emociones_evolucion_activas]},
    )
    st.plotly_chart(fig_evolucion, width="stretch")
    st.caption(f"Personajes {genero_elegido.lower()}s en este recorte: **{len(df_genero_evolucion):,}**")

else:
    evolucion = (
        df_filtrado.groupby(["Oscar_Year", "Gender_ES"])[emocion_evolucion]
        .mean()
        .reset_index()
    )

    fig_evolucion = px.line(
        evolucion, x="Oscar_Year", y=emocion_evolucion, color="Gender_ES",
        title=f"Media de {NOMBRES_EMOCIONES[emocion_evolucion].lower()} por año y género",
        labels={"Oscar_Year": "Año", emocion_evolucion: f"Media de {NOMBRES_EMOCIONES[emocion_evolucion].lower()}", "Gender_ES": "Género"},
        color_discrete_map=COLORES_GENERO,
        markers=True,
    )
    st.plotly_chart(fig_evolucion, width="stretch")
    st.caption(f"Personajes en este recorte: **{len(df_filtrado):,}**. " + texto_exclusion)

st.caption(
    "Cada punto es la media de esa emoción entre todos los personajes de ese "
    "año y género que cumplen los filtros generales de arriba. Algunos años "
    "tienen pocos personajes (mínimo ~65), así que las oscilaciones bruscas "
    "de un año a otro pueden deberse al tamaño de muestra, no a una tendencia real."
)

st.divider()

# ----------------------------
# COMPARACIÓN DETALLADA POR EMOCIÓN (barras)
# ----------------------------
st.subheader("Comparación detallada, emoción por emoción")

excluir_neutro_barras = pill_excluir_neutro("pill_barras")
emociones_barras = [e for e in EMOCIONES if not (excluir_neutro_barras and e == "Emotion_Neutral")]

promedio_largo = df_filtrado.groupby("Gender_ES")[emociones_barras].mean().reset_index().melt(
    id_vars="Gender_ES", var_name="Emoción", value_name="Probabilidad promedio"
)
promedio_largo["Emoción"] = promedio_largo["Emoción"].map(NOMBRES_EMOCIONES)

fig_barras_emociones = px.bar(
    promedio_largo, x="Emoción", y="Probabilidad promedio", color="Gender_ES", barmode="group",
    title="Probabilidad promedio de cada emoción, por género",
    color_discrete_map=COLORES_GENERO,
    text="Probabilidad promedio",
)
fig_barras_emociones.update_traces(texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False)
st.plotly_chart(fig_barras_emociones, width="stretch")

st.caption(f"Personajes en este recorte: **{len(df_filtrado):,}**. " + texto_exclusion)

st.divider()