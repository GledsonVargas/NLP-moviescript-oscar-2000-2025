import re
 
import streamlit as st
import pandas as pd
import plotly.express as px
 
st.set_page_config(
    page_title="NLP y Género en el Cine",
    page_icon="🎬",
    layout="wide"
)

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

 
st.title("NLP y narrativa cinematográfica")
st.markdown("### Análisis de la representación femenina y de la evolución discursiva en los guiones ganadores del Óscar (2000–2025)")
 
st.markdown(
    """
    El cine en particular, y los medios de comunicación de masas en general, actúan como poderosos mecanismos que proyectan y consolidan los valores predominantes de la sociedad. En este contexto, el cine contemporáneo surge como un punto de encuentro entre patrones sociales, psicológicos y lingüísticos que no solo buscan entretener, sino que también moldean activamente las estructuras colectivas. Dada esta profunda influencia sobre la cultura popular, las producciones cinematográficas funcionan con frecuencia como un reflejo de las actitudes culturales en torno a los roles y expectativas de género.
    En consecuencia, la representación de sesgos en la pantalla influye en las creencias del público, fomentando un entorno en el que los contenidos audiovisuales no solo reflejan la realidad, sino que también establecen normas de comportamiento predominantes (Yu et al., 2022). Esta dinámica perpetúa un ciclo en el que los estereotipos sociales condicionan la perspectiva de los cineastas —predominantemente hombres— quienes, a su vez, producen obras que terminan validando dichos prejuicios en el imaginario colectivo (Simonton, 2004).
    
    No obstante, más allá de la perpetuación de los sesgos, el cine también posee un potencial transformador. Puede convertirse en una herramienta de activismo al reconfigurar las percepciones sociales, cuestionar los arquetipos heredados y fomentar un discurso global más inclusivo y equitativo (Piyumali y Sandaruwan, 2025).
    En este marco, el estudio de los diálogos adquiere un especial interés analítico, ya que las narrativas verbales presentes en la pantalla contribuyen a la construcción del imaginario colectivo (Cape, 2003; Schofield y Mehr, 2016). Así, la convergencia entre el impacto sociocultural del cine, la riqueza lingüística de sus guiones y la disponibilidad actual de grandes corpus textuales convierte al diálogo cinematográfico en un escenario idóneo para la investigación.
    
    En este ámbito, el análisis computacional basado en el Procesamiento del Lenguaje Natural (PLN) se ha consolidado como una herramienta fundamental para identificar estereotipos a partir de los patrones lingüísticos de los personajes (Martínez et al., 2022). Las técnicas de PLN permiten procesar grandes volúmenes de texto para realizar análisis de sentimiento, modelado de temas (topic modeling) y detección de patrones (Cini, 2025). De este modo, los investigadores en ciencias sociales computacionales pueden examinar las dinámicas de género y las desigualdades presentes en las narrativas contemporáneas mediante metodologías empíricas y automatizadas que ofrecen un elevado grado de precisión en la medición de la representación de género, transformando los datos cualitativos de los guiones en métricas cuantitativas rigurosas (Kagan et al., 2020).
    Sobre esta base, la presente investigación tiene como objetivo analizar la representación de las mujeres en el cine comercial y el impacto que ejerce la presencia femenina en puestos clave de liderazgo creativo, concretamente en la dirección y el guion, sobre dichas narrativas. Para ello, el estudio aplica técnicas de Procesamiento del Lenguaje Natural (PLN) y minería de texto a los guiones completos de las películas ganadoras del Premio de la Academia en las categorías de Mejor Guion Adaptado, Mejor Guion Original y Mejor Película entre los años 2000 y 2025. Mediante la medición cuantitativa de la densidad discursiva, la polaridad emocional y la evolución de los arquetipos femeninos, este trabajo pretende determinar si la reciente apertura institucional de la Academia se traduce en un cambio estructural y paritario dentro del discurso cinematográfico de las películas galardonadas (Wilk, 2024). En última instancia, esta investigación pone de relieve la importancia de los estudios que evidencian el valor social de la inclusión en los ámbitos de la narrativa y la creación audiovisual.
    """
)
 
st.divider()
 
# ----------------------------
# CARGA DE DATOS
# ----------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_pickle("Dataset_final.pkl")
    return df
 
def split_sentences(texto):
    oraciones = re.split(r'(?<=[.!?])\s+', texto.strip())
    return [o for o in oraciones if len(o.strip()) > 0]
 
@st.cache_data
def calcular_total_frases(df):
    total = 0
    for _, row in df.drop_duplicates(subset="IMDb_ID").iterrows():
        for texto in row["Script_Dict"].values():
            if isinstance(texto, str) and texto.strip():
                total += len(split_sentences(texto))
    return total
 
df = cargar_datos()
df_unicas = df.drop_duplicates(subset="IMDb_ID")
 
# ----------------------------
# MÉTRICAS GENERALES DEL DATASET
# ----------------------------
st.subheader("Vista general del dataset")
st.caption(
    "Estas cifras incluyen las 77 nominaciones completas, incluyendo 2 películas "
    "(Talk to Her y Anatomy of a Fall) que se excluyen de los análisis de NLP "
    "por no estar predominantemente en inglés. Por eso los totales aquí son "
    "ligeramente superiores a los de las páginas de análisis de NLP."
)
 
total_personajes = (
    df["Male_Characters_Count"].sum()
    + df["Female_Characters_Count"].sum()
    + df["Unknown_Characters_Count"].sum()
)
total_palabras = df["Words_Male"].sum() + df["Words_Female"].sum() + df["Words_Unknown"].sum()
total_frases = calcular_total_frases(df)
 
resumen_genero = pd.DataFrame({
    "Género": ["Masculino", "Femenino", "Desconocido"],
    "Personajes": [
        df["Male_Characters_Count"].sum(),
        df["Female_Characters_Count"].sum(),
        df["Unknown_Characters_Count"].sum(),
    ],
    "Palabras totales": [
        df["Words_Male"].sum(),
        df["Words_Female"].sum(),
        df["Words_Unknown"].sum(),
    ],
})
 
colores_genero = {"Masculino": "#3B6EA5", "Femenino": "#C1447E", "Desconocido": "#b0b0b0"}
 
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Películas únicas", f"{df_unicas.shape[0]}")
col2.metric("Nominaciones registradas", f"{len(df)}")
col3.metric("Personajes analizados", f"{int(total_personajes):,}")
col4.metric("Palabras", f"{int(total_palabras):,}")
col5.metric("Frases totales", f"{total_frases:,}")
 
st.markdown("")
 
col_izq, col_der = st.columns(2)
with col_izq:
    fig_personajes = px.pie(
        resumen_genero, names="Género", values="Personajes",
        title="Distribución de personajes por género",
        color="Género", color_discrete_map=colores_genero, hole=0.45
    )
    fig_personajes.update_traces(textinfo="value+percent", texttemplate="%{value:,}<br>(%{percent})")
    st.plotly_chart(fig_personajes, width="stretch")
with col_der:
    fig_palabras = px.pie(
        resumen_genero, names="Género", values="Palabras totales",
        title="Distribución de palabras de diálogo por género",
        color="Género", color_discrete_map=colores_genero, hole=0.45
    )
    fig_palabras.update_traces(textinfo="value+percent", texttemplate="%{value:,}<br>(%{percent})")
    st.plotly_chart(fig_palabras, width="stretch")
 
st.divider()

 
# ----------------------------
# DIRECTORES Y GUIONISTAS POR GÉNERO
# ----------------------------
st.subheader("Directores y guionistas por género")
 
rol_seleccionado = st.segmented_control(
    "Selecciona qué rol quieres ver:",
    options=["Directores", "Guionistas"],
    default="Directores",
)
 
TRADUCCION_AWARD_ROL = {
    "Best Picture": "Mejor Película",
    "Adapted Screenplay": "Mejor Guion Adaptado",
    "Original Screenplay": "Mejor Guion Original",
}
TRADUCCION_AWARD_ROL_INV = {v: k for k, v in TRADUCCION_AWARD_ROL.items()}
 
award_rol_seleccionado = st.segmented_control(
    "Categoría de Oscar",
    options=["Todas"] + list(TRADUCCION_AWARD_ROL.values()),
    default="Todas",
)
 
columna_male = "male_director" if rol_seleccionado == "Directores" else "male_writer"
columna_female = "female_director" if rol_seleccionado == "Directores" else "female_writer"
 
if award_rol_seleccionado and award_rol_seleccionado != "Todas":
    # Filtramos por categoria de Oscar: cada fila ya es una nominacion, no hace falta deduplicar
    base_rol = df[df["Award"] == TRADUCCION_AWARD_ROL_INV[award_rol_seleccionado]]
    sufijo_titulo = f" — {award_rol_seleccionado}"
else:
    # Sin filtro de Award: usamos peliculas unicas para no contar dos veces
    # a un mismo director/guionista si su pelicula fue nominada en varias categorias
    base_rol = df_unicas
    sufijo_titulo = " (todas las categorías, por película única)"
 
resumen_rol = pd.DataFrame({
    "Género": ["Masculino", "Femenino"],
    "Cantidad": [base_rol[columna_male].sum(), base_rol[columna_female].sum()]
})
titulo_rol = f"Distribución de {rol_seleccionado.lower()} por género{sufijo_titulo}"
 
col_grafico, col_texto = st.columns([2, 1])
 
with col_grafico:
    fig_rol = px.pie(
        resumen_rol, names="Género", values="Cantidad",
        title=titulo_rol, color="Género", color_discrete_map=colores_genero, hole=0.45
    )
    fig_rol.update_traces(textinfo="label+value", textposition="outside")
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
# GÉNEROS CINEMATOGRÁFICOS (drama, comedia, etc.)
# ----------------------------
st.subheader("Géneros cinematográficos de las películas")
st.caption("Nota: esto se refiere al género del filme (drama, comedia...), no al género de los personajes.")
 
vista_genero_cine = st.radio(
    "¿Cómo quieres ver la distribución?",
    ["Total", "Por categoría de Oscar (Award)", "Por representación de género de personajes"],
    horizontal=True
)
 
col_genero_cine = "Genres"
df[col_genero_cine] = df[col_genero_cine].astype(str)
 
if vista_genero_cine == "Total":
    generos_explotados = (
        df_unicas[col_genero_cine].str.split(",").explode().str.strip()
    )
    generos_explotados = generos_explotados[generos_explotados != ""].dropna()
 
    conteo = generos_explotados.value_counts()
    tabla_generos = pd.DataFrame({
        "Género cinematográfico": conteo.index,
        "Cantidad": conteo.values,
        "Porcentaje (%)": (conteo / conteo.sum() * 100).round(2).values
    })
 
    fig = px.bar(
        tabla_generos, x="Género cinematográfico", y="Porcentaje (%)", text="Porcentaje (%)",
        title="Distribución porcentual de géneros cinematográficos"
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, width="stretch")
    with st.expander("Ver tabla completa"):
        st.dataframe(tabla_generos, width="stretch")
 
elif vista_genero_cine == "Por categoría de Oscar (Award)":
    tmp = df[["Award", col_genero_cine]].copy()
    tmp[col_genero_cine] = tmp[col_genero_cine].str.split(",")
    tmp = tmp.explode(col_genero_cine)
    tmp[col_genero_cine] = tmp[col_genero_cine].str.strip()
    tmp = tmp[tmp[col_genero_cine] != ""].dropna()
 
    conteo_award_genero = tmp.groupby(["Award", col_genero_cine]).size().reset_index(name="Cantidad")
 
    fig = px.bar(
        conteo_award_genero, x=col_genero_cine, y="Cantidad", color="Award",
        title="Géneros cinematográficos por categoría de Oscar", barmode="stack"
    )
    st.plotly_chart(fig, width="stretch")
 
else:  # Por representación de género de personajes
    tmp = df_unicas[[col_genero_cine, "Male_Characters_Count", "Female_Characters_Count"]].copy()
    tmp[col_genero_cine] = tmp[col_genero_cine].str.split(",")
    tmp = tmp.explode(col_genero_cine)
    tmp[col_genero_cine] = tmp[col_genero_cine].str.strip()
    tmp = tmp[tmp[col_genero_cine] != ""].dropna()
 
    resumen = tmp.groupby(col_genero_cine)[["Male_Characters_Count", "Female_Characters_Count"]].mean().reset_index()
    resumen = resumen.rename(columns={
        "Male_Characters_Count": "Promedio personajes masculinos",
        "Female_Characters_Count": "Promedio personajes femeninos"
    })
    resumen_largo = resumen.melt(id_vars=col_genero_cine, var_name="Tipo", value_name="Promedio")
 
    fig = px.bar(
        resumen_largo, x=col_genero_cine, y="Promedio", color="Tipo", barmode="group",
        title="Promedio de personajes masculinos/femeninos por género cinematográfico",
        color_discrete_map={"Promedio personajes masculinos": "#3B6EA5", "Promedio personajes femeninos": "#C1447E"}
    )
    st.plotly_chart(fig, width="stretch")
 
st.divider()

 
# ----------------------------
# TABLA COMPLETA DEL DATASET
# ----------------------------
st.subheader("Explorar el dataset completo")

columnas_mostrar = [
    "Title", "Oscar_Year", "Award", "Genres", "Rating_Score",
    "Top_Cast", "Director", "male_director", "female_director",
    "Writers", "male_writer", "female_writer", "Synopsis",
    "Script_Dict", "Characters_Genders", "Male_Characters_Count", "Female_Characters_Count", 
    "Unknown_Characters_Count", "Words_Male", "Words_Female", "Words_Unknown",
    "AverageWords_male", "AverageWords_female", "AverageWords_unknown"
]
 
nombres_legibles = {
    "Title": "Título",
    "Oscar_Year": "Año",
    "Award": "Oscar",
    "Genres": "Géneros",
    "Rating_Score": "Puntuación IMDB",
    "Top_Cast": "Reparto principal",
    "Director": "Director/a",
    "male_director": "Directores (H)",
    "female_director": "Directoras (M)",
    "Writers": "Guionistas",
    "male_writer": "Guionistas (H)",
    "female_writer": "Guionistas (M)",
    "Synopsis": "Sinopsis",
    "Script_Dict" : "Guion", 
    "Characters_Genders" : "Género de los personajes",
    "Male_Characters_Count": "Personajes masculinos",
    "Female_Characters_Count": "Personajes femeninos",
    "Unknown_Characters_Count" : "Personajes género desconocido",
    "Words_Male": "Palabras (H)",
    "Words_Female": "Palabras (M)",
    "Words_Unknown" : "Palabras (Desc.)",
    "AverageWords_male" : "Media palabras por personaje (H)",
    "AverageWords_female" : "Media palabras por personaje (M)",
    "AverageWords_unknown" : "Media palabras por personaje (Desc)"
}
 
tabla_final = df[columnas_mostrar].rename(columns=nombres_legibles)
st.dataframe(tabla_final, width="stretch")
