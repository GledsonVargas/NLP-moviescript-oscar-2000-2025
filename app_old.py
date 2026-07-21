import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="NLP y Género en el Cine",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 NLP y Género en el Cine")
st.markdown("### Proyecto de análisis de guiones y representación de género")

# ----------------------------
# CARGA DE DATOS
# ----------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_pickle("Dataset_final.pkl")  # cambia ruta si hace falta
    return df

df = cargar_datos()

# ----------------------------
# AJUSTA AQUÍ EL NOMBRE DE LA COLUMNA
# ----------------------------
col_genero = "Genres"  # cambia esto por el nombre real de tu columna

# ----------------------------
# LIMPIEZA Y SEPARACIÓN DE GÉNEROS
# ----------------------------
df[col_genero] = df[col_genero].astype(str)

# Separamos por coma y explotamos para que cada género sea una fila
generos_explotados = (
    df[col_genero]
    .str.split(",")
    .explode()
    .str.strip()
)

# Quitamos valores vacíos o nulos
generos_explotados = generos_explotados[generos_explotados != ""]
generos_explotados = generos_explotados.dropna()

# ----------------------------
# CÁLCULO DE FRECUENCIAS Y PORCENTAJES
# ----------------------------
conteo = generos_explotados.value_counts()
porcentaje = (conteo / conteo.sum()) * 100

tabla_generos = pd.DataFrame({
    "Género": conteo.index,
    "Cantidad": conteo.values,
    "Porcentaje (%)": porcentaje.values
})

# ----------------------------
# MÉTRICAS EN FORMATO TARJETAS
# ----------------------------
st.subheader("📌 Porcentaje por género")

# mostramos máximo 6 métricas para que no se rompa el layout
top_n = 5
tabla_top = tabla_generos.head(top_n)

cols = st.columns(len(tabla_top))

for i, row in enumerate(tabla_top.itertuples(index=False)):
    cols[i].metric(
        label=row[0],  # Género
        value=f"{row[2]:.2f}%",
        delta=f"{row[1]} apariciones"
    )

# ----------------------------
# GRÁFICO DE BARRAS
# ----------------------------
st.subheader("📊 Gráfico de barras")

fig = px.bar(
    tabla_generos,
    x="Género",
    y="Porcentaje (%)",
    text="Porcentaje (%)",
    title="Distribución porcentual de géneros (conteo individual)"
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>Porcentaje: %{y:.2f}%<extra></extra>"
)    
fig.update_layout(
    yaxis_title="Porcentaje (%)",
    xaxis_title="Género",
    uniformtext_minsize=10,
    uniformtext_mode="hide"
)

st.plotly_chart(fig, width='stretch')

# ----------------------------
# TABLA COMPLETA
# ----------------------------
tabla_generos["Porcentaje (%)"] = tabla_generos["Porcentaje (%)"].round(2)
st.subheader("📋 Tabla de distribución completa")
st.dataframe(tabla_generos, width='stretch')