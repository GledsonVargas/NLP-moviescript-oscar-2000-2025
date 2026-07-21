import streamlit as st
import pandas as pd

st.title("📊 Exploración del dataset")

# Primer dataset
@st.cache_data
def cargar_dataset():
    return pd.read_parquet("Dataset_final.parquet")

df = cargar_dataset()

# Según dataset
@st.cache_data
def cargar_dataset():
    return pd.read_pickle("personajes_rol_bechdel.pkl")

df2 = cargar_dataset()

# Texto en la página

st.success("Dataset cargado correctamente.")
st.write("Filas:", df.shape[0])
st.write("Columnas:", df.shape[1])

st.write("Vista previa:")
st.dataframe(df)

st.write("Filas:", df2.shape[0])
st.write("Columnas:", df2.shape[1])
st.dataframe(df2)