import pickle
import spacy

# Cargar modelo
nlp = spacy.load("en_core_web_lg")

# Cargar dataset
with open("Dataset_final.pkl", "rb") as f:
    df = pickle.load(f)

# Función que calcula el ratio de agencia de un texto
def calcular_agencia(texto):
    doc = nlp(texto)
    nsubj = sum(1 for token in doc if token.dep_ == "nsubj")
    nsubjpass = sum(1 for token in doc if token.dep_ == "nsubjpass")
    total = nsubj + nsubjpass
    if total == 0:
        return None
    return nsubj / total

# Calcular agencia para todos los personajes del dataset
resultados = []

for _, pelicula in df.iterrows():
    script = pelicula['Script_Dict']
    generos = pelicula['Characters_Genders']
    
    for personaje, texto in script.items():
        genero = generos.get(personaje, 'unknown')
        if genero == 'unknown':
            continue
        ratio = calcular_agencia(texto)
        if ratio is None:
            continue
        resultados.append({
            'titulo': pelicula['Title'],
            'premio': pelicula['Premio'],
            'personaje': personaje,
            'genero': genero,
            'agencia': ratio
        })

print(f"\nTotal personajes analizados: {len(resultados)}")
print(f"Ejemplo: {resultados[0]}")

import pandas as pd

# Convertir a DataFrame
df_agencia = pd.DataFrame(resultados)

# Resumen por género
resumen = df_agencia.groupby('genero')['agencia'].agg(['mean', 'std', 'count']).round(3)
resumen.columns = ['Media', 'Desviación', 'Personajes']
print("\nAgencia narrativa por género:")
print(resumen)

# Resumen por género y categoría de premio
resumen_premio = df_agencia.groupby(['premio', 'genero'])['agencia'].mean().round(3).unstack()
print("\nAgencia por categoría de premio:")
print(resumen_premio)