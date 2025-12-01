import pandas as pd

# --- Define las listas que ya conoces de tu script de entrenamiento ---
NUMERIC_FEATURES = [
    'rooms', 'bathrooms', 'half_bathrooms', 'parking_spots', 'construction_area',
    'land_area', 'longitude', 'latitude', 'views', 'likes', 'shares', 'inquiries', 'appears'
]
CATEGORICAL_FEATURES = ['property_type', 'municipality', 'neighborhood']

# --- Carga el CSV para descubrir las columnas binarias ---
df = pd.read_csv('data/data_neojaus.csv')
# Usamos el mismo filtro inicial para ser consistentes
df_venta = df[df['is_venta'] == 't'].copy()

# --- Lógica exacta de tu notebook para encontrar las columnas binarias ('t'/'f') ---
binary_features = [
    col for col in df_venta.columns
    if set(df_venta[col].dropna().unique()) == {'t', 'f'}
]

# --- Combina y muestra la lista final ---
features_usadas = NUMERIC_FEATURES + CATEGORICAL_FEATURES + binary_features

print("=============================================")
print("=== LISTA DE COLUMNAS PARA EL MODELO ===")
print("=============================================")
# Imprime una columna por línea para que sea fácil de leer
for feature in features_usadas:
    print(feature)

print("\n\n=============================================")
print("=== LISTA PARA COPIAR Y PEGAR ===")
print("=============================================")
# Imprime la lista completa en formato de Python para copiarla fácilmente
print(str(features_usadas))