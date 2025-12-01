# =================================================================
# ARCHIVO: entrenamiento.py
# =================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
import os

# --- 1. CONFIGURACIÓN Y CONSTANTES ---
# (Aplicando la sugerencia de no "hardcodear")
RUTA_DATOS = 'data/data_neojaus.csv'
CARPETA_MODELOS = 'saved_models/'
UMBRAL_MINIMO_ZONA = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

FINAL_NUMERIC_FEATURES = [
    'latitude',
    'longitude',
    'construction_area',
    'land_area',
    'rooms',
    'bathrooms',
    'parking_spots',
    'half_bathrooms'
]

# Lista con las 3 features categóricas de tu selección
FINAL_CATEGORICAL_FEATURES = [
    'neighborhood',
    'municipality',
    'property_type'
]

# Lista con las 14 features binarias de tu selección
FINAL_BINARY_FEATURES = [
    'walk_in_closets',
    'step_free_entryway',
    'green_areas',
    'clubhouse',
    'is_condo',
    'stainless_steel_appliances',
    'bike_parking',
    'washing_room',
    'pet_park',
    'soccer_court',
    'wifi_common_areas',
    'guests_parking',
    'granite_countertops',
    'bathtub'
]

# --- 2. FUNCIONES AUXILIARES (MODULARIDAD) ---

def _cargar_y_filtrar_datos(ruta: str) -> pd.DataFrame:
    """Carga los datos y aplica el filtro inicial."""
    df = pd.read_csv(ruta)
    df_venta = df[(df['is_venta'] == 't') & (df['pricing_sales_price'].notna())].copy()

    # Eliminar outliers
    y = df_venta['pricing_sales_price']
    Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 3 * IQR
    mask_outliers = y <= upper_bound

    print(f"Datos después de filtrar outliers: {mask_outliers.sum()} filas")
    return df_venta[mask_outliers]


def _preparar_features(df: pd.DataFrame) -> (pd.DataFrame, dict, dict):
    """
    Preprocesa los datos usando las listas finales de features:
    convierte binarias, imputa nulos y codifica categóricas.
    """
    # 1. Combina las 3 listas finales para obtener el set completo
    features_usadas = FINAL_NUMERIC_FEATURES + FINAL_CATEGORICAL_FEATURES + FINAL_BINARY_FEATURES
    X = df[features_usadas].copy()

    # 2. Convierte las binarias seleccionadas a 0/1
    for col in FINAL_BINARY_FEATURES:
        X[col] = (X[col] == 't').astype(int)

    # 3. Imputa nulos en las numéricas
    imputer_numeric = SimpleImputer(strategy='median')
    X[FINAL_NUMERIC_FEATURES] = imputer_numeric.fit_transform(X[FINAL_NUMERIC_FEATURES])

    # 4. Codifica categóricas
    label_encoders = {}
    for col in FINAL_CATEGORICAL_FEATURES:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    return X, imputer_numeric, label_encoders


# --- 3. FUNCIÓN PRINCIPAL (ORQUESTADORA) ---

def entrenar_y_guardar_modelo():
    """
    Función principal que orquesta el pipeline de entrenamiento.
    """
    print("Iniciando el pipeline de entrenamiento...")

    # Cargar y limpiar datos
    df_clean = _cargar_y_filtrar_datos(RUTA_DATOS)

    # Preparar features y target
    X, imputer_numeric, label_encoders = _preparar_features(df_clean)
    y_price = df_clean['pricing_sales_price']

    # Dividir en train y test para una evaluación realista
    X_train, X_test, y_train_price, y_test_price = train_test_split(
        X, y_price, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Datos de entrenamiento: {X_train.shape}, Datos de test: {X_test.shape}")

    # Entrenar Regresor
    print("\nEntrenando modelo de regresión...")
    regresor = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    regresor.fit(X_train, y_train_price)
    test_score_r2 = regresor.score(X_test, y_test_price)
    print(f"R² Score en Test: {test_score_r2:.3f}")  # ¡Métrica mucho más confiable!

    # Entrenar Clasificador (usando todos los datos para definir categorías)
    print("\nEntrenando modelo de clasificación...")
    precios_estimados = regresor.predict(X)
    desviacion = ((y_price - precios_estimados) / precios_estimados) * 100

    def clasificar_precio(d):
        if d <= -15:
            return "BAJO"
        elif d >= 15:
            return "ALTO"
        else:
            return "NORMAL"

    y_category = desviacion.apply(clasificar_precio)

    # Dividir de nuevo con el target categórico
    X_train_clf, X_test_clf, y_train_cat, y_test_cat = train_test_split(
        X, y_category, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_category
    )

    clasificador = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE,
                                          n_jobs=-1)
    clasificador.fit(X_train_clf, y_train_cat)
    test_accuracy = clasificador.score(X_test_clf, y_test_cat)
    print(f"Accuracy en Test: {test_accuracy:.3f}")
    # =============================================================
    # NUEVO: ANÁLISIS DE IMPORTANCIA DE FEATURES
    # =============================================================
    print("\n=== ANALIZANDO IMPORTANCIA DE FEATURES ===")
    importancias = clasificador.feature_importances_
    df_importancias = pd.DataFrame({
        'feature': X.columns,  # Usamos X para tener los nombres originales
        'importance': importancias
    }).sort_values('importance', ascending=False)

    print("Top 25 features más importantes:")
    print(df_importancias.head(25))

    # Guardar el gráfico para revisión
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 12))
    sns.barplot(x='importance', y='feature', data=df_importancias.head(25))
    plt.title('Top 25 Features más Importantes')
    plt.tight_layout()
    plt.savefig(f'{CARPETA_MODELOS}importancia_features.png')
    print("Gráfico de importancia guardado en 'modelos_guardados/importancia_features.png'")
    # plt.show() # Puedes comentar plt.show() para que no detenga el script

    # Guardar artefactos
    print("\nGuardando artefactos...")
    os.makedirs(CARPETA_MODELOS, exist_ok=True)
    joblib.dump(clasificador, f'{CARPETA_MODELOS}clasificador_precios.pkl')
    joblib.dump(imputer_numeric, f'{CARPETA_MODELOS}imputer_numeric.pkl')
    joblib.dump(label_encoders, f'{CARPETA_MODELOS}label_encoders.pkl')

    # Guardar zonas confiables y listas de features
    conteo_por_barrio = X['neighborhood'].value_counts()
    zonas_confiables = conteo_por_barrio[conteo_por_barrio >= UMBRAL_MINIMO_ZONA].index.tolist()
    joblib.dump(zonas_confiables, f'{CARPETA_MODELOS}zonas_confiables.pkl')
    joblib.dump(FINAL_NUMERIC_FEATURES + FINAL_CATEGORICAL_FEATURES + FINAL_BINARY_FEATURES, f'{CARPETA_MODELOS}features_usadas.pkl')

    print("✅ Pipeline completado.")


if __name__ == '__main__':
    entrenar_y_guardar_modelo()