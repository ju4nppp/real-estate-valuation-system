import gradio as gr
import joblib
import os
import pandas as pd

# ================================================================
# ESTO ES UNA VERSIÓN SIMPLIFICADA DE TU inference_v2.py
# ADAPTADA PARA GRADIO
# ================================================================

# --- 1. Cargar Artefactos ---
MODELS_FOLDER = 'app_models/'  # La carpeta que creaste

REGRESSOR = joblib.load(os.path.join(MODELS_FOLDER, 'regressor_model.pkl'))
IMPUTER_NUMERIC = joblib.load(os.path.join(MODELS_FOLDER, 'imputer_numeric.pkl'))
RELIABLE_ZONES = joblib.load(os.path.join(MODELS_FOLDER, 'reliable_zones.pkl'))
THRESHOLDS = joblib.load(os.path.join(MODELS_FOLDER, 'thresholds.pkl'))

# Listas de Features (necesarias para el preprocesamiento)
FINAL_NUMERIC_FEATURES = [
    'superficieConstruccionM2', 'banios', 'espaciosEstacionamiento',
    'recamaras', 'latitud', 'longitud', 'edad'
]
FINAL_CATEGORICAL_FEATURES = [
    'entidadFederativa', 'delegacionMunicipio', 'colonia',
    'tipoBien', 'estadoConservacion'
]
FEATURES_USED = FINAL_NUMERIC_FEATURES + FINAL_CATEGORICAL_FEATURES


# --- 2. La Función de Inferencia (El "cerebro" de la App) ---

def get_property_label_demo(
        # Recibimos las 12 variables de los campos de Gradio
        superficieConstruccionM2, banios, espaciosEstacionamiento, recamaras, latitud, longitud, edad,
        entidadFederativa, delegacionMunicipio, colonia, tipoBien, estadoConservacion,
        valorComercial  # El precio real para comparar
):
    # Ensamblar el diccionario de datos
    property_data = {
        'superficieConstruccionM2': superficieConstruccionM2,
        'banios': banios,
        'espaciosEstacionamiento': espaciosEstacionamiento,
        'recamaras': recamaras,
        'latitud': latitud,
        'longitud': longitud,
        'edad': edad,
        'entidadFederativa': entidadFederativa,
        'delegacionMunicipio': delegacionMunicipio,
        'colonia': colonia,
        'tipoBien': tipoBien,
        'estadoConservacion': estadoConservacion,
        'valorComercial': valorComercial
    }

    # --- Lógica de inferencia (copiada de inference_v2.py) ---

    # 1. Estandarizar a mayúsculas
    for col in FINAL_CATEGORICAL_FEATURES:
        if isinstance(property_data.get(col), str):
            property_data[col] = property_data[col].upper()

    # 2. Validar Zona
    colonia_upper = property_data['colonia']
    if colonia_upper not in RELIABLE_ZONES:
        return f"ERROR: Zona no confiable ('{colonia_upper}'). No hay suficientes datos.", "N/A", "N/A"

    # 3. Preparar Datos
    X_pred = pd.DataFrame([property_data])
    X_pred = X_pred.reindex(columns=FEATURES_USED)
    X_pred[FINAL_NUMERIC_FEATURES] = IMPUTER_NUMERIC.transform(X_pred[FINAL_NUMERIC_FEATURES])
    for col in FINAL_CATEGORICAL_FEATURES:
        X_pred[col] = X_pred[col].astype('category')

    # 4. Predecir Precio
    estimated_price = REGRESSOR.predict(X_pred)[0]

    # 5. Etiquetar
    real_price = property_data['valorComercial']
    deviation = ((real_price - estimated_price) / estimated_price) * 100

    low_thresh = THRESHOLDS['low_threshold_pct']
    high_thresh = THRESHOLDS['high_threshold_pct']

    if deviation <= low_thresh:
        label = "LOW 📉"
    elif deviation >= high_thresh:
        label = "HIGH 📈"
    else:
        label = "NORMAL ✅"

    return f"Etiqueta: {label}", f"Precio Estimado: ${estimated_price:,.2f}", f"Desviación: {deviation:,.2f}%"


# --- 3. Crear la Interfaz de Gradio ---

# Creamos 13 sliders/cajas de texto
inputs = [
    # Numéricos
    gr.Number(label="Superficie Construcción (M2)", value=150),
    gr.Number(label="Baños", value=2),
    gr.Number(label="Estacionamiento", value=1),
    gr.Number(label="Recámaras", value=3),
    gr.Number(label="Latitud", value=19.4326),
    gr.Number(label="Longitud", value=-99.1332),
    gr.Number(label="Edad", value=10),
    # Categóricos
    gr.Textbox(label="Entidad Federativa", value="Distrito Federal"),
    gr.Textbox(label="Delegación/Municipio", value="Cuauhtémoc"),
    gr.Textbox(label="Colonia", value="CENTRO"),
    gr.Textbox(label="Tipo de Bien", value="Casa"),
    gr.Textbox(label="Estado de Conservación", value="Bueno"),
    # El precio real
    gr.Number(label="Precio Real (Valor Comercial)", value=5000000)
]

# Creamos 3 cajas de texto para el resultado
outputs = [
    gr.Textbox(label="Resultado"),
    gr.Textbox(label="Tasación Virtual"),
    gr.Textbox(label="Análisis")
]

# Lanzamos la App
demo = gr.Interface(
    fn=get_property_label_demo,
    inputs=inputs,
    outputs=outputs,
    title="Tasadur Virtual v2",
    description="Demo del modelo de precios. Ingresa los datos de una propiedad para estimar su valor y clasificarla."
)

demo.launch()