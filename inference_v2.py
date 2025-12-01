# =================================================================
# FILE: inference_v2.py (Live Prediction Script - Robust Version)
# =================================================================

import pandas as pd
import joblib
import os

# --- 1. CONFIGURATION AND CONSTANTS ---

MODELS_FOLDER = 'saved_models_v2/'
TARGET_VARIABLE = 'valorComercial'

# These lists MUST match the ones used in 'training_v2.py'
FINAL_NUMERIC_FEATURES = [
    'superficieConstruccionM2', 'banios', 'espaciosEstacionamiento',
    'recamaras', 'latitud', 'longitud', 'edad'
]
FINAL_CATEGORICAL_FEATURES = [
    'entidadFederativa', 'delegacionMunicipio', 'colonia',
    'tipoBien', 'estadoConservacion'
]
FEATURES_USED = FINAL_NUMERIC_FEATURES + FINAL_CATEGORICAL_FEATURES

# --- 2. LOAD ALL ARTIFACTS ON STARTUP ---

print("Loading model artifacts...")
try:
    REGRESSOR = joblib.load(os.path.join(MODELS_FOLDER, 'regressor_model.pkl'))
    IMPUTER_NUMERIC = joblib.load(os.path.join(MODELS_FOLDER, 'imputer_numeric.pkl'))
    RELIABLE_ZONES = joblib.load(os.path.join(MODELS_FOLDER, 'reliable_zones.pkl'))
    THRESHOLDS = joblib.load(os.path.join(MODELS_FOLDER, 'thresholds.pkl'))
    print("✅ Artifacts loaded successfully.")
except FileNotFoundError:
    print(f"❌ ERROR: Model artifacts not found in '{MODELS_FOLDER}'.")
    print("Please run 'training_v2.py' first.")
    REGRESSOR = None


# --- 3. THE PREDICTION FUNCTION ---

def get_property_label(property_data: dict) -> dict:
    """
    Takes a dictionary of a single property, runs all business logic,
    and returns a classification label and details.
    """
    if not REGRESSOR:
        return {"error": "Model is not loaded. Please check server."}

    # ================================================================
    # NEW: Standardize all categorical inputs to UPPERCASE
    # ================================================================
    for col in FINAL_CATEGORICAL_FEATURES:
        if isinstance(property_data.get(col), str):
            property_data[col] = property_data[col].upper()
    # ================================================================

    # --- Gate 1: Check if the 'colonia' is reliable ---
    try:
        colonia = property_data['colonia']
        if colonia not in RELIABLE_ZONES:
            return {
                "error": f"No hay suficientes datos en la colonia '{colonia}' para una estimación confiable."
            }
    except KeyError:
        return {"error": "Faltan datos de 'colonia'."}

    # --- Step 2: Prepare the data ---
    try:
        X_pred = pd.DataFrame([property_data])
        X_pred = X_pred.reindex(columns=FEATURES_USED)

        X_pred[FINAL_NUMERIC_FEATURES] = IMPUTER_NUMERIC.transform(X_pred[FINAL_NUMERIC_FEATURES])

        for col in FINAL_CATEGORICAL_FEATURES:
            X_pred[col] = X_pred[col].astype('category')

    except Exception as e:
        return {"error": f"Error al preprocesar los datos: {e}"}

    # --- Step 3: Predict the price (The Core Logic) ---
    estimated_price = REGRESSOR.predict(X_pred)[0]

    # --- Step 4: Apply Business Logic to Classify ---
    try:
        real_price = property_data[TARGET_VARIABLE]
        deviation = ((real_price - estimated_price) / estimated_price) * 100

        low_thresh = THRESHOLDS['low_threshold_pct']
        high_thresh = THRESHOLDS['high_threshold_pct']

        if deviation <= low_thresh:
            label = "LOW"
        elif deviation >= high_thresh:
            label = "HIGH"
        else:
            label = "NORMAL"

        return {
            "label": label,
            "real_price": float(real_price),
            "estimated_price": float(round(estimated_price, 2)),
            "deviation_pct": float(round(deviation, 2))
        }

    except KeyError:
        return {"error": f"Falta el dato '{TARGET_VARIABLE}' para calcular la desviación."}
    except Exception as e:
        return {"error": f"Error al clasificar: {e}"}


# --- 4. TEST BLOCK (to run this file directly) ---
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("--- RUNNING INFERENCE TEST ---")
    print("=" * 50)

    # --- Test Case 1: A reliable property (using mixed/lowercase) ---
    # Now this test should work, because our function converts it to uppercase
    sample_property_good = {
        'superficieConstruccionM2': 114.61,
        'banios': 2,
        'espaciosEstacionamiento': 2,
        'recamaras': 3,
        'latitud': 20.72343,
        'longitud': -103.42333,
        'edad': 6,
        'entidadFederativa': 'JALISCO',  # <--- Lowercase
        'delegacionMunicipio': 'ZAPOPAN',  # <--- Lowercase
        'colonia': 'SANTA MARGARITA',  # <--- Lowercase
        'tipoBien': 'DEPARTAMENTO EN CONDOMINIO',
        'estadoConservacion': 'BUENO',
        'valorComercial': 5787000
    }

    # We need a REAL colonia from your reliable_zones.pkl file for this to pass
    # PLEASE REPLACE 'CENTRO' with a real 'colonia' from your data
    real_colonia = 'CENTRO'  # <--!! CAMBIA ESTO por una colonia real de tu lista

    print(f"\nTest 1: Predicting property in a reliable zone ('{real_colonia}')...")
    sample_property_good['colonia'] = real_colonia
    label = get_property_label(sample_property_good)
    print(label)

    # --- Test Case 2: An unreliable zone ---
    sample_property_bad_zone = sample_property_good.copy()
    sample_property_bad_zone['colonia'] = 'COLONIA INVENTADA 123'

    print(f"\nTest 2: Predicting property in 'COLONIA INVENTADA 123'...")
    label_bad = get_property_label(sample_property_bad_zone)
    print(label_bad)

    # --- Test Case 3: Missing key data ---
    sample_property_missing_data = {'latitud': 19.43, 'longitud': -99.13}

    print(f"\nTest 3: Predicting with missing data...")
    label_missing = get_property_label(sample_property_missing_data)
    print(label_missing)