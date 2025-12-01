# =================================================================
# FILE: training_v2.py (Final - Regressor-Only Logic)
# =================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION AND CONSTANTS ---

DATA_PATH = 'data/subset_pap.csv'
MODELS_FOLDER = 'saved_models_v2/'
TARGET_VARIABLE = 'valorComercial'
MINIMUM_ZONE_COUNT = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

FINAL_NUMERIC_FEATURES = [
    'superficieConstruccionM2',
    'banios',
    'espaciosEstacionamiento',
    'recamaras',
    'latitud',
    'longitud',
    'edad'
]

FINAL_CATEGORICAL_FEATURES = [
    'entidadFederativa',
    'delegacionMunicipio',
    'colonia',
    'tipoBien',
    'estadoConservacion'
]


# --- 2. HELPER FUNCTIONS ---

def _load_and_clean_data(path: str) -> pd.DataFrame:
    """Loads and cleans the dataset."""
    df = pd.read_csv(path)
    df = df[df[TARGET_VARIABLE].notna()].copy()

    # Remove extreme outliers based on price
    y = df[TARGET_VARIABLE]
    Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 3 * IQR
    mask_outliers = (y <= upper_bound)

    print(f"Data after outlier removal: {mask_outliers.sum()} rows")
    return df[mask_outliers].reset_index(drop=True)


def _prepare_features(df: pd.DataFrame):
    """
    Prepares the feature set (X) for LightGBM.
    Handles missing values and sets correct data types for categorical features.
    """
    features_used = FINAL_NUMERIC_FEATURES + FINAL_CATEGORICAL_FEATURES
    X = df[features_used].copy()

    # Impute missing values for numeric features
    imputer_numeric = SimpleImputer(strategy='median')
    X[FINAL_NUMERIC_FEATURES] = imputer_numeric.fit_transform(X[FINAL_NUMERIC_FEATURES])

    # Convert categorical features to 'category' dtype for LightGBM
    for col in FINAL_CATEGORICAL_FEATURES:
        X[col] = X[col].astype('category')

    return X, imputer_numeric


# --- 3. MAIN TRAINING PIPELINE ---

def train_and_save_model():
    """Main pipeline to train and save the regression model and business logic."""
    print("Starting training pipeline (Regressor-Only)...")

    # Load and prepare data
    df_clean = _load_and_clean_data(DATA_PATH)
    X, imputer_numeric = _prepare_features(df_clean)
    y = df_clean[TARGET_VARIABLE]

    # Split data for regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train data: {X_train.shape}, Test data: {X_test.shape}")

    # --- 1. Train Regression Model with LGBMRegressor ---
    print("\nTraining LGBM regression model...")
    regressor = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    regressor.fit(
        X_train, y_train,
        categorical_feature=FINAL_CATEGORICAL_FEATURES
    )
    test_score_r2 = regressor.score(X_test, y_test)
    print(f"R² Score on Test: {test_score_r2:.3f}")

    # --- 2. Calculate Deviation Thresholds (Business Logic) ---
    print("\nCalculating deviation thresholds...")
    # Use all data (X) to get a stable representation of deviation
    predicted_prices = regressor.predict(X)
    deviation = ((y - predicted_prices) / predicted_prices) * 100

    lower_threshold = deviation.quantile(0.20)
    upper_threshold = deviation.quantile(0.80)

    print(f"  LOW Threshold (20th percentile): {lower_threshold:.2f}%")
    print(f"  HIGH Threshold (80th percentile): {upper_threshold:.2f}%")

    # --- 3. Feature Importance Analysis (from the Regressor) ---
    print("\n=== FEATURE IMPORTANCE ANALYSIS (from Regressor) ===")
    importances = regressor.feature_importances_
    df_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("Top features based on regression model:")
    print(df_importances.head())

    os.makedirs(MODELS_FOLDER, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=df_importances.head(15))
    plt.title('Top 15 Most Important Features (Regressor Model)')
    plt.tight_layout()
    plt.savefig(f'{MODELS_FOLDER}feature_importance.png')
    print(f"Feature importance chart saved to '{MODELS_FOLDER}feature_importance.png'")

    # --- 4. Calculate and Save Reliable Zones ---
    print("\nCalculating and saving reliable zones...")
    zone_counts = df_clean['colonia'].value_counts()
    reliable_zones = zone_counts[zone_counts >= MINIMUM_ZONE_COUNT].index.tolist()
    print(f"Found {len(reliable_zones)} reliable zones (colonias) with {MINIMUM_ZONE_COUNT}+ properties.")

    # --- 5. Save Artifacts ---
    print("\nSaving models and artifacts...")

    # Create dictionary for the thresholds
    thresholds = {
        'low_threshold_pct': lower_threshold,
        'high_threshold_pct': upper_threshold
    }

    joblib.dump(regressor, f'{MODELS_FOLDER}regressor_model.pkl')
    joblib.dump(imputer_numeric, f'{MODELS_FOLDER}imputer_numeric.pkl')
    joblib.dump(reliable_zones, f'{MODELS_FOLDER}reliable_zones.pkl')
    joblib.dump(thresholds, f'{MODELS_FOLDER}thresholds.pkl')  # <-- Save the thresholds
    joblib.dump(FINAL_NUMERIC_FEATURES + FINAL_CATEGORICAL_FEATURES, f'{MODELS_FOLDER}features_used.pkl')

    print("✅ Training pipeline completed successfully.")


if __name__ == '__main__':
    train_and_save_model()