import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

# NEU: Die Feature-Liste wurde um die Lag-Features erweitert
FEATURES_LIST = [
    'daily_return', 'SMA_10', 'SMA_50', 'sma_signal', 'RSI_14',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'ATRr_14',
    'daily_return_lag_1', 'RSI_14_lag_1',
    'daily_return_lag_2', 'RSI_14_lag_2',
    'daily_return_lag_3', 'RSI_14_lag_3'
]

def train_regression_model(data, target_column_name):
    print(f"\n--- Starte Modell-Wettbewerb für Ziel: {target_column_name} ---")

    if not all(col in data.columns for col in FEATURES_LIST + [target_column_name]):
        print("FEHLER: Notwendige Spalten nicht gefunden.")
        return None, None

    X = data[FEATURES_LIST]
    y = data[target_column_name]

    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modell 1: Random Forest
    print("Trainiere RandomForest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_score = rf_model.score(X_test_scaled, y_test)
    print(f"RandomForest Test-Score (R²): {rf_score:.4f}")

    # Modell 2: LightGBM
    print("Trainiere LightGBM...")
    lgbm_model = LGBMRegressor(random_state=42)
    lgbm_model.fit(X_train_scaled, y_train)
    lgbm_score = lgbm_model.score(X_test_scaled, y_test)
    print(f"LightGBM Test-Score (R²): {lgbm_score:.4f}")

    if lgbm_score > rf_score:
        print("----> SIEGER: LightGBM!")
        return lgbm_model, scaler
    else:
        print("----> SIEGER: RandomForest!")
        return rf_model, scaler