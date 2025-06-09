# train_model.py (Version 2.0 - Regression)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor # KORREKTUR: Dieser Import wurde hinzugefügt
from sklearn.preprocessing import StandardScaler
    
# Die Feature-Liste bleibt gleich
FEATURES_LIST = [
    'daily_return', 'SMA_10', 'SMA_50', 'sma_signal', 'RSI_14',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'ATRr_14'
]

def train_regression_model(data, target_column_name):
    """
    Trainiert ein Regressions-Modell, um eine bestimmte Ziel-Spalte
    (z.B. den zukünftigen Tiefstpunkt) vorherzusagen.
    """
    print(f"\n--- Starte Regressions-Training für Ziel: {target_column_name} ---")

    if not all(col in data.columns for col in FEATURES_LIST + [target_column_name]):
        print(f"FEHLER: Notwendige Spalten für Ziel '{target_column_name}' nicht gefunden.")
        return None, None

    X = data[FEATURES_LIST]
    y = data[target_column_name]

    # Daten aufteilen und skalieren (wie vorher)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # NEUES MODELL: RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    print("Trainiere Regressor-Modell...")
    model.fit(X_train_scaled, y_train)
    print("Training abgeschlossen.")
    
    return model, scaler