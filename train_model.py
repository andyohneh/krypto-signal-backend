import pandas as pd
from sklearn.linear_model import LinearRegression # Oder dein bevorzugtes Modell
from sklearn.model_selection import train_test_split
import numpy as np

# ACHTUNG: FEATURES_LIST muss hier definiert sein und die korrekten Spaltennamen enthalten,
# die von feature_engineer.py erstellt werden.
FEATURES_LIST = [
    'Close',
    'daily_return',
    'SMA_10',
    'SMA_50',
    'sma_signal',
    'RSI_14',
    'MACD_12_26_9',
    'MACDh_12_26_9',
    'MACDs_12_26_9',
    'ATRr_14'
]

def train_regression_model(data, target_column_name):
    """
    Trainiert ein Regressionsmodell.
    Args:
        data (pd.DataFrame): DataFrame mit Features und der Zielspalte.
        target_column_name (str): Der Name der Zielspalte.
    Returns:
        Ein trainiertes Regressionsmodell.
    """
    print(f"--- Starte Regressions-Training für Ziel: {target_column_name} ---")

    # Überprüfe, ob alle notwendigen Spalten vorhanden sind
    required_columns = FEATURES_LIST + [target_column_name]
    if not all(col in data.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in data.columns]
        print(f"FEHLER: Notwendige Spalten für Ziel '{target_column_name}' nicht gefunden: {missing_cols}")
        # Du könntest hier einen Fehler auslösen oder None zurückgeben,
        # je nachdem, wie du Fehler behandeln möchtest.
        return None # Oder raise ValueError("...")

    # Trenne Features (X) und Ziel (y)
    X = data[FEATURES_LIST]
    y = data[target_column_name]

    # Handle NaN-Werte (falls noch welche vorhanden sind, sollten sie aber durch feature_engineer behoben sein)
    X = X.fillna(0) # Oder eine andere Strategie
    y = y.fillna(0) # Oder eine andere Strategie

    # Optional: Daten aufteilen in Trainings- und Testsets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modell initialisieren und trainieren
    model = LinearRegression() # Oder dein bevorzugtes Modell
    model.fit(X, y)

    print(f"--- Regressions-Training für Ziel '{target_column_name}' abgeschlossen. ---")
    return model

# Andere Funktionen, falls vorhanden, bleiben unverändert.