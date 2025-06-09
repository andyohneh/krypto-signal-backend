import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pickle # Hinzugefügt, falls du Modelle hier speicherst oder lädst, auch wenn es in run_training_pipeline passiert

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
        # Wichtig: Wenn Spalten fehlen, muss das Training abgebrochen werden,
        # da das Modell sonst nicht korrekt trainiert werden kann.
        return None # Oder raise ValueError("...")

    # Trenne Features (X) und Ziel (y)
    X = data[FEATURES_LIST]
    y = data[target_column_name]

    # Handle NaN-Werte (falls noch welche vorhanden sind, sollten sie aber durch feature_engineer behoben sein)
    # Wichtig: Diese .fillna(0) Rufe sind ein Fallback. Idealerweise sollte feature_engineer.py
    # sicherstellen, dass keine NaN-Werte in FEATURES_LIST oder target_column_name ankommen.
    X = X.fillna(0)
    y = y.fillna(0)

    # Optional: Daten aufteilen in Trainings- und Testsets
    # Wenn du train_test_split verwendest, denk daran, dass du dann auch die Testdaten
    # für die Evaluierung verwenden solltest. Für ein einfaches Training kann es entfallen.
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modell initialisieren und trainieren
    model = LinearRegression() # Hier könntest du auch andere Modelle verwenden (z.B. RandomForestRegressor, GradientBoostingRegressor)
    model.fit(X, y) # Trainiere das Modell mit den Features (X) und dem Ziel (y)

    print(f"--- Regressions-Training für Ziel '{target_column_name}' abgeschlossen. ---")
    return model

# Hier könnten weitere Funktionen stehen, die du in train_model.py hast.
# Falls du keine weiteren Funktionen hast, ist das Ende der Datei.