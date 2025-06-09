# train_model.py (Version mit Hyperparameter-Tuning)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV # NEU: Import für die Parametersuche

def train_and_evaluate_model(data):
    """
    Sucht jetzt nach den besten Hyperparametern für das RandomForest-Modell,
    trainiert es und gibt die besten gefundenen Komponenten zurück.
    """
    if data is None or data.empty:
        print("FEHLER: Leere Daten übergeben.")
        return None, None, 0.0

    print("Starte Modelltraining mit Hyperparameter-Tuning...")

    # Datenvorbereitung (unverändert)
    data['target'] = np.where(data['Adj Close'].shift(-1) > data['Adj Close'], 1, 0)
    data.dropna(inplace=True)
    features = [
        'daily_return', 'SMA_10', 'SMA_50', 'sma_signal', 'RSI_14',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'
    ]
    X = data[features]
    y = data['target']

    # Daten aufteilen (unverändert)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    if len(X_train) < 10: # Brauchen eine Mindestanzahl an Daten
        print("FEHLER: Nicht genügend Trainingsdaten.")
        return None, None, 0.0

    # Daten skalieren (unverändert)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Daten erfolgreich skaliert.")

    # --- NEU: Hyperparameter-Tuning mit GridSearchCV ---

    # 1. Definiere das "Gitter" der Parameter, die wir testen wollen
    param_grid = {
        'n_estimators': [100, 150],       # Anzahl der Bäume im Wald
        'max_depth': [10, 20, None],      # Maximale Tiefe eines Baumes (None = unbegrenzt)
        'min_samples_split': [2, 5],      # Mindestanzahl an Samples für einen Split
        'min_samples_leaf': [1, 2]        # Mindestanzahl an Samples in einem Blatt
    }

    # 2. Erstelle das GridSearchCV-Objekt
    # Es wird alle Kombinationen der Parameter testen (2*3*2*2 = 24 Modelle)
    # cv=3 bedeutet, es macht eine 3-fache Kreuzvalidierung, um die Ergebnisse stabiler zu machen.
    # n_jobs=-1 nutzt alle verfügbaren CPU-Kerne, um das Training zu beschleunigen.
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

    # 3. Führe die Suche auf den Trainingsdaten aus (das kann jetzt länger dauern!)
    print("Starte die Suche nach den besten Parametern... das kann einige Minuten dauern.")
    grid_search.fit(X_train_scaled, y_train)

    # 4. Hole dir das beste gefundene Modell
    print(f"Beste gefundene Parameter: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    # 5. Bewerte das beste Modell
    test_accuracy = best_model.score(X_test_scaled, y_test)
    print(f"Genauigkeit des optimierten Modells auf Test-Daten: {test_accuracy:.2%}")

    # 6. Gib das BESTE Modell und den Scaler zurück
    return best_model, scaler, test_accuracy