# train_model.py (Version 6 - Speichert Modelle in die DB)

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# NEU: Importiere die notwendigen DB-Komponenten
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import LargeBinary

# --- Datenbank-Setup für ein Standalone-Skript ---
# Wir erstellen eine temporäre App, nur um den DB-Kontext zu haben
app = Flask(__name__)
# Lese die DB-URL aus den Umgebungsvariablen (die wir im Cron Job gesetzt haben)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Wir definieren das Modell hier erneut, damit das Skript die Tabellenstruktur kennt
class TrainedModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    data = db.Column(LargeBinary, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())

def save_artifact_to_db(name, artifact):
    """Serialisiert ein Python-Objekt (Modell/Scaler) und speichert es in der DB."""
    print(f"Speichere '{name}' in der Datenbank...")

    # "pickle" wandelt das Objekt in einen Binär-String um
    pickled_artifact = pickle.dumps(artifact)

    # Prüfe, ob bereits ein Eintrag mit diesem Namen existiert
    existing_artifact = TrainedModel.query.filter_by(name=name).first()

    if existing_artifact:
        # Wenn ja, aktualisiere nur die Daten
        existing_artifact.data = pickled_artifact
        print(f"'{name}' in der DB aktualisiert.")
    else:
        # Wenn nicht, erstelle einen neuen Eintrag
        new_artifact = TrainedModel(name=name, data=pickled_artifact)
        db.session.add(new_artifact)
        print(f"'{name}' neu in der DB erstellt.")

    db.session.commit()

def train_model_for_asset(input_filename, model_name, scaler_name):
    # Der Anfang der Funktion bleibt gleich
    print(f"\n--- Starte DB-Modelltraining für {model_name} ---")
    data = pd.read_csv(input_filename, index_col=0, parse_dates=True)
    data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    data.dropna(inplace=True)
    features = ['daily_return', 'SMA_10', 'SMA_50', 'sma_signal', 'RSI_14']
    X = data[features]
    y = data['target']
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # Die Auswertung bleibt zur Kontrolle
    test_accuracy = model.score(X_test_scaled, y_test)
    print(f"Modell-Genauigkeit auf Test-Daten: {test_accuracy:.2%}")

    # NEU: Speichern in die Datenbank statt in eine Datei
    with app.app_context():
        save_artifact_to_db(name=model_name, artifact=model)
        save_artifact_to_db(name=scaler_name, artifact=scaler)

if __name__ == '__main__':
    # Wir müssen den Code in einen App-Kontext packen, damit die DB-Verbindung funktioniert
    with app.app_context():
        db.create_all() # Stellt sicher, dass die Tabelle existiert

    # Prozess für Bitcoin
    train_model_for_asset(
        "btc_data_with_features.csv",
        "btc_model",      # Nur noch der Name, kein Dateipfad
        "btc_scaler"      # Nur noch der Name, kein Dateipfad
    )

    # Prozess für Gold
    train_model_for_asset(
        "gold_data_with_features.csv",
        "gold_model",     # Nur noch der Name, kein Dateipfad
        "gold_scaler"     # Nur noch der Name, kein Dateipfad
    )

    print("\nAlle Modelle und Scaler wurden in die Datenbank geschrieben.")