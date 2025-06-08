# train_model.py (Finale Version - mit DB-Bugfix)

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import LargeBinary

# --- Datenbank-Setup ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class TrainedModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    data = db.Column(LargeBinary, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())

def save_artifact_to_db(name, artifact):
    """Serialisiert ein Objekt und speichert es in der DB (Update oder Insert)."""
    print(f"Speichere '{name}' in der Datenbank...")
    pickled_artifact = pickle.dumps(artifact)

    with app.app_context():
        # KORREKTUR: Wir suchen immer nach dem 'name', nicht der 'id'.
        existing_artifact = TrainedModel.query.filter_by(name=name).first()

        if existing_artifact:
            existing_artifact.data = pickled_artifact
            print(f"'{name}' in der DB aktualisiert.")
        else:
            new_artifact = TrainedModel(name=name, data=pickled_artifact)
            db.session.add(new_artifact)
            print(f"'{name}' neu in der DB erstellt.")

        db.session.commit()

def train_model_for_asset(input_filename, model_name, scaler_name):
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

    test_accuracy = model.score(X_test_scaled, y_test)
    print(f"Modell-Genauigkeit auf Test-Daten: {test_accuracy:.2%}")

    save_artifact_to_db(name=model_name, artifact=model)
    save_artifact_to_db(name=scaler_name, artifact=scaler)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    train_model_for_asset("btc_data_with_features.csv", "btc_model", "btc_scaler")
    train_model_for_asset("gold_data_with_features.csv", "gold_model", "gold_scaler")

    print("\nAlle Modelle und Scaler wurden in die Datenbank geschrieben.")