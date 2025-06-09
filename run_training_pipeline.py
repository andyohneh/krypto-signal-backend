# run_training_pipeline.py (Version 2.0 - Trainiert Regressions-Modelle)

import os
import json
import pickle
from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import LargeBinary, func

# Importiere unsere sauberen Helfer-Funktionen
from data_manager import download_historical_data
from feature_engineer import add_features_to_data, create_regression_targets
from train_model import train_regression_model

# --- Setup (unverändert) ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Datenbank-Modelle (unverändert) ---
class TrainedModel(db.Model):
    id = db.Column(db.Integer, primary_key=True); name = db.Column(db.String(80), unique=True, nullable=False)
    data = db.Column(LargeBinary, nullable=False); timestamp = db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())

def save_artifact_to_db(name, artifact):
    """Speichert ein Artefakt (Modell/Scaler) in der DB."""
    print(f"Speichere '{name}' in der Datenbank...")
    pickled_artifact = pickle.dumps(artifact)
    existing_artifact = TrainedModel.query.filter_by(name=name).first()
    if existing_artifact:
        existing_artifact.data = pickled_artifact
        print(f"'{name}' in der DB aktualisiert.")
    else:
        new_artifact = TrainedModel(name=name, data=pickled_artifact)
        db.session.add(new_artifact)
        print(f"'{name}' neu in der DB erstellt.")
    db.session.commit()

def run_full_regression_pipeline():
    print("Starte die vollständige Regressions-Trainings-Pipeline...")

    with app.app_context():
        db.create_all()

        asset_map = {
            "BTC": {"ticker": "BTC-USD"},
            "Gold": {"ticker": "GC=F"}
        }

        for asset_name, details in asset_map.items():
            print(f"\n--- Verarbeite {asset_name} ---")

            # 1. Daten holen und Features erstellen
            raw_data = download_historical_data(details["ticker"], period="2y") # Längere Historie für bessere Modelle
            featured_data = add_features_to_data(raw_data)

            # 2. Regressions-Ziele erstellen (z.B. für die nächsten 7 Tage)
            final_data_for_training = create_regression_targets(featured_data, future_days=7)

            if final_data_for_training is not None:
                # 3. Zwei Modelle trainieren: eins für den Tiefst-, eins für den Höchstpreis

                # Modell für den Tiefstpreis (potenzieller Einstieg)
                low_model, low_scaler = train_regression_model(final_data_for_training, 'future_7d_low')
                if low_model and low_scaler:
                    save_artifact_to_db(f"{asset_name.lower()}_low_model", low_model)
                    save_artifact_to_db(f"{asset_name.lower()}_low_scaler", low_scaler)

                # Modell für den Höchstpreis (potenzieller Take Profit)
                high_model, high_scaler = train_regression_model(final_data_for_training, 'future_7d_high')
                if high_model and high_scaler:
                    save_artifact_to_db(f"{asset_name.lower()}_high_model", high_model)
                    save_artifact_to_db(f"{asset_name.lower()}_high_scaler", high_scaler)
            else:
                print(f"Konnte Trainingsdaten für {asset_name} nicht erstellen.")

    print("\n\nRegressions-Pipeline erfolgreich durchgelaufen!")

if __name__ == '__main__':
    run_full_regression_pipeline()