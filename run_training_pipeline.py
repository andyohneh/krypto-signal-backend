# run_training_pipeline.py (Finaler Debug-Modus mit Spurensuchern)

import os, json, pickle, requests
from dotenv import load_dotenv
load_dotenv()
print("[OK] .env geladen")

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import LargeBinary, func
import firebase_admin
from firebase_admin import credentials, messaging
print("[OK] Alle Haupt-Bibliotheken importiert")

from data_manager import download_historical_data
from feature_engineer import add_features_to_data, create_regression_targets
from train_model import train_regression_model, FEATURES_LIST
print("[OK] Alle Helfer-Skripte importiert")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
db = SQLAlchemy(app)
print("[OK] Flask-App und DB-Verbindung konfiguriert")

class Settings(db.Model):
    id=db.Column(db.Integer, primary_key=True); update_interval_minutes=db.Column(db.Integer, default=15); last_btc_signal=db.Column(db.String(100), default='N/A'); last_gold_signal=db.Column(db.String(100), default='N/A')
class TrainedModel(db.Model):
    id=db.Column(db.Integer, primary_key=True); name=db.Column(db.String(80), unique=True, nullable=False); data=db.Column(LargeBinary, nullable=False)
class Device(db.Model):
    id=db.Column(db.Integer, primary_key=True); fcm_token=db.Column(db.String(255), unique=True, nullable=False)
print("[OK] DB-Modelle definiert")

def run_full_pipeline():
    print("\n>>> Starte die vollständige Pipeline-Funktion...")

    with app.app_context():
        print("    Betrete App-Kontext...")

        print("    [Schritt 1] Führe db.create_all() aus...")
        db.create_all()
        print("    [OK] DB-Tabellen geprüft/erstellt.")

        asset_map = {"BTC": {"ticker": "BTC-USD"}, "Gold": {"ticker": "GC=F"}}

        for asset_name, details in asset_map.items():
            print(f"\n    --- Verarbeite {asset_name} ---")

            print(f"    [Schritt 2] Lade Daten für {asset_name}...")
            raw_data = download_historical_data(details["ticker"], period="2y")
            print(f"    [OK] Daten für {asset_name} geladen.")

            print(f"    [Schritt 3] Erstelle Features für {asset_name}...")
            featured_data = add_features_to_data(raw_data)
            print(f"    [OK] Features für {asset_name} erstellt.")

            print(f"    [Schritt 4] Erstelle Regressions-Ziele für {asset_name}...")
            final_data_for_training = create_regression_targets(featured_data, future_days=7)
            print(f"    [OK] Regressions-Ziele für {asset_name} erstellt.")

            if final_data_for_training is not None:
                print(f"    [Schritt 5] Trainiere Modelle für {asset_name}...")

                low_model, low_scaler = train_regression_model(final_data_for_training, 'future_7d_low')
                high_model, high_scaler = train_regression_model(final_data_for_training, 'future_7d_high')

                print(f"    [OK] Modelle für {asset_name} trainiert.")
            else:
                print(f"    WARNUNG: Keine Trainingsdaten für {asset_name} verfügbar.")

    print("\n\n>>> Pipeline erfolgreich durchgelaufen! (Zumindest der Trainings-Teil)")

if __name__ == '__main__':
    run_full_pipeline()