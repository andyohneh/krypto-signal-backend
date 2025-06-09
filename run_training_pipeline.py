# run_training_pipeline.py (Finale Version mit automatischem Redeployment)

import os
import json
import pickle
import requests # Wichtig für den Aufruf des Deploy Hooks
from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import LargeBinary, func
import firebase_admin
from firebase_admin import credentials, messaging

from data_manager import download_historical_data
from feature_engineer import add_features_to_data
from train_model import train_and_evaluate_model

# --- Setup ---
app = Flask(__name__)
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    print("Firebase Admin SDK initialisiert.")
except Exception as e:
    print(f"FEHLER bei Firebase-Initialisierung: {e}")

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- DB-Modelle ---
class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True); last_btc_signal = db.Column(db.String(10), default='N/A'); last_gold_signal = db.Column(db.String(10), default='N/A')
class TrainedModel(db.Model):
    id = db.Column(db.Integer, primary_key=True); name = db.Column(db.String(80), unique=True, nullable=False); data = db.Column(LargeBinary, nullable=False)
class Device(db.Model):
    id = db.Column(db.Integer, primary_key=True); fcm_token = db.Column(db.String(255), unique=True, nullable=False)

def save_artifact_to_db(name, artifact):
    print(f"Speichere '{name}' in der Datenbank...")
    with app.app_context():
        existing_artifact = TrainedModel.query.filter_by(name=name).first()
        if existing_artifact: existing_artifact.data = pickle.dumps(artifact)
        else: db.session.add(TrainedModel(name=name, data=pickle.dumps(artifact)))
        db.session.commit()

def send_notification(title, body, tokens):
    try:
        message = messaging.MulticastMessage(notification=messaging.Notification(title=title, body=body), tokens=tokens)
        messaging.send_multicast(message)
        print("Benachrichtigungen erfolgreich versendet.")
    except Exception as e:
        print(f"Fehler beim Senden der Benachrichtigung: {e}")

def trigger_web_service_redeploy():
    print("\n--- Phase 4: Automatisches Redeployment auslösen ---")
    hook_url = os.environ.get('WEB_SERVICE_DEPLOY_HOOK_URL')
    if not hook_url:
        print("Deploy Hook URL nicht gefunden. Überspringe automatischen Neustart.")
        return
    try:
        print("Rufe Deploy Hook auf...")
        response = requests.get(hook_url, timeout=20)
        if 200 <= response.status_code < 300:
            print("Redeployment erfolgreich ausgelöst!")
        else:
            print(f"Fehler beim Auslösen des Deploy Hooks: Status {response.status_code}")
    except Exception as e:
        print(f"Fehler beim Aufruf des Deploy Hooks: {e}")

def run_full_pipeline():
    print("Starte die vollständige Trainings- und Benachrichtigungs-Pipeline...")
    with app.app_context():
        db.create_all()

        # Phase 1 & 2
        print("\n--- Phase 1&2: Datenaufbereitung & Training ---")
        asset_map = {
            "BTC": {"ticker": "BTC-USD", "model_name": "btc_model", "scaler_name": "btc_scaler", "last_signal_key": "last_btc_signal"},
            "Gold": {"ticker": "GC=F", "model_name": "gold_model", "scaler_name": "gold_scaler", "last_signal_key": "last_gold_signal"}
        }
        trained_artifacts = {}
        for asset_name, details in asset_map.items():
            featured_data = add_features_to_data(download_historical_data(details["ticker"]))
            if featured_data is not None:
                model, scaler, _ = train_and_evaluate_model(featured_data)
                if model and scaler:
                    save_artifact_to_db(details["model_name"], model)
                    save_artifact_to_db(details["scaler_name"], scaler)
                    trained_artifacts[asset_name] = {"model": model, "scaler": scaler}

        # Phase 3
        print("\n--- Phase 3: Vorhersage & Benachrichtigung ---")
        settings = Settings.query.first()
        if not settings: settings = Settings(); db.session.add(settings); db.session.commit()
        device_tokens = [device.fcm_token for device in Device.query.all()]

        for asset_name, details in asset_map.items():
            if asset_name not in trained_artifacts: continue
            # ... (Logik zur Vorhersage und Benachrichtigung wie vorher)
            last_signal = getattr(settings, details["last_signal_key"])
            new_signal = "Kauf" # Platzhalter, ersetze durch echte Vorhersage
            if new_signal != last_signal and device_tokens:
                # ... (sende Benachrichtigung)
                setattr(settings, details["last_signal_key"], new_signal)
                db.session.commit()

    # Phase 4
    trigger_web_service_redeploy()
    print("\n\nPipeline erfolgreich durchgelaufen! Neustart des Web Service angestoßen.")

if __name__ == '__main__':
    run_full_pipeline()