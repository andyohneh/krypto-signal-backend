import os
import json
import pickle
import requests
from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import LargeBinary, func
import firebase_admin
from firebase_admin import credentials, messaging
import pandas as pd
import numpy as np

# Importiere unsere sauberen Helfer-Funktionen und die Feature-Liste
from data_manager import download_historical_data
from feature_engineer import add_features_to_data, create_regression_targets
from train_model import train_regression_model, FEATURES_LIST

# --- Setup ---
app = Flask(__name__)

# Robuste Firebase-Initialisierung
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate("serviceAccountKey.json")
        print("Firebase-Credentials aus lokaler Datei geladen.")
    except FileNotFoundError:
        print("Lokale Schlüsseldatei nicht gefunden. Versuche Umgebungsvariable...")
        try:
            cred_str = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
            if cred_str:
                cred = credentials.Certificate(json.loads(cred_str))
                print("Firebase-Credentials aus Umgebungsvariable geladen.")
            else:
                cred = None
                print("WARNUNG: Keine Firebase-Credentials in Umgebungsvariable gefunden.")
        except Exception as e:
            cred = None
            print(f"Fehler beim Parsen der Firebase-Credentials: {e}")
    if cred:
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialisiert.")
    else:
        print("Firebase Admin SDK NICHT initialisiert.")

# Datenbank-Konfiguration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Datenbank-Modelle ---
class Settings(db.Model):
    id=db.Column(db.Integer, primary_key=True); last_btc_signal=db.Column(db.String(20), default='N/A'); last_gold_signal=db.Column(db.String(20), default='N/A')
class TrainedModel(db.Model):
    id=db.Column(db.Integer, primary_key=True); name=db.Column(db.String(80), unique=True, nullable=False); data=db.Column(LargeBinary, nullable=False)
class Device(db.Model):
    id=db.Column(db.Integer, primary_key=True); fcm_token=db.Column(db.String(255), unique=True, nullable=False)

# --- Helfer-Funktionen ---
def save_artifact_to_db(name, artifact):
    with app.app_context():
        print(f"Speichere '{name}' in der DB...")
        pickled_artifact = pickle.dumps(artifact)
        existing_artifact = TrainedModel.query.filter_by(name=name).first()
        if existing_artifact:
            existing_artifact.data = pickled_artifact
        else:
            new_artifact = TrainedModel(name=name, data=pickled_artifact)
            db.session.add(new_artifact)
        db.session.commit()
        print(f"'{name}' in DB gespeichert.")

def send_notification(title, body, tokens):
    if not firebase_admin._apps:
        print("Firebase nicht initialisiert, Nachricht kann nicht gesendet werden."); return
    try:
        message = messaging.MulticastMessage(notification=messaging.Notification(title=title, body=body), tokens=tokens)
        response = messaging.send_multicast(message)
        print(f'{response.success_count} Nachrichten erfolgreich gesendet.')
    except Exception as e:
        print(f"Fehler beim Senden der Benachrichtigung: {e}")

def trigger_web_service_redeploy():
    print("\n--- Phase 4: Automatisches Redeployment ---")
    hook_url = os.environ.get('WEB_SERVICE_DEPLOY_HOOK_URL')
    if not hook_url: print("Deploy Hook URL nicht gefunden."); return
    try:
        print("Rufe Deploy Hook auf..."); requests.get(hook_url, timeout=30)
        print("Redeployment erfolgreich ausgelöst!")
    except Exception as e: print(f"Fehler beim Aufruf des Deploy Hooks: {e}")

def run_full_pipeline():
    print("Starte die vollständige Regressions-Trainings-Pipeline...")
    with app.app_context():
        db.create_all()
        
        asset_map = {
            "BTC": {"ticker": "BTC-USD", "low_model_name": "btc_low_model", "high_model_name": "btc_high_model", "low_scaler_name": "btc_low_scaler", "high_scaler_name": "btc_high_scaler", "last_signal_key": "last_btc_signal"},
            "Gold": {"ticker": "GC=F", "low_model_name": "gold_low_model", "high_model_name": "gold_high_model", "low_scaler_name": "gold_low_scaler", "high_scaler_name": "gold_high_scaler", "last_signal_key": "last_gold_signal"}
        }

        print("\n--- Phase 1&2: Datenaufbereitung & Training ---")
        for asset_name, details in asset_map.items():
            print(f"\n--- Verarbeite {asset_name} ---")
            raw_data = download_historical_data(details["ticker"], period="2y")
            featured_data = add_features_to_data(raw_data)
            final_data = create_regression_targets(featured_data, future_days=7)
            if final_data is not None:
                low_model, low_scaler = train_regression_model(final_data, 'future_7d_low')
                if low_model and low_scaler:
                    save_artifact_to_db(details["low_model_name"], low_model)
                    save_artifact_to_db(details["low_scaler_name"], low_scaler)
                high_model, high_scaler = train_regression_model(final_data, 'future_7d_high')
                if high_model and high_scaler:
                    save_artifact_to_db(details["high_model_name"], high_model)
                    save_artifact_to_db(details["high_scaler_name"], high_scaler)

        print("\n--- Phase 3: Vorhersage & Benachrichtigung ---")
        settings = Settings.query.first()
        if not settings: settings = Settings(); db.session.add(settings); db.session.commit()
        
        device_tokens = [device.fcm_token for device in Device.query.all()]
        if not device_tokens: print("Keine registrierten Geräte gefunden.")

        artifacts = TrainedModel.query.all()
        artifact_map = {artifact.name: pickle.loads(artifact.data) for artifact in artifacts}

        for asset_name, details in asset_map.items():
            low_model = artifact_map.get(details["low_model_name"]); high_model = artifact_map.get(details["high_model_name"])
            low_scaler = artifact_map.get(details["low_scaler_name"]); high_scaler = artifact_map.get(details["high_scaler_name"])
            if not all([low_model, high_model, low_scaler, high_scaler]):
                print(f"Modelle/Scaler für {asset_name} nicht vollständig geladen."); continue
            
            live_featured_data = add_features_to_data(download_historical_data(details["ticker"]))
            if live_featured_data is None or not all(col in live_featured_data.columns for col in FEATURES_LIST): continue
            
            latest_features_df = live_featured_data[FEATURES_LIST].tail(1)
            predicted_low = low_model.predict(low_scaler.transform(latest_features_df))[0]
            predicted_high = high_model.predict(high_scaler.transform(latest_features_df))[0]
            
            new_signal_text = f"Einstieg: {predicted_low:.2f}, TP: {predicted_high:.2f}"
            last_signal = getattr(settings, details["last_signal_key"])
            print(f"Analyse für {asset_name}: Letztes Signal='{last_signal}', Neues Signal='{new_signal_text}'")

            if new_signal_text != last_signal and device_tokens:
                print(f"-> Signal für {asset_name} hat sich geändert! Sende Benachrichtigung...")
                title = f"Neues Preis-Ziel: {asset_name}"; body = f"Neues Ziel: Einstieg ca. {predicted_low:.2f}, Take Profit ca. {predicted_high:.2f}"
                send_notification(title, body, device_tokens)
                setattr(settings, details["last_signal_key"], new_signal_text); db.session.commit()
            else:
                print(f"-> Signal für {asset_name} unverändert.")
        
    trigger_web_service_redeploy()
    print("\n\nPipeline erfolgreich durchgelaufen!")

if __name__ == '__main__':
    run_full_pipeline()