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

from data_manager import download_historical_data
from feature_engineer import add_features_to_data
from train_model import train_and_evaluate_model, FEATURES_LIST

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Settings(db.Model):
    id=db.Column(db.Integer, primary_key=True); last_btc_signal=db.Column(db.String(10), default='N/A'); last_gold_signal=db.Column(db.String(10), default='N/A'); bitcoin_tp_percentage=db.Column(db.Float, default=2.5); bitcoin_sl_percentage=db.Column(db.Float, default=1.5); xauusd_tp_percentage=db.Column(db.Float, default=1.8); xauusd_sl_percentage=db.Column(db.Float, default=0.8); update_interval_minutes=db.Column(db.Integer, default=15)
class TrainedModel(db.Model):
    id=db.Column(db.Integer, primary_key=True); name=db.Column(db.String(80), unique=True, nullable=False); data=db.Column(LargeBinary, nullable=False); timestamp=db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())
class Device(db.Model):
    id=db.Column(db.Integer, primary_key=True); fcm_token=db.Column(db.String(255), unique=True, nullable=False); timestamp=db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())

def save_artifact_to_db(name, artifact):
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

def send_notification(title, body, tokens):
    try:
        message = messaging.MulticastMessage(notification=messaging.Notification(title=title, body=body), tokens=tokens)
        response = messaging.send_multicast(message)
        print(f'{response.success_count} Nachrichten erfolgreich gesendet.')
    except Exception as e: print(f"Fehler beim Senden der Benachrichtigung: {e}")

def trigger_web_service_redeploy():
    print("\n--- Phase 4: Automatisches Redeployment auslösen ---")
    hook_url = os.environ.get('WEB_SERVICE_DEPLOY_HOOK_URL')
    if not hook_url: print("Deploy Hook URL nicht gefunden."); return
    try:
        print("Rufe Deploy Hook auf..."); requests.get(hook_url, timeout=30)
        print("Redeployment erfolgreich ausgelöst!")
    except Exception as e: print(f"Fehler beim Aufruf des Deploy Hooks: {e}")

def run_full_pipeline():
    print("Starte die vollständige Trainings- und Benachrichtigungs-Pipeline...")

    if not firebase_admin._apps:
        try:
            print("Initialisiere Firebase Admin SDK...")
            cred = credentials.Certificate("serviceAccountKey.json")
            firebase_admin.initialize_app(cred)
            print("Firebase Admin SDK initialisiert.")
        except Exception as e:
            print(f"FEHLER bei Firebase-Initialisierung: {e}"); return

    with app.app_context():
        db.create_all()

        asset_map = {"BTC": {"ticker": "BTC-USD", "model_name": "btc_model", "scaler_name": "btc_scaler", "last_signal_key": "last_btc_signal"},"Gold": {"ticker": "GC=F", "model_name": "gold_model", "scaler_name": "gold_scaler", "last_signal_key": "last_gold_signal"}}

        print("\n--- Phase 1&2: Datenaufbereitung & Training ---")
        for asset_name, details in asset_map.items():
            print(f"\n--- Verarbeite {asset_name} ---")
            featured_data = add_features_to_data(download_historical_data(details["ticker"]))
            if featured_data is not None:
                model, scaler, _ = train_and_evaluate_model(featured_data)
                if model and scaler: save_artifact_to_db(details["model_name"], model); save_artifact_to_db(details["scaler_name"], scaler)

        print("\n--- Phase 3: Vorhersage & Benachrichtigung ---")
        settings = Settings.query.first()
        if not settings: settings = Settings(); db.session.add(settings); db.session.commit()

        device_tokens = [device.fcm_token for device in Device.query.all()]
        if not device_tokens: print("Keine registrierten Geräte gefunden. Überspringe Benachrichtigungen.")

        artifacts = TrainedModel.query.all()
        artifact_map = {artifact.name: pickle.loads(artifact.data) for artifact in artifacts}

        for asset_name, details in asset_map.items():
            model = artifact_map.get(details["model_name"]); scaler = artifact_map.get(details["scaler_name"])
            if not model or not scaler: print(f"Modell/Scaler für {asset_name} nicht geladen."); continue

            live_featured_data = add_features_to_data(download_historical_data(details["ticker"]))
            if live_featured_data is None or not all(col in live_featured_data.columns for col in FEATURES_LIST): continue

            features_for_scaling = live_featured_data[FEATURES_LIST]
            scaled_features = scaler.transform(features_for_scaling)
            latest_features = scaled_features[-1].reshape(1, -1)
            prediction = model.predict(latest_features)[0]
            new_signal = "Kauf" if prediction == 1 else "Verkauf"
            last_signal = getattr(settings, details["last_signal_key"])
            print(f"Analyse für {asset_name}: Letztes Signal='{last_signal}', Neues Signal='{new_signal}'")

            if new_signal != last_signal and device_tokens:
                print(f"-> Signal für {asset_name} hat sich geändert! Sende Benachrichtigung...")
                title = f"Neues Signal: {asset_name}"; body = f"Das Handelssignal für {asset_name} ist jetzt: {new_signal.upper()}"
                send_notification(title, body, device_tokens)
                setattr(settings, details["last_signal_key"], new_signal); db.session.commit()
            else: print(f"-> Signal für {asset_name} unverändert.")

    trigger_web_service_redeploy()
    print("\n\nPipeline erfolgreich durchgelaufen!")