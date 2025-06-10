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
from feature_engineer import add_features_to_data, create_regression_targets
from train_model import train_regression_model, FEATURES_LIST

app = Flask(__name__)

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate("serviceAccountKey.json")
        print("Firebase-Credentials aus lokaler Datei geladen.")
    except FileNotFoundError:
        print("Lokale Schlüsseldatei nicht gefunden. Versuche Umgebungsvariable...")
        try:
            cred_str = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
            if cred_str: cred = credentials.Certificate(json.loads(cred_str))
            else: cred = None
        except Exception as e: cred = None; print(f"Fehler bei Firebase aus Umgebungsvariable: {e}")
    if cred:
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialisiert.")
    else:
        print("Firebase Admin SDK NICHT initialisiert.")

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Settings(db.Model):
    id=db.Column(db.Integer, primary_key=True); update_interval_minutes=db.Column(db.Integer, default=15); last_btc_signal=db.Column(db.String(100), default='N/A'); last_gold_signal=db.Column(db.String(100), default='N/A')
class TrainedModel(db.Model):
    id=db.Column(db.Integer, primary_key=True); name=db.Column(db.String(80), unique=True, nullable=False); data=db.Column(LargeBinary, nullable=False); timestamp=db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())
class Device(db.Model):
    id=db.Column(db.Integer, primary_key=True); fcm_token=db.Column(db.String(255), unique=True, nullable=False); timestamp=db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())

def save_artifact_to_db(name, artifact):
    with app.app_context():
        pickled_artifact = pickle.dumps(artifact)
        existing_artifact = TrainedModel.query.filter_by(name=name).first()
        if existing_artifact: existing_artifact.data = pickled_artifact
        else: db.session.add(TrainedModel(name=name, data=pickled_artifact))
        db.session.commit()
        print(f"'{name}' in DB gespeichert/aktualisiert.")

def send_notification(title, body, tokens):
    if not firebase_admin._apps: print("Firebase nicht initialisiert, kann keine Nachricht senden."); return
    try:
        message = messaging.MulticastMessage(notification=messaging.Notification(title=title, body=body), tokens=tokens)
        response = messaging.send_multicast(message)
        print(f'{response.success_count} Nachrichten erfolgreich gesendet.')
    except Exception as e: print(f"Fehler beim Senden der Benachrichtigung: {e}")

def trigger_web_service_redeploy():
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
        asset_map = {"BTC": {"ticker": "BTC-USD", "low_model_name": "btc_low_model", "high_model_name": "btc_high_model", "low_scaler_name": "btc_low_scaler", "high_scaler_name": "btc_high_scaler", "last_signal_key": "last_btc_signal"},
                     "Gold": {"ticker": "GC=F", "low_model_name": "gold_low_model", "high_model_name": "gold_high_model", "low_scaler_name": "gold_low_scaler", "high_scaler_name": "gold_high_scaler", "last_signal_key": "last_gold_signal"}}
        
        for asset_name, details in asset_map.items():
            print(f"\n--- Verarbeite {asset_name} ---")
            final_data = create_regression_targets(add_features_to_data(download_historical_data(details["ticker"], "2y")))
            if final_data is not None:
                low_model, low_scaler = train_regression_model(final_data, 'future_7d_low')
                if low_model and low_scaler: save_artifact_to_db(details["low_model_name"], low_model); save_artifact_to_db(details["low_scaler_name"], low_scaler)
                high_model, high_scaler = train_regression_model(final_data, 'future_7d_high')
                if high_model and high_scaler: save_artifact_to_db(details["high_model_name"], high_model); save_artifact_to_db(details["high_scaler_name"], high_scaler)
        
        settings = Settings.query.first()
        if not settings: settings = Settings(); db.session.add(settings); db.session.commit()
        device_tokens = [device.fcm_token for device in Device.query.all()]
        artifacts = TrainedModel.query.all()
        artifact_map = {artifact.name: pickle.loads(artifact.data) for artifact in artifacts}
        for asset_name, details in asset_map.items():
            low_model = artifact_map.get(details["low_model_name"]); high_model = artifact_map.get(details["high_model_name"])
            low_scaler = artifact_map.get(details["low_scaler_name"]); high_scaler = artifact_map.get(details["high_scaler_name"])
            if not all([low_model, high_model, low_scaler, high_scaler]): continue
            live_featured_data = add_features_to_data(download_historical_data(details["ticker"]))
            if live_featured_data is None or not all(col in live_featured_data.columns for col in FEATURES_LIST): continue
            latest_features_df = live_featured_data[FEATURES_LIST].tail(1)
            predicted_low = low_model.predict(low_scaler.transform(latest_features_df))[0]
            predicted_high = high_model.predict(high_scaler.transform(latest_features_df))[0]
            new_signal_text = f"Einstieg: {predicted_low:.2f}, TP: {predicted_high:.2f}"
            last_signal = getattr(settings, details["last_signal_key"])
            if new_signal_text != last_signal and device_tokens:
                title = f"Neues Preis-Ziel: {asset_name}"; body = f"Neues Ziel: Einstieg ca. {predicted_low:.2f}, TP ca. {predicted_high:.2f}"
                send_notification(title, body, device_tokens)
                setattr(settings, details["last_signal_key"], new_signal_text); db.session.commit()
    
    trigger_web_service_redeploy()
    print("\n\nPipeline erfolgreich durchgelaufen!")

if __name__ == '__main__':
    run_full_pipeline()