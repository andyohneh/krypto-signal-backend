# run_training_pipeline.py (Finale Version - Lädt Key aus Datei)

import os
import json
import pickle
from dotenv import load_dotenv

# Lade die .env-Datei (jetzt nur noch für DATABASE_URL)
load_dotenv()

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import LargeBinary, func
import firebase_admin
from firebase_admin import credentials, messaging

# Importiere unsere Helfer-Funktionen
from data_manager import download_historical_data
from feature_engineer import add_features_to_data
from train_model import train_and_evaluate_model

# --- Setup für Datenbank und Firebase ---
app = Flask(__name__)

# --- NEUE, ROBUSTERE FIREBASE-INITIALISIERUNG ---
# Lade den Schlüssel direkt aus der .json-Datei
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    print("Firebase Admin SDK erfolgreich aus Datei initialisiert.")
except Exception as e:
    print(f"FEHLER bei Firebase-Initialisierung: {e}. Stelle sicher, dass die 'serviceAccountKey.json' im Projektordner liegt.")

# Lade die Datenbank-URL aus den Umgebungsvariablen (lokal aus .env, auf Render aus den Settings)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ... (Der Rest des Skripts von hier an ist exakt gleich wie in der letzten Version) ...
# --- Datenbank-Modelle ---
class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    last_btc_signal = db.Column(db.String(10), default='N/A'); last_gold_signal = db.Column(db.String(10), default='N/A')
    bitcoin_tp_percentage = db.Column(db.Float, default=2.5); bitcoin_sl_percentage = db.Column(db.Float, default=1.5)
    xauusd_tp_percentage = db.Column(db.Float, default=1.8); xauusd_sl_percentage = db.Column(db.Float, default=0.8)
    update_interval_minutes = db.Column(db.Integer, default=15)
class TrainedModel(db.Model):
    id = db.Column(db.Integer, primary_key=True); name = db.Column(db.String(80), unique=True, nullable=False)
    data = db.Column(LargeBinary, nullable=False); timestamp = db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())
class Device(db.Model):
    id = db.Column(db.Integer, primary_key=True); fcm_token = db.Column(db.String(255), unique=True, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())
def save_artifact_to_db(name, artifact):
    print(f"Speichere '{name}' in der Datenbank...")
    pickled_artifact = pickle.dumps(artifact)
    with app.app_context():
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
    except Exception as e:
        print(f"Fehler beim Senden der Benachrichtigung: {e}")
def run_full_pipeline():
    print("Starte die vollständige Trainings- und Benachrichtigungs-Pipeline...")
    with app.app_context():
        db.create_all()
        asset_map = {
            "BTC": {"ticker": "BTC-USD", "model_name": "btc_model", "scaler_name": "btc_scaler", "last_signal_key": "last_btc_signal"},
            "Gold": {"ticker": "GC=F", "model_name": "gold_model", "scaler_name": "gold_scaler", "last_signal_key": "last_gold_signal"}
        }
        for asset_name, details in asset_map.items():
            print(f"\n--- Verarbeite {asset_name} ---")
            raw_data = download_historical_data(details["ticker"])
            featured_data = add_features_to_data(raw_data)
            if featured_data is not None:
                model, scaler, accuracy = train_and_evaluate_model(featured_data)
                if model and scaler:
                    save_artifact_to_db(name=details["model_name"], artifact=model)
                    save_artifact_to_db(name=details["scaler_name"], artifact=scaler)
        
        print("\n--- Prüfe auf Signaländerungen für Benachrichtigungen ---")
        settings = Settings.query.first()
        if not settings:
             settings = Settings(); db.session.add(settings); db.session.commit()
        device_tokens = [device.fcm_token for device in Device.query.all()]
        if not device_tokens:
            print("Keine registrierten Geräte gefunden. Überspringe Benachrichtigungen.")
        
        artifacts = TrainedModel.query.all()
        artifact_map = {artifact.name: pickle.loads(artifact.data) for artifact in artifacts}

        for asset_name, details in asset_map.items():
            model = artifact_map.get(details["model_name"])
            scaler = artifact_map.get(details["scaler_name"])
            if not model or not scaler: continue
            
            live_featured_data = add_features_to_data(download_historical_data(details["ticker"]))
            if live_featured_data is None: continue

            features_for_scaling = live_featured_data[features]
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
                setattr(settings, details["last_signal_key"], new_signal)
                db.session.commit()
            else:
                print(f"-> Signal für {asset_name} unverändert.")
        
    print("\n\nPipeline erfolgreich durchgelaufen!")

if __name__ == '__main__':
    features = ['daily_return', 'SMA_10', 'SMA_50', 'sma_signal', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
    run_full_pipeline()