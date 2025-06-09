import os
import json
import pickle
import requests
from dotenv import load_dotenv

# Lade die .env-Datei für die lokale Entwicklung
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
cred = None # Initialisiere cred hier als None, um NameError zu vermeiden
if not firebase_admin._apps:
    try:
        # 1. Versuch: Lade aus lokaler Datei (für deinen PC)
        cred = credentials.Certificate("serviceAccountKey.json")
        print("Firebase-Credentials aus lokaler Datei geladen.")
    except FileNotFoundError:
        # 2. Versuch: Lade aus Umgebungsvariable (für Render)
        print("Lokale Schlüsseldatei nicht gefunden. Versuche Umgebungsvariable...")
        try:
            cred_str = os.environ.get('FIREBASE_SERVICE_ACCOUNT_2', 'FIREBASE_SERVICE_ACCOUNT_JSON') # Test, ob 2ter Name klappt 
            if cred_str:
                cred = credentials.Certificate(json.loads(cred_str))
                print("Firebase-Credentials aus Umgebungsvariable geladen.")
            else:
                print("WARNUNG: FIREBASE_SERVICE_ACCOUNT_JSON Variable nicht gefunden.")
        except Exception as e:
            print(f"Fehler beim Parsen/Laden der Firebase-Credentials aus Umgebungsvariable: {e}")
    except Exception as e: # Catch all für andere Fehler beim Laden der Datei
        print(f"Ein anderer Fehler bei Firebase-Credential-Suche: {e}")

    if cred:
        try:
            firebase_admin.initialize_app(cred)
            print("Firebase Admin SDK initialisiert.")
        except Exception as e:
            print(f"FEHLER bei Firebase-Initialisierung mit geladenen Credentials: {e}")
            cred = None # Setze cred auf None, wenn Initialisierung fehlschlägt
    else:
        print("Firebase Admin SDK NICHT initialisiert. Benachrichtigungen werden fehlschlagen.")


# Datenbank-Konfiguration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Datenbank-Modelle ---
class Settings(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    # KORREKTUR: Länge auf 100 erhöht, um längere Signaltexte zu speichern
    last_btc_signal=db.Column(db.String(100), default='N/A')
    last_gold_signal=db.Column(db.String(100), default='N/A')
    # Die anderen Spalten, die in app.py definiert sind, müssen hier auch existieren,
    # damit das Modell konsistent ist, auch wenn sie in diesem Skript nicht direkt genutzt werden.
    bitcoin_tp_percentage = db.Column(db.Float, default=2.5)
    bitcoin_sl_percentage = db.Column(db.Float, default=1.5)
    xauusd_tp_percentage = db.Column(db.Float, default=1.8)
    xauusd_sl_percentage = db.Column(db.Float, default=0.8)
    update_interval_minutes = db.Column(db.Integer, default=15)


class TrainedModel(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    name=db.Column(db.String(80), unique=True, nullable=False)
    data=db.Column(LargeBinary, nullable=False)
    timestamp=db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())

class Device(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    fcm_token=db.Column(db.String(255), unique=True, nullable=False)
    timestamp=db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())

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
    # KORREKTUR: Prüfe Firebase Admin SDK Initialisierung hier vor dem Senden
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
        print("Rufe Deploy Hook auf...")
        response = requests.get(hook_url, timeout=30)
        if 200 <= response.status_code < 300: print("Redeployment erfolgreich ausgelöst!")
        else: print(f"Fehler beim Auslösen des Deploy Hooks: Status {response.status_code}")
    except Exception as e: print(f"Fehler beim Aufruf des Deploy Hooks: {e}")

# --- Haupt-Logik ---
def run_full_pipeline():
    print("Starte die vollständige Regressions-Trainings-Pipeline...")
    with app.app_context():
        db.create_all() # Stellt sicher, dass alle Tabellen (inkl. Settings mit neuer Spaltenlänge) existieren
        
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
                # Trainiere und speichere Low-Modell und Scaler
                low_model, low_scaler = train_regression_model(final_data, 'future_7d_low')
                if low_model and low_scaler:
                    save_artifact_to_db(details["low_model_name"], low_model)
                    save_artifact_to_db(details["low_scaler_name"], low_scaler)
                
                # Trainiere und speichere High-Modell und Scaler
                high_model, high_scaler = train_regression_model(final_data, 'future_7d_high')
                if high_model and high_scaler:
                    save_artifact_to_db(details["high_model_name"], high_model)
                    save_artifact_to_db(details["high_scaler_name"], high_scaler)
            else: print(f"Konnte Trainingsdaten für {asset_name} nicht erstellen.")

        print("\n--- Phase 3: Vorhersage & Benachrichtigung ---")
        settings = Settings.query.first()
        if not settings:
            # Dies sollte nach db.create_all() nicht passieren, aber zur Sicherheit
            settings = Settings(); db.session.add(settings); db.session.commit()
        
        device_tokens = [device.fcm_token for device in Device.query.all()]
        if not device_tokens: print("Keine registrierten Geräte gefunden. Überspringe Benachrichtigungen.")

        artifacts = TrainedModel.query.all()
        artifact_map = {artifact.name: pickle.loads(artifact.data) for artifact in artifacts}

        for asset_name, details in asset_map.items():
            # Stelle sicher, dass beide Modelle und Scaler vorhanden sind, bevor wir fortfahren
            low_model = artifact_map.get(details["low_model_name"])
            high_model = artifact_map.get(details["high_model_name"])
            low_scaler = artifact_map.get(details["low_scaler_name"])
            high_scaler = artifact_map.get(details["high_scaler_name"])

            if not all([low_model, high_model, low_scaler, high_scaler]):
                print(f"Modelle/Scaler für {asset_name} nicht vollständig geladen. Überspringe Vorhersage/Benachrichtigung.")
                continue
            
            # Hole Live-Features und mache Vorhersagen
            live_featured_data = add_features_to_data(download_historical_data(details["ticker"]))
            if live_featured_data is None or not all(col in live_featured_data.columns for col in FEATURES_LIST):
                print(f"Konnte keine Live-Features für {asset_name} erstellen."); continue
            
            latest_features_df = live_featured_data[FEATURES_LIST].tail(1)
            predicted_low = low_model.predict(low_scaler.transform(latest_features_df))[0]
            predicted_high = high_model.predict(high_scaler.transform(latest_features_df))[0]
            
            # Erstelle den Signal-Text (der jetzt länger ist)
            new_signal_text = f"Einstieg: {predicted_low:.2f}, TP: {predicted_high:.2f}"
            
            last_signal = getattr(settings, details["last_signal_key"])
            print(f"Analyse für {asset_name}: Letztes Signal='{last_signal}', Neues Signal='{new_signal_text}'")

            # Benachrichtigungslogik: Sende, wenn sich das Signal ändert ODER wenn es "N/A" ist (erster Lauf)
            if new_signal_text != last_signal and device_tokens:
                print(f"-> Signal für {asset_name} hat sich geändert! Sende Benachrichtigung...")
                title = f"Neues Preis-Ziel: {asset_name}"
                body = f"Neues Ziel: Einstieg ca. {predicted_low:.2f}, Take Profit ca. {predicted_high:.2f}"
                send_notification(title, body, device_tokens)
                
                # Speichere das neue Signal in der Datenbank
                setattr(settings, details["last_signal_key"], new_signal_text)
                db.session.commit()
            else:
                print(f"-> Signal für {asset_name} unverändert. Keine Aktion nötig.")
    
    # Phase 4: Redeployment auslösen
    trigger_web_service_redeploy()
    print("\n\nPipeline erfolgreich durchgelaufen!")

if __name__ == '__main__':
    run_full_pipeline()