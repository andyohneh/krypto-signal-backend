# run_training_pipeline.py (Finale Version mit automatischer Benachrichtigung)

import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import LargeBinary, func
import firebase_admin
from firebase_admin import credentials, messaging

# Importiere unsere Helfer-Funktionen
from data_manager import download_historical_data
from feature_engineer import add_features_to_data
from train_model import train_model_for_asset

# --- Setup für Datenbank und Firebase (wird vom Cron Job benötigt) ---
app = Flask(__name__)
try:
    # Lade die Firebase-Zugangsdaten aus den Umgebungsvariablen
    cred_str = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
    if cred_str:
        cred = credentials.Certificate(json.loads(cred_str))
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialisiert.")
    else:
        print("WARNUNG: Firebase nicht initialisiert. Keine Benachrichtigungen möglich.")
except Exception as e:
    print(f"FEHLER bei Firebase-Initialisierung: {e}")

# Lade die Datenbank-URL aus den Umgebungsvariablen
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Datenbank-Modelle (müssen hier bekannt sein, um mit der DB zu sprechen) ---
class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    last_btc_signal = db.Column(db.String(10), default='N/A')
    last_gold_signal = db.Column(db.String(10), default='N/A')
    # Die anderen Spalten brauchen wir für diesen Job nicht, aber das Modell muss vollständig sein
    bitcoin_tp_percentage = db.Column(db.Float, default=2.5)
    bitcoin_sl_percentage = db.Column(db.Float, default=1.5)
    xauusd_tp_percentage = db.Column(db.Float, default=1.8)
    xauusd_sl_percentage = db.Column(db.Float, default=0.8)
    update_interval_minutes = db.Column(db.Integer, default=15)

class TrainedModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    data = db.Column(LargeBinary, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())

class Device(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fcm_token = db.Column(db.String(255), unique=True, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())

def send_notification(title, body, tokens):
    """Sendet eine Push-Benachrichtigung an eine Liste von Geräte-Tokens."""
    try:
        message = messaging.MulticastMessage(
            notification=messaging.Notification(title=title, body=body),
            tokens=tokens,
        )
        response = messaging.send_multicast(message)
        print(f'{response.success_count} Nachrichten erfolgreich gesendet.')
        if response.failure_count > 0:
            print(f'{response.failure_count} Nachrichten konnten nicht gesendet werden.')
    except Exception as e:
        print(f"Fehler beim Senden der Benachrichtigung: {e}")

def run_full_pipeline():
    print("Starte die vollständige Trainings- und Benachrichtigungs-Pipeline...")
    
    # Der App-Kontext wird für alle DB-Operationen benötigt
    with app.app_context():
        # Schritt 1 & 2: Daten laden, Features erstellen und CSVs speichern
        print("\n--- Phase 1: Datenaufbereitung ---")
        for ticker in ["BTC-USD", "GC=F"]:
            raw_data = download_historical_data(ticker)
            if raw_data is not None:
                featured_data = add_features_to_data(raw_data)
                if featured_data is not None:
                    filename = f"{ticker.lower().split('-')[0]}_data_with_features.csv"
                    featured_data.to_csv(filename)
                    print(f"{ticker}-Daten mit Features erfolgreich gespeichert.")

        # Schritt 3: Modelle mit den neuen Daten trainieren und in die DB speichern
        print("\n--- Phase 2: Modelltraining ---")
        train_model_for_asset("btc_data_with_features.csv", "btc_model", "btc_scaler")
        train_model_for_asset("gold_data_with_features.csv", "gold_model", "gold_scaler")

        # Schritt 4: Artefakte und Einstellungen aus der DB laden für die Vorhersage
        print("\n--- Phase 3: Vorhersage & Benachrichtigung ---")
        artifacts = TrainedModel.query.all()
        artifact_map = {artifact.name: pickle.loads(artifact.data) for artifact in artifacts}
        settings = Settings.query.first()
        device_tokens = [device.fcm_token for device in Device.query.all()]
        
        if not device_tokens:
            print("Keine registrierten Geräte gefunden. Überspringe Benachrichtigungen.")
        
        # Schritt 5: Für jedes Asset Signal prüfen und ggf. benachrichtigen
        asset_map = {
            "BTC": {"ticker": "BTC-USD", "model": artifact_map.get('btc_model'), "scaler": artifact_map.get('btc_scaler'), "last_signal_key": "last_btc_signal"},
            "Gold": {"ticker": "GC=F", "model": artifact_map.get('gold_model'), "scaler": artifact_map.get('gold_scaler'), "last_signal_key": "last_gold_signal"}
        }

        for asset_name, details in asset_map.items():
            model, scaler = details["model"], details["scaler"]
            if not model or not scaler:
                print(f"Modell/Scaler für {asset_name} nicht geladen. Überspringe.")
                continue

            # Live-Features für die Vorhersage holen
            live_features_df = add_features_to_data(download_historical_data(details["ticker"]))
            if live_features_df is None: continue

            features_for_scaling = live_features_df[['daily_return', 'SMA_10', 'SMA_50', 'sma_signal', 'RSI_14']]
            scaled_features = scaler.transform(features_for_scaling)
            latest_features = scaled_features[-1].reshape(1, -1)

            # Neue Vorhersage machen
            prediction = model.predict(latest_features)[0]
            new_signal = "Kauf" if prediction == 1 else "Verkauf"
            last_signal = getattr(settings, details["last_signal_key"])
            print(f"Analyse für {asset_name}: Letztes Signal='{last_signal}', Neues Signal='{new_signal}'")

            # Vergleichen und benachrichtigen
            if new_signal != last_signal and device_tokens:
                print(f"-> Signal für {asset_name} hat sich geändert! Sende Benachrichtigung...")
                title = f"Neues Signal: {asset_name}"
                body = f"Das Handelssignal für {asset_name} ist jetzt: {new_signal.upper()}"
                send_notification(title, body, device_tokens)
                
                # Letztes Signal in der DB aktualisieren
                setattr(settings, details["last_signal_key"], new_signal)
                db.session.commit()
            else:
                print(f"-> Signal für {asset_name} unverändert. Keine Aktion nötig.")
        
    print("\n\nPipeline erfolgreich durchgelaufen!")

if __name__ == '__main__':
    run_full_pipeline()