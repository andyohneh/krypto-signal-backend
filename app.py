import os
import json
import requests
import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import LargeBinary, func
import firebase_admin
from firebase_admin import credentials, messaging
from dotenv import load_dotenv

# Lade Umgebungsvariablen aus .env-Datei (lokal)
load_dotenv()

# Importiere deine lokalen Module
from data_manager import download_historical_data
from feature_engineer import add_features_to_data, create_regression_targets
from train_model import FEATURES_LIST, train_regression_model

app = Flask(__name__)

# Konfiguration der Datenbank
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Firebase-Initialisierung ---
# Diese Initialisierung ist für den Web Service (app.py) relevant
# Sie muss nur einmal erfolgen.
if not firebase_admin._apps:
    cred = None  # WICHTIG: cred hier initialisieren
    try:
        # Versuch 1: Lade aus lokaler Datei (für lokale Entwicklung)
        cred = credentials.Certificate("serviceAccountKey.json")
        print("Firebase-Credentials aus lokaler Datei geladen.")
    except FileNotFoundError:
        print("Lokale Schlüsseldatei nicht gefunden. Versuche Umgebungsvariable...")
        try:
            # Versuch 2: Lade aus Umgebungsvariable (für Render)
            cred_str = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
            if cred_str:
                cred = credentials.Certificate(json.loads(cred_str))
                print("Firebase-Credentials aus Umgebungsvariable geladen.")
            else:
                print("WARNUNG: FIREBASE_SERVICE_ACCOUNT_JSON Variable nicht gefunden.")
        except json.JSONDecodeError as e:
            print(f"FEHLER beim Parsen der Firebase JSON Umgebungsvariable: {e}")
        except Exception as e:
            print(f"Allgemeiner FEHLER beim Laden der Firebase-Credentials aus Umgebungsvariable: {e}")
    except Exception as e:
        print(f"Allgemeiner FEHLER beim Laden der Firebase-Credentials: {e}")

    if cred:
        try:
            firebase_admin.initialize_app(cred)
            print("Firebase erfolgreich initialisiert.")
        except ValueError as e:
            print(f"Firebase bereits initialisiert oder Fehler: {e}") # Sollte hier nicht passieren, da wir prüfen
        except Exception as e:
            print(f"FEHLER bei der Firebase-Initialisierung: {e}")
    else:
        print("FEHLER: Firebase-Credentials konnten NICHT geladen werden. Firebase wird NICHT initialisiert.")

# --- Datenbankmodelle ---
class Device(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fcm_token = db.Column(db.String(255), unique=True, nullable=False)
    timestamp = db.Column(db.DateTime, default=func.now())

class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    last_btc_signal = db.Column(db.String(100), default='N/A') # Erhöht auf 100
    last_gold_signal = db.Column(db.String(100), default='N/A') # Erhöht auf 100
    scaler_btc_low = db.Column(LargeBinary)
    model_btc_low = db.Column(LargeBinary)
    scaler_btc_high = db.Column(LargeBinary)
    model_btc_high = db.Column(LargeBinary)
    scaler_gold_low = db.Column(LargeBinary)
    model_gold_low = db.Column(LargeBinary)
    scaler_gold_high = db.Column(LargeBinary)
    model_gold_high = db.Column(LargeBinary)
    model_update_timestamp = db.Column(db.DateTime, default=func.now())

    def update_model(self, asset_type, model_type, scaler, model):
        scaler_col = f'scaler_{asset_type}_{model_type}'
        model_col = f'model_{asset_type}_{model_type}'
        setattr(self, scaler_col, pickle.dumps(scaler))
        setattr(self, model_col, pickle.dumps(model))

    def get_model(self, asset_type, model_type):
        scaler_col = f'scaler_{asset_type}_{model_type}'
        model_col = f'model_{asset_type}_{model_type}'
        scaler = pickle.loads(getattr(self, scaler_col)) if getattr(self, scaler_col) else None
        model = pickle.loads(getattr(self, model_col)) if getattr(self, model_col) else None
        return scaler, model

with app.app_context():
    db.create_all()
    # Sicherstellen, dass immer ein Einstellungs-Eintrag existiert
    if not Settings.query.first():
        db.session.add(Settings())
        db.session.commit()
        print("Initialer Settings-Eintrag erstellt.")

# --- Hilfsfunktion für Benachrichtigungen ---
def send_notification(title, body, tokens):
    if not tokens:
        print("Keine Tokens für den Versand von Benachrichtigungen vorhanden.")
        return

    # Firebase-Initialisierung prüfen
    if not firebase_admin._apps:
        print("Firebase ist nicht initialisiert. Nachricht kann nicht gesendet werden.")
        return

    # Sende Nachrichten in Batches, um API-Limits zu beachten
    message = messaging.MulticastMessage(
        notification=messaging.Notification(title=title, body=body),
        tokens=tokens,
    )
    try:
        response = messaging.send_multicast(message)
        print(f"Erfolgreich {response.success_count} Nachrichten gesendet, {response.failure_count} Fehler.")
        if response.failure_count > 0:
            for resp in response.responses:
                if not resp.success:
                    print(f"Fehler beim Senden: {resp.exception}")
    except Exception as e:
        print(f"Fehler beim Senden der Benachrichtigung: {e}")

# --- Routen ---
@app.route('/')
def home():
    return "KI-Modell Training Service läuft!"

@app.route('/register_device', methods=['POST'])
def register_device():
    data = request.get_json()
    token = data.get('token')

    if not token:
        return jsonify({"status": "error", "message": "Kein Token erhalten."}), 400

    with app.app_context():
        existing_device = Device.query.filter_by(fcm_token=token).first()
        if existing_device:
            existing_device.timestamp = func.now()
            db.session.commit()
            print(f"Geräte-Token {token[:15]}... bereits vorhanden, Zeitstempel aktualisiert.")
            return jsonify({"status": "success", "message": "Gerät bereits registriert."})
        else:
            new_device = Device(fcm_token=token)
            db.session.add(new_device)
            db.session.commit()
            print(f"Neues Gerät mit Token {token[:15]}... registriert.")
            return jsonify({"status": "success", "message": "Gerät erfolgreich registriert."})

@app.route('/send_test_notification', methods=['POST'])
def send_test_notification():
    data = request.get_json()
    token = data.get('token')

    if not token:
        return jsonify({"status": "error", "message": "Kein Token erhalten."}), 400

    title = "Test!"
    body = "Funktioniert! 🎉"
    send_notification(title, body, [token]) # Nutze die neue send_notification Funktion

    return jsonify({"status": "success", "message": "Testnachricht gesendet (siehe Logs für Details)."})

@app.route('/get_latest_signals', methods=['GET'])
def get_latest_signals():
    with app.app_context():
        settings = Settings.query.first()
        if settings:
            return jsonify({
                "status": "success",
                "btc_signal": settings.last_btc_signal,
                "gold_signal": settings.last_gold_signal,
                "last_update": settings.model_update_timestamp.isoformat()
            })
        else:
            return jsonify({"status": "error", "message": "Keine Signale gefunden."}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.getenv("PORT", 5000))