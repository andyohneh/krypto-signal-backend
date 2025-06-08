# app.py (Plan B, Schritt 1: Datenbank vorbereiten)

import os
import json
import requests
import joblib
import numpy as np
import pickle # NEU: Wird später zum Serialisieren der Modelle benötigt
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import LargeBinary # NEU: Für Binärdaten in der DB
import firebase_admin
from firebase_admin import credentials, messaging

# Importiere unsere Helfer-Funktionen
from data_manager import download_historical_data
from feature_engineer import add_features_to_data


# --- Initialisierungen ---
app = Flask(__name__)
try:
    # Firebase initialisieren
    cred_str = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
    if cred_str:
        cred = credentials.Certificate(json.loads(cred_str))
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialisiert.")
except Exception as e:
    print(f"FEHLER bei Firebase-Initialisierung: {e}")

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# --- Datenbank-Modelle ---

class Settings(db.Model):
    # Diese Klasse bleibt unverändert
    id = db.Column(db.Integer, primary_key=True)
    bitcoin_tp_percentage = db.Column(db.Float, default=2.5)
    bitcoin_sl_percentage = db.Column(db.Float, default=1.5)
    xauusd_tp_percentage = db.Column(db.Float, default=1.8)
    xauusd_sl_percentage = db.Column(db.Float, default=0.8)
    update_interval_minutes = db.Column(db.Integer, default=15)

# NEU: Eine Tabelle zum Speichern unserer trainierten Modelle und Scaler
class TrainedModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False) # z.B. 'btc_model', 'btc_scaler'
    data = db.Column(LargeBinary, nullable=False) # Hier kommt das "eingescannte" Modell hinein
    timestamp = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())


# --- Platzhalter für Modelle ---
# Die Logik zum Laden aus der DB kommt im letzten Schritt.
# Vorerst setzen wir sie auf None, damit die App starten kann.
btc_model, gold_model, btc_scaler, gold_scaler = None, None, None, None
print("Modell-Variablen initialisiert. Lade-Logik wird später implementiert.")


# --- Datenbank- und Helfer-Funktionen ---
def load_settings_from_db():
    settings = Settings.query.first()
    if not settings:
        settings = Settings()
        db.session.add(settings)
        db.session.commit()
    # Wandelt das DB-Objekt in ein Dictionary um
    return {c.name: getattr(settings, c.name) for c in settings.__table__.columns}

def save_settings(new_settings):
    s = Settings.query.first()
    if s:
        for k, v in new_settings.items():
            if hasattr(s, k):
                setattr(s, k, v)
        db.session.commit()

# Diese Funktion wird vorerst nicht mehr direkt hier gebraucht,
# da die Logik zum Laden noch fehlt, aber wir lassen sie für die Struktur drin.
def get_scaled_live_features(ticker, scaler):
    raw_data = download_historical_data(ticker, period="3mo", interval="1d")
    if raw_data is None: return None
    featured_data = add_features_to_data(raw_data)
    if featured_data is None: return None
    features_for_scaling = featured_data[['daily_return', 'SMA_10', 'SMA_50', 'sma_signal', 'RSI_14']]
    scaled_features = scaler.transform(features_for_scaling)
    return scaled_features[-1].reshape(1, -1)


# --- App-Kontext zum Start ---
with app.app_context():
    # Dieser Befehl erstellt jetzt BEIDE Tabellen (Settings und TrainedModel),
    # falls sie noch nicht existieren.
    db.create_all()
    current_settings = load_settings_from_db()

print(f"Einstellungen beim Start geladen: {current_settings}")


# --- API-Routen ---
@app.route('/')
def home():
    return "Krypto Helfer Backend - Bereit für DB-Modell-Speicherung."

@app.route('/get_signals')
def get_signals():
    # HINWEIS: Diese Route wird einen Fehler produzieren, bis wir die Lade-Logik implementieren,
    # da die Modelle 'None' sind. Das ist okay für diesen Zwischenschritt.
    if not btc_model or not gold_model:
        return jsonify({
            "global_error": "Modelle werden noch nicht aus der DB geladen. Bitte nächsten Schritt ausführen."
        })
    # Die alte Logik würde hier folgen...
    return jsonify({"status": "ok"})


@app.route('/save_settings', methods=['POST'])
def save_app_settings():
    global current_settings; data = request.get_json()
    if data:
        current_settings.update(data); save_settings(current_settings)
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@app.route('/send_test_notification', methods=['POST'])
def send_test_notification():
    data = request.get_json(); token = data.get('token')
    if not token: return jsonify({"status": "error"}), 400
    try:
        message = messaging.Message(notification=messaging.Notification(title='Test!', body='Funktioniert! 🎉'), token=token)
        response = messaging.send(message)
        return jsonify({"status": "success", "response": str(response)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500