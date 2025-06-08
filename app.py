# app.py (FINALE PRODUKTIONSVERSION - Lädt Modelle aus der DB)

import os
import json
import requests
import pickle
import numpy as np
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import LargeBinary
import firebase_admin
from firebase_admin import credentials, messaging

# Importiere unsere Helfer-Funktionen
from data_manager import download_historical_data
from feature_engineer import add_features_to_data

# --- Initialisierungen ---
app = Flask(__name__)
try:
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
    id = db.Column(db.Integer, primary_key=True)
    bitcoin_tp_percentage = db.Column(db.Float, default=2.5)
    bitcoin_sl_percentage = db.Column(db.Float, default=1.5)
    xauusd_tp_percentage = db.Column(db.Float, default=1.8)
    xauusd_sl_percentage = db.Column(db.Float, default=0.8)
    update_interval_minutes = db.Column(db.Integer, default=15)

class TrainedModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    data = db.Column(LargeBinary, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())

# --- Globale Variablen für geladene Artefakte ---
current_settings = {}
btc_model, gold_model, btc_scaler, gold_scaler = None, None, None, None

# --- Datenbank- & Helfer-Funktionen ---
def load_artifacts_from_db():
    """Lädt alle Modelle und Scaler aus der Datenbank in die globalen Variablen."""
    global btc_model, gold_model, btc_scaler, gold_scaler
    print("Versuche, Modelle und Scaler aus der DB zu laden...")
    try:
        # Wichtig: Wir müssen im App-Kontext sein, um auf die DB zuzugreifen
        with app.app_context():
            artifacts = TrainedModel.query.all()
            if not artifacts:
                print("WARNUNG: Keine Modelle in der DB gefunden. Bitte den Cron Job ausführen.")
                return

            artifact_map = {}
            for artifact in artifacts:
                artifact_map[artifact.name] = pickle.loads(artifact.data)
            
            btc_model = artifact_map.get('btc_model')
            gold_model = artifact_map.get('gold_model')
            btc_scaler = artifact_map.get('btc_scaler')
            gold_scaler = artifact_map.get('gold_scaler')
            
            # Überprüfen, ob alle geladen wurden
            if all([btc_model, gold_model, btc_scaler, gold_scaler]):
                print(f"Erfolgreich {len(artifacts)} Modelle/Scaler aus der DB geladen.")
            else:
                print("WARNUNG: Einige Modelle/Scaler konnten nicht in der DB gefunden werden.")

    except Exception as e:
        print(f"FEHLER beim Laden der Artefakte aus der DB: {e}")

def load_settings_from_db():
    settings = Settings.query.first()
    if not settings:
        settings = Settings(); db.session.add(settings); db.session.commit()
    return {c.name: getattr(settings, c.name) for c in settings.__table__.columns if c.name != 'id'}

def save_settings(new_settings):
    s = Settings.query.first()
    if s: [setattr(s, k, v) for k, v in new_settings.items() if hasattr(s, k)]; db.session.commit()

def get_scaled_live_features(ticker, scaler):
    raw_data = download_historical_data(ticker, period="3mo", interval="1d")
    if raw_data is None: return None
    featured_data = add_features_to_data(raw_data)
    if featured_data is None: return None
    features_for_scaling = featured_data[['daily_return', 'SMA_10', 'SMA_50', 'sma_signal', 'RSI_14']]
    scaled_features = scaler.transform(features_for_scaling)
    return scaled_features[-1].reshape(1, -1)

# --- App-Start ---
with app.app_context():
    db.create_all()
    current_settings = load_settings_from_db()
load_artifacts_from_db() # Lade die Modelle beim Start

# --- API-Routen ---
@app.route('/')
def home():
    return "Krypto Helfer Backend - Finale KI-Modelle sind live!"

# ... (alle anderen Routen wie /get_signals, /save_settings etc. bleiben exakt wie in der letzten Version) ...
@app.route('/get_signals')
def get_signals():
    global current_settings
    bitcoin_data, gold_data, error_msg = {}, {}, ""
    if btc_model and btc_scaler:
        try:
            price = float(requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT").json()['price'])
            features = get_scaled_live_features("BTC-USD", btc_scaler)
            if features is not None:
                prediction = btc_model.predict(features)[0]
                signal = "Kauf" if prediction == 1 else "Verkauf"
                tp = price * (1 + current_settings.get("bitcoin_tp_percentage", 2.5)/100)
                sl = price * (1 - current_settings.get("bitcoin_sl_percentage", 1.5)/100)
                bitcoin_data = {"price": round(price,2), "entry": round(price,2), "take_profit": round(tp,2), "stop_loss": round(sl,2), "signal_type": signal}
            else: error_msg += "BTC Feature-Erstellung fehlgeschlagen. "; bitcoin_data={"signal_type":"Fehler"}
        except Exception as e: error_msg += f"BTC Fehler: {e}. "; bitcoin_data={"signal_type":"Fehler"}
    else: error_msg += "BTC Modell/Scaler nicht geladen. "
    
    if gold_model and gold_scaler:
        try:
            FMP_API_KEY = os.environ.get('FMP_API_KEY')
            price = float(requests.get(f'https://financialmodelingprep.com/api/v3/quote/XAUUSD?apikey={FMP_API_KEY}').json()[0]['price'])
            features = get_scaled_live_features("GC=F", gold_scaler)
            if features is not None:
                prediction = gold_model.predict(features)[0]
                signal = "Kauf" if prediction == 1 else "Verkauf"
                tp = price * (1 + current_settings.get("xauusd_tp_percentage", 1.8)/100)
                sl = price * (1 - current_settings.get("xauusd_sl_percentage", 0.8)/100)
                gold_data = {"price": round(price,2), "entry": round(price,2), "take_profit": round(tp,2), "stop_loss": round(sl,2), "signal_type": signal}
            else: error_msg += "Gold Feature-Erstellung fehlgeschlagen. "; gold_data={"signal_type":"Fehler"}
        except Exception as e: error_msg += f"Gold Fehler: {e}. "; gold_data={"signal_type":"Fehler"}
    else: error_msg += "Gold Modell/Scaler fehlt. "

    response = {"bitcoin": bitcoin_data, "gold": gold_data, "settings": current_settings}
    if error_msg: response["global_error"] = error_msg.strip()
    return jsonify(response)
    
@app.route('/save_settings', methods=['POST'])
def save_app_settings():
    global current_settings; data = request.get_json()
    if data: current_settings.update(data); save_settings(current_settings); return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

# NEUER API-ENDPUNKT ZUM REGISTRIEREN VON GERÄTEN
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
    data = request.get_json(); token = data.get('token')
    if not token: return jsonify({"status": "error"}), 400
    try:
        message = messaging.Message(notification=messaging.Notification(title='Test!', body='Funktioniert! 🎉'), token=token)
        response = messaging.send(message)
        return jsonify({"status": "success", "response": str(response)})
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 500