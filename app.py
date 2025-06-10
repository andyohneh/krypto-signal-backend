import os, json, pickle, requests, numpy as np, pandas as pd, pandas_ta as ta
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import LargeBinary, func
import firebase_admin
from firebase_admin import credentials, messaging
from dotenv import load_dotenv
from data_manager import download_historical_data
from feature_engineer import add_features_to_data
from train_model import FEATURES_LIST

load_dotenv()
app = Flask(__name__)

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate("serviceAccountKey.json")
    except FileNotFoundError:
        try:
            cred_str = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
            if cred_str: cred = credentials.Certificate(json.loads(cred_str))
            else: cred = None
        except Exception as e: cred = None
    if cred: firebase_admin.initialize_app(cred)

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Settings(db.Model):
    id=db.Column(db.Integer, primary_key=True); update_interval_minutes=db.Column(db.Integer, default=15); last_btc_signal=db.Column(db.String(100), default='N/A'); last_gold_signal=db.Column(db.String(100), default='N/A')
class TrainedModel(db.Model):
    id=db.Column(db.Integer, primary_key=True); name=db.Column(db.String(80), unique=True, nullable=False); data=db.Column(LargeBinary, nullable=False); timestamp=db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())
class Device(db.Model):
    id=db.Column(db.Integer, primary_key=True); fcm_token=db.Column(db.String(255), unique=True, nullable=False); timestamp=db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())

models = {}
def load_artifacts_from_db():
    global models
    with app.app_context():
        try:
            for m in TrainedModel.query.all(): models[m.name] = pickle.loads(m.data)
            print(f"Erfolgreich {len(models)} Artefakte geladen.")
        except Exception as e: print(f"FEHLER beim Laden der Artefakte: {e}")
def load_settings_from_db():
    settings = Settings.query.first()
    if not settings: settings = Settings(); db.session.add(settings); db.session.commit()
    return {c.name: getattr(settings, c.name) for c in settings.__table__.columns if c.name != 'id'}

with app.app_context():
    db.create_all()
    current_settings = load_settings_from_db()
load_artifacts_from_db()

@app.route('/get_chart_data/<ticker_symbol>')
def get_chart_data(ticker_symbol):
    try:
        data = download_historical_data(ticker_symbol, period="6mo", interval="1d")
        if data is None: return jsonify({"error": "Rohdaten laden fehlgeschlagen"}), 500
        data['SMA_10'] = data['Adj Close'].rolling(window=10).mean()
        data['SMA_50'] = data['Adj Close'].rolling(window=50).mean()
        data['RSI_14'] = ta.rsi(data['Adj Close'], length=14)
        chart_columns = ['Adj Close', 'SMA_10', 'SMA_50', 'RSI_14']
        chart_data = data[chart_columns].copy()
        chart_data.rename(columns={'Adj Close': 'price', 'SMA_10': 'sma_short', 'SMA_50': 'sma_long', 'RSI_14': 'rsi'}, inplace=True)
        chart_data.reset_index(inplace=True)
        chart_data['Date'] = chart_data['Date'].dt.strftime('%Y-%m-%d')
        return jsonify(chart_data.dropna().to_dict(orient="records"))
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/get_signals')
def get_signals():
    global current_settings
    bitcoin_data, gold_data, error_msg = {}, {}, ""
    btc_keys = ['btc_low_model', 'btc_low_scaler', 'btc_high_model', 'btc_high_scaler']
    if all(k in models for k in btc_keys):
        try:
            current_price = float(requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT").json()['price'])
            features = add_features_to_data(download_historical_data("BTC-USD", "3mo")).tail(1)
            if not features.empty:
                low_pred = models['btc_low_model'].predict(models['btc_low_scaler'].transform(features[FEATURES_LIST]))[0]
                high_pred = models['btc_high_model'].predict(models['btc_high_scaler'].transform(features[FEATURES_LIST]))[0]
                sl = low_pred - (features['ATRr_14'].iloc[0] * 1.5)
                bitcoin_data = {"price": round(current_price, 2), "entry": round(low_pred, 2), "take_profit": round(high_pred, 2), "stop_loss": round(sl, 2)}
            else: error_msg += "BTC Feature-Erstellung fehlgeschlagen. "
        except Exception as e: error_msg += f"BTC Fehler: {e}. "; bitcoin_data={"price":"Fehler"}
    else: error_msg += "BTC Modelle nicht geladen. "
    
    # ... (analoge Logik für Gold) ...

    return jsonify({"bitcoin": bitcoin_data, "gold": gold_data, "settings": current_settings, "global_error": error_msg.strip()})

# ... (andere Routen)
if __name__ == '__main__':
    app.run()