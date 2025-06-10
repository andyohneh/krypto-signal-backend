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

from data_manager import download_historical_data
from feature_engineer import add_features_to_data, create_regression_targets
from train_model import train_regression_model, FEATURES_LIST

# --- Setup ---
app = Flask(__name__)
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate("serviceAccountKey.json")
    except FileNotFoundError:
        try:
            cred_str = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
            if cred_str: cred = credentials.Certificate(json.loads(cred_str))
            else: cred = None
        except Exception: cred = None
    if cred: firebase_admin.initialize_app(cred)

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Datenbank-Modelle ---
class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    update_interval_minutes = db.Column(db.Integer, default=15)
    last_btc_signal = db.Column(db.String(100), default='N/A')
    last_gold_signal = db.Column(db.String(100), default='N/A')

class TrainedModel(db.Model):
    id=db.Column(db.Integer, primary_key=True); name=db.Column(db.String(80), unique=True, nullable=False); data=db.Column(LargeBinary, nullable=False); timestamp=db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())

class Device(db.Model):
    id=db.Column(db.Integer, primary_key=True); fcm_token=db.Column(db.String(255), unique=True, nullable=False); timestamp=db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())

# --- Helfer-Funktionen ---
def save_artifact_to_db(name, artifact):
    with app.app_context():
        print(f"Speichere '{name}' in der DB...")
        pickled_artifact = pickle.dumps(artifact)
        existing_artifact = TrainedModel.query.filter_by(name=name).first()
        if existing_artifact: existing_artifact.data = pickled_artifact
        else: db.session.add(TrainedModel(name=name, data=pickled_artifact))
        db.session.commit()
        print(f"'{name}' in DB gespeichert.")

def send_notification(title, body, tokens):
    # ... (Code bleibt gleich)
    pass

def trigger_web_service_redeploy():
    # ... (Code bleibt gleich)
    pass

def run_full_pipeline():
    print("Starte die vollständige Regressions-Trainings-Pipeline...")
    with app.app_context():
        db.create_all()
        # ... (Rest der Logik bleibt gleich)
        
if __name__ == '__main__':
    run_full_pipeline()