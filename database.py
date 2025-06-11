# lib/database.py (Finale Version mit Backtest-Tabelle)

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import LargeBinary, func, DateTime, String, Integer, Float # Importiere zusätzliche Typen

db = SQLAlchemy()

class Settings(db.Model):
    __tablename__ = 'settings'
    id = db.Column(Integer, primary_key=True)
    update_interval_minutes = db.Column(Integer, default=15)
    last_btc_signal = db.Column(String(100), default='N/A')
    last_gold_signal = db.Column(String(100), default='N/A')
    
    # NEU: Speicher für die besten Strategie-Parameter
    btc_entry_threshold = db.Column(Float, default=5.0)
    btc_sl_multiplier = db.Column(Float, default=1.5)
    gold_entry_threshold = db.Column(Float, default=5.0)
    gold_sl_multiplier = db.Column(Float, default=1.5)

class TrainedModel(db.Model):
    __tablename__ = 'trained_model'
    id = db.Column(Integer, primary_key=True)
    name = db.Column(String(80), unique=True, nullable=False)
    data = db.Column(LargeBinary, nullable=False)
    timestamp = db.Column(DateTime, server_default=func.now(), onupdate=func.now())

class Device(db.Model):
    __tablename__ = 'device'
    id = db.Column(Integer, primary_key=True)
    fcm_token = db.Column(String(255), unique=True, nullable=False)
    timestamp = db.Column(DateTime, server_default=func.now(), onupdate=func.now())

# NEU: Die Tabelle für unsere Backtest-Ergebnisse
class BacktestResult(db.Model):
    __tablename__ = 'backtest_result'
    id = db.Column(Integer, primary_key=True)
    asset_name = db.Column(String(50), nullable=False)
    date = db.Column(DateTime, nullable=False)
    balance = db.Column(Float, nullable=False)