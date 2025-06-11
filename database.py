# lib/database.py

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import LargeBinary, func

# 1. Erstelle das SQLAlchemy-Objekt. Es ist noch nicht mit einer App verbunden.
db = SQLAlchemy()

# 2. Definiere alle unsere Datenbank-Modelle (die Baupläne) an diesem einen, zentralen Ort.

class Settings(db.Model):
    __tablename__ = 'settings'
    id = db.Column(db.Integer, primary_key=True)
    update_interval_minutes = db.Column(db.Integer, default=15)
    last_btc_signal = db.Column(db.String(100), default='N/A')
    last_gold_signal = db.Column(db.String(100), default='N/A')

class TrainedModel(db.Model):
    __tablename__ = 'trained_model'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())

class Device(db.Model):
    __tablename__ = 'device'
    id = db.Column(db.Integer, primary_key=True)
    fcm_token = db.Column(db.String(255), unique=True, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())