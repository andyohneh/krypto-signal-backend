# run_training_pipeline.py (Temporärer DEBUG-Code)
import sys
import os
print("--- Test 1: Skript-Start ---")
print(f"Python Version: {sys.version}")
print("Skript wird ausgeführt. Python-Interpreter funktioniert.")

try:
    print("\n--- Test 2: Importiere einfache Bibliotheken ---")
    import json
    import pickle
    import requests
    from dotenv import load_dotenv
    print("Einfache Bibliotheken erfolgreich importiert.")
except Exception as e:
    print(f"FEHLER bei einfachen Imports: {e}")

try:
    print("\n--- Test 3: Importiere komplexe Daten-Bibliotheken ---")
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    print("Daten-Bibliotheken (pandas, numpy, sklearn) erfolgreich importiert.")
except Exception as e:
    print(f"FEHLER beim Import der Daten-Bibliotheken: {e}")

try:
    print("\n--- Test 4: Importiere komplexe Cloud-Bibliotheken ---")
    from flask import Flask
    from flask_sqlalchemy import SQLAlchemy
    import firebase_admin
    print("Cloud-Bibliotheken (Flask, SQLAlchemy, Firebase) erfolgreich importiert.")
except Exception as e:
    print(f"FEHLER beim Import der Cloud-Bibliotheken: {e}")

try:
    print("\n--- Test 5: Importiere unsere Helfer-Skripte ---")
    from data_manager import download_historical_data
    from feature_engineer import add_features_to_data
    from train_model import train_and_evaluate_model, FEATURES_LIST
    print("Helfer-Skripte erfolgreich importiert.")
except Exception as e:
    print(f"FEHLER beim Import der Helfer-Skripte: {e}")

print("\n\nDebug-Skript erfolgreich bis zum Ende durchgelaufen!")