# backtester.py (Finale Version - Speichert beste Parameter in DB)

import os
import pickle
from dotenv import load_dotenv
load_dotenv()

from flask import Flask
import pandas as pd

# Importiere jetzt auch Settings aus unserer zentralen DB-Datei
from database import db, TrainedModel, BacktestResult, Settings
from data_manager import download_historical_data
from feature_engineer import add_features_to_data
from train_model import FEATURES_LIST

# --- Setup ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def run_backtest_simulation(ticker, model_prefix, initial_capital=100.0, 
                            entry_threshold_percent=5.0, 
                            sl_atr_multiplier=1.5):
    
    print(f"\n--- Starte Backtest für {ticker} | Einstieg: >{entry_threshold_percent}% | SL: {sl_atr_multiplier}x ATR ---")

    with app.app_context():
        models = {}
        required_keys = [f"{model_prefix}_low_model", f"{model_prefix}_low_scaler", f"{model_prefix}_high_model", f"{model_prefix}_high_scaler"]
        for key in required_keys:
            artifact = TrainedModel.query.filter_by(name=key).first()
            if not artifact: print(f"FEHLER: Artefakt '{key}' nicht gefunden."); return 0.0, []
            models[key] = pickle.loads(artifact.data)
        
        historical_data = download_historical_data(ticker, period="2y")
        featured_data = add_features_to_data(historical_data)
        if featured_data is None: return 0.0, []

        capital = initial_capital
        in_trade = False
        entry_price, take_profit_target, stop_loss_target = 0, 0, 0
        portfolio_history = []

        for i in range(len(featured_data) - 1):
            current_day = featured_data.iloc[i]
            next_day = featured_data.iloc[i+1]
            
            if in_trade:
                if next_day['Low'] <= stop_loss_target:
                    capital *= (1 + ((stop_loss_target / entry_price) - 1)); in_trade = False
                elif next_day['High'] >= take_profit_target:
                    capital *= (1 + ((take_profit_target / entry_price) - 1)); in_trade = False

            if not in_trade:
                features_df = pd.DataFrame([current_day[FEATURES_LIST]])
                predicted_low = models[f'{model_prefix}_low_model'].predict(models[f'{model_prefix}_low_scaler'].transform(features_df))[0]
                predicted_high = models[f'{model_prefix}_high_model'].predict(models[f'{model_prefix}_high_scaler'].transform(features_df))[0]
                
                if predicted_low > 0 and ((predicted_high / predicted_low) - 1) * 100 > entry_threshold_percent:
                    in_trade = True; entry_price = next_day['Open']
                    take_profit_target = predicted_high
                    stop_loss_target = predicted_low - (current_day['ATRr_14'] * sl_atr_multiplier)
            
            portfolio_history.append({'date': current_day.name, 'balance': capital})
        
        print(f"--> Testlauf beendet. Endkapital: {capital:.2f}")
        return capital, portfolio_history

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    # --- DAS STRATEGIE-LABOR ---
    assets_to_test = {
        "BTC": {"ticker": "BTC-USD", "prefix": "btc"},
        "Gold": {"ticker": "GC=F", "prefix": "gold"}
    }
    
    entry_thresholds = [3, 5, 7]
    sl_multipliers = [1.0, 1.5, 2.0]
    
    # Führe die Forschung für jedes Asset durch
    for asset_name, asset_details in assets_to_test.items():
        print("\n" + "="*40 + f" {asset_name.upper()} FORSCHUNG START " + "="*40)
        results = []
        for threshold in entry_thresholds:
            for multiplier in sl_multipliers:
                final_capital, history = run_backtest_simulation(
                    ticker=asset_details["ticker"], 
                    model_prefix=asset_details["prefix"],
                    entry_threshold_percent=threshold,
                    sl_atr_multiplier=multiplier
                )
                if final_capital > 0:
                    results.append({'params': {'Einstieg': threshold, 'SL': multiplier}, 'result': final_capital, 'history': history})
        
        # Finde und speichere die beste Strategie für dieses Asset
        if results:
            best_run = max(results, key=lambda x: x['result'])
            print(f"\nBester {asset_name} End-Kontostand: {best_run['result']:.2f}")
            print(f"Beste {asset_name} Strategie-Parameter: {best_run['params']}")
            
            # --- NEU: SPEICHERE DIE BESTEN PARAMETER IN DER DB ---
            with app.app_context():
                settings = Settings.query.first()
                if not settings: settings = Settings()
                
                # Setze die Attribute dynamisch basierend auf dem Asset-Namen
                setattr(settings, f"{asset_details['prefix']}_entry_threshold", best_run['params']['Einstieg'])
                setattr(settings, f"{asset_details['prefix']}_sl_multiplier", best_run['params']['SL'])
                
                db.session.add(settings)
                db.session.commit()
                print(f"Beste Strategie-Parameter für {asset_name} in DB gespeichert.")

                # Speichere die beste Historie für den Chart in der DB
                BacktestResult.query.filter_by(asset_name=asset_details["ticker"]).delete()
                for record in best_run['history']:
                    db.session.add(BacktestResult(asset_name=asset_details["ticker"], date=record['date'], balance=record['balance']))
                db.session.commit()
                print(f"Beste {asset_name} Backtest-Historie in DB gespeichert.")
    
    print("\n\nFORSCHUNG ABGESCHLOSSEN!")