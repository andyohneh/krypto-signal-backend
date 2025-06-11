# backtester.py (Finale Version - Labor für BTC & Gold)

import os
import pickle
from dotenv import load_dotenv
load_dotenv()

from flask import Flask
import pandas as pd

from database import db, TrainedModel, BacktestResult
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
            if not artifact:
                print(f"FEHLER: Artefakt '{key}' nicht in der DB gefunden. Breche diesen Testlauf ab.")
                return 0.0, []
            models[key] = pickle.loads(artifact.data)
        
        historical_data = download_historical_data(ticker, period="2y")
        featured_data = add_features_to_data(historical_data)
        if featured_data is None:
            print("Konnte Features nicht erstellen."); return 0.0, []

        capital = initial_capital
        in_trade = False
        entry_price, take_profit_target, stop_loss_target = 0, 0, 0
        portfolio_history = []

        for i in range(len(featured_data) - 1):
            current_day = featured_data.iloc[i]
            next_day = featured_data.iloc[i+1]
            
            if in_trade:
                if next_day['Low'] <= stop_loss_target:
                    capital *= (1 + ((stop_loss_target / entry_price) - 1))
                    in_trade = False
                elif next_day['High'] >= take_profit_target:
                    capital *= (1 + ((take_profit_target / entry_price) - 1))
                    in_trade = False

            if not in_trade:
                features_df = pd.DataFrame([current_day[FEATURES_LIST]])
                predicted_low = models[f'{model_prefix}_low_model'].predict(models[f'{model_prefix}_low_scaler'].transform(features_df))[0]
                predicted_high = models[f'{model_prefix}_high_model'].predict(models[f'{model_prefix}_high_scaler'].transform(features_df))[0]
                
                if predicted_low > 0 and ((predicted_high / predicted_low) - 1) * 100 > entry_threshold_percent:
                    in_trade = True
                    entry_price = next_day['Open']
                    take_profit_target = predicted_high
                    stop_loss_target = predicted_low - (current_day['ATRr_14'] * sl_atr_multiplier)
            
            portfolio_history.append({'date': current_day.name, 'balance': capital})
        
        print(f"--> Testlauf beendet. Endkapital: {capital:.2f}")
        return capital, portfolio_history

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    # --- DAS STRATEGIE-LABOR ---
    entry_thresholds = [3, 5, 7]
    sl_multipliers = [1.0, 1.5, 2.0]
    
    # --- BITCOIN FORSCHUNG ---
    print("\n" + "="*40 + " BITCOIN FORSCHUNG START " + "="*40)
    btc_results = []
    for threshold in entry_thresholds:
        for multiplier in sl_multipliers:
            final_capital, history = run_backtest_simulation("BTC-USD", "btc", entry_threshold_percent=threshold, sl_atr_multiplier=multiplier)
            if final_capital > 0:
                btc_results.append({'params': {'Einstieg': threshold, 'SL': multiplier}, 'result': final_capital, 'history': history})
    
    # --- GOLD FORSCHUNG ---
    print("\n" + "="*40 + " GOLD FORSCHUNG START " + "="*40)
    gold_results = []
    for threshold in entry_thresholds:
        for multiplier in sl_multipliers:
            final_capital, history = run_backtest_simulation("GC=F", "gold", entry_threshold_percent=threshold, sl_atr_multiplier=multiplier)
            if final_capital > 0:
                gold_results.append({'params': {'Einstieg': threshold, 'SL': multiplier}, 'result': final_capital, 'history': history})

    # --- FINALE ERGEBNISSE & SPEICHERN IN DB ---
    print("\n" + "="*80)
    print("FORSCHUNG ABGESCHLOSSEN!")
    
    # Beste BTC Strategie finden und speichern
    if btc_results:
        best_btc_run = max(btc_results, key=lambda x: x['result'])
        print(f"\nBester BTC End-Kontostand: {best_btc_run['result']:.2f}")
        print(f"Beste BTC Strategie-Parameter: {best_btc_run['params']}")
        with app.app_context():
            BacktestResult.query.filter_by(asset_name="BTC-USD").delete()
            for record in best_btc_run['history']:
                db.session.add(BacktestResult(asset_name="BTC-USD", date=record['date'], balance=record['balance']))
            db.session.commit()
            print("Beste BTC Backtest-Historie in DB gespeichert.")

    # Beste Gold Strategie finden und speichern
    if gold_results:
        best_gold_run = max(gold_results, key=lambda x: x['result'])
        print(f"\nBester Gold End-Kontostand: {best_gold_run['result']:.2f}")
        print(f"Beste Gold Strategie-Parameter: {best_gold_run['params']}")
        with app.app_context():
            BacktestResult.query.filter_by(asset_name="GC=F").delete()
            for record in best_gold_run['history']:
                db.session.add(BacktestResult(asset_name="GC=F", date=record['date'], balance=record['balance']))
            db.session.commit()
            print("Beste Gold Backtest-Historie in DB gespeichert.")
            
    print("="*80)