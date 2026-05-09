import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings

from src.data_generator import generate_em_microstructure_data
from src.features import engineer_microstructure_features
from src.backtesters import (
    backtest_illiquid_strategy,
    backtest_twap_passive_execution,
    backtest_hourly_strategy
)

warnings.filterwarnings('ignore')

def print_metrics(name, returns, periods_per_year):
    total_return = (returns.iloc[-1] / returns.iloc[0]) - 1
    pct_returns = returns.pct_change().fillna(0)
    ann_vol = pct_returns.std() * np.sqrt(periods_per_year)
    sharpe = (pct_returns.mean() / pct_returns.std()) * np.sqrt(periods_per_year) if pct_returns.std() != 0 else 0
    
    print(f"--- {name} Metrics ---")
    print(f"Total Return:   {total_return * 100:.2f}%")
    print(f"Annualized Vol: {ann_vol * 100:.2f}%")
    print(f"Sharpe Ratio:   {sharpe:.2f}\n")

def main():
    print("=====================================================")
    print(" EM Microstructure: Illiquid Environments StatArb")
    print("=====================================================\n")

    # 1. Generate Synthetic Data
    print("[1/5] Generating synthetic IDX microstructure data...")
    df = generate_em_microstructure_data(n_assets=15, n_days=60)
    print(f"      Generated {len(df)} rows of 1-minute tick data.\n")

    # 2. Feature Engineering
    print("[2/5] Engineering microstructure features (OFI, Amihud, Roll)...")
    df_features = engineer_microstructure_features(df)

    # 3. Model Training (1-Minute Frequency)
    print("[3/5] Training cross-sectional LightGBM Alpha model...")
    features_cols = ['OFI_Proxy', 'OFI_EMA_15', 'Amihud_EMA_30', 'Roll_Spread', 'Return_5m', 'Return_15m', 'Is_Stale']
    
    split_idx = int(len(df_features) * 0.7)
    train_data = df_features.iloc[:split_idx]
    test_data = df_features.iloc[split_idx:].copy()

    lgb_params = {
        'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05,
        'num_leaves': 31, 'max_depth': 5, 'feature_fraction': 0.8, 'verbose': -1
    }
    
    train_dataset = lgb.Dataset(train_data[features_cols], label=train_data['Target_15m_Fwd'])
    model = lgb.train(lgb_params, train_dataset, num_boost_round=150)
    
    test_data['Alpha_Score'] = model.predict(test_data[features_cols])
    test_data['Rank'] = test_data.groupby(test_data.index)['Alpha_Score'].rank(pct=True)

    # 4. Run Backtests
    print("\n[4/5] Running High-Frequency Execution Engines...")
    
    # Experiment 1: Naive Execution (Aggressive Taker)
    print("      -> Simulating Naive Execution (Market Orders)...")
    bt_naive = backtest_illiquid_strategy(test_data)
    
    # Experiment 2: TWAP Execution (Passive Maker)
    print("      -> Simulating TWAP Algorithm (Limit Orders)...")
    bt_twap = backtest_twap_passive_execution(test_data)

    # 5. Hourly Timeframe Shift
    print("\n[5/5] Resampling to 1-Hour bars for Timeframe Shift Experiment...")
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    hourly_data = []
    for asset, group in df.groupby('Asset'):
        resampled = group.resample('60T').agg(agg_dict).dropna()
        resampled['Asset'] = asset
        hourly_data.append(resampled)
    df_hourly = pd.concat(hourly_data).sort_index()

    # Engineer hourly features
    df_h_features = df_hourly.copy()
    grouped_h = df_h_features.groupby('Asset')
    df_h_features['Return_1H'] = grouped_h['Close'].pct_change(1)
    df_h_features['Return_3H'] = grouped_h['Close'].pct_change(3)
    df_h_features['Vol_3H'] = grouped_h['Close'].pct_change(1).transform(lambda x: x.rolling(3).std())
    df_h_features['Target_1H_Fwd'] = grouped_h['Close'].transform(lambda x: x.pct_change(1).shift(-1))
    df_h_features['Spread_Cost'] = df_h_features['Close'] * 0.005
    df_h_features = df_h_features.dropna()

    split_idx_h = int(len(df_h_features) * 0.7)
    train_data_h = df_h_features.iloc[:split_idx_h]
    test_data_h = df_h_features.iloc[split_idx_h:].copy()
    
    lgb_train_h = lgb.Dataset(train_data_h[['Return_1H', 'Return_3H', 'Vol_3H']], label=train_data_h['Target_1H_Fwd'])
    model_h = lgb.train({'objective': 'regression', 'verbose': -1, 'learning_rate': 0.01}, lgb_train_h, num_boost_round=50)
    
    test_data_h['Alpha_Score'] = model_h.predict(test_data_h[['Return_1H', 'Return_3H', 'Vol_3H']])
    test_data_h['Rank'] = test_data_h.groupby(test_data_h.index)['Alpha_Score'].rank(pct=True)

    print("      -> Simulating Hourly Execution...")
    bt_hourly = backtest_hourly_strategy(test_data_h)

    # --- Print Final Results ---
    print("\n=====================================================")
    print(" FINAL SIMULATION RESULTS")
    print("=====================================================\n")
    
    MINUTES_PER_YEAR = 252 * 390
    HOURS_PER_YEAR = 252 * 6.5
    
    print_metrics("Experiment 1: Naive Market Orders", bt_naive['Portfolio_Value'], MINUTES_PER_YEAR)
    print_metrics("Experiment 2: TWAP Passive Limit Orders", bt_twap['Portfolio_Value'], MINUTES_PER_YEAR)
    print_metrics("Experiment 3: Hourly Timeframe Shift", bt_hourly['Portfolio_Value'], HOURS_PER_YEAR)

    print("Pipeline Complete. Review notebooks/ for visual charts.")

if __name__ == "__main__":
    main()
