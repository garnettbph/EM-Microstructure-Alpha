import numpy as np
import pandas as pd

def engineer_microstructure_features(df):
    """
    Engineers microstructure features to capture order book imbalance and noise.
    Calculates Order Flow Imbalance (OFI), Amihud Illiquidity, and Roll's Measure.
    """
    features = pd.DataFrame(index=df.index)
    
    # Group by asset
    grouped = df.groupby('Asset')
    
    # 1. Tick Rule for Order Flow Imbalance (OFI) proxy
    delta_price = grouped['Close'].diff()
    tick_sign = np.sign(delta_price).replace(0, method='ffill').fillna(0)
    features['OFI_Proxy'] = tick_sign * df['Volume']
    features['OFI_EMA_15'] = grouped['OFI_Proxy'].transform(lambda x: x.ewm(span=15).mean())
    
    # 2. Amihud Illiquidity Measure (Absolute Return / Volume)
    returns = grouped['Close'].pct_change()
    features['Amihud_Illiq'] = returns.abs() / (df['Volume'] + 1) # +1 to avoid div by zero
    features['Amihud_EMA_30'] = grouped['Amihud_Illiq'].transform(lambda x: x.ewm(span=30).mean())
    
    # 3. Roll's Measure (Microstructure Noise / Bid-Ask Spread proxy)
    def rolling_roll_measure(series, window=30):
        cov = series.rolling(window).cov(series.shift(1))
        # Handle positive covariance (where Roll fails) by setting to 0
        return 2 * np.sqrt(np.where(cov < 0, -cov, 0))
    
    features['Roll_Spread'] = grouped['Close'].transform(lambda x: rolling_roll_measure(x.diff()))
    
    # 4. Micro-Momentum (Short term mean-reversion vs momentum)
    features['Return_5m'] = grouped['Close'].pct_change(5)
    features['Return_15m'] = grouped['Close'].pct_change(15)
    
    # 5. Stale Price Indicator (Binary)
    features['Is_Stale'] = (df['Volume'] == 0).astype(int)
    
    # Target: 15-minute forward return
    features['Target_15m_Fwd'] = grouped['Close'].transform(lambda x: x.pct_change(15).shift(-15))
    
    # Combine back with original data
    return pd.concat([df, features], axis=1).dropna()
