import numpy as np
import pandas as pd

def backtest_illiquid_strategy(data, initial_capital=1000000, max_participation=0.05, base_bps_cost=10):
    """
    Custom event-driven backtester for illiquid markets. Enforces maximum volume 
    participation caps, models dynamic slippage using Roll's spread, and rejects 
    trades during zero-volume minutes.
    """
    portfolio_value = initial_capital
    cash = initial_capital
    positions = {asset: 0 for asset in data['Asset'].unique()}
    
    history = []
    
    # Iterate minute by minute
    grouped_time = data.groupby(data.index)
    
    for timestamp, current_market in grouped_time:
        longs = current_market[current_market['Rank'] >= 0.8]
        shorts = current_market[current_market['Rank'] <= 0.2]
        
        target_positions = {}
        long_alloc = (portfolio_value * 1.0) / len(longs) if len(longs) > 0 else 0
        short_alloc = -(portfolio_value * 1.0) / len(shorts) if len(shorts) > 0 else 0
        
        for _, row in longs.iterrows():
            target_positions[row['Asset']] = long_alloc / row['Close']
        for _, row in shorts.iterrows():
            target_positions[row['Asset']] = short_alloc / row['Close']
            
        # Execution Engine
        for asset in positions.keys():
            current_shares = positions[asset]
            target_shares = target_positions.get(asset, 0)
            shares_to_trade = target_shares - current_shares
            
            if shares_to_trade != 0:
                asset_data = current_market[current_market['Asset'] == asset]
                if asset_data.empty: continue
                
                asset_data = asset_data.iloc[0]
                available_volume = asset_data['Volume']
                
                # 1. STALE PRICE / ZERO VOLUME CHECK
                if available_volume == 0:
                    continue # Cannot trade, stock is stale
                    
                # 2. MAXIMUM PARTICIPATION CAP
                max_shares_allowed = available_volume * max_participation
                if abs(shares_to_trade) > max_shares_allowed:
                    shares_to_trade = np.sign(shares_to_trade) * max_shares_allowed
                
                # 3. DYNAMIC SLIPPAGE (Bid-Ask Spread + Market Impact)
                spread_cost = asset_data['Roll_Spread'] if not pd.isna(asset_data['Roll_Spread']) else (asset_data['Close'] * 0.005)
                exec_price = asset_data['Close'] + (np.sign(shares_to_trade) * (spread_cost / 2))
                
                # Calculate trade value and fixed costs
                trade_value = abs(shares_to_trade) * exec_price
                commission = trade_value * (base_bps_cost / 10000)
                
                # Execute Trade
                cash -= (shares_to_trade * exec_price) + commission
                positions[asset] += shares_to_trade
                
        # Mark to Market (Evaluate Portfolio at True Close)
        m2m_value = cash
        for asset, shares in positions.items():
            asset_data = current_market[current_market['Asset'] == asset]
            if not asset_data.empty:
                m2m_value += shares * asset_data.iloc[0]['Close']
                
        portfolio_value = m2m_value
        history.append({'Datetime': timestamp, 'Portfolio_Value': portfolio_value})
        
    return pd.DataFrame(history).set_index('Datetime')


def backtest_twap_passive_execution(data, initial_capital=1000000, twap_minutes=5, max_participation=0.02, maker_bps_cost=2):
    """
    Simulates a TWAP execution algorithm using passive limit orders.
    Avoids spread crossing penalties and pays lower maker fees.
    """
    portfolio_value = initial_capital
    cash = initial_capital
    positions = {asset: 0 for asset in data['Asset'].unique()}
    pending_orders = {asset: {'shares_per_min': 0, 'minutes_left': 0} for asset in data['Asset'].unique()}
    
    history = []
    grouped_time = data.groupby(data.index)
    
    for timestamp, current_market in grouped_time:
        # Signal Generation
        if timestamp.minute % twap_minutes == 0:
            longs = current_market[current_market['Rank'] >= 0.8]
            shorts = current_market[current_market['Rank'] <= 0.2]
            
            target_positions = {}
            long_alloc = (portfolio_value * 1.0) / len(longs) if len(longs) > 0 else 0
            short_alloc = -(portfolio_value * 1.0) / len(shorts) if len(shorts) > 0 else 0
            
            for _, row in longs.iterrows(): target_positions[row['Asset']] = long_alloc / row['Close']
            for _, row in shorts.iterrows(): target_positions[row['Asset']] = short_alloc / row['Close']
                
            # Create the TWAP slices
            for asset in positions.keys():
                target = target_positions.get(asset, 0)
                diff = target - positions[asset]
                if abs(diff) > 0:
                    pending_orders[asset] = {
                        'shares_per_min': diff / twap_minutes, 
                        'minutes_left': twap_minutes
                    }
                    
        # Execution Engine
        for asset in positions.keys():
            order = pending_orders[asset]
            if order['minutes_left'] > 0:
                asset_data = current_market[current_market['Asset'] == asset]
                if not asset_data.empty:
                    asset_data = asset_data.iloc[0]
                    available_volume = asset_data['Volume']
                    
                    if available_volume > 0:
                        shares_to_trade = order['shares_per_min']
                        max_shares_allowed = available_volume * max_participation
                        
                        if abs(shares_to_trade) > max_shares_allowed:
                            shares_to_trade = np.sign(shares_to_trade) * max_shares_allowed
                        
                        # PASSIVE EXECUTION: No spread penalty
                        exec_price = asset_data['Close'] 
                        trade_value = abs(shares_to_trade) * exec_price
                        commission = trade_value * (maker_bps_cost / 10000)
                        
                        cash -= (shares_to_trade * exec_price) + commission
                        positions[asset] += shares_to_trade
                
                order['minutes_left'] -= 1
                
        # Mark to Market
        m2m_value = cash
        for asset, shares in positions.items():
            asset_data = current_market[current_market['Asset'] == asset]
            if not asset_data.empty:
                m2m_value += shares * asset_data.iloc[0]['Close']
                
        portfolio_value = m2m_value
        history.append({'Datetime': timestamp, 'Portfolio_Value': portfolio_value})
        
    return pd.DataFrame(history).set_index('Datetime')


def backtest_hourly_strategy(data, initial_capital=1000000, base_bps_cost=10):
    """
    Backtester designed for lower-frequency (e.g., hourly) data. Applies a 
    fixed, conservative spread penalty to model transaction costs on longer timeframes.
    """
    portfolio_value = initial_capital
    cash = initial_capital
    positions = {asset: 0 for asset in data['Asset'].unique()}
    history = []
    
    grouped_time = data.groupby(data.index)
    
    for timestamp, current_market in grouped_time:
        longs = current_market[current_market['Rank'] >= 0.7]
        shorts = current_market[current_market['Rank'] <= 0.3]
        
        target_positions = {}
        long_alloc = (portfolio_value * 1.0) / len(longs) if len(longs) > 0 else 0
        short_alloc = -(portfolio_value * 1.0) / len(shorts) if len(shorts) > 0 else 0
        
        for _, row in longs.iterrows(): target_positions[row['Asset']] = long_alloc / row['Close']
        for _, row in shorts.iterrows(): target_positions[row['Asset']] = short_alloc / row['Close']
            
        for asset in positions.keys():
            target_shares = target_positions.get(asset, 0)
            shares_to_trade = target_shares - positions[asset]
            
            if shares_to_trade != 0:
                asset_data = current_market[current_market['Asset'] == asset]
                if not asset_data.empty:
                    asset_data = asset_data.iloc[0]
                    
                    spread_cost = asset_data['Spread_Cost']
                    exec_price = asset_data['Close'] + (np.sign(shares_to_trade) * (spread_cost / 2))
                    
                    trade_value = abs(shares_to_trade) * exec_price
                    commission = trade_value * (base_bps_cost / 10000)
                    
                    cash -= (shares_to_trade * exec_price) + commission
                    positions[asset] += shares_to_trade
                    
        # Mark to Market
        m2m_value = cash
        for asset, shares in positions.items():
            asset_data = current_market[current_market['Asset'] == asset]
            if not asset_data.empty:
                m2m_value += shares * asset_data.iloc[0]['Close']
                
        portfolio_value = m2m_value
        history.append({'Datetime': timestamp, 'Portfolio_Value': portfolio_value})
        
    return pd.DataFrame(history).set_index('Datetime')
