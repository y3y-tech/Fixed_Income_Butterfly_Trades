import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

gsw = pd.read_excel('data/gsw_yields.xlsx', index_col=0, parse_dates=True)

maturities = [2,5,10]
yields = gsw[maturities].dropna()
yields.columns = ['2Y','5Y','10Y']

yields = yields.loc['2015':]

yields_diff = yields.diff(periods=1).dropna()

scaler = StandardScaler()
yields_diff_scaled = scaler.fit_transform(yields_diff)

pca = PCA(n_components=3)
pca.fit(yields_diff_scaled)
explained_variance = pca.explained_variance_ratio_
components = pca.components_

weights = np.array(-components[2] / components[2][1])

# Calculate the butterfly spread
butterfly_spread = weights[0] * yields_diff['2Y'] + weights[1] * yields_diff['5Y'] + weights[2] * yields_diff['10Y']

def mean_reversion(look_back_window, entry_threshold, exit_threshold, spread_data):
    """
    Generate mean reversion trading signals based on z-scores
    
    Parameters:
    - look_back_window: rolling window for mean/std calculation
    - entry_threshold: z-score threshold to enter position
    - exit_threshold: z-score threshold to exit position
    - spread_data: time series of the butterfly spread
    """
    # Calculate rolling mean and std
    rolling_mean = spread_data.rolling(window=look_back_window).mean()
    rolling_std = spread_data.rolling(window=look_back_window).std()
    
    # Calculate z-score
    z_scores = (spread_data - rolling_mean) / rolling_std
    
    # Generate trading signals
    signals = pd.DataFrame(index=spread_data.index)
    signals['Position'] = 0  # 1 for long, -1 for short, 0 for neutral
    signals['Z_Score'] = z_scores
    
    # Entry signals
    signals.loc[z_scores > entry_threshold, 'Position'] = -1  # Short when spread is high
    signals.loc[z_scores < -entry_threshold, 'Position'] = 1   # Long when spread is low
    
    # Exit signals (overwrite entry signals when within exit threshold)
    signals.loc[abs(z_scores) < exit_threshold, 'Position'] = 999
    
    # Forward fill positions (hold position until exit signal)
    signals['Position'] = signals['Position'].replace(0, np.nan).ffill().fillna(0)

    # 4. Convert the placeholder back to a real neutral position
    signals['Position'] = signals['Position'].replace(999, 0)
    
    return signals, z_scores

def backtest_strategy(signals, spread_data):
    """
    Backtest the mean reversion strategy
    
    Parameters:
    - signals: DataFrame with Position column
    - spread_data: time series of the butterfly spread
    """
    # Calculate daily spread changes (this is your P&L driver)
    daily_changes = spread_data.diff()
    
    # Calculate strategy returns (position from previous day * today's change)
    # Use shift(1) because you enter position based on yesterday's signal
    strategy_returns = signals['Position'].shift(1) * daily_changes
    
    # Calculate cumulative returns
    cumulative_returns = strategy_returns.cumsum()
    
    return strategy_returns, cumulative_returns


look_back_window = 20
entry_threshold = 1.5  # Enter when spread is 1.5 std devs away
exit_threshold = 0.5   # Exit when spread is within 0.5 std devs
signals, z_scores = mean_reversion(look_back_window, entry_threshold, exit_threshold, butterfly_spread)
strategy_returns, cumulative_returns = backtest_strategy(signals, butterfly_spread)