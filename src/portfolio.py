import numpy as np
import pandas as pd

# Generates our portfolio by converting z-score signals into weights. Can be tested with different threshold values.
def gen_portfolio(signal_df, pairs, in_sample_px, threshold = 0.2):
  pos = pd.DataFrame(0.0, index=signal_df.index, columns=in_sample_px.columns)

  for pair in pairs:
    coin_x, coin_y = pair
    z_scores = signal_df[(pair, 'z_score')]
    betas = signal_df[(pair, 'beta')]
    # Entry signals
    short_x = z_scores > 1
    long_x = z_scores < -1
    
    pos.loc[short_x, coin_x] += -1 # Decrement position on days where short_x is True
    pos.loc[long_x, coin_x] += 1 # Increment position on days where long_x is True
    pos.loc[short_x, coin_y] += betas[short_x] # Coin Y gets the opposite trade, scaled by beta
    pos.loc[long_x, coin_y] += -betas[long_x]
    # Exit signals
    exit_signal = z_scores.abs() <= threshold
    pos.loc[exit_signal, coin_x] = 0
    pos.loc[exit_signal, coin_y] = 0

  pos = pos.replace(0, np.nan).ffill().fillna(0)

  row_sums = pos.abs().sum(axis=1)
  pos = pos.divide(row_sums.where(row_sums != 0, other=1), axis=0) # Divides weights by total positions so weights sum to 1

  return pos

# Absolute sum of weight changes by day, helps compute transaction costs
def compute_turnover(pos):
  daily_changes = pos.fillna(0).diff().abs()
  return daily_changes.sum(axis=1)

def compute_net_returns(pos, prices, tcost_bps):
  coin_returns = prices.pct_change()
  # Safety check that coins are in both pos and prices
  shared_coins = pos.columns.intersection(coin_returns.columns)
  pos = pos[shared_coins]
  coin_returns = coin_returns[shared_coins]
  
  gross_returns = (pos.shift(1) * coin_returns).sum(axis=1)
  turnover = computer_turnover(pos)
  tcost = turnover * tcost_bps * 1e-4
  net_returns = gross_returns - tcost
  return net_returns
