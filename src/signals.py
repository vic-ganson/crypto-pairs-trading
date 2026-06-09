import numpy as np
import pandas as pd

# Estimates dynamic beta and alpha between two log price series using a Kalman filter. Delta and R are conventional right now.
def kalman_filter(px_x, px_y, delta=1e-4, R=1e-2):
  # Fix and forward-fill potential missing data before taking logs
  log_x = np.log(px_x.ffill().replace(0, np.nan))
  log_y = np.log(px_y.ffill().replace(0, np.nan))
  
  n = len(log_x)
  theta = np.zeros((n,2)) # theta[:, 0] = beta, theta[:,1] = alpha
  P = np.zeros((n,2,2)) # stores uncertainty
  P[0] = np.eye(2)
  Q = delta * np.eye(2) # process noise
  spreads = np.full(n, np.nan)

  for t in range(1, n):
    x_t = log_x.iloc[t]
    y_t = log_y.iloc[t]

    if np.isnan(x_t) or np.isnan(y_t):
        theta[t] = theta[t - 1]
        P[t] = P[t - 1]
        continue
      
    F = np.array([x_t, 1.0])
    
    theta_pred = theta[t - 1]
    P_pred = P[t - 1] + Q
    
    y_pred = F @ theta_pred
    spread = y_t - y_pred
    S = F @ P_pred @ F.T + R # variance of spread
    
    K = P_pred @ F / S # Kalman gain, how much we trust the new observation compared to prior estimate
    
    theta[t] = theta_pred + K * spread
    P[t] = P_pred - np.outer(K, F) @ P_pred
    spreads[t] = spread

  beta_series = pd.Series(theta[:, 0], index=log_x.index)
  alpha_series = pd.Series(theta[:, 1], index=log_x.index)
  spread_series = pd.Series(spreads, index=log_x.index)
  return beta_series, alpha_series, spread_series

# Computes rolling z-score of the Kalman spread
def compute_zscore(spread, window=90):
  mean = spread.rolling(window=window, min_periods=1).mean()
  std = spread.rolling(window=window, min_periods=1).std()
  return (spread - mean) / std

# Stores pairs with associated beta, alpha, spread, and z_score in DataFrame
def gen_signals(px, pairs, delta=1e-4, R=1e-2, zscore_window=90):
    signal_df = {}

    for pair in pairs:
        coin_x, coin_y = pair
        beta, alpha, spread = kalman_filter(px[coin_x], px[coin_y], delta=delta, R=R)
        z_score = compute_zscore(spread, window=zscore_window)

        signal_df[(pair, 'beta')] = beta
        signal_df[(pair, 'alpha')] = alpha
        signal_df[(pair, 'spread')] = spread
        signal_df[(pair, 'z_score')] = z_score

    return pd.DataFrame(signal_df)
