import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def adf_of_pair(coin_x, coin_y, sample_px):
  sample_log_px = np.log(sample_px)
  
  X = sample_log_px[coin_x].fillna(0).values
  Y = sample_log_px[coin_y].fillna(0).values
  
  model = sm.OLS(Y, sm.add_constant(X)).fit()
  alpha = model.params[0]
  beta = model.params[1]
  residuals = Y - beta * X - alpha

  adf_result = adfuller(residuals)
  p_value = adf_result[1]
  test_stat = adf_result[0]

  return (coin_x, coin_y), (p_value, test_stat)

def best_pairs(sample_px, p_threshold = 0.05):
  coins = sample_px.columns.tolist()
  best_pairs = {}

  for coin_x, coin_y in combinations(coins, 2):
    pair, (p_value, test_stat) = adf_of_pair(coin_x, coin_y, sample_x)
    for base, partner in [(coin_x, coin_y), (coin_y, coin_x)]:
      if p_value < p_threshold:
        if base not in best_pairs or test_stat < best_pairs[base][1]:
          best_pairs[base] = (partner, test_stat, p_value)

  final_pairs = [(base, info[0]) for base, info in best_pairs.items()]
  return final_pairs
