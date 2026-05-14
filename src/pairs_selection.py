
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def adf_of_pair(coin_x, coin_y, sample_px):
  sample_log_px = np.log(sample_px)
  
  X = sample_log_px[coin_x].fillna(0).values
  Y = _sample_log_px[coin_y].fillna(0).values
  
  model = sm.OLS(Y, sm.add_constant(X)).fit()
  alpha = model.params[0]
  beta = model.params[1]
  residuals = Y - beta * X - alpha

  adf_result = adfuller(residuals)
  p_value = adf_result[1]
  test_stat = adf_result[0]

  return (coin_x, coin_y), (p_value, test_stat)
