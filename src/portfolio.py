import numpy as np
import pandas as pd

# Generates our portfolio by converting z-score signals into weights. Can be tested with different threshold values.
def gen_portfolio(signal_df, pairs, in_sample_px, threshold = 0.2):
  pos = pd.DataFrame(
