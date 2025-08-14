#%% 
"""
This code uses DMD to forecast the near future in the crypto market, for high volume
and high market cap coins. It uses 8 symbols, and for building the A operator, it
first preprocesses the historical data, converting the data to a standard normal
distribution via the log returns and z-score normalization to get the underlying
dynamics of the forecast as accurate as possible.
After the forecast, data is converted back to prices to check for the actual
performance.
"""

import pandas as pd
import numpy as np
import shelve
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
