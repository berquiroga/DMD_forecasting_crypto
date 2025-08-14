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

#%%
# Preprocessing functions
# Data is already stored in a shelve. Function to obtain 1000 closing price data
def get_timeframe_data(timeframe):
    database = shelve.open("shelve_historical_data")
    timeframe_symbols = database[f"{timeframe}_symbols"]
    timeframe_data = database[timeframe]
    return timeframe_data, timeframe_symbols


# Retrieve only data for the symbols and timeframe wanted from the shelve
def get_symbols_data(symbols, timeframe):
    timeframe_historical_data, timeframe_symbols = get_timeframe_data(timeframe)
    for i in range(len(symbols)):
        index = timeframe_symbols.index(symbols[i])
        if i == 0:
            symbols_historical_data = timeframe_historical_data[index,:]
        else:
            symbols_historical_data = np.vstack((symbols_historical_data, timeframe_historical_data[index,:]))
    return symbols_historical_data


# Compute log returns for all columns on the numpy array
def compute_log_returns(timeframe_data):
    rows = timeframe_data.shape[0]
    columns = timeframe_data.shape[1]
    log_returns = np.zeros((rows, columns - 1))
    for i in range(rows):
        for j in range(columns - 1):
            log_returns[i,j] = np.log(timeframe_data[i, j+1] / timeframe_data[i,j])
    return log_returns


# Compute the z-score for the log returns to normalize
def get_z_scores(log_returns):
    rows_log_returns = log_returns.shape[0]
    columns_log_returns = log_returns.shape[1]
    means = log_returns.mean(axis=1)
    stds = log_returns.std(axis=1)
    z_scores = np.zeros((rows_log_returns, columns_log_returns))
    for i in range(rows_log_returns):
        for j in range(columns_log_returns):
            z_scores[i,j] = (log_returns[i,j] - means[i]) / stds[i]
    return z_scores, means, stds

#%%
# Build Snapshots required to forecast
# Build X and X' snapshot matrices from normalized data
def build_snapshot_matrices(z_score_normalized_data):
    X = z_score_normalized_data[:, :-1]
    X_prime = z_score_normalized_data[:, 1:]
    return X, X_prime

#%%
# DMD functions
# Function to perform a full rank DMD if low rank matrix
def dmd_full_rank(X, X_prime):
    A = X_prime @ np.linalg.pinv(X)
    eigvals, eigvecs = np.linalg.eig(A)
    return A, eigvals, eigvecs
    

# Function to perform a truncated DMD with SVD if high rank matrix
def dmd_low_rank(X, X_prime, r):
    U, S, Vh = np.linalg.svd(X)
    U_r = U[:, 0:r]
    S_r = np.diag(S[0:r])
    V = Vh.T.conj()
    V_r = V[:, 0:r]
    # Build A tilde and DMD modes
    A_tilde = U_r.T @ X_prime @ V_r @ np.linalg.inv(S_r)
    eigvals, W = np.linalg.eig(A_tilde) # Eigenvalues and eigenvectors
    eigvecs = X_prime @ V_r @ np.linalg.inv(S_r) @ W # DMD modes
    return A_tilde, eigvals, eigvecs
