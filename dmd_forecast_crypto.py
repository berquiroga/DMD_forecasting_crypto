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

#%%
# Forecasting
# Function to forecast a determined amount of steps using the A operator
def forecast(A, z_score_data, future_steps):
    x = z_score_data[:, -1]
    forecasts = np.zeros((A.shape[0], future_steps))
    for i in range(future_steps):
        x_i = A @ x
        forecasts[:,i] = x_i
        x = x_i
    return forecasts


# Function to obtain the success rate of a sampling forecasting window with fixed input and future steps through a training set
def get_successful_forecasts_over_window(training_set_z_score, training_set, means, stds, input_steps, future_steps):
    forecasts = training_set_z_score.shape[1] - input_steps - future_steps + 1
    successful_forecasts = []
    # If input bars equals 1, it is not possible to build DMD so no profits
    if input_steps == 1:
        successful_forecasts = np.zeros((training_set_z_score.shape[0], forecasts))
        return successful_forecasts
    for i in range(forecasts):
        # Slice data to use only the input steps
        forecast_data_z_score = training_set_z_score[:, i:input_steps + i]
        # Compute A operator from data
        X, X_prime = build_snapshot_matrices(forecast_data_z_score)
        A, eigenvals, eigenvecs = dmd_full_rank(X, X_prime)
        # Forecast the future steps of data
        forecasted_results_z_score = forecast(A, forecast_data_z_score, future_steps)
        # Determine which forecasts were correct
        current_step = training_set[:, input_steps + i - 1]
        future_step = training_set[:, input_steps + future_steps + i - 1]
        forecasted_results = reverse_z_scores(forecasted_results_z_score, current_step, means, stds)
        forecasted_step = forecasted_results[:, -1]
        successful_forecasts.append(determine_successful_forecasts_direction(current_step, future_step, forecasted_step))
    return np.array(successful_forecasts).T
        

# Function to determine which forecasts were successful per step
def determine_successful_forecasts_direction(initial_price, actual_price_movement, forecasted_price_movement):
    successful_forecasts = []
    actual_price_difference = actual_price_movement - initial_price
    forecasted_price_difference = forecasted_price_movement - initial_price
    for initial, actual, forecasted in zip(initial_price, actual_price_difference, forecasted_price_difference):
        if forecasted > 0 and actual / initial > 0.01:
            successful_forecasts.append(1)
        elif forecasted < 0 and actual / initial < 0.01:
            successful_forecasts.append(1)
        else:
            successful_forecasts.append(0)
    return successful_forecasts
    
    
# Compute the success rate for a set
def compute_sliding_window_success_rate(successful_forecasts, input_steps, future_steps):
    # Generate dictionary where each symbol present is a key and it retrieves the success rate for the input/future combination
    sample_success_rate = {}
    global symbols
    sample_success_rate["input"] = input_steps
    sample_success_rate["future"] = future_steps
    forecasts = successful_forecasts.shape[1]
    for j in range(successful_forecasts.shape[0]):
        success = 0
        for forecast in successful_forecasts[j, :]:
            if forecast != 0:
                success += 1
        sample_success_rate[symbols[j]] = success / forecasts
    return sample_success_rate


# Compute the success rate of all possible combinations of input steps and future steps for all symbols
def compute_all_combinations_success_rates(training_set_z_score, training_set, means, stds, input_steps, future_steps):
    combinations_success_rates = []
    for i in range(input_steps):
        for j in range(future_steps):
            sliding_window_successful_forecasts = get_successful_forecasts_over_window(training_set_z_score, training_set, means, stds, i+1, j+1)
            sliding_window_success_rate = compute_sliding_window_success_rate(sliding_window_successful_forecasts, i+1, j+1)
            combinations_success_rates.append(sliding_window_success_rate)
    return combinations_success_rates


# Process all combinations success rates and average the symbols perfomance. Return a matrix
def compute_average_success_rate_matrix(combinations_success_rates):
    global symbols
    combinations = []
    for success_rate in combinations_success_rates:
        average = 0
        for symbol in symbols:
            average += success_rate[symbol]
        average = average / len(symbols)
        average_success_rate = {
            "input": success_rate["input"],
            "future": success_rate["future"],
            "average": average,
        }
        combinations.append(average_success_rate)
    rows = combinations[-1]["input"]
    columns = combinations[-1]["future"]
    combinations_average_matrix = np.empty((rows, columns))
    for combination in combinations:
        i = combination["input"] - 1
        j = combination["future"] - 1
        combinations_average_matrix[i, j] = combination["average"]
    higher_average_success_rate = {
            "input": None,
            "future": None,
            "average": 0,        
    }
    for i in range(rows):
        for j in range(columns):
            if combinations_average_matrix[i, j] > higher_average_success_rate["average"]:
                higher_average_success_rate["average"] = combinations_average_matrix[i, j]
                higher_average_success_rate["input"] = i + 1
                higher_average_success_rate["future"] = j + 1
    return combinations_average_matrix, higher_average_success_rate

#%%
# Post-processing functions
# Function to convert forecasted prices back to price action
def reverse_z_scores(z_score_forecasts_data, last_price, means, stds):
    rows = z_score_forecasts_data.shape[0]
    columns = z_score_forecasts_data.shape[1]
    price_forecasts = np.zeros_like(z_score_forecasts_data)
    unnormalized_log_returns = np.zeros_like(z_score_forecasts_data)
    for i in range(rows):
        for j in range(columns):
            unnormalized_log_returns[i,j] = (z_score_forecasts_data[i,j] * stds[i]) + means[i]
            if j == 0:
                price_forecasts[i,j] = last_price[i] * np.exp(unnormalized_log_returns[i,j])
            else:
                price_forecasts[i,j] = price_forecasts[i, j-1] * np.exp(unnormalized_log_returns[i,j])
    return price_forecasts
