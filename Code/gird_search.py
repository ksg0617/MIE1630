from data_loader import load_data

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import itertools
import matplotlib.pyplot as plt

from scipy.optimize import minimize
import time

def compute_forecast_metrics(forecast, test_ts_data):
    """
    Compute RMSE, MAE, and MAPE for forecast and test time series data.

    Parameters:
    - forecast (array): The forecasted values.
    - test_ts_data (array): The actual observed values.

    Returns:
    - dict: A dictionary containing RMSE, MAE, and MAPE.
    """
    
    if len(forecast) != len(test_ts_data):
        raise ValueError("Forecast and test time series data must have the same length.")
    
    # Compute errors
    errors = forecast - test_ts_data
    
    # RMSE
    rmse = np.sqrt(np.mean(errors**2))
    
    # MAE
    mae = np.mean(np.abs(errors))
    
    # MAPE (avoid division by zero by replacing zero values in the test data with a small epsilon)
    epsilon = np.finfo(float).eps  # Smallest positive float
    test_ts_data_safe = np.where(test_ts_data == 0, epsilon, test_ts_data)
    mape = np.mean(np.abs(errors / test_ts_data_safe)) * 100
    
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

# Record start time
start_time = time.time()

#data_file_path = "D:/MIE1630/Data/AirQualityUCI.csv"
#data_file_path = "D:/MIE1630/Data/electricity.csv"
data_file_path = "D:/MIE1630/Data/traffic.csv"

train_ts_data, test_ts_data = load_data(data_file_path)

forecast_horizon = 48

train = train_ts_data[:-forecast_horizon]
val =train_ts_data[-forecast_horizon:]

# Define more granular ranges for the parameters
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 10 values from 0.0 to 1.0
betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
param_grid = list(itertools.product(alphas, betas, gammas))

best_score = float('inf')
best_params = {'alpha': None, 'beta': None, 'gamma': None}

# Perform grid search
for alpha, beta, gamma in param_grid:
    print(alpha, beta, gamma)
    try:
        model = ExponentialSmoothing(
            train,
            seasonal_periods=24,
            trend='add',
            seasonal='add'
        )
        fit = model.fit(
            smoothing_level=alpha,
            smoothing_trend=beta,
            smoothing_seasonal=gamma,
            optimized=False
        )
        forecast = fit.forecast(len(val))
        error = mean_squared_error(val, forecast)
        if error < best_score:
            best_score = error
            best_params['alpha'] = alpha
            best_params['beta'] = beta
            best_params['gamma'] = gamma
    except Exception as e:
        continue

print('Best Parameters:')
print('Alpha:', best_params['alpha'])
print('Beta:', best_params['beta'])
print('Gamma:', best_params['gamma'])
print('Best MSE:', best_score)

train_ts_data, test_ts_data = load_data(data_file_path)

# Evaluate the model with the best hyperparameters
model_hw_best = ExponentialSmoothing(
    train_ts_data,
    trend='add',
    seasonal='add',
    seasonal_periods=24
)
model_fit_best = model_hw_best.fit(
    smoothing_level=best_params['alpha'],
    smoothing_trend=best_params['beta'],
    smoothing_seasonal= best_params['gamma'],
    optimized=False
)
forecast_best = model_fit_best.forecast(len(test_ts_data))


# Calculate the RMSE on the test data
metrics = compute_forecast_metrics(forecast_best, test_ts_data)
print(metrics)

# Record end time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")

# Specify the number of last data points to display
n_last_points = 50  # Adjust this number based on how many data points you want to see

# Slice the last n data points from the training data
train_last = train_ts_data[-n_last_points:]

# Plot the forecasts vs actuals
plt.figure(figsize=(12, 6))
plt.plot(train_last.index, train_last.values, label='Training Data')
plt.plot(test_ts_data.index, test_ts_data.values, label='Actual Test Data')
plt.plot(test_ts_data.index, forecast_best.values, label='Forecasted Data')
plt.title('Actual vs Forecasted')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()

if data_file_path == "D:/MIE1630/Data/AirQualityUCI.csv":
    plt.savefig("D:/MIE1630/Results/Grid_Search/" + "air_quality.png")
elif data_file_path == "D:/MIE1630/Data/electricity.csv":
    plt.savefig("D:/MIE1630/Results/Grid_Search/" + "electricity.png")
elif data_file_path == "D:/MIE1630/Data/traffic.csv":
    plt.savefig("D:/MIE1630/Results/Grid_Search/" + "traffic.png")

plt.show()