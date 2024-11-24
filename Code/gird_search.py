from data_loader import load_data

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import itertools
import matplotlib.pyplot as plt

from scipy.optimize import minimize

# data_file_path = "D:/MIE1630/Data/AirQualityUCI.csv"
# data_file_path = "D:/MIE1630/Data/electricity.csv"
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


mse_best = np.mean((forecast_best - test_ts_data) ** 2)
print(f"Test MSE with Grid Search: {mse_best}")

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