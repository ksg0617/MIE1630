# Import necessary libraries
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import random
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
import random
from data_loader import load_data
import time

# Record start time
start_time = time.time()

random.seed(98)

# data_file_path = "D:/MIE1630/Data/AirQualityUCI.csv"
# data_file_path = "D:/MIE1630/Data/electricity.csv"
data_file_path = "D:/MIE1630/Data/traffic.csv"

train_ts_data, test_ts_data = load_data(data_file_path)

forecast_horizon = 48

train = train_ts_data[:-forecast_horizon]
val =train_ts_data[-forecast_horizon:]

# Random Search Hyperparameter Optimization
n_iterations = 100
best_score = float('inf')
best_params = {'alpha': None, 'beta': None, 'gamma': None}
error_scores = []

print("Starting Random Search...\n")
for i in range(n_iterations):
    # Randomly sample hyperparameters from uniform distribution between 0 and 1
    alpha = random.uniform(0, 1)
    beta = random.uniform(0, 1)
    gamma = random.uniform(0, 1)
    
    try:
        # Fit the model on the training data
        model = ExponentialSmoothing(
            train,
            trend='add',
            seasonal='add',
            seasonal_periods=24
        )
        model_fit = model.fit(
            smoothing_level=alpha,
            smoothing_trend=beta,
            smoothing_seasonal=gamma,
            optimized=False  # Disable internal optimization
        )
        
        # Make predictions on the test set
        predictions = model_fit.forecast(steps=len(val))
        
        # Calculate the error metric (Mean Squared Error)
        mse = mean_squared_error(val, predictions)
        error_scores.append(mse)
        
        # Check if this is the best score so far
        if mse < best_score:
            best_score = mse
            best_params['alpha'] = alpha
            best_params['beta'] = beta
            best_params['gamma'] = gamma
            
        print(f"Iteration {i+1}/{n_iterations}: MSE = {mse:.4f}, "
              f"alpha = {alpha:.4f}, beta = {beta:.4f}, gamma = {gamma:.4f}")
    except Exception as e:
        # Handle exceptions due to invalid configurations
        print(f"Iteration {i+1}/{n_iterations}: Error - {e}")

# Print the best hyperparameters
print("\nBest Hyperparameters Found:")
print(f"Alpha: {best_params['alpha']:.4f}")
print(f"Beta: {best_params['beta']:.4f}")
print(f"Gamma: {best_params['gamma']:.4f}")
print(f"Best MSE: {best_score:.4f}")

# Refit the model with the best hyperparameters on the full training data
best_model = ExponentialSmoothing(
    train_ts_data,
    trend='add',
    seasonal='add',
    seasonal_periods=24
)
best_model_fit = best_model.fit(
    smoothing_level=best_params['alpha'],
    smoothing_trend=best_params['beta'],
    smoothing_seasonal=best_params['gamma'],
    optimized=False
)

# Forecast on the test set
best_predictions = best_model_fit.forecast(steps=len(test_ts_data))
mse_best = np.mean((best_predictions - test_ts_data) ** 2)
print(f"Test MSE with Random Search: {mse_best}")

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
plt.plot(test_ts_data.index, best_predictions.values, label='Forecasted Data')
plt.title('Actual vs Forecasted')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()

if data_file_path == "D:/MIE1630/Data/AirQualityUCI.csv":
    plt.savefig("D:/MIE1630/Results/Random_Search/" + "air_quality.png")
elif data_file_path == "D:/MIE1630/Data/electricity.csv":
    plt.savefig("D:/MIE1630/Results/Random_Search/" + "electricity.png")
elif data_file_path == "D:/MIE1630/Data/traffic.csv":
    plt.savefig("D:/MIE1630/Results/Random_Search/" + "traffic.png")
plt.show()