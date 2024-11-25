import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import random
import torch

np.random.seed(98)
random.seed(98)
torch.manual_seed(98)

from holt_winters_env import HoltWintersEnv
from data_loader import load_data
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

# Instantiate and check the environment
env = HoltWintersEnv(train_ts_data, window_size=96, forecast_horizon=48)
check_env(env)

# Create the RL agent
model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=5000)

# Evaluate the agent
state, _ = env.reset()
rewards = []
alphas, betas, gammas = [], [], []
while True:
    action, _ = model.predict(state)
    state, reward, terminated, truncated, _ = env.step(action)
    rewards.append(reward)
    alphas.append(action[0])
    betas.append(action[1])
    gammas.append(action[2])
    if terminated or truncated:
        break

print(f"Total Reward: {sum(rewards)}")

# Reset the environment
last_state_index = len(train_ts_data) - 96
state = train_ts_data[last_state_index:].values.astype(np.float32)

# Get the action (hyperparameters) from the trained agent
action, _ = model.predict(state)
alpha, beta, gamma = action

# Fit the Holt-Winters model on the entire training data using the agent's hyperparameters
model_hw = ExponentialSmoothing(
    train_ts_data,
    trend='add',
    seasonal='add',
    seasonal_periods=24
)

try:
    model_fit = model_hw.fit(smoothing_level=alpha, smoothing_trend=beta,
                             smoothing_seasonal=gamma, optimized=False)
    forecast = model_fit.forecast(len(test_ts_data))
except ValueError as e:
    print(f"Model fitting failed: {e}")
    forecast = pd.Series([np.nan]*len(test_ts_data), index=test_ts_data.index)

# Calculate the RMSE on the test data
metrics = compute_forecast_metrics(forecast, test_ts_data)
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
plt.plot(test_ts_data.index, forecast.values, label='Forecasted Data')
plt.title('Actual vs Forecasted')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()

if data_file_path == "D:/MIE1630/Data/AirQualityUCI.csv":
    plt.savefig("D:/MIE1630/Results/RL/" + "air_quality.png")
elif data_file_path == "D:/MIE1630/Data/electricity.csv":
    plt.savefig("D:/MIE1630/Results/RL/" + "electricity.png")
elif data_file_path == "D:/MIE1630/Data/traffic.csv":
    plt.savefig("D:/MIE1630/Results/RL/" + "traffic.png")
plt.show()
