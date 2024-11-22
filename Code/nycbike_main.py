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

from nycbike import HoltWintersEnv
from data_loader import load_nycbike_data

train_ts_data, test_ts_data = load_nycbike_data()
train_ts_data += 1 - train_ts_data.min()  # Shift data to make it strictly positive
test_ts_data += 1 - test_ts_data.min() 

# Instantiate and check the environment
env = HoltWintersEnv(train_ts_data, window_size=48, forecast_horizon=12)
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

# Plot the hyperparameters over time
plt.figure(figsize=(12, 4))
plt.plot(alphas, label='Alpha')
plt.plot(betas, label='Beta')
plt.plot(gammas, label='Gamma')
plt.title('Hyperparameters Over Time')
plt.legend()
plt.show()

# Reset the environment
last_state_index = len(train_ts_data) - 48
state = train_ts_data[last_state_index:len(train_ts_data)].values.astype(np.float32)

# Get the action (hyperparameters) from the trained agent
action, _ = model.predict(state)
alpha, beta, gamma = action

# Fit the Holt-Winters model on the entire training data using the agent's hyperparameters
model_hw = ExponentialSmoothing(
    train_ts_data,
    trend='add',
    seasonal='add',
    seasonal_periods=12  
)

try:
    model_fit = model_hw.fit(smoothing_level=alpha, smoothing_trend=beta,
                             smoothing_seasonal=gamma, optimized=False)
    forecast = model_fit.forecast(len(test_ts_data))
except ValueError as e:
    print(f"Model fitting failed: {e}")
    forecast = pd.Series([np.nan]*len(test_ts_data), index=test_ts_data.index)

# Calculate the MSE on the test data
mse = np.mean((forecast - test_ts_data) ** 2)
print(f"Test MSE: {mse}")

# Plot the forecasts vs actuals
plt.figure(figsize=(12, 6))
plt.plot(train_ts_data.index, train_ts_data.values, label='Training Data')
plt.plot(test_ts_data.index, test_ts_data.values, label='Actual Test Data')
plt.plot(test_ts_data.index, forecast.values, label='Forecasted Data')
plt.title('Actual vs Forecasted Passengers (Test Data)')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()