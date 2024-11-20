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
import random
import torch

np.random.seed(99)
random.seed(99)
torch.manual_seed(99)

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the dataset directly from a CSV file
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
data = pd.read_csv(url)

# Parse the 'Month' column to datetime
data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m')

# Set 'Month' as the index
data.set_index('Month', inplace=True)

# Extract the time series data
ts_data = data['Passengers']

ts_data.index = pd.to_datetime(ts_data.index)
ts_data = ts_data.asfreq('MS') # frequency here is monthly start

# Split the data into training and testing
train_ts_data = ts_data.iloc[:120]  # First 120 data points
test_ts_data = ts_data.iloc[120:]   # Last 24 data points

class HoltWintersEnv(gym.Env):
    def __init__(self, ts_data, window_size=24, forecast_horizon=1):
        super(HoltWintersEnv, self).__init__()
        self.ts_data = ts_data
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.max_steps = len(ts_data) - window_size - forecast_horizon
        self.current_step = 0

        # Action space: Continuous values for alpha, beta, gamma
        self.action_space = spaces.Box(low=0.01, high=1.0, shape=(3,), dtype=np.float32)

        # Number of additional features: trend, mean, variance, month_sin, month_cos
        self.num_additional_features = 5

        # Observation space: Recent time series data + additional features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size + self.num_additional_features,),
            dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Check if a custom starting point is provided
        if options and 'current_step' in options:
            self.current_step = options['current_step']
        else:
            self.current_step = 0  # Default behavior

        state = self.ts_data[self.current_step:self.current_step + self.window_size].values

        # Compute additional features
        # Trend (slope of linear regression)
        y = state
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        trend = slope

        # Mean and variance over the window
        mean = np.mean(y)
        variance = np.var(y)

        # Seasonality: compute month_sin and month_cos
        current_date = self.ts_data.index[self.current_step + self.window_size - 1]
        month = current_date.month
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        additional_features = np.array([trend, mean, variance, month_sin, month_cos])

        # Concatenate state and additional features
        observation = np.concatenate([state, additional_features])

        return observation.astype(np.float32), {}

    def step(self, action):
        alpha, beta, gamma = action

        # Get the data for model fitting
        start = self.current_step
        end = self.current_step + self.window_size
        train_data = self.ts_data[start:end]

        # Fit Holt-Winters model
        model = ExponentialSmoothing(
            train_data,
            trend='mul',
            seasonal='mul',
            seasonal_periods=12
        )
        try:
            model_fit = model.fit(smoothing_level=alpha, smoothing_trend=beta,
                                  smoothing_seasonal=gamma, optimized=False)
        except ValueError as e:
            # Penalize invalid parameter combinations
            reward = -100.0
            terminated = False
            truncated = False
            if self.current_step >= self.max_steps:
                terminated = True
                next_state = np.zeros(self.window_size + self.num_additional_features)
            else:
                next_state = self.ts_data[self.current_step:self.current_step + self.window_size].values

                # Compute additional features
                y = next_state
                x = np.arange(len(y))
                slope, intercept = np.polyfit(x, y, 1)
                trend = slope

                mean = np.mean(y)
                variance = np.var(y)

                current_date = self.ts_data.index[self.current_step + self.window_size - 1]
                month = current_date.month
                month_sin = np.sin(2 * np.pi * month / 12)
                month_cos = np.cos(2 * np.pi * month / 12)

                additional_features = np.array([trend, mean, variance, month_sin, month_cos])
                next_state = np.concatenate([next_state, additional_features])

            return next_state.astype(np.float32), reward, terminated, truncated, {}

        # Forecast
        forecast = model_fit.forecast(self.forecast_horizon)
        actual = self.ts_data[end:end + self.forecast_horizon]

        # Calculate reward
        mse = np.mean((forecast - actual) ** 2)
        reward = -mse / np.mean(self.ts_data ** 2)  # Normalized MSE

        # Prepare for next step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        if not terminated:
            next_state = self.ts_data[self.current_step:self.current_step + self.window_size].values

            # Compute additional features
            y = next_state
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            trend = slope

            mean = np.mean(y)
            variance = np.var(y)

            current_date = self.ts_data.index[self.current_step + self.window_size - 1]
            month = current_date.month
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            additional_features = np.array([trend, mean, variance, month_sin, month_cos])
            next_state = np.concatenate([next_state, additional_features])

        else:
            next_state = np.zeros(self.window_size + self.num_additional_features)

        return next_state.astype(np.float32), reward, terminated, truncated, {}

    def render(self):
        pass  # Rendering is not implemented


# Instantiate and check the environment
env = HoltWintersEnv(ts_data, window_size=24, forecast_horizon=1)
check_env(env)

# Create the RL agent
model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=5000)

# Evaluate the agent
# Set the starting point to the end of the training data
start_step = len(train_ts_data) - env.window_size

# Reset the environment with the custom starting point
state, _ = env.reset(options={'current_step': start_step})

action, _ = model.predict(state)
alpha, beta, gamma = action
print(f"Alpha: {alpha}, Beta: {beta}, Gamma: {gamma}")

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
state, _ = env.reset()

# Get the action (hyperparameters) from the trained agent
action, _ = model.predict(state)
alpha, beta, gamma = action

# Fit the Holt-Winters model on the entire training data using the agent's hyperparameters
model_hw = ExponentialSmoothing(
    train_ts_data,
    trend='mul',
    seasonal='mul',
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

# Traditional optimization
model_hw_opt = ExponentialSmoothing(
    train_ts_data,
    trend='mul',
    seasonal='mul',
    seasonal_periods=12
).fit(optimized=True)

forecast_opt = model_hw_opt.forecast(len(test_ts_data))
mse_opt = np.mean((forecast_opt - test_ts_data) ** 2)
print(f"Test MSE with Traditional Optimization: {mse_opt}")

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(train_ts_data.index, train_ts_data.values, label='Training Data')
plt.plot(test_ts_data.index, test_ts_data.values, label='Actual Test Data')
plt.plot(test_ts_data.index, forecast.values, label='RL Forecast')
plt.plot(test_ts_data.index, forecast_opt.values, label='Optimized Forecast')
plt.title('Actual vs Forecasted Passengers (Test Data)')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()

