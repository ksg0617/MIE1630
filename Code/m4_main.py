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

# Load the training data
train_df = pd.read_csv('D:/MIE1630/Data/Monthly-train.csv')

# Load the testing data
test_df = pd.read_csv('D:/MIE1630/Data/Monthly-test.csv')

# Assign column names
train_df.columns = ['ID'] + ['V'+str(i) for i in range(1, train_df.shape[1])]
test_df.columns = ['ID'] + ['V'+str(i) for i in range(1, test_df.shape[1])]

# Select a time series, e.g., the first one
ts_id = train_df.iloc[0]['ID']

# Get the training data for this time series
train_ts_data = train_df.iloc[0, 1:].dropna().values.astype(float)

# Get the testing data for this time series
test_ts_data = test_df.iloc[0, 1:].dropna().values.astype(float)

# Number of periods in training data
n_train = len(train_ts_data)

# Create date range for training data
train_dates = pd.date_range(start='2000-01-01', periods=n_train, freq='M')

# Create pandas Series with the dates
train_ts_series = pd.Series(train_ts_data, index=train_dates)

# Similarly for test data
n_test = len(test_ts_data)
test_dates = pd.date_range(start=train_dates[-1] + pd.offsets.MonthBegin(), periods=n_test, freq='M')
test_ts_series = pd.Series(test_ts_data, index=test_dates)

# ts_data is the training data
ts_data = train_ts_series

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

        # Observation space: Recent time series data
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        state = self.ts_data[self.current_step:self.current_step + self.window_size].values
        return state.astype(np.float32), {}

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
        except Exception as e:
            # Penalize invalid parameter combinations
            reward = -100.0
            terminated = False
            truncated = False
            if self.current_step >= self.max_steps:
                terminated = True
                next_state = np.zeros(self.window_size)
            else:
                next_state = self.ts_data[self.current_step:self.current_step + self.window_size].values
            return next_state.astype(np.float32), reward, terminated, truncated, {}

        # Forecast
        forecast = model_fit.forecast(self.forecast_horizon)
        actual = self.ts_data[end:end + self.forecast_horizon]

        # reward normalization
        mse = np.mean((forecast - actual) ** 2)
        reward = -mse / np.mean(self.ts_data ** 2)  # Normalized MSE

        # Prepare for next step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        if not terminated:
            next_state = self.ts_data[self.current_step:self.current_step + self.window_size].values
        else:
            next_state = np.zeros(self.window_size)

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
state = ts_data[-env.window_size:].values.astype(np.float32)
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
    ts_data,
    trend='mul',
    seasonal='mul',
    seasonal_periods=12
)

try:
    model_fit = model_hw.fit(smoothing_level=alpha, smoothing_trend=beta,
                             smoothing_seasonal=gamma, optimized=False)
    forecast = model_fit.forecast(len(test_ts_series))
except Exception as e:
    print(f"Model fitting failed: {e}")
    forecast = pd.Series([np.nan]*len(test_ts_series), index=test_ts_series.index)

# Calculate the MSE on the test data
mse = np.mean((forecast - test_ts_series) ** 2)
print(f"Test MSE: {mse}")

# Plot the forecasts vs actuals
plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data.values, label='Training Data')
plt.plot(test_ts_series.index, test_ts_series.values, label='Actual Test Data')
plt.plot(test_ts_series.index, forecast.values, label='Forecasted Data')
plt.title('Actual vs Forecasted Values (Test Data)')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.show()

# Traditional optimization
model_hw_opt = ExponentialSmoothing(
    ts_data,
    trend='mul',
    seasonal='mul',
    seasonal_periods=12
).fit(optimized=True)

forecast_opt = model_hw_opt.forecast(len(test_ts_series))
mse_opt = np.mean((forecast_opt - test_ts_series) ** 2)
print(f"Test MSE with Traditional Optimization: {mse_opt}")

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data.values, label='Training Data')
plt.plot(test_ts_series.index, test_ts_series.values, label='Actual Test Data')
plt.plot(test_ts_series.index, forecast.values, label='RL Forecast')
plt.plot(test_ts_series.index, forecast_opt.values, label='Optimized Forecast')
plt.title('Actual vs Forecasted Values (Test Data)')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.show()
