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
            trend='add',
            seasonal='add',
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
            self.current_step += 1
            if self.current_step >= self.max_steps:
                terminated = True
                next_state = np.zeros(self.window_size)
            else:
                next_state = self.ts_data[self.current_step:self.current_step + self.window_size].values
            return next_state.astype(np.float32), reward, terminated, truncated, {}

        # Forecast
        forecast = model_fit.forecast(self.forecast_horizon)
        actual = self.ts_data[end:end + self.forecast_horizon]

        # Calculate reward
        mse = np.mean((forecast - actual) ** 2)
        reward = -mse  # Negative MSE as reward

        # Prepare for next step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False  # Set to True if you have a time limit
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

